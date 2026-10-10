import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.musc import MuSc


VARIANTS = {
    "static_fuser_self": {
        "dynamic": False,
        "mode": "fuser",
        "retrain_policy": "on_change",
    },
    "static_fuser_loo": {
        "dynamic": False,
        "mode": "static_loo",
        "retrain_policy": "on_change",
    },
    "dynamic_fuser_frozen": {
        "dynamic": True,
        "mode": "fuser",
        "retrain_policy": "fit_once",
    },
    "dynamic_fuser_retrain": {
        "dynamic": True,
        "mode": "fuser",
        "retrain_policy": "on_change",
    },
}


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            parsed = {"seed": int(row["seed"]), "variant": row["variant"]}
            for key, value in row.items():
                if key in {"seed", "variant"}:
                    continue
                if key == "fallback_reason_counts":
                    parsed[key] = value
                else:
                    parsed[key] = float(value) if value else None
            rows.append(parsed)
    return rows


def read_labels(model, category):
    dataset = model.load_datasets(category, divide_num=1, divide_iter=0)
    labels = []
    for index in range(len(dataset)):
        item = dataset[index]
        value = item["is_anomaly"]
        labels.append(bool(value.item() if hasattr(value, "item") else value))
    return labels


def expert_lifecycle_metrics(timeline, labels):
    experts = {}
    deleted = set()
    for record in timeline:
        for expert in record["active_experts_before"] + record["active_experts_after"]:
            experts[expert["expert_id"]] = expert["image_id"]
        deleted.update(record["deleted_expert_ids"])

    final_ids = {
        expert["expert_id"] for expert in timeline[-1]["active_experts_after"]
    }
    normal_ids = {
        expert_id
        for expert_id, image_id in experts.items()
        if not labels[image_id]
    }
    anomaly_ids = set(experts) - normal_ids
    return {
        "admitted_expert_count": len(experts),
        "normal_expert_count": len(normal_ids),
        "anomaly_expert_count": len(anomaly_ids),
        "normal_expert_deletion_rate": (
            len(normal_ids & deleted) / len(normal_ids) if normal_ids else None
        ),
        "anomaly_expert_deletion_rate": (
            len(anomaly_ids & deleted) / len(anomaly_ids) if anomaly_ids else None
        ),
        "final_normal_expert_count": len(final_ids & normal_ids),
        "final_anomaly_expert_count": len(final_ids & anomaly_ids),
    }


def scoring_metrics(scoring):
    covered = [row for row in scoring if row["expert_image_ids"]]
    fallback = [row for row in scoring if row["fallback_reason"] is not None]
    fallback_steps = {row["step"] for row in fallback}
    first_image_steps = {
        row["step"]
        for row in scoring
        if row["fallback_reason"] == "first_image_unavailable"
    }
    fuser_records = [row for row in scoring if "fuser_mean" in row]
    fallback_reason_counts = {}
    for row in fallback:
        reason = row["fallback_reason"]
        fallback_reason_counts[reason] = fallback_reason_counts.get(reason, 0) + 1
    return {
        "scoring_record_count": len(scoring),
        "expert_scoring_coverage": len(covered) / len(scoring) if scoring else None,
        "fallback_record_count": len(fallback),
        "fallback_step_count": len(fallback_steps),
        "first_image_unavailable_step_count": len(first_image_steps),
        "fuser_retrain_count": sum(row["fuser_retrained"] for row in scoring),
        "training_recompute_count": sum(
            row.get("training_recomputed", False) for row in scoring
        ),
        "fuser_score_min": (
            min(row["fuser_min"] for row in fuser_records)
            if fuser_records else None
        ),
        "fuser_score_max": (
            max(row["fuser_max"] for row in fuser_records)
            if fuser_records else None
        ),
        "fuser_score_mean": _mean_field(fuser_records, "fuser_mean"),
        "fuser_negative_ratio": _mean_field(
            fuser_records, "fuser_negative_ratio"
        ),
        "training_image_count_max": max(
            (row["training_image_count"] for row in scoring), default=0
        ),
        "training_patch_pair_count_max": max(
            (row["training_patch_pair_count"] for row in scoring), default=0
        ),
        "dino_score_mean": _mean_field(scoring, "dino_mean"),
        "clip_score_mean": _mean_field(scoring, "clip_mean"),
        "dino_clip_correlation_mean": _mean_field(
            scoring, "dino_clip_correlation"
        ),
        "dino_clip_z_disagreement_mean": _mean_field(
            scoring, "dino_clip_z_disagreement"
        ),
        "fallback_reason_counts": json.dumps(
            fallback_reason_counts, ensure_ascii=False, sort_keys=True
        ),
    }


def _mean_field(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def run_variant(base_config, args, seed, variant_name, timeline_path=None):
    variant = VARIANTS[variant_name]
    cfg = copy.deepcopy(base_config)
    cfg["datasets"]["dataset_name"] = "mvtec_ad"
    cfg["datasets"]["data_path"] = str(Path(args.data_root).resolve())
    cfg["datasets"]["class_name"] = args.category
    cfg["datasets"]["divide_num"] = 1
    cfg["testing"]["vis"] = False
    cfg["testing"]["save_excel"] = False
    cfg["testing"]["max_samples"] = args.max_samples
    cfg["testing"]["shuffle_stream"] = True
    cfg["models"]["feature_layers"] = [23]
    cfg["models"]["r_list"] = [1]
    cfg["models"]["r_list2"] = [1]
    run_output = Path(args.output_dir) / f"seed_{seed}" / variant_name
    cfg["testing"]["output_dir"] = str(run_output)
    cfg["models"].setdefault("dynamic_committee", {})["enabled"] = variant[
        "dynamic"
    ]
    cfg["models"]["dynamic_committee"]["timeline_path"] = (
        str(timeline_path) if timeline_path else None
    )
    cfg["models"].setdefault("dynamic_fusion", {})["mode"] = variant["mode"]
    cfg["models"]["dynamic_fusion"]["retrain_policy"] = variant[
        "retrain_policy"
    ]

    model = MuSc(cfg, seed=seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    image_metric, pixel_metric = model.make_category_data(args.category)
    runtime_seconds = time.perf_counter() - started
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(model.device) / 1024 / 1024
        if torch.cuda.is_available()
        else 0.0
    )

    row = {
        "seed": seed,
        "variant": variant_name,
        "image_auroc": float(image_metric[0]),
        "image_f1": float(image_metric[1]),
        "image_ap": float(image_metric[2]),
        "pixel_auroc": float(pixel_metric[0]),
        "pixel_f1": float(pixel_metric[1]),
        "pixel_ap": float(pixel_metric[2]),
        "aupro": float(pixel_metric[3]),
        "runtime_seconds": runtime_seconds,
        "peak_memory_mb": peak_memory_mb,
        "expert_scoring_coverage": None,
        "fallback_record_count": None,
        "fallback_step_count": None,
        "normal_expert_deletion_rate": None,
        "anomaly_expert_deletion_rate": None,
        "final_normal_expert_count": None,
        "final_anomaly_expert_count": None,
        "scoring_record_count": None,
        "first_image_unavailable_step_count": None,
        "admitted_expert_count": None,
        "normal_expert_count": None,
        "anomaly_expert_count": None,
        "fuser_retrain_count": None,
        "training_recompute_count": None,
        "fuser_score_min": None,
        "fuser_score_max": None,
        "fuser_score_mean": None,
        "fuser_negative_ratio": None,
        "training_image_count_max": None,
        "training_patch_pair_count_max": None,
        "dino_score_mean": None,
        "clip_score_mean": None,
        "dino_clip_correlation_mean": None,
        "dino_clip_z_disagreement_mean": None,
        "fallback_reason_counts": None,
    }
    timeline = None
    if variant["dynamic"]:
        audit_files = list(run_output.rglob(f"{args.category}_dynamic_audit_*.json"))
        if len(audit_files) != 1:
            raise RuntimeError(
                f"expected one dynamic audit file, found {len(audit_files)}"
            )
        audit = json.loads(audit_files[0].read_text(encoding="utf-8"))
        timeline = audit["timeline"]
        row.update(scoring_metrics(audit["scoring"]))
        row.update(expert_lifecycle_metrics(timeline, read_labels(model, args.category)))
    return row, timeline


def aggregate(rows):
    summaries = []
    numeric_fields = [
        key
        for key in rows[0]
        if key not in {"seed", "variant", "fallback_reason_counts"}
    ]
    for variant_name in VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant_name]
        if not variant_rows:
            continue
        summary = {"variant": variant_name, "seed_count": len(variant_rows)}
        for field in numeric_fields:
            values = [row[field] for row in variant_rows if row[field] is not None]
            summary[f"{field}_mean"] = sum(values) / len(values) if values else None
        summaries.append(summary)
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the bottle dynamic DINO+CLIP+fuser ablation."
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "musc.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(VARIANTS),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_config(args.config)
    runs_path = output_dir / "multisource_ablation_runs.csv"
    rows = read_csv(runs_path) if args.resume and runs_path.exists() else []
    completed = {(row["seed"], row["variant"]) for row in rows}
    for seed in args.seeds:
        timeline_path = output_dir / f"seed_{seed}" / "dynamic_timeline.json"
        reference_timeline = (
            json.loads(timeline_path.read_text(encoding="utf-8"))
            if timeline_path.exists()
            else None
        )
        for variant_name in args.variants:
            if (seed, variant_name) in completed:
                continue
            row, timeline = run_variant(
                base_config,
                args,
                seed,
                variant_name,
                timeline_path if reference_timeline is not None else None,
            )
            rows.append(row)
            if timeline is not None:
                if reference_timeline is None:
                    reference_timeline = timeline
                    timeline_path.parent.mkdir(parents=True, exist_ok=True)
                    timeline_path.write_text(
                        json.dumps(timeline, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                elif timeline != reference_timeline:
                    raise RuntimeError(
                        f"dynamic variants produced different timelines for seed {seed}"
                    )
            write_csv(runs_path, rows)

    summaries = aggregate(rows)
    write_csv(output_dir / "multisource_ablation_summary.csv", summaries)
    (output_dir / "multisource_ablation_summary.json").write_text(
        json.dumps(
            {"runs": rows, "summary": summaries},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
