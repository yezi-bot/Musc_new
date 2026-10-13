import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.mvtec import DatasetSplit, MVTecDataset
from examples.run_dynamic_dino import load_dinov2, seed_everything
from models.backbone.open_clip.transform import image_transform
from models.modules._DYNAMIC_DINO_STREAM import chunked_dino_score
from models.modules._LNAMD import LNAMD
from models.modules._RsCIN import RsCIN
from utils.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay cross-category DINO-only runs and perform an offline "
            "GT-only contaminated-expert oracle audit."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--dinov2-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--metric-tolerance", type=float, default=5e-5)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--sequences", nargs="+")
    return parser.parse_args()


def metric_dict(image_metric, pixel_metric):
    return {
        "image_auroc": float(image_metric[0]),
        "image_f1": float(image_metric[1]),
        "image_ap": float(image_metric[2]),
        "pixel_auroc": float(pixel_metric[0]),
        "pixel_f1": float(pixel_metric[1]),
        "pixel_ap": float(pixel_metric[2]),
        "aupro": float(pixel_metric[3]),
    }


def safe_mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def interpolate_patch_score(patch_score, image_size):
    grid_size = math.isqrt(patch_score.numel())
    if grid_size * grid_size != patch_score.numel():
        raise ValueError("patch count must form a square grid")
    return F.interpolate(
        patch_score.reshape(1, 1, grid_size, grid_size),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=True,
    )[0, 0].detach().cpu()


def dino_patch_score(query, references, device, topmin_min):
    if not references:
        return torch.zeros(
            query.shape[0], device=device, dtype=torch.float32
        )
    return chunked_dino_score(
        query=query,
        references=references,
        device=device,
        topmin_min=topmin_min,
        topmin_max=0.3,
        chunk_size=8,
    )


class CrossCategoryFeatureExtractor:
    def __init__(self, model, device):
        self.model = model
        self.device = torch.device(device)
        self.aggregator = LNAMD(
            device=self.device,
            r=1,
            feature_dim=1024,
            feature_layer=[24],
        )

    def extract(self, image):
        image = image.unsqueeze(0).to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type == "cuda",
        ):
            patch_tokens = self.model.get_intermediate_layers(
                image,
                n=[23],
                return_class_token=False,
            )
            patch_tokens = [
                torch.cat(
                    [torch.zeros_like(tokens[:, :1]), tokens], dim=1
                )
                for tokens in patch_tokens
            ]
            features = self.aggregator._embed(patch_tokens)
            features /= features.norm(dim=-1, keepdim=True)
            class_token = self.model(image)
        return (
            features[0, :, 0, :].detach().cpu(),
            class_token[0].detach().float().cpu().numpy(),
        )


def selected_stream(data_root, category, seed, max_samples, transform, image_size):
    dataset = MVTecDataset(
        source=data_root,
        classname=category,
        resize=image_size,
        imagesize=image_size,
        split=DatasetSplit.TEST,
        clip_transformer=transform,
        random_seed=seed,
    )
    indices = list(range(len(dataset)))
    if max_samples < len(indices):
        indices = (
            torch.linspace(0, len(indices) - 1, steps=max_samples)
            .round()
            .long()
            .unique(sorted=True)
            .tolist()
        )
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(
        len(indices), generator=generator
    ).tolist()
    return [dataset[indices[index]] for index in permutation]


def extract_category_stream(
    data_root,
    category,
    seed,
    max_samples,
    transform,
    image_size,
    extractor,
):
    samples = selected_stream(
        data_root,
        category,
        seed,
        max_samples,
        transform,
        image_size,
    )
    output = []
    for index, sample in enumerate(samples):
        print(
            f"  extracting {category} seed={seed}: "
            f"{index + 1}/{len(samples)}",
            end="\r",
            flush=True,
        )
        features, class_token = extractor.extract(sample["image"])
        output.append(
            {
                "category": category,
                "anomaly_type": Path(sample["image_path"]).parent.name,
                "is_anomaly": bool(sample["is_anomaly"]),
                "image_path": str(sample["image_path"]),
                "mask": (sample["mask"] > 0.5).numpy().astype(np.int32),
                "features": features,
                "class_token": class_token,
            }
        )
    print(" " * 80, end="\r", flush=True)
    return output


def find_audit_path(experiment_root, run):
    sequence_dir = run["sequence"].replace("->", "_to_")
    root = (
        experiment_root
        / sequence_dir
        / f"seed_{run['seed']}"
        / run["variant"]
    )
    candidates = list(root.rglob("*_dynamic_audit_*.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one audit below {root}, found {len(candidates)}"
        )
    return candidates[0]


def validate_audit(audit, stream):
    timeline = audit["timeline"]
    scoring = audit["scoring"]
    expected_steps = list(range(len(stream)))
    if [int(row["step"]) for row in timeline] != expected_steps:
        raise ValueError("timeline steps must be contiguous and ordered")
    if [int(row["step"]) for row in scoring] != expected_steps:
        raise ValueError("scoring steps must be contiguous and ordered")
    if any(row["requested_mode"] != "dino_only" for row in scoring):
        raise ValueError("only DINO-only cross-category runs are supported")
    if any(int(row["aggregation_r"]) != 1 for row in scoring):
        raise ValueError("analysis expects aggregation r=1")
    if any(int(row["dino_layer"]) != 23 for row in scoring):
        raise ValueError("analysis expects DINO layer 23")
    supported_modes = {
        "dino_fallback",
        "dino_only",
        "dino_provisional",
    }
    unsupported_modes = sorted(
        {
            row["effective_mode"]
            for row in scoring
            if row["effective_mode"] not in supported_modes
        }
    )
    if unsupported_modes:
        raise ValueError(
            f"unsupported effective scoring modes: {unsupported_modes}"
        )


def used_reference_ids(scoring_row):
    effective_mode = scoring_row["effective_mode"]
    formal = []
    provisional = []
    if effective_mode in {
        "dino_only",
        "dino_provisional",
        "dino_channel_irbank",
    }:
        formal = [int(value) for value in scoring_row["expert_image_ids"]]
    if effective_mode in {"dino_provisional", "dino_channel_irbank"}:
        provisional = [
            int(value)
            for value in scoring_row.get("provisional_expert_image_ids", [])
        ]
    return formal, provisional


def score_step(stream, step, scoring_row, device, image_size, oracle=False):
    current = stream[step]["features"]
    formal_ids, provisional_ids = used_reference_ids(scoring_row)
    if oracle:
        formal_ids = [
            image_id
            for image_id in formal_ids
            if not stream[image_id]["is_anomaly"]
        ]
        provisional_ids = [
            image_id
            for image_id in provisional_ids
            if not stream[image_id]["is_anomaly"]
        ]

    if formal_ids:
        base_score = dino_patch_score(
            current,
            [stream[image_id]["features"] for image_id in formal_ids],
            device,
            topmin_min=0.02,
        )
    else:
        history_start = int(scoring_row.get("segment_start", 0))
        base_score = dino_patch_score(
            current,
            [stream[index]["features"] for index in range(history_start, step)],
            device,
            topmin_min=0.0,
        )

    if provisional_ids and scoring_row["effective_mode"] == "dino_provisional":
        provisional_score = dino_patch_score(
            current,
            [stream[image_id]["features"] for image_id in provisional_ids],
            device,
            topmin_min=0.02,
        )
        weight = float(scoring_row["provisional_expert_weight"])
        base_score = (1.0 - weight) * base_score + weight * provisional_score
    return interpolate_patch_score(base_score, image_size)


def phase_summary(rows, phase):
    selected = [row for row in rows if row["phase"] == phase]
    result = {"step_count": len(selected)}
    for label_name, is_anomaly in (("normal", False), ("anomaly", True)):
        label_rows = [
            row for row in selected if row["is_anomaly"] is is_anomaly
        ]
        result[f"{label_name}_count"] = len(label_rows)
        result[f"{label_name}_original_score_mean"] = safe_mean(
            [row["original_score"] for row in label_rows]
        )
        result[f"{label_name}_oracle_score_mean"] = safe_mean(
            [row["oracle_score"] for row in label_rows]
        )
        result[f"{label_name}_oracle_delta_mean"] = safe_mean(
            [row["oracle_score_delta"] for row in label_rows]
        )
    return result


def admission_rows(audit, stream):
    rows = []
    for record in audit["timeline"]:
        admitted_id = record.get("admitted_expert_id")
        if admitted_id is None:
            continue
        matches = [
            expert
            for expert in record.get("active_experts_after", [])
            if int(expert["expert_id"]) == int(admitted_id)
        ]
        if len(matches) != 1:
            raise ValueError("admitted expert is absent from active_experts_after")
        expert = matches[0]
        image_id = int(expert["image_id"])
        rows.append(
            {
                "expert_id": int(admitted_id),
                "image_id": image_id,
                "admission_step": int(record["step"]),
                "category": stream[image_id]["category"],
                "anomaly_type": stream[image_id]["anomaly_type"],
                "is_anomaly": stream[image_id]["is_anomaly"],
            }
        )
    return rows


def compare_metric(name, actual, expected, tolerance):
    error = abs(float(actual) - float(expected))
    if error > tolerance:
        raise RuntimeError(
            f"replayed {name} error {error:.6g} exceeds tolerance {tolerance}"
        )
    return error


def run_analysis(run, audit, stream, device, image_size, metric_tolerance):
    validate_audit(audit, stream)
    scoring = audit["scoring"]
    boundary = int(run["boundary"])
    rows = []
    original_maps = []
    oracle_maps = []
    contaminated_exposure_steps = 0
    stale_post_switch_exposure_steps = 0
    used_reference_slots = 0
    contaminated_reference_slots = 0
    stale_post_switch_reference_slots = 0

    for step, scoring_row in enumerate(scoring):
        formal_ids, provisional_ids = used_reference_ids(scoring_row)
        used_ids = formal_ids + provisional_ids
        contaminated_ids = [
            image_id for image_id in used_ids if stream[image_id]["is_anomaly"]
        ]
        stale_ids = [
            image_id
            for image_id in used_ids
            if step >= boundary and stream[image_id]["category"] != stream[step]["category"]
        ]
        if contaminated_ids:
            contaminated_exposure_steps += 1
        if stale_ids:
            stale_post_switch_exposure_steps += 1
        used_reference_slots += len(used_ids)
        contaminated_reference_slots += len(contaminated_ids)
        stale_post_switch_reference_slots += len(stale_ids)

        original_map = score_step(
            stream, step, scoring_row, device, image_size, oracle=False
        )
        oracle_map = (
            score_step(
                stream, step, scoring_row, device, image_size, oracle=True
            )
            if contaminated_ids
            else original_map
        )
        original_score = float(original_map.max())
        oracle_score = float(oracle_map.max())
        if contaminated_ids:
            phase = "contaminated_exposure"
        elif step < boundary:
            phase = "pre_switch_clean_or_fallback"
        else:
            phase = "post_switch_clean_or_fallback"
        rows.append(
            {
                "sequence": run["sequence"],
                "seed": int(run["seed"]),
                "variant": run["variant"],
                "step": step,
                "segment": "pre_switch" if step < boundary else "post_switch",
                "category": stream[step]["category"],
                "image_path": stream[step]["image_path"],
                "anomaly_type": stream[step]["anomaly_type"],
                "is_anomaly": stream[step]["is_anomaly"],
                "effective_mode": scoring_row["effective_mode"],
                "phase": phase,
                "formal_reference_ids": " ".join(map(str, formal_ids)),
                "provisional_reference_ids": " ".join(map(str, provisional_ids)),
                "contaminated_reference_ids": " ".join(
                    map(str, contaminated_ids)
                ),
                "stale_reference_ids": " ".join(map(str, stale_ids)),
                "original_score": original_score,
                "oracle_score": oracle_score,
                "oracle_score_delta": oracle_score - original_score,
            }
        )
        original_maps.append(original_map.numpy())
        oracle_maps.append(oracle_map.numpy())

    labels = np.asarray(
        [int(sample["is_anomaly"]) for sample in stream], dtype=np.int32
    )
    masks = np.stack([sample["mask"] for sample in stream])
    class_tokens = [sample["class_token"] for sample in stream]
    original_raw_scores = np.asarray(
        [row["original_score"] for row in rows]
    )
    oracle_raw_scores = np.asarray([row["oracle_score"] for row in rows])
    original_image_scores = RsCIN(
        original_raw_scores, class_tokens, k_list=[1, 2, 3]
    )
    oracle_image_scores = RsCIN(
        oracle_raw_scores, class_tokens, k_list=[1, 2, 3]
    )
    original_image_metric, original_pixel_metric = compute_metrics(
        labels,
        original_image_scores,
        masks,
        np.stack(original_maps),
    )
    oracle_image_metric, oracle_pixel_metric = compute_metrics(
        labels,
        oracle_image_scores,
        masks,
        np.stack(oracle_maps),
    )
    original_metrics = metric_dict(original_image_metric, original_pixel_metric)
    oracle_metrics = metric_dict(oracle_image_metric, oracle_pixel_metric)
    raw_image_metric, _ = compute_metrics(labels, original_raw_scores)
    replay_errors = {
        "image_auroc": compare_metric(
            "image_auroc",
            original_metrics["image_auroc"],
            run["image_auroc"],
            metric_tolerance,
        ),
        "pixel_auroc": compare_metric(
            "pixel_auroc",
            original_metrics["pixel_auroc"],
            run["pixel_auroc"],
            metric_tolerance,
        ),
        "aupro": compare_metric(
            "aupro",
            original_metrics["aupro"],
            run["aupro"],
            metric_tolerance,
        ),
        "raw_image_auroc": compare_metric(
            "raw_image_auroc",
            raw_image_metric[0],
            run["raw_image_auroc"],
            metric_tolerance,
        ),
    }
    metric_delta = {
        key: oracle_metrics[key] - original_metrics[key]
        for key in original_metrics
    }
    admissions = admission_rows(audit, stream)
    return {
        "sequence": run["sequence"],
        "seed": int(run["seed"]),
        "variant": run["variant"],
        "boundary": boundary,
        "step_count": len(stream),
        "formal_admission_count": len(admissions),
        "contaminated_formal_admission_count": sum(
            int(row["is_anomaly"]) for row in admissions
        ),
        "formal_admissions": admissions,
        "used_reference_steps": sum(
            bool(used_reference_ids(row)[0] + used_reference_ids(row)[1])
            for row in scoring
        ),
        "contaminated_exposure_steps": contaminated_exposure_steps,
        "contaminated_exposure_fraction_of_all_steps": (
            contaminated_exposure_steps / len(stream)
        ),
        "stale_post_switch_exposure_steps": stale_post_switch_exposure_steps,
        "stale_post_switch_exposure_fraction": (
            stale_post_switch_exposure_steps / (len(stream) - boundary)
        ),
        "used_reference_slots": used_reference_slots,
        "contaminated_reference_slots": contaminated_reference_slots,
        "contaminated_reference_slot_fraction": (
            contaminated_reference_slots / used_reference_slots
            if used_reference_slots
            else 0.0
        ),
        "stale_post_switch_reference_slots": stale_post_switch_reference_slots,
        "replay_metric_errors": replay_errors,
        "original_metrics": original_metrics,
        "oracle_metrics": oracle_metrics,
        "oracle_minus_original": metric_delta,
        "score_phases": {
            phase: phase_summary(rows, phase)
            for phase in (
                "pre_switch_clean_or_fallback",
                "contaminated_exposure",
                "post_switch_clean_or_fallback",
            )
        },
    }, rows


def aggregate_group(results):
    metric_names = list(results[0]["original_metrics"])
    total_steps = sum(result["step_count"] for result in results)
    total_post_steps = sum(
        result["step_count"] - result["boundary"] for result in results
    )
    total_slots = sum(result["used_reference_slots"] for result in results)
    return {
        "run_count": len(results),
        "formal_admission_count": sum(
            result["formal_admission_count"] for result in results
        ),
        "contaminated_formal_admission_count": sum(
            result["contaminated_formal_admission_count"] for result in results
        ),
        "contaminated_exposure_steps": sum(
            result["contaminated_exposure_steps"] for result in results
        ),
        "contaminated_exposure_fraction_of_all_steps": (
            sum(result["contaminated_exposure_steps"] for result in results)
            / total_steps
        ),
        "stale_post_switch_exposure_steps": sum(
            result["stale_post_switch_exposure_steps"] for result in results
        ),
        "stale_post_switch_exposure_fraction": (
            sum(result["stale_post_switch_exposure_steps"] for result in results)
            / total_post_steps
        ),
        "used_reference_slots": total_slots,
        "contaminated_reference_slots": sum(
            result["contaminated_reference_slots"] for result in results
        ),
        "contaminated_reference_slot_fraction": (
            sum(result["contaminated_reference_slots"] for result in results)
            / total_slots
            if total_slots
            else 0.0
        ),
        "original_metric_means": {
            key: safe_mean([result["original_metrics"][key] for result in results])
            for key in metric_names
        },
        "oracle_metric_means": {
            key: safe_mean([result["oracle_metrics"][key] for result in results])
            for key in metric_names
        },
        "oracle_minus_original_means": {
            key: safe_mean(
                [result["oracle_minus_original"][key] for result in results]
            )
            for key in metric_names
        },
    }


def aggregate_results(results):
    grouped = defaultdict(list)
    for result in results:
        grouped[f"{result['sequence']}|{result['variant']}"].append(result)
    return {
        "overall": aggregate_group(results),
        "by_sequence_variant": {
            key: aggregate_group(group) for key, group in sorted(grouped.items())
        },
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    summary_path = Path(
        args.summary or experiment_root / "cross_category_switch_summary.json"
    ).resolve()
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = source_summary["runs"]
    if args.seeds is not None:
        runs = [run for run in runs if int(run["seed"]) in args.seeds]
    if args.variants is not None:
        runs = [run for run in runs if run["variant"] in args.variants]
    if args.sequences is not None:
        runs = [run for run in runs if run["sequence"] in args.sequences]
    if not runs:
        raise ValueError("no runs match the requested filters")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_dinov2(args.dinov2_repo, args.checkpoint, device)
    extractor = CrossCategoryFeatureExtractor(model, device)
    transform = image_transform(
        args.image_size,
        is_train=False,
    )

    results = []
    all_rows = []
    runs_by_seed = defaultdict(list)
    for run in runs:
        runs_by_seed[int(run["seed"])].append(run)
    for seed, seed_runs in sorted(runs_by_seed.items()):
        seed_everything(seed)
        categories = sorted(
            {
                category
                for run in seed_runs
                for category in run["sequence"].split("->")
            }
        )
        segment_sizes = {int(run["segment_samples"]) for run in seed_runs}
        if len(segment_sizes) != 1:
            raise ValueError("all runs for a seed must share segment_samples")
        segment_samples = segment_sizes.pop()
        category_streams = {}
        for category in categories:
            category_streams[category] = extract_category_stream(
                args.data_root,
                category,
                seed,
                segment_samples,
                transform,
                args.image_size,
                extractor,
            )
        for run in seed_runs:
            sequence = run["sequence"].split("->")
            stream = [
                sample
                for category in sequence
                for sample in category_streams[category]
            ]
            audit_path = find_audit_path(experiment_root, run)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            print(
                f"Analyzing {run['sequence']} seed={seed} "
                f"variant={run['variant']}...",
                flush=True,
            )
            result, rows = run_analysis(
                run,
                audit,
                stream,
                device,
                args.image_size,
                args.metric_tolerance,
            )
            result["source_audit"] = str(audit_path)
            results.append(result)
            all_rows.extend(rows)
        del category_streams
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = {
        "analysis_scope": (
            "cross_category_offline_gt_oracle_remove_anomalous_expert_roles"
        ),
        "gt_usage": (
            "GT is used only after each original online run to label and remove "
            "anomalous formal/provisional expert references. It never changes "
            "the online timeline, admission, deletion, or channel state."
        ),
        "oracle_empty_committee_policy": (
            "When GT removal leaves no clean formal expert, replay uses the "
            "original strict-history DINO fallback; contaminated images remain "
            "ordinary history but no longer act as experts."
        ),
        "source_summary": str(summary_path),
        "source_experiment_root": str(experiment_root),
        "data_root": str(Path(args.data_root).resolve()),
        "dinov2_repo": str(Path(args.dinov2_repo).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "image_size": args.image_size,
        "aggregate": aggregate_results(results),
        "runs": results,
    }
    (output_dir / "cross_category_contamination_impact.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "cross_category_contamination_steps.csv", all_rows)


if __name__ == "__main__":
    main()
