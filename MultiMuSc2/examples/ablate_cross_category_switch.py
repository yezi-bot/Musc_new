import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.musc import MuSc


VARIANTS = {
    "source_static_fuser": {
        "dynamic": False,
        "mode": "fuser",
        "reset": False,
    },
    "strict_history_dino": {
        "dynamic": True,
        "mode": "strict_history_dino",
        "reset": False,
    },
    "dynamic_dino_no_reset": {
        "dynamic": True,
        "mode": "dino_only",
        "reset": False,
    },
    "dynamic_dino_oracle_reset": {
        "dynamic": True,
        "mode": "dino_only",
        "reset": True,
    },
}


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def safe_auroc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


def window_auroc(labels, scores, start, length):
    stop = min(start + length, len(labels))
    return safe_auroc(labels[start:stop], scores[start:stop])


def good_score_ratio(labels, scores, boundary, window=10):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores)
    pre_slice = slice(max(0, boundary - window), boundary)
    post_slice = slice(boundary, min(len(labels), boundary + window))
    pre = scores[pre_slice][~labels[pre_slice]]
    post = scores[post_slice][~labels[post_slice]]
    if pre.size == 0 or post.size == 0:
        return None
    denominator = float(np.median(pre))
    if denominator == 0.0:
        return None
    return float(np.median(post) / denominator)


def audit_metrics(audit_path, boundary):
    if audit_path is None:
        return {
            "post_old_expert_reference_fraction": None,
            "new_expert_latency": None,
            "old_expert_clear_latency": None,
            "state_reset_count": None,
            "causal_reference_violations": None,
        }
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    scoring = audit["scoring"]
    post = [record for record in scoring if record["step"] >= boundary]
    old_reference_records = [
        record
        for record in post
        if any(image_id < boundary for image_id in record["expert_image_ids"])
    ]
    first_new = next(
        (
            record["step"]
            for record in post
            if any(image_id >= boundary for image_id in record["expert_image_ids"])
        ),
        None,
    )
    first_cleared = next(
        (
            record["step"]
            for record in post
            if not any(image_id < boundary for image_id in record["expert_image_ids"])
        ),
        None,
    )
    violations = sum(
        image_id >= record["step"]
        for record in scoring
        for image_id in record["expert_image_ids"]
    )
    return {
        "post_old_expert_reference_fraction": (
            len(old_reference_records) / len(post) if post else None
        ),
        "new_expert_latency": (
            first_new - boundary if first_new is not None else None
        ),
        "old_expert_clear_latency": (
            first_cleared - boundary if first_cleared is not None else None
        ),
        "state_reset_count": sum(
            record.get("state_reset_before", False) for record in scoring
        ),
        "causal_reference_violations": int(violations),
    }


def run_variant(base_config, args, sequence, seed, variant_name):
    variant = VARIANTS[variant_name]
    sequence_name = "_to_".join(sequence)
    run_output = Path(args.output_dir) / sequence_name / f"seed_{seed}" / variant_name
    cfg = copy.deepcopy(base_config)
    cfg["datasets"]["dataset_name"] = "mvtec_ad"
    cfg["datasets"]["data_path"] = str(Path(args.data_root).resolve())
    cfg["datasets"]["class_name"] = list(sequence)
    cfg["datasets"]["divide_num"] = 1
    cfg["testing"]["output_dir"] = str(run_output)
    cfg["testing"]["vis"] = False
    cfg["testing"]["save_excel"] = False
    cfg["testing"]["shuffle_stream"] = True
    cfg["testing"]["max_samples"] = args.segment_samples
    cfg["models"]["feature_layers"] = [23]
    cfg["models"]["r_list"] = [1]
    cfg["models"]["r_list2"] = [1]
    cfg["models"].setdefault("dynamic_committee", {})["enabled"] = variant[
        "dynamic"
    ]
    cfg["models"]["dynamic_committee"][
        "reset_at_category_boundaries"
    ] = variant["reset"]
    cfg["models"].setdefault("dynamic_fusion", {})["mode"] = variant["mode"]

    model = MuSc(cfg, seed=seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    image_metric, pixel_metric, predictions = model.make_category_data(
        list(sequence),
        return_predictions=True,
    )
    runtime_seconds = time.perf_counter() - started
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(model.device) / 1024 / 1024
        if torch.cuda.is_available()
        else 0.0
    )
    boundaries = predictions["stream_boundaries"]
    if len(boundaries) != 1:
        raise RuntimeError("two-category streams must have exactly one boundary")
    boundary = boundaries[0]
    labels = np.asarray(predictions["image_labels"])
    raw_scores = np.asarray(predictions["raw_image_scores"])
    pixel_labels = np.asarray(predictions["pixel_labels"])
    pixel_scores = np.asarray(predictions["pixel_scores"])
    audit_files = list(run_output.rglob(f"{sequence_name}_dynamic_audit_*.json"))
    audit_path = audit_files[0] if len(audit_files) == 1 else None

    row = {
        "sequence": "->".join(sequence),
        "seed": seed,
        "variant": variant_name,
        "segment_samples": args.segment_samples,
        "boundary": boundary,
        "image_auroc": float(image_metric[0]),
        "pixel_auroc": float(pixel_metric[0]),
        "aupro": float(pixel_metric[3]),
        "raw_image_auroc": safe_auroc(labels, raw_scores),
        "pre_raw_image_auroc": safe_auroc(
            labels[:boundary], raw_scores[:boundary]
        ),
        "post_raw_image_auroc": safe_auroc(
            labels[boundary:], raw_scores[boundary:]
        ),
        "post5_raw_image_auroc": window_auroc(
            labels, raw_scores, boundary, 5
        ),
        "post10_raw_image_auroc": window_auroc(
            labels, raw_scores, boundary, 10
        ),
        "post20_raw_image_auroc": window_auroc(
            labels, raw_scores, boundary, 20
        ),
        "post10_good_score_ratio": good_score_ratio(
            labels, raw_scores, boundary, window=10
        ),
        "pre_pixel_auroc": safe_auroc(
            pixel_labels[:boundary].ravel(), pixel_scores[:boundary].ravel()
        ),
        "post_pixel_auroc": safe_auroc(
            pixel_labels[boundary:].ravel(), pixel_scores[boundary:].ravel()
        ),
        "runtime_seconds": runtime_seconds,
        "peak_memory_mb": peak_memory_mb,
    }
    row.update(audit_metrics(audit_path, boundary))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def write_results(output_dir, rows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cross_category_switch_runs.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "cross_category_switch_summary.json").write_text(
        json.dumps({"runs": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate abrupt two-category stream switches."
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "musc.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["bottle:cable", "cable:bottle"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--segment-samples", type=int, default=40)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(VARIANTS),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.segment_samples < 5:
        raise ValueError("segment-samples must be at least 5")
    sequences = []
    for value in args.sequences:
        categories = value.split(":")
        if len(categories) != 2 or categories[0] == categories[1]:
            raise ValueError("each sequence must contain two different categories")
        sequences.append(categories)

    base_config = load_config(args.config)
    rows = []
    for sequence in sequences:
        for seed in args.seeds:
            for variant_name in args.variants:
                rows.append(
                    run_variant(
                        base_config,
                        args,
                        sequence,
                        seed,
                        variant_name,
                    )
                )
                write_results(args.output_dir, rows)


if __name__ == "__main__":
    main()
