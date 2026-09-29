"""Evaluate image-level AUROC for the current DINO dynamic EX-MSM pipeline."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from ablate_stage4_lifecycle import file_sha256
from validate_density_bottle import load_samples, write_csv
from validate_dynamic_ex_msm import (
    compute_online_dino_msm_image_score,
    read_stage1_records,
    read_stage3_experts,
)


EVALUATIONS = (
    ("similarity_ttl", 0),
    ("similarity_ttl", 1),
    ("channel_ttl", 0),
    ("channel_ttl", 1),
)


def expert_id(expert):
    return f"s{expert['stream_seed']}-e{expert['expert_index']}"


def read_lifecycle_timeline(path):
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "step": int(row["step"]),
                    "variant": row["variant"],
                    "committee_floor": int(row["committee_floor"]),
                    "expert_id": row["expert_id"],
                    "stream_seed": int(row["stream_seed"]),
                    "expert_image_id": int(row["expert_image_id"]),
                    "alive": row["alive"].lower() == "true",
                    "deleted_this_step": row["deleted_this_step"].lower()
                    == "true",
                    "event": row["event"],
                }
            )
    return rows


def load_features(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["features"] if isinstance(payload, dict) else payload


def validate_inputs(
    samples,
    features,
    stage1_records,
    experts,
    timeline,
    manifest,
    features_path,
    stage1_path,
    stage3_path,
):
    errors = []
    if len(samples) != len(features):
        errors.append("sample and feature counts differ")
    local_records = [row for row in stage1_records if row["method"] == "local"]
    stream_seeds = sorted({row["stream_seed"] for row in local_records})
    for stream_seed in stream_seeds:
        rows = sorted(
            (row for row in local_records if row["stream_seed"] == stream_seed),
            key=lambda row: row["stream_step"],
        )
        if [row["stream_step"] for row in rows] != list(range(len(samples))):
            errors.append(f"seed {stream_seed}: incomplete Stage 1 steps")
        if sorted(row["image_id"] for row in rows) != list(range(len(samples))):
            errors.append(f"seed {stream_seed}: Stage 1 image ids are not a permutation")
        for row in rows:
            if row["image_type"] != samples[row["image_id"]]["kind"]:
                errors.append(f"seed {stream_seed}: image type mismatch")
                break

    expert_ids = {expert_id(expert) for expert in experts}
    if len(expert_ids) != len(experts):
        errors.append("expert ids are not unique")
    if len(experts) != 8:
        errors.append(f"expected 8 Stage 3 experts, got {len(experts)}")

    timeline_keys = {
        (row["variant"], row["committee_floor"], row["expert_id"])
        for row in timeline
    }
    for variant, floor in EVALUATIONS:
        for eid in expert_ids:
            if (variant, floor, eid) not in timeline_keys:
                errors.append(f"missing lifecycle timeline: {variant} floor={floor} {eid}")

    expected_hashes = {
        "feature_cache_sha256": file_sha256(features_path),
        "stage1_records_sha256": file_sha256(stage1_path),
        "stage3_experts_sha256": file_sha256(stage3_path),
    }
    for key, value in expected_hashes.items():
        if manifest.get(key) != value:
            errors.append(f"input manifest mismatch: {key}")
    reproduction = manifest.get("stage1_numerical_reproduction", {})
    if reproduction.get("mismatch_count") != 0:
        errors.append("input manifest did not pass Stage 1 numerical reproduction")

    return {
        "errors": errors,
        "sample_count": len(samples),
        "feature_count": len(features),
        "stream_seeds": stream_seeds,
        "expert_ids": sorted(expert_ids),
        "experiment_id": manifest.get("experiment_id"),
        "verified_hashes": expected_hashes,
    }


def lifecycle_windows(timeline, variant, floor):
    selected = [
        row
        for row in timeline
        if row["variant"] == variant and row["committee_floor"] == floor
    ]
    windows = {}
    for row in selected:
        state = windows.setdefault(
            row["expert_id"],
            {
                "stream_seed": row["stream_seed"],
                "expert_image_id": row["expert_image_id"],
                "admission_step": None,
                "deletion_step": None,
            },
        )
        if row["event"] == "admitted":
            state["admission_step"] = row["step"]
        if row["deleted_this_step"]:
            state["deletion_step"] = row["step"]
    return windows


def active_experts_before_step(windows, stream_seed, step):
    return sorted(
        (
            eid,
            state["expert_image_id"],
        )
        for eid, state in windows.items()
        if state["stream_seed"] == stream_seed
        and state["admission_step"] is not None
        and state["admission_step"] < step
        and (state["deletion_step"] is None or step <= state["deletion_step"])
    )


def score_dynamic_streams(
    samples,
    features,
    stage1_records,
    timeline,
    device,
    reference_chunk_size=8,
):
    local_records = [row for row in stage1_records if row["method"] == "local"]
    rows = []
    score_cache = {}
    for variant, floor in EVALUATIONS:
        windows = lifecycle_windows(timeline, variant, floor)
        for stream_seed in sorted({row["stream_seed"] for row in local_records}):
            stream_rows = sorted(
                (row for row in local_records if row["stream_seed"] == stream_seed),
                key=lambda row: row["stream_step"],
            )
            for row in stream_rows:
                step = row["stream_step"]
                active = active_experts_before_step(windows, stream_seed, step)
                active_ids = [item[0] for item in active]
                reference_image_ids = tuple(item[1] for item in active)
                if reference_image_ids:
                    cache_key = (row["image_id"], reference_image_ids)
                    if cache_key not in score_cache:
                        score_cache[cache_key] = compute_online_dino_msm_image_score(
                            features[row["image_id"]],
                            [features[image_id] for image_id in reference_image_ids],
                            device=device,
                            reference_chunk_size=reference_chunk_size,
                        )
                    score = score_cache[cache_key]
                    source = "dynamic_expert_committee"
                else:
                    score = row["ms_score"]
                    source = (
                        "strictly_prior_history_fallback"
                        if score is not None
                        else "unavailable_first_image"
                    )
                rows.append(
                    {
                        "method": f"{variant}_floor{floor}",
                        "variant": variant,
                        "committee_floor": floor,
                        "stream_seed": stream_seed,
                        "stream_step": step,
                        "image_id": row["image_id"],
                        "image_type": samples[row["image_id"]]["kind"],
                        "gt_debug": int(samples[row["image_id"]]["is_anomaly"]),
                        "anomaly_score": score,
                        "score_source": source,
                        "active_expert_count": len(active_ids),
                        "active_expert_ids": ";".join(active_ids),
                        "reference_image_ids": ";".join(
                            str(image_id) for image_id in reference_image_ids
                        ),
                    }
                )

    for row in local_records:
        rows.append(
            {
                "method": "online_history_baseline",
                "variant": "history",
                "committee_floor": "",
                "stream_seed": row["stream_seed"],
                "stream_step": row["stream_step"],
                "image_id": row["image_id"],
                "image_type": samples[row["image_id"]]["kind"],
                "gt_debug": int(samples[row["image_id"]]["is_anomaly"]),
                "anomaly_score": row["ms_score"],
                "score_source": (
                    "strictly_prior_history"
                    if row["ms_score"] is not None
                    else "unavailable_first_image"
                ),
                "active_expert_count": 0,
                "active_expert_ids": "",
                "reference_image_ids": "",
            }
        )
    return rows


def compute_auroc(gt, scores):
    gt = np.asarray(gt, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(gt)) != 2:
        raise ValueError("AUROC requires both normal and anomalous images")
    return float(roc_auc_score(gt, scores))


def aggregate_and_evaluate(score_rows, samples):
    methods = sorted({row["method"] for row in score_rows})
    metric_rows = []
    final_rows = []
    for method in methods:
        method_rows = [row for row in score_rows if row["method"] == method]
        for stream_seed in sorted({row["stream_seed"] for row in method_rows}):
            stream_all = [
                row for row in method_rows if row["stream_seed"] == stream_seed
            ]
            available = [
                row
                for row in stream_all
                if row["anomaly_score"] is not None
            ]
            metric_rows.append(
                {
                    "method": method,
                    "scope": "stream",
                    "stream_seed": stream_seed,
                    "image_count": len(available),
                    "normal_count": sum(not row["gt_debug"] for row in available),
                    "anomaly_count": sum(row["gt_debug"] for row in available),
                    "dynamic_expert_score_count": sum(
                        row["score_source"] == "dynamic_expert_committee"
                        for row in stream_all
                    ),
                    "history_fallback_count": sum(
                        "history" in row["score_source"] for row in stream_all
                    ),
                    "unavailable_count": sum(
                        row["anomaly_score"] is None for row in stream_all
                    ),
                    "image_level_auroc": compute_auroc(
                        [row["gt_debug"] for row in available],
                        [row["anomaly_score"] for row in available],
                    ),
                }
            )

        by_image = defaultdict(list)
        for row in method_rows:
            if row["anomaly_score"] is not None:
                by_image[row["image_id"]].append(float(row["anomaly_score"]))
        if set(by_image) != set(range(len(samples))):
            missing = sorted(set(range(len(samples))) - set(by_image))
            raise RuntimeError(f"{method}: missing final scores for images {missing}")
        ensemble_rows = []
        for image_id, sample in enumerate(samples):
            values = by_image[image_id]
            final = {
                "method": method,
                "image_id": image_id,
                "image_name": sample["path"].name,
                "image_type": sample["kind"],
                "gt_debug": int(sample["is_anomaly"]),
                "final_anomaly_score": float(np.mean(values)),
                "score_std_across_streams": float(np.std(values)),
                "available_stream_count": len(values),
            }
            final_rows.append(final)
            ensemble_rows.append(final)
        metric_rows.append(
            {
                "method": method,
                "scope": "ensemble_mean",
                "stream_seed": "",
                "image_count": len(ensemble_rows),
                "normal_count": sum(not row["gt_debug"] for row in ensemble_rows),
                "anomaly_count": sum(row["gt_debug"] for row in ensemble_rows),
                "dynamic_expert_score_count": sum(
                    row["score_source"] == "dynamic_expert_committee"
                    for row in method_rows
                ),
                "history_fallback_count": sum(
                    "history" in row["score_source"] for row in method_rows
                ),
                "unavailable_count": sum(
                    row["anomaly_score"] is None for row in method_rows
                ),
                "image_level_auroc": compute_auroc(
                    [row["gt_debug"] for row in ensemble_rows],
                    [row["final_anomaly_score"] for row in ensemble_rows],
                ),
            }
        )
    return final_rows, metric_rows


def summarize_methods(metric_rows):
    summaries = []
    for method in sorted({row["method"] for row in metric_rows}):
        stream_rows = [
            row
            for row in metric_rows
            if row["method"] == method and row["scope"] == "stream"
        ]
        ensemble = next(
            row
            for row in metric_rows
            if row["method"] == method and row["scope"] == "ensemble_mean"
        )
        stream_aurocs = [row["image_level_auroc"] for row in stream_rows]
        total_steps = sum(
            row["dynamic_expert_score_count"]
            + row["history_fallback_count"]
            + row["unavailable_count"]
            for row in stream_rows
        )
        summaries.append(
            {
                "method": method,
                "mean_stream_auroc": float(np.mean(stream_aurocs)),
                "std_stream_auroc": float(np.std(stream_aurocs)),
                "ensemble_image_auroc": ensemble["image_level_auroc"],
                "dynamic_expert_score_count": sum(
                    row["dynamic_expert_score_count"] for row in stream_rows
                ),
                "history_fallback_count": sum(
                    row["history_fallback_count"] for row in stream_rows
                ),
                "unavailable_first_image_count": sum(
                    row["unavailable_count"] for row in stream_rows
                ),
                "dynamic_expert_coverage": (
                    sum(row["dynamic_expert_score_count"] for row in stream_rows)
                    / total_steps
                    if total_steps
                    else None
                ),
            }
        )
    return summaries


def plot_roc_curves(path, final_rows, metric_rows):
    fig, axis = plt.subplots(figsize=(8, 7))
    methods = sorted({row["method"] for row in final_rows})
    ensemble_metrics = {
        row["method"]: row["image_level_auroc"]
        for row in metric_rows
        if row["scope"] == "ensemble_mean"
    }
    for method in methods:
        rows = [row for row in final_rows if row["method"] == method]
        fpr, tpr, _ = roc_curve(
            [row["gt_debug"] for row in rows],
            [row["final_anomaly_score"] for row in rows],
        )
        axis.plot(fpr, tpr, label=f"{method} ({ensemble_metrics[method]:.3f})")
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("false positive rate")
    axis.set_ylabel("true positive rate")
    axis.set_title("MVTec AD bottle: image-level ROC")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_score_distributions(path, final_rows):
    methods = sorted({row["method"] for row in final_rows})
    fig, axes = plt.subplots(len(methods), 1, figsize=(10, 3.2 * len(methods)))
    axes = np.atleast_1d(axes)
    for axis, method in zip(axes, methods):
        rows = [row for row in final_rows if row["method"] == method]
        normal = [row["final_anomaly_score"] for row in rows if not row["gt_debug"]]
        anomaly = [row["final_anomaly_score"] for row in rows if row["gt_debug"]]
        axis.hist(normal, bins=15, alpha=0.65, density=True, label="normal")
        axis.hist(anomaly, bins=15, alpha=0.65, density=True, label="anomaly")
        axis.set_title(method)
        axis.set_xlabel("final anomaly score")
        axis.set_ylabel("density")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def validate_outputs(score_rows, final_rows, metric_rows, samples):
    errors = []
    methods = sorted({row["method"] for row in score_rows})
    for method in methods:
        rows = [row for row in final_rows if row["method"] == method]
        if len(rows) != len(samples):
            errors.append(f"{method}: final score count is not {len(samples)}")
        if any(not np.isfinite(row["final_anomaly_score"]) for row in rows):
            errors.append(f"{method}: non-finite final anomaly score")
        if any(row["available_stream_count"] < 2 for row in rows):
            errors.append(f"{method}: an image has fewer than two available stream scores")
    if any(
        not 0.0 <= row["image_level_auroc"] <= 1.0 for row in metric_rows
    ):
        errors.append("AUROC is outside [0, 1]")
    for row in score_rows:
        if row["score_source"] == "dynamic_expert_committee":
            current_id = str(row["image_id"])
            if current_id in row["reference_image_ids"].split(";"):
                errors.append(
                    f"self-reference detected: {row['method']} seed={row['stream_seed']} "
                    f"step={row['stream_step']}"
                )
                break
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--features-cache", required=True)
    parser.add_argument("--stage1-records", required=True)
    parser.add_argument("--stage3-experts", required=True)
    parser.add_argument("--lifecycle-timeline", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--reference-chunk-size", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.data_root)
    features = load_features(args.features_cache)
    stage1_records = read_stage1_records(args.stage1_records)
    experts = read_stage3_experts(args.stage3_experts)
    timeline = read_lifecycle_timeline(args.lifecycle_timeline)
    manifest = json.loads(Path(args.input_manifest).read_text(encoding="utf-8"))
    input_validation = validate_inputs(
        samples,
        features,
        stage1_records,
        experts,
        timeline,
        manifest,
        args.features_cache,
        args.stage1_records,
        args.stage3_experts,
    )
    if input_validation["errors"]:
        raise RuntimeError(
            "Input validation failed: " + "; ".join(input_validation["errors"])
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_rows = score_dynamic_streams(
        samples,
        features,
        stage1_records,
        timeline,
        device,
        reference_chunk_size=args.reference_chunk_size,
    )
    final_rows, metric_rows = aggregate_and_evaluate(score_rows, samples)
    method_summaries = summarize_methods(metric_rows)
    errors = validate_outputs(score_rows, final_rows, metric_rows, samples)

    write_csv(output_dir / "bottle_dynamic_scores_by_stream.csv", score_rows)
    write_csv(output_dir / "bottle_dynamic_final_scores.csv", final_rows)
    write_csv(
        output_dir / "bottle_final_anomaly_scores.csv",
        [row for row in final_rows if row["method"] == "channel_ttl_floor0"],
    )
    write_csv(output_dir / "bottle_dynamic_image_auroc.csv", metric_rows)
    write_csv(output_dir / "bottle_dynamic_method_summary.csv", method_summaries)
    plot_roc_curves(output_dir / "bottle_dynamic_image_roc.png", final_rows, metric_rows)
    plot_score_distributions(
        output_dir / "bottle_dynamic_score_distributions.png", final_rows
    )

    summary = {
        "experiment": "Current bottle DINO dynamic EX-MSM image scoring",
        "device": str(device),
        "input_validation": input_validation,
        "scoring": {
            "reference_timing": "experts alive at the start of the current step",
            "patch_score": "mean nearest 30% reference-image distances",
            "image_score": "maximum patch score",
            "cold_start": "strictly-prior online history score; first image unavailable",
            "final_score": "mean available score for each image across three streams",
            "gt_usage": "AUROC only",
            "not_included": ["CLIP", "MSM2 fusion", "RsCIN", "pixel AUROC"],
        },
        "primary_method": "channel_ttl_floor0",
        "validation_errors": errors,
        "method_summary": method_summaries,
        "metrics": metric_rows,
    }
    (output_dir / "bottle_dynamic_image_auroc.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        raise RuntimeError("Output validation failed: " + "; ".join(errors))


if __name__ == "__main__":
    main()
