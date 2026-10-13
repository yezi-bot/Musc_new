import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from MultiMuSc2.datasets.mvtec import DatasetSplit, MVTecDataset
from MultiMuSc2.examples.run_dynamic_dino import (
    DinoFeatureExtractor,
    load_dinov2,
    seed_everything,
)
from MultiMuSc2.models.modules._DYNAMIC_DINO_STREAM import (
    chunked_dino_score,
    feature_key,
)
from MultiMuSc2.utils.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Offline GT-only audit of contaminated experts in the Bottle "
            "latest_only runs."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--dinov2-repo", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--score-tolerance", type=float, default=5e-4)
    return parser.parse_args()


def normalized_path(value):
    return os.path.normcase(os.path.normpath(str(value)))


def dataset_index_by_relative_path(dataset, data_root):
    mapping = {}
    for index, (_, _, image_path, _) in enumerate(dataset.data_to_iterate):
        relative = os.path.relpath(image_path, data_root)
        key = normalized_path(relative)
        if key in mapping:
            raise ValueError(f"duplicate dataset path: {relative}")
        mapping[key] = index
    return mapping


def score_map(
    query_features,
    reference_features,
    feature_layers,
    r_list,
    device,
    image_size,
    topmin_min,
    topmin_max,
    reference_chunk_size,
):
    if not reference_features:
        raise ValueError("offline replay requires at least one reference")
    layer_scores = []
    for r in r_list:
        for layer in feature_layers:
            key = feature_key(r, layer)
            layer_scores.append(
                chunked_dino_score(
                    query=query_features[key],
                    references=[features[key] for features in reference_features],
                    device=device,
                    topmin_min=topmin_min,
                    topmin_max=topmin_max,
                    chunk_size=reference_chunk_size,
                )
            )
    patch_score = torch.stack(layer_scores).mean(dim=0)
    grid_size = math.isqrt(patch_score.numel())
    if grid_size * grid_size != patch_score.numel():
        raise ValueError("patch count must form a square grid")
    anomaly_map = F.interpolate(
        patch_score.reshape(1, 1, grid_size, grid_size),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=True,
    )[0, 0]
    return anomaly_map.detach().cpu()


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


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def phase_score_summary(rows, phase):
    selected = [row for row in rows if row["phase"] == phase]
    result = {"step_count": len(selected)}
    for label_name, is_anomaly in (("normal", False), ("anomaly", True)):
        label_rows = [
            row for row in selected if row["is_anomaly"] is is_anomaly
        ]
        result[f"{label_name}_count"] = len(label_rows)
        result[f"{label_name}_original_score_mean"] = mean_or_none(
            [row["original_score"] for row in label_rows]
        )
        result[f"{label_name}_oracle_score_mean"] = mean_or_none(
            [row["oracle_score"] for row in label_rows]
        )
        result[f"{label_name}_oracle_delta_mean"] = mean_or_none(
            [row["oracle_score"] - row["original_score"] for row in label_rows]
        )
    return result


def contaminated_expert_rows(timeline):
    admissions = []
    for record in timeline:
        for event in record.get("expert_lifecycle_events", []):
            if (
                event.get("decision") == "admitted"
                and event.get("expert_is_anomaly_debug") is True
            ):
                admissions.append(event)

    rows = []
    for event in admissions:
        image_id = int(event["expert_image_id"])
        referenced_steps = [
            int(record["step"])
            for record in timeline
            if image_id in {
                int(value)
                for value in record.get("scoring_reference_image_ids", [])
            }
            and record.get("fallback_reason") is None
        ]
        rows.append(
            {
                "expert_id": int(event["expert_id"]),
                "expert_image_id": image_id,
                "anomaly_type": event.get("expert_anomaly_type_debug"),
                "admission_step": int(event["admission_step"]),
                "reference_step_count": len(referenced_steps),
                "first_reference_step": (
                    min(referenced_steps) if referenced_steps else None
                ),
                "last_reference_step": (
                    max(referenced_steps) if referenced_steps else None
                ),
            }
        )
    return rows


def run_seed(
    seed,
    timeline_path,
    metrics_path,
    dataset,
    dataset_mapping,
    extractor,
    device,
    config,
    score_tolerance,
):
    seed_everything(seed)
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    if [int(record["step"]) for record in timeline] != list(
        range(len(timeline))
    ):
        raise ValueError("timeline steps must be contiguous and ordered")
    stored_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    feature_layers = [int(value) for value in config["feature_layers"]]
    r_list = [int(value) for value in config["r_list"]]
    scoring = config["scoring"]
    image_size = int(config["image_size"])

    last_reference_step = {}
    for record in timeline:
        step = int(record["step"])
        for image_id in record.get("scoring_reference_image_ids", []):
            last_reference_step[int(image_id)] = step

    feature_bank = {}
    rows = []
    labels = []
    masks = []
    original_maps = []
    oracle_maps = []
    max_score_error = 0.0
    expert_scored_steps = 0
    contaminated_exposure_steps = 0
    expert_reference_slots = 0
    contaminated_reference_slots = 0

    contaminated_steps = []
    for record in timeline:
        if record.get("fallback_reason") is not None:
            continue
        reference_ids = [
            int(value) for value in record.get("scoring_reference_image_ids", [])
        ]
        if not reference_ids:
            continue
        expert_scored_steps += 1
        contaminated = [
            image_id
            for image_id in reference_ids
            if timeline[image_id]["anomaly_type"] != "good"
        ]
        expert_reference_slots += len(reference_ids)
        contaminated_reference_slots += len(contaminated)
        if contaminated:
            contaminated_exposure_steps += 1
            contaminated_steps.append(int(record["step"]))

    first_contaminated_step = (
        min(contaminated_steps) if contaminated_steps else None
    )
    last_contaminated_step = (
        max(contaminated_steps) if contaminated_steps else None
    )

    for record in timeline:
        step = int(record["step"])
        relative_path = normalized_path(record["image_path"])
        if relative_path not in dataset_mapping:
            raise ValueError(f"timeline image not found in dataset: {relative_path}")
        sample = dataset[dataset_mapping[relative_path]]
        current_features = extractor.extract(sample["image"].unsqueeze(0))

        reference_ids = [
            int(value) for value in record.get("scoring_reference_image_ids", [])
        ]
        if not record["available"]:
            if reference_ids:
                raise ValueError("unavailable step unexpectedly has references")
        else:
            references = [feature_bank[image_id] for image_id in reference_ids]
            topmin_min = (
                0.0
                if record.get("fallback_reason") == "strict_history_dino"
                else float(scoring["topmin_min"])
            )
            original_map = score_map(
                current_features,
                references,
                feature_layers,
                r_list,
                device,
                image_size,
                topmin_min,
                float(scoring["topmin_max"]),
                int(scoring["reference_chunk_size"]),
            )
            original_score = float(original_map.max())
            score_error = abs(original_score - float(record["image_score"]))
            max_score_error = max(max_score_error, score_error)
            if score_error > score_tolerance:
                raise RuntimeError(
                    f"seed {seed} step {step} replay score error "
                    f"{score_error:.6g} exceeds tolerance {score_tolerance}"
                )

            contaminated_reference_ids = (
                [
                    image_id
                    for image_id in reference_ids
                    if timeline[image_id]["anomaly_type"] != "good"
                ]
                if record.get("fallback_reason") is None
                else []
            )
            clean_reference_ids = [
                image_id
                for image_id in reference_ids
                if image_id not in contaminated_reference_ids
            ]
            if contaminated_reference_ids:
                if record.get("fallback_reason") is not None:
                    oracle_map = original_map
                else:
                    if not clean_reference_ids:
                        raise RuntimeError(
                            "oracle removal left an empty active committee"
                        )
                    oracle_map = score_map(
                        current_features,
                        [feature_bank[value] for value in clean_reference_ids],
                        feature_layers,
                        r_list,
                        device,
                        image_size,
                        float(scoring["topmin_min"]),
                        float(scoring["topmin_max"]),
                        int(scoring["reference_chunk_size"]),
                    )
            else:
                oracle_map = original_map

            oracle_score = float(oracle_map.max())
            is_anomaly = record["anomaly_type"] != "good"
            if contaminated_reference_ids:
                phase = "contaminated_exposure"
            elif (
                first_contaminated_step is not None
                and step < first_contaminated_step
            ):
                phase = "before_contamination"
            elif (
                last_contaminated_step is not None
                and step > last_contaminated_step
            ):
                phase = "after_contamination"
            else:
                phase = "clean_or_fallback"

            rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "image_path": record["image_path"],
                    "anomaly_type": record["anomaly_type"],
                    "is_anomaly": is_anomaly,
                    "phase": phase,
                    "fallback_reason": record.get("fallback_reason"),
                    "reference_count": len(reference_ids),
                    "contaminated_reference_count": len(
                        contaminated_reference_ids
                    ),
                    "contaminated_reference_ids": " ".join(
                        str(value) for value in contaminated_reference_ids
                    ),
                    "original_score": original_score,
                    "oracle_score": oracle_score,
                    "oracle_score_delta": oracle_score - original_score,
                }
            )
            labels.append(int(is_anomaly))
            masks.append((sample["mask"] > 0.5).numpy().astype(np.int32))
            original_maps.append(original_map.numpy())
            oracle_maps.append(oracle_map.numpy())

        if step in last_reference_step:
            feature_bank[step] = current_features
        del current_features
        expired = [
            image_id
            for image_id, last_step in last_reference_step.items()
            if last_step == step and image_id in feature_bank
        ]
        for image_id in expired:
            del feature_bank[image_id]

    gt_sp = np.asarray(labels, dtype=np.int32)
    gt_px = np.stack(masks)
    original_image_metric, original_pixel_metric = compute_metrics(
        gt_sp=gt_sp,
        pr_sp=np.asarray([row["original_score"] for row in rows]),
        gt_px=gt_px,
        pr_px=np.stack(original_maps),
    )
    oracle_image_metric, oracle_pixel_metric = compute_metrics(
        gt_sp=gt_sp,
        pr_sp=np.asarray([row["oracle_score"] for row in rows]),
        gt_px=gt_px,
        pr_px=np.stack(oracle_maps),
    )
    original_metrics = metric_dict(original_image_metric, original_pixel_metric)
    oracle_metrics = metric_dict(oracle_image_metric, oracle_pixel_metric)
    metric_deltas = {
        key: oracle_metrics[key] - original_metrics[key]
        for key in original_metrics
    }
    stored_metric_errors = {
        key: original_metrics[key] - float(stored_metrics[key])
        for key in original_metrics
    }

    result = {
        "seed": seed,
        "available_steps": len(rows),
        "expert_scored_steps": expert_scored_steps,
        "contaminated_exposure_steps": contaminated_exposure_steps,
        "contaminated_exposure_fraction_of_expert_steps": (
            contaminated_exposure_steps / expert_scored_steps
            if expert_scored_steps
            else 0.0
        ),
        "expert_reference_slots": expert_reference_slots,
        "contaminated_reference_slots": contaminated_reference_slots,
        "contaminated_reference_slot_fraction": (
            contaminated_reference_slots / expert_reference_slots
            if expert_reference_slots
            else 0.0
        ),
        "first_contaminated_exposure_step": first_contaminated_step,
        "last_contaminated_exposure_step": last_contaminated_step,
        "contaminated_experts": contaminated_expert_rows(timeline),
        "max_replay_image_score_error": max_score_error,
        "stored_metric_errors": stored_metric_errors,
        "original_metrics": original_metrics,
        "oracle_metrics": oracle_metrics,
        "oracle_minus_original": metric_deltas,
        "score_phases": {
            phase: phase_score_summary(rows, phase)
            for phase in (
                "before_contamination",
                "contaminated_exposure",
                "after_contamination",
                "clean_or_fallback",
            )
        },
    }
    return result, rows


def aggregate_results(results):
    metric_names = list(results[0]["original_metrics"])
    expert_steps = sum(result["expert_scored_steps"] for result in results)
    reference_slots = sum(
        result["expert_reference_slots"] for result in results
    )
    return {
        "seed_count": len(results),
        "total_expert_scored_steps": expert_steps,
        "total_contaminated_exposure_steps": sum(
            result["contaminated_exposure_steps"] for result in results
        ),
        "pooled_contaminated_exposure_fraction": (
            sum(result["contaminated_exposure_steps"] for result in results)
            / expert_steps
            if expert_steps
            else 0.0
        ),
        "total_expert_reference_slots": reference_slots,
        "total_contaminated_reference_slots": sum(
            result["contaminated_reference_slots"] for result in results
        ),
        "pooled_contaminated_reference_slot_fraction": (
            sum(result["contaminated_reference_slots"] for result in results)
            / reference_slots
            if reference_slots
            else 0.0
        ),
        "original_metric_means": {
            key: float(np.mean([result["original_metrics"][key] for result in results]))
            for key in metric_names
        },
        "oracle_metric_means": {
            key: float(np.mean([result["oracle_metrics"][key] for result in results]))
            for key in metric_names
        },
        "oracle_minus_original_means": {
            key: float(
                np.mean(
                    [result["oracle_minus_original"][key] for result in results]
                )
            )
            for key in metric_names
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
    summary_path = Path(args.summary).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    resolved = summary["resolved_configs"]["latest_only"]
    models = resolved["models"]
    checkpoint = Path(args.checkpoint or summary["checkpoint"]).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_dinov2(args.dinov2_repo, checkpoint, device)
    extractor = DinoFeatureExtractor(
        model=model,
        device=device,
        feature_layers=models["feature_layers"],
        r_list=models["r_list"],
    )
    dataset = MVTecDataset(
        source=args.data_root,
        classname=resolved["datasets"]["class_name"],
        resize=int(resolved["datasets"]["img_resize"]),
        imagesize=int(resolved["datasets"]["img_resize"]),
        split=DatasetSplit.TEST,
    )
    dataset_mapping = dataset_index_by_relative_path(dataset, args.data_root)
    config = {
        "feature_layers": models["feature_layers"],
        "r_list": models["r_list"],
        "scoring": models["scoring"],
        "image_size": int(resolved["datasets"]["img_resize"]),
    }

    results = []
    all_rows = []
    for seed in args.seeds:
        print(f"Replaying seed {seed}...", flush=True)
        result, rows = run_seed(
            seed=seed,
            timeline_path=experiment_root / f"seed_{seed}" / "timeline.json",
            metrics_path=experiment_root / f"seed_{seed}" / "metrics.json",
            dataset=dataset,
            dataset_mapping=dataset_mapping,
            extractor=extractor,
            device=device,
            config=config,
            score_tolerance=args.score_tolerance,
        )
        results.append(result)
        all_rows.extend(rows)
        print(
            f"Seed {seed} complete: "
            f"exposure_steps={result['contaminated_exposure_steps']}, "
            f"image_auroc_delta="
            f"{result['oracle_minus_original']['image_auroc']:.6f}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = {
        "analysis_scope": "offline_gt_oracle_remove_abnormal_active_experts",
        "gt_usage": (
            "GT is used only after the original online run to label and remove "
            "contaminated expert references. It is not an online method."
        ),
        "source_summary": str(summary_path),
        "source_experiment_root": str(experiment_root),
        "data_root": str(Path(args.data_root).resolve()),
        "dinov2_repo": str(Path(args.dinov2_repo).resolve()),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "config": config,
        "aggregate": aggregate_results(results),
        "seeds": results,
    }
    (output_dir / "contamination_impact.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "contamination_steps.csv", all_rows)


if __name__ == "__main__":
    main()
