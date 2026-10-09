import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from MultiMuSc2.datasets.mvtec import DatasetSplit, MVTecDataset
from MultiMuSc2.examples.run_dynamic_dino import (
    DinoFeatureExtractor,
    git_commit,
    load_dinov2,
    select_dataset,
    sha256_file,
)
from MultiMuSc2.models.modules._CAUSAL_POSITION_LOF import (
    CausalPositionLofMemory,
)
from MultiMuSc2.models.modules._DYNAMIC_DINO_STREAM import feature_key


CSV_FIELDS = [
    "seed",
    "step",
    "image_path",
    "anomaly_type_debug",
    "is_anomaly_debug",
    "admitted_by",
    "expert_ids_by_variant",
    "position_radius",
    "patch_count",
    "anomaly_patch_count_debug",
    "anomaly_patch_fraction_debug",
    "mean",
    "median",
    "q90",
    "q95",
    "q99",
    "max",
    "top01_mean",
    "top05_mean",
    "top10_mean",
    "top15_mean",
    "fraction_gt_1_0",
    "fraction_gt_1_2",
    "fraction_gt_1_5",
    "fraction_gt_2_0",
    "patch_auroc_debug",
    "patch_ap_debug",
    "top01_gt_precision_debug",
    "top05_gt_precision_debug",
    "top15_gt_precision_debug",
    "npz_path",
    "figure_path",
]

SUMMARY_METRICS = [
    "mean",
    "median",
    "q90",
    "q95",
    "q99",
    "max",
    "top01_mean",
    "top05_mean",
    "top10_mean",
    "top15_mean",
    "fraction_gt_1_0",
    "fraction_gt_1_2",
    "fraction_gt_1_5",
    "fraction_gt_2_0",
]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose patch-level causal LOF for the union of admitted experts."
        )
    )
    parser.add_argument(
        "--config",
        default=PROJECT_ROOT / "configs" / "dynamic_dino.yaml",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dinov2-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--timeline-root",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--position-radii",
        nargs="+",
        type=int,
        default=[0, 1],
    )
    return parser.parse_args()


def parse_named_paths(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"timeline root must use NAME=PATH syntax: {value}"
            )
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if "@" in label:
            name, raw_seed = label.rsplit("@", 1)
            try:
                seed = int(raw_seed)
            except ValueError as error:
                raise ValueError(
                    f"invalid seed override in timeline label: {label}"
                ) from error
        else:
            name = label
            seed = None
        name = name.strip()
        key = (name, seed)
        if not name or key in result:
            raise ValueError(f"invalid or duplicate timeline label: {label}")
        path = Path(raw_path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"timeline root does not exist: {path}")
        result[key] = path
    return result


def normalized_relative_path(value):
    return str(value).replace("\\", "/").casefold()


def read_timeline(root, seed):
    path = root / f"seed_{seed}" / "timeline.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing timeline: {path}")
    timeline = json.loads(path.read_text(encoding="utf-8"))
    if not timeline:
        raise ValueError(f"timeline is empty: {path}")
    for expected_step, record in enumerate(timeline):
        if int(record["step"]) != expected_step:
            raise ValueError(f"timeline steps are not contiguous: {path}")
    return timeline


def timeline_roots_for_seed(timeline_roots, seed):
    names = {name for name, _ in timeline_roots}
    selected = {}
    for name in names:
        root = timeline_roots.get(
            (name, seed),
            timeline_roots.get((name, None)),
        )
        if root is None:
            raise ValueError(
                f"no timeline root configured for {name}, seed {seed}"
            )
        selected[name] = root
    return selected


def collect_admission_union(timeline_roots, seed):
    selected_roots = timeline_roots_for_seed(timeline_roots, seed)
    timelines = {
        name: read_timeline(root, seed)
        for name, root in selected_roots.items()
    }
    lengths = {len(timeline) for timeline in timelines.values()}
    if len(lengths) != 1:
        raise ValueError(f"timeline lengths differ for seed {seed}")

    reference_name = next(iter(timelines))
    reference = timelines[reference_name]
    targets = {}
    for name, timeline in timelines.items():
        for step, (expected, record) in enumerate(zip(reference, timeline)):
            if normalized_relative_path(expected["image_path"]) != (
                normalized_relative_path(record["image_path"])
            ):
                raise ValueError(
                    f"timeline image order differs at seed {seed}, step {step}"
                )
            expert_id = record.get("admitted_expert_id")
            if expert_id is None:
                continue
            target = targets.setdefault(
                step,
                {
                    "image_path": record["image_path"],
                    "anomaly_type_debug": record.get("anomaly_type"),
                    "expert_ids_by_variant": {},
                },
            )
            target["expert_ids_by_variant"][name] = int(expert_id)
    return reference, targets


def patch_mask_from_pixel_mask(mask, grid_size):
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError("mask must have shape [1, height, width]")
    pooled = F.adaptive_max_pool2d(
        mask.float().unsqueeze(0),
        output_size=(grid_size, grid_size),
    )
    return pooled[0, 0].reshape(-1) > 0.5


def top_mean(scores, fraction):
    count = max(1, int(math.ceil(scores.numel() * fraction)))
    return float(torch.topk(scores, k=count).values.mean())


def top_gt_precision(scores, patch_mask, fraction):
    count = max(1, int(math.ceil(scores.numel() * fraction)))
    indices = torch.topk(scores, k=count).indices
    return float(patch_mask[indices].float().mean())


def patch_score_summary(scores, patch_mask):
    scores = scores.detach().float().cpu().reshape(-1)
    patch_mask = patch_mask.detach().bool().cpu().reshape(-1)
    if scores.shape != patch_mask.shape:
        raise ValueError("score and mask shapes differ")
    if not torch.isfinite(scores).all():
        raise ValueError("patch LOF contains non-finite values")

    quantiles = torch.quantile(
        scores,
        torch.tensor([0.90, 0.95, 0.99]),
    )
    result = {
        "patch_count": int(scores.numel()),
        "anomaly_patch_count_debug": int(patch_mask.sum()),
        "anomaly_patch_fraction_debug": float(patch_mask.float().mean()),
        "mean": float(scores.mean()),
        "median": float(scores.median()),
        "q90": float(quantiles[0]),
        "q95": float(quantiles[1]),
        "q99": float(quantiles[2]),
        "max": float(scores.max()),
        "top01_mean": top_mean(scores, 0.01),
        "top05_mean": top_mean(scores, 0.05),
        "top10_mean": top_mean(scores, 0.10),
        "top15_mean": top_mean(scores, 0.15),
        "fraction_gt_1_0": float((scores > 1.0).float().mean()),
        "fraction_gt_1_2": float((scores > 1.2).float().mean()),
        "fraction_gt_1_5": float((scores > 1.5).float().mean()),
        "fraction_gt_2_0": float((scores > 2.0).float().mean()),
        "patch_auroc_debug": None,
        "patch_ap_debug": None,
        "top01_gt_precision_debug": top_gt_precision(
            scores, patch_mask, 0.01
        ),
        "top05_gt_precision_debug": top_gt_precision(
            scores, patch_mask, 0.05
        ),
        "top15_gt_precision_debug": top_gt_precision(
            scores, patch_mask, 0.15
        ),
    }
    labels = patch_mask.numpy().astype(np.int32)
    if labels.min() != labels.max():
        values = scores.numpy()
        result["patch_auroc_debug"] = float(
            roc_auc_score(labels, values)
        )
        result["patch_ap_debug"] = float(
            average_precision_score(labels, values)
        )
    return result


def denormalize_image(image):
    restored = image.detach().float().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return restored.clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def save_diagnostic_figure(
    path,
    image,
    scores,
    patch_mask,
    grid_size,
    title,
):
    score_map = scores.reshape(grid_size, grid_size).numpy()
    mask_map = patch_mask.reshape(grid_size, grid_size).numpy()
    top_count = max(1, int(math.ceil(scores.numel() * 0.05)))
    top_mask = torch.zeros_like(scores, dtype=torch.bool)
    top_mask[torch.topk(scores, k=top_count).indices] = True
    top_map = top_mask.reshape(grid_size, grid_size).numpy()
    image_np = denormalize_image(image)

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    heat = axes[1].imshow(score_map, cmap="magma")
    axes[1].set_title("Patch LOF")
    figure.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(mask_map, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("GT patch mask (debug)")
    axes[3].imshow(mask_map, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(
        np.ma.masked_where(~top_map, top_map),
        cmap="autumn",
        alpha=0.8,
        vmin=0,
        vmax=1,
    )
    axes[3].set_title("Top 5% LOF over GT")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_csv(path, rows, fields=CSV_FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def unique_image_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["image_path"], row["position_radius"])].append(row)

    result = []
    for (image_path, radius), group in sorted(grouped.items()):
        first = group[0]
        row = {
            "seed": ";".join(str(item["seed"]) for item in group),
            "step": ";".join(str(item["step"]) for item in group),
            "image_path": image_path,
            "anomaly_type_debug": first["anomaly_type_debug"],
            "is_anomaly_debug": first["is_anomaly_debug"],
            "admitted_by": ";".join(
                sorted(
                    {
                        name
                        for item in group
                        for name in item["admitted_by"].split(";")
                        if name
                    }
                )
            ),
            "expert_ids_by_variant": "",
            "position_radius": radius,
            "patch_count": first["patch_count"],
            "anomaly_patch_count_debug": first[
                "anomaly_patch_count_debug"
            ],
            "anomaly_patch_fraction_debug": first[
                "anomaly_patch_fraction_debug"
            ],
            "npz_path": "",
            "figure_path": "",
        }
        for metric in SUMMARY_METRICS:
            row[metric] = float(
                np.mean([float(item[metric]) for item in group])
            )
        for metric in (
            "patch_auroc_debug",
            "patch_ap_debug",
            "top01_gt_precision_debug",
            "top05_gt_precision_debug",
            "top15_gt_precision_debug",
        ):
            values = [
                float(item[metric])
                for item in group
                if item[metric] not in (None, "")
            ]
            row[metric] = float(np.mean(values)) if values else None
        result.append(row)
    return result


def group_summary(rows):
    summary = {}
    for radius in sorted({int(row["position_radius"]) for row in rows}):
        radius_rows = [
            row for row in rows if int(row["position_radius"]) == radius
        ]
        groups = {}
        for label, is_anomaly in (("normal", 0), ("abnormal", 1)):
            selected = [
                row
                for row in radius_rows
                if int(row["is_anomaly_debug"]) == is_anomaly
            ]
            values = {"count": len(selected)}
            for metric in SUMMARY_METRICS:
                metric_values = [float(row[metric]) for row in selected]
                values[f"{metric}_mean"] = (
                    float(np.mean(metric_values))
                    if metric_values
                    else None
                )
                values[f"{metric}_median"] = (
                    float(np.median(metric_values))
                    if metric_values
                    else None
                )
            groups[label] = values

        expert_metrics = {}
        labels = np.asarray(
            [int(row["is_anomaly_debug"]) for row in radius_rows]
        )
        if labels.size and labels.min() != labels.max():
            for metric in SUMMARY_METRICS:
                values = np.asarray(
                    [float(row[metric]) for row in radius_rows]
                )
                expert_metrics[metric] = {
                    "expert_image_auroc_debug": float(
                        roc_auc_score(labels, values)
                    ),
                    "expert_image_ap_debug": float(
                        average_precision_score(labels, values)
                    ),
                }
        summary[f"radius_{radius}"] = {
            "groups": groups,
            "expert_metrics_debug": expert_metrics,
        }
    return summary


def run_seed(
    seed,
    config,
    args,
    timeline_roots,
    model,
    device,
    output_dir,
):
    reference_timeline, targets = collect_admission_union(
        timeline_roots,
        seed,
    )
    dataset = MVTecDataset(
        source=args.data_root,
        classname=config["datasets"]["class_name"],
        resize=int(config["datasets"]["img_resize"]),
        imagesize=int(config["datasets"]["img_resize"]),
        split=DatasetSplit.TEST,
    )
    dataset = select_dataset(dataset, seed, max_samples=None)
    if len(dataset) != len(reference_timeline):
        raise ValueError(
            f"dataset and timeline lengths differ for seed {seed}"
        )

    committee = config["models"]["dynamic_committee"]
    committee_r = int(committee["r"])
    committee_layer = int(committee["dino_layer"])
    extractor = DinoFeatureExtractor(
        model=model,
        device=device,
        feature_layers=[committee_layer],
        r_list=[committee_r],
    )
    memory = CausalPositionLofMemory(
        k=int(committee.get("lof_k", 6)),
        device=device,
        position_chunk_size=int(
            committee.get("lof_position_chunk_size", 64)
        ),
        image_chunk_size=int(
            committee.get("lof_image_chunk_size", 8)
        ),
    )
    key = feature_key(committee_r, committee_layer)
    rows = []
    seed_dir = output_dir / f"seed_{seed}"
    values_dir = seed_dir / "values"
    figures_dir = seed_dir / "figures"
    values_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for step in range(len(dataset)):
        sample = dataset[step]
        relative_path = os.path.relpath(
            sample["image_path"],
            args.data_root,
        )
        expected_path = reference_timeline[step]["image_path"]
        if normalized_relative_path(relative_path) != (
            normalized_relative_path(expected_path)
        ):
            raise ValueError(
                f"dataset order differs at seed {seed}, step {step}"
            )

        features = extractor.extract(sample["image"].unsqueeze(0))[key]
        if step in targets:
            patch_count = int(features.shape[0])
            grid_size = math.isqrt(patch_count)
            if grid_size * grid_size != patch_count:
                raise ValueError("patch count must form a square grid")
            target = targets[step]
            admitted_by = sorted(target["expert_ids_by_variant"])
            anomaly_type = Path(sample["image_path"]).parent.name

            scores_by_radius = {}
            for radius in args.position_radii:
                scores = memory.score(
                    features,
                    position_radius=radius,
                )
                if scores is None:
                    raise RuntimeError(
                        f"LOF unavailable at admitted step {step}"
                    )
                scores_by_radius[radius] = scores

            # GT is read only after all causal LOF scores have been computed.
            patch_mask = patch_mask_from_pixel_mask(
                sample["mask"],
                grid_size,
            )
            for radius, scores in scores_by_radius.items():
                stem = (
                    f"seed_{seed}_step_{step:04d}_{anomaly_type}_r{radius}"
                )
                npz_path = values_dir / f"{stem}.npz"
                figure_path = figures_dir / f"{stem}.png"
                np.savez_compressed(
                    npz_path,
                    lof_scores=scores.numpy().reshape(grid_size, grid_size),
                    gt_patch_mask_debug=patch_mask.numpy().reshape(
                        grid_size, grid_size
                    ),
                    image_path=str(relative_path),
                    seed=seed,
                    step=step,
                    position_radius=radius,
                )
                save_diagnostic_figure(
                    figure_path,
                    sample["image"],
                    scores,
                    patch_mask,
                    grid_size,
                    (
                        f"seed={seed}, step={step}, type={anomaly_type}, "
                        f"radius={radius}, admitted_by={','.join(admitted_by)}"
                    ),
                )
                row = {
                    "seed": seed,
                    "step": step,
                    "image_path": relative_path,
                    "anomaly_type_debug": anomaly_type,
                    "is_anomaly_debug": int(sample["is_anomaly"]),
                    "admitted_by": ";".join(admitted_by),
                    "expert_ids_by_variant": json.dumps(
                        target["expert_ids_by_variant"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "position_radius": radius,
                    **patch_score_summary(scores, patch_mask),
                    "npz_path": os.path.relpath(npz_path, output_dir),
                    "figure_path": os.path.relpath(
                        figure_path, output_dir
                    ),
                }
                rows.append(row)
        memory.update(features)

    if len(rows) != len(targets) * len(args.position_radii):
        raise RuntimeError("diagnostic output count does not match targets")
    write_csv(seed_dir / "expert_patch_lof_summary.csv", rows)
    return rows, sorted(targets)


def validate_config(config, position_radii):
    committee = config["models"]["dynamic_committee"]
    if int(committee["r"]) != 1 or int(committee["dino_layer"]) != 23:
        raise ValueError("diagnostic requires committee r=1 and DINO layer 23")
    if any(radius < 0 for radius in position_radii):
        raise ValueError("position radii must be non-negative")
    if len(set(position_radii)) != len(position_radii):
        raise ValueError("position radii must be unique")


def main():
    args = parse_args()
    timeline_roots = parse_named_paths(args.timeline_root)
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config, args.position_radii)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_dinov2(
        args.dinov2_repo,
        args.checkpoint,
        device,
    )

    started = time.perf_counter()
    all_rows = []
    target_steps = {}
    for seed in args.seeds:
        rows, steps = run_seed(
            seed,
            config,
            args,
            timeline_roots,
            model,
            device,
            output_dir,
        )
        all_rows.extend(rows)
        target_steps[str(seed)] = steps

    write_csv(output_dir / "expert_patch_lof_summary.csv", all_rows)
    unique_rows = unique_image_rows(all_rows)
    write_csv(output_dir / "expert_patch_lof_unique_images.csv", unique_rows)
    metadata = {
        "git_commit": git_commit(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu"
        ),
        "seeds": args.seeds,
        "position_radii": args.position_radii,
        "timeline_roots": {
            (
                name if seed is None else f"{name}@{seed}"
            ): str(path)
            for (name, seed), path in timeline_roots.items()
        },
        "target_steps": target_steps,
        "occurrence_summary": group_summary(all_rows),
        "unique_image_summary": group_summary(unique_rows),
        "runtime_seconds": time.perf_counter() - started,
        "gt_usage": (
            "GT is used only after causal LOF scoring for offline debug metrics."
        ),
    }
    (output_dir / "expert_patch_lof_diagnostic.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
