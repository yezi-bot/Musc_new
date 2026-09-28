"""Validate stage 1 image-level Channel support for dynamic EX-MSM."""

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from validate_density_bottle import ChannelMemory, extract_features, load_samples, write_csv


def compute_online_dino_msm_image_score(
    current, reference_features, device, reference_chunk_size=8
):
    """Score one image against strictly earlier images in the current stream."""
    if reference_chunk_size < 1:
        raise ValueError("reference_chunk_size must be at least 1")
    if not reference_features:
        return None

    current = current.float().to(device)
    patch_count = current.shape[0]
    patch_to_image = []
    for start in range(0, len(reference_features), reference_chunk_size):
        chunk = torch.stack(
            reference_features[start : start + reference_chunk_size]
        ).float().to(device)
        distances = torch.cdist(current, chunk.reshape(-1, chunk.shape[-1]))
        distances = distances.reshape(patch_count, chunk.shape[0], patch_count).amin(dim=2)
        patch_to_image.append(distances)
    patch_to_image = torch.cat(patch_to_image, dim=1)
    k_max = max(1, int(patch_to_image.shape[1] * 0.3))
    patch_scores = torch.topk(
        patch_to_image, k=k_max, dim=1, largest=False, sorted=False
    ).values.mean(dim=1)
    return float(patch_scores.max().cpu())


def compute_patch_channel_distances(
    current, mature_channels, grid_size, device, position_radius=None
):
    """Return each patch's nearest mature seed distance under an optional position window."""
    seeds = torch.stack([channel.seed for channel in mature_channels]).float().to(device)
    distances = torch.cdist(current.float().to(device), seeds)
    if position_radius is not None:
        patch_ids = torch.arange(current.shape[0], device=device)
        channel_ids = torch.tensor(
            [channel.patch_ids[-1] for channel in mature_channels], device=device
        )
        patch_rows = patch_ids[:, None] // grid_size
        patch_cols = patch_ids[:, None] % grid_size
        channel_rows = channel_ids[None, :] // grid_size
        channel_cols = channel_ids[None, :] % grid_size
        valid = (patch_rows - channel_rows).abs() <= position_radius
        valid &= (patch_cols - channel_cols).abs() <= position_radius
        distances = distances.masked_fill(~valid, float("inf"))
    return distances.amin(dim=1).cpu()


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def run_stage1_image_support(
    features,
    samples,
    device,
    output_dir,
    ttl=5,
    density_k=5,
    alpha=0.5,
    threshold_quantile=0.7,
    stream_seeds=(0, 1, 2),
    msm_chunk_size=8,
):
    """Measure online Channel support before every Channel memory update."""
    grid_size = int(math.sqrt(features[0].shape[0]))
    records = []
    for stream_seed in stream_seeds:
        order = list(range(len(samples)))
        random.Random(stream_seed).shuffle(order)
        memories = {
            "global": ChannelMemory(
                max_ttl=ttl, density_k=density_k, device=device, position_radius=None
            ),
            "local": ChannelMemory(
                max_ttl=ttl, density_k=density_k, device=device, position_radius=1
            ),
        }
        distance_history = {"global": [], "local": []}
        reference_features = []
        for stream_step, image_id in enumerate(order):
            current = features[image_id]
            ms_score = compute_online_dino_msm_image_score(
                current,
                reference_features,
                device=device,
                reference_chunk_size=msm_chunk_size,
            )
            for method, memory in memories.items():
                mature = memory.weighted_mature_channels()
                threshold = None
                support = None
                current_distances = None
                if mature:
                    current_distances = compute_patch_channel_distances(
                        current,
                        mature,
                        grid_size,
                        device,
                        position_radius=memory.position_radius,
                    )
                    if distance_history[method]:
                        threshold = float(
                            np.quantile(distance_history[method], threshold_quantile)
                        )
                        support = float((current_distances <= threshold).float().mean())
                records.append(
                    {
                        "stream_seed": stream_seed,
                        "stream_step": stream_step,
                        "image_id": image_id,
                        "image_type": samples[image_id]["kind"],
                        "method": method,
                        "reliable_mature_channels": len(mature),
                        "online_threshold": threshold,
                        "channel_support": support,
                        "ms_score": ms_score,
                        "ms_score_source": "single-layer online DINO MSM-style score, image max",
                    }
                )
                if current_distances is not None:
                    finite_distances = current_distances[torch.isfinite(current_distances)]
                    distance_history[method].extend(finite_distances.tolist())
                _, patch_weights = memory.patch_reliability(
                    current, grid_size, reliability_mode="rank"
                )
                soft_weights = alpha + (1.0 - alpha) * patch_weights
                memory.update(
                    current,
                    image_id,
                    patch_densities=None,
                    patch_reliabilities=soft_weights,
                )
            reference_features.append(current)

    summary = {
        "stage": 1,
        "stream_seeds": list(stream_seeds),
        "threshold": {
            "quantile": threshold_quantile,
            "source": "strictly prior finite NN distances within each stream and method",
        },
        "ms_score": {
            "available": True,
            "source": "single-layer DINO MSM-style score against strictly prior stream images",
            "image_reduction": "maximum patch score",
            "unavailable_warmup_images": len(stream_seeds),
        },
        "methods": {},
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    bins = np.linspace(0.0, 1.0, 21)
    for axis, method in zip(axes, ("global", "local")):
        available = [
            row for row in records if row["method"] == method and row["channel_support"] is not None
        ]
        normal = [row["channel_support"] for row in available if row["image_type"] == "good"]
        anomaly = [row["channel_support"] for row in available if row["image_type"] != "good"]
        pairwise = [
            float(normal_value > anomaly_value) + 0.5 * float(normal_value == anomaly_value)
            for normal_value in normal
            for anomaly_value in anomaly
        ]
        summary["methods"][method] = {
            "available_records": len(available),
            "unavailable_warmup_records": len(stream_seeds) * len(samples) - len(available),
            "normal": summarize(normal),
            "anomaly": summarize(anomaly),
            "normal_minus_anomaly_mean": (
                float(np.mean(normal) - np.mean(anomaly)) if normal and anomaly else None
            ),
            "support_separation_auc": float(np.mean(pairwise)) if pairwise else None,
        }
        if normal:
            axis.hist(normal, bins=bins, alpha=0.65, density=True, label="normal")
        if anomaly:
            axis.hist(anomaly, bins=bins, alpha=0.65, density=True, label="anomaly")
        axis.set_title(f"{method.title()} online support")
        axis.set_xlabel("image-level Channel support")
        if normal or anomaly:
            axis.legend()
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_online_support_distributions.png", dpi=200)
    plt.close(fig)
    write_csv(output_dir / "stage1_online_image_support.csv", records)
    (output_dir / "stage1_online_image_support_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dinov2-repo", required=True)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--layer-index", type=int, default=23)
    parser.add_argument("--ttl", type=int, default=5)
    parser.add_argument("--density-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--threshold-quantile", type=float, default=0.7)
    parser.add_argument("--features-cache", type=str, default=None)
    parser.add_argument("--stream-seeds", type=str, default="0,1,2")
    parser.add_argument("--msm-chunk-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.data_root)
    if not samples:
        raise RuntimeError("No bottle test images found")
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_cache = (
        Path(args.features_cache)
        if args.features_cache
        else output_dir / "bottle_dino_features.pt"
    )
    if features_cache.exists():
        features = torch.load(features_cache, map_location="cpu", weights_only=False)
        print(f"loaded feature cache: {features_cache}", flush=True)
    else:
        model = torch.hub.load(args.dinov2_repo, "dinov2_vitl14", source="local")
        model.eval().to(device)
        preprocess = transforms.Compose(
            [
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        features = extract_features(samples, model, preprocess, device, args.layer_index)
        torch.save(features, features_cache)

    features = features[: len(samples)]
    if len(features) != len(samples):
        raise RuntimeError("Feature cache contains fewer entries than the selected samples")
    stream_seeds = tuple(int(value.strip()) for value in args.stream_seeds.split(","))
    summary = run_stage1_image_support(
        features,
        samples,
        device,
        output_dir,
        ttl=args.ttl,
        density_k=args.density_k,
        alpha=args.alpha,
        threshold_quantile=args.threshold_quantile,
        stream_seeds=stream_seeds,
        msm_chunk_size=args.msm_chunk_size,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
