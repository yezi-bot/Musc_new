"""Validate span-only versus density-filtered channel contamination on MVTec bottle."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


ANOMALY_TYPES = {"broken_large", "broken_small", "contamination"}


class Channel:
    def __init__(self, feature, image_id, patch_id, ttl):
        self.seed = feature.detach().clone()
        self.features = [feature.detach().clone()]
        self.image_ids = [image_id]
        self.patch_ids = [patch_id]
        self.densities = []
        self.reliabilities = []
        self.span = 1
        self.effective_span = 1.0
        self.ttl = ttl

    @property
    def median_density(self):
        if not self.densities:
            return None
        return float(np.median(self.densities))


class ChannelMemory:
    def __init__(self, max_ttl=5, mature_span=3, density_k=5, device="cpu", position_radius=None):
        self.channels = []
        self.max_ttl = max_ttl
        self.mature_span = mature_span
        self.density_k = density_k
        self.device = torch.device(device)
        self.position_radius = position_radius
        self.deleted_last_step = 0
        self.matched_last_step = 0
        self.created_last_step = 0

    def mature_channels(self):
        return [c for c in self.channels if c.span >= self.mature_span]

    def weighted_mature_channels(self):
        return [c for c in self.channels if c.effective_span >= self.mature_span]

    def patch_density(self, features):
        """Compute density against historical mature seeds before updating memory."""
        mature = self.mature_channels()
        if not mature:
            return None
        seeds = torch.stack([c.seed for c in mature], dim=0).to(self.device)
        distances = torch.cdist(features.float().to(self.device), seeds.float())
        k = min(self.density_k, distances.shape[1])
        nearest = torch.topk(distances, k=k, dim=1, largest=False).values
        return nearest.mean(dim=1).cpu()

    def patch_reliability(self, features, grid_size, reliability_mode="rank"):
        """Position-aware kNN outlier score and soft reliability for each patch."""
        current = features.float().to(self.device)
        if self.channels:
            candidates = torch.stack([c.features[-1] for c in self.channels]).to(self.device)
            distances = torch.cdist(current, candidates)
            if self.position_radius is not None:
                patch_ids = torch.arange(current.shape[0], device=self.device)
                channel_ids = torch.tensor(
                    [c.patch_ids[-1] for c in self.channels], device=self.device
                )
                rows = patch_ids[:, None] // grid_size
                cols = patch_ids[:, None] % grid_size
                channel_rows = channel_ids[None, :] // grid_size
                channel_cols = channel_ids[None, :] % grid_size
                valid = (rows - channel_rows).abs() <= self.position_radius
                valid &= (cols - channel_cols).abs() <= self.position_radius
                distances = distances.masked_fill(~valid, float("inf"))
            scores = torch.empty(current.shape[0], device=self.device)
            for patch_id in range(current.shape[0]):
                row_distances = distances[patch_id]
                finite = row_distances[torch.isfinite(row_distances)]
                if finite.numel() == 0:
                    finite = torch.cdist(current[patch_id : patch_id + 1], current)[0]
                    finite[patch_id] = float("inf")
                    finite = finite[torch.isfinite(finite)]
                k = min(self.density_k, finite.numel())
                scores[patch_id] = torch.topk(finite, k=k, largest=False).values.mean()
        else:
            distances = torch.cdist(current, current)
            patch_ids = torch.arange(current.shape[0], device=self.device)
            rows = patch_ids[:, None] // grid_size
            cols = patch_ids[:, None] % grid_size
            valid = (rows - rows.T).abs() <= 1
            valid &= (cols - cols.T).abs() <= 1
            valid.fill_diagonal_(False)
            distances = distances.masked_fill(~valid, float("inf"))
            scores = torch.empty(current.shape[0], device=self.device)
            for patch_id in range(current.shape[0]):
                finite = distances[patch_id][torch.isfinite(distances[patch_id])]
                k = min(self.density_k, finite.numel())
                scores[patch_id] = torch.topk(finite, k=k, largest=False).values.mean()
        if reliability_mode == "rank":
            order = torch.argsort(scores)
            ranks = torch.empty_like(scores)
            ranks[order] = torch.arange(scores.numel(), device=self.device, dtype=scores.dtype)
            reliability = 1.0 - ranks / max(scores.numel() - 1, 1)
        elif reliability_mode == "distance":
            scale = torch.median(scores).clamp_min(1e-6)
            reliability = scale / (scale + scores)
        else:
            raise ValueError(f"Unknown reliability mode: {reliability_mode}")
        reliability = reliability.clamp_min(0.1).clamp_max(1.0)
        return scores.cpu(), reliability.cpu()

    def update(self, features, image_id, patch_densities, patch_reliabilities=None):
        for channel in self.channels:
            channel.ttl -= 1
        before_delete = len(self.channels)
        self.channels = [c for c in self.channels if c.ttl > 0]
        self.deleted_last_step = before_delete - len(self.channels)
        self.matched_last_step = 0
        self.created_last_step = 0

        if not self.channels:
            for patch_id, feature in enumerate(features):
                channel = Channel(feature, image_id, patch_id, self.max_ttl)
                if patch_densities is not None:
                    channel.densities.append(float(patch_densities[patch_id]))
                if patch_reliabilities is not None:
                    weight = float(patch_reliabilities[patch_id])
                    channel.reliabilities.append(weight)
                    channel.effective_span = weight
                self.channels.append(channel)
            self.created_last_step = features.shape[0]
            return

        seeds = torch.stack([c.seed for c in self.channels], dim=0).to(self.device)
        distances = torch.cdist(features.float().to(self.device), seeds.float())
        if self.position_radius is not None:
            grid_size = int(math.sqrt(features.shape[0]))
            patch_ids = torch.arange(features.shape[0], device=self.device)
            channel_ids = torch.tensor(
                [c.patch_ids[-1] for c in self.channels], device=self.device
            )
            patch_rows = (patch_ids[:, None] // grid_size)
            patch_cols = (patch_ids[:, None] % grid_size)
            channel_rows = channel_ids[None, :] // grid_size
            channel_cols = channel_ids[None, :] % grid_size
            valid = (patch_rows - channel_rows).abs() <= self.position_radius
            valid &= (patch_cols - channel_cols).abs() <= self.position_radius
            distances = distances.masked_fill(~valid, float("inf"))
        patch_to_channel = distances.argmin(dim=1)
        channel_to_patch = distances.argmin(dim=0)
        matched = set()

        for patch_id in range(features.shape[0]):
            channel_id = int(patch_to_channel[patch_id])
            if not torch.isfinite(distances[patch_id, channel_id]):
                continue
            if int(channel_to_patch[channel_id]) != patch_id:
                continue
            channel = self.channels[channel_id]
            channel.features.append(features[patch_id].detach().clone())
            channel.image_ids.append(image_id)
            channel.patch_ids.append(patch_id)
            if patch_densities is not None:
                channel.densities.append(float(patch_densities[patch_id]))
            if patch_reliabilities is not None:
                weight = float(patch_reliabilities[patch_id])
                channel.reliabilities.append(weight)
                channel.effective_span += weight
            channel.span += 1
            channel.ttl = self.max_ttl
            matched.add(patch_id)
            self.matched_last_step += 1

        for patch_id, feature in enumerate(features):
            if patch_id in matched:
                continue
            channel = Channel(feature, image_id, patch_id, self.max_ttl)
            if patch_densities is not None:
                channel.densities.append(float(patch_densities[patch_id]))
            if patch_reliabilities is not None:
                weight = float(patch_reliabilities[patch_id])
                channel.reliabilities.append(weight)
                channel.effective_span = weight
            self.channels.append(channel)
            self.created_last_step += 1


def load_samples(data_root):
    root = Path(data_root) / "bottle"
    samples = []
    for category in sorted((root / "test").iterdir()):
        if not category.is_dir():
            continue
        for image_path in sorted(category.glob("*.png")):
            if category.name == "good":
                mask = np.zeros((1024, 1024), dtype=np.uint8)
            else:
                mask_path = root / "ground_truth" / category.name / f"{image_path.stem}_mask.png"
                mask = np.asarray(Image.open(mask_path).convert("L"))
            samples.append(
                {
                    "path": image_path,
                    "kind": category.name,
                    "mask": mask,
                    "is_anomaly": category.name in ANOMALY_TYPES,
                }
            )
    return samples


def patch_is_anomalous(samples, image_id, patch_id, grid_size):
    mask = torch.from_numpy(samples[image_id]["mask"].astype(np.float32))[None, None]
    mask = torch.nn.functional.interpolate(mask, size=(grid_size, grid_size), mode="nearest")[0, 0]
    row, col = divmod(patch_id, grid_size)
    return bool(mask[row, col] > 0)


def seed_is_anomalous(channel, samples, grid_size):
    return patch_is_anomalous(samples, channel.image_ids[0], channel.patch_ids[0], grid_size)


def channel_stats(channels, samples, grid_size, threshold=None, weighted=False):
    span_channels = [
        c for c in channels if (c.effective_span if weighted else c.span) >= 3
    ]
    if threshold is None:
        selected = span_channels
    else:
        selected = [
            c for c in span_channels
            if c.median_density is None or c.median_density <= threshold
        ]
    anomaly_count = sum(seed_is_anomalous(c, samples, grid_size) for c in selected)
    densities = [c.median_density for c in span_channels if c.median_density is not None]
    return {
        "span_mature": len(span_channels),
        "selected": len(selected),
        "removed": len(span_channels) - len(selected),
        "anomaly_seed_count": anomaly_count,
        "contamination": anomaly_count / len(selected) if selected else 0.0,
        "densities": densities,
    }


def run_clean_baseline(features, samples, device, ttl=5):
    memory = ChannelMemory(max_ttl=ttl, device=device, position_radius=None)
    rows = []
    grid_size = int(math.sqrt(features[0].shape[0]))
    for image_id, current in enumerate(features):
        memory.update(current, image_id, patch_densities=None)
        mature = memory.mature_channels()
        normal_mature = sum(not seed_is_anomalous(c, samples, grid_size) for c in mature)
        anomaly_mature = len(mature) - normal_mature
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(samples[image_id]["path"]),
                "image_type": samples[image_id]["kind"],
                "matching": "global",
                "maturity": "span>=3",
                "total_channels": len(memory.channels),
                "mature_channels": len(mature),
                "normal_mature_channels": normal_mature,
                "anomaly_mature_channels": anomaly_mature,
                "contamination": anomaly_mature / len(mature) if mature else 0.0,
                "deleted_channels": memory.deleted_last_step,
            }
        )
    return rows


def run_position_baseline(features, samples, device, ttl=5):
    memory = ChannelMemory(max_ttl=ttl, device=device, position_radius=1)
    rows = []
    grid_size = int(math.sqrt(features[0].shape[0]))
    for image_id, current in enumerate(features):
        memory.update(current, image_id, patch_densities=None)
        mature = memory.mature_channels()
        normal_mature = sum(not seed_is_anomalous(c, samples, grid_size) for c in mature)
        anomaly_mature = len(mature) - normal_mature
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(samples[image_id]["path"]),
                "image_type": samples[image_id]["kind"],
                "matching": "local_3x3",
                "maturity": "span>=3",
                "total_channels": len(memory.channels),
                "mature_channels": len(mature),
                "normal_mature_channels": normal_mature,
                "anomaly_mature_channels": anomaly_mature,
                "contamination": anomaly_mature / len(mature) if mature else 0.0,
                "deleted_channels": memory.deleted_last_step,
            }
        )
    return rows


def run_stage3_reliability(features, samples, device, ttl=5, density_k=5):
    memory = ChannelMemory(
        max_ttl=ttl,
        density_k=density_k,
        device=device,
        position_radius=1,
    )
    rows = []
    patch_records = []
    grid_size = int(math.sqrt(features[0].shape[0]))
    for image_id, current in enumerate(features):
        patch_scores, patch_weights = memory.patch_reliability(current, grid_size)
        normal_weights = []
        anomaly_weights = []
        for patch_id, weight in enumerate(patch_weights.tolist()):
            anomaly = patch_is_anomalous(samples, image_id, patch_id, grid_size)
            (anomaly_weights if anomaly else normal_weights).append(weight)
            patch_records.append(
                {
                    "image_id": image_id,
                    "patch_id": patch_id,
                    "row": patch_id // grid_size,
                    "col": patch_id % grid_size,
                    "is_anomaly": int(anomaly),
                    "outlier_distance": float(patch_scores[patch_id]),
                    "reliability": weight,
                }
            )
        memory.update(
            current,
            image_id,
            patch_densities=None,
            patch_reliabilities=patch_weights,
        )
        mature = memory.mature_channels()
        normal_mature = sum(not seed_is_anomalous(c, samples, grid_size) for c in mature)
        anomaly_mature = len(mature) - normal_mature
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(samples[image_id]["path"]),
                "image_type": samples[image_id]["kind"],
                "matching": "local_3x3",
                "maturity": "span>=3",
                "reliability_used_for_maturity": False,
                "total_channels": len(memory.channels),
                "mature_channels": len(mature),
                "normal_mature_channels": normal_mature,
                "anomaly_mature_channels": anomaly_mature,
                "contamination": anomaly_mature / len(mature) if mature else 0.0,
                "normal_patch_mean_reliability": float(np.mean(normal_weights)) if normal_weights else None,
                "anomaly_patch_mean_reliability": float(np.mean(anomaly_weights)) if anomaly_weights else None,
                "deleted_channels": memory.deleted_last_step,
            }
        )
    return rows, patch_records


def run_stage4_update_audit(features, samples, device, ttl=5, density_k=5):
    memory = ChannelMemory(
        max_ttl=ttl,
        density_k=density_k,
        device=device,
        position_radius=1,
    )
    rows = []
    grid_size = int(math.sqrt(features[0].shape[0]))
    for image_id, current in enumerate(features):
        _, patch_weights = memory.patch_reliability(current, grid_size)
        low_reliability = int((patch_weights < 0.2).sum())
        memory.update(
            current,
            image_id,
            patch_densities=None,
            patch_reliabilities=patch_weights,
        )
        mature = memory.mature_channels()
        normal_mature = sum(not seed_is_anomalous(c, samples, grid_size) for c in mature)
        anomaly_mature = len(mature) - normal_mature
        forwarded = memory.matched_last_step + memory.created_last_step
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(samples[image_id]["path"]),
                "image_type": samples[image_id]["kind"],
                "matching": "local_3x3",
                "maturity": "span>=3",
                "reliability_used_for_filtering": False,
                "patch_count": int(current.shape[0]),
                "low_reliability_patches": low_reliability,
                "matched_updates": memory.matched_last_step,
                "created_channels": memory.created_last_step,
                "forwarded_patches": forwarded,
                "all_patches_forwarded": forwarded == current.shape[0],
                "mature_channels": len(mature),
                "normal_mature_channels": normal_mature,
                "anomaly_mature_channels": anomaly_mature,
                "contamination": anomaly_mature / len(mature) if mature else 0.0,
                "deleted_channels": memory.deleted_last_step,
            }
        )
    return rows


def run_stage5_soft_span(
    features,
    samples,
    device,
    ttl=5,
    density_k=5,
    alpha=0.5,
    reliability_mode="rank",
    position_radius=1,
):
    memory = ChannelMemory(
        max_ttl=ttl,
        density_k=density_k,
        device=device,
        position_radius=position_radius,
    )
    rows = []
    channel_records = []
    patch_records = []
    grid_size = int(math.sqrt(features[0].shape[0]))
    for image_id, current in enumerate(features):
        patch_scores, patch_weights = memory.patch_reliability(
            current, grid_size, reliability_mode=reliability_mode
        )
        soft_weights = alpha + (1.0 - alpha) * patch_weights
        for patch_id, weight in enumerate(patch_weights.tolist()):
            patch_records.append(
                {
                    "image_id": image_id,
                    "patch_id": patch_id,
                    "is_anomaly": int(patch_is_anomalous(samples, image_id, patch_id, grid_size)),
                    "reliability": weight,
                    "soft_weight": float(soft_weights[patch_id]),
                }
            )
        memory.update(
            current,
            image_id,
            patch_densities=None,
            patch_reliabilities=soft_weights,
        )
        mature = memory.weighted_mature_channels()
        normal_mature = sum(not seed_is_anomalous(c, samples, grid_size) for c in mature)
        anomaly_mature = len(mature) - normal_mature
        for channel_id, channel in enumerate(memory.channels):
            channel_records.append(
                {
                    "image_id": image_id,
                    "channel_id": channel_id,
                    "seed_image_id": channel.image_ids[0],
                    "seed_patch_id": channel.patch_ids[0],
                    "seed_anomaly": int(seed_is_anomalous(channel, samples, grid_size)),
                    "span": channel.span,
                    "effective_span": channel.effective_span,
                    "retained": int(channel.effective_span >= 3),
                }
            )
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(samples[image_id]["path"]),
                "image_type": samples[image_id]["kind"],
                "matching": "local_3x3",
                "reliability_mode": reliability_mode,
                "maturity": "effective_span>=3",
                "alpha": alpha,
                "total_channels": len(memory.channels),
                "mature_channels": len(mature),
                "normal_mature_channels": normal_mature,
                "anomaly_mature_channels": anomaly_mature,
                "contamination": anomaly_mature / len(mature) if mature else 0.0,
                "mean_reliability": float(patch_weights.mean()),
                "mean_soft_weight": float(soft_weights.mean()),
                "deleted_channels": memory.deleted_last_step,
            }
        )
    return rows, channel_records, patch_records


def run_stage7_semantic_audit(
    features, samples, device, ttl=5, density_k=5, alpha=0.5
):
    """Audit all, span-mature, and reliability-mature Channel semantics."""
    memory = ChannelMemory(
        max_ttl=ttl,
        density_k=density_k,
        device=device,
        position_radius=1,
    )
    rows = []
    channel_records = []
    grid_size = int(math.sqrt(features[0].shape[0]))
    for image_id, current in enumerate(features):
        _, patch_weights = memory.patch_reliability(current, grid_size, reliability_mode="rank")
        soft_weights = alpha + (1.0 - alpha) * patch_weights
        memory.update(
            current,
            image_id,
            patch_densities=None,
            patch_reliabilities=soft_weights,
        )
        all_channels = list(memory.channels)
        span_mature = [c for c in all_channels if c.span >= 3]
        reliable_mature = [c for c in all_channels if c.effective_span >= 3]

        def stats(channels):
            anomaly_count = sum(seed_is_anomalous(c, samples, grid_size) for c in channels)
            return {
                "count": len(channels),
                "normal_count": len(channels) - anomaly_count,
                "anomaly_count": anomaly_count,
                "contamination": anomaly_count / len(channels) if channels else 0.0,
            }

        all_stats = stats(all_channels)
        mature_stats = stats(span_mature)
        reliable_stats = stats(reliable_mature)
        for channel_id, channel in enumerate(all_channels):
            channel_records.append(
                {
                    "image_id": image_id,
                    "channel_id": channel_id,
                    "seed_image_id": channel.image_ids[0],
                    "seed_patch_id": channel.patch_ids[0],
                    "seed_anomaly": int(seed_is_anomalous(channel, samples, grid_size)),
                    "span": channel.span,
                    "effective_span": channel.effective_span,
                    "is_all_channel": 1,
                    "is_mature_channel": int(channel.span >= 3),
                    "is_reliable_mature_channel": int(channel.effective_span >= 3),
                }
            )
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(samples[image_id]["path"]),
                "image_type": samples[image_id]["kind"],
                "matching": "local_3x3",
                "reliability_mode": "rank",
                "alpha": alpha,
                "all_channels": all_stats["count"],
                "normal_all_channels": all_stats["normal_count"],
                "anomaly_all_channels": all_stats["anomaly_count"],
                "all_channel_contamination": all_stats["contamination"],
                "mature_channels": mature_stats["count"],
                "normal_mature_channels": mature_stats["normal_count"],
                "anomaly_mature_channels": mature_stats["anomaly_count"],
                "mature_contamination": mature_stats["contamination"],
                "reliable_mature_channels": reliable_stats["count"],
                "normal_reliable_mature_channels": reliable_stats["normal_count"],
                "anomaly_reliable_mature_channels": reliable_stats["anomaly_count"],
                "reliable_mature_contamination": reliable_stats["contamination"],
                "removed_by_effective_span": mature_stats["count"] - reliable_stats["count"],
                "deleted_channels": memory.deleted_last_step,
            }
        )
    return rows, channel_records


def run_stage9_nn_distance_audit(
    features, samples, device, ttl=5, density_k=5, alpha=0.5
):
    """Compare global/local reliable mature Channel coverage by NN distance."""
    memories = {
        "global": ChannelMemory(
            max_ttl=ttl, density_k=density_k, device=device, position_radius=None
        ),
        "local": ChannelMemory(
            max_ttl=ttl, density_k=density_k, device=device, position_radius=1
        ),
    }
    image_rows = []
    patch_records = []
    grid_size = int(math.sqrt(features[0].shape[0]))

    def nearest_distances(memory, current):
        mature = memory.weighted_mature_channels()
        if not mature:
            return torch.full((current.shape[0],), float("nan"))
        seeds = torch.stack([channel.seed for channel in mature]).to(device)
        return torch.cdist(current.float().to(device), seeds.float()).min(dim=1).values.cpu()

    def finite_mean(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else None

    def finite_median(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else None

    for image_id, current in enumerate(features):
        distances = {
            name: nearest_distances(memory, current) for name, memory in memories.items()
        }
        normal_global = []
        normal_local = []
        anomaly_global = []
        anomaly_local = []
        for patch_id in range(current.shape[0]):
            anomaly = patch_is_anomalous(samples, image_id, patch_id, grid_size)
            global_distance = float(distances["global"][patch_id])
            local_distance = float(distances["local"][patch_id])
            if anomaly:
                anomaly_global.append(global_distance)
                anomaly_local.append(local_distance)
            else:
                normal_global.append(global_distance)
                normal_local.append(local_distance)
            patch_records.append(
                {
                    "image_id": image_id,
                    "patch_id": patch_id,
                    "row": patch_id // grid_size,
                    "col": patch_id % grid_size,
                    "is_anomaly": int(anomaly),
                    "nearest_global_distance": global_distance,
                    "nearest_local_distance": local_distance,
                }
            )

        for name, memory in memories.items():
            _, patch_weights = memory.patch_reliability(current, grid_size, reliability_mode="rank")
            soft_weights = alpha + (1.0 - alpha) * patch_weights
            memory.update(
                current,
                image_id,
                patch_densities=None,
                patch_reliabilities=soft_weights,
            )

        global_mature = memories["global"].weighted_mature_channels()
        local_mature = memories["local"].weighted_mature_channels()
        image_rows.append(
            {
                "image_id": image_id,
                "image_path": str(samples[image_id]["path"]),
                "image_type": samples[image_id]["kind"],
                "global_reliable_mature_channels": len(global_mature),
                "local_reliable_mature_channels": len(local_mature),
                "global_mean_nn_distance": finite_mean(distances["global"]),
                "global_median_nn_distance": finite_median(distances["global"]),
                "local_mean_nn_distance": finite_mean(distances["local"]),
                "local_median_nn_distance": finite_median(distances["local"]),
                "normal_global_mean": finite_mean(normal_global),
                "normal_local_mean": finite_mean(normal_local),
                "normal_global_median": finite_median(normal_global),
                "normal_local_median": finite_median(normal_local),
                "anomaly_global_mean": finite_mean(anomaly_global),
                "anomaly_local_mean": finite_mean(anomaly_local),
                "anomaly_global_median": finite_median(anomaly_global),
                "anomaly_local_median": finite_median(anomaly_local),
            }
        )
    return image_rows, patch_records


def run_weighted_span(features, samples, device, ttl=5, density_k=5):
    """Scheme B: local position matching plus soft patch reliability."""
    grid_size = int(math.sqrt(features[0].shape[0]))
    memory = ChannelMemory(
        max_ttl=ttl,
        density_k=density_k,
        device=device,
        position_radius=1,
    )
    rows = []
    scores = []
    channel_records = []
    for image_id, current in enumerate(features):
        patch_scores, patch_weights = memory.patch_reliability(current, grid_size)
        scores.extend(patch_scores.tolist())
        memory.update(
            current,
            image_id,
            patch_scores,
            patch_reliabilities=patch_weights,
        )
        stats = channel_stats(memory.channels, samples, grid_size, weighted=True)
        for channel_id, channel in enumerate(memory.channels):
            channel_records.append(
                {
                    "image_id": image_id,
                    "channel_id": channel_id,
                    "seed_image_id": channel.image_ids[0],
                    "seed_patch_id": channel.patch_ids[0],
                    "seed_anomaly": int(seed_is_anomalous(channel, samples, grid_size)),
                    "span": channel.span,
                    "effective_span": channel.effective_span,
                    "retained": int(channel.effective_span >= 3),
                    "ttl": channel.ttl,
                }
            )
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(samples[image_id]["path"]),
                "image_type": samples[image_id]["kind"],
                "matching": "local_3x3",
                "maturity": "effective_span>=3",
                "mean_patch_reliability": float(patch_weights.mean()),
                "min_patch_reliability": float(patch_weights.min()),
                **{k: v for k, v in stats.items() if k != "densities"},
            }
        )
    return rows, scores, channel_records


def extract_features(samples, model, preprocess, device, layer_index):
    features = []
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            image = preprocess(Image.open(sample["path"]).convert("RGB")).unsqueeze(0).to(device)
            tokens = model.get_intermediate_layers(
                image, n=[layer_index], return_class_token=False
            )[0]
            tokens = tokens.squeeze(0).float()
            tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            features.append(tokens.cpu())
            print(f"feature {index + 1}/{len(samples)}", flush=True)
    return features


def run_once(features, samples, device, threshold=None, ttl=5):
    memory = ChannelMemory(max_ttl=ttl, device=device)
    rows = []
    all_densities = []
    grid_size = int(math.sqrt(features[0].shape[0]))
    for image_id, current in enumerate(features):
        patch_densities = memory.patch_density(current)
        if patch_densities is not None:
            all_densities.extend(patch_densities.tolist())
        memory.update(current, image_id, patch_densities)
        stats = channel_stats(memory.channels, samples, grid_size, threshold)
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(samples[image_id]["path"]),
                "image_type": samples[image_id]["kind"],
                "threshold": threshold,
                **{k: v for k, v in stats.items() if k != "densities"},
            }
        )
    return rows, all_densities


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path, baseline, density_runs, weighted_rows=None):
    plt.figure(figsize=(11, 5))
    x = [row["image_id"] for row in baseline]
    plt.plot(x, [row["contamination"] for row in baseline], label="span-only", linewidth=2)
    for label, rows in density_runs.items():
        plt.plot(x, [row["contamination"] for row in rows], label=f"density q={label}")
    if weighted_rows is not None:
        plt.plot(
            x,
            [row["contamination"] for row in weighted_rows],
            label="scheme B: effective span",
            linewidth=2,
        )
    plt.xlabel("Image index")
    plt.ylabel("Anomaly seed ratio")
    plt.title("MVTec AD bottle: Channel contamination")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def collect_density_records(features, samples, device, ttl=5):
    memory = ChannelMemory(max_ttl=ttl, device=device)
    records = []
    grid_size = int(math.sqrt(features[0].shape[0]))
    for image_id, current in enumerate(features):
        patch_densities = memory.patch_density(current)
        memory.update(current, image_id, patch_densities)
        for channel in memory.mature_channels():
            density = channel.median_density
            if density is None:
                continue
            records.append(
                {
                    "image_id": image_id,
                    "seed_image_id": channel.image_ids[0],
                    "seed_patch_id": channel.patch_ids[0],
                    "seed_anomaly": int(seed_is_anomalous(channel, samples, grid_size)),
                    "span": channel.span,
                    "median_density": density,
                }
            )
    return records


def plot_density_distributions(path, records):
    normal = [r["median_density"] for r in records if not r["seed_anomaly"]]
    anomaly = [r["median_density"] for r in records if r["seed_anomaly"]]
    plt.figure(figsize=(9, 5))
    if normal:
        plt.hist(normal, bins=40, alpha=0.65, label=f"normal seed (n={len(normal)})")
    if anomaly:
        plt.hist(anomaly, bins=40, alpha=0.65, label=f"anomaly seed (n={len(anomaly)})")
    plt.xlabel("Channel median density")
    plt.ylabel("Channel count")
    plt.title("MVTec AD bottle: Channel density distributions")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dinov2-repo", required=True)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--layer-index", type=int, default=23)
    parser.add_argument("--ttl", type=int, default=5)
    parser.add_argument("--features-cache", type=str, default=None)
    parser.add_argument("--scheme-b-only", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--stage", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.data_root)
    if not samples:
        raise RuntimeError("No bottle test images found")
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_cache = Path(args.features_cache) if args.features_cache else output_dir / "bottle_dino_features.pt"
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

    if args.max_samples is not None:
        features = features[: args.max_samples]

    if args.stage == 1:
        baseline = run_clean_baseline(features, samples, device, ttl=args.ttl)
        write_csv(output_dir / "stage1_global_span.csv", baseline)
        values = np.asarray([row["contamination"] for row in baseline], dtype=np.float64)
        summary = {
            "stage": 1,
            "sample_count": len(samples),
            "matching": "global",
            "density": False,
            "effective_span": False,
            "maturity": "span>=3",
            "mean_contamination": float(values.mean()),
            "final_contamination": float(values[-1]),
            "mean_total_channels": float(np.mean([r["total_channels"] for r in baseline])),
            "mean_mature_channels": float(np.mean([r["mature_channels"] for r in baseline])),
            "normal_mature_channels": int(sum(r["normal_mature_channels"] for r in baseline)),
            "anomaly_mature_channels": int(sum(r["anomaly_mature_channels"] for r in baseline)),
            "deleted_channels": int(sum(r["deleted_channels"] for r in baseline)),
        }
        (output_dir / "stage1_global_span_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.stage == 2:
        position_rows = run_position_baseline(features, samples, device, ttl=args.ttl)
        write_csv(output_dir / "stage2_position_span.csv", position_rows)
        values = np.asarray([row["contamination"] for row in position_rows], dtype=np.float64)
        summary = {
            "stage": 2,
            "sample_count": len(samples),
            "matching": "local_3x3",
            "density": False,
            "effective_span": False,
            "maturity": "span>=3",
            "mean_contamination": float(values.mean()),
            "final_contamination": float(values[-1]),
            "mean_total_channels": float(np.mean([r["total_channels"] for r in position_rows])),
            "mean_mature_channels": float(np.mean([r["mature_channels"] for r in position_rows])),
            "normal_mature_channels": int(sum(r["normal_mature_channels"] for r in position_rows)),
            "anomaly_mature_channels": int(sum(r["anomaly_mature_channels"] for r in position_rows)),
            "deleted_channels": int(sum(r["deleted_channels"] for r in position_rows)),
        }
        (output_dir / "stage2_position_span_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.stage == 3:
        reliability_rows, patch_records = run_stage3_reliability(
            features, samples, device, ttl=args.ttl
        )
        write_csv(output_dir / "stage3_position_reliability.csv", reliability_rows)
        write_csv(output_dir / "stage3_patch_reliability_records.csv", patch_records)
        values = np.asarray([row["contamination"] for row in reliability_rows], dtype=np.float64)
        normal_weights = [
            r["reliability"] for r in patch_records if not r["is_anomaly"]
        ]
        anomaly_weights = [
            r["reliability"] for r in patch_records if r["is_anomaly"]
        ]
        summary = {
            "stage": 3,
            "sample_count": len(samples),
            "matching": "local_3x3",
            "reliability": "position-aware kNN rank",
            "reliability_used_for_maturity": False,
            "maturity": "span>=3",
            "mean_contamination": float(values.mean()),
            "final_contamination": float(values[-1]),
            "mean_mature_channels": float(np.mean([r["mature_channels"] for r in reliability_rows])),
            "normal_patch_count": len(normal_weights),
            "anomaly_patch_count": len(anomaly_weights),
            "normal_patch_mean_reliability": float(np.mean(normal_weights)),
            "anomaly_patch_mean_reliability": float(np.mean(anomaly_weights)),
            "normal_patch_median_reliability": float(np.median(normal_weights)),
            "anomaly_patch_median_reliability": float(np.median(anomaly_weights)),
        }
        (output_dir / "stage3_position_reliability_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.stage == 4:
        audit_rows = run_stage4_update_audit(features, samples, device, ttl=args.ttl)
        write_csv(output_dir / "stage4_no_filter_update_audit.csv", audit_rows)
        values = np.asarray([row["contamination"] for row in audit_rows], dtype=np.float64)
        low_count = sum(row["low_reliability_patches"] for row in audit_rows)
        summary = {
            "stage": 4,
            "sample_count": len(samples),
            "matching": "local_3x3",
            "reliability_used_for_filtering": False,
            "maturity": "span>=3",
            "mean_contamination": float(values.mean()),
            "final_contamination": float(values[-1]),
            "mean_mature_channels": float(np.mean([r["mature_channels"] for r in audit_rows])),
            "low_reliability_patches": int(low_count),
            "forwarded_patches": int(sum(r["forwarded_patches"] for r in audit_rows)),
            "all_patches_forwarded": bool(all(r["all_patches_forwarded"] for r in audit_rows)),
            "matched_updates": int(sum(r["matched_updates"] for r in audit_rows)),
            "created_channels": int(sum(r["created_channels"] for r in audit_rows)),
            "deleted_channels": int(sum(r["deleted_channels"] for r in audit_rows)),
        }
        (output_dir / "stage4_no_filter_update_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.stage == 5:
        soft_rows, channel_records, patch_records = run_stage5_soft_span(
            features, samples, device, ttl=args.ttl, alpha=0.5, reliability_mode="rank"
        )
        write_csv(output_dir / "stage5_soft_effective_span.csv", soft_rows)
        write_csv(output_dir / "stage5_channel_records.csv", channel_records)
        write_csv(output_dir / "stage5_patch_records.csv", patch_records)
        values = np.asarray([row["contamination"] for row in soft_rows], dtype=np.float64)
        groups = {}
        for record in channel_records:
            key = (record["seed_image_id"], record["seed_patch_id"])
            groups.setdefault(key, []).append(record["retained"])
        labels = {
            (record["seed_image_id"], record["seed_patch_id"]): record["seed_anomaly"]
            for record in channel_records
        }
        retention = {}
        for label, anomaly in (("normal", 0), ("anomaly", 1)):
            selected = [history for key, history in groups.items() if labels[key] == anomaly]
            observation_values = [
                value for key, history in groups.items() if labels[key] == anomaly for value in history
            ]
            retention[label] = {
                "unique_seed_channels": len(selected),
                "ever_retained_rate": float(np.mean([any(v) for v in selected])) if selected else None,
                "observation_retention_rate": float(np.mean(observation_values)) if observation_values else None,
            }
        summary = {
            "stage": 5,
            "sample_count": len(samples),
            "matching": "local_3x3",
            "reliability_mode": "rank",
            "alpha": 0.5,
            "maturity": "effective_span>=3",
            "mean_contamination": float(values.mean()),
            "final_contamination": float(values[-1]),
            "mean_total_channels": float(np.mean([r["total_channels"] for r in soft_rows])),
            "mean_mature_channels": float(np.mean([r["mature_channels"] for r in soft_rows])),
            "normal_patch_mean_reliability": float(np.mean([r["reliability"] for r in patch_records if not r["is_anomaly"]])),
            "anomaly_patch_mean_reliability": float(np.mean([r["reliability"] for r in patch_records if r["is_anomaly"]])),
            "retention": retention,
            "deleted_channels": int(sum(r["deleted_channels"] for r in soft_rows)),
        }
        (output_dir / "stage5_soft_effective_span_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.stage == 6:
        comparison = {}
        for mode in ("rank", "distance"):
            soft_rows, channel_records, patch_records = run_stage5_soft_span(
                features,
                samples,
                device,
                ttl=args.ttl,
                alpha=0.5,
                reliability_mode=mode,
            )
            write_csv(output_dir / f"stage6_{mode}_soft_span.csv", soft_rows)
            write_csv(output_dir / f"stage6_{mode}_patch_records.csv", patch_records)
            values = np.asarray([row["contamination"] for row in soft_rows], dtype=np.float64)
            groups = {}
            labels = {}
            for record in channel_records:
                key = (record["seed_image_id"], record["seed_patch_id"])
                groups.setdefault(key, []).append(record["retained"])
                labels[key] = record["seed_anomaly"]
            retention = {}
            for label, anomaly in (("normal", 0), ("anomaly", 1)):
                selected = [history for key, history in groups.items() if labels[key] == anomaly]
                observation_values = [
                    value for key, history in groups.items() if labels[key] == anomaly for value in history
                ]
                retention[label] = {
                    "unique_seed_channels": len(selected),
                    "ever_retained_rate": float(np.mean([any(v) for v in selected])) if selected else None,
                    "observation_retention_rate": float(np.mean(observation_values)) if observation_values else None,
                }
            normal_patch = [r["reliability"] for r in patch_records if not r["is_anomaly"]]
            anomaly_patch = [r["reliability"] for r in patch_records if r["is_anomaly"]]
            comparison[mode] = {
                "mean_contamination": float(values.mean()),
                "final_contamination": float(values[-1]),
                "mean_mature_channels": float(np.mean([r["mature_channels"] for r in soft_rows])),
                "normal_patch_mean_reliability": float(np.mean(normal_patch)),
                "anomaly_patch_mean_reliability": float(np.mean(anomaly_patch)),
                "normal_patch_median_reliability": float(np.median(normal_patch)),
                "anomaly_patch_median_reliability": float(np.median(anomaly_patch)),
                "retention": retention,
            }
        summary = {
            "stage": 6,
            "sample_count": len(samples),
            "matching": "local_3x3",
            "alpha": 0.5,
            "maturity": "effective_span>=3",
            "comparison": comparison,
        }
        (output_dir / "stage6_reliability_comparison_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.stage == 7:
        semantic_rows, channel_records = run_stage7_semantic_audit(
            features, samples, device, ttl=args.ttl, alpha=0.5
        )
        write_csv(output_dir / "stage7_channel_semantics.csv", semantic_rows)
        write_csv(output_dir / "stage7_channel_records.csv", channel_records)
        summary = {
            "stage": 7,
            "sample_count": len(samples),
            "matching": "local_3x3",
            "reliability_mode": "rank",
            "alpha": 0.5,
            "definitions": {
                "all_channels": "TTL-cleaned channels after the current update",
                "mature_channels": "span>=3",
                "reliable_mature_channels": "effective_span>=3",
            },
            "mean": {
                "all_channels": float(np.mean([r["all_channels"] for r in semantic_rows])),
                "mature_channels": float(np.mean([r["mature_channels"] for r in semantic_rows])),
                "reliable_mature_channels": float(
                    np.mean([r["reliable_mature_channels"] for r in semantic_rows])
                ),
                "all_channel_contamination": float(
                    np.mean([r["all_channel_contamination"] for r in semantic_rows])
                ),
                "mature_contamination": float(
                    np.mean([r["mature_contamination"] for r in semantic_rows])
                ),
                "reliable_mature_contamination": float(
                    np.mean([r["reliable_mature_contamination"] for r in semantic_rows])
                ),
                "removed_by_effective_span": float(
                    np.mean([r["removed_by_effective_span"] for r in semantic_rows])
                ),
            },
            "retention": {
                "mature_over_all": float(
                    np.sum([r["mature_channels"] for r in semantic_rows])
                    / np.sum([r["all_channels"] for r in semantic_rows])
                ),
                "reliable_mature_over_mature": float(
                    np.sum([r["reliable_mature_channels"] for r in semantic_rows])
                    / np.sum([r["mature_channels"] for r in semantic_rows])
                ),
            },
        }
        (output_dir / "stage7_channel_semantics_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.stage == 8:
        ablations = {}
        global_rows = run_clean_baseline(features, samples, device, ttl=args.ttl)
        local_rows = run_position_baseline(features, samples, device, ttl=args.ttl)
        soft_rows, _, _ = run_stage5_soft_span(
            features, samples, device, ttl=args.ttl, alpha=0.5, reliability_mode="rank"
        )
        global_soft_rows, _, _ = run_stage5_soft_span(
            features,
            samples,
            device,
            ttl=args.ttl,
            alpha=0.5,
            reliability_mode="rank",
            position_radius=None,
        )
        runs = {
            "global_span": global_rows,
            "local_span": local_rows,
            "local_soft_effective_span": soft_rows,
            "global_soft_effective_span": global_soft_rows,
        }
        for name, rows in runs.items():
            write_csv(output_dir / f"stage8_{name}.csv", rows)
            values = np.asarray([row["contamination"] for row in rows], dtype=np.float64)
            mature_key = "mature_channels" if name != "local_soft_effective_span" else "mature_channels"
            ablations[name] = {
                "mean_contamination": float(values.mean()),
                "final_contamination": float(values[-1]),
                "mean_total_channels": float(np.mean([r["total_channels"] for r in rows])),
                "mean_mature_channels": float(np.mean([r[mature_key] for r in rows])),
                "deleted_channels": int(sum(r["deleted_channels"] for r in rows)),
            }
        baseline_values = np.asarray([r["contamination"] for r in global_rows], dtype=np.float64)
        for name, rows in runs.items():
            values = np.asarray([r["contamination"] for r in rows], dtype=np.float64)
            ablations[name]["lower_than_global_fraction"] = float(np.mean(values < baseline_values))
        summary = {
            "stage": 8,
            "sample_count": len(samples),
            "ttl": args.ttl,
            "ablation_definitions": {
                "global_span": "global matching, span>=3",
                "local_span": "3x3 position matching, span>=3",
                "local_soft_effective_span": "3x3 position matching, rank reliability, alpha=0.5, effective_span>=3",
                "global_soft_effective_span": "global matching, rank reliability, alpha=0.5, effective_span>=3",
            },
            "ablations": ablations,
            "incremental_effects": {
                "position_vs_global_mean_contamination_delta": ablations["local_span"]["mean_contamination"] - ablations["global_span"]["mean_contamination"],
                "soft_effective_vs_local_mean_contamination_delta": ablations["local_soft_effective_span"]["mean_contamination"] - ablations["local_span"]["mean_contamination"],
                "global_soft_vs_global_mean_contamination_delta": ablations["global_soft_effective_span"]["mean_contamination"] - ablations["global_span"]["mean_contamination"],
                "position_effect_with_soft_effective_span_delta": ablations["local_soft_effective_span"]["mean_contamination"] - ablations["global_soft_effective_span"]["mean_contamination"],
            },
        }
        (output_dir / "stage8_channel_ablations_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.stage == 9:
        image_rows, patch_records = run_stage9_nn_distance_audit(
            features, samples, device, ttl=args.ttl, alpha=0.5
        )
        write_csv(output_dir / "stage9_nn_distance_by_image.csv", image_rows)
        write_csv(output_dir / "stage9_nn_distance_by_patch.csv", patch_records)

        def values(key, rows):
            result = [row[key] for row in rows if row[key] is not None]
            return np.asarray(result, dtype=np.float64)

        summary = {
            "stage": 9,
            "sample_count": len(samples),
            "matching": {
                "global": "global matching",
                "local": "3x3 position matching",
            },
            "reliability": "rank",
            "alpha": 0.5,
            "maturity": "effective_span>=3",
            "coverage_threshold": None,
            "distance_summary": {},
        }
        for name in ("global", "local"):
            mean_values = values(f"{name}_mean_nn_distance", image_rows)
            median_values = values(f"{name}_median_nn_distance", image_rows)
            normal_values = values(f"normal_{name}_mean", image_rows)
            anomaly_values = values(f"anomaly_{name}_mean", image_rows)
            summary["distance_summary"][name] = {
                "mean_image_nn_distance": float(mean_values.mean()) if mean_values.size else None,
                "median_image_nn_distance": float(np.median(median_values)) if median_values.size else None,
                "normal_patch_mean_distance": float(normal_values.mean()) if normal_values.size else None,
                "anomaly_patch_mean_distance": float(anomaly_values.mean()) if anomaly_values.size else None,
                "mean_reliable_mature_channels": float(
                    np.mean([row[f"{name}_reliable_mature_channels"] for row in image_rows])
                ),
            }
        normal_global = values("normal_global_mean", image_rows)
        normal_local = values("normal_local_mean", image_rows)
        anomaly_global = values("anomaly_global_mean", image_rows)
        anomaly_local = values("anomaly_local_mean", image_rows)

        def safe_difference(left, right):
            return float(left.mean() - right.mean()) if left.size and right.size else None

        def safe_lower_fraction(left, right):
            return float(np.mean(left < right)) if left.size and right.size else None

        summary["comparison"] = {
            "normal_local_minus_global_mean_distance": safe_difference(normal_local, normal_global),
            "anomaly_local_minus_global_mean_distance": safe_difference(anomaly_local, anomaly_global),
            "normal_local_lower_fraction": safe_lower_fraction(normal_local, normal_global),
            "anomaly_local_lower_fraction": safe_lower_fraction(anomaly_local, anomaly_global),
        }
        (output_dir / "stage9_nn_distance_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    baseline, density_values = run_once(features, samples, device, threshold=None, ttl=args.ttl)
    weighted_rows, weighted_scores, weighted_channel_records = run_weighted_span(
        features, samples, device, ttl=args.ttl
    )
    density_values = np.asarray(density_values, dtype=np.float64)
    quantiles = [0.50, 0.70, 0.80, 0.90]
    thresholds = {str(q): float(np.quantile(density_values, q)) for q in quantiles if density_values.size}

    density_runs = {}
    if not args.scheme_b_only:
        for label, threshold in thresholds.items():
            rows, _ = run_once(features, samples, device, threshold=threshold, ttl=args.ttl)
            density_runs[label] = rows
            write_csv(output_dir / f"bottle_density_q{int(float(label) * 100)}.csv", rows)
    write_csv(output_dir / "bottle_span_only.csv", baseline)
    write_csv(output_dir / "bottle_scheme_b_effective_span.csv", weighted_rows)
    write_csv(
        output_dir / "bottle_scheme_b_channel_records.csv",
        weighted_channel_records,
    )
    plot_curves(
        output_dir / "bottle_contamination_curves.png",
        baseline,
        density_runs,
        weighted_rows=weighted_rows,
    )
    if args.scheme_b_only:
        density_records = []
    else:
        density_records = collect_density_records(features, samples, device, ttl=args.ttl)
        write_csv(output_dir / "bottle_channel_density.csv", density_records)
        plot_density_distributions(output_dir / "bottle_density_distributions.png", density_records)

    normal_density = [r["median_density"] for r in density_records if not r["seed_anomaly"]]
    anomaly_density = [r["median_density"] for r in density_records if r["seed_anomaly"]]
    summary = {
        "sample_count": len(samples),
        "thresholds": thresholds,
        "density_records": {
            "normal_seed_count": len(normal_density),
            "anomaly_seed_count": len(anomaly_density),
            "normal_median": float(np.median(normal_density)) if normal_density else None,
            "anomaly_median": float(np.median(anomaly_density)) if anomaly_density else None,
        },
        "runs": {},
        "scheme_b_channel_retention": {},
    }
    for label, group in (
        ("normal", [r for r in weighted_channel_records if not r["seed_anomaly"]]),
        ("anomaly", [r for r in weighted_channel_records if r["seed_anomaly"]]),
    ):
        retained = [r["retained"] for r in group]
        summary["scheme_b_channel_retention"][label] = {
            "records": len(group),
            "retained_records": int(sum(retained)),
            "retention_rate": float(np.mean(retained)) if retained else None,
            "mean_effective_span": float(np.mean([r["effective_span"] for r in group])) if group else None,
        }
    for label, rows in {
        "span_only": baseline,
        **density_runs,
        "scheme_b_effective_span": weighted_rows,
    }.items():
        values = np.asarray([row["contamination"] for row in rows], dtype=np.float64)
        selected = np.asarray([row["selected"] for row in rows], dtype=np.float64)
        summary["runs"][label] = {
            "mean_contamination": float(values.mean()),
            "final_contamination": float(values[-1]),
            "mean_selected_channels": float(selected.mean()),
            "final_selected_channels": float(selected[-1]),
            "contamination_lower_fraction": float(
                np.mean(values < np.asarray([r["contamination"] for r in baseline]))
            ) if label != "span_only" else None,
        }
    (output_dir / "bottle_density_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
