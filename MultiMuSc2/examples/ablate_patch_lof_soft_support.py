import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from MultiMuSc2.datasets.mvtec import DatasetSplit, MVTecDataset
from MultiMuSc2.examples.ablate_dynamic_expert_reentry import (
    mean_summary,
    summarize_timeline,
)
from MultiMuSc2.examples.run_dynamic_dino import (
    DinoFeatureExtractor,
    annotate_expert_debug_labels,
    git_commit,
    lifecycle_csv_rows,
    load_dinov2,
    seed_everything,
    select_dataset,
    sha256_file,
    write_csv,
    write_lifecycle_csv,
)
from MultiMuSc2.models.modules._CAUSAL_POSITION_LOF import (
    CausalPositionLofMemory,
)
from MultiMuSc2.models.modules._CHANNEL import ChannelMemory
from MultiMuSc2.models.modules._DYNAMIC_DINO_STREAM import (
    chunked_dino_score,
)
from MultiMuSc2.models.modules._DYNAMIC_EXPERT import (
    DynamicExpertManager,
)


VARIANTS = {
    "baseline": {
        "soft_candidate": False,
        "soft_span_product": False,
    },
    "soft_candidate_only": {
        "soft_candidate": True,
        "soft_span_product": False,
    },
    "soft_span_product": {
        "soft_candidate": False,
        "soft_span_product": True,
    },
    "soft_candidate_span_product": {
        "soft_candidate": True,
        "soft_span_product": True,
    },
}

STEP_FIELDS = [
    "variant",
    "seed",
    "step",
    "image_path",
    "anomaly_type_debug",
    "is_anomaly_debug",
    "lof_available",
    "mean_patch_weight",
    "low_weight_patch_fraction",
    "fraction_lof_gt_1_2",
    "fraction_lof_gt_1_5",
    "raw_channel_support",
    "soft_channel_support",
    "applied_channel_support",
    "raw_span_increment",
    "weighted_span_increment",
    "applied_span_increment",
    "admitted_expert_id",
    "deleted_expert_ids",
    "active_experts_before",
    "active_experts_after",
    "channel_count_after",
    "mature_channel_count_after",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ablate radius-0 patch-LOF soft support and effective span."
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
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def lof_to_patch_weight(lof_scores):
    if lof_scores is None:
        return None
    scores = torch.as_tensor(lof_scores, dtype=torch.float32).cpu()
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("lof_scores must be a non-empty vector")
    if not torch.isfinite(scores).all():
        raise ValueError("lof_scores contain non-finite values")
    if (scores <= 0).any():
        raise ValueError("lof_scores must be positive")
    return (1.0 / scores.clamp(min=1.0)).clamp(
        min=1.0e-6,
        max=1.0,
    )


def support_from_distances(distances, threshold, patch_weights=None):
    if distances is None or threshold is None:
        return None
    values = torch.as_tensor(distances, dtype=torch.float32).cpu()
    if values.ndim != 1:
        raise ValueError("distances must be a vector")
    supported = (values <= float(threshold)).float()
    if patch_weights is None:
        return float(supported.mean())
    weights = torch.as_tensor(
        patch_weights,
        dtype=torch.float32,
    ).cpu()
    if weights.shape != supported.shape:
        raise ValueError("patch_weights must match distances")
    if not torch.isfinite(weights).all():
        raise ValueError("patch_weights contain non-finite values")
    if ((weights <= 0) | (weights > 1)).any():
        raise ValueError("patch_weights must be in (0, 1]")
    return float((supported * weights).mean())


def patch_weight_audit(lof_scores, patch_count):
    if lof_scores is None:
        return {
            "lof_available": False,
            "mean_patch_weight": 1.0,
            "low_weight_patch_fraction": 0.0,
            "fraction_lof_gt_1_2": 0.0,
            "fraction_lof_gt_1_5": 0.0,
        }
    scores = torch.as_tensor(lof_scores, dtype=torch.float32).cpu()
    if scores.shape != (patch_count,):
        raise ValueError("LOF score count does not match patch count")
    weights = lof_to_patch_weight(scores)
    return {
        "lof_available": True,
        "mean_patch_weight": float(weights.mean()),
        "low_weight_patch_fraction": float((weights < 0.8).float().mean()),
        "fraction_lof_gt_1_2": float((scores > 1.2).float().mean()),
        "fraction_lof_gt_1_5": float((scores > 1.5).float().mean()),
    }


def validate_config(config):
    if config["models"]["dynamic_fusion"]["mode"] != "dino_only":
        raise ValueError("soft LOF ablation requires dino_only mode")
    if config["testing"].get("use_rscin", False):
        raise ValueError("strict online ablation does not allow RsCIN")
    committee = config["models"]["dynamic_committee"]
    if int(committee["r"]) != 1:
        raise ValueError("soft LOF ablation requires committee r=1")
    if int(committee["dino_layer"]) != 23:
        raise ValueError("soft LOF ablation requires DINO layer 23")


def config_for_variant(base_config, variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    config = copy.deepcopy(base_config)
    committee = config["models"]["dynamic_committee"]
    committee.update(
        {
            "representative_mode": "latest_candidate",
            "admission_ttl_mode": "historical_gap",
            "lof_k": 6,
            "lof_position_radius": 0,
            "lof_position_chunk_size": 64,
            "lof_image_chunk_size": 8,
        }
    )
    return config


def extract_seed_stream(config, args, seed, model, device):
    seed_everything(seed)
    dataset = MVTecDataset(
        source=args.data_root,
        classname=config["datasets"]["class_name"],
        resize=int(config["datasets"]["img_resize"]),
        imagesize=int(config["datasets"]["img_resize"]),
        split=DatasetSplit.TEST,
    )
    dataset = select_dataset(dataset, seed, args.max_samples)
    if len(dataset) < 2:
        raise ValueError("strict online ablation needs at least two images")

    committee = config["models"]["dynamic_committee"]
    extractor = DinoFeatureExtractor(
        model=model,
        device=device,
        feature_layers=[int(committee["dino_layer"])],
        r_list=[int(committee["r"])],
    )
    feature_key = (
        f"r{int(committee['r'])}_l{int(committee['dino_layer'])}"
    )
    lof_memory = CausalPositionLofMemory(
        k=int(committee.get("lof_k", 6)),
        device=device,
        position_chunk_size=int(
            committee.get("lof_position_chunk_size", 64)
        ),
        image_chunk_size=int(
            committee.get("lof_image_chunk_size", 8)
        ),
    )

    stream = []
    for step in range(len(dataset)):
        sample = dataset[step]
        current = extractor.extract(sample["image"].unsqueeze(0))[
            feature_key
        ]
        lof_scores = lof_memory.score(
            current,
            position_radius=0,
        )
        lof_memory.update(current)
        stream.append(
            {
                "step": step,
                "features": current.detach().cpu().clone(),
                "lof_scores": (
                    lof_scores.detach().cpu().clone()
                    if lof_scores is not None
                    else None
                ),
                "image_path": os.path.relpath(
                    sample["image_path"],
                    args.data_root,
                ),
                "anomaly_type": Path(sample["image_path"]).parent.name,
                "is_anomaly": int(sample["is_anomaly"]),
            }
        )
    return stream


class SoftLofLifecycleState:
    def __init__(self, config, device, variant):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        self.variant = variant
        self.settings = VARIANTS[variant]
        self.device = torch.device(device)
        committee = config["models"]["dynamic_committee"]
        scoring = config["models"]["scoring"]
        self.channel_quantile = float(
            committee.get("channel_distance_quantile", 0.7)
        )
        self.reliability_alpha = float(
            committee.get("reliability_alpha", 0.5)
        )
        self.reference_chunk_size = int(
            scoring.get("reference_chunk_size", 8)
        )

        self.memory = ChannelMemory(
            max_ttl=int(committee.get("channel_ttl", 5)),
            mature_span=float(committee.get("mature_span", 3.0)),
            density_k=int(committee.get("density_k", 5)),
            device=self.device,
            position_radius=int(committee.get("position_radius", 1)),
        )
        self.manager = DynamicExpertManager(
            ms_quantile=float(committee.get("ms_quantile", 0.3)),
            support_quantile=float(
                committee.get("support_quantile", 0.7)
            ),
            duplicate_similarity=float(
                committee.get("duplicate_similarity", 0.98)
            ),
            min_cluster_support=int(
                committee.get("min_cluster_support", 2)
            ),
            committee_cap=int(committee.get("committee_cap", 5)),
            base_ttl=int(committee.get("base_ttl", 5)),
            max_ttl=int(committee.get("max_ttl", 20)),
            ttl_gap_multiplier=float(
                committee.get("ttl_gap_multiplier", 2.0)
            ),
            representative_mode=committee.get(
                "representative_mode",
                "latest_candidate",
            ),
            admission_ttl_mode=committee.get(
                "admission_ttl_mode",
                "historical_gap",
            ),
        )
        self.features = []
        self.patch_weights = []
        self.distance_history = []
        self.timeline = []

    def _distance_threshold(self):
        if not self.distance_history:
            return None
        return float(
            np.quantile(
                self.distance_history,
                self.channel_quantile,
            )
        )

    def _distances(self, features, grid_size):
        return self.memory.patch_to_mature_distances(
            features,
            grid_size,
        )

    def _baseline_span_weights(self, features, grid_size):
        _, reliability = self.memory.patch_reliability(
            features,
            grid_size,
        )
        return (
            self.reliability_alpha
            + (1.0 - self.reliability_alpha) * reliability
        )

    def process(self, features, lof_scores):
        step = len(self.features)
        patch_count = int(features.shape[0])
        grid_size = math.isqrt(patch_count)
        if grid_size * grid_size != patch_count:
            raise ValueError("patch count must form a square grid")

        active_before = self.manager.active_experts_before_step()
        expert_states_before = self.manager.expert_audit_snapshot()
        ms_patch_score = chunked_dino_score(
            query=features,
            references=self.features,
            device=self.device,
            topmin_min=0.0,
            topmin_max=0.3,
            chunk_size=self.reference_chunk_size,
        )
        ms_score = (
            float(ms_patch_score.max())
            if ms_patch_score is not None
            else None
        )
        distance_threshold = self._distance_threshold()
        current_distances = self._distances(features, grid_size)
        raw_support = support_from_distances(
            current_distances,
            distance_threshold,
        )
        current_weights = lof_to_patch_weight(lof_scores)
        neutral_weights = torch.ones(patch_count)
        support_weights = (
            current_weights
            if current_weights is not None
            else neutral_weights
        )
        soft_support = support_from_distances(
            current_distances,
            distance_threshold,
            support_weights,
        )
        applied_support = (
            soft_support
            if self.settings["soft_support"]
            else raw_support
        )

        raw_expert_supports = {}
        soft_expert_supports = {}
        applied_expert_supports = {}
        for expert in active_before:
            expert_id = int(expert["expert_id"])
            image_id = int(expert["image_id"])
            if image_id >= step:
                raise RuntimeError("expert must be strictly historical")
            expert_features = self.features[image_id]
            distances = self._distances(expert_features, grid_size)
            raw = support_from_distances(
                distances,
                distance_threshold,
            )
            stored_weights = self.patch_weights[image_id]
            weights = (
                stored_weights
                if stored_weights is not None
                else neutral_weights
            )
            soft = support_from_distances(
                distances,
                distance_threshold,
                weights,
            )
            raw_expert_supports[expert_id] = raw
            soft_expert_supports[expert_id] = soft
            applied_expert_supports[expert_id] = (
                soft if self.settings["soft_support"] else raw
            )

        event = self.manager.advance(
            step=step,
            image_id=step,
            dino_patch_features=features,
            ms_score=ms_score,
            channel_support=applied_support,
            expert_channel_supports=applied_expert_supports,
        )

        if current_distances is not None:
            finite = current_distances[torch.isfinite(current_distances)]
            self.distance_history.extend(finite.tolist())

        baseline_span_weights = self._baseline_span_weights(
            features,
            grid_size,
        )
        soft_span_weights = (
            current_weights
            if current_weights is not None
            else baseline_span_weights
        )
        applied_span_weights = (
            soft_span_weights
            if self.settings["soft_span"]
            else baseline_span_weights
        )
        self.memory.update(
            features,
            image_id=step,
            grid_size=grid_size,
            patch_reliabilities=applied_span_weights,
        )

        self.features.append(features.detach().cpu().clone())
        self.patch_weights.append(
            current_weights.detach().cpu().clone()
            if current_weights is not None
            else None
        )
        audit = patch_weight_audit(lof_scores, patch_count)
        record = {
            "step": step,
            "ms_score": ms_score,
            "channel_distance_threshold": distance_threshold,
            "raw_channel_support": raw_support,
            "soft_channel_support": soft_support,
            "channel_support": applied_support,
            "raw_expert_channel_supports": raw_expert_supports,
            "soft_expert_channel_supports": soft_expert_supports,
            "expert_channel_supports": applied_expert_supports,
            **audit,
            "raw_span_increment": float(patch_count),
            "weighted_span_increment": float(
                soft_span_weights.sum()
            ),
            "applied_span_increment": float(
                applied_span_weights.sum()
            ),
            "active_experts_before": active_before,
            "expert_states_before": expert_states_before,
            "admitted_expert_id": event["admitted_expert_id"],
            "deleted_expert_ids": event["deleted_expert_ids"],
            "expert_lifecycle_events": event[
                "expert_lifecycle_events"
            ],
            "active_experts_after": (
                self.manager.active_experts_before_step()
            ),
            "expert_states_after": self.manager.expert_audit_snapshot(),
            "channel_count_after": len(self.memory.channels),
            "mature_channel_count_after": len(
                self.memory.mature_channels()
            ),
            "ms_threshold": event["ms_threshold"],
            "support_threshold": event["support_threshold"],
            "is_candidate": event["is_candidate"],
            "candidate_cluster_id": event["cluster_id"],
        }
        self.timeline.append(record)
        return record


def step_rows(timeline, variant, seed):
    rows = []
    for record in timeline:
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "step": record["step"],
                "image_path": record["image_path"],
                "anomaly_type_debug": record["anomaly_type"],
                "is_anomaly_debug": int(
                    record["anomaly_type"] != "good"
                ),
                "lof_available": record["lof_available"],
                "mean_patch_weight": record["mean_patch_weight"],
                "low_weight_patch_fraction": record[
                    "low_weight_patch_fraction"
                ],
                "fraction_lof_gt_1_2": record[
                    "fraction_lof_gt_1_2"
                ],
                "fraction_lof_gt_1_5": record[
                    "fraction_lof_gt_1_5"
                ],
                "raw_channel_support": record["raw_channel_support"],
                "soft_channel_support": record["soft_channel_support"],
                "applied_channel_support": record["channel_support"],
                "raw_span_increment": record["raw_span_increment"],
                "weighted_span_increment": record[
                    "weighted_span_increment"
                ],
                "applied_span_increment": record[
                    "applied_span_increment"
                ],
                "admitted_expert_id": record["admitted_expert_id"],
                "deleted_expert_ids": json.dumps(
                    record["deleted_expert_ids"]
                ),
                "active_experts_before": json.dumps(
                    [
                        expert["expert_id"]
                        for expert in record["active_experts_before"]
                    ]
                ),
                "active_experts_after": json.dumps(
                    [
                        expert["expert_id"]
                        for expert in record["active_experts_after"]
                    ]
                ),
                "channel_count_after": record["channel_count_after"],
                "mature_channel_count_after": record[
                    "mature_channel_count_after"
                ],
            }
        )
    return rows


def summarize_soft_signal(timeline):
    available = [record for record in timeline if record["lof_available"]]
    return {
        "lof_available_steps": len(available),
        "lof_unavailable_steps": len(timeline) - len(available),
        "mean_patch_weight": (
            float(
                np.mean(
                    [record["mean_patch_weight"] for record in available]
                )
            )
            if available
            else None
        ),
        "mean_low_weight_patch_fraction": (
            float(
                np.mean(
                    [
                        record["low_weight_patch_fraction"]
                        for record in available
                    ]
                )
            )
            if available
            else None
        ),
    }


def run_variant(stream, config, variant, seed, device, output_dir):
    state = SoftLofLifecycleState(config, device, variant)
    started = time.perf_counter()
    for item in stream:
        record = state.process(item["features"], item["lof_scores"])
        record["image_path"] = item["image_path"]
        record["anomaly_type"] = item["anomaly_type"]
    runtime_seconds = time.perf_counter() - started
    annotate_expert_debug_labels(state.timeline)

    seed_dir = output_dir / variant / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "timeline.json").write_text(
        json.dumps(state.timeline, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(
        seed_dir / "soft_lof_steps.csv",
        step_rows(state.timeline, variant, seed),
    )
    write_lifecycle_csv(
        seed_dir / "expert_lifecycle.csv",
        lifecycle_csv_rows(state.timeline),
    )
    return {
        "variant": variant,
        "seed": seed,
        "sample_count": len(stream),
        "runtime_seconds": runtime_seconds,
        **summarize_timeline(state.timeline),
        **summarize_soft_signal(state.timeline),
    }


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(base_config)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_dinov2(
        args.dinov2_repo,
        args.checkpoint,
        device,
    )

    rows = []
    extraction_seconds = {}
    resolved_configs = {}
    for seed in args.seeds:
        started = time.perf_counter()
        stream = extract_seed_stream(
            base_config,
            args,
            seed,
            model,
            device,
        )
        extraction_seconds[str(seed)] = time.perf_counter() - started
        for variant in args.variants:
            config = config_for_variant(base_config, variant)
            resolved_configs[variant] = config
            rows.append(
                run_variant(
                    stream,
                    config,
                    variant,
                    seed,
                    device,
                    output_dir,
                )
            )
        del stream
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(output_dir / "soft_lof_runs.csv", rows)
    summary = {
        variant: mean_summary(
            [row for row in rows if row["variant"] == variant]
        )
        for variant in args.variants
    }
    metadata = {
        "git_commit": git_commit(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu"
        ),
        "seeds": args.seeds,
        "max_samples": args.max_samples,
        "variants": {
            name: VARIANTS[name] for name in args.variants
        },
        "lof_weight_formula": "1 / max(lof, 1)",
        "soft_support_formula": "mean(supported_patch * patch_weight)",
        "lof_position_radius": 0,
        "lof_k": 6,
        "early_unavailable_fallback": (
            "neutral support weight; distance-rank span weight"
        ),
        "extraction_seconds": extraction_seconds,
        "resolved_configs": resolved_configs,
        "runs": rows,
        "summary": summary,
        "gt_usage": "GT is added after online replay for debug summaries.",
    }
    (output_dir / "soft_lof_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
