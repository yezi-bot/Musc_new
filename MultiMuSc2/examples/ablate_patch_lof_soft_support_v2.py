import argparse
import copy
import json
import math
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

from MultiMuSc2.examples.ablate_dynamic_expert_reentry import (
    mean_summary,
    summarize_timeline,
)
from MultiMuSc2.examples.ablate_patch_lof_soft_support import (
    SoftLofLifecycleState,
    extract_seed_stream,
    lof_to_patch_weight,
    patch_weight_audit,
    support_from_distances,
    validate_config,
)
from MultiMuSc2.examples.run_dynamic_dino import (
    annotate_expert_debug_labels,
    git_commit,
    lifecycle_csv_rows,
    load_dinov2,
    sha256_file,
    write_csv,
    write_lifecycle_csv,
)
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
    "soft_candidate_support",
    "applied_candidate_support",
    "lifecycle_support_threshold",
    "candidate_support_threshold",
    "raw_span_increment",
    "lof_span_increment",
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
            "Ablate split candidate support and multiplicative LOF span."
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


class SplitSupportExpertManager(DynamicExpertManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_support_history = []
        self.soft_support_history = []

    def _support_threshold(self, history):
        if not history:
            return None
        return float(np.quantile(history, self.support_quantile))

    def advance_split_support(
        self,
        step,
        image_id,
        dino_patch_features,
        ms_score,
        raw_channel_support,
        soft_candidate_support,
        raw_expert_supports,
        use_soft_candidate,
    ):
        if step != self.last_step + 1:
            raise ValueError(
                f"step must be {self.last_step + 1}, got {step}"
            )
        if not isinstance(raw_expert_supports, dict):
            raise TypeError("raw_expert_supports must be a dict")

        ms_score = self._optional_score("ms_score", ms_score)
        raw_channel_support = self._optional_score(
            "raw_channel_support",
            raw_channel_support,
            unit_interval=True,
        )
        soft_candidate_support = self._optional_score(
            "soft_candidate_support",
            soft_candidate_support,
            unit_interval=True,
        )
        ms_threshold = (
            float(np.quantile(self.ms_history, self.ms_quantile))
            if self.ms_history
            else None
        )
        lifecycle_support_threshold = self._support_threshold(
            self.raw_support_history
        )
        if use_soft_candidate:
            candidate_support = soft_candidate_support
            candidate_support_threshold = self._support_threshold(
                self.soft_support_history
            )
        else:
            candidate_support = raw_channel_support
            candidate_support_threshold = lifecycle_support_threshold

        (
            deleted_expert_ids,
            expert_lifecycle_events,
        ) = self._advance_lifecycle(
            step,
            lifecycle_support_threshold,
            raw_expert_supports,
        )
        (
            is_candidate,
            cluster_id,
            admitted_expert_id,
        ) = self._consider_candidate(
            step,
            int(image_id),
            dino_patch_features,
            ms_score,
            candidate_support,
            ms_threshold,
            candidate_support_threshold,
        )

        if admitted_expert_id is not None:
            admitted_expert = next(
                expert
                for expert in self.active_experts
                if expert["expert_id"] == admitted_expert_id
            )
            expert_lifecycle_events.append(
                {
                    "step": int(step),
                    "expert_id": int(admitted_expert_id),
                    "expert_image_id": int(admitted_expert["image_id"]),
                    "cluster_id": int(admitted_expert["cluster_id"]),
                    "admission_step": int(step),
                    "age": 0,
                    "channel_support": None,
                    "support_threshold": candidate_support_threshold,
                    "support_available": False,
                    "threshold_available": (
                        candidate_support_threshold is not None
                    ),
                    "support_gap": None,
                    "ttl_before": None,
                    "ttl_after": int(admitted_expert["ttl"]),
                    "patience_before": None,
                    "patience_after": int(admitted_expert["patience"]),
                    "last_supported_step_before": None,
                    "last_supported_step_after": int(step),
                    "signal_decision": "admitted",
                    "decision": "admitted",
                    "refreshed": False,
                    "deleted_this_step": False,
                    "alive_after": True,
                    "admission_candidate_image_id": int(image_id),
                    "admission_candidate_ms_score": ms_score,
                    "admission_ms_threshold": ms_threshold,
                    "admission_candidate_channel_support": (
                        candidate_support
                    ),
                    "admission_support_threshold": (
                        candidate_support_threshold
                    ),
                    "representative_mode": admitted_expert[
                        "representative_mode"
                    ],
                    "admission_ttl_mode": admitted_expert[
                        "admission_ttl_mode"
                    ],
                }
            )

        if ms_score is not None:
            self.ms_history.append(ms_score)
        if raw_channel_support is not None:
            self.raw_support_history.append(raw_channel_support)
        if soft_candidate_support is not None:
            self.soft_support_history.append(soft_candidate_support)
        self.last_step = step

        return {
            "step": step,
            "image_id": int(image_id),
            "ms_threshold": ms_threshold,
            "lifecycle_support_threshold": lifecycle_support_threshold,
            "candidate_support_threshold": candidate_support_threshold,
            "candidate_support": candidate_support,
            "is_candidate": is_candidate,
            "cluster_id": cluster_id,
            "admitted_expert_id": admitted_expert_id,
            "deleted_expert_ids": deleted_expert_ids,
            "expert_lifecycle_events": expert_lifecycle_events,
            "active_expert_ids_after_step": [
                expert["expert_id"] for expert in self.active_experts
            ],
        }


def build_split_manager(config):
    committee = config["models"]["dynamic_committee"]
    return SplitSupportExpertManager(
        ms_quantile=float(committee.get("ms_quantile", 0.3)),
        support_quantile=float(committee.get("support_quantile", 0.7)),
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


class SoftLofLifecycleStateV2(SoftLofLifecycleState):
    manager: SplitSupportExpertManager
    def __init__(self, config, device, variant):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        super().__init__(config, device, "baseline")
        self.variant = variant
        self.settings = VARIANTS[variant]
        self.manager = build_split_manager(config)

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
        lof_weights = (
            current_weights
            if current_weights is not None
            else neutral_weights
        )
        soft_candidate_support = support_from_distances(
            current_distances,
            distance_threshold,
            lof_weights,
        )

        raw_expert_supports = {}
        soft_expert_supports_debug = {}
        for expert in active_before:
            expert_id = int(expert["expert_id"])
            image_id = int(expert["image_id"])
            if image_id >= step:
                raise RuntimeError("expert must be strictly historical")
            expert_features = self.features[image_id]
            distances = self._distances(expert_features, grid_size)
            raw_expert_supports[expert_id] = support_from_distances(
                distances,
                distance_threshold,
            )
            stored_weights = self.patch_weights[image_id]
            expert_weights = (
                stored_weights
                if stored_weights is not None
                else neutral_weights
            )
            soft_expert_supports_debug[expert_id] = (
                support_from_distances(
                    distances,
                    distance_threshold,
                    expert_weights,
                )
            )

        event = self.manager .advance_split_support(
            step=step,
            image_id=step,
            dino_patch_features=features,
            ms_score=ms_score,
            raw_channel_support=raw_support,
            soft_candidate_support=soft_candidate_support,
            raw_expert_supports=raw_expert_supports,
            use_soft_candidate=self.settings["soft_candidate"],
        )

        if current_distances is not None:
            finite = current_distances[torch.isfinite(current_distances)]
            self.distance_history.extend(finite.tolist())

        baseline_span_weights = self._baseline_span_weights(
            features,
            grid_size,
        )
        lof_span_weights = baseline_span_weights * lof_weights
        if (lof_span_weights > baseline_span_weights + 1.0e-7).any():
            raise RuntimeError("LOF span product increased baseline weight")
        applied_span_weights = (
            lof_span_weights
            if self.settings["soft_span_product"]
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
            "soft_candidate_support": soft_candidate_support,
            "channel_support": event["candidate_support"],
            "raw_expert_channel_supports": raw_expert_supports,
            "soft_expert_channel_supports_debug": (
                soft_expert_supports_debug
            ),
            "expert_channel_supports": raw_expert_supports,
            **audit,
            "raw_span_increment": float(baseline_span_weights.sum()),
            "lof_span_increment": float(lof_span_weights.sum()),
            "applied_span_increment": float(applied_span_weights.sum()),
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
            "lifecycle_support_threshold": event[
                "lifecycle_support_threshold"
            ],
            "candidate_support_threshold": event[
                "candidate_support_threshold"
            ],
            "support_threshold": event["candidate_support_threshold"],
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
                "soft_candidate_support": record[
                    "soft_candidate_support"
                ],
                "applied_candidate_support": record["channel_support"],
                "lifecycle_support_threshold": record[
                    "lifecycle_support_threshold"
                ],
                "candidate_support_threshold": record[
                    "candidate_support_threshold"
                ],
                "raw_span_increment": record["raw_span_increment"],
                "lof_span_increment": record["lof_span_increment"],
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


def summarize_signal(timeline):
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
        "mean_span_weight_ratio": float(
            np.mean(
                [
                    record["applied_span_increment"]
                    / record["raw_span_increment"]
                    for record in timeline
                ]
            )
        ),
    }


def run_variant(stream, config, variant, seed, device, output_dir):
    state = SoftLofLifecycleStateV2(config, device, variant)
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
        seed_dir / "soft_lof_v2_steps.csv",
        step_rows(state.timeline, variant, seed),
    )
    write_lifecycle_csv(
        seed_dir / "expert_lifecycle.csv",
        lifecycle_csv_rows(state.timeline),
    )
    return {
        "variant": variant,
        **VARIANTS[variant],
        "seed": seed,
        "sample_count": len(stream),
        "runtime_seconds": runtime_seconds,
        **summarize_timeline(state.timeline),
        **summarize_signal(state.timeline),
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

    write_csv(output_dir / "soft_lof_v2_runs.csv", rows)
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
        "candidate_support_formula": (
            "mean(supported_patch * lof_weight)"
        ),
        "span_formula": "distance_rank_weight * lof_weight",
        "lifecycle_rule": (
            "raw expert support compared with raw support history"
        ),
        "candidate_rule": (
            "selected raw/soft support compared with matching history"
        ),
        "lof_position_radius": 0,
        "lof_k": 6,
        "early_unavailable_fallback": (
            "lof_weight=1; candidate and span equal baseline"
        ),
        "extraction_seconds": extraction_seconds,
        "resolved_configs": resolved_configs,
        "runs": rows,
        "summary": summary,
        "gt_usage": "GT is added after online replay for debug summaries.",
    }
    (output_dir / "soft_lof_v2_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
