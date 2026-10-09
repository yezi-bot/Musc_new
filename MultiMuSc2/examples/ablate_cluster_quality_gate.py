import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    extract_seed_stream,
    validate_config,
)
from MultiMuSc2.examples.ablate_patch_lof_soft_support_v2 import (
    SoftLofLifecycleStateV2,
    SplitSupportExpertManager,
    config_for_variant,
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


VARIANTS = {
    "baseline": {
        "gate_first_admission": False,
        "gate_reentry": False,
    },
    "first_min_gate": {
        "gate_first_admission": True,
        "gate_reentry": False,
    },
    "reentry_median_gate": {
        "gate_first_admission": False,
        "gate_reentry": True,
    },
    "split_quality_gate": {
        "gate_first_admission": True,
        "gate_reentry": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ablate causal cluster-quality gates for first admission "
            "and re-entry."
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
    parser.add_argument(
        "--quality-quantile",
        type=float,
        default=0.7,
    )
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


class QualityGateExpertManager(SplitSupportExpertManager):
    def __init__(
        self,
        *args,
        quality_quantile,
        gate_first_admission,
        gate_reentry,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.quality_quantile = float(quality_quantile)
        self.gate_first_admission = bool(gate_first_admission)
        self.gate_reentry = bool(gate_reentry)
        self.candidate_quality_history = []
        self._reset_quality_audit()

    def _reset_quality_audit(self):
        self.last_candidate_quality = None
        self.last_quality_threshold = None
        self.last_gate_quality = None
        self.last_gate_statistic = None
        self.last_gate_applied = False
        self.last_gate_available = False
        self.last_gate_pass = None
        self.last_admission_kind = None
        self.last_admission_blocked = False
        self.last_cluster_member_qualities = []

    def advance_split_support(self, *args, **kwargs):
        self._reset_quality_audit()
        return super().advance_split_support(*args, **kwargs)

    def _consider_candidate(
        self,
        step,
        image_id,
        dino_patch_features,
        ms_score,
        channel_support,
        ms_threshold,
        support_threshold,
    ):
        is_candidate = bool(
            ms_score is not None
            and channel_support is not None
            and ms_threshold is not None
            and support_threshold is not None
            and ms_score <= ms_threshold
            and channel_support >= support_threshold
        )
        if not is_candidate:
            return False, None, None

        embedding = self._image_embedding(dino_patch_features)
        quality = (
            channel_support / max(support_threshold, 1.0e-12)
            - ms_score / max(ms_threshold, 1.0e-12)
        )
        quality_threshold = (
            float(
                np.quantile(
                    self.candidate_quality_history,
                    self.quality_quantile,
                )
            )
            if self.candidate_quality_history
            else None
        )
        self.last_candidate_quality = float(quality)
        self.last_quality_threshold = quality_threshold
        self.candidate_quality_history.append(float(quality))

        cluster_id = None
        if self.clusters:
            centroids = torch.stack(
                [
                    F.normalize(cluster["embedding_sum"], dim=0)
                    for cluster in self.clusters
                ]
            )
            similarities = centroids @ embedding
            max_similarity, best_cluster = similarities.max(dim=0)
            if float(max_similarity) >= self.duplicate_similarity:
                cluster_id = int(best_cluster)

        if cluster_id is None:
            cluster_id = len(self.clusters)
            self.clusters.append(
                {
                    "embedding_sum": torch.zeros_like(embedding),
                    "members": [],
                    "active_expert_id": None,
                    "ever_admitted": False,
                    "admission_count": 0,
                }
            )

        cluster = self.clusters[cluster_id]
        cluster.setdefault("ever_admitted", False)
        cluster.setdefault("admission_count", 0)
        cluster["members"].append(
            {
                "step": step,
                "image_id": image_id,
                "embedding": embedding,
                "quality": float(quality),
                "ms_score": ms_score,
                "channel_support": channel_support,
                "ms_threshold": ms_threshold,
                "support_threshold": support_threshold,
            }
        )
        cluster["embedding_sum"] += embedding
        self.last_cluster_member_qualities = [
            float(member["quality"])
            for member in cluster["members"]
        ]

        if cluster["active_expert_id"] is not None:
            return True, cluster_id, None
        if len(cluster["members"]) < self.min_cluster_support:
            return True, cluster_id, None

        is_reentry = bool(cluster["ever_admitted"])
        self.last_admission_kind = (
            "reentry" if is_reentry else "first_admission"
        )
        if is_reentry:
            gate_applied = self.gate_reentry
            gate_quality = float(
                statistics.median(
                    self.last_cluster_member_qualities
                )
            )
            gate_statistic = "median_quality"
        else:
            gate_applied = self.gate_first_admission
            gate_quality = float(
                min(self.last_cluster_member_qualities)
            )
            gate_statistic = "minimum_quality"

        self.last_gate_applied = gate_applied
        self.last_gate_quality = gate_quality
        self.last_gate_statistic = gate_statistic
        self.last_gate_available = quality_threshold is not None
        if gate_applied:
            self.last_gate_pass = bool(
                quality_threshold is not None
                and gate_quality >= quality_threshold
            )
            if not self.last_gate_pass:
                self.last_admission_blocked = True
                return True, cluster_id, None
        else:
            self.last_gate_pass = None

        if len(self.active_experts) >= self.committee_cap:
            return True, cluster_id, None

        if self.representative_mode == "historical_quality":
            representative = max(
                cluster["members"],
                key=lambda member: member["quality"],
            )
        else:
            representative = cluster["members"][-1]

        member_steps = sorted(
            member["step"] for member in cluster["members"]
        )
        historical_max_support_gap = max(
            later - earlier
            for earlier, later in zip(
                member_steps,
                member_steps[1:],
            )
        )
        if self.admission_ttl_mode == "historical_gap":
            max_support_gap = historical_max_support_gap
            patience = self._patience_from_gap(max_support_gap)
        else:
            max_support_gap = 0
            patience = self.base_ttl

        expert = {
            "expert_id": self.next_expert_id,
            "image_id": representative["image_id"],
            "cluster_id": cluster_id,
            "admission_step": step,
            "cluster_support_at_admission": len(cluster["members"]),
            "max_support_gap": max_support_gap,
            "last_supported_step": step,
            "support_count": len(cluster["members"]),
            "patience": patience,
            "ttl": patience,
            "deletion_step": None,
            "last_event": "admitted",
            "last_channel_support": None,
            "representative_mode": self.representative_mode,
            "admission_ttl_mode": self.admission_ttl_mode,
        }
        cluster["active_expert_id"] = self.next_expert_id
        cluster["ever_admitted"] = True
        cluster["admission_count"] += 1
        self.active_experts.append(expert)
        self.admitted_count += 1
        self.next_expert_id += 1
        return True, cluster_id, expert["expert_id"]


def build_quality_manager(config, variant, quality_quantile):
    committee = config["models"]["dynamic_committee"]
    settings = VARIANTS[variant]
    return QualityGateExpertManager(
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
        quality_quantile=quality_quantile,
        gate_first_admission=settings["gate_first_admission"],
        gate_reentry=settings["gate_reentry"],
    )


class QualityGateLifecycleState(SoftLofLifecycleStateV2):
    manager: QualityGateExpertManager

    def __init__(
        self,
        config,
        device,
        variant,
        quality_quantile,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        super().__init__(config, device, "baseline")
        self.variant = variant
        self.manager = build_quality_manager(
            config,
            variant,
            quality_quantile,
        )

    def process(self, features, lof_scores):
        record = super().process(features, lof_scores)
        record.update(
            {
                "candidate_quality": (
                    self.manager.last_candidate_quality
                ),
                "quality_threshold": (
                    self.manager.last_quality_threshold
                ),
                "quality_gate_statistic": (
                    self.manager.last_gate_statistic
                ),
                "quality_gate_value": (
                    self.manager.last_gate_quality
                ),
                "quality_gate_applied": (
                    self.manager.last_gate_applied
                ),
                "quality_gate_available": (
                    self.manager.last_gate_available
                ),
                "quality_gate_pass": self.manager.last_gate_pass,
                "quality_admission_kind": (
                    self.manager.last_admission_kind
                ),
                "quality_admission_blocked": (
                    self.manager.last_admission_blocked
                ),
                "cluster_member_qualities": list(
                    self.manager.last_cluster_member_qualities
                ),
            }
        )
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
                "is_candidate": record["is_candidate"],
                "candidate_cluster_id": record[
                    "candidate_cluster_id"
                ],
                "candidate_quality": record["candidate_quality"],
                "quality_threshold": record["quality_threshold"],
                "quality_gate_statistic": record[
                    "quality_gate_statistic"
                ],
                "quality_gate_value": record["quality_gate_value"],
                "quality_gate_applied": record[
                    "quality_gate_applied"
                ],
                "quality_gate_available": record[
                    "quality_gate_available"
                ],
                "quality_gate_pass": record["quality_gate_pass"],
                "quality_admission_kind": record[
                    "quality_admission_kind"
                ],
                "quality_admission_blocked": record[
                    "quality_admission_blocked"
                ],
                "cluster_member_qualities": json.dumps(
                    record["cluster_member_qualities"]
                ),
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
            }
        )
    return rows


def summarize_quality_gate(timeline):
    blocked = [
        record
        for record in timeline
        if record["quality_admission_blocked"]
    ]
    unavailable = [
        record
        for record in timeline
        if record["quality_gate_applied"]
        and not record["quality_gate_available"]
    ]
    return {
        "quality_blocked_event_count": len(blocked),
        "quality_blocked_normal_events": sum(
            record["anomaly_type"] == "good" for record in blocked
        ),
        "quality_blocked_abnormal_events": sum(
            record["anomaly_type"] != "good" for record in blocked
        ),
        "quality_gate_unavailable_events": len(unavailable),
        "first_admission_blocked_events": sum(
            record["quality_admission_kind"] == "first_admission"
            for record in blocked
        ),
        "reentry_blocked_events": sum(
            record["quality_admission_kind"] == "reentry"
            for record in blocked
        ),
    }


def run_variant(
    stream,
    config,
    variant,
    seed,
    device,
    output_dir,
    quality_quantile,
):
    state = QualityGateLifecycleState(
        config=config,
        device=device,
        variant=variant,
        quality_quantile=quality_quantile,
    )
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
        seed_dir / "cluster_quality_gate_steps.csv",
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
        **summarize_quality_gate(state.timeline),
    }


def main():
    args = parse_args()
    if not 0.0 < args.quality_quantile < 1.0:
        raise ValueError("quality_quantile must be between 0 and 1")
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
            config = config_for_variant(base_config, "baseline")
            resolved_configs[variant] = copy.deepcopy(config)
            rows.append(
                run_variant(
                    stream=stream,
                    config=config,
                    variant=variant,
                    seed=seed,
                    device=device,
                    output_dir=output_dir,
                    quality_quantile=args.quality_quantile,
                )
            )
        del stream
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(output_dir / "cluster_quality_gate_runs.csv", rows)
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
        "quality_quantile": args.quality_quantile,
        "variants": {
            name: VARIANTS[name] for name in args.variants
        },
        "quality_formula": (
            "channel_support / support_threshold "
            "- ms_score / ms_threshold"
        ),
        "threshold_rule": (
            "strict-history quantile of all earlier base-candidate qualities"
        ),
        "first_admission_rule": "minimum cluster-member quality",
        "reentry_rule": "median cluster-member quality",
        "unavailable_rule": (
            "keep cluster provisional and retry on a later member"
        ),
        "channel_rule": "unchanged baseline distance-rank ChannelMemory",
        "lifecycle_rule": "unchanged raw Channel-support TTL",
        "extraction_seconds": extraction_seconds,
        "resolved_configs": resolved_configs,
        "runs": rows,
        "summary": summary,
        "gt_usage": "GT is attached after online replay for debug only.",
    }
    (output_dir / "cluster_quality_gate_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
