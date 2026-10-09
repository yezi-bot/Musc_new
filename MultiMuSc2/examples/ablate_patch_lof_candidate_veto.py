import argparse
import copy
import json
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
    "baseline": {"tail_veto": False},
    "lof_tail_veto": {"tail_veto": True},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ablate a causal high-LOF patch-ratio veto for expert candidates."
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
        "--patch-lof-quantile",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--risk-quantile",
        type=float,
        default=0.7,
    )
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def validate_quantile(name, value):
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


class TailVetoExpertManager(SplitSupportExpertManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tail_veto_enabled = False
        self.current_lof_tail_risk = None
        self.current_lof_tail_risk_threshold = None
        self.last_base_is_candidate = False
        self.last_candidate_vetoed = False

    def configure_tail_veto(self, enabled, risk, threshold):
        self.tail_veto_enabled = bool(enabled)
        self.current_lof_tail_risk = risk
        self.current_lof_tail_risk_threshold = threshold
        self.last_base_is_candidate = False
        self.last_candidate_vetoed = False

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
        self.last_base_is_candidate = bool(
            ms_score is not None
            and channel_support is not None
            and ms_threshold is not None
            and support_threshold is not None
            and ms_score <= ms_threshold
            and channel_support >= support_threshold
        )
        veto_ready = bool(
            self.current_lof_tail_risk is not None
            and self.current_lof_tail_risk_threshold is not None
        )
        self.last_candidate_vetoed = bool(
            self.tail_veto_enabled
            and veto_ready
            and self.last_base_is_candidate
            and self.current_lof_tail_risk
            > self.current_lof_tail_risk_threshold
        )
        if self.last_candidate_vetoed:
            return False, None, None
        return super()._consider_candidate(
            step,
            image_id,
            dino_patch_features,
            ms_score,
            channel_support,
            ms_threshold,
            support_threshold,
        )


def build_tail_veto_manager(config):
    committee = config["models"]["dynamic_committee"]
    return TailVetoExpertManager(
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


class TailVetoLifecycleState(SoftLofLifecycleStateV2):
    manager: TailVetoExpertManager

    def __init__(
        self,
        config,
        device,
        variant,
        patch_lof_quantile,
        risk_quantile,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        super().__init__(config, device, "baseline")
        self.variant = variant
        self.tail_veto_enabled = VARIANTS[variant]["tail_veto"]
        self.patch_lof_quantile = float(patch_lof_quantile)
        self.risk_quantile = float(risk_quantile)
        self.manager = build_tail_veto_manager(config)
        self.patch_lof_history = []
        self.lof_tail_risk_history = []

    def _current_tail_statistics(self, lof_scores, patch_count):
        if lof_scores is None:
            return None, None, None, None
        scores = torch.as_tensor(
            lof_scores,
            dtype=torch.float32,
        ).cpu()
        if scores.shape != (patch_count,):
            raise ValueError("LOF score count does not match patch count")
        if not torch.isfinite(scores).all():
            raise ValueError("LOF scores contain non-finite values")
        if (scores <= 0).any():
            raise ValueError("LOF scores must be positive")

        patch_threshold = (
            float(
                np.quantile(
                    self.patch_lof_history,
                    self.patch_lof_quantile,
                )
            )
            if self.patch_lof_history
            else None
        )
        risk = (
            float((scores > patch_threshold).float().mean())
            if patch_threshold is not None
            else None
        )
        risk_threshold = (
            float(
                np.quantile(
                    self.lof_tail_risk_history,
                    self.risk_quantile,
                )
            )
            if self.lof_tail_risk_history
            else None
        )
        return scores, patch_threshold, risk, risk_threshold

    def process(self, features, lof_scores):
        patch_count = int(features.shape[0])
        (
            scores,
            patch_threshold,
            risk,
            risk_threshold,
        ) = self._current_tail_statistics(lof_scores, patch_count)
        self.manager.configure_tail_veto(
            enabled=self.tail_veto_enabled,
            risk=risk,
            threshold=risk_threshold,
        )

        record = super().process(features, lof_scores)
        veto_ready = risk is not None and risk_threshold is not None
        record.update(
            {
                "patch_lof_threshold": patch_threshold,
                "lof_tail_risk": risk,
                "lof_tail_risk_threshold": risk_threshold,
                "tail_veto_ready": veto_ready,
                "base_is_candidate": (
                    self.manager.last_base_is_candidate
                ),
                "candidate_vetoed": (
                    self.manager.last_candidate_vetoed
                ),
            }
        )

        if scores is not None:
            self.patch_lof_history.extend(scores.tolist())
        if risk is not None:
            self.lof_tail_risk_history.append(risk)
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
                "patch_lof_threshold": record[
                    "patch_lof_threshold"
                ],
                "lof_tail_risk": record["lof_tail_risk"],
                "lof_tail_risk_threshold": record[
                    "lof_tail_risk_threshold"
                ],
                "tail_veto_ready": record["tail_veto_ready"],
                "base_is_candidate": record["base_is_candidate"],
                "candidate_vetoed": record["candidate_vetoed"],
                "is_candidate": record["is_candidate"],
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
                "channel_support": record["channel_support"],
                "support_threshold": record["support_threshold"],
                "channel_count_after": record["channel_count_after"],
                "mature_channel_count_after": record[
                    "mature_channel_count_after"
                ],
            }
        )
    return rows


def summarize_veto(timeline):
    vetoed = [record for record in timeline if record["candidate_vetoed"]]
    ready = [record for record in timeline if record["tail_veto_ready"]]
    risks = [
        record["lof_tail_risk"]
        for record in timeline
        if record["lof_tail_risk"] is not None
    ]
    return {
        "tail_veto_ready_steps": len(ready),
        "base_candidate_count": sum(
            bool(record["base_is_candidate"]) for record in timeline
        ),
        "candidate_veto_count": len(vetoed),
        "vetoed_normal": sum(
            record["anomaly_type"] == "good" for record in vetoed
        ),
        "vetoed_abnormal": sum(
            record["anomaly_type"] != "good" for record in vetoed
        ),
        "mean_lof_tail_risk": (
            float(np.mean(risks)) if risks else None
        ),
    }


def run_variant(
    stream,
    config,
    variant,
    seed,
    device,
    output_dir,
    patch_lof_quantile,
    risk_quantile,
):
    state = TailVetoLifecycleState(
        config=config,
        device=device,
        variant=variant,
        patch_lof_quantile=patch_lof_quantile,
        risk_quantile=risk_quantile,
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
        seed_dir / "lof_tail_veto_steps.csv",
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
        **summarize_veto(state.timeline),
    }


def main():
    args = parse_args()
    validate_quantile("patch_lof_quantile", args.patch_lof_quantile)
    validate_quantile("risk_quantile", args.risk_quantile)
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
                    patch_lof_quantile=args.patch_lof_quantile,
                    risk_quantile=args.risk_quantile,
                )
            )
        del stream
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(output_dir / "lof_tail_veto_runs.csv", rows)
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
        "patch_lof_quantile": args.patch_lof_quantile,
        "risk_quantile": args.risk_quantile,
        "patch_threshold_rule": (
            "strict-history patch LOF quantile"
        ),
        "risk_rule": (
            "fraction of current patches above the historical patch threshold"
        ),
        "veto_rule": (
            "veto an otherwise valid candidate when current risk is above "
            "the strict-history risk quantile"
        ),
        "early_unavailable_fallback": "no veto",
        "lifecycle_rule": "unchanged raw Channel support TTL",
        "channel_update_rule": "unchanged baseline distance-rank span",
        "extraction_seconds": extraction_seconds,
        "resolved_configs": resolved_configs,
        "runs": rows,
        "summary": summary,
        "gt_usage": "GT is added after online replay for debug summaries.",
    }
    (output_dir / "lof_tail_veto_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
