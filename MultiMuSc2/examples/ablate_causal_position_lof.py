import argparse
import copy
import csv
import json
import sys
from pathlib import Path

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
from MultiMuSc2.examples.run_dynamic_dino import (
    git_commit,
    load_dinov2,
    run_seed,
    sha256_file,
    write_csv,
)


VARIANTS = {
    "distance_rank": {
        "patch_reliability_mode": "distance_rank",
        "lof_position_radius": 0,
        "lof_soft_floor": True,
        "lof_reliability_minimum": 0.1,
    },
    "causal_lof_exact": {
        "patch_reliability_mode": "causal_lof",
        "lof_position_radius": 0,
        "lof_soft_floor": True,
        "lof_reliability_minimum": 0.1,
    },
    "causal_lof_radius1": {
        "patch_reliability_mode": "causal_lof",
        "lof_position_radius": 1,
        "lof_soft_floor": True,
        "lof_reliability_minimum": 0.1,
    },
    "causal_lof_radius1_no_floor": {
        "patch_reliability_mode": "causal_lof",
        "lof_position_radius": 1,
        "lof_soft_floor": False,
        "lof_reliability_minimum": 1.0e-6,
    },
}


LOF_LIFECYCLE_FIELDS = [
    "variant",
    "seed",
    "step",
    "stream_anomaly_type_debug",
    "expert_id",
    "expert_image_id",
    "expert_admission_step",
    "admission_phase",
    "expert_anomaly_type_debug",
    "raw_channel_support",
    "gated_channel_support",
    "lof_tail",
    "lof_threshold",
    "lof_calibration_step",
    "lof_calibration_history_count",
    "lof_backfilled_this_step",
    "lof_rejected",
    "ttl_before",
    "ttl_after",
    "signal_decision",
    "deleted_this_step",
]


def stream_phase(step, sample_count):
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    phase_index = min(2, int(step) * 3 // sample_count)
    return ("early", "middle", "late")[phase_index]


def required_expert_label(state):
    value = state.get("expert_is_anomaly_debug")
    if value is None:
        raise ValueError(
            "expert_is_anomaly_debug is required for LOF summary"
        )
    return bool(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablate strictly causal positional LOF signals."
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
            "lof_tail_fraction": 0.15,
            "lof_history_quantile": 0.85,
            "lof_reliability_minimum": 0.1,
            "lof_position_chunk_size": 64,
            "lof_image_chunk_size": 8,
        }
    )
    committee.update(VARIANTS[variant])
    return config


def lof_audit_rows(timeline, variant, seed):
    rows = []
    sample_count = len(timeline)
    for record in timeline:
        states = {
            int(expert["expert_id"]): expert
            for expert in record.get("expert_states_before", [])
        }
        lifecycle = {
            int(event["expert_id"]): event
            for event in record.get("expert_lifecycle_events", [])
            if event.get("decision") != "admitted"
        }
        raw_supports = record.get("raw_expert_channel_supports", {})
        gated_supports = record.get("expert_channel_supports", {})
        lof_tails = record.get("expert_lof_tails", {})
        lof_thresholds = record.get("expert_lof_thresholds", {})
        calibration_steps = record.get(
            "expert_lof_calibration_steps",
            {},
        )
        calibration_history_counts = record.get(
            "expert_lof_calibration_history_counts",
            {},
        )
        backfilled = record.get("expert_lof_backfilled", {})
        rejected = record.get("expert_lof_rejected", {})

        for expert_id, state in states.items():
            key = str(expert_id)
            event = lifecycle.get(expert_id, {})
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "step": record["step"],
                    "stream_anomaly_type_debug": record[
                        "anomaly_type"
                    ],
                    "expert_id": expert_id,
                    "expert_image_id": state["image_id"],
                    "expert_admission_step": state["admission_step"],
                    "admission_phase": stream_phase(
                        state["admission_step"],
                        sample_count,
                    ),
                    "expert_anomaly_type_debug": state.get(
                        "expert_anomaly_type_debug"
                    ),
                    "raw_channel_support": raw_supports.get(
                        key,
                        raw_supports.get(expert_id),
                    ),
                    "gated_channel_support": gated_supports.get(
                        key,
                        gated_supports.get(expert_id),
                    ),
                    "lof_tail": lof_tails.get(
                        key,
                        lof_tails.get(expert_id),
                    ),
                    "lof_threshold": lof_thresholds.get(
                        key,
                        lof_thresholds.get(expert_id),
                    ),
                    "lof_calibration_step": calibration_steps.get(
                        key,
                        calibration_steps.get(expert_id),
                    ),
                    "lof_calibration_history_count": (
                        calibration_history_counts.get(
                            key,
                            calibration_history_counts.get(expert_id),
                        )
                    ),
                    "lof_backfilled_this_step": backfilled.get(
                        key,
                        backfilled.get(expert_id, False),
                    ),
                    "lof_rejected": rejected.get(
                        key,
                        rejected.get(expert_id, False),
                    ),
                    "ttl_before": event.get("ttl_before"),
                    "ttl_after": event.get("ttl_after"),
                    "signal_decision": event.get(
                        "signal_decision"
                    ),
                    "deleted_this_step": event.get(
                        "deleted_this_step"
                    ),
                }
            )
    return rows


def write_lof_lifecycle_csv(path, rows):
    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=LOF_LIFECYCLE_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_lof(timeline):
    labels = ("normal", "abnormal")
    phases = ("early", "middle", "late")
    step_available = {label: 0 for label in labels}
    step_rejected = {label: 0 for label in labels}
    unique_available = {label: set() for label in labels}
    unique_rejected = {label: set() for label in labels}
    phase_available = {
        phase: {label: 0 for label in labels}
        for phase in phases
    }
    phase_rejected = {
        phase: {label: 0 for label in labels}
        for phase in phases
    }
    available = 0
    unavailable = 0
    sample_count = len(timeline)
    for record in timeline:
        states = {
            int(expert["expert_id"]): expert
            for expert in record.get("expert_states_before", [])
        }
        tails = record.get("expert_lof_tails", {})
        rejected = record.get("expert_lof_rejected", {})
        for expert_id, state in states.items():
            is_abnormal = required_expert_label(state)
            label = "abnormal" if is_abnormal else "normal"
            phase = stream_phase(
                state["admission_step"],
                sample_count,
            )
            key = str(expert_id)
            tail = tails.get(key, tails.get(expert_id))
            is_rejected = bool(
                rejected.get(key, rejected.get(expert_id, False))
            )
            if tail is None:
                unavailable += 1
                continue
            available += 1
            step_available[label] += 1
            unique_available[label].add(expert_id)
            phase_available[phase][label] += 1
            if is_rejected:
                step_rejected[label] += 1
                unique_rejected[label].add(expert_id)
                phase_rejected[phase][label] += 1

    summary = {
        "lof_available_expert_steps": available,
        "lof_unavailable_expert_steps": unavailable,
    }
    for label in labels:
        summary[f"lof_{label}_available_expert_steps"] = (
            step_available[label]
        )
        summary[f"lof_{label}_rejected_expert_steps"] = (
            step_rejected[label]
        )
        summary[f"lof_{label}_expert_step_rejection_rate"] = (
            step_rejected[label] / step_available[label]
            if step_available[label]
            else 0.0
        )
        summary[f"lof_{label}_available_unique_experts"] = len(
            unique_available[label]
        )
        summary[f"lof_{label}_rejected_unique_experts"] = len(
            unique_rejected[label]
        )
        summary[f"lof_{label}_unique_expert_rejection_rate"] = (
            len(unique_rejected[label]) / len(unique_available[label])
            if unique_available[label]
            else 0.0
        )
    for phase in phases:
        for label in labels:
            denominator = phase_available[phase][label]
            numerator = phase_rejected[phase][label]
            summary[
                f"lof_{phase}_{label}_available_expert_steps"
            ] = denominator
            summary[
                f"lof_{phase}_{label}_rejected_expert_steps"
            ] = numerator
            summary[
                f"lof_{phase}_{label}_expert_step_rejection_rate"
            ] = numerator / denominator if denominator else 0.0
    return summary


def validate_config(config):
    if config["models"]["dynamic_fusion"]["mode"] != "dino_only":
        raise ValueError("LOF ablation only accepts dino_only mode")
    if config["testing"].get("use_rscin", False):
        raise ValueError("strict online ablation does not allow RsCIN")


def main():
    args = parse_args()
    with Path(args.config).resolve().open(
        "r",
        encoding="utf-8",
    ) as handle:
        base_config = yaml.safe_load(handle)
    validate_config(base_config)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model = load_dinov2(
        args.dinov2_repo,
        args.checkpoint,
        device,
    )

    rows = []
    summaries = {}
    resolved_configs = {}
    for variant in args.variants:
        settings = VARIANTS[variant]
        config = config_for_variant(base_config, variant)
        resolved_configs[variant] = config
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_rows = []

        for seed in args.seeds:
            metrics = run_seed(
                config,
                args,
                seed,
                model,
                device,
                variant_dir,
            )
            timeline_path = (
                variant_dir / f"seed_{seed}" / "timeline.json"
            )
            timeline = json.loads(
                timeline_path.read_text(encoding="utf-8")
            )
            audit = lof_audit_rows(timeline, variant, seed)
            write_lof_lifecycle_csv(
                variant_dir / f"seed_{seed}" / "lof_lifecycle.csv",
                audit,
            )
            row = {
                "variant": variant,
                **settings,
                **metrics,
                **summarize_timeline(timeline),
                **summarize_lof(timeline),
            }
            rows.append(row)
            variant_rows.append(row)

        write_csv(
            variant_dir / "causal_lof_runs.csv",
            variant_rows,
        )
        summaries[variant] = mean_summary(variant_rows)

    write_csv(output_dir / "causal_lof_runs.csv", rows)
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
        "variants": VARIANTS,
        "variants": {
            variant: VARIANTS[variant]
            for variant in args.variants
        },
        "resolved_configs": resolved_configs,
        "runs": rows,
        "summary": summaries,
    }
    (output_dir / "causal_lof_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()