import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __package__:
    from .run_dynamic_dino import (
        git_commit,
        load_dinov2,
        run_seed,
        sha256_file,
        write_csv,
    )
else:
    from run_dynamic_dino import (
        git_commit,
        load_dinov2,
        run_seed,
        sha256_file,
        write_csv,
    )


VARIANTS = {
    "baseline": {
        "representative_mode": "historical_quality",
        "admission_ttl_mode": "historical_gap",
    },
    "latest_only": {
        "representative_mode": "latest_candidate",
        "admission_ttl_mode": "historical_gap",
    },
    "base_ttl_only": {
        "representative_mode": "historical_quality",
        "admission_ttl_mode": "base",
    },
    "latest_base_ttl": {
        "representative_mode": "latest_candidate",
        "admission_ttl_mode": "base",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ablate cluster representative selection and admission TTL."
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
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def config_for_variant(base_config, variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")

    config = copy.deepcopy(base_config)
    committee = config["models"]["dynamic_committee"]
    committee.update(VARIANTS[variant])
    return config


def _event_is_abnormal(event):
    value = event.get("expert_is_anomaly_debug")
    if value is None:
        raise ValueError(
            "expert_is_anomaly_debug is required for offline summary"
        )
    return bool(value)


def summarize_timeline(timeline):
    admission_events = []
    deletion_events = []

    for record in timeline:
        for event in record.get("expert_lifecycle_events", []):
            if event.get("decision") == "admitted":
                admission_events.append(event)
            if event.get("deleted_this_step", False):
                deletion_events.append(event)

    admitted_normal = sum(
        not _event_is_abnormal(event)
        for event in admission_events
    )
    admitted_abnormal = sum(
        _event_is_abnormal(event)
        for event in admission_events
    )
    deleted_normal = sum(
        not _event_is_abnormal(event)
        for event in deletion_events
    )
    deleted_abnormal = sum(
        _event_is_abnormal(event)
        for event in deletion_events
    )

    final_experts = (
        timeline[-1].get("expert_states_after", [])
        if timeline
        else []
    )
    final_normal = sum(
        expert.get("expert_is_anomaly_debug") is False
        for expert in final_experts
    )
    final_abnormal = sum(
        expert.get("expert_is_anomaly_debug") is True
        for expert in final_experts
    )

    cluster_admissions = Counter(
        int(event["cluster_id"])
        for event in admission_events
    )
    repeated_cluster_admissions = sum(
        count - 1
        for count in cluster_admissions.values()
        if count > 1
    )
    historical_representative_admissions = sum(
        int(event["expert_image_id"])
        != int(event["admission_candidate_image_id"])
        for event in admission_events
    )

    return {
        "admitted_normal": admitted_normal,
        "admitted_abnormal": admitted_abnormal,
        "deleted_normal": deleted_normal,
        "deleted_abnormal": deleted_abnormal,
        "final_normal": final_normal,
        "final_abnormal": final_abnormal,
        "normal_false_deletion_rate": (
            deleted_normal / admitted_normal
            if admitted_normal
            else 0.0
        ),
        "abnormal_deletion_rate": (
            deleted_abnormal / admitted_abnormal
            if admitted_abnormal
            else 0.0
        ),
        "repeated_cluster_admissions": repeated_cluster_admissions,
        "historical_representative_admissions": (
            historical_representative_admissions
        ),
    }


def load_timeline(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean_summary(rows):
    summary = {}
    if not rows:
        return summary

    excluded = {
        "variant",
        "representative_mode",
        "admission_ttl_mode",
        "seed",
    }
    for key in rows[0]:
        if key in excluded:
            continue
        values = [
            row[key]
            for row in rows
            if isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
        ]
        if values:
            summary[f"{key}_mean"] = sum(values) / len(values)
    return summary


def validate_config(config):
    if config["models"]["dynamic_fusion"]["mode"] != "dino_only":
        raise ValueError(
            "ablation runner only accepts dino_only mode"
        )
    if config["testing"].get("use_rscin", False):
        raise ValueError(
            "strict online ablation does not allow RsCIN"
        )


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
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
    variant_summaries = {}
    resolved_configs = {}

    for variant, settings in VARIANTS.items():
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
            timeline = load_timeline(
                variant_dir / f"seed_{seed}" / "timeline.json"
            )
            row = {
                "variant": variant,
                **settings,
                **metrics,
                **summarize_timeline(timeline),
            }
            rows.append(row)
            variant_rows.append(row)

        write_csv(
            variant_dir / "reentry_ablation_runs.csv",
            variant_rows,
        )
        variant_summaries[variant] = mean_summary(variant_rows)

    write_csv(output_dir / "reentry_ablation_runs.csv", rows)
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
        "resolved_configs": resolved_configs,
        "runs": rows,
        "summary": variant_summaries,
    }
    (output_dir / "reentry_ablation_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
