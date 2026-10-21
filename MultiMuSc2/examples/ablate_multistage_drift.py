import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from datasets.gradual_drift import build_multistage_drift_schedule
from examples.ablate_gradual_drift import (
    VARIANTS,
    load_config,
    run_variant,
    safe_auroc,
)


PRESSURE_VARIANTS = (
    "photometric_multistage_cap2_reject",
    "photometric_multistage_cap2_replace",
)


def _segment_metrics(step_rows, schedule):
    metrics = {}
    segment_ids = []
    for item in schedule:
        if (
            item["segment_kind"] == "plateau"
            and item["segment_id"] not in segment_ids
        ):
            segment_ids.append(item["segment_id"])

    for segment_id in segment_ids:
        indices = [
            index
            for index, item in enumerate(schedule)
            if item["segment_id"] == segment_id
        ]
        labels = np.asarray(
            [int(step_rows[index]["is_anomaly"]) for index in indices]
        )
        scores = np.asarray(
            [float(step_rows[index]["raw_image_score"]) for index in indices]
        )
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        prefix = segment_id
        metrics[f"{prefix}_sample_count"] = len(indices)
        metrics[f"{prefix}_raw_auroc"] = safe_auroc(labels, scores)
        metrics[f"{prefix}_normal_score_mean"] = (
            float(np.mean(normal_scores)) if normal_scores.size else None
        )
        metrics[f"{prefix}_anomaly_score_mean"] = (
            float(np.mean(anomaly_scores)) if anomaly_scores.size else None
        )
        metrics[f"{prefix}_max_active_experts"] = max(
            int(step_rows[index]["active_expert_count_after"])
            for index in indices
        )
        metrics[f"{prefix}_max_expert_slots"] = max(
            int(step_rows[index]["expert_slot_count_after"])
            for index in indices
        )
        metrics[f"{prefix}_max_dormant_experts"] = max(
            int(step_rows[index]["dormant_expert_count_after"])
            for index in indices
        )
    return metrics


def _read_step_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_results(output_dir, schedule, rows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "multistage_drift_runs.csv"
    json_path = output_dir / "multistage_drift_summary.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(
            {
                "schedule": schedule,
                "runs": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed-budget dynamic experts under a same-category "
            "multi-stage photometric stream."
        )
    )
    parser.add_argument(
        "--config",
        default=PROJECT_ROOT / "configs" / "musc.yaml",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--sequence-name", default="abcd_recovery")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--anomaly-ratio", type=float, default=(1.0 / 3.0))
    parser.add_argument("--initial-plateau-steps", type=int, default=4)
    parser.add_argument("--transition-steps", type=int, default=2)
    parser.add_argument("--plateau-steps", type=int, default=4)
    parser.add_argument("--lab-ewma-alpha", type=float, default=0.3)
    parser.add_argument("--lab-ewma-threshold-k", type=float, default=3.0)
    parser.add_argument("--lab-ewma-baseline-window", type=int, default=8)
    parser.add_argument("--lab-ewma-confirm-steps", type=int, default=2)
    parser.add_argument(
        "--photometric-support-confirm-steps",
        type=int,
        default=2,
    )
    parser.add_argument("--photometric-l-ewma-floor", type=float, default=0.04)
    parser.add_argument("--photometric-ab-ewma-floor", type=float, default=0.01)
    parser.add_argument(
        "--photometric-consistency-fraction",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=PRESSURE_VARIANTS,
        default=list(PRESSURE_VARIANTS),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    schedule = build_multistage_drift_schedule(
        initial_plateau_steps=args.initial_plateau_steps,
        transition_steps=args.transition_steps,
        plateau_steps=args.plateau_steps,
    )
    args.stream_samples = len(schedule)
    args.brightness_delta = 0.25
    args.contrast_delta = 0.08
    args.saturation_delta = 0.30
    args.hue_delta = 0.04

    base_config = load_config(args.config)
    rows = []
    for seed in args.seeds:
        for variant_name in args.variants:
            print(
                f"Running category={args.category}, "
                f"sequence={args.sequence_name}, seed={seed}, "
                f"variant={variant_name}"
            )
            row = run_variant(
                base_config=base_config,
                args=args,
                drift_type="multistage",
                drift_level=args.sequence_name,
                seed=seed,
                variant_name=variant_name,
                schedule_override=schedule,
            )
            run_output = (
                Path(args.output_dir)
                / args.category
                / "multistage"
                / args.sequence_name
                / f"seed_{seed}"
                / variant_name
            )
            step_rows = _read_step_rows(
                run_output / "gradual_drift_steps.csv"
            )
            row["schedule_name"] = args.sequence_name
            row["initial_plateau_steps"] = args.initial_plateau_steps
            row["transition_steps"] = args.transition_steps
            row["plateau_steps"] = args.plateau_steps
            row.update(_segment_metrics(step_rows, schedule))
            rows.append(row)
            write_results(args.output_dir, schedule, rows)


if __name__ == "__main__":
    main()
