import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from datasets.gradual_drift import (
    build_multistage_drift_schedule,
    build_repeated_state_drift_schedule,
)
from examples.ablate_gradual_drift import (
    VARIANTS,
    load_config,
    run_variant,
    safe_auroc,
)


PRESSURE_VARIANTS = (
    "photometric_multistage_cap2_reject",
    "photometric_multistage_cap2_replace",
    "photometric_multistage_cap2_dual_replace",
    "photometric_multistage_cap2_dual_replace_fast_reentry",
    "photometric_multistage_cap2_dual_replace_history_reentry",
    "photometric_multistage_cap2_state_distribution_only",
    "photometric_multistage_cap2_state_distribution_temporary",
    "photometric_multistage_cap2_adaptive_evidence",
    "photometric_multistage_cap2_adaptive_evidence_bootstrap_provisional",
    "photometric_multistage_cap2_adaptive_evidence_bootstrap_confidence",
    "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential",
    "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential_stable_anchor",
    "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential_denstream",
    "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential_denstream_fitness",
    "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential_denstream_fitness_coreset",
    "priority4_min_loss_coreset",
    "priority4_joint_coreset",
    "priority4_patch_coreset",
    "priority4_patch_victim",
    "priority4_patch_kcenter_victim",
    "priority5_task_aware_virtual_swap",
    "priority5_task_aware_soft_reliability",
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


def _json_value(value, fallback):
    if value in (None, ""):
        return fallback
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def _optional_int(value):
    if value in (None, "", "None"):
        return None
    return int(value)


def _optional_float(value):
    if value in (None, "", "None"):
        return None
    return float(value)


def _has_existing_expert_support(step_row):
    threshold = _optional_float(step_row.get("support_threshold"))
    if threshold is None:
        return False
    active_ids = {
        int(expert_id)
        for expert_id in _json_value(
            step_row.get("active_expert_ids_before"), []
        )
    }
    supports = _json_value(
        step_row.get("current_expert_supports"), {}
    )
    return any(
        int(expert_id) in active_ids
        and support is not None
        and float(support) >= threshold
        for expert_id, support in supports.items()
    )


def _supported_expert_ids(step_row):
    threshold = _optional_float(step_row.get("support_threshold"))
    if threshold is None:
        return set()
    active_ids = {
        int(expert_id)
        for expert_id in _json_value(
            step_row.get("active_expert_ids_before"), []
        )
    }
    supports = _json_value(
        step_row.get("current_expert_supports"), {}
    )
    return {
        int(expert_id)
        for expert_id, support in supports.items()
        if int(expert_id) in active_ids
        and support is not None
        and float(support) >= threshold
    }


def _first_matching_delay(step_rows, predicate, after_scoring=False):
    for offset, step_row in enumerate(step_rows):
        if predicate(step_row):
            return offset + (2 if after_scoring else 1)
    return None


def _first_event_delay(step_rows, event_name):
    for offset, step_row in enumerate(step_rows):
        if event_name == "existing_support":
            found = _has_existing_expert_support(step_row)
        elif event_name == "admission":
            found = _optional_int(step_row.get("admitted_expert_id")) is not None
        elif event_name == "reactivation":
            found = any(
                bool(event.get("reactivated_this_step", False))
                for event in _json_value(
                    step_row.get("expert_lifecycle_events"), []
                )
            )
        else:
            raise ValueError(f"unknown event_name: {event_name}")
        if found:
            return offset + (1 if event_name == "existing_support" else 2)
    return None


def _repeated_state_metrics(step_rows, schedule, committee_cap):
    metrics = {}
    plateau_segments = []
    for item in schedule:
        if item["segment_kind"] != "plateau":
            continue
        if item["segment_id"] not in plateau_segments:
            plateau_segments.append(item["segment_id"])

    first_segment_by_state = {}
    revisit_segments = []
    for segment_id in plateau_segments:
        item = next(
            schedule_item
            for schedule_item in schedule
            if schedule_item["segment_id"] == segment_id
        )
        state_id = item["state_id"]
        if state_id not in first_segment_by_state:
            first_segment_by_state[state_id] = segment_id
        elif item.get("is_revisit", False):
            revisit_segments.append(segment_id)

    for segment_id in revisit_segments:
        indices = [
            index
            for index, item in enumerate(schedule)
            if item["segment_id"] == segment_id
        ]
        segment_rows = [step_rows[index] for index in indices]
        segment_item = schedule[indices[0]]
        first_segment_id = first_segment_by_state[segment_item["state_id"]]
        first_indices = [
            index
            for index, item in enumerate(schedule)
            if item["segment_id"] == first_segment_id
        ]
        first_expert_scored_normal_scores = [
            float(step_rows[index]["raw_image_score"])
            for index in first_indices
            if int(step_rows[index]["is_anomaly"]) == 0
            and bool(
                _json_value(
                    step_rows[index].get("active_expert_ids_before"), []
                )
            )
        ]
        revisit_normal_scores = [
            float(row["raw_image_score"])
            for row in segment_rows
            if int(row["is_anomaly"]) == 0
        ]
        first_normal_mean = (
            float(np.mean(first_expert_scored_normal_scores))
            if first_expert_scored_normal_scores
            else None
        )
        revisit_normal_mean = (
            float(np.mean(revisit_normal_scores))
            if revisit_normal_scores
            else None
        )
        prefix = segment_id
        prefix_rows = segment_rows[:3]
        metrics[f"{prefix}_first_3_raw_scores"] = json.dumps(
            [float(row["raw_image_score"]) for row in prefix_rows]
        )
        metrics[f"{prefix}_first_3_anomaly_flags"] = json.dumps(
            [int(row["is_anomaly"]) for row in prefix_rows]
        )
        for position in range(3):
            row = prefix_rows[position] if position < len(prefix_rows) else None
            metrics[f"{prefix}_first_{position + 1}_raw_score"] = (
                float(row["raw_image_score"]) if row is not None else None
            )
            metrics[f"{prefix}_first_{position + 1}_is_anomaly"] = (
                int(row["is_anomaly"]) if row is not None else None
            )

        normal_rows = [
            row for row in segment_rows if int(row["is_anomaly"]) == 0
        ]
        supported_normal_count = sum(
            _has_existing_expert_support(row) for row in normal_rows
        )
        existing_delay = _first_event_delay(
            segment_rows, "existing_support"
        )
        reactivation_delay = _first_event_delay(
            segment_rows, "reactivation"
        )
        admission_delay = _first_event_delay(segment_rows, "admission")
        takeover_delays = [
            delay
            for delay in (
                existing_delay,
                reactivation_delay,
                admission_delay,
            )
            if delay is not None
        ]
        metrics[f"{prefix}_first_existing_support_delay_images"] = (
            existing_delay
        )
        metrics[f"{prefix}_first_reactivation_delay_images"] = (
            reactivation_delay
        )
        metrics[f"{prefix}_first_admission_delay_images"] = admission_delay
        metrics[f"{prefix}_autonomous_takeover_delay_images"] = (
            min(takeover_delays) if takeover_delays else None
        )
        metrics[f"{prefix}_normal_existing_support_fraction"] = (
            supported_normal_count / len(normal_rows)
            if normal_rows
            else None
        )
        metrics[f"{prefix}_first_visit_normal_score_mean"] = (
            first_normal_mean
        )
        metrics[f"{prefix}_first_visit_expert_scored_normal_count"] = len(
            first_expert_scored_normal_scores
        )
        metrics[f"{prefix}_revisit_normal_score_mean"] = revisit_normal_mean
        metrics[f"{prefix}_normal_score_delta_vs_first_visit"] = (
            revisit_normal_mean - first_normal_mean
            if revisit_normal_mean is not None
            and first_normal_mean is not None
            else None
        )
        metrics[f"{prefix}_normal_score_ratio_vs_first_visit"] = (
            revisit_normal_mean / first_normal_mean
            if revisit_normal_mean is not None
            and first_normal_mean not in (None, 0.0)
            else None
        )

        state_id = segment_item["state_id"]
        state_visit = int(segment_item["state_visit"])
        arrival_indices = [
            index
            for index, item in enumerate(schedule)
            if item["state_id"] == state_id
            and int(item.get("state_visit", 1)) == state_visit
        ]
        arrival_rows = [step_rows[index] for index in arrival_indices]
        original_expert_ids = {
            _optional_int(step_rows[index].get("admitted_expert_id"))
            for index, item in enumerate(schedule)
            if item["state_id"] == state_id
            and int(item.get("state_visit", 1)) == 1
            and index < arrival_indices[0]
        }
        original_expert_ids.discard(None)
        slot_ids_at_arrival = {
            int(expert_id)
            for expert_id in _json_value(
                arrival_rows[0].get("expert_slot_ids_before"), []
            )
        }
        active_ids_at_arrival = {
            int(expert_id)
            for expert_id in _json_value(
                arrival_rows[0].get("active_expert_ids_before"), []
            )
        }
        original_support_delay = _first_matching_delay(
            arrival_rows,
            lambda row: bool(
                _supported_expert_ids(row) & original_expert_ids
            ),
        )
        original_reactivation_delay = _first_matching_delay(
            arrival_rows,
            lambda row: any(
                int(event.get("expert_id", -1)) in original_expert_ids
                and bool(event.get("reactivated_this_step", False))
                for event in _json_value(
                    row.get("expert_lifecycle_events"), []
                )
            ),
            after_scoring=True,
        )
        rebuilt_admission_delay = _first_matching_delay(
            arrival_rows,
            lambda row: (
                _optional_int(row.get("admitted_expert_id")) is not None
                and _optional_int(row.get("admitted_expert_id"))
                not in original_expert_ids
            ),
            after_scoring=True,
        )
        rebuilt_expert_ids = {
            _optional_int(row.get("admitted_expert_id"))
            for row in arrival_rows
            if _optional_int(row.get("admitted_expert_id")) is not None
            and _optional_int(row.get("admitted_expert_id"))
            not in original_expert_ids
        }
        rebuilt_support_delay = _first_matching_delay(
            arrival_rows,
            lambda row: bool(
                _supported_expert_ids(row) & rebuilt_expert_ids
            ),
        )
        any_support_delay = _first_matching_delay(
            arrival_rows,
            lambda row: bool(_supported_expert_ids(row)),
        )
        other_support_delay = _first_matching_delay(
            arrival_rows,
            lambda row: bool(
                _supported_expert_ids(row)
                - original_expert_ids
                - rebuilt_expert_ids
            ),
        )
        support_sources = [
            (original_support_delay, "original_expert_support"),
            (rebuilt_support_delay, "rebuilt_expert_support"),
            (other_support_delay, "other_expert_support"),
        ]
        support_sources = [
            item for item in support_sources if item[0] is not None
        ]
        if support_sources:
            _, takeover_source = min(support_sources, key=lambda item: item[0])
        elif original_reactivation_delay is not None:
            takeover_source = "original_expert_reactivated_unverified"
        elif rebuilt_admission_delay is not None:
            takeover_source = "rebuilt_expert_admitted_unverified"
        else:
            takeover_source = "none"
        metrics[f"{prefix}_arrival_first_3_raw_scores"] = json.dumps(
            [
                float(row["raw_image_score"])
                for row in arrival_rows[:3]
            ]
        )
        metrics[f"{prefix}_arrival_first_3_anomaly_flags"] = json.dumps(
            [int(row["is_anomaly"]) for row in arrival_rows[:3]]
        )
        metrics[f"{prefix}_original_expert_ids"] = json.dumps(
            sorted(original_expert_ids)
        )
        metrics[f"{prefix}_original_expert_in_slots_at_arrival"] = bool(
            original_expert_ids & slot_ids_at_arrival
        )
        metrics[f"{prefix}_original_expert_active_at_arrival"] = bool(
            original_expert_ids & active_ids_at_arrival
        )
        metrics[f"{prefix}_original_expert_support_delay_images"] = (
            original_support_delay
        )
        metrics[f"{prefix}_original_expert_reactivation_delay_images"] = (
            original_reactivation_delay
        )
        metrics[f"{prefix}_rebuilt_expert_admission_delay_images"] = (
            rebuilt_admission_delay
        )
        metrics[f"{prefix}_rebuilt_expert_support_delay_images"] = (
            rebuilt_support_delay
        )
        metrics[f"{prefix}_arrival_any_support_delay_images"] = (
            any_support_delay
        )
        metrics[f"{prefix}_arrival_other_support_delay_images"] = (
            other_support_delay
        )
        metrics[f"{prefix}_arrival_takeover_source"] = takeover_source

    replacement_events = []
    admission_counts_by_state = {}
    for index, row in enumerate(step_rows):
        state_id = schedule[index]["state_id"]
        admitted_expert_id = _optional_int(row.get("admitted_expert_id"))
        if admitted_expert_id is not None:
            admission_counts_by_state[state_id] = (
                admission_counts_by_state.get(state_id, 0) + 1
            )
        replacement = _json_value(
            row.get("budget_replacement_event"), None
        )
        if not replacement:
            continue
        victim_coverage = replacement.get("victim_coverage") or {}
        victim_lab = replacement.get("victim_lab_uniqueness") or {}
        replacement_events.append(
            {
                "step": int(row["step"]),
                "segment_id": schedule[index]["segment_id"],
                "state_id": state_id,
                "state_visit": schedule[index].get("state_visit"),
                "is_revisit": bool(schedule[index].get("is_revisit", False)),
                "victim_expert_id": int(replacement["expert_id"]),
                "victim_unique_fraction": victim_coverage.get(
                    "unique_fraction"
                ),
                "victim_coverage_fraction": victim_coverage.get(
                    "coverage_fraction"
                ),
                "victim_photometrically_redundant": victim_lab.get(
                    "photometrically_redundant"
                ),
                "candidate_admission_evidence": replacement.get(
                    "candidate_admission_evidence"
                ),
                "new_coverage_fraction": replacement.get(
                    "new_coverage_fraction"
                ),
            }
        )

    replacement_steps = [event["step"] for event in replacement_events]
    replacement_victims = [
        event["victim_expert_id"] for event in replacement_events
    ]
    max_slots = max(int(row["expert_slot_count_after"]) for row in step_rows)
    metrics.update(
        {
            "revisit_segment_count": len(revisit_segments),
            "revisit_segment_ids": json.dumps(revisit_segments),
            "budget_respected": bool(max_slots <= int(committee_cap)),
            "observed_max_expert_slots": max_slots,
            "replacement_event_count": len(replacement_events),
            "revisit_replacement_event_count": sum(
                bool(event["is_revisit"]) for event in replacement_events
            ),
            "replacement_events": json.dumps(
                replacement_events, ensure_ascii=False
            ),
            "repeated_replacement_victim_count": (
                len(replacement_victims) - len(set(replacement_victims))
            ),
            "minimum_replacement_gap_steps": (
                min(
                    second - first
                    for first, second in zip(
                        replacement_steps, replacement_steps[1:]
                    )
                )
                if len(replacement_steps) >= 2
                else None
            ),
            "admission_counts_by_state": json.dumps(
                admission_counts_by_state, ensure_ascii=False
            ),
        }
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
    parser.add_argument(
        "--normal-source",
        choices=("test-good", "train-good", "test-and-train-good"),
        default="test-good",
        help=(
            "Normal-image pool. Anomalies always come from test. Use "
            "test-and-train-good only for an explicitly labelled mechanism "
            "stress test, not for the no-precollection primary result."
        ),
    )
    parser.add_argument("--sequence-name", default="abcd_recovery")
    parser.add_argument(
        "--state-sequence",
        nargs="+",
        choices=("a", "b", "c", "d"),
        default=None,
        help=(
            "Optional repeated-state sequence, for example: "
            "--state-sequence a b d a b a b d"
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--anomaly-ratio", type=float, default=(1.0 / 3.0))
    parser.add_argument(
        "--allow-record-reuse",
        action="store_true",
        help=(
            "Deterministically reuse source images when a long synthetic "
            "stream exceeds the available dataset records."
        ),
    )
    parser.add_argument("--initial-plateau-steps", type=int, default=4)
    parser.add_argument("--transition-steps", type=int, default=2)
    parser.add_argument("--plateau-steps", type=int, default=4)
    parser.add_argument(
        "--final-plateau-extra-steps",
        type=int,
        default=0,
        help="Extra observations appended only to the final plateau.",
    )
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
    if args.state_sequence is None:
        schedule = build_multistage_drift_schedule(
            initial_plateau_steps=args.initial_plateau_steps,
            transition_steps=args.transition_steps,
            plateau_steps=args.plateau_steps,
        )
    else:
        schedule = build_repeated_state_drift_schedule(
            state_sequence=tuple(args.state_sequence),
            initial_plateau_steps=args.initial_plateau_steps,
            transition_steps=args.transition_steps,
            plateau_steps=args.plateau_steps,
            final_plateau_extra_steps=args.final_plateau_extra_steps,
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
            row["final_plateau_extra_steps"] = (
                args.final_plateau_extra_steps
            )
            row.update(_segment_metrics(step_rows, schedule))
            row.update(
                _repeated_state_metrics(
                    step_rows,
                    schedule,
                    committee_cap=row["committee_cap"],
                )
            )
            rows.append(row)
            write_results(args.output_dir, schedule, rows)


if __name__ == "__main__":
    main()
