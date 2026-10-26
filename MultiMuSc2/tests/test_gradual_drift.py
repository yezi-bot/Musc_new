import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from MultiMuSc2.datasets.gradual_drift import (
    apply_photometric_drift,
    build_multistage_drift_schedule,
    build_repeated_state_drift_schedule,
    build_segment_balanced_stream_records,
    lab_robust_feature,
)
from MultiMuSc2.examples.ablate_gradual_drift import (
    VARIANTS,
    build_lab_audit_profiles,
    build_lab_ewma_profiles,
    drift_candidate_audit_metrics,
    expert_lab_metrics,
    expert_metrics,
    lab_metrics,
    select_stream_source_records,
)
from MultiMuSc2.examples.ablate_multistage_drift import (
    _repeated_state_metrics,
)


class GradualDriftLabAuditTest(unittest.TestCase):
    def test_dual_replace_variants_use_real_two_vs_three_confirmation(self):
        confirm_three = VARIANTS[
            "photometric_multistage_cap2_dual_replace"
        ]
        confirm_two = VARIANTS[
            "photometric_multistage_cap2_dual_replace_fast_reentry"
        ]
        history_reentry = VARIANTS[
            "photometric_multistage_cap2_dual_replace_history_reentry"
        ]

        self.assertEqual(
            confirm_three["photometric_dynamic_min_low_risk_members"], 3
        )
        self.assertEqual(
            confirm_three["photometric_min_low_risk_members"], 3
        )
        self.assertEqual(
            confirm_two["photometric_dynamic_min_low_risk_members"], 2
        )
        self.assertEqual(
            confirm_two["photometric_min_low_risk_members"], 2
        )
        self.assertTrue(
            history_reentry["photometric_history_reentry_enabled"]
        )
        self.assertEqual(
            history_reentry["photometric_history_reentry_members"], 2
        )

    def test_state_ablation_variants_enable_mechanisms_incrementally(self):
        history = VARIANTS[
            "photometric_multistage_cap2_dual_replace_history_reentry"
        ]
        distribution = VARIANTS[
            "photometric_multistage_cap2_state_distribution_only"
        ]
        temporary = VARIANTS[
            "photometric_multistage_cap2_state_distribution_temporary"
        ]
        adaptive = VARIANTS[
            "photometric_multistage_cap2_adaptive_evidence"
        ]

        self.assertFalse(
            history.get("photometric_state_distribution_enabled", False)
        )
        self.assertFalse(
            history.get("photometric_temporary_state_enabled", False)
        )
        self.assertFalse(
            history.get("photometric_adaptive_candidate_evidence", False)
        )
        self.assertTrue(distribution["photometric_state_distribution_enabled"])
        self.assertFalse(distribution["photometric_temporary_state_enabled"])
        self.assertFalse(
            distribution["photometric_adaptive_candidate_evidence"]
        )
        self.assertTrue(temporary["photometric_state_distribution_enabled"])
        self.assertTrue(temporary["photometric_temporary_state_enabled"])
        self.assertFalse(temporary["photometric_adaptive_candidate_evidence"])
        self.assertTrue(adaptive["photometric_state_distribution_enabled"])
        self.assertTrue(adaptive["photometric_temporary_state_enabled"])
        self.assertTrue(adaptive["photometric_adaptive_candidate_evidence"])

        confidence = VARIANTS[
            "photometric_multistage_cap2_adaptive_evidence_bootstrap_confidence"
        ]
        self.assertEqual(confidence["cold_start_mode"], "confidence")
        self.assertEqual(confidence["cold_start_confidence_history"], 16)
        self.assertFalse(confidence["cold_start_provisional_enabled"])
        prequential = VARIANTS[
            "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential"
        ]
        self.assertEqual(
            prequential["cold_start_reference_confidence_mode"],
            "prequential",
        )
        stable_anchor = VARIANTS[
            "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential_stable_anchor"
        ]
        self.assertEqual(
            stable_anchor["cold_start_reference_confidence_mode"],
            "prequential",
        )
        self.assertTrue(
            stable_anchor["photometric_stable_lab_anchor_enabled"]
        )

    def test_expert_lab_metrics_audits_frozen_anchor_and_candidate_state(self):
        anchor = [40.0, 2.0, 1.0, 3.0, 2.0, 4.0]
        timeline = [
            {
                "step": step,
                "lab_drift_profile": {
                    "expert_id": 0,
                    "drift_confirmed": step == 1,
                    "l_ewma": 0.1 + step,
                    "ab_ewma": 0.0,
                    "stable_lab_anchor_enabled": True,
                    "anchor_frozen": True,
                    "lab_anchor_center": anchor,
                    "candidate_lab_count": step + 1,
                    "anchor_to_candidate_total_distance": 0.2 + step,
                    "anchor_to_coverage_total_distance": 0.1 + step,
                },
            }
            for step in range(2)
        ]
        schedule = [{"phase": "ramp_up"} for _ in timeline]

        metrics = expert_lab_metrics(timeline, schedule)

        self.assertEqual(metrics["stable_lab_anchor_observed_steps"], 2)
        self.assertEqual(metrics["stable_lab_anchor_frozen_steps"], 2)
        self.assertEqual(metrics["stable_lab_anchor_change_count"], 0)
        self.assertEqual(metrics["lab_candidate_state_observed_steps"], 2)
        self.assertEqual(metrics["lab_candidate_anchor_distance_max"], 1.2)

    def test_stream_source_can_add_train_good_without_reusing_test_good(self):
        test_temp_root = Path(__file__).resolve().parents[2] / ".smoke"
        test_temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_temp_root) as temp_dir:
            train_good_dir = (
                Path(temp_dir) / "bottle" / "train" / "good"
            )
            train_good_dir.mkdir(parents=True)
            for name in ("000.png", "001.png"):
                (train_good_dir / name).touch()
            test_records = [
                ("bottle", "good", "test_good.png", None),
                ("bottle", "broken", "test_bad.png", "mask.png"),
            ]

            selected, metadata = select_stream_source_records(
                records=test_records,
                data_root=temp_dir,
                category="bottle",
                normal_source="test-and-train-good",
            )

            self.assertEqual(len(selected), 4)
            self.assertEqual(metadata["available_test_good_count"], 1)
            self.assertEqual(metadata["available_train_good_count"], 2)
            self.assertEqual(metadata["available_selected_good_count"], 3)
            self.assertEqual(metadata["available_test_anomaly_count"], 1)

    def test_repeated_state_schedule_supports_d_recovery_pressure(self):
        schedule = build_repeated_state_drift_schedule(
            state_sequence=("a", "b", "d", "a", "b", "a", "b", "d")
        )

        self.assertEqual(len(schedule), 46)
        self.assertEqual(schedule[-1]["state_id"], "d")
        self.assertEqual(schedule[-1]["state_visit"], 2)
        self.assertAlmostEqual(schedule[-1]["brightness"], 0.80)
        self.assertAlmostEqual(schedule[-1]["saturation"], 0.85)

    def test_repeated_state_schedule_extends_only_final_plateau(self):
        schedule = build_repeated_state_drift_schedule(
            state_sequence=("a", "b", "d", "a", "b", "a", "b", "d"),
            final_plateau_extra_steps=2,
        )

        first_d_plateau = [
            item
            for item in schedule
            if item["segment_id"] == "state_d_visit1"
        ]
        second_d_plateau = [
            item
            for item in schedule
            if item["segment_id"] == "state_d_visit2"
        ]

        self.assertEqual(len(schedule), 48)
        self.assertEqual(len(first_d_plateau), 4)
        self.assertEqual(len(second_d_plateau), 6)
        self.assertEqual(schedule[-1]["state_id"], "d")
        self.assertEqual(schedule[-1]["state_visit"], 2)

    def test_expert_metrics_counts_budget_replacement_rejection_reason(self):
        timeline = [
            {
                "active_experts_after": [],
                "deleted_expert_ids": [],
                "admitted_expert_id": None,
                "budget_replacement": None,
                "budget_replacement_evaluation": {
                    "outcome": "insufficient_new_coverage",
                    "new_coverage_fraction": 0.0,
                },
            }
        ]

        metrics = expert_metrics(timeline, labels=np.asarray([0]))

        self.assertEqual(metrics["budget_replacement_attempt_count"], 1)
        self.assertEqual(
            metrics["budget_replacement_insufficient_coverage_count"],
            1,
        )
        self.assertEqual(
            metrics[
                "budget_replacement_evaluated_new_coverage_fraction_median"
            ],
            0.0,
        )

    def test_expert_metrics_reports_coverage_loss_replacement_audit(self):
        timeline = [
            {
                "active_experts_after": [],
                "deleted_expert_ids": [],
                "admitted_expert_id": None,
                "budget_replacement": None,
                "budget_replacement_evaluation": {
                    "outcome": "no_positive_coverage_gain",
                    "new_coverage_fraction": 1.0,
                    "coverage_removal_loss": 0.2,
                    "coverage_swap_gain": -0.1,
                },
            }
        ]

        metrics = expert_metrics(timeline, labels=np.asarray([0]))

        self.assertEqual(
            metrics[
                "budget_replacement_no_positive_coverage_gain_count"
            ],
            1,
        )
        self.assertEqual(
            metrics["budget_replacement_coverage_removal_loss_median"],
            0.2,
        )
        self.assertEqual(
            metrics["budget_replacement_coverage_swap_gain_median"],
            -0.1,
        )

    def test_expert_metrics_counts_candidate_eviction(self):
        timeline = [
            {
                "active_experts_after": [],
                "deleted_expert_ids": [],
                "admitted_expert_id": None,
                "budget_replacement": None,
                "budget_replacement_evaluation": {
                    "outcome": "candidate_evicted_minimum_coverage_loss",
                    "new_coverage_fraction": 0.5,
                    "coverage_eviction_loss": 0.0,
                },
            }
        ]

        metrics = expert_metrics(timeline, labels=np.asarray([0]))

        self.assertEqual(
            metrics["budget_replacement_candidate_evicted_count"],
            1,
        )
        self.assertEqual(metrics["budget_replacement_accepted_count"], 0)
        self.assertEqual(
            metrics[
                "budget_replacement_coverage_eviction_loss_median"
            ],
            0.0,
        )

    def test_expert_metrics_counts_photometric_state_only_from_timeline(self):
        timeline = [
            {
                "active_experts_after": [],
                "deleted_expert_ids": [],
                "admitted_expert_id": None,
                "novel_quarantine": {
                    "decision": "recorded_photometric_state_only"
                },
            }
        ]

        metrics = expert_metrics(timeline, labels=np.asarray([0]))

        self.assertEqual(metrics["photometric_state_only_count"], 1)

    def test_expert_metrics_counts_bootstrap_provisional_lifecycle(self):
        timeline = [
            {
                "step": 5,
                "active_experts_after": [],
                "deleted_expert_ids": [],
                "admitted_expert_id": None,
                "bootstrap": {"decision": "provisional_created"},
            },
            {
                "step": 6,
                "active_experts_after": [],
                "deleted_expert_ids": [],
                "admitted_expert_id": None,
                "bootstrap": {"decision": "provisional_confirming"},
            },
            {
                "step": 7,
                "active_experts_after": [],
                "deleted_expert_ids": [],
                "admitted_expert_id": None,
                "bootstrap": {"decision": "rolled_back"},
            },
            {
                "step": 8,
                "active_experts_after": [],
                "deleted_expert_ids": [],
                "admitted_expert_id": None,
                "bootstrap": {"decision": "provisional_created"},
            },
            {
                "step": 10,
                "active_experts_after": [
                    {
                        "expert_id": 0,
                        "image_id": 0,
                        "admission_route": "bootstrap",
                    }
                ],
                "deleted_expert_ids": [],
                "admitted_expert_id": 0,
                "bootstrap": {
                    "decision": "admitted",
                    "confirmation_steps": 2,
                    "selected_image_ids": [0, 1],
                },
            },
        ]
        scoring = [
            {"step": 5, "provisional_state": None},
            {"step": 6, "provisional_state": "bootstrap_provisional"},
            {"step": 7, "provisional_state": "bootstrap_provisional"},
            {"step": 8, "provisional_state": None},
            {"step": 9, "provisional_state": "bootstrap_provisional"},
        ]

        metrics = expert_metrics(
            timeline,
            labels=np.asarray([0, 0]),
            scoring=scoring,
        )

        self.assertEqual(metrics["bootstrap_provisional_created_count"], 2)
        self.assertEqual(
            metrics["bootstrap_provisional_confirming_step_count"], 1
        )
        self.assertEqual(metrics["bootstrap_provisional_rolled_back_count"], 1)
        self.assertEqual(metrics["bootstrap_provisional_admitted_count"], 1)
        self.assertEqual(metrics["bootstrap_provisional_scoring_step_count"], 3)
        self.assertEqual(
            metrics["bootstrap_provisional_confirmation_delay_mean"], 2.0
        )

    def test_expert_metrics_counts_continuous_bootstrap_confidence(self):
        bootstrap_expert = {
            "expert_id": 0,
            "image_id": 1,
            "admission_route": "bootstrap_confidence",
        }
        timeline = [
            {
                "step": 0,
                "active_experts_after": [bootstrap_expert],
                "expert_slots_after": [bootstrap_expert],
                "deleted_expert_ids": [],
                "admitted_expert_id": 0,
                "bootstrap": {
                    "decision": "confidence_state_created",
                    "state_confidence": 0.0,
                    "normality_confidence": 0.0,
                    "reference_confidence": 0.0,
                    "selected_image_ids": [0],
                },
            },
            {
                "step": 1,
                "active_experts_after": [bootstrap_expert],
                "expert_slots_after": [bootstrap_expert],
                "deleted_expert_ids": [],
                "admitted_expert_id": None,
                "bootstrap": {
                    "decision": "confidence_state_updated",
                    "state_confidence": 0.5,
                    "normality_confidence": 0.8,
                    "reference_confidence": 0.4,
                    "prequential_support": 0.8,
                },
            },
        ]
        scoring = [
            {"step": 0, "active_expert_ids": [], "expert_image_ids": []},
            {
                "step": 1,
                "active_expert_ids": [0],
                "expert_image_ids": [1],
                "active_expert_scoring_weights": [0.25],
            },
        ]

        metrics = expert_metrics(
            timeline,
            labels=np.asarray([0, 1]),
            scoring=scoring,
        )

        self.assertEqual(metrics["bootstrap_confidence_created_count"], 1)
        self.assertEqual(metrics["bootstrap_confidence_update_count"], 1)
        self.assertEqual(metrics["bootstrap_confidence_first_positive_step"], 1)
        self.assertEqual(metrics["bootstrap_confidence_final"], 0.5)
        self.assertEqual(metrics["bootstrap_confidence_max"], 0.5)
        self.assertEqual(
            metrics["bootstrap_reference_confidence_first_positive_step"], 1
        )
        self.assertEqual(
            metrics["bootstrap_reference_confidence_final"], 0.4
        )
        self.assertEqual(
            metrics["bootstrap_normality_confidence_final"], 0.8
        )
        self.assertEqual(
            metrics["bootstrap_prequential_support_mean"], 0.8
        )
        self.assertEqual(
            metrics["bootstrap_confidence_anomaly_exposure"], 0.25
        )
        self.assertEqual(
            metrics["bootstrap_confidence_member_anomaly_exposure"], 0.25
        )
        self.assertEqual(
            metrics[
                "bootstrap_confidence_member_anomaly_scoring_step_count"
            ],
            1,
        )
        self.assertEqual(
            metrics["bootstrap_reference_first_weighted_scoring_step"], 1
        )

    def test_expert_metrics_uses_state_lifecycle_events_as_source_of_truth(self):
        timeline = [
            {
                "active_experts_after": [],
                "deleted_expert_ids": [],
                "admitted_expert_id": None,
                "novel_quarantine": {
                    "decision": "matured_photometric_state_only"
                },
                "historical_states": [
                    {"state_id": 0},
                    {"state_id": 1},
                ],
                "state_lifecycle_events": [
                    {"event_type": "create", "status_after": "temporary"},
                    {"event_type": "match", "status_after": "temporary"},
                    {"event_type": "update", "status_after": "stable"},
                    {"event_type": "merge", "status_after": "stable"},
                    {"event_type": "mature", "status_after": "stable"},
                    {"event_type": "evict", "status_after": "temporary"},
                ],
            }
        ]

        metrics = expert_metrics(timeline, labels=np.asarray([0]))

        self.assertEqual(metrics["photometric_state_only_count"], 1)
        self.assertEqual(metrics["historical_state_created_count"], 1)
        self.assertEqual(metrics["historical_state_matched_count"], 1)
        self.assertEqual(metrics["historical_state_updated_count"], 1)
        self.assertEqual(metrics["historical_state_merged_count"], 1)
        self.assertEqual(metrics["historical_state_matured_count"], 1)
        self.assertEqual(metrics["historical_state_evicted_count"], 1)
        self.assertEqual(metrics["temporary_state_created_count"], 1)
        self.assertEqual(metrics["max_historical_state_count"], 2)
        self.assertEqual(metrics["final_historical_state_count"], 2)

    def test_expert_metrics_counts_history_reentry_outcomes(self):
        decisions = (
            "history_shared_coverage",
            "history_restored_free_slot",
            "history_restored_by_replacement",
            "history_materialized_free_slot",
            "history_materialized_by_replacement",
            "history_reentry_rejected",
        )
        timeline = [
            {
                "active_experts_after": [],
                "deleted_expert_ids": [],
                "admitted_expert_id": None,
                "history_reentry": {"decision": decision},
                "historical_states": [{"state_id": 0}],
            }
            for decision in decisions
        ]

        metrics = expert_metrics(timeline, labels=np.zeros(len(timeline)))

        self.assertEqual(metrics["history_shared_coverage_count"], 1)
        self.assertEqual(metrics["history_restored_free_slot_count"], 1)
        self.assertEqual(
            metrics["history_restored_by_replacement_count"], 1
        )
        self.assertEqual(metrics["history_reentry_rejected_count"], 1)
        self.assertEqual(metrics["history_materialized_free_slot_count"], 1)
        self.assertEqual(
            metrics["history_materialized_by_replacement_count"], 1
        )
        self.assertEqual(metrics["max_historical_state_count"], 1)

    def test_expert_metrics_counts_polluted_expert_scoring_exposure(self):
        clean_expert = {
            "expert_id": 0,
            "image_id": 0,
            "admission_route": "bootstrap",
        }
        polluted_expert = {
            "expert_id": 1,
            "image_id": 1,
            "admission_route": "photometric_drift",
        }
        timeline = [
            {
                "step": 0,
                "active_experts_after": [clean_expert],
                "expert_slots_after": [clean_expert],
                "deleted_expert_ids": [],
                "admitted_expert_id": 0,
            },
            {
                "step": 1,
                "active_experts_after": [clean_expert, polluted_expert],
                "expert_slots_after": [clean_expert, polluted_expert],
                "deleted_expert_ids": [],
                "admitted_expert_id": 1,
            },
            *[
                {
                    "step": step,
                    "active_experts_after": [
                        clean_expert,
                        polluted_expert,
                    ],
                    "expert_slots_after": [
                        clean_expert,
                        polluted_expert,
                    ],
                    "deleted_expert_ids": [],
                    "admitted_expert_id": None,
                }
                for step in range(2, 5)
            ],
        ]
        scoring = [
            {"step": 0, "active_expert_ids": []},
            {"step": 1, "active_expert_ids": [0, 1]},
            {"step": 2, "active_expert_ids": [0, 1]},
            {"step": 3, "active_expert_ids": [1]},
            {"step": 3, "active_expert_ids": [0, 1]},
            {"step": 4, "active_expert_ids": [0]},
        ]

        metrics = expert_metrics(
            timeline,
            labels=np.asarray([0, 1, 0, 1, 0]),
            scoring=scoring,
        )

        self.assertEqual(metrics["polluted_expert_admissions"], 1)
        self.assertEqual(metrics["polluted_expert_scoring_step_count"], 2)
        self.assertEqual(
            metrics["polluted_expert_future_anomaly_scoring_count"], 1
        )

    def test_multistage_schedule_has_distinct_absolute_states(self):
        schedule = build_multistage_drift_schedule()

        self.assertEqual(len(schedule), 28)
        self.assertEqual(schedule[0]["segment_id"], "state_a_initial")
        self.assertEqual(schedule[-1]["segment_id"], "state_a_recovery")
        self.assertEqual(schedule[-1]["brightness"], 1.0)
        self.assertEqual(schedule[-1]["hue"], 0.0)
        plateau_ids = {
            item["segment_id"]
            for item in schedule
            if item["segment_kind"] == "plateau"
        }
        self.assertEqual(
            plateau_ids,
            {
                "state_a_initial",
                "state_b",
                "state_c",
                "state_d",
                "state_a_recovery",
            },
        )
        transition = [
            item
            for item in schedule
            if item["segment_id"] == "transition_a_to_b"
        ]
        self.assertEqual(len(transition), 2)
        self.assertLess(
            transition[0]["brightness"],
            transition[1]["brightness"],
        )

    def test_repeated_state_schedule_keeps_visits_distinct(self):
        schedule = build_repeated_state_drift_schedule()

        self.assertEqual(len(schedule), 46)
        plateau_ids = []
        for item in schedule:
            if (
                item["segment_kind"] == "plateau"
                and item["segment_id"] not in plateau_ids
            ):
                plateau_ids.append(item["segment_id"])
        self.assertEqual(
            plateau_ids,
            [
                "state_a_visit1",
                "state_b_visit1",
                "state_c_visit1",
                "state_a_visit2",
                "state_b_visit2",
                "state_a_visit3",
                "state_b_visit3",
                "state_c_visit2",
            ],
        )
        self.assertFalse(schedule[0]["is_revisit"])
        self.assertTrue(schedule[-1]["is_revisit"])
        self.assertEqual(schedule[-1]["state_id"], "c")
        self.assertEqual(schedule[-1]["state_visit"], 2)

    def test_repeated_state_metrics_measure_retakeover_and_budget(self):
        schedule = build_repeated_state_drift_schedule()
        rows = []
        for index, item in enumerate(schedule):
            is_revisit = bool(item.get("is_revisit", False))
            score = 0.12 if is_revisit else 0.10
            current_supports = {}
            active_ids = [0]
            if item["segment_id"] == "state_a_visit2":
                current_supports = {"0": 0.8}
                active_ids = [0]
            replacement = None
            if (
                item["segment_id"] == "state_a_visit2"
                and (
                    index == 0
                    or schedule[index - 1]["segment_id"]
                    != item["segment_id"]
                )
            ):
                replacement = {
                    "expert_id": 1,
                    "victim_coverage": {
                        "unique_fraction": 0.1,
                        "coverage_fraction": 0.4,
                    },
                    "victim_lab_uniqueness": {
                        "photometrically_redundant": True,
                    },
                    "candidate_admission_evidence": "lab",
                    "new_coverage_fraction": 0.2,
                }
            rows.append(
                {
                    "step": str(index),
                    "is_anomaly": "0",
                    "raw_image_score": str(score),
                    "support_threshold": "0.5",
                    "current_expert_supports": json.dumps(
                        current_supports
                    ),
                    "active_expert_ids_before": json.dumps(active_ids),
                    "expert_slot_ids_before": json.dumps(active_ids),
                    "expert_lifecycle_events": "[]",
                    "admitted_expert_id": "",
                    "budget_replacement_event": json.dumps(replacement),
                    "expert_slot_count_after": "2",
                }
            )

        metrics = _repeated_state_metrics(rows, schedule, committee_cap=2)

        self.assertEqual(metrics["revisit_segment_count"], 5)
        self.assertTrue(metrics["budget_respected"])
        self.assertEqual(metrics["replacement_event_count"], 1)
        self.assertEqual(metrics["revisit_replacement_event_count"], 1)
        self.assertEqual(
            metrics[
                "state_a_visit2_first_existing_support_delay_images"
            ],
            1,
        )
        self.assertAlmostEqual(
            metrics["state_a_visit2_normal_score_delta_vs_first_visit"],
            0.02,
        )
        self.assertIn("state_c_visit2_arrival_takeover_source", metrics)

    def test_repeated_state_metrics_distinguish_deleted_c_from_rebuild(self):
        schedule = build_repeated_state_drift_schedule()
        c_revisit_start = next(
            index
            for index, item in enumerate(schedule)
            if item["state_id"] == "c" and int(item["state_visit"]) == 2
        )
        rows = []
        for index, item in enumerate(schedule):
            admitted_expert_id = None
            if item["segment_id"] == "state_c_visit1" and not any(
                row["admitted_expert_id"] == "1" for row in rows
            ):
                admitted_expert_id = 1
            if (
                item["segment_id"]
                == "transition_b_visit3_to_c_visit2"
                and not any(
                    row["admitted_expert_id"] == "2" for row in rows
                )
            ):
                admitted_expert_id = 2
            is_c_revisit = (
                item["state_id"] == "c"
                and int(item["state_visit"]) == 2
            )
            rebuilt_is_available = index > c_revisit_start
            active_ids = [0, 2] if rebuilt_is_available else [0]
            current_supports = (
                {"2": 0.8} if rebuilt_is_available else {}
            )
            rows.append(
                {
                    "step": str(index),
                    "is_anomaly": "0",
                    "raw_image_score": "0.1",
                    "support_threshold": "0.5",
                    "current_expert_supports": json.dumps(current_supports),
                    "active_expert_ids_before": json.dumps(active_ids),
                    "expert_slot_ids_before": (
                        json.dumps(active_ids)
                        if is_c_revisit else "[0, 1]"
                    ),
                    "expert_lifecycle_events": "[]",
                    "admitted_expert_id": (
                        "" if admitted_expert_id is None
                        else str(admitted_expert_id)
                    ),
                    "budget_replacement_event": "",
                    "expert_slot_count_after": "2",
                }
            )

        metrics = _repeated_state_metrics(rows, schedule, committee_cap=2)

        self.assertEqual(metrics["state_c_visit2_original_expert_ids"], "[1]")
        self.assertFalse(
            metrics["state_c_visit2_original_expert_in_slots_at_arrival"]
        )
        self.assertEqual(
            metrics["state_c_visit2_rebuilt_expert_admission_delay_images"],
            2,
        )
        self.assertEqual(
            metrics["state_c_visit2_rebuilt_expert_support_delay_images"],
            2,
        )
        self.assertEqual(
            metrics["state_c_visit2_arrival_takeover_source"],
            "rebuilt_expert_support",
        )

    def test_segment_balanced_records_preserve_each_segment_mix(self):
        records = [
            ("bottle", "good", f"good_{index}.png", None)
            for index in range(12)
        ]
        records += [
            ("bottle", "broken", f"bad_{index}.png", None)
            for index in range(6)
        ]

        stream = build_segment_balanced_stream_records(
            records=records,
            segment_sizes=[4, 2, 4],
            anomaly_ratio=(1.0 / 3.0),
            seed=0,
        )

        self.assertEqual(len(stream), 10)
        offset = 0
        for size in (4, 2, 4):
            segment = stream[offset:offset + size]
            labels = [record[1] != "good" for record in segment]
            self.assertGreater(sum(labels), 0)
            self.assertLess(sum(labels), size)
            offset += size

    def test_segment_balanced_records_can_reuse_for_long_stream(self):
        records = [
            ("bottle", "good", f"good_{index}.png", None)
            for index in range(4)
        ]
        records += [
            ("bottle", "broken", f"bad_{index}.png", None)
            for index in range(2)
        ]

        stream = build_segment_balanced_stream_records(
            records=records,
            segment_sizes=[4, 4, 4],
            anomaly_ratio=(1.0 / 3.0),
            seed=0,
            allow_reuse=True,
        )

        self.assertEqual(len(stream), 12)
        self.assertLess(
            len({record[2] for record in stream}),
            len(stream),
        )
        for offset in range(0, len(stream), 4):
            labels = [
                record[1] != "good" for record in stream[offset:offset + 4]
            ]
            self.assertEqual(sum(labels), 1)

    def test_no_drift_level_stays_zero_drift_during_nonzero_schedule(self):
        timeline = [
            {
                "drift_candidate_audit": {
                    "decision": "audited",
                    "selected_image_ids": [0, 1, 2],
                }
            }
        ]
        schedule = [
            {"severity": 0.5},
            {"severity": 0.75},
            {"severity": 1.0},
        ]

        metrics = drift_candidate_audit_metrics(
            timeline=timeline,
            schedule=schedule,
            labels=np.asarray([0, 0, 0]),
            drift_level="no_drift",
        )

        self.assertEqual(metrics["drift_pool_audit_true_drift_count"], 0)
        self.assertEqual(metrics["drift_pool_audit_zero_drift_count"], 1)

    def test_brightness_increases_lab_l_feature(self):
        image = Image.new("RGB", (8, 8), (96, 96, 96))
        baseline = lab_robust_feature(image)
        brighter = lab_robust_feature(
            apply_photometric_drift(
                image,
                {
                    "brightness": 1.25,
                    "contrast": 1.0,
                    "saturation": 1.0,
                    "hue": 0.0,
                },
            )
        )

        self.assertGreater(brighter[0], baseline[0])
        self.assertLess(abs(brighter[2] - baseline[2]), 1e-3)
        self.assertLess(abs(brighter[4] - baseline[4]), 1e-3)

    def test_lab_audit_starts_after_first_expert_admission(self):
        lab_features = np.asarray(
            [
                [40.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [41.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [42.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [45.0, 2.0, 1.0, 3.0, 2.0, 4.0],
            ]
        )
        timeline = [
            {
                "step": 0,
                "admitted_expert_id": None,
                "active_experts_after": [],
            },
            {
                "step": 1,
                "admitted_expert_id": None,
                "active_experts_after": [],
            },
            {
                "step": 2,
                "admitted_expert_id": 0,
                "active_experts_after": [
                    {
                        "expert_id": 0,
                        "image_id": 1,
                    }
                ],
            },
            {
                "step": 3,
                "admitted_expert_id": None,
                "active_experts_after": [
                    {
                        "expert_id": 0,
                        "image_id": 1,
                    }
                ],
            },
        ]

        profiles = build_lab_audit_profiles(
            lab_features,
            timeline,
        )

        self.assertIsNone(profiles[2]["total_distance"])
        self.assertEqual(profiles[3]["reference_image_id"], 1)
        self.assertAlmostEqual(profiles[3]["l_distance"], 0.04)

    def test_clean_metrics_use_labels_only_for_offline_subset(self):
        profiles = [
            {
                "reference_expert_id": 0,
                "reference_image_id": 0,
                "reference_admission_step": 0,
                "l_distance": 0.1,
                "ab_distance": 0.2,
                "total_distance": 0.3,
            },
            {
                "reference_expert_id": 0,
                "reference_image_id": 0,
                "reference_admission_step": 0,
                "l_distance": 0.5,
                "ab_distance": 0.6,
                "total_distance": 0.7,
            },
        ]
        schedule = [
            {"phase": "plateau"},
            {"phase": "plateau"},
        ]

        metrics = lab_metrics(
            profiles,
            schedule,
            labels=np.asarray([0, 1]),
        )

        self.assertAlmostEqual(
            metrics["plateau_lab_total_distance_median"],
            0.5,
        )
        self.assertAlmostEqual(
            metrics[
                "plateau_lab_total_distance_clean_median"
            ],
            0.3,
        )

    def test_lab_ewma_uses_historical_baseline_and_confirmation(self):
        lab_features = np.asarray(
            [
                [40.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [41.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [42.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [60.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [70.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [41.0, 2.0, 1.0, 3.0, 2.0, 4.0],
            ]
        )
        reference = {
            "expert_id": 0,
            "image_id": 0,
            "admission_step": 2,
        }

        profiles = build_lab_ewma_profiles(
            lab_features,
            reference=reference,
            alpha=1.0,
            threshold_k=3.0,
            baseline_window=8,
            confirm_steps=2,
        )

        self.assertIsNone(profiles[2])
        self.assertEqual(profiles[3]["baseline_start"], 0)
        self.assertEqual(profiles[3]["baseline_end"], 2)
        self.assertEqual(profiles[3]["baseline_count"], 3)
        self.assertTrue(profiles[3]["l_hit"])
        self.assertFalse(profiles[3]["l_confirmed"])
        self.assertTrue(profiles[4]["l_confirmed"])
        self.assertTrue(profiles[4]["new_confirmation"])
        self.assertFalse(profiles[5]["l_hit"])
        self.assertFalse(profiles[5]["l_confirmed"])
        self.assertFalse(profiles[5]["new_confirmation"])


if __name__ == "__main__":
    unittest.main()
