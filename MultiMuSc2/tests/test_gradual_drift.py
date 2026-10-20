import unittest

import numpy as np
from PIL import Image

from MultiMuSc2.datasets.gradual_drift import (
    apply_photometric_drift,
    lab_robust_feature,
)
from MultiMuSc2.examples.ablate_gradual_drift import (
    build_lab_audit_profiles,
    build_lab_ewma_profiles,
    drift_candidate_audit_metrics,
    lab_metrics,
)


class GradualDriftLabAuditTest(unittest.TestCase):
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
