import unittest

import torch

from MultiMuSc2.models.modules._DYNAMIC_PIPELINE import (
    _controlled_contamination_patch_ids,
    _hierarchical_dino_score,
    _image_msm_score,
    _positioned_memory_scores,
    _strict_history_fallback,
    build_dynamic_committee_timeline,
    score_dynamic_msm2_layer,
)


class ControlledContaminationPatchSelectionTest(unittest.TestCase):
    def test_patch_mode_selects_only_largest_distances(self):
        selected = _controlled_contamination_patch_ids(
            torch.arange(6, dtype=torch.float32),
            patch_count=4,
            contamination_mode="patch",
        )
        self.assertEqual(selected, [5, 4, 3, 2])

    def test_mixed_mode_alternates_large_and_locally_plausible_patches(self):
        selected = _controlled_contamination_patch_ids(
            torch.arange(6, dtype=torch.float32),
            patch_count=4,
            contamination_mode="mixed",
        )
        self.assertEqual(selected, [5, 0, 4, 1])
from MultiMuSc2.models.modules._MSM import (
    MSM2_online,
    aggregate_reference_distances,
)


def synthetic_features(image_count=5):
    images = []
    for image_id in range(image_count):
        offset = image_id * 0.05
        images.append(
            torch.tensor(
                [
                    [0.0 + offset, 0.0],
                    [0.0 + offset, 1.0],
                    [1.0 + offset, 0.0],
                    [1.0 + offset, 1.0],
                ],
                dtype=torch.float32,
            )
        )
    return torch.stack(images)


def scoring_timeline(image_count=4):
    timeline = []
    for step in range(image_count):
        if step < 2:
            active = []
            members = []
        else:
            active = [
                {
                    "expert_id": 0,
                    "image_id": 0,
                    "cluster_id": 0,
                    "admission_step": 1,
                }
            ]
            members = [
                {"step": 0, "image_id": 0, "cluster_id": 0},
                {"step": 1, "image_id": 1, "cluster_id": 0},
            ]
        timeline.append(
            {
                "step": step,
                "active_experts_before": active,
                "fuser_training_members_before": members,
            }
        )
    return timeline


class DynamicPipelineTest(unittest.TestCase):
    def test_reentry_counterfactual_audit_does_not_change_scores(self):
        dino = synthetic_features(4)
        timeline = scoring_timeline(4)
        timeline[3]["active_experts_before"] = [
            {
                "expert_id": 0,
                "image_id": 0,
                "cluster_id": 0,
                "admission_step": 1,
            },
            {
                "expert_id": 1,
                "image_id": 1,
                "cluster_id": 1,
                "admission_step": 2,
                "admission_route": "history_reentry_observer_channel_gate",
            },
        ]
        baseline_timeline = [
            {
                **record,
                "active_experts_before": [
                    {
                        key: value
                        for key, value in expert.items()
                        if key != "admission_route"
                    }
                    for expert in record["active_experts_before"]
                ],
            }
            for record in timeline
        ]

        audited_scores, audits = score_dynamic_msm2_layer(
            dino,
            None,
            timeline,
            fusion_mode="dino_only",
        )
        baseline_scores, _ = score_dynamic_msm2_layer(
            dino,
            None,
            baseline_timeline,
            fusion_mode="dino_only",
        )

        self.assertTrue(torch.equal(audited_scores, baseline_scores))
        self.assertEqual(audits[3]["reentry_expert_image_ids"], [1])
        self.assertIsNotNone(audits[3]["reentry_counterfactual_score"])
        self.assertAlmostEqual(
            audits[3]["reentry_score_regret"],
            audits[3]["score_all"]
            - audits[3]["reentry_counterfactual_score"],
        )

    def test_confidence_expert_blends_with_strict_history(self):
        dino = synthetic_features(3)
        timeline = scoring_timeline(3)
        timeline[2]["active_experts_before"][0]["scoring_weight"] = 0.25
        scores, audits = score_dynamic_msm2_layer(
            dino,
            None,
            timeline,
            fusion_mode="dino_only",
        )
        expert_score, _ = MSM2_online(
            dino[2],
            None,
            dino[[0]],
            None,
            fusion_mode="dino_only",
        )
        history_score, _ = _strict_history_fallback(
            dino[2],
            dino[:2],
            reason="test",
        )

        self.assertTrue(
            torch.allclose(
                scores[2],
                history_score + 0.25 * (expert_score - history_score),
            )
        )
        self.assertEqual(audits[2]["effective_mode"], "dino_confidence_blend")
        self.assertEqual(audits[2]["active_committee_confidence"], 0.25)
        self.assertEqual(audits[2]["expert_image_ids"], [0])

    def test_full_and_confidence_experts_report_actual_contributions(self):
        dino = synthetic_features(3)
        timeline = scoring_timeline(3)
        timeline[2]["active_experts_before"] = [
            {
                "expert_id": 10,
                "image_id": 0,
                "cluster_id": 0,
                "admission_step": 0,
                "scoring_weight": 1.0,
                "scoring_patch_reliabilities": [1.0] * 4,
            },
            {
                "expert_id": 11,
                "image_id": 1,
                "cluster_id": 1,
                "admission_step": 1,
                "scoring_weight": 0.25,
                "scoring_patch_reliabilities": [0.5] * 4,
            },
        ]

        _, audits = score_dynamic_msm2_layer(
            dino,
            None,
            timeline,
            fusion_mode="dino_only",
        )

        contributions = audits[2]["selected_reference_contributions"]
        self.assertEqual(
            [item["expert_id"] for item in contributions],
            [10, 11],
        )
        self.assertEqual(
            [item["blend_weight"] for item in contributions],
            [0.75, 0.25],
        )
        self.assertEqual(
            [item["raw_weight"] for item in contributions],
            [3.0, 1.0],
        )
        self.assertEqual(
            [item["effective_weight"] for item in contributions],
            [3.0, 0.5],
        )

    def test_hierarchical_all_one_global_reliability_matches_cross_patch(self):
        dino = synthetic_features(3)
        timeline = scoring_timeline(3)
        timeline[2]["active_experts_before"] = [
            {
                "expert_id": 10,
                "image_id": 0,
                "scoring_weight": 1.0,
                "global_member_reliability": 1.0,
                "scoring_patch_reliabilities": [1.0, 0.5, 1.0, 1.0],
            },
            {
                "expert_id": 11,
                "image_id": 1,
                "scoring_weight": 1.0,
                "global_member_reliability": 1.0,
                "scoring_patch_reliabilities": [0.8, 1.0, 1.0, 0.7],
            },
        ]

        cross_scores, _ = score_dynamic_msm2_layer(
            dino,
            None,
            timeline,
            fusion_mode="dino_only",
        )
        hierarchical_scores, audits = score_dynamic_msm2_layer(
            dino,
            None,
            timeline,
            fusion_mode="dino_only",
            hierarchical_reliability_enabled=True,
        )

        self.assertTrue(torch.equal(cross_scores[2], hierarchical_scores[2]))
        self.assertEqual(
            audits[2]["effective_mode"],
            "dino_hierarchical_reliability",
        )

    def test_hierarchical_global_reliability_shrinks_to_history(self):
        dino = synthetic_features(3)
        history_score, _ = _strict_history_fallback(
            dino[2],
            dino[:2],
            reason="test",
        )
        expert_score, _ = MSM2_online(
            dino[2],
            None,
            dino[[0]],
            None,
            fusion_mode="dino_only",
            dino_patch_reliabilities=torch.ones(1, 4),
        )
        for reliability in (0.0, 0.5, 1.0):
            score, _, _ = _hierarchical_dino_score(
                current_dino=dino[2],
                expert_dino=dino[0:1],
                expert_patch_reliabilities=torch.ones(1, 4),
                expert_global_reliabilities=[reliability],
                expert_ids=[10],
                expert_image_ids=[0],
                history_score=history_score,
                topmin_min=0.0,
                topmin_max=0.3,
                reliability_penalty=1.0,
            )
            expected = history_score + reliability * (
                expert_score - history_score
            )
            self.assertTrue(torch.allclose(score, expected))

    def test_hierarchical_scoring_is_expert_order_invariant(self):
        dino = synthetic_features(3)
        experts = [
            {
                "expert_id": 10,
                "image_id": 0,
                "global_member_reliability": 0.4,
                "scoring_patch_reliabilities": [1.0, 0.5, 1.0, 1.0],
            },
            {
                "expert_id": 11,
                "image_id": 1,
                "global_member_reliability": 0.8,
                "scoring_patch_reliabilities": [0.8, 1.0, 1.0, 0.7],
            },
        ]
        scores = []
        for active in (experts, list(reversed(experts))):
            timeline = scoring_timeline(3)
            timeline[2]["active_experts_before"] = active
            result, audits = score_dynamic_msm2_layer(
                dino,
                None,
                timeline,
                fusion_mode="dino_only",
                hierarchical_reliability_enabled=True,
            )
            scores.append(result[2])
            for contribution in audits[2][
                "selected_reference_contributions"
            ]:
                self.assertLessEqual(
                    contribution["global_blend_contribution"],
                    contribution["local_effective_contribution"],
                )
                self.assertLessEqual(
                    contribution["local_effective_contribution"],
                    contribution["raw_contribution"],
                )
        self.assertTrue(torch.allclose(scores[0], scores[1]))

    def test_confidence_state_is_strictly_historical_during_scoring(self):
        dino = synthetic_features(4)
        timeline = build_dynamic_committee_timeline(
            dino,
            device="cpu",
            manager_kwargs={
                "cold_start_mode": "confidence",
                "cold_start_confidence_history": 16,
                "min_cluster_support": 10,
                "committee_cap": 2,
            },
        )
        _, audits = score_dynamic_msm2_layer(
            dino,
            None,
            timeline,
            fusion_mode="dino_only",
        )

        self.assertEqual(timeline[0]["active_experts_before"], [])
        for step in range(1, len(timeline)):
            active = timeline[step]["active_experts_before"]
            self.assertTrue(active)
            self.assertTrue(
                all(
                    image_id < step
                    for image_id in active[0]["state_member_image_ids"]
                )
            )
            self.assertNotIn(step, audits[step]["expert_image_ids"])
            self.assertEqual(
                audits[step]["effective_mode"],
                "dino_confidence_blend",
            )

    def test_stable_lab_anchor_is_frozen_from_past_only(self):
        dino = synthetic_features(5)
        lab = torch.tensor(
            [
                [40.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [41.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [70.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [80.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [90.0, 2.0, 1.0, 3.0, 2.0, 4.0],
            ],
            dtype=torch.float64,
        )
        timeline = build_dynamic_committee_timeline(
            dino,
            device="cpu",
            lab_features=lab,
            manager_kwargs={
                "cold_start_mode": "confidence",
                "cold_start_reference_confidence_mode": "prequential",
                "cold_start_lab_enabled": True,
                "photometric_stable_lab_anchor_enabled": True,
                "min_cluster_support": 10,
                "committee_cap": 2,
            },
        )

        frozen = [
            record["active_experts_before"][0]["lab_state"]
            for record in timeline
            if record["active_experts_before"]
            and record["active_experts_before"][0]["lab_state"].get(
                "anchor_frozen", False
            )
        ]

        self.assertTrue(frozen)
        anchor = frozen[0]["anchor_center"]
        self.assertTrue(
            all(state["anchor_center"] == anchor for state in frozen)
        )
        for record in timeline:
            if not record["active_experts_before"]:
                continue
            state = record["active_experts_before"][0]["lab_state"]
            if state.get("anchor_frozen", False):
                self.assertTrue(
                    all(
                        image_id < record["step"]
                        for image_id in state["anchor_member_image_ids"]
                    )
                )

    def test_confidence_weight_remains_individual_with_full_expert(self):
        dino = synthetic_features(3)
        timeline = scoring_timeline(3)
        timeline[2]["active_experts_before"] = [
            {
                "expert_id": 0,
                "image_id": 0,
                "cluster_id": 0,
                "admission_step": 0,
            },
            {
                "expert_id": 1,
                "image_id": 1,
                "cluster_id": 1,
                "admission_step": 1,
                "scoring_weight": 0.25,
            },
        ]
        scores, audits = score_dynamic_msm2_layer(
            dino,
            None,
            timeline,
            fusion_mode="dino_only",
        )
        full_score, _ = MSM2_online(
            dino[2], None, dino[[0]], None, fusion_mode="dino_only"
        )
        confidence_score, _ = MSM2_online(
            dino[2], None, dino[[1]], None, fusion_mode="dino_only"
        )
        expected = full_score + 0.25 * (confidence_score - full_score)

        self.assertTrue(torch.allclose(scores[2], expected))
        self.assertEqual(audits[2]["effective_mode"], "dino_confidence_blend")
        self.assertEqual(
            audits[2]["active_expert_scoring_weights"], [1.0, 0.25]
        )

    def test_photometric_drift_rejects_high_risk_pool_without_cross_category_route(
        self,
    ):
        dino = synthetic_features(10)
        dino[4:] = -dino[4:]
        lab = torch.tensor(
            [
                [40.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [40.1, 2.0, 1.0, 3.0, 2.0, 4.0],
                [40.2, 2.0, 1.0, 3.0, 2.0, 4.0],
                [40.3, 2.0, 1.0, 3.0, 2.0, 4.0],
                [70.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [70.1, 2.0, 1.0, 3.0, 2.0, 4.0],
                [70.2, 2.0, 1.0, 3.0, 2.0, 4.0],
                [70.3, 2.0, 1.0, 3.0, 2.0, 4.0],
                [70.4, 2.0, 1.0, 3.0, 2.0, 4.0],
                [70.5, 2.0, 1.0, 3.0, 2.0, 4.0],
            ],
            dtype=torch.float64,
        )
        timeline = build_dynamic_committee_timeline(
            dino,
            device="cpu",
            lab_features=lab,
            manager_kwargs={
                "cold_start_mode": "adaptive",
                "cold_start_min_samples": 4,
                "cold_start_max_samples": 6,
                "cold_start_required_members": 3,
                "cold_start_lab_enabled": True,
                "candidate_mode": "legacy",
                "photometric_candidate_enabled": True,
                "photometric_quarantine_steps": 4,
                "photometric_prebuffer_size": 2,
                "lab_ewma_confirm_steps": 2,
                "novel_admission_mode": "channel_gate",
                "committee_cap": 3,
            },
        )

        self.assertFalse(timeline[4]["lab_drift_profile"]["just_confirmed"])
        self.assertTrue(timeline[5]["lab_drift_profile"]["just_confirmed"])
        self.assertFalse(timeline[5]["photometric_gate"]["triggered"])
        self.assertTrue(timeline[6]["photometric_gate"]["triggered"])
        self.assertEqual(timeline[6]["candidate_route"], "photometric_drift")
        self.assertEqual(timeline[6]["novel_quarantine"]["decision"], "opened")
        self.assertEqual(timeline[6]["novel_quarantine"]["image_ids"], [5, 6])
        self.assertEqual(timeline[7]["novel_quarantine"]["decision"], "collecting")
        self.assertEqual(
            timeline[8]["novel_quarantine"]["decision"],
            "rejected_photometric_insufficient_low_risk",
        )
        self.assertEqual(timeline[8]["novel_quarantine"]["low_risk_count"], 1)
        self.assertIsNone(timeline[8]["admission_route"])
        self.assertTrue(
            all(
                record["candidate_route"] != "new_distribution"
                for record in timeline
            )
        )
        self.assertNotIn(
            1,
            [
                expert["expert_id"]
                for expert in timeline[8]["active_experts_before"]
            ],
        )
        self.assertNotIn(
            1,
            [
                expert["expert_id"]
                for expert in timeline[9]["active_experts_before"]
            ],
        )

    def test_stable_lab_does_not_open_photometric_route(self):
        dino = synthetic_features(8)
        lab = torch.tensor(
            [[40.0, 2.0, 1.0, 3.0, 2.0, 4.0]] * 8,
            dtype=torch.float64,
        )
        timeline = build_dynamic_committee_timeline(
            dino,
            device="cpu",
            lab_features=lab,
            manager_kwargs={
                "cold_start_mode": "adaptive",
                "cold_start_min_samples": 4,
                "cold_start_max_samples": 6,
                "cold_start_required_members": 3,
                "cold_start_lab_enabled": True,
                "photometric_candidate_enabled": True,
                "novel_admission_mode": "channel_gate",
                "committee_cap": 3,
            },
        )

        self.assertTrue(
            all(
                record["candidate_route"] != "photometric_drift"
                for record in timeline
            )
        )
        self.assertTrue(
            all(
                record["admission_route"]
                != "photometric_drift_channel_gate"
                for record in timeline
            )
        )

    def test_small_lab_false_confirmation_stays_below_photometric_floor(self):
        dino = synthetic_features(10)
        lab = torch.tensor(
            [
                [40.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [40.1, 2.0, 1.0, 3.0, 2.0, 4.0],
                [40.2, 2.0, 1.0, 3.0, 2.0, 4.0],
                [40.3, 2.0, 1.0, 3.0, 2.0, 4.0],
                [43.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [42.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [43.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [42.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [43.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [42.0, 2.0, 1.0, 3.0, 2.0, 4.0],
            ],
            dtype=torch.float64,
        )
        timeline = build_dynamic_committee_timeline(
            dino,
            device="cpu",
            lab_features=lab,
            manager_kwargs={
                "cold_start_mode": "adaptive",
                "cold_start_min_samples": 4,
                "cold_start_max_samples": 6,
                "cold_start_required_members": 3,
                "cold_start_lab_enabled": True,
                "photometric_candidate_enabled": True,
                "novel_admission_mode": "channel_gate",
                "committee_cap": 3,
            },
        )

        self.assertTrue(
            any(
                (record.get("lab_drift_profile") or {}).get(
                    "just_confirmed", False
                )
                for record in timeline
            )
        )
        self.assertTrue(
            all(
                not (record.get("photometric_gate") or {}).get(
                    "triggered", False
                )
                for record in timeline
            )
        )
        self.assertTrue(
            all(
                record.get("admission_route")
                != "photometric_drift_channel_gate"
                for record in timeline
            )
        )
    def test_bootstrap_lab_state_is_used_only_after_admission(self):
        dino = synthetic_features(5)
        lab = torch.tensor(
            [
                [40.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [40.5, 2.0, 1.0, 3.0, 2.0, 4.0],
                [41.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [41.5, 2.0, 1.0, 3.0, 2.0, 4.0],
                [50.0, 2.0, 1.0, 3.0, 2.0, 4.0],
            ],
            dtype=torch.float64,
        )
        timeline = build_dynamic_committee_timeline(
            dino,
            device="cpu",
            lab_features=lab,
            manager_kwargs={
                "cold_start_mode": "adaptive",
                "cold_start_min_samples": 4,
                "cold_start_max_samples": 6,
                "cold_start_required_members": 3,
                "cold_start_lab_enabled": True,
            },
        )

        self.assertEqual(timeline[3]["admitted_expert_id"], 0)
        self.assertIsNone(timeline[3]["lab_drift_profile"])
        self.assertEqual(timeline[4]["lab_drift_profile"]["expert_id"], 0)

    def test_drift_candidate_audit_is_causal_and_read_only(self):
        base = synthetic_features(1)[0]
        dino = torch.stack([base for _ in range(9)])
        lab = torch.tensor(
            [
                [40.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [40.1, 2.0, 1.0, 3.0, 2.0, 4.0],
                [40.2, 2.0, 1.0, 3.0, 2.0, 4.0],
                [40.3, 2.0, 1.0, 3.0, 2.0, 4.0],
                [70.0, 2.0, 1.0, 3.0, 2.0, 4.0],
                [70.1, 2.0, 1.0, 3.0, 2.0, 4.0],
                [70.2, 2.0, 1.0, 3.0, 2.0, 4.0],
                [70.3, 2.0, 1.0, 3.0, 2.0, 4.0],
                [70.4, 2.0, 1.0, 3.0, 2.0, 4.0],
            ],
            dtype=torch.float64,
        )
        timeline = build_dynamic_committee_timeline(
            dino,
            device="cpu",
            lab_features=lab,
            manager_kwargs={
                "cold_start_mode": "adaptive",
                "cold_start_min_samples": 4,
                "cold_start_max_samples": 6,
                "cold_start_required_members": 3,
                "cold_start_lab_enabled": True,
                "drift_candidate_audit_enabled": True,
                "drift_candidate_pool_size": 4,
                "drift_candidate_keep": 3,
                "committee_cap": 1,
            },
        )

        event = timeline[8]["drift_candidate_audit"]
        self.assertEqual(event["decision"], "audited")
        self.assertTrue(
            all(image_id <= 8 for image_id in event["selected_image_ids"])
        )
        self.assertEqual(len(timeline[8]["active_experts_after"]), 1)
        self.assertIsNone(timeline[8]["admitted_expert_id"])

    def test_positioned_memory_scores_leave_uncovered_patches_invalid(self):
        features = synthetic_features(2)
        scores, valid = _positioned_memory_scores(
            features[1],
            [features[0]],
            [[0]],
            grid_size=2,
            position_radius=0,
        )

        self.assertEqual(valid.tolist(), [True, False, False, False])
        self.assertAlmostEqual(float(scores[0]), 0.05, places=6)
        self.assertTrue(torch.equal(scores[1:], torch.zeros(3)))

    def test_irbank_replaces_only_position_covered_dino_scores(self):
        dino = synthetic_features(4)
        timeline = scoring_timeline()
        timeline[3]["provisional_expert_before"] = {
            "image_id": 1,
            "image_ids": [1],
            "memory_members": [{"image_id": 1, "patch_ids": [0]}],
            "representative_step": 1,
            "start_step": 0,
            "pool_size": 1,
            "weight": 1.0,
            "state": "confirmed_irbank",
        }
        base_score, _ = MSM2_online(
            dino[3],
            None,
            dino[[0]],
            None,
            fusion_mode="dino_only",
            topmin_min=0.02,
            topmin_max=0.3,
        )

        scores, audits = score_dynamic_msm2_layer(
            dino,
            None,
            timeline,
            fusion_mode="dino_only",
            position_radius=0,
        )

        self.assertAlmostEqual(float(scores[3, 0]), 0.1, places=6)
        self.assertTrue(torch.allclose(scores[3, 1:], base_score[1:]))
        self.assertEqual(audits[3]["effective_mode"], "dino_channel_irbank")
        self.assertEqual(audits[3]["memory_patch_count"], 1)
        self.assertEqual(audits[3]["memory_valid_fraction"], 0.25)
        self.assertAlmostEqual(audits[3]["memory_score_max"], 0.1, places=6)
        self.assertAlmostEqual(
            audits[3]["base_score_max_before_memory"],
            float(base_score.max()),
            places=6,
        )

    def test_channel_gate_uses_recent_strict_history_fallback(self):
        dino = synthetic_features(5)
        timeline = scoring_timeline(5)
        timeline[4]["provisional_expert_before"] = {
            "image_id": 3,
            "image_ids": [2, 3],
            "memory_members": [
                {"image_id": 2, "patch_ids": [0]},
                {"image_id": 3, "patch_ids": [0]},
            ],
            "representative_step": 3,
            "start_step": 2,
            "pool_size": 2,
            "weight": 0.25,
            "state": "collecting_channel_gate",
        }
        expected, _ = _strict_history_fallback(
            dino[4],
            dino[2:4],
            "expected",
        )

        scores, audits = score_dynamic_msm2_layer(
            dino,
            None,
            timeline,
            fusion_mode="dino_only",
            transition_fallback_window=2,
        )

        self.assertTrue(torch.allclose(scores[4], expected))
        self.assertEqual(
            audits[4]["effective_mode"],
            "recent_history_dino_fallback",
        )
        self.assertEqual(
            audits[4]["fallback_reason"],
            "category_transition_recent_history",
        )
        self.assertEqual(audits[4]["fallback_history_image_ids"], [2, 3])
        self.assertNotIn("memory_patch_count", audits[4])

    def test_channel_gate_rolling_evidence_triggers_and_latches(self):
        dino = synthetic_features(7)
        timeline = scoring_timeline(7)
        timeline[1]["candidate_route"] = "new_distribution"
        timeline[3]["candidate_route"] = "new_distribution"
        provisional = {
            "image_id": 3,
            "image_ids": [3],
            "representative_step": 3,
            "start_step": 1,
            "pool_size": 3,
            "weight": 0.25,
            "state": "collecting_channel_gate",
        }
        for step in (4, 5, 6):
            timeline[step]["provisional_expert_before"] = provisional

        _, audits = score_dynamic_msm2_layer(
            dino,
            None,
            timeline,
            fusion_mode="dino_only",
            transition_fallback_window=2,
            transition_fallback_require_pre_pool_expert=True,
            transition_fallback_evidence_window=4,
            transition_fallback_min_hits=2,
            transition_fallback_latch=True,
        )

        self.assertEqual(audits[4]["fallback_recent_evidence_steps"], [1, 3])
        self.assertEqual(audits[4]["fallback_recent_evidence_hits"], 2)
        self.assertTrue(audits[4]["transition_fallback_triggered"])
        self.assertTrue(audits[4]["transition_fallback_latched"])
        self.assertEqual(audits[6]["fallback_recent_evidence_steps"], [3])
        self.assertEqual(audits[6]["fallback_recent_evidence_hits"], 1)
        self.assertTrue(audits[6]["transition_fallback_triggered"])
        self.assertTrue(audits[6]["transition_fallback_latched"])

    def test_channel_gate_rolling_evidence_requires_old_expert(self):
        dino = synthetic_features(5)
        timeline = scoring_timeline(5)
        timeline[1]["candidate_route"] = "new_distribution"
        timeline[3]["candidate_route"] = "new_distribution"
        timeline[4]["active_experts_before"] = [
            {"expert_id": 0, "image_id": 2}
        ]
        timeline[4]["provisional_expert_before"] = {
            "image_id": 3,
            "image_ids": [3],
            "representative_step": 3,
            "start_step": 1,
            "pool_size": 3,
            "weight": 0.25,
            "state": "collecting_channel_gate",
        }

        _, audits = score_dynamic_msm2_layer(
            dino,
            None,
            timeline,
            fusion_mode="dino_only",
            transition_fallback_window=2,
            transition_fallback_require_pre_pool_expert=True,
            transition_fallback_evidence_window=4,
            transition_fallback_min_hits=2,
            transition_fallback_latch=True,
        )

        self.assertEqual(audits[4]["fallback_recent_evidence_hits"], 2)
        self.assertFalse(audits[4]["fallback_has_pre_pool_expert"])
        self.assertFalse(audits[4]["transition_fallback_triggered"])

    def test_channel_gate_can_disable_recent_history_fallback(self):
        features = torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.9, 0.1]],
                [[0.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        timeline = [
            {
                "active_experts_before": [],
                "provisional_expert_before": None,
                "fuser_training_members_before": [],
            },
            {
                "active_experts_before": [],
                "provisional_expert_before": None,
                "fuser_training_members_before": [],
            },
            {
                "active_experts_before": [{"expert_id": 0, "image_id": 0}],
                "provisional_expert_before": {
                    "image_id": 1,
                    "image_ids": [1],
                    "weight": 0.25,
                    "pool_size": 1,
                    "state": "collecting_channel_gate",
                },
                "fuser_training_members_before": [],
            },
        ]

        _, audits = score_dynamic_msm2_layer(
            features,
            None,
            timeline,
            fusion_mode="dino_only",
            transition_fallback_window=0,
        )

        self.assertEqual(audits[2]["fallback_history_image_ids"], [])
        self.assertNotEqual(
            audits[2]["fallback_reason"],
            "category_transition_recent_history",
        )

    def test_chunked_online_msm_matches_unchunked_result(self):
        features = synthetic_features(6)
        expected = float(
            aggregate_reference_distances(
                features[-1],
                features[:-1],
                topmin_min=0,
                topmin_max=0.3,
            ).max()
        )
        actual = _image_msm_score(
            features[-1],
            [feature for feature in features[:-1]],
            reference_chunk_size=2,
        )

        self.assertAlmostEqual(actual, expected, places=6)

    def test_chunked_strict_history_fallback_matches_direct_result(self):
        features = synthetic_features(6)
        expected = aggregate_reference_distances(
            features[-1],
            features[:-1],
            topmin_min=0,
            topmin_max=0.3,
        )
        actual, reason = _strict_history_fallback(
            features[-1],
            features[:-1],
            "strict_history_test",
            reference_chunk_size=2,
        )

        self.assertTrue(torch.allclose(actual, expected))
        self.assertEqual(reason, "strict_history_test")

    def test_timeline_snapshots_are_strictly_pre_update(self):
        timeline = build_dynamic_committee_timeline(
            synthetic_features(),
            device="cpu",
            manager_kwargs={"base_ttl": 2, "max_ttl": 4},
        )

        self.assertEqual(len(timeline), 5)
        self.assertIsNone(timeline[0]["ms_score"])
        self.assertIsNone(timeline[0]["channel_support"])
        for record in timeline:
            self.assertTrue(
                all(
                    member["step"] < record["step"]
                    for member in record["fuser_training_members_before"]
                )
            )

    def test_controlled_history_eviction_is_post_update_and_causal(self):
        timeline = build_dynamic_committee_timeline(
            synthetic_features(5),
            device="cpu",
            manager_kwargs={
                "cold_start_mode": "confidence",
                "cold_start_confidence_history": 16,
                "min_cluster_support": 10,
                "committee_cap": 2,
            },
            controlled_history_evictions={
                1: {"expert_id": 0},
            },
        )

        eviction = timeline[1]["controlled_history_eviction"]
        self.assertTrue(eviction["evicted"])
        self.assertEqual(eviction["target_expert_id"], 0)
        self.assertTrue(
            all(
                expert["expert_id"] != 0
                for expert in timeline[2]["active_experts_before"]
            )
        )
        for record in timeline:
            self.assertTrue(
                all(
                    expert["image_id"] < record["step"]
                    for expert in record["active_experts_before"]
                )
            )

    def test_empty_contamination_config_preserves_baseline(self):
        features = synthetic_features(5)
        manager_kwargs = {
            "cold_start_mode": "confidence",
            "candidate_reliability_enabled": True,
            "candidate_patch_reliability_mode": "cross_patch_robust",
            "min_cluster_support": 2,
            "committee_cap": 2,
        }
        baseline_timeline = build_dynamic_committee_timeline(
            features,
            device="cpu",
            manager_kwargs=manager_kwargs,
        )
        disabled_timeline = build_dynamic_committee_timeline(
            features,
            device="cpu",
            manager_kwargs=manager_kwargs,
            contamination_injections={},
        )

        def timeline_projection(timeline):
            return [
                {
                    "step": record["step"],
                    "active_before": [
                        (
                            expert["expert_id"],
                            expert["image_id"],
                            expert.get("scoring_weight", 1.0),
                            expert.get("scoring_patch_reliabilities"),
                        )
                        for expert in record["active_experts_before"]
                    ],
                    "active_after": [
                        (expert["expert_id"], expert["image_id"])
                        for expert in record["active_experts_after"]
                    ],
                    "training_members": [
                        (member["step"], member["image_id"])
                        for member in record[
                            "fuser_training_members_before"
                        ]
                    ],
                    "controlled_contamination": record[
                        "controlled_contamination"
                    ],
                }
                for record in timeline
            ]

        self.assertEqual(
            timeline_projection(disabled_timeline),
            timeline_projection(baseline_timeline),
        )
        baseline_scores, baseline_audits = score_dynamic_msm2_layer(
            features,
            None,
            baseline_timeline,
            fusion_mode="dino_only",
        )
        disabled_scores, disabled_audits = score_dynamic_msm2_layer(
            features,
            None,
            disabled_timeline,
            fusion_mode="dino_only",
        )
        self.assertEqual(len(disabled_scores), len(baseline_scores))
        for disabled, baseline in zip(disabled_scores, baseline_scores):
            self.assertTrue(torch.equal(disabled, baseline))
        self.assertEqual(
            [audit["expert_image_ids"] for audit in disabled_audits],
            [audit["expert_image_ids"] for audit in baseline_audits],
        )

    def test_controlled_contamination_is_admitted_after_scoring(self):
        features = synthetic_features(4)
        features[1, 0] = torch.tensor([8.0, 8.0])
        timeline = build_dynamic_committee_timeline(
            features,
            device="cpu",
            manager_kwargs={
                "cold_start_mode": "confidence",
                "candidate_reliability_enabled": True,
                "candidate_patch_reliability_mode": "cross_patch_robust",
                "min_cluster_support": 10,
                "committee_cap": 2,
            },
            contamination_injections={
                1: {
                    "source_label": "defect",
                    "source_path": "controlled-defect.png",
                }
            },
        )

        injection = timeline[1]["controlled_contamination"]
        self.assertTrue(injection["injected"])
        self.assertEqual(injection["source_image_id"], 1)
        self.assertEqual(injection["reliability_context_count"], 1)
        self.assertNotIn(
            injection["injected_expert_id"],
            [
                expert["expert_id"]
                for expert in timeline[1]["active_experts_before"]
            ],
        )
        self.assertIn(
            injection["injected_expert_id"],
            [
                expert["expert_id"]
                for expert in timeline[2]["active_experts_before"]
            ],
        )
        self.assertIn(
            0,
            [
                expert["image_id"]
                for expert in timeline[2]["active_experts_before"]
                if expert["expert_id"]
                != injection["injected_expert_id"]
            ],
        )

        scores, audits = score_dynamic_msm2_layer(
            features,
            None,
            timeline,
            fusion_mode="dino_only",
            image_score_size=3,
        )
        self.assertIsNotNone(
            audits[2]["score_without_contaminated_reference"]
        )
        expected_score = float(
            torch.nn.functional.interpolate(
                scores[2].reshape(1, 1, 2, 2),
                size=3,
                mode="bilinear",
                align_corners=True,
            ).max()
        )
        self.assertAlmostEqual(audits[2]["score_all"], expected_score)
        self.assertEqual(audits[2]["image_score_size"], 3)
        for audit in audits:
            self.assertTrue(
                all(
                    int(image_id) < int(audit["step"])
                    for image_id in audit["expert_image_ids"]
                )
            )

    def test_hierarchical_patch_and_mixed_injections_are_causal(self):
        features = synthetic_features(4)
        features[1, 0] = torch.tensor([8.0, 8.0])
        for mode in ("patch", "mixed"):
            timeline = build_dynamic_committee_timeline(
                features,
                device="cpu",
                manager_kwargs={
                    "cold_start_mode": "confidence",
                    "candidate_reliability_enabled": True,
                    "candidate_patch_reliability_mode": (
                        "hierarchical_global_local"
                    ),
                    "min_cluster_support": 10,
                    "committee_cap": 2,
                },
                contamination_injections={
                    1: {
                        "mode": mode,
                        "patch_ratio": 0.25,
                        "source_label": "defect",
                    }
                },
            )
            injection = timeline[1]["controlled_contamination"]
            injected_id = injection["injected_expert_id"]
            self.assertTrue(injection["injected"])
            self.assertEqual(injection["contamination_mode"], mode)
            self.assertEqual(
                injection["base_patch_reliability_source"],
                "feature_provenance_hybrid",
            )
            self.assertNotIn(
                injected_id,
                [
                    expert["expert_id"]
                    for expert in timeline[1]["active_experts_before"]
                ],
            )
            injected = next(
                expert
                for expert in timeline[2]["active_experts_before"]
                if int(expert["expert_id"]) == int(injected_id)
            )
            self.assertEqual(injected["controlled_contamination_mode"], mode)
            self.assertIn("local_patch_reliabilities", injected)
            self.assertIn("global_member_reliability", injected)
            self.assertLessEqual(
                injection["base_patch_reliability_mean"],
                1.0,
            )
            self.assertGreater(
                injection["base_patch_reliability_min"],
                0.0,
            )
            if mode == "patch":
                self.assertEqual(
                    injected["reliability_audit"][
                        "controlled_global_context"
                    ],
                    "step_start_clean_expert",
                )

            _, audits = score_dynamic_msm2_layer(
                features,
                None,
                timeline,
                fusion_mode="dino_only",
                hierarchical_reliability_enabled=True,
            )
            for audit in audits:
                self.assertTrue(
                    all(
                        int(image_id) < int(audit["step"])
                        for image_id in audit["expert_image_ids"]
                    )
                )

    def test_mixed_replay_scores_the_causal_hybrid_feature(self):
        features = synthetic_features(4)
        features[1, 0] = torch.tensor([8.0, 8.0])
        timeline = build_dynamic_committee_timeline(
            features,
            device="cpu",
            manager_kwargs={
                "cold_start_mode": "confidence",
                "candidate_reliability_enabled": True,
                "candidate_patch_reliability_mode": (
                    "hierarchical_global_local"
                ),
                "min_cluster_support": 10,
                "committee_cap": 2,
            },
            contamination_injections={
                1: {
                    "mode": "mixed",
                    "patch_ratio": 0.25,
                    "source_label": "defect",
                    "replay_steps": [2],
                }
            },
        )
        replay = timeline[2]["controlled_contamination_replay"]
        self.assertEqual(replay["source_image_id"], 1)
        self.assertEqual(replay["clean_image_id"], 0)
        self.assertLess(replay["source_image_id"], 2)
        self.assertLess(replay["clean_image_id"], 2)

        replay_scores, _ = score_dynamic_msm2_layer(
            features,
            None,
            timeline,
            fusion_mode="dino_only",
            hierarchical_reliability_enabled=True,
        )
        manual_features = features.clone()
        patch_ids = torch.as_tensor(replay["patch_ids"], dtype=torch.long)
        manual_features[2] = manual_features[replay["clean_image_id"]]
        manual_features[2].index_copy_(
            0,
            patch_ids,
            manual_features[replay["source_image_id"]].index_select(
                0,
                patch_ids,
            ),
        )
        manual_timeline = [dict(record) for record in timeline]
        manual_timeline[2].pop("controlled_contamination_replay")
        manual_scores, _ = score_dynamic_msm2_layer(
            manual_features,
            None,
            manual_timeline,
            fusion_mode="dino_only",
            hierarchical_reliability_enabled=True,
        )
        self.assertTrue(torch.equal(replay_scores[2], manual_scores[2]))

    def test_short_ms_uses_only_recent_strict_history(self):
        features = synthetic_features(6)
        timeline = build_dynamic_committee_timeline(
            features,
            device="cpu",
            ms_short_windows=(2, 4),
        )

        expected_long = _image_msm_score(
            features[5],
            list(features[:5]),
        )
        expected_short = _image_msm_score(
            features[5],
            list(features[3:5]),
        )

        self.assertAlmostEqual(timeline[5]["ms_score"], expected_long)
        self.assertAlmostEqual(
            timeline[5]["ms_short_scores"]["2"],
            expected_short,
        )
        self.assertEqual(
            timeline[5]["ms_short_history_sizes"],
            {"2": 2, "4": 4},
        )

    def test_all_modes_share_timeline_and_return_finite_scores(self):
        dino = synthetic_features(4)
        clip = dino * 1.7
        timeline = scoring_timeline()

        for mode in (
            "fuser",
            "fixed",
            "dino_only",
            "clip_only",
            "strict_history_dino",
        ):
            scores, audits = score_dynamic_msm2_layer(
                dino,
                None if mode in {"dino_only", "strict_history_dino"} else clip,
                timeline,
                fusion_mode=mode,
            )
            self.assertEqual(scores.shape, (4, 4))
            self.assertTrue(torch.isfinite(scores).all())
            self.assertEqual(audits[0]["fallback_reason"], "first_image_unavailable")
            if mode == "clip_only":
                self.assertEqual(audits[1]["effective_mode"], "clip_fallback")

    def test_provisional_expert_blends_with_existing_dino_path(self):
        dino = synthetic_features(4)
        timeline = scoring_timeline()
        timeline[2]["provisional_expert_before"] = {
            "image_id": 1,
            "representative_step": 1,
            "start_step": 0,
            "pool_size": 2,
            "weight": 0.25,
        }
        base_score, _ = MSM2_online(
            dino[2],
            None,
            dino[[0]],
            None,
            fusion_mode="dino_only",
            topmin_min=0.02,
            topmin_max=0.3,
        )
        provisional_score, _ = MSM2_online(
            dino[2],
            None,
            dino[[1]],
            None,
            fusion_mode="dino_only",
            topmin_min=0.02,
            topmin_max=0.3,
        )

        scores, audits = score_dynamic_msm2_layer(
            dino,
            None,
            timeline,
            fusion_mode="dino_only",
        )

        self.assertTrue(
            torch.allclose(
                scores[2],
                0.75 * base_score + 0.25 * provisional_score,
            )
        )
        self.assertEqual(audits[2]["effective_mode"], "dino_provisional")
        self.assertEqual(audits[2]["provisional_expert_image_id"], 1)
        self.assertEqual(audits[2]["provisional_expert_image_ids"], [1])
        self.assertEqual(audits[2]["provisional_pool_size"], 2)

    def test_confirmed_soft_committee_uses_all_historical_members(self):
        dino = synthetic_features(4)
        timeline = scoring_timeline()
        timeline[3]["provisional_expert_before"] = {
            "image_id": 1,
            "image_ids": [1, 2],
            "representative_step": 1,
            "start_step": 0,
            "pool_size": 4,
            "weight": 0.5,
            "state": "confirmed",
        }
        base_score, _ = MSM2_online(
            dino[3],
            None,
            dino[[0]],
            None,
            fusion_mode="dino_only",
            topmin_min=0.02,
            topmin_max=0.3,
        )
        committee_score, _ = MSM2_online(
            dino[3],
            None,
            dino[[1, 2]],
            None,
            fusion_mode="dino_only",
            topmin_min=0.02,
            topmin_max=0.3,
        )

        scores, audits = score_dynamic_msm2_layer(
            dino,
            None,
            timeline,
            fusion_mode="dino_only",
        )

        self.assertTrue(
            torch.allclose(scores[3], 0.5 * base_score + 0.5 * committee_score)
        )
        self.assertEqual(audits[3]["provisional_expert_image_ids"], [1, 2])


    def test_reset_starts_a_new_causal_history_segment(self):
        features = synthetic_features(6)
        timeline = build_dynamic_committee_timeline(
            features,
            device="cpu",
            reset_steps=[3],
        )
        scores, audits = score_dynamic_msm2_layer(
            features,
            None,
            timeline,
            fusion_mode="strict_history_dino",
        )
        expected = aggregate_reference_distances(
            features[4],
            features[3:4],
            topmin_min=0,
            topmin_max=0.3,
        )

        self.assertTrue(timeline[3]["state_reset_before"])
        self.assertEqual(timeline[3]["segment_start"], 3)
        self.assertIsNone(timeline[3]["ms_score"])
        self.assertEqual(audits[3]["fallback_reason"], "first_image_unavailable")
        self.assertTrue(torch.equal(scores[3], torch.zeros_like(scores[3])))
        self.assertTrue(torch.allclose(scores[4], expected))
        self.assertNotIn("dino_mean", audits[4])

    def test_fuser_retrains_only_when_member_signature_changes(self):
        dino = synthetic_features(4)
        _, audits = score_dynamic_msm2_layer(
            dino,
            dino * 2.0,
            scoring_timeline(),
            fusion_mode="fuser",
        )

        self.assertTrue(audits[2]["fuser_retrained"])
        self.assertFalse(audits[3]["fuser_retrained"])

    def test_fixed_recomputes_scales_without_reporting_fuser_retrain(self):
        dino = synthetic_features(4)
        _, audits = score_dynamic_msm2_layer(
            dino,
            dino * 2.0,
            scoring_timeline(),
            fusion_mode="fixed",
        )

        self.assertTrue(audits[2]["training_recomputed"])
        self.assertFalse(audits[2]["fuser_retrained"])

    def test_fit_once_freezes_fuser_after_first_successful_fit(self):
        dino = synthetic_features(5)
        timeline = scoring_timeline(5)
        timeline[4]["fuser_training_members_before"] = [
            {"step": 0, "image_id": 0, "cluster_id": 0},
            {"step": 1, "image_id": 1, "cluster_id": 0},
            {"step": 2, "image_id": 2, "cluster_id": 0},
        ]
        _, audits = score_dynamic_msm2_layer(
            dino,
            dino * 2.0,
            timeline,
            fusion_mode="fuser",
            retrain_policy="fit_once",
        )

        self.assertTrue(audits[2]["fuser_retrained"])
        self.assertFalse(audits[4]["fuser_retrained"])
        self.assertEqual(audits[4]["training_image_count"], 2)

    def test_committee_gate_waits_for_three_stable_active_experts(self):
        dino = synthetic_features(6)
        timeline = scoring_timeline(6)
        active = [
            {
                "expert_id": image_id,
                "image_id": image_id,
                "cluster_id": image_id,
                "admission_step": image_id,
            }
            for image_id in range(3)
        ]
        timeline[4]["active_experts_before"] = active
        timeline[5]["active_experts_before"] = active
        gated_scores, audits = score_dynamic_msm2_layer(
            dino,
            dino * 2.0,
            timeline,
            fusion_mode="fuser",
            retrain_policy="committee_gate",
            training_source="active_committee",
            committee_min_experts=3,
            committee_stable_steps=2,
        )
        dino_scores, _ = score_dynamic_msm2_layer(
            dino,
            None,
            timeline,
            fusion_mode="dino_only",
        )

        self.assertEqual(audits[4]["fallback_reason"], "committee_not_stable")
        self.assertEqual(audits[4]["effective_mode"], "dino_gate")
        self.assertTrue(torch.allclose(gated_scores[4], dino_scores[4]))
        self.assertFalse(audits[4]["fuser_retrained"])
        self.assertTrue(audits[5]["fuser_retrained"])
        self.assertEqual(audits[5]["fuser_training_image_ids"], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
