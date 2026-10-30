import unittest

import numpy as np
import torch

from MultiMuSc2.models.modules._DYNAMIC_EXPERT import DynamicExpertManager


def patch_features(x, y):
    return torch.tensor([[x, y], [x, y]], dtype=torch.float32)


def prepared_manager(committee_cap=1):
    manager = DynamicExpertManager(
        min_cluster_support=2,
        committee_cap=committee_cap,
        base_ttl=1,
        max_ttl=1,
    )
    manager.ms_history.append(1.0)
    manager.support_history.append(0.5)
    return manager


class DynamicExpertManagerTest(unittest.TestCase):
    @staticmethod
    def denstream_member(step, image_id, channel_support=0.8):
        return {
            "step": int(step),
            "image_id": int(image_id),
            "embedding": torch.tensor([1.0, 0.0]),
            "lab_feature": np.asarray(
                [50.0, 5.0, 128.0, 8.0, 128.0, 8.0],
                dtype=np.float64,
            ),
            "channel_support": float(channel_support),
            "provisional_channel_support": float(channel_support),
        }

    @staticmethod
    def reliability_member(manager, image_id, features):
        base = torch.ones(features.shape[0])
        return {
            "step": int(image_id),
            "image_id": int(image_id),
            "embedding": torch.nn.functional.normalize(
                features.mean(dim=0), dim=0
            ),
            "ms_score": 0.5,
            "historical_ms_threshold": 1.0,
            "base_patch_reliabilities": base,
            "patch_reliabilities": base.clone(),
            "reliability_patch_bank": manager._candidate_patch_bank(
                features, base
            ),
        }

    def test_bidirectional_support_is_stronger_than_one_way_support(self):
        manager = DynamicExpertManager(
            candidate_reliability_enabled=True,
            budget_patch_prototype_count=4,
            candidate_reliability_position_radius=1,
            candidate_unidirectional_weight=0.25,
        )
        left_features = torch.eye(4)
        mutual_members = [
            self.reliability_member(manager, 0, left_features),
            self.reliability_member(manager, 1, left_features.clone()),
        ]
        manager._refresh_candidate_reliabilities(mutual_members)
        mutual_reliability = mutual_members[0]["member_reliability"]

        one_way_features = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 4,
            dtype=torch.float32,
        )
        one_way_members = [
            self.reliability_member(manager, 0, left_features),
            self.reliability_member(manager, 1, one_way_features),
        ]
        manager._refresh_candidate_reliabilities(one_way_members)

        self.assertGreater(
            mutual_reliability,
            one_way_members[0]["member_reliability"],
        )
        self.assertGreater(
            mutual_members[0]["reliability_audit"]["bidirectional_support"],
            one_way_members[0]["reliability_audit"]["bidirectional_support"],
        )
        self.assertTrue(
            torch.all(one_way_members[0]["patch_reliabilities"] > 0.0)
        )

    def test_bidirectional_support_uses_local_neighborhood_on_real_grid(self):
        manager = DynamicExpertManager(
            candidate_reliability_enabled=True,
            budget_patch_prototype_count=32,
            candidate_reliability_position_radius=1,
            candidate_unidirectional_weight=0.25,
        )
        grid_size = 37
        patch_count = grid_size * grid_size
        base = torch.zeros(patch_count, 4)
        base[:, 0] = 1.0
        shifted = base.clone()
        shifted[grid_size + 1 :, 1] = 1.0
        left = self.reliability_member(manager, 0, base)
        right = self.reliability_member(manager, 1, shifted)
        _, _, audit = manager._bidirectional_member_support(left, right)

        self.assertGreater(audit["mutual_fraction"], 0.0)
        self.assertLess(audit["mutual_fraction"], 1.0)

    def test_cross_patch_reliability_preserves_uniform_low_support(self):
        manager = DynamicExpertManager(
            candidate_reliability_enabled=True,
            candidate_patch_reliability_mode="cross_patch_robust",
        )

        reliability = manager._cross_patch_scoring_reliability(
            torch.full((16,), 0.35)
        )

        self.assertTrue(torch.equal(reliability, torch.ones(16)))

    def test_cross_patch_reliability_suppresses_only_low_outlier(self):
        manager = DynamicExpertManager(
            candidate_reliability_enabled=True,
            candidate_patch_reliability_mode="cross_patch_robust",
        )
        support = torch.full((16,), 0.9)
        support[-1] = 0.2

        reliability = manager._cross_patch_scoring_reliability(support)

        self.assertTrue(torch.equal(reliability[:-1], torch.ones(15)))
        self.assertLess(float(reliability[-1]), 1.0)

    def test_cross_patch_mode_separates_member_and_scoring_reliability(self):
        manager = DynamicExpertManager(
            candidate_reliability_enabled=True,
            candidate_patch_reliability_mode="cross_patch_robust",
            candidate_reliability_position_radius=0,
        )
        features = torch.eye(4)
        members = [
            self.reliability_member(manager, 0, features),
            self.reliability_member(manager, 1, features.clone()),
        ]
        members[0]["channel_support"] = 0.2
        members[1]["channel_support"] = 0.8

        manager._refresh_candidate_reliabilities(members)

        self.assertLess(
            members[0]["member_reliability"],
            members[1]["member_reliability"],
        )
        self.assertTrue(
            torch.equal(
                members[0]["scoring_patch_reliabilities"],
                torch.ones(4),
            )
        )
        self.assertEqual(
            members[0]["reliability_audit"]["channel_evidence"],
            0.2,
        )

    def test_hierarchical_reliability_filters_local_before_global(self):
        manager = DynamicExpertManager(
            candidate_reliability_enabled=True,
            candidate_patch_reliability_mode="hierarchical_global_local",
            candidate_reliability_position_radius=0,
        )
        clean = torch.eye(4)
        partial = clean.clone()
        partial[-1] = torch.tensor([-1.0, 0.0, 0.0, 0.0])
        members = [
            self.reliability_member(manager, 0, clean),
            self.reliability_member(manager, 1, partial),
        ]
        members[0]["base_patch_reliabilities"] = torch.tensor(
            [0.8, 0.7, 0.6, 0.5]
        )
        for member in members:
            member["channel_support"] = 0.8

        manager._refresh_candidate_reliabilities(members)

        local = members[0]["local_patch_reliabilities"]
        self.assertTrue(torch.all((local > 0.0) & (local <= 1.0)))
        self.assertLess(float(local[-1]), float(local[:-1].mean()))
        self.assertTrue(
            torch.equal(local, members[0]["scoring_patch_reliabilities"])
        )
        self.assertTrue(
            torch.all(
                local <= members[0]["base_patch_reliabilities"]
            )
        )
        self.assertEqual(
            members[0]["global_member_reliability"],
            members[0]["member_reliability"],
        )
        self.assertIn("structure_evidence", members[0])
        self.assertLessEqual(members[0]["structure_evidence"], 1.0)
        self.assertAlmostEqual(
            members[0]["global_member_reliability"],
            max(
                manager.candidate_reliability_floor,
                0.8
                * members[0]["reliability_audit"]["risk_evidence"]
                * members[0]["structure_evidence"],
            ),
        )

    def test_hierarchical_global_reliability_uses_channel_conflict(self):
        manager = DynamicExpertManager(
            candidate_reliability_enabled=True,
            candidate_patch_reliability_mode="hierarchical_global_local",
            candidate_reliability_position_radius=0,
        )
        features = torch.eye(4)
        members = [
            self.reliability_member(manager, 0, features),
            self.reliability_member(manager, 1, features.clone()),
        ]
        members[0]["channel_support"] = 0.2
        members[1]["channel_support"] = 0.8

        manager._refresh_candidate_reliabilities(members)

        self.assertTrue(
            torch.equal(
                members[0]["local_patch_reliabilities"],
                torch.ones(4),
            )
        )
        self.assertLess(
            members[0]["global_member_reliability"],
            members[1]["global_member_reliability"],
        )

    def test_hierarchical_missing_risk_history_is_not_normal_evidence(self):
        manager = DynamicExpertManager(
            candidate_reliability_enabled=True,
            candidate_patch_reliability_mode="hierarchical_global_local",
            candidate_reliability_position_radius=0,
        )
        features = torch.eye(4)
        members = [
            self.reliability_member(manager, 0, features),
            self.reliability_member(manager, 1, features.clone()),
        ]
        for member in members:
            member.pop("historical_ms_threshold")
            member["channel_support"] = 0.8

        manager._refresh_candidate_reliabilities(members)

        self.assertEqual(
            members[0]["reliability_audit"]["risk_evidence"],
            manager.candidate_reliability_floor,
        )
        self.assertEqual(
            members[0]["reliability_audit"]["risk_evidence_source"],
            "missing_history",
        )

    def test_rejects_unknown_patch_reliability_mode(self):
        with self.assertRaisesRegex(
            ValueError,
            "candidate_patch_reliability_mode",
        ):
            DynamicExpertManager(
                candidate_patch_reliability_mode="unknown",
            )

    def test_soft_candidate_keeps_members_without_memory_patch_ids(self):
        manager = DynamicExpertManager(
            candidate_reliability_enabled=True,
            novel_memory_keep_fraction=0.5,
        )
        members = [
            {
                "image_id": 0,
                "step": 0,
                "ms_short_score": 0.2,
                "memory_patch_ids": [],
                "member_reliability": 0.8,
            },
            {
                "image_id": 1,
                "step": 1,
                "ms_short_score": 0.3,
                "memory_patch_ids": [],
                "member_reliability": 0.7,
            },
        ]

        selected = manager._select_irbank_members(members, min_keep=1)

        self.assertEqual(
            [item["image_id"] for item in selected],
            [0, 1],
        )

    def test_member_reliability_weights_lab_state_update(self):
        manager = DynamicExpertManager(
            candidate_reliability_enabled=True,
            cold_start_lab_enabled=True,
        )
        members = [
            {
                "step": 0,
                "image_id": 0,
                "lab_feature": np.zeros(6),
                "member_reliability": 0.9,
            },
            {
                "step": 1,
                "image_id": 1,
                "lab_feature": np.full(6, 100.0),
                "member_reliability": 0.1,
            },
        ]

        state = manager._build_lab_state(members, step=1)

        np.testing.assert_allclose(state["center"], np.full(6, 10.0))

    def test_denstream_temporary_uses_effective_support_not_event_count(self):
        manager = DynamicExpertManager(
            photometric_state_distribution_enabled=True,
            photometric_temporary_state_enabled=True,
            photometric_denstream_support_enabled=True,
            photometric_denstream_decay=0.9,
            photometric_denstream_mature_support=0.5,
            photometric_denstream_discard_support=0.05,
        )
        first_members = [
            self.denstream_member(0, 0),
            self.denstream_member(0, 1),
        ]
        evidence = {
            "increment": 0.2,
            "dino_evidence": 1.0,
            "channel_evidence": 0.5,
            "risk_evidence": 0.4,
        }
        created = manager._record_photometric_state(
            first_members,
            0,
            state_status="temporary",
            source_type="temporary",
            temporary_support_evidence=evidence,
        )
        state_id = created["historical_state_id"]
        manager._decay_temporary_states(1)
        updated = manager._update_matched_historical_state(
            {"matched_state_id": state_id},
            [self.denstream_member(1, 2), self.denstream_member(1, 3)],
            1,
            temporary_support_evidence=evidence,
        )

        self.assertEqual(updated["state_status_after"], "temporary")
        self.assertEqual(
            manager.historical_states[0]["support_event_count"], 2
        )
        self.assertAlmostEqual(
            manager.historical_states[0]["temporary_effective_support"],
            0.38,
        )

        manager._decay_temporary_states(2)
        updated = manager._update_matched_historical_state(
            {"matched_state_id": state_id},
            [self.denstream_member(2, 4), self.denstream_member(2, 5)],
            2,
            temporary_support_evidence=evidence,
        )
        self.assertEqual(updated["state_status_after"], "stable")
        self.assertTrue(
            any(
                event["event_type"] == "mature"
                and event["reason"]
                == "denstream_effective_support_matured"
                for event in manager.state_lifecycle_events
            )
        )

    def test_fixed_temporary_maturation_is_unchanged_when_denstream_disabled(self):
        manager = DynamicExpertManager(
            photometric_state_distribution_enabled=True,
            photometric_temporary_state_enabled=True,
            photometric_denstream_support_enabled=False,
        )
        members = [
            self.denstream_member(0, 0),
            self.denstream_member(0, 1),
        ]
        created = manager._record_photometric_state(
            members,
            0,
            state_status="temporary",
            source_type="temporary",
        )
        updated = manager._update_matched_historical_state(
            {"matched_state_id": created["historical_state_id"]},
            [self.denstream_member(1, 2), self.denstream_member(1, 3)],
            1,
        )

        self.assertEqual(updated["state_status_after"], "stable")
        self.assertNotIn(
            "temporary_effective_support", manager.historical_states[0]
        )

    def test_denstream_temporary_decays_and_is_evicted_without_support(self):
        manager = DynamicExpertManager(
            photometric_state_distribution_enabled=True,
            photometric_temporary_state_enabled=True,
            photometric_denstream_support_enabled=True,
            photometric_denstream_decay=0.5,
            photometric_denstream_mature_support=0.5,
            photometric_denstream_discard_support=0.1,
        )
        manager._record_photometric_state(
            [self.denstream_member(0, 0), self.denstream_member(0, 1)],
            0,
            state_status="temporary",
            source_type="temporary",
            temporary_support_evidence={
                "increment": 0.2,
                "dino_evidence": 1.0,
                "channel_evidence": 0.5,
                "risk_evidence": 0.4,
            },
        )
        manager.state_lifecycle_events = []
        manager._decay_temporary_states(2)

        self.assertEqual(manager.historical_states, [])
        self.assertEqual(
            manager.state_lifecycle_events[-1]["reason"],
            "temporary_effective_support_decay",
        )

    def test_denstream_support_increment_is_continuous_evidence_product(self):
        manager = DynamicExpertManager(
            photometric_denstream_support_enabled=True,
        )
        strong_members = [
            self.denstream_member(0, 0, 0.9),
            self.denstream_member(0, 1, 0.9),
        ]
        weak_members = [
            self.denstream_member(0, 0, 0.2),
            {
                **self.denstream_member(0, 1, 0.2),
                "embedding": torch.tensor([0.91, 0.41460824]),
            },
        ]
        strong = manager._temporary_support_evidence(
            strong_members,
            {"risk_ratios": [0.5, 0.5]},
        )
        weak = manager._temporary_support_evidence(
            weak_members,
            {"risk_ratios": [1.2, 1.2]},
        )

        self.assertGreater(strong["increment"], weak["increment"])
        expected_increment = 0.0
        for evidence in strong["member_evidence"]:
            expected_increment = (
                manager.photometric_denstream_decay * expected_increment
                + evidence
            )
        self.assertAlmostEqual(
            strong["increment"],
            expected_increment,
        )
        self.assertEqual(
            len(strong["member_evidence"]),
            len(strong_members),
        )

    def test_fitness_history_uses_joint_similarity_and_adaptive_size(self):
        manager = DynamicExpertManager(
            photometric_fitness_history_enabled=True,
            photometric_history_max=16,
        )
        current = {
            **self.denstream_member(10, 10, 0.9),
            "current_expert_supports": {0: 0.9},
        }
        recent_similar = {
            **self.denstream_member(9, 1, 0.88),
            "current_expert_supports": {0: 0.88},
        }
        older_similar = {
            **self.denstream_member(0, 0, 0.9),
            "current_expert_supports": {0: 0.9},
        }
        recent_dissimilar = {
            **self.denstream_member(8, 2, 0.0),
            "embedding": torch.tensor([-1.0, 0.0]),
            "lab_feature": np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
            ),
            "current_expert_supports": {0: 0.0},
        }
        future = {
            **self.denstream_member(11, 11, 0.9),
            "current_expert_supports": {0: 0.9},
        }

        selected, profile = manager._fitness_history_profile(
            [older_similar, recent_dissimilar, recent_similar, future],
            current,
        )

        self.assertEqual(profile["available_count"], 3)
        self.assertEqual(profile["selected_count"], 1)
        self.assertEqual(profile["selected_image_ids"], [1])
        self.assertEqual(
            [member["image_id"] for member in selected], [1]
        )
        self.assertTrue(
            all(member["step"] < current["step"] for member in selected)
        )
        self.assertNotIn("history_relevance", recent_similar)
        self.assertGreater(selected[0]["history_relevance"], 0.0)

    def test_confidence_cold_start_updates_continuously_after_scoring(self):
        manager = DynamicExpertManager(
            cold_start_mode="confidence",
            cold_start_confidence_history=16,
            min_cluster_support=10,
            committee_cap=2,
        )
        features = patch_features(1.0, 0.0)
        events = []
        confidences = []
        for step in range(3):
            events.append(
                manager.advance(
                    step=step,
                    image_id=step,
                    dino_patch_features=features,
                    ms_score=0.1 + 0.1 * step,
                    channel_support=1.0,
                    expert_channel_supports=(
                        {} if step == 0 else {0: 1.0}
                    ),
                )
            )
            confidences.append(
                manager.active_experts[0]["state_confidence"]
            )

        self.assertEqual(events[0]["admitted_expert_id"], 0)
        self.assertEqual(
            events[0]["bootstrap"]["decision"],
            "confidence_state_created",
        )
        self.assertEqual(confidences[0], 0.0)
        self.assertGreater(confidences[1], confidences[0])
        self.assertGreater(confidences[2], confidences[1])
        self.assertEqual(
            manager.active_experts[0]["state_member_image_ids"],
            [0, 1, 2],
        )
        self.assertNotIn(
            events[2]["bootstrap"]["decision"],
            {"provisional_created", "admitted"},
        )

    def test_confidence_bootstrap_materializes_scoring_patch_reliability(self):
        manager = DynamicExpertManager(
            cold_start_mode="confidence",
            candidate_reliability_enabled=True,
            candidate_patch_reliability_mode="cross_patch_robust",
            min_cluster_support=10,
            committee_cap=2,
        )
        reliability = torch.tensor([0.6, 0.9])
        for step in range(2):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=0.1,
                channel_support=1.0,
                expert_channel_supports=(
                    {} if step == 0 else {0: 1.0}
                ),
                patch_reliabilities=reliability,
            )

        expert = manager.active_experts[0]
        self.assertEqual(
            expert["scoring_patch_reliabilities"],
            [1.0, 1.0],
        )
        training_members = manager.fuser_training_members(2)
        self.assertEqual(len(training_members), 2)
        self.assertTrue(
            all(
                member["scoring_patch_reliabilities"] == [1.0, 1.0]
                for member in training_members
            )
        )
        self.assertTrue(
            any(
                member["patch_reliabilities"]
                != member["scoring_patch_reliabilities"]
                for member in training_members
            )
        )

    def test_legacy_reliability_leaves_bootstrap_scoring_unchanged(self):
        manager = DynamicExpertManager(
            cold_start_mode="confidence",
            candidate_reliability_enabled=True,
            candidate_patch_reliability_mode="legacy_rank_product",
            min_cluster_support=10,
            committee_cap=2,
        )

        manager.advance(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            channel_support=1.0,
            expert_channel_supports={},
            patch_reliabilities=torch.tensor([0.6, 0.9]),
        )

        self.assertNotIn(
            "scoring_patch_reliabilities",
            manager.active_experts[0],
        )

    def test_confidence_cold_start_downweights_structural_risk_outlier(self):
        manager = DynamicExpertManager(
            cold_start_mode="confidence",
            cold_start_confidence_history=16,
            min_cluster_support=10,
            committee_cap=2,
        )
        feature_rows = [
            patch_features(1.0, 0.0),
            patch_features(1.0, 0.0),
            patch_features(1.0, 0.0),
            patch_features(0.0, 1.0),
        ]
        for step, features in enumerate(feature_rows):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1 if step < 3 else 10.0,
                channel_support=1.0 if step < 3 else 0.0,
                expert_channel_supports=(
                    {} if step == 0 else {0: 1.0}
                ),
            )

        event = manager.last_bootstrap_event
        self.assertLess(
            event["member_weights"][3],
            min(event["member_weights"][:3]),
        )
        self.assertIn(event["representative_image_id"], {0, 1, 2})

    def test_prequential_confidence_separates_state_and_reference_trust(self):
        manager = DynamicExpertManager(
            cold_start_mode="confidence",
            cold_start_confidence_history=16,
            cold_start_reference_confidence_mode="prequential",
            min_cluster_support=10,
            committee_cap=2,
        )
        for step in range(2):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=0.1,
                channel_support=1.0,
                expert_channel_supports=(
                    {} if step == 0 else {0: 1.0}
                ),
            )

        trusted = manager.last_bootstrap_event
        self.assertGreater(trusted["state_confidence"], 0.0)
        self.assertEqual(trusted["normality_confidence"], 1.0)
        self.assertEqual(
            trusted["reference_confidence"],
            trusted["state_confidence"],
        )

        manager.advance(
            step=2,
            image_id=2,
            dino_patch_features=patch_features(0.0, 1.0),
            ms_score=10.0,
            channel_support=0.0,
            expert_channel_supports={0: 1.0},
        )

        challenged = manager.last_bootstrap_event
        self.assertEqual(challenged["prequential_support_count"], 2)
        self.assertLess(challenged["prequential_support"], 0.5)
        self.assertLess(
            challenged["normality_confidence"],
            trusted["normality_confidence"],
        )
        self.assertLess(
            challenged["reference_confidence"],
            trusted["reference_confidence"],
        )
        expert = manager.active_experts[0]
        self.assertEqual(
            expert["scoring_weight"],
            expert["reference_confidence"],
        )
        self.assertNotEqual(
            expert["state_confidence"],
            expert["reference_confidence"],
        )

    def test_prequential_support_uses_only_prior_members(self):
        manager = DynamicExpertManager(
            cold_start_mode="confidence",
            cold_start_reference_confidence_mode="prequential",
            min_cluster_support=10,
            committee_cap=2,
        )
        manager.advance(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            channel_support=1.0,
            expert_channel_supports={},
        )
        manager.advance(
            step=1,
            image_id=1,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.2,
            channel_support=0.8,
            expert_channel_supports={0: 1.0},
        )

        event = manager.last_bootstrap_event
        self.assertEqual(event["prequential_profile"]["evidence_count"], 2)
        self.assertIsNone(
            event["prequential_profile"]["structural_support"]
        )
        self.assertEqual(
            manager.active_experts[0]["state_member_image_ids"], [0, 1]
        )

    def test_stable_lab_anchor_freezes_after_reference_becomes_trusted(self):
        manager = DynamicExpertManager(
            cold_start_mode="confidence",
            cold_start_reference_confidence_mode="prequential",
            cold_start_lab_enabled=True,
            photometric_stable_lab_anchor_enabled=True,
            min_cluster_support=10,
            committee_cap=2,
        )
        features = patch_features(1.0, 0.0)
        for step, lightness in enumerate((40.0, 41.0)):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                channel_support=1.0,
                expert_channel_supports=(
                    {} if step == 0 else {0: 1.0}
                ),
                lab_feature=torch.tensor(
                    [lightness, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
            )

        state = manager.active_experts[0]["lab_state"]
        frozen_anchor = list(state["anchor_center"])
        self.assertTrue(state["anchor_frozen"])

        manager.advance(
            step=2,
            image_id=2,
            dino_patch_features=features,
            ms_score=0.1,
            channel_support=1.0,
            expert_channel_supports={0: 1.0},
            lab_feature=torch.tensor(
                [80.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
        )

        state = manager.active_experts[0]["lab_state"]
        self.assertEqual(state["anchor_center"], frozen_anchor)
        self.assertNotEqual(state["coverage_center"], frozen_anchor)

    def test_stable_lab_anchor_accumulates_separate_drift_candidate(self):
        manager = DynamicExpertManager(
            cold_start_mode="confidence",
            cold_start_reference_confidence_mode="prequential",
            cold_start_lab_enabled=True,
            photometric_stable_lab_anchor_enabled=True,
            min_cluster_support=10,
            committee_cap=2,
        )
        features = patch_features(1.0, 0.0)
        for step, lightness in enumerate((40.0, 41.0)):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                channel_support=1.0,
                expert_channel_supports=(
                    {} if step == 0 else {0: 1.0}
                ),
                lab_feature=torch.tensor(
                    [lightness, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
            )

        anchor = list(
            manager.active_experts[0]["lab_state"]["anchor_center"]
        )
        first = manager.advance_lab_state(
            torch.tensor([80.0, 2.0, 1.0, 3.0, 2.0, 4.0]),
            step=2,
            image_id=2,
        )
        second = manager.advance_lab_state(
            torch.tensor([82.0, 2.0, 1.0, 3.0, 2.0, 4.0]),
            step=3,
            image_id=3,
        )

        self.assertEqual(first["candidate_lab_count"], 1)
        self.assertEqual(second["candidate_lab_count"], 2)
        self.assertEqual(second["lab_anchor_center"], anchor)
        self.assertGreater(
            second["anchor_to_candidate_total_distance"], 0.0
        )
        self.assertTrue(second["drift_confirmed"])

    def test_legacy_lab_center_still_updates_when_anchor_is_disabled(self):
        manager = DynamicExpertManager(
            cold_start_mode="confidence",
            cold_start_reference_confidence_mode="prequential",
            cold_start_lab_enabled=True,
            photometric_stable_lab_anchor_enabled=False,
            min_cluster_support=10,
            committee_cap=2,
        )
        features = patch_features(1.0, 0.0)
        centers = []
        for step, lightness in enumerate((40.0, 41.0, 80.0)):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                channel_support=1.0,
                expert_channel_supports=(
                    {} if step == 0 else {0: 1.0}
                ),
                lab_feature=torch.tensor(
                    [lightness, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
            )
            centers.append(
                manager.active_experts[0]["lab_state"]["center"][0]
            )

        self.assertNotEqual(centers[1], centers[2])
        self.assertNotIn(
            "anchor_center", manager.active_experts[0]["lab_state"]
        )

    def test_confidence_update_preserves_lab_detector_continuity(self):
        manager = DynamicExpertManager(
            cold_start_mode="confidence",
            cold_start_reference_confidence_mode="prequential",
            cold_start_lab_enabled=True,
            photometric_stable_lab_anchor_enabled=False,
            lab_ewma_confirm_steps=2,
            min_cluster_support=10,
            committee_cap=2,
        )
        features = patch_features(1.0, 0.0)
        for step, lightness in enumerate((40.0, 41.0)):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                channel_support=1.0,
                expert_channel_supports=(
                    {} if step == 0 else {0: 1.0}
                ),
                lab_feature=torch.tensor(
                    [lightness, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
            )

        initial_state = manager.active_experts[0]["lab_state"]
        detector_center = list(initial_state["detector_center"])
        first = manager.advance_lab_state(
            torch.tensor([80.0, 2.0, 1.0, 3.0, 2.0, 4.0]),
            step=2,
            image_id=2,
        )
        self.assertEqual(first["l_streak"], 1)
        manager.advance(
            step=2,
            image_id=2,
            dino_patch_features=features,
            ms_score=0.1,
            channel_support=1.0,
            expert_channel_supports={0: 1.0},
            lab_feature=torch.tensor(
                [80.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            lab_drift_profile=first,
        )

        updated_state = manager.active_experts[0]["lab_state"]
        self.assertEqual(updated_state["detector_center"], detector_center)
        self.assertEqual(updated_state["l_streak"], 1)
        self.assertNotEqual(updated_state["coverage_center"], detector_center)
        second = manager.advance_lab_state(
            torch.tensor([82.0, 2.0, 1.0, 3.0, 2.0, 4.0]),
            step=3,
            image_id=3,
        )
        self.assertEqual(second["l_streak"], 2)
        self.assertTrue(second["drift_confirmed"])

    def test_confidence_state_defers_update_during_drift_hypothesis(self):
        manager = DynamicExpertManager(
            cold_start_mode="confidence",
            cold_start_reference_confidence_mode="prequential",
            cold_start_lab_enabled=True,
            min_cluster_support=10,
            committee_cap=2,
        )
        features = patch_features(1.0, 0.0)
        for step, lightness in enumerate((40.0, 41.0)):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                channel_support=1.0,
                expert_channel_supports=(
                    {} if step == 0 else {0: 1.0}
                ),
                lab_feature=torch.tensor(
                    [lightness, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
            )

        member_ids_before = list(
            manager.active_experts[0]["state_member_image_ids"]
        )
        manager.advance(
            step=2,
            image_id=2,
            dino_patch_features=features,
            ms_score=0.1,
            channel_support=1.0,
            expert_channel_supports={0: 1.0},
            lab_feature=torch.tensor(
                [80.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            lab_drift_profile={"drift_confirmed": True},
        )

        self.assertEqual(
            manager.active_experts[0]["state_member_image_ids"],
            member_ids_before,
        )
        self.assertEqual(
            manager.last_bootstrap_event["decision"],
            "confidence_state_update_deferred",
        )

    def test_confidence_cold_start_respects_history_budget(self):
        manager = DynamicExpertManager(
            cold_start_mode="confidence",
            cold_start_confidence_history=3,
            min_cluster_support=10,
            committee_cap=2,
        )
        for step in range(5):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=0.1 + step,
                channel_support=1.0,
                expert_channel_supports=(
                    {} if step == 0 else {0: 1.0}
                ),
            )

        self.assertEqual(
            manager.active_experts[0]["state_member_image_ids"],
            [2, 3, 4],
        )

    def test_retired_confidence_state_is_not_recreated(self):
        manager = DynamicExpertManager(
            cold_start_mode="confidence",
            cold_start_confidence_history=3,
            min_cluster_support=10,
            committee_cap=2,
        )
        manager.advance(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            channel_support=1.0,
            expert_channel_supports={},
        )
        manager.active_experts.clear()
        manager.clusters[0]["active_expert_id"] = None

        event = manager.advance(
            step=1,
            image_id=1,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=1.0,
            channel_support=0.0,
            expert_channel_supports={},
        )

        self.assertIsNone(event["admitted_expert_id"])
        self.assertEqual(manager.active_experts, [])
        self.assertIsNone(event["bootstrap"])

    def test_adaptive_cold_start_admits_consistent_medoid(self):
        manager = DynamicExpertManager(
            cold_start_mode="adaptive",
            cold_start_min_samples=4,
            cold_start_max_samples=6,
            cold_start_required_members=3,
        )
        features = patch_features(1.0, 0.0)
        events = [
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=None if step == 0 else 0.1,
                channel_support=None if step == 0 else 1.0,
                expert_channel_supports={},
            )
            for step in range(4)
        ]

        self.assertTrue(
            all(event["admitted_expert_id"] is None for event in events[:3])
        )
        self.assertEqual(events[3]["admitted_expert_id"], 0)
        self.assertEqual(events[3]["admission_route"], "bootstrap")
        self.assertEqual(events[3]["bootstrap"]["pool_size"], 4)
        self.assertEqual(len(events[3]["bootstrap"]["selected_image_ids"]), 3)

    def test_adaptive_bootstrap_materializes_scoring_patch_reliability(self):
        manager = DynamicExpertManager(
            cold_start_mode="adaptive",
            cold_start_min_samples=4,
            cold_start_max_samples=6,
            cold_start_required_members=3,
            candidate_reliability_enabled=True,
            candidate_patch_reliability_mode="cross_patch_robust",
        )
        for step in range(4):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=None if step == 0 else 0.1,
                channel_support=None if step == 0 else 1.0,
                expert_channel_supports={},
                patch_reliabilities=torch.tensor([0.6, 0.9]),
            )

        expert = manager.active_experts[0]
        self.assertEqual(
            expert["scoring_patch_reliabilities"],
            [1.0, 1.0],
        )
        self.assertEqual(len(expert["reference_member_image_ids"]), 3)

    def test_bootstrap_expert_carries_lab_baseline(self):
        manager = DynamicExpertManager(
            cold_start_mode="adaptive",
            cold_start_min_samples=4,
            cold_start_max_samples=6,
            cold_start_required_members=3,
            cold_start_lab_enabled=True,
        )
        features = patch_features(1.0, 0.0)
        for step in range(4):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=None if step == 0 else 0.1,
                channel_support=None if step == 0 else 1.0,
                expert_channel_supports={},
                lab_feature=torch.tensor(
                    [40.0 + step, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
            )

        state = manager.active_experts[0]["lab_state"]
        self.assertIsNotNone(state)
        self.assertEqual(len(state["member_image_ids"]), 3)
        self.assertEqual(len(state["center"]), 6)

    def test_bootstrap_provisional_requires_future_confirmation(self):
        manager = DynamicExpertManager(
            cold_start_mode="adaptive",
            cold_start_min_samples=4,
            cold_start_max_samples=6,
            cold_start_required_members=3,
            cold_start_lab_enabled=True,
            cold_start_provisional_enabled=True,
            cold_start_provisional_confirm_steps=2,
            cold_start_provisional_max_wait_steps=4,
        )
        features = patch_features(1.0, 0.0)
        events = []
        for step in range(6):
            events.append(
                manager.advance(
                    step=step,
                    image_id=step,
                    dino_patch_features=features,
                    ms_score=None if step == 0 else 0.1,
                    channel_support=None if step == 0 else 1.0,
                    provisional_channel_support=(
                        None if step == 0 else 0.0
                    ),
                    expert_channel_supports={},
                    lab_feature=torch.tensor(
                        [40.0 + step, 2.0, 1.0, 3.0, 2.0, 4.0]
                    ),
                )
            )
            if step == 3:
                snapshot = manager.provisional_expert_before_step()
                self.assertIsNotNone(snapshot)
                self.assertEqual(snapshot["state"], "bootstrap_provisional")
                self.assertTrue(
                    all(image_id <= step for image_id in snapshot["image_ids"])
                )
                self.assertEqual(manager.active_experts, [])
                self.assertEqual(manager.historical_states, [])

        self.assertEqual(events[3]["bootstrap"]["decision"], "provisional_created")
        self.assertEqual(
            events[4]["bootstrap"]["decision"], "provisional_confirming"
        )
        self.assertEqual(events[5]["bootstrap"]["decision"], "admitted")
        self.assertEqual(events[5]["admitted_expert_id"], 0)
        self.assertEqual(len(manager.active_experts), 1)
        self.assertEqual(len(manager.historical_states), 1)
        self.assertIsNone(manager.provisional_expert_before_step())

    def test_bootstrap_provisional_rolls_back_without_forced_admission(self):
        manager = DynamicExpertManager(
            cold_start_mode="adaptive",
            cold_start_min_samples=4,
            cold_start_max_samples=4,
            cold_start_required_members=3,
            cold_start_provisional_enabled=True,
            cold_start_provisional_confirm_steps=2,
            cold_start_provisional_max_wait_steps=2,
        )
        initial = patch_features(1.0, 0.0)
        changed = patch_features(0.0, 1.0)
        events = []
        for step in range(6):
            events.append(
                manager.advance(
                    step=step,
                    image_id=step,
                    dino_patch_features=initial if step < 4 else changed,
                    ms_score=None if step == 0 else 0.1,
                    channel_support=None if step == 0 else 1.0,
                    provisional_channel_support=(
                        None if step == 0 else 0.5
                    ),
                    expert_channel_supports={},
                )
            )

        self.assertEqual(events[3]["bootstrap"]["decision"], "provisional_created")
        self.assertEqual(events[5]["bootstrap"]["decision"], "rolled_back")
        self.assertTrue(
            all(event["admitted_expert_id"] is None for event in events)
        )
        self.assertEqual(manager.active_experts, [])
        self.assertEqual(manager.historical_states, [])
        self.assertIsNone(manager.provisional_expert_before_step())
        self.assertEqual(
            events[5]["bootstrap"]["selected_image_ids"], []
        )

    def test_bootstrap_provisional_rejects_high_risk_future_support(self):
        manager = DynamicExpertManager(
            cold_start_mode="adaptive",
            cold_start_min_samples=4,
            cold_start_max_samples=4,
            cold_start_required_members=3,
            cold_start_provisional_enabled=True,
            cold_start_provisional_confirm_steps=2,
            cold_start_provisional_max_wait_steps=2,
        )
        features = patch_features(1.0, 0.0)
        events = []
        for step in range(6):
            events.append(
                manager.advance(
                    step=step,
                    image_id=step,
                    dino_patch_features=features,
                    ms_score=(
                        None
                        if step == 0
                        else 0.1
                        if step < 4
                        else 1.0
                    ),
                    channel_support=None if step == 0 else 1.0,
                    provisional_channel_support=(
                        None if step == 0 else 1.0
                    ),
                    expert_channel_supports={},
                )
            )

        self.assertEqual(events[3]["bootstrap"]["decision"], "provisional_created")
        self.assertEqual(events[5]["bootstrap"]["decision"], "rolled_back")
        self.assertEqual(events[5]["bootstrap"]["confirmation_steps"], 0)
        self.assertEqual(
            events[5]["bootstrap"]["confirmation_risk_ceiling"], 0.1
        )
        self.assertEqual(manager.active_experts, [])
        self.assertEqual(manager.historical_states, [])

    def test_drift_candidate_audit_selects_low_ms_without_admission(self):
        manager = DynamicExpertManager(
            cold_start_mode="adaptive",
            cold_start_min_samples=4,
            cold_start_max_samples=6,
            cold_start_required_members=3,
            cold_start_lab_enabled=True,
            drift_candidate_audit_enabled=True,
            drift_candidate_pool_size=4,
            drift_candidate_keep=3,
        )
        features = patch_features(1.0, 0.0)
        for step in range(4):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=None if step == 0 else 0.1,
                channel_support=None if step == 0 else 1.0,
                expert_channel_supports={},
                lab_feature=torch.tensor(
                    [40.0 + step, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
            )

        scores = [0.4, 0.1, 0.3, 0.2]
        events = []
        for offset, score in enumerate(scores):
            step = offset + 4
            events.append(
                manager.advance_drift_candidate_audit(
                    step=step,
                    image_id=step,
                    dino_patch_features=features,
                    lab_feature=torch.tensor(
                        [60.0 + offset, 2.0, 1.0, 3.0, 2.0, 4.0]
                    ),
                    ms_short_score=score,
                    lab_drift_profile={
                        "expert_id": 0,
                        "drift_confirmed": True,
                    },
                    channel_support=0.8,
                    mature_match_fraction=0.5,
                    recent_provisional_match_fraction=0.1,
                )
            )

        self.assertEqual(events[0]["decision"], "opened")
        self.assertEqual(events[-1]["decision"], "audited")
        self.assertEqual(events[-1]["selected_image_ids"], [5, 7, 6])
        self.assertAlmostEqual(events[-1]["dino_consistency_min"], 1.0)
        self.assertEqual(manager.admitted_count, 1)
        self.assertEqual(len(manager.active_experts), 1)

    def test_latest_candidate_selects_newest_cluster_member(self):
        manager = DynamicExpertManager(
            representative_mode="latest_candidate",
            min_cluster_support=2,
            committee_cap=1,
        )
        manager.ms_history.append(1.0)
        manager.support_history.append(0.5)
        features = patch_features(1.0, 0.0)

        self.advance_candidate(manager, 0, 10, features)
        admitted = self.advance_candidate(manager, 1, 11, features)

        self.assertEqual(admitted["admitted_expert_id"], 0)
        self.assertEqual(manager.active_experts[0]["image_id"], 11)
        self.assertEqual(
            manager.active_experts[0]["representative_mode"],
            "latest_candidate",
        )

    def test_confirmed_member_count_cannot_exceed_tail(self):
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            DynamicExpertManager(
                novel_quarantine_tail=2,
                novel_confirmed_members=3,
            )

    def advance_candidate(
        self,
        manager,
        step,
        image_id,
        features,
        expert_supports=None,
    ):
        return manager.advance(
            step=step,
            image_id=image_id,
            dino_patch_features=features,
            ms_score=0.1,
            channel_support=1.0,
            expert_channel_supports=expert_supports or {},
        )

    def test_deleted_cluster_can_be_readmitted(self):
        manager = prepared_manager()
        features = patch_features(1.0, 0.0)

        self.advance_candidate(manager, 0, 10, features)
        admitted = self.advance_candidate(manager, 1, 11, features)
        readmitted = self.advance_candidate(
            manager,
            2,
            12,
            features,
            {0: 0.0},
        )

        self.assertEqual(admitted["admitted_expert_id"], 0)
        self.assertEqual(readmitted["deleted_expert_ids"], [0])
        self.assertEqual(readmitted["admitted_expert_id"], 1)
        self.assertEqual(readmitted["active_expert_ids_after_step"], [1])
        self.assertEqual(manager.clusters[0]["active_expert_id"], 1)

    def test_committee_cap_counts_only_active_experts(self):
        manager = prepared_manager(committee_cap=1)
        first = patch_features(1.0, 0.0)
        second = patch_features(0.0, 1.0)

        self.advance_candidate(manager, 0, 20, first)
        self.advance_candidate(manager, 1, 21, first)
        self.advance_candidate(manager, 2, 22, second, {0: 0.0})
        replacement = self.advance_candidate(manager, 3, 23, second)

        self.assertEqual(manager.admitted_count, 2)
        self.assertEqual(replacement["admitted_expert_id"], 1)
        self.assertEqual(replacement["active_expert_ids_after_step"], [1])

    def test_snapshot_and_fuser_members_are_strictly_historical(self):
        manager = prepared_manager()
        features = patch_features(1.0, 0.0)

        self.advance_candidate(manager, 0, 30, features)
        self.advance_candidate(manager, 1, 31, features)

        self.assertEqual(
            manager.active_experts_before_step(),
            [
                {
                    "expert_id": 0,
                    "image_id": 30,
                    "cluster_id": 0,
                    "admission_step": 1,
                }
            ],
        )
        self.assertEqual(manager.fuser_training_image_ids(1), [30])
        self.assertEqual(manager.fuser_training_image_ids(2), [30, 31])
        self.assertEqual(
            manager.fuser_training_members(2),
            [
                {"step": 0, "image_id": 30, "cluster_id": 0},
                {"step": 1, "image_id": 31, "cluster_id": 0},
            ],
        )

    def test_new_distribution_route_uses_repeated_provisional_channels(self):
        manager = DynamicExpertManager(
            candidate_mode="dual_path",
            novel_support_quantile=0.3,
            novel_provisional_min=0.05,
            min_cluster_support=2,
        )
        manager.ms_history.append(1.0)
        manager.support_history.extend([0.7, 0.8, 0.9])

        event = manager.advance(
            step=0,
            image_id=10,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=1.2,
            channel_support=0.1,
            expert_channel_supports={},
            provisional_channel_support=0.2,
        )

        self.assertEqual(event["candidate_route"], "new_distribution")
        self.assertEqual(event["admitted_expert_id"], 0)
        self.assertEqual(manager.active_experts[0]["admission_route"], "new_distribution")

    def test_new_distribution_route_requires_high_ms_score(self):
        manager = DynamicExpertManager(
            candidate_mode="dual_path",
            novel_ms_quantile=0.9,
            novel_support_quantile=0.3,
            novel_provisional_min=0.05,
        )
        manager.ms_history.extend([0.8, 1.0, 1.2])
        manager.support_history.extend([0.7, 0.8, 0.9])

        event = manager.advance(
            step=0,
            image_id=10,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=1.0,
            channel_support=0.1,
            expert_channel_supports={},
            provisional_channel_support=0.2,
        )

        self.assertIsNone(event["candidate_route"])
        self.assertIsNone(event["admitted_expert_id"])

    def test_quarantine_restarts_and_selects_from_stable_tail(self):
        manager = DynamicExpertManager(
            candidate_mode="dual_path",
            novel_admission_mode="quarantine",
            novel_quarantine_steps=8,
            novel_quarantine_tail=4,
            novel_ratio_threshold=0.95,
            novel_provisional_min=0.05,
        )
        manager.ms_history.extend([1.0] * 8)
        manager.support_history.extend([0.7, 0.8, 0.9])
        short_scores = [1.2, 1.1, 1.0, 0.98, 0.96, 0.91, 0.93, 0.80, 0.90]
        ratios = [1.0, 0.9, 0.91, 0.92, 0.90, 0.89, 0.91, 0.88, 0.92]

        events = []
        for step in range(9):
            trigger = step < 2
            events.append(
                manager.advance(
                    step=step,
                    image_id=100 + step,
                    dino_patch_features=patch_features(1.0 + step, 1.0),
                    ms_score=1.2 if trigger else 0.5,
                    channel_support=0.1,
                    expert_channel_supports={},
                    provisional_channel_support=0.2 if trigger else 0.0,
                    ms_short_score=short_scores[step],
                    ms_short_ratio=ratios[step],
                )
            )

        self.assertEqual(events[1]["novel_quarantine"]["decision"], "restarted")
        self.assertEqual(events[8]["admission_route"], "new_distribution_quarantine")
        self.assertEqual(events[8]["novel_quarantine"]["selected_image_id"], 107)
        self.assertEqual(manager.active_experts[0]["image_id"], 107)

    def test_photometric_restart_keeps_trigger_expert_and_prebuffer(self):
        manager = DynamicExpertManager(
            candidate_mode="dual_path",
            novel_admission_mode="channel_gate",
            novel_quarantine_steps=8,
            photometric_quarantine_steps=4,
            photometric_prebuffer_size=2,
        )
        features = patch_features(1.0, 0.0)
        manager.novel_quarantine = [
            manager._quarantine_member(
                step=4,
                image_id=4,
                dino_patch_features=features,
                ms_score=1.2,
                ms_short_score=1.2,
                ms_short_ratio=1.1,
                channel_support=0.1,
                provisional_channel_support=0.2,
                memory_patch_ids=[],
                lab_feature=torch.tensor(
                    [40.0, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                candidate_route="new_distribution",
            )
        ]
        manager.novel_quarantine_route = "new_distribution"
        seed_member = manager._quarantine_member(
            step=5,
            image_id=5,
            dino_patch_features=features,
            ms_score=0.8,
            ms_short_score=0.8,
            ms_short_ratio=0.9,
            channel_support=0.8,
            provisional_channel_support=0.0,
            memory_patch_ids=[],
            lab_feature=torch.tensor(
                [60.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            candidate_route="history",
        )

        manager._advance_novel_quarantine(
            step=6,
            image_id=6,
            dino_patch_features=features,
            ms_score=0.7,
            ms_short_score=0.7,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.0,
            trigger_route="photometric_drift",
            memory_patch_ids=[],
            lab_feature=torch.tensor(
                [61.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            photometric_seed_members=[seed_member],
            photometric_trigger_expert_id=0,
            historical_ms_threshold=1.0,
        )

        self.assertEqual(manager.novel_quarantine_route, "photometric_drift")
        self.assertEqual(manager.photometric_trigger_expert_id, 0)
        self.assertEqual(
            [item["image_id"] for item in manager.novel_quarantine],
            [5, 6],
        )

    def test_photometric_pool_uses_snapshot_after_trigger_expert_expires(self):
        manager = DynamicExpertManager(
            cold_start_mode="adaptive",
            cold_start_min_samples=4,
            cold_start_max_samples=4,
            cold_start_required_members=3,
            cold_start_lab_enabled=True,
            base_ttl=1,
            max_ttl=1,
            novel_admission_mode="quarantine",
            photometric_quarantine_steps=2,
        )
        features = patch_features(1.0, 0.0)
        for step in range(4):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=None if step == 0 else 0.1,
                channel_support=None if step == 0 else 1.0,
                expert_channel_supports={},
                lab_feature=torch.tensor(
                    [40.0 + step, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
            )

        manager._advance_novel_quarantine(
            step=4,
            image_id=4,
            dino_patch_features=features,
            ms_score=0.8,
            ms_short_score=0.8,
            ms_short_ratio=0.8,
            channel_support=0.1,
            provisional_channel_support=0.0,
            trigger_route="photometric_drift",
            memory_patch_ids=[],
            lab_feature=torch.tensor(
                [60.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            photometric_trigger_expert_id=0,
            historical_ms_threshold=1.0,
        )
        self.assertIsNotNone(manager.photometric_trigger_lab_center)
        self.assertIsNone(manager.provisional_expert_before_step())

        deleted, _ = manager._advance_lifecycle(
            step=5,
            support_threshold=0.5,
            expert_channel_supports={0: 0.0},
        )
        self.assertEqual(deleted, [0])
        self.assertEqual(manager.active_experts, [])

        _, admitted_expert_id = manager._advance_novel_quarantine(
            step=5,
            image_id=5,
            dino_patch_features=features,
            ms_score=0.8,
            ms_short_score=0.8,
            ms_short_ratio=0.8,
            channel_support=0.1,
            provisional_channel_support=0.0,
            trigger_route=None,
            memory_patch_ids=[],
            lab_feature=torch.tensor(
                [61.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            historical_ms_threshold=1.0,
        )

        self.assertEqual(admitted_expert_id, 1)
        self.assertEqual(manager.active_experts[0]["expert_id"], 1)
        self.assertEqual(
            manager.last_novel_quarantine_event["decision"],
            "admitted",
        )
        self.assertEqual(
            manager.last_novel_quarantine_event["photometric_profile"][
                "reason"
            ],
            "consistent",
        )
        self.assertEqual(
            manager.last_novel_quarantine_event["low_risk_count"],
            2,
        )

    def test_photometric_pool_rejects_without_two_historical_low_risk_members(
        self,
    ):
        manager = DynamicExpertManager(
            cold_start_mode="adaptive",
            cold_start_min_samples=4,
            cold_start_max_samples=4,
            cold_start_required_members=3,
            cold_start_lab_enabled=True,
            novel_admission_mode="quarantine",
            photometric_quarantine_steps=2,
            photometric_min_low_risk_members=2,
        )
        features = patch_features(1.0, 0.0)
        for step in range(4):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=None if step == 0 else 0.1,
                channel_support=None if step == 0 else 1.0,
                expert_channel_supports={},
                lab_feature=torch.tensor(
                    [40.0 + step, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
            )

        manager._advance_novel_quarantine(
            step=4,
            image_id=4,
            dino_patch_features=features,
            ms_score=0.8,
            ms_short_score=0.8,
            ms_short_ratio=0.8,
            channel_support=0.1,
            provisional_channel_support=0.0,
            trigger_route="photometric_drift",
            memory_patch_ids=[],
            lab_feature=torch.tensor(
                [60.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            photometric_trigger_expert_id=0,
            historical_ms_threshold=0.5,
        )
        _, admitted_expert_id = manager._advance_novel_quarantine(
            step=5,
            image_id=5,
            dino_patch_features=features,
            ms_score=0.8,
            ms_short_score=0.8,
            ms_short_ratio=0.8,
            channel_support=0.1,
            provisional_channel_support=0.0,
            trigger_route=None,
            memory_patch_ids=[],
            lab_feature=torch.tensor(
                [61.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            historical_ms_threshold=1.0,
        )

        self.assertIsNone(admitted_expert_id)
        self.assertEqual(
            manager.last_novel_quarantine_event["decision"],
            "rejected_photometric_insufficient_low_risk",
        )
        self.assertEqual(
            manager.last_novel_quarantine_event["low_risk_count"],
            1,
        )
        self.assertEqual(
            manager.last_novel_quarantine_event["low_risk_image_ids"],
            [5],
        )

    def test_rejected_photometric_pool_does_not_protect_trigger_ttl(self):
        manager = DynamicExpertManager(
            cold_start_mode="adaptive",
            cold_start_min_samples=4,
            cold_start_max_samples=4,
            cold_start_required_members=3,
            cold_start_lab_enabled=True,
            base_ttl=1,
            max_ttl=1,
            novel_admission_mode="quarantine",
            photometric_quarantine_steps=2,
        )
        features = patch_features(1.0, 0.0)
        for step in range(4):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=None if step == 0 else 0.1,
                channel_support=None if step == 0 else 1.0,
                expert_channel_supports={},
                lab_feature=torch.tensor(
                    [40.0 + step, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
            )

        manager._advance_novel_quarantine(
            step=4,
            image_id=4,
            dino_patch_features=features,
            ms_score=0.8,
            ms_short_score=0.8,
            ms_short_ratio=0.8,
            channel_support=0.1,
            provisional_channel_support=0.0,
            trigger_route="photometric_drift",
            memory_patch_ids=[],
            lab_feature=torch.tensor(
                [60.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            photometric_trigger_expert_id=0,
            historical_ms_threshold=1.0,
        )

        deleted, _ = manager._advance_lifecycle(
            step=5,
            support_threshold=0.5,
            expert_channel_supports={0: 0.0},
        )
        self.assertEqual(deleted, [0])

        _, admitted_expert_id = manager._advance_novel_quarantine(
            step=5,
            image_id=5,
            dino_patch_features=features,
            ms_score=0.8,
            ms_short_score=0.8,
            ms_short_ratio=0.8,
            channel_support=0.1,
            provisional_channel_support=0.0,
            trigger_route=None,
            memory_patch_ids=[],
            lab_feature=torch.tensor(
                [20.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            historical_ms_threshold=1.0,
        )

        self.assertIsNone(admitted_expert_id)
        self.assertEqual(
            manager.last_novel_quarantine_event["decision"],
            "rejected_photometric_inconsistent",
        )
        self.assertEqual(manager.active_experts, [])

    def test_photometric_consistency_uses_only_historical_low_risk_members(
        self,
    ):
        manager = DynamicExpertManager(
            novel_admission_mode="quarantine",
            photometric_quarantine_steps=3,
            photometric_min_low_risk_members=2,
        )
        features = patch_features(1.0, 0.0)
        manager._advance_novel_quarantine(
            step=4,
            image_id=4,
            dino_patch_features=features,
            ms_score=1.2,
            ms_short_score=1.2,
            ms_short_ratio=1.0,
            channel_support=0.1,
            provisional_channel_support=0.0,
            trigger_route="photometric_drift",
            lab_feature=torch.tensor(
                [20.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            photometric_trigger_expert_id=0,
            photometric_trigger_lab_center=[
                40.0, 2.0, 1.0, 3.0, 2.0, 4.0
            ],
            historical_ms_threshold=1.0,
        )
        manager._advance_novel_quarantine(
            step=5,
            image_id=5,
            dino_patch_features=features,
            ms_score=0.8,
            ms_short_score=0.8,
            ms_short_ratio=1.0,
            channel_support=0.1,
            provisional_channel_support=0.0,
            trigger_route=None,
            lab_feature=torch.tensor(
                [60.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            historical_ms_threshold=1.0,
        )
        _, admitted_expert_id = manager._advance_novel_quarantine(
            step=6,
            image_id=6,
            dino_patch_features=features,
            ms_score=0.7,
            ms_short_score=0.7,
            ms_short_ratio=1.0,
            channel_support=0.1,
            provisional_channel_support=0.0,
            trigger_route=None,
            lab_feature=torch.tensor(
                [61.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            historical_ms_threshold=1.0,
        )

        self.assertIsNotNone(admitted_expert_id)
        self.assertEqual(
            manager.last_novel_quarantine_event["low_risk_image_ids"],
            [5, 6],
        )
        self.assertEqual(
            manager.last_novel_quarantine_event["photometric_profile"][
                "required_count"
            ],
            2,
        )

    def test_photometric_pool_opens_from_snapshot_after_expert_deleted(self):
        manager = DynamicExpertManager(
            novel_admission_mode="quarantine",
            photometric_quarantine_steps=2,
            photometric_min_low_risk_members=2,
        )
        features = patch_features(1.0, 0.0)
        manager._advance_novel_quarantine(
            step=4,
            image_id=4,
            dino_patch_features=features,
            ms_score=0.8,
            ms_short_score=0.8,
            ms_short_ratio=1.0,
            channel_support=0.1,
            provisional_channel_support=0.0,
            trigger_route="photometric_drift",
            lab_feature=torch.tensor(
                [60.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            photometric_trigger_expert_id=0,
            photometric_trigger_lab_center=[
                40.0, 2.0, 1.0, 3.0, 2.0, 4.0
            ],
            historical_ms_threshold=1.0,
        )
        self.assertEqual(manager.active_experts, [])
        self.assertIsNotNone(manager.photometric_trigger_lab_center)

        _, admitted_expert_id = manager._advance_novel_quarantine(
            step=5,
            image_id=5,
            dino_patch_features=features,
            ms_score=0.7,
            ms_short_score=0.7,
            ms_short_ratio=1.0,
            channel_support=0.1,
            provisional_channel_support=0.0,
            trigger_route=None,
            lab_feature=torch.tensor(
                [61.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            historical_ms_threshold=1.0,
        )

        self.assertIsNotNone(admitted_expert_id)
        self.assertEqual(
            manager.last_novel_quarantine_event["decision"],
            "admitted",
        )

    def test_candidate_pipeline_passes_lab_snapshot_after_expert_deleted(self):
        manager = DynamicExpertManager(
            novel_admission_mode="quarantine",
            photometric_candidate_enabled=True,
            photometric_support_confirm_steps=1,
            photometric_quarantine_steps=2,
        )
        lab_center = [40.0, 2.0, 1.0, 3.0, 2.0, 4.0]

        is_candidate, _, _, route = manager._consider_candidate(
            step=4,
            image_id=4,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.8,
            channel_support=0.1,
            ms_threshold=1.0,
            support_threshold=0.5,
            provisional_channel_support=0.0,
            novel_ms_threshold=None,
            novel_support_threshold=None,
            ms_short_score=0.8,
            ms_short_ratio=1.0,
            memory_patch_ids=[],
            lab_feature=torch.tensor(
                [60.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            lab_drift_profile={
                "expert_id": 0,
                "lab_center": lab_center,
                "l_ewma": 1.0,
                "ab_ewma": 0.0,
                "l_threshold": 0.1,
                "ab_threshold": 0.1,
                "drift_confirmed": True,
            },
            current_expert_supports={0: 0.1},
        )

        self.assertTrue(is_candidate)
        self.assertEqual(route, "photometric_drift")
        self.assertEqual(manager.active_experts, [])
        self.assertEqual(
            manager.photometric_trigger_lab_center.tolist(),
            lab_center,
        )

    def test_active_photometric_pool_continues_without_lab_profile(self):
        manager = DynamicExpertManager(
            novel_admission_mode="quarantine",
            photometric_candidate_enabled=True,
            photometric_relevant_history_enabled=True,
            photometric_min_low_risk_members=2,
        )
        features = patch_features(1.0, 0.0)
        manager._advance_novel_quarantine(
            step=4,
            image_id=4,
            dino_patch_features=features,
            ms_score=0.4,
            ms_short_score=0.4,
            ms_short_ratio=1.0,
            channel_support=0.1,
            provisional_channel_support=0.0,
            trigger_route="photometric_drift",
            lab_feature=torch.tensor(
                [60.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            photometric_trigger_expert_id=0,
            photometric_trigger_lab_center=[
                40.0, 2.0, 1.0, 3.0, 2.0, 4.0
            ],
            historical_ms_threshold=1.0,
        )

        is_candidate, _, _, route = manager._consider_candidate(
            step=5,
            image_id=5,
            dino_patch_features=features,
            ms_score=0.3,
            channel_support=0.1,
            ms_threshold=1.0,
            support_threshold=0.5,
            provisional_channel_support=0.0,
            novel_ms_threshold=None,
            novel_support_threshold=None,
            ms_short_score=0.3,
            ms_short_ratio=1.0,
            memory_patch_ids=[],
            lab_feature=torch.tensor(
                [61.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            lab_drift_profile=None,
            current_expert_supports={},
        )

        self.assertTrue(is_candidate)
        self.assertEqual(route, "photometric_drift")

    def test_quarantine_rejects_non_shift_pool(self):
        manager = DynamicExpertManager(
            candidate_mode="dual_path",
            novel_admission_mode="quarantine",
            novel_quarantine_steps=4,
            novel_quarantine_tail=2,
            novel_ratio_threshold=0.95,
        )
        manager.ms_history.extend([1.0] * 8)
        manager.support_history.extend([0.7, 0.8, 0.9])

        events = []
        for step in range(4):
            events.append(
                manager.advance(
                    step=step,
                    image_id=step,
                    dino_patch_features=patch_features(1.0 + step, 1.0),
                    ms_score=1.2 if step == 0 else 0.5,
                    channel_support=0.1,
                    expert_channel_supports={},
                    provisional_channel_support=0.2 if step == 0 else 0.0,
                    ms_short_score=1.0,
                    ms_short_ratio=1.01,
                )
            )

        self.assertEqual(events[-1]["novel_quarantine"]["decision"], "rejected")
        self.assertIsNone(events[-1]["admitted_expert_id"])
        self.assertEqual(manager.active_experts, [])

    def test_provisional_expert_is_historical_and_not_a_fuser_member(self):
        manager = DynamicExpertManager(
            candidate_mode="dual_path",
            novel_admission_mode="provisional",
            novel_quarantine_steps=4,
            novel_quarantine_tail=2,
            novel_ratio_threshold=0.95,
            novel_provisional_weight=0.25,
        )
        manager.ms_history.extend([1.0] * 8)
        manager.support_history.extend([0.7, 0.8, 0.9])

        first = manager.advance(
            step=0,
            image_id=10,
            dino_patch_features=patch_features(1.0, 1.0),
            ms_score=1.2,
            channel_support=0.1,
            expert_channel_supports={},
            provisional_channel_support=0.2,
            ms_short_score=1.1,
            ms_short_ratio=0.9,
        )
        snapshot = manager.provisional_expert_before_step()

        self.assertEqual(first["candidate_route"], "new_distribution")
        self.assertEqual(snapshot["image_id"], 10)
        self.assertEqual(snapshot["image_ids"], [10])
        self.assertEqual(snapshot["representative_step"], 0)
        self.assertEqual(snapshot["weight"], 0.25)
        self.assertEqual(manager.fuser_training_members(1), [])

        events = []
        for step, short_score in zip(range(1, 4), [1.0, 0.8, 0.9]):
            events.append(
                manager.advance(
                    step=step,
                    image_id=10 + step,
                    dino_patch_features=patch_features(1.0 + step, 1.0),
                    ms_score=0.5,
                    channel_support=0.1,
                    expert_channel_supports={},
                    provisional_channel_support=0.0,
                    ms_short_score=short_score,
                    ms_short_ratio=0.9,
                )
            )

        self.assertEqual(
            events[-1]["admission_route"],
            "new_distribution_provisional",
        )
        self.assertEqual(manager.active_experts[0]["image_id"], 12)
        self.assertIsNone(manager.provisional_expert_before_step())

    def test_soft_confirmation_stays_out_of_formal_committee(self):
        manager = DynamicExpertManager(
            candidate_mode="dual_path",
            novel_admission_mode="soft_confirmed",
            novel_quarantine_steps=4,
            novel_quarantine_tail=2,
            novel_ratio_threshold=0.95,
            novel_provisional_weight=0.25,
            novel_confirmed_weight=0.5,
            novel_confirmed_ttl=2,
            novel_confirmed_members=2,
        )
        manager.ms_history.extend([1.0] * 8)
        manager.support_history.extend([0.7, 0.8, 0.9])

        events = []
        for step, short_score in enumerate([1.1, 1.0, 0.8, 0.9]):
            events.append(
                manager.advance(
                    step=step,
                    image_id=10 + step,
                    dino_patch_features=patch_features(1.0 + step, 1.0),
                    ms_score=1.2 if step == 0 else 0.5,
                    channel_support=0.1,
                    expert_channel_supports={},
                    provisional_channel_support=0.2 if step == 0 else 0.0,
                    ms_short_score=short_score,
                    ms_short_ratio=0.9,
                )
            )

        snapshot = manager.provisional_expert_before_step()
        self.assertEqual(
            events[-1]["novel_quarantine"]["decision"],
            "confirmed_soft",
        )
        self.assertEqual(snapshot["image_id"], 12)
        self.assertEqual(snapshot["image_ids"], [12, 13])
        self.assertEqual(snapshot["weight"], 0.5)
        self.assertEqual(snapshot["state"], "confirmed")
        self.assertEqual(
            events[-1]["novel_quarantine"]["confirmed_image_ids"],
            [12, 13],
        )
        self.assertEqual(manager.active_experts, [])
        self.assertEqual(manager.fuser_training_members(4), [])

        for step in range(4, 6):
            manager.advance(
                step=step,
                image_id=10 + step,
                dino_patch_features=patch_features(1.0 + step, 1.0),
                ms_score=0.5,
                channel_support=0.1,
                expert_channel_supports={},
                provisional_channel_support=0.0,
                ms_short_score=0.9,
                ms_short_ratio=0.9,
            )
        self.assertIsNone(manager.provisional_expert_before_step())

    def test_channel_irbank_rebuilds_from_lower_half_and_refreshes_ttl(self):
        manager = DynamicExpertManager(
            candidate_mode="dual_path",
            novel_admission_mode="channel_irbank",
            novel_quarantine_steps=4,
            novel_quarantine_tail=4,
            novel_ratio_threshold=0.95,
            novel_confirmed_ttl=2,
            novel_memory_window=6,
            novel_memory_keep_fraction=0.5,
        )
        manager.ms_history.extend([1.0] * 8)
        manager.support_history.extend([0.7, 0.8, 0.9])

        events = []
        for step, short_score in enumerate([1.1, 1.0, 0.8, 0.9]):
            events.append(
                manager.advance(
                    step=step,
                    image_id=10 + step,
                    dino_patch_features=patch_features(1.0 + step, 1.0),
                    ms_score=1.2 if step == 0 else 0.5,
                    channel_support=0.1,
                    expert_channel_supports={},
                    provisional_channel_support=0.2 if step == 0 else 0.0,
                    ms_short_score=short_score,
                    ms_short_ratio=0.9,
                    memory_patch_ids=[0],
                )
            )

        snapshot = manager.provisional_expert_before_step()
        self.assertEqual(
            events[-1]["novel_quarantine"]["decision"],
            "confirmed_irbank",
        )
        self.assertEqual(snapshot["image_ids"], [12, 13])
        self.assertEqual(snapshot["state"], "confirmed_irbank")
        self.assertEqual(
            snapshot["memory_members"],
            [
                {"image_id": 12, "patch_ids": [0]},
                {"image_id": 13, "patch_ids": [0]},
            ],
        )
        self.assertEqual(manager.active_experts, [])

        manager.advance(
            step=4,
            image_id=14,
            dino_patch_features=patch_features(5.0, 1.0),
            ms_score=0.5,
            channel_support=0.1,
            expert_channel_supports={},
            provisional_channel_support=0.0,
            ms_short_score=0.1,
            ms_short_ratio=0.9,
            memory_patch_ids=[1],
        )
        refreshed = manager.provisional_expert_before_step()
        self.assertEqual(refreshed["image_ids"], [14, 12, 13])
        self.assertEqual(manager.confirmed_provisional["ttl"], 2)

    def test_channel_gate_admits_formal_expert_after_confirmation(self):
        manager = DynamicExpertManager(
            candidate_mode="dual_path",
            novel_admission_mode="channel_gate",
            novel_quarantine_steps=4,
            novel_quarantine_tail=4,
            novel_ratio_threshold=0.95,
            novel_memory_window=6,
            novel_memory_keep_fraction=0.5,
        )
        manager.ms_history.extend([1.0] * 8)
        manager.support_history.extend([0.7, 0.8, 0.9])

        events = []
        for step, short_score in enumerate([1.1, 1.0, 0.8, 0.9]):
            events.append(
                manager.advance(
                    step=step,
                    image_id=10 + step,
                    dino_patch_features=patch_features(1.0 + step, 1.0),
                    ms_score=1.2 if step == 0 else 0.5,
                    channel_support=0.1,
                    expert_channel_supports={},
                    provisional_channel_support=0.2 if step == 0 else 0.0,
                    ms_short_score=short_score,
                    ms_short_ratio=0.9,
                    memory_patch_ids=[0],
                )
            )

        self.assertEqual(
            events[-1]["novel_quarantine"]["decision"],
            "confirmed_channel_gate",
        )
        self.assertEqual(events[-1]["admitted_expert_id"], 0)
        self.assertEqual(events[-1]["admission_route"], "new_distribution_channel_gate")
        self.assertEqual(manager.active_experts[0]["image_id"], 12)
        self.assertIsNone(manager.provisional_expert_before_step())

    def test_irbank_min_keep_preserves_two_reliable_members(self):
        manager = DynamicExpertManager(novel_memory_keep_fraction=0.5)
        pool = [
            {
                "image_id": 1,
                "ms_short_score": 0.2,
                "memory_patch_ids": [0],
            },
            {
                "image_id": 2,
                "ms_short_score": 0.1,
                "memory_patch_ids": [1],
            },
            {
                "image_id": 3,
                "ms_short_score": 0.0,
                "memory_patch_ids": [],
            },
        ]

        default_selected = manager._select_irbank_members(pool)
        drift_selected = manager._select_irbank_members(pool, min_keep=2)

        self.assertEqual(
            [member["image_id"] for member in default_selected],
            [2],
        )
        self.assertEqual(
            [member["image_id"] for member in drift_selected],
            [2, 1],
        )

    def test_adaptive_evidence_stages_first_joint_state_as_temporary(self):
        manager = DynamicExpertManager(
            photometric_adaptive_candidate_evidence=True,
            budget_min_new_coverage_fraction=0.5,
        )
        manager.active_experts = [{"expert_id": 4}]
        members = [
            {
                "image_id": image_id,
                "embedding": torch.tensor([1.0, 0.0]),
                "memory_patch_ids": [image_id],
                "historical_low_risk": True,
                "historical_ms_threshold": 0.5,
                "ms_score": 0.1,
                "lab_feature": torch.tensor(
                    [80.0, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                "current_expert_supports": {4: 0.1},
            }
            for image_id in (10, 11)
        ]
        profile = {
            "consistent": True,
            "selected_inlier_image_ids": [10, 11],
        }

        evidence = manager._candidate_admission_evidence(
            members,
            profile,
            support_threshold=0.5,
        )

        self.assertEqual(evidence["decision"], "new_temporary_state")
        self.assertEqual(evidence["new_coverage_fraction"], 1.0)
        self.assertEqual(evidence["strict_low_risk_count"], 2)

    def test_adaptive_evidence_stages_first_covered_state_as_temporary(self):
        manager = DynamicExpertManager(
            photometric_adaptive_candidate_evidence=True,
            budget_min_new_coverage_fraction=0.5,
        )
        manager.active_experts = [{"expert_id": 4}]
        members = [
            {
                "image_id": image_id,
                "embedding": torch.tensor([1.0, 0.0]),
                "memory_patch_ids": [image_id],
                "historical_low_risk": True,
                "historical_ms_threshold": 0.5,
                "ms_score": 0.1,
                "lab_feature": torch.tensor(
                    [80.0, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                "current_expert_supports": {4: 0.9},
            }
            for image_id in (10, 11)
        ]
        profile = {
            "consistent": True,
            "selected_inlier_image_ids": [10, 11],
        }

        evidence = manager._candidate_admission_evidence(
            members,
            profile,
            support_threshold=0.5,
        )

        self.assertEqual(evidence["decision"], "new_temporary_state")
        self.assertEqual(evidence["new_coverage_fraction"], 0.0)

    def test_fitness_denstream_active_match_uses_linked_temporary_state(self):
        def decision(split_enabled):
            manager = DynamicExpertManager(
                cold_start_lab_enabled=True,
                photometric_candidate_window_mode="history",
                photometric_fitness_history_enabled=True,
                photometric_fitness_adaptive_evidence_enabled=(
                    split_enabled
                ),
                photometric_adaptive_candidate_evidence=True,
                photometric_state_distribution_enabled=True,
                photometric_temporary_state_enabled=True,
                photometric_denstream_support_enabled=split_enabled,
                photometric_history_reentry_enabled=True,
                budget_min_new_coverage_fraction=0.5,
                budget_unified_delta_j_dino_score_enabled=split_enabled,
            )
            anchor = manager._quarantine_member(
                step=0,
                image_id=0,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.8,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [80.0, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
            )
            anchor_peer = manager._quarantine_member(
                step=1,
                image_id=1,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.8,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [80.0, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
            )
            manager._admit_quarantined_member(
                anchor,
                0,
                cluster_members=[anchor, anchor_peer],
                admission_route="photometric_drift_channel_gate",
            )
            members = []
            for image_id in (10, 11):
                member = manager._quarantine_member(
                    step=image_id,
                    image_id=image_id,
                    dino_patch_features=patch_features(1.0, 0.0),
                    ms_score=0.1,
                    ms_short_score=0.1,
                    ms_short_ratio=0.8,
                    channel_support=0.9,
                    provisional_channel_support=0.5,
                    memory_patch_ids=[0],
                    lab_feature=torch.tensor(
                        [80.0, 2.0, 1.0, 3.0, 2.0, 4.0]
                    ),
                    historical_ms_threshold=0.5,
                    current_expert_supports={0: 0.9},
                )
                members.append(member)
            evidence = manager._candidate_admission_evidence(
                members,
                {
                    "consistent": True,
                    "selected_inlier_image_ids": [10, 11],
                },
                support_threshold=0.5,
            )
            return manager, evidence

        _, old = decision(False)
        manager, new = decision(True)
        self.assertEqual(old["decision"], "existing_state_supported", old)
        self.assertEqual(new["decision"], "new_temporary_state", new)
        self.assertEqual(new["new_coverage_fraction"], 0.0)
        self.assertEqual(new["linked_active_state_id"], 0)

        active_center = manager.historical_states[0]["lab_center"].copy()
        other_active = dict(manager.historical_states[0])
        other_active.update(
            {
                "state_id": 1,
                "source_expert_id": 1,
                "active_expert_id": 1,
                "cluster_id": 1,
            }
        )
        manager.historical_states.append(other_active)
        manager.next_historical_state_id = 2
        recorded = manager._record_photometric_state(
            new["selected_members"],
            step=11,
            state_status="temporary",
            source_type="temporary",
            temporary_support_evidence=new["temporary_support_evidence"],
            linked_active_state_id=new["linked_active_state_id"],
        )

        self.assertEqual(len(manager.historical_states), 3)
        self.assertEqual(
            recorded["linked_active_state_id"],
            new["linked_active_state_id"],
        )
        self.assertTrue(
            np.array_equal(
                manager.historical_states[0]["lab_center"], active_center
            )
        )

    def test_adaptive_pool_materializes_channel_novel_state_after_two_events(self):
        manager = DynamicExpertManager(
            novel_admission_mode="channel_gate",
            photometric_candidate_enabled=True,
            photometric_candidate_window_mode="dynamic_low_risk",
            photometric_dynamic_min_low_risk_members=3,
            photometric_dynamic_max_low_risk_members=6,
            photometric_min_low_risk_members=3,
            photometric_adaptive_candidate_evidence=True,
            photometric_episode_enabled=True,
        )
        manager.active_experts = [{"expert_id": 0}]
        manager.next_expert_id = 1
        manager.photometric_episode_active = True
        features = patch_features(1.0, 0.0)

        admitted_id = None
        for step in range(4):
            _, admitted_id = manager._advance_novel_quarantine(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=1.2,
                channel_support=0.1,
                provisional_channel_support=0.5,
                trigger_route=(
                    "photometric_drift" if step in {0, 2} else None
                ),
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [80.0, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                photometric_trigger_expert_id=0,
                photometric_trigger_lab_center=[
                    40.0, 2.0, 1.0, 3.0, 2.0, 4.0
                ],
                historical_ms_threshold=0.5,
                current_expert_supports={0: 0.1},
                support_threshold=0.5,
            )

        self.assertEqual(
            admitted_id,
            1,
            manager.last_novel_quarantine_event,
        )
        self.assertEqual(
            manager.last_novel_quarantine_event["confirmation_reason"],
            "new_scoring_expert",
        )
        self.assertEqual(len(manager.historical_states), 1)
        self.assertEqual(
            manager.historical_states[0]["state_status"], "stable"
        )

    def test_fitness_history_can_reuse_adaptive_candidate_evidence(self):
        manager = DynamicExpertManager(
            novel_admission_mode="channel_gate",
            photometric_candidate_enabled=True,
            photometric_candidate_window_mode="history",
            photometric_fitness_history_enabled=True,
            photometric_fitness_adaptive_evidence_enabled=True,
            photometric_dynamic_min_low_risk_members=3,
            photometric_dynamic_max_low_risk_members=6,
            photometric_min_low_risk_members=3,
            photometric_adaptive_candidate_evidence=True,
            photometric_episode_enabled=True,
        )
        manager.active_experts = [{"expert_id": 0}]
        manager.next_expert_id = 1
        manager.photometric_episode_active = True
        features = patch_features(1.0, 0.0)

        admitted_id = None
        for step in range(4):
            _, admitted_id = manager._advance_novel_quarantine(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=1.2,
                channel_support=0.1,
                provisional_channel_support=0.5,
                trigger_route=(
                    "photometric_drift" if step in {0, 2} else None
                ),
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [80.0, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                photometric_trigger_expert_id=0,
                photometric_trigger_lab_center=[
                    40.0, 2.0, 1.0, 3.0, 2.0, 4.0
                ],
                historical_ms_threshold=0.5,
                current_expert_supports={0: 0.1},
                support_threshold=0.5,
            )

        self.assertEqual(
            admitted_id,
            1,
            manager.last_novel_quarantine_event,
        )
        self.assertEqual(
            manager.last_novel_quarantine_event["confirmation_reason"],
            "new_scoring_expert",
        )

    def test_adaptive_pool_records_covered_state_without_expert(self):
        manager = DynamicExpertManager(
            novel_admission_mode="channel_gate",
            photometric_candidate_enabled=True,
            photometric_candidate_window_mode="dynamic_low_risk",
            photometric_dynamic_min_low_risk_members=3,
            photometric_dynamic_max_low_risk_members=6,
            photometric_min_low_risk_members=3,
            photometric_adaptive_candidate_evidence=True,
            photometric_episode_enabled=True,
        )
        manager.active_experts = [{"expert_id": 0}]
        manager.next_expert_id = 1
        manager.photometric_episode_active = True
        features = patch_features(1.0, 0.0)

        admitted_id = None
        for step in range(4):
            _, admitted_id = manager._advance_novel_quarantine(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=1.2,
                channel_support=0.9,
                provisional_channel_support=0.5,
                trigger_route=(
                    "photometric_drift" if step in {0, 2} else None
                ),
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [80.0, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                photometric_trigger_expert_id=0,
                photometric_trigger_lab_center=[
                    40.0, 2.0, 1.0, 3.0, 2.0, 4.0
                ],
                historical_ms_threshold=0.5,
                current_expert_supports={0: 0.9},
                support_threshold=0.5,
            )

        self.assertIsNone(admitted_id)
        self.assertEqual(
            manager.last_novel_quarantine_event["decision"],
            "matured_photometric_state_only",
        )
        self.assertEqual(len(manager.historical_states), 1)
        self.assertEqual(
            manager.historical_states[-1]["source_type"],
            "photometric_only",
        )
        self.assertEqual(len(manager.active_experts), 1)

    def test_denstream_maturity_does_not_bypass_formal_admission(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            novel_admission_mode="channel_gate",
            cold_start_lab_enabled=True,
            photometric_candidate_enabled=True,
            photometric_candidate_window_mode="dynamic_low_risk",
            photometric_dynamic_min_low_risk_members=3,
            photometric_dynamic_max_low_risk_members=6,
            photometric_min_low_risk_members=3,
            photometric_adaptive_candidate_evidence=True,
            photometric_denstream_support_enabled=True,
            photometric_denstream_decay=0.9,
            photometric_denstream_mature_support=0.5,
            photometric_episode_enabled=True,
        )
        manager.active_experts = [{"expert_id": 0}]
        manager.next_expert_id = 1
        manager.photometric_episode_active = True
        features = patch_features(1.0, 0.0)

        admitted_id = None
        for step in range(2):
            _, admitted_id = manager._advance_novel_quarantine(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=1.2,
                channel_support=0.9,
                provisional_channel_support=0.5,
                trigger_route=("photometric_drift" if step == 0 else None),
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [80.0, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                photometric_trigger_expert_id=0,
                photometric_trigger_lab_center=[
                    40.0, 2.0, 1.0, 3.0, 2.0, 4.0
                ],
                historical_ms_threshold=0.5,
                current_expert_supports={0: 0.9},
                support_threshold=0.5,
            )

        self.assertIsNone(admitted_id)
        self.assertEqual(len(manager.active_experts), 1)
        self.assertEqual(len(manager.historical_states), 1)
        self.assertEqual(
            manager.historical_states[0]["state_status"], "stable"
        )
        self.assertFalse(
            manager.last_novel_quarantine_event["created_full_expert"]
        )
        self.assertTrue(
            any(
                event["event_type"] == "mature"
                and event["reason"]
                == "denstream_effective_support_matured"
                for event in manager.state_lifecycle_events
            )
        )

    def test_fitness_denstream_maturity_reuses_formal_admission(self):
        def exercise(reject):
            manager = DynamicExpertManager(
                committee_cap=2,
                novel_admission_mode="channel_gate",
                cold_start_lab_enabled=True,
                photometric_candidate_enabled=True,
                photometric_candidate_window_mode="history",
                photometric_fitness_history_enabled=True,
                photometric_fitness_adaptive_evidence_enabled=True,
                photometric_adaptive_candidate_evidence=True,
                photometric_state_distribution_enabled=True,
                photometric_temporary_state_enabled=True,
                photometric_denstream_support_enabled=True,
                photometric_denstream_mature_support=0.5,
                budget_unified_delta_j_dino_score_enabled=True,
            )
            manager.active_experts = [{"expert_id": 0}]
            manager.next_expert_id = 1
            admission_calls = []

            def candidate_evidence(candidate_members, *_args, **_kwargs):
                temporary = next(
                    (
                        state
                        for state in manager.historical_states
                        if state.get("state_status") == "temporary"
                    ),
                    None,
                )
                return {
                    "decision": (
                        "update_temporary_state"
                        if temporary is not None
                        else "new_temporary_state"
                    ),
                    "selected_members": list(candidate_members),
                    "has_reliable_patch_memory": True,
                    "channel_novel": True,
                    "fitness_denstream_delta_j_enabled": True,
                    "linked_active_state_id": None,
                    "history_match": (
                        {
                            "matched_state_id": temporary["state_id"],
                            "matched_profile": {
                                "state_status": "temporary"
                            },
                        }
                        if temporary is not None
                        else {}
                    ),
                    "temporary_support_evidence": {
                        "increment": 0.6,
                        "dino_evidence": 1.0,
                        "channel_evidence": 0.8,
                        "risk_evidence": 0.75,
                    },
                }

            def admit(member, step, **kwargs):
                admission_calls.append(
                    {
                        "member": member,
                        "step": step,
                        **kwargs,
                    }
                )
                if reject:
                    manager.last_budget_replacement_evaluation = {
                        "outcome": "unified_delta_j_rejected"
                    }
                    return None, None
                return 1, 1

            manager._candidate_admission_evidence = candidate_evidence
            manager._admit_quarantined_member = admit
            features = patch_features(1.0, 0.0)
            for step in range(4):
                manager._advance_novel_quarantine(
                    step=step,
                    image_id=step,
                    dino_patch_features=features,
                    ms_score=0.1,
                    ms_short_score=0.1,
                    ms_short_ratio=1.2,
                    channel_support=0.8,
                    provisional_channel_support=0.5,
                    trigger_route=(
                        "photometric_drift" if step in {0, 2} else None
                    ),
                    memory_patch_ids=[0],
                    lab_feature=torch.tensor(
                        [80.0, 2.0, 1.0, 3.0, 2.0, 4.0]
                    ),
                    photometric_trigger_expert_id=0,
                    photometric_trigger_lab_center=[
                        40.0, 2.0, 1.0, 3.0, 2.0, 4.0
                    ],
                    historical_ms_threshold=0.5,
                    current_expert_supports={0: 0.1},
                    support_threshold=0.5,
                )
            return manager, admission_calls

        accepted, accepted_calls = exercise(reject=False)
        self.assertEqual(len(accepted_calls), 1)
        self.assertEqual(len(accepted.historical_states), 1)
        self.assertEqual(len(accepted.active_experts), 1)
        self.assertEqual(
            accepted_calls[0]["candidate_evidence"]["associated_state_id"],
            accepted.historical_states[0]["state_id"],
        )
        self.assertTrue(
            accepted.last_novel_quarantine_event["created_full_expert"]
        )
        self.assertTrue(
            any(
                event["event_type"] == "create"
                and event["status_after"] == "temporary"
                for event in accepted.state_lifecycle_events
            )
        )

        rejected, rejected_calls = exercise(reject=True)
        self.assertEqual(len(rejected_calls), 1)
        self.assertEqual(len(rejected.historical_states), 1)
        self.assertEqual(len(rejected.active_experts), 1)
        self.assertEqual(
            rejected.last_novel_quarantine_event["decision"],
            "retained_history_no_safe_swap",
        )
        self.assertEqual(
            rejected.last_novel_quarantine_event["historical_state"][
                "historical_state_id"
            ],
            rejected_calls[0]["candidate_evidence"]["associated_state_id"],
        )

    def test_historical_state_uses_robust_distribution(self):
        manager = DynamicExpertManager(
            cold_start_lab_enabled=True,
            photometric_history_reentry_members=2,
            photometric_adaptive_candidate_evidence=True,
        )

        def member(step, value):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [value, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports={},
            )

        manager._record_photometric_state(
            [member(0, 60.0), member(1, 61.0), member(2, 80.0)],
            step=2,
            state_status="temporary",
            source_type="temporary",
        )
        state = manager.historical_states[0]
        match = manager._historical_state_match(
            [member(3, 64.0), member(4, 65.0)]
        )

        self.assertEqual(state["lab_center"][0], 61.0)
        self.assertGreater(state["l_radius"], manager.photometric_l_ewma_floor)
        self.assertEqual(state["member_image_ids"], [0, 1, 2])
        self.assertTrue(match["matched"])
        self.assertEqual(match["matched_source_type"], "temporary")

    def test_historical_state_match_is_unified_across_source_types(self):
        source_types = ("active", "retired", "photometric_only", "temporary")
        for index, source_type in enumerate(source_types):
            manager = DynamicExpertManager(
                cold_start_lab_enabled=True,
                photometric_history_reentry_members=2,
                photometric_adaptive_candidate_evidence=True,
            )
            members = []
            for offset in range(2):
                members.append(
                    manager._quarantine_member(
                        step=offset,
                        image_id=offset,
                        dino_patch_features=patch_features(1.0, 0.0),
                        ms_score=0.1,
                        ms_short_score=0.1,
                        ms_short_ratio=0.8,
                        channel_support=0.2,
                        provisional_channel_support=0.5,
                        memory_patch_ids=[0],
                        lab_feature=torch.tensor(
                            [70.0 + offset, 2.0, 1.0, 3.0, 2.0, 4.0]
                        ),
                        historical_ms_threshold=0.5,
                        current_expert_supports={},
                    )
                )
            manager._record_photometric_state(
                members,
                step=1,
                state_status=(
                    "temporary" if source_type == "temporary" else "stable"
                ),
                source_type=source_type,
            )
            match = manager._historical_state_match(members)
            self.assertTrue(match["matched"], msg=source_type)
            self.assertEqual(match["matched_source_type"], source_type)

    def test_matching_active_state_enriches_cluster_without_new_expert(self):
        manager = DynamicExpertManager(
            cold_start_lab_enabled=True,
            photometric_history_reentry_members=2,
            photometric_adaptive_candidate_evidence=True,
        )

        def member(step, value, support):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=support,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [value, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports={0: support},
            )

        initial = [member(0, 70.0, 0.1), member(1, 71.0, 0.1)]
        manager._admit_quarantined_member(
            initial[0],
            step=1,
            cluster_members=initial,
            admission_route="photometric_drift_channel_gate",
        )
        support_count_before = manager.active_experts[0]["support_count"]
        last_supported_before = manager.active_experts[0][
            "last_supported_step"
        ]
        del manager.clusters[0]["members"][0]["lab_feature"]
        candidates = [member(2, 72.0, 0.1), member(3, 73.0, 0.1)]
        evidence = manager._candidate_admission_evidence(
            candidates,
            {
                "consistent": True,
                "selected_inlier_image_ids": [2, 3],
            },
            support_threshold=0.5,
        )
        update = manager._update_matched_historical_state(
            evidence["history_match"], candidates, step=3
        )

        self.assertEqual(evidence["decision"], "update_existing_state")
        self.assertEqual(len(manager.active_experts), 1)
        self.assertEqual(len(manager.clusters[0]["members"]), 4)
        self.assertEqual(update["members_added_to_active_cluster"], 2)
        self.assertEqual(
            manager.active_experts[0]["support_count"], support_count_before
        )
        self.assertEqual(
            manager.active_experts[0]["last_supported_step"],
            last_supported_before,
        )
        self.assertEqual(
            manager.active_experts[0]["lab_state"]["center"][0], 72.0
        )

    def test_relevant_lab_history_returns_limited_robust_suffix(self):
        manager = DynamicExpertManager(
            cold_start_mode="adaptive",
            cold_start_min_samples=4,
            cold_start_max_samples=4,
            cold_start_required_members=3,
            cold_start_lab_enabled=True,
            photometric_relevant_history_enabled=True,
            photometric_history_min=3,
            photometric_history_max=5,
        )
        features = patch_features(1.0, 0.0)
        for step in range(4):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=None if step == 0 else 0.1,
                channel_support=None if step == 0 else 1.0,
                expert_channel_supports={},
                lab_feature=torch.tensor(
                    [40.0 + step, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
            )

        profile = None
        for step in range(4, 10):
            profile = manager.advance_lab_state(
                torch.tensor(
                    [44.0 + step, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                step=step,
                image_id=step,
            )

        self.assertTrue(profile["relevant_history_compatible"])
        self.assertLessEqual(profile["relevant_history_size"], 5)
        self.assertEqual(
            profile["relevant_image_ids"][-1],
            9,
        )
        self.assertEqual(
            len(manager.active_experts[0]["lab_state"]["recent_lab_history"]),
            5,
        )

    def test_relevant_photometric_profile_uses_lab_and_dino_support(self):
        manager = DynamicExpertManager(
            photometric_relevant_history_enabled=True,
            photometric_history_distance_threshold=3.0,
            photometric_dino_similarity_floor=0.9,
        )
        features = patch_features(1.0, 0.0)
        pool = [
            manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [60.0 + step, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
            )
            for step in range(3)
        ]
        profile = manager._photometric_pool_profile(
            pool,
            trigger_expert_id=0,
            trigger_lab_center=[40.0, 2.0, 1.0, 3.0, 2.0, 4.0],
        )

        self.assertTrue(profile["consistent"])
        self.assertEqual(profile["lab_inlier_count"], 3)
        self.assertEqual(profile["dino_inlier_count"], 3)

    def test_photometric_admission_replaces_most_redundant_expert(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            photometric_relevant_history_enabled=True,
        )
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            memory_patch_ids=[0],
            historical_ms_threshold=0.5,
        )
        second = {
            **first,
            "step": 1,
            "image_id": 1,
            "embedding": manager._image_embedding(
                patch_features(0.0, 1.0)
            ),
        }
        manager._admit_quarantined_member(first, 0, admission_route="legacy")
        manager._admit_quarantined_member(second, 1, admission_route="legacy")
        manager.coverage_history = [
            {
                "step": 2,
                "support_threshold": 0.5,
                "supports": {0: 0.9, 1: 0.9},
            },
            {
                "step": 3,
                "support_threshold": 0.5,
                "supports": {0: 0.9, 1: 0.1},
            },
        ]
        candidate_a = {
            **first,
            "step": 4,
            "image_id": 4,
            "current_expert_supports": {0: 0.9, 1: 0.1},
        }
        candidate_b = {
            **candidate_a,
            "step": 5,
            "image_id": 5,
            "current_expert_supports": {0: 0.1, 1: 0.1},
        }

        _, expert_id = manager._admit_quarantined_member(
            candidate_a,
            step=5,
            cluster_members=[candidate_a, candidate_b],
            admission_route="photometric_drift_channel_gate",
            support_threshold=0.5,
        )

        self.assertEqual(expert_id, 2)
        self.assertEqual(len(manager.active_experts), 2)
        self.assertEqual(
            [expert["expert_id"] for expert in manager.active_experts],
            [0, 2],
        )
        self.assertEqual(manager.retired_experts[-1]["expert_id"], 1)
        self.assertEqual(
            manager.retired_experts[-1]["last_event"],
            "replaced_coverage_budget",
        )
        self.assertEqual(
            manager.last_budget_replacement["reason"],
            "coverage_budget_replacement",
        )
        self.assertAlmostEqual(
            manager.last_budget_replacement["new_coverage_fraction"],
            0.5,
        )
        self.assertEqual(
            manager.last_budget_replacement_evaluation["outcome"],
            "accepted",
        )

    def test_coreset_replacement_minimizes_recent_coverage_loss(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            budget_coverage_loss_enabled=True,
            budget_coverage_channel_weight=1.0,
            budget_coverage_lab_weight=0.0,
            budget_coverage_dino_weight=0.0,
        )
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        second = {
            **first,
            "step": 1,
            "image_id": 1,
            "embedding": manager._image_embedding(
                patch_features(0.0, 1.0)
            ),
        }
        manager._admit_quarantined_member(first, 0, admission_route="legacy")
        manager._admit_quarantined_member(second, 1, admission_route="legacy")
        manager.coverage_history = [
            {
                "step": 2,
                "image_id": 2,
                "support_threshold": 0.5,
                "supports": {0: 0.9, 1: 0.9},
                "embedding": None,
                "lab_feature": None,
            },
            {
                "step": 3,
                "image_id": 3,
                "support_threshold": 0.5,
                "supports": {0: 0.9, 1: 0.1},
                "embedding": None,
                "lab_feature": None,
            },
        ]
        candidates = [
            {
                **first,
                "step": step,
                "image_id": step,
                "current_expert_supports": {0: 0.1, 1: 0.1},
            }
            for step in (4, 5)
        ]

        victim_id, audit = manager._select_budget_replacement(
            candidates,
            support_threshold=0.5,
            admission_route="legacy",
        )

        self.assertEqual(victim_id, 1)
        self.assertEqual(
            audit["replacement_reason"],
            "coreset_coverage_replacement",
        )
        evaluation = manager.last_budget_replacement_evaluation
        self.assertEqual(evaluation["outcome"], "accepted")
        self.assertGreater(evaluation["coverage_swap_gain"], 0.0)
        self.assertLessEqual(
            evaluation["coverage_loss_profiles"][1]["removal_loss"],
            evaluation["coverage_loss_profiles"][0]["removal_loss"],
        )
        self.assertEqual(
            set(evaluation["coverage_loss_profiles"][0]),
            {
                "current_coverage",
                "coverage_without_expert",
                "coverage_with_candidate",
                "removal_loss",
                "swap_gain",
            },
        )

    def test_unified_delta_j_shadow_does_not_change_replacement(self):
        def run(shadow_enabled):
            manager = DynamicExpertManager(
                committee_cap=2,
                budget_coverage_loss_enabled=True,
                budget_coverage_channel_weight=1.0,
                budget_coverage_lab_weight=0.0,
                budget_coverage_dino_weight=0.0,
                budget_unified_delta_j_shadow_enabled=shadow_enabled,
            )
            first = manager._quarantine_member(
                step=0,
                image_id=0,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.8,
                provisional_channel_support=0.5,
                historical_ms_threshold=0.5,
            )
            second = {
                **first,
                "step": 1,
                "image_id": 1,
                "embedding": manager._image_embedding(
                    patch_features(0.0, 1.0)
                ),
            }
            manager._admit_quarantined_member(
                first, 0, admission_route="legacy"
            )
            manager._admit_quarantined_member(
                second, 1, admission_route="legacy"
            )
            manager.coverage_history = [
                {
                    "step": 2,
                    "support_threshold": 0.5,
                    "supports": {0: 0.9, 1: 0.9},
                },
                {
                    "step": 3,
                    "support_threshold": 0.5,
                    "supports": {0: 0.9, 1: 0.1},
                },
            ]
            candidates = [
                {
                    **first,
                    "step": step,
                    "image_id": step,
                    "current_expert_supports": {0: 0.1, 1: 0.1},
                }
                for step in (4, 5)
            ]
            _, admitted_id = manager._admit_quarantined_member(
                candidates[0],
                step=5,
                cluster_members=candidates,
                admission_route="legacy",
                support_threshold=0.5,
            )
            return manager, admitted_id

        baseline, baseline_admitted = run(False)
        shadow, shadow_admitted = run(True)

        self.assertEqual(shadow_admitted, baseline_admitted)
        self.assertEqual(
            [expert["expert_id"] for expert in shadow.active_experts],
            [expert["expert_id"] for expert in baseline.active_experts],
        )
        self.assertEqual(
            shadow.last_budget_replacement_evaluation["outcome"],
            baseline.last_budget_replacement_evaluation["outcome"],
        )
        audit = shadow.last_budget_replacement_evaluation[
            "unified_delta_j_shadow"
        ]
        self.assertEqual(
            audit["formula"],
            "coverage_gain - pollution_penalty - "
            "redundancy_penalty - normal_score_penalty + "
            "reliability_eviction_gain",
        )
        self.assertIn("candidate_gain", audit)
        self.assertIn("victim_profiles", audit)
        self.assertEqual(
            audit["existing_selected_victim_expert_id"],
            shadow.last_budget_replacement["expert_id"],
        )
        self.assertIsInstance(audit["agrees_with_existing_decision"], bool)
        self.assertTrue(audit["agrees_with_existing_decision"])

    def test_unified_delta_j_shadow_audits_redundant_free_slot_admission(self):
        def run(shadow_enabled, control_enabled=False):
            manager = DynamicExpertManager(
                committee_cap=2,
                budget_virtual_swap_enabled=True,
                budget_unified_delta_j_shadow_enabled=shadow_enabled,
                budget_unified_delta_j_enabled=control_enabled,
            )
            first = manager._quarantine_member(
                step=0,
                image_id=0,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.8,
                provisional_channel_support=0.5,
                historical_ms_threshold=0.5,
            )
            manager._admit_quarantined_member(
                first, 0, admission_route="legacy"
            )
            manager.coverage_history = [
                {
                    "step": 0,
                    "image_id": 0,
                    "support_threshold": 0.5,
                    "supports": {0: 0.9},
                    "embedding": first["embedding"],
                    "patch_prototypes": first["patch_prototypes"],
                    "lab_feature": None,
                }
            ]
            second = {
                **first,
                "step": 1,
                "image_id": 1,
                "current_expert_supports": {0: 0.9},
            }
            _, admitted_id = manager._admit_quarantined_member(
                second,
                1,
                admission_route="legacy",
                support_threshold=0.5,
            )
            return manager, admitted_id

        baseline, baseline_admitted = run(False)
        shadow, shadow_admitted = run(True)

        self.assertEqual(shadow_admitted, baseline_admitted)
        self.assertEqual(
            [expert["expert_id"] for expert in shadow.active_experts],
            [expert["expert_id"] for expert in baseline.active_experts],
        )
        evaluation = shadow.last_budget_replacement_evaluation
        self.assertEqual(evaluation["outcome"], "free_slot_admitted")
        audit = evaluation["unified_delta_j_shadow"]
        self.assertEqual(audit["formal_change_type"], "free_slot_admission")
        self.assertEqual(audit["actual_decision"], "accept")
        self.assertFalse(audit["would_accept"])
        self.assertFalse(audit["agrees_with_existing_decision"])
        self.assertEqual(audit["coverage_gain"], 0.0)

        controlled, controlled_admitted = run(True, True)
        self.assertIsNone(controlled_admitted)
        self.assertEqual(len(controlled.active_experts), 1)
        controlled_evaluation = controlled.last_budget_replacement_evaluation
        self.assertEqual(
            controlled_evaluation["outcome"], "unified_delta_j_rejected"
        )
        controlled_audit = controlled_evaluation["unified_delta_j_shadow"]
        self.assertEqual(controlled_audit["actual_decision"], "reject")
        self.assertTrue(controlled_audit["agrees_with_existing_decision"])

    def test_unified_delta_j_shadow_agrees_with_virtual_swap_rejection(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            budget_virtual_swap_enabled=True,
            budget_unified_delta_j_shadow_enabled=True,
        )
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        second = {
            **first,
            "step": 1,
            "image_id": 1,
            "embedding": manager._image_embedding(
                patch_features(0.0, 1.0)
            ),
            "patch_prototypes": patch_features(0.0, 1.0),
        }
        manager._admit_quarantined_member(
            first, 0, admission_route="legacy"
        )
        manager._admit_quarantined_member(
            second, 1, admission_route="legacy"
        )
        manager.coverage_history = [
            {
                "step": 2,
                "image_id": 2,
                "support_threshold": 0.5,
                "supports": {0: 0.9, 1: 0.1},
                "embedding": first["embedding"],
                "patch_prototypes": first["patch_prototypes"],
                "lab_feature": None,
            },
            {
                "step": 3,
                "image_id": 3,
                "support_threshold": 0.5,
                "supports": {0: 0.1, 1: 0.9},
                "embedding": second["embedding"],
                "patch_prototypes": second["patch_prototypes"],
                "lab_feature": None,
            },
        ]
        candidates = [
            {
                **first,
                "step": step,
                "image_id": step,
                "current_expert_supports": {0: 0.1, 1: 0.1},
            }
            for step in (4, 5)
        ]

        _, admitted_id = manager._admit_quarantined_member(
            candidates[0],
            step=5,
            cluster_members=candidates,
            admission_route="legacy",
            support_threshold=0.5,
        )

        self.assertIsNone(admitted_id)
        evaluation = manager.last_budget_replacement_evaluation
        self.assertEqual(
            evaluation["outcome"], "virtual_swap_no_positive_task_gain"
        )
        audit = evaluation["unified_delta_j_shadow"]
        self.assertEqual(audit["actual_decision"], "reject")
        self.assertFalse(audit["would_accept"])
        self.assertTrue(audit["agrees_with_existing_decision"])

    def test_virtual_swap_audits_dino_score_harm_for_unique_victim(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            budget_virtual_swap_enabled=True,
            budget_patch_coverage_enabled=True,
            budget_unified_delta_j_shadow_enabled=True,
        )
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        second = manager._quarantine_member(
            step=1,
            image_id=1,
            dino_patch_features=patch_features(0.0, 1.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        manager._admit_quarantined_member(first, 0, admission_route="legacy")
        manager._admit_quarantined_member(second, 1, admission_route="legacy")
        manager.coverage_history = [
            {
                "step": 2,
                "image_id": 2,
                "support_threshold": 0.5,
                "supports": {0: 0.9, 1: 0.1},
                "embedding": first["embedding"],
                "patch_prototypes": first["patch_prototypes"],
                "lab_feature": None,
            },
            {
                "step": 3,
                "image_id": 3,
                "support_threshold": 0.5,
                "supports": {0: 0.1, 1: 0.9},
                "embedding": second["embedding"],
                "patch_prototypes": second["patch_prototypes"],
                "lab_feature": None,
            },
        ]
        candidate = {
            **first,
            "step": 4,
            "image_id": 4,
            "current_expert_supports": {0: 0.9, 1: 0.1},
        }

        profiles = manager._virtual_swap_profiles(
            [candidate], support_threshold=0.5
        )

        safe_profile = profiles[0]["dino_normal_score_shadow"]
        harmful_profile = profiles[1]["dino_normal_score_shadow"]
        self.assertTrue(safe_profile["available"])
        self.assertEqual(
            safe_profile["score_source"],
            "shared_msm_representative_reference",
        )
        self.assertAlmostEqual(safe_profile["normal_score_penalty"], 0.0)
        self.assertGreater(harmful_profile["normal_score_penalty"], 0.0)
        self.assertEqual(harmful_profile["evaluated_image_ids"], [2, 3])

    def test_virtual_addition_audits_history_to_full_expert_regime_switch(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            budget_virtual_swap_enabled=True,
            budget_patch_coverage_enabled=True,
            budget_unified_delta_j_shadow_enabled=True,
        )
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        manager._admit_quarantined_member(first, 0, admission_route="legacy")
        manager.active_experts[0]["scoring_weight"] = 0.25
        state_b = manager._quarantine_member(
            step=1,
            image_id=1,
            dino_patch_features=patch_features(0.0, 1.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        manager.coverage_history = [
            {
                "step": step,
                "image_id": step,
                "support_threshold": 0.5,
                "supports": {0: 0.9},
                "embedding": member["embedding"],
                "patch_prototypes": member["patch_prototypes"],
                "lab_feature": None,
            }
            for step, member in enumerate(
                [first, first, state_b, state_b], start=2
            )
        ]
        candidate = {
            **first,
            "step": 6,
            "image_id": 6,
            "current_expert_supports": {0: 0.9},
        }

        profile = manager._virtual_addition_profile(
            [candidate], support_threshold=0.5
        )["dino_normal_score_shadow"]

        self.assertTrue(profile["available"])
        self.assertTrue(
            all(
                regime.startswith("strict_history_fallback")
                for regime in profile["baseline_score_regimes"]
            )
        )
        self.assertTrue(
            all(
                regime.startswith("full_expert")
                for regime in profile["changed_score_regimes"]
            )
        )
        self.assertGreater(profile["normal_score_penalty"], 0.0)
        manager._audit_free_slot_unified_delta_j_shadow(
            [candidate],
            support_threshold=0.5,
            admission_route="photometric_drift_channel_gate",
        )
        shadow = manager.last_budget_replacement_evaluation[
            "unified_delta_j_shadow"
        ]
        self.assertIn("dino_score_delta_j", shadow)
        self.assertFalse(shadow["dino_score_would_accept"])
        confidence_profiles = shadow["confidence_weight_profiles"]
        self.assertEqual(set(confidence_profiles), {"0.5", "0.25"})
        for confidence_profile in confidence_profiles.values():
            score_profile = confidence_profile[
                "dino_normal_score_shadow"
            ]
            self.assertTrue(score_profile["available"])
            self.assertTrue(
                all(
                    regime.startswith(
                        "strict_history_fallback+confidence_blend"
                    )
                    for regime in score_profile["changed_score_regimes"]
                )
            )
            self.assertLess(
                score_profile["normal_score_penalty"],
                profile["normal_score_penalty"],
            )

        controlled = DynamicExpertManager(
            committee_cap=2,
            budget_virtual_swap_enabled=True,
            budget_patch_coverage_enabled=True,
            budget_unified_delta_j_dino_score_enabled=True,
        )
        controlled_first = controlled._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        controlled._admit_quarantined_member(
            controlled_first, 0, admission_route="legacy"
        )
        controlled.active_experts[0]["scoring_weight"] = 0.25
        controlled.coverage_history = list(manager.coverage_history)
        controlled_candidate = {
            **state_b,
            "step": 6,
            "image_id": 6,
            "current_expert_supports": {0: 0.1},
        }
        _, controlled_id = controlled._admit_quarantined_member(
            controlled_candidate,
            step=6,
            cluster_members=[controlled_candidate],
            admission_route="photometric_drift_channel_gate",
            support_threshold=0.5,
            candidate_evidence={
                "fitness_denstream_delta_j_enabled": True,
            },
        )
        self.assertIsNotNone(
            controlled_id,
            controlled.last_budget_replacement_evaluation,
        )
        admitted = next(
            expert
            for expert in controlled.active_experts
            if expert["expert_id"] == controlled_id
        )
        self.assertEqual(admitted["scoring_weight"], 0.25)
        self.assertEqual(
            admitted["confidence_source"],
            "unified_delta_j_dino_score",
        )
        controlled_shadow = controlled.last_budget_replacement_evaluation[
            "unified_delta_j_shadow"
        ]
        self.assertAlmostEqual(
            controlled_shadow["dino_score_coverage_gain"],
            controlled_shadow["patch_kcenter_candidate_profile"][
                "eviction_loss"
            ],
        )
        self.assertGreater(
            controlled_shadow["dino_score_coverage_gain"],
            controlled_shadow["candidate_gain"],
        )
        self.assertEqual(
            controlled_shadow["confidence_weight_profiles"]["0.25"][
                "dino_score_coverage_gain"
            ],
            controlled_shadow["dino_score_coverage_gain"],
        )
        self.assertEqual(
            controlled_shadow["dino_score_recommended_scoring_weight"],
            0.25,
        )
        self.assertIn("state_coverage_gain", controlled_shadow)
        self.assertIn("combined_coverage_gain", controlled_shadow)

    def test_effective_redundancy_penalty_respects_scoring_weight(self):
        manager = DynamicExpertManager(committee_cap=2)
        manager.active_experts = [
            {"expert_id": 0, "scoring_weight": 1.0}
        ]
        members = [{"current_expert_supports": {0: 0.8}}]

        low = manager._candidate_effective_redundancy_delta(
            members, support_threshold=0.5, scoring_weight=0.25
        )
        high = manager._candidate_effective_redundancy_delta(
            members, support_threshold=0.5, scoring_weight=1.0
        )

        self.assertGreater(low, 0.0)
        self.assertLess(low, high)

    def test_full_budget_continuous_evidence_uses_virtual_swap_then_low_weight(self):
        manager = DynamicExpertManager(
            committee_cap=1,
            novel_provisional_weight=0.25,
        )
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        manager._admit_quarantined_member(first, 0, admission_route="legacy")
        calls = []

        def select(*args, **kwargs):
            calls.append((args, kwargs))
            manager.last_budget_replacement_evaluation = {
                "outcome": "accepted"
            }
            return 0, {"replacement_reason": "virtual_swap"}

        def replace(expert_id, *_args, **_kwargs):
            manager.active_experts = [
                expert
                for expert in manager.active_experts
                if int(expert["expert_id"]) != int(expert_id)
            ]
            return True

        manager._select_budget_replacement = select
        manager._replace_budgeted_expert = replace
        candidate = {
            **first,
            "step": 1,
            "image_id": 1,
            "current_expert_supports": {0: 0.1},
        }

        _, admitted_id = manager._admit_quarantined_member(
            candidate,
            1,
            admission_route="photometric_drift_channel_gate",
            support_threshold=0.5,
            candidate_evidence={
                "fitness_denstream_delta_j_enabled": True,
            },
        )

        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(admitted_id)
        self.assertEqual(len(manager.active_experts), 1)
        self.assertEqual(manager.active_experts[0]["scoring_weight"], 0.25)

    def test_min_loss_coreset_does_not_reject_an_admitted_candidate(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            budget_coverage_loss_enabled=True,
            budget_coverage_min_loss_only=True,
            budget_coverage_channel_weight=1.0,
            budget_coverage_lab_weight=0.0,
            budget_coverage_dino_weight=0.0,
        )
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        second = {
            **first,
            "step": 1,
            "image_id": 1,
            "embedding": manager._image_embedding(
                patch_features(0.0, 1.0)
            ),
        }
        manager._admit_quarantined_member(first, 0, admission_route="legacy")
        manager._admit_quarantined_member(second, 1, admission_route="legacy")
        manager.coverage_history = [
            {
                "step": step,
                "image_id": step,
                "support_threshold": 0.5,
                "supports": (
                    {0: 0.9, 1: 0.1}
                    if step < 6
                    else {0: 0.1, 1: 0.9}
                ),
                "embedding": None,
                "lab_feature": None,
            }
            for step in range(2, 10)
        ]
        candidates = [
            {
                **first,
                "step": step,
                "image_id": step,
                "current_expert_supports": {0: 0.1, 1: 0.1},
            }
            for step in (10, 11)
        ]

        victim_id, audit = manager._select_budget_replacement(
            candidates,
            support_threshold=0.5,
            admission_route="legacy",
        )

        self.assertIsNotNone(victim_id)
        self.assertEqual(
            audit["replacement_reason"],
            "coreset_minimum_coverage_loss_replacement",
        )
        evaluation = manager.last_budget_replacement_evaluation
        self.assertEqual(evaluation["outcome"], "accepted")
        self.assertEqual(
            evaluation["coverage_selection_mode"],
            "minimum_removal_loss",
        )
        profile = evaluation["coverage_loss_profiles"][victim_id]
        self.assertLess(profile["swap_gain"], 0.0)
        self.assertEqual(
            profile["removal_loss"],
            min(
                item["removal_loss"]
                for item in evaluation["coverage_loss_profiles"].values()
            ),
        )
        manager._replace_budgeted_expert(victim_id, 12, audit)
        self.assertEqual(
            manager.retired_experts[-1]["last_event"],
            "replaced_coreset_coverage",
        )

    def test_joint_coreset_can_evict_candidate_at_capacity(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            budget_coverage_loss_enabled=True,
            budget_coverage_joint_eviction_enabled=True,
            budget_coverage_channel_weight=1.0,
            budget_coverage_lab_weight=0.0,
            budget_coverage_dino_weight=0.0,
        )
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        second = {
            **first,
            "step": 1,
            "image_id": 1,
            "embedding": manager._image_embedding(
                patch_features(0.0, 1.0)
            ),
        }
        manager._admit_quarantined_member(first, 0, admission_route="legacy")
        manager._admit_quarantined_member(second, 1, admission_route="legacy")
        manager.coverage_history = [
            {
                "step": step,
                "image_id": step,
                "support_threshold": 0.5,
                "supports": (
                    {0: 0.9, 1: 0.1}
                    if step < 6
                    else {0: 0.1, 1: 0.9}
                ),
                "embedding": None,
                "lab_feature": None,
            }
            for step in range(2, 10)
        ]
        candidates = [
            {
                **first,
                "step": step,
                "image_id": step,
                "current_expert_supports": {0: 0.9, 1: 0.1},
            }
            for step in (10, 11)
        ]

        victim_id, audit = manager._select_budget_replacement(
            candidates,
            support_threshold=0.5,
            admission_route="legacy",
            candidate_evidence={
                "new_coverage_fraction": 0.5,
                "new_coverage_count": 1,
                "candidate_observation_count": 2,
            },
        )

        self.assertIsNone(victim_id)
        self.assertIsNone(audit)
        evaluation = manager.last_budget_replacement_evaluation
        self.assertEqual(
            evaluation["outcome"],
            "candidate_evicted_minimum_coverage_loss",
        )
        self.assertEqual(evaluation["evicted_entity_id"], "candidate")
        self.assertEqual(
            evaluation["coverage_selection_mode"],
            "joint_minimum_loss",
        )

    def test_joint_coreset_can_evict_existing_expert(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            budget_coverage_loss_enabled=True,
            budget_coverage_joint_eviction_enabled=True,
            budget_coverage_channel_weight=1.0,
            budget_coverage_lab_weight=0.0,
            budget_coverage_dino_weight=0.0,
        )
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        second = {
            **first,
            "step": 1,
            "image_id": 1,
            "embedding": manager._image_embedding(
                patch_features(0.0, 1.0)
            ),
        }
        manager._admit_quarantined_member(first, 0, admission_route="legacy")
        manager._admit_quarantined_member(second, 1, admission_route="legacy")
        manager.coverage_history = [
            {
                "step": step,
                "image_id": step,
                "support_threshold": 0.5,
                "supports": {0: 0.9, 1: 0.1},
                "embedding": None,
                "lab_feature": None,
            }
            for step in range(2, 10)
        ]
        candidates = [
            {
                **second,
                "step": step,
                "image_id": step,
                "current_expert_supports": {0: 0.1, 1: 0.1},
            }
            for step in (10, 11, 12, 13)
        ]

        victim_id, audit = manager._select_budget_replacement(
            candidates,
            support_threshold=0.5,
            admission_route="legacy",
        )

        self.assertEqual(victim_id, 1)
        self.assertEqual(
            audit["replacement_reason"],
            "coreset_joint_minimum_loss_replacement",
        )
        evaluation = manager.last_budget_replacement_evaluation
        self.assertEqual(evaluation["outcome"], "accepted")
        self.assertEqual(evaluation["evicted_entity_id"], 1)
        manager._replace_budgeted_expert(victim_id, 14, audit)
        self.assertEqual(
            manager.retired_experts[-1]["last_event"],
            "replaced_coreset_coverage",
        )

    def test_patch_coreset_is_bounded_and_deterministic(self):
        manager = DynamicExpertManager(
            budget_patch_coverage_enabled=True,
            budget_patch_prototype_count=2,
        )
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=torch.float32,
        )

        first = manager._patch_coreset(features)
        second = manager._patch_coreset(features)

        self.assertEqual(tuple(first.shape), (2, 2))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(
            torch.allclose(first.norm(dim=1), torch.ones(2))
        )

    def test_patch_coverage_preserves_complementary_patch_modes(self):
        manager = DynamicExpertManager(
            budget_patch_coverage_enabled=True,
            budget_patch_prototype_count=2,
            budget_coverage_channel_weight=0.0,
            budget_coverage_lab_weight=0.0,
            budget_coverage_dino_weight=1.0,
        )
        records = [
            {
                "image_id": 0,
                "support_threshold": 0.5,
                "supports": {},
                "embedding": None,
                "patch_prototypes": torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0]],
                    dtype=torch.float32,
                ),
                "lab_feature": None,
            }
        ]

        def entity(expert_id, prototype):
            return {
                "expert_id": expert_id,
                "embedding": None,
                "patch_prototypes": torch.tensor(
                    [prototype], dtype=torch.float32
                ),
                "lab_center": None,
                "lab_l_scale": 1.0,
                "lab_ab_scale": 1.0,
                "candidate_image_ids": set(),
            }

        horizontal = entity(0, [1.0, 0.0])
        vertical = entity(1, [0.0, 1.0])

        one_mode = manager._coverage_loss_objective(
            records, [horizontal]
        )
        both_modes = manager._coverage_loss_objective(
            records, [horizontal, vertical]
        )

        self.assertGreater(both_modes["dino"], one_mode["dino"])
        self.assertAlmostEqual(both_modes["dino"], 1.0)

    def test_patch_victim_selection_ranks_only_replaceable_experts(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            budget_patch_coverage_enabled=True,
            budget_patch_victim_selection_enabled=True,
        )
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        second = manager._quarantine_member(
            step=1,
            image_id=1,
            dino_patch_features=patch_features(0.0, 1.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        manager._admit_quarantined_member(first, 0, admission_route="legacy")
        manager._admit_quarantined_member(second, 1, admission_route="legacy")
        manager.coverage_history = [
            {
                "step": 2,
                "image_id": 2,
                "support_threshold": 0.5,
                "supports": {0: 0.9, 1: 0.9},
                "embedding": first["embedding"],
                "patch_prototypes": first["patch_prototypes"],
                "lab_feature": None,
            }
        ]
        manager._coverage_loss_joint_eviction_profiles = (
            lambda candidate_members, support_threshold: {
                0: {"eviction_loss": 0.8},
                1: {"eviction_loss": 0.1},
                "candidate": {"eviction_loss": 0.2},
            }
        )
        candidates = [
            {
                **first,
                "step": 3,
                "image_id": 3,
                "current_expert_supports": {0: 0.1, 1: 0.1},
            }
        ]

        victim_id, audit = manager._select_budget_replacement(
            candidates,
            support_threshold=0.5,
            admission_route="legacy",
            candidate_evidence={
                "new_coverage_fraction": 1.0,
                "new_coverage_count": 1,
                "candidate_observation_count": 1,
            },
        )

        self.assertEqual(victim_id, 1)
        self.assertEqual(
            audit["replacement_reason"],
            "coreset_patch_victim_replacement",
        )
        self.assertEqual(
            manager.last_budget_replacement_evaluation[
                "coverage_selection_mode"
            ],
            "patch_minimum_victim_loss",
        )

    def test_patch_kcenter_profiles_use_worst_patch_distance(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            budget_patch_coverage_enabled=True,
            budget_patch_victim_selection_enabled=True,
            budget_patch_kcenter_victim_selection_enabled=True,
        )
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        second = manager._quarantine_member(
            step=1,
            image_id=1,
            dino_patch_features=patch_features(0.0, 1.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        manager._admit_quarantined_member(first, 0, admission_route="legacy")
        manager._admit_quarantined_member(second, 1, admission_route="legacy")
        manager.coverage_history = [
            {
                "step": 2,
                "image_id": 2,
                "support_threshold": 0.5,
                "supports": {0: 0.9, 1: 0.9},
                "embedding": first["embedding"],
                "patch_prototypes": torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
                ),
                "lab_feature": None,
            }
        ]
        profiles = manager._patch_kcenter_eviction_profiles(
            [first], 0.5
        )
        self.assertIn(0, profiles)
        self.assertIn(1, profiles)
        self.assertGreaterEqual(profiles[0]["eviction_loss"], 0.0)
        self.assertGreaterEqual(profiles[1]["eviction_loss"], 0.0)
        self.assertEqual(
            profiles[0]["coverage_objective"],
            "patch_kcenter_max_cosine_distance",
        )

    def test_patch_kcenter_does_not_evict_semantically_unique_expert(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            budget_patch_coverage_enabled=True,
            budget_patch_victim_selection_enabled=True,
            budget_patch_kcenter_victim_selection_enabled=True,
        )
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        second = manager._quarantine_member(
            step=1,
            image_id=1,
            dino_patch_features=patch_features(0.0, 1.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        manager._admit_quarantined_member(first, 0, admission_route="legacy")
        manager._admit_quarantined_member(second, 1, admission_route="legacy")
        manager.active_experts[0]["last_supported_step"] = -10
        manager.active_experts[1]["last_supported_step"] = 2
        manager._expert_coverage_profiles = lambda: {
            0: {
                "observed_count": 18,
                "covered_count": 17,
                "unique_count": 15,
                "redundant_count": 2,
                "coverage_fraction": 17 / 18,
                "unique_fraction": 15 / 18,
                "redundant_fraction": 2 / 18,
            },
            1: {
                "observed_count": 8,
                "covered_count": 2,
                "unique_count": 0,
                "redundant_count": 2,
                "coverage_fraction": 0.25,
                "unique_fraction": 0.0,
                "redundant_fraction": 0.25,
            },
        }
        manager._expert_lab_uniqueness_profiles = lambda: {
            0: {
                "has_lab_state": True,
                "nearest_relative_distance": None,
                "photometrically_redundant": False,
            },
            1: {
                "has_lab_state": False,
                "nearest_relative_distance": None,
                "photometrically_redundant": False,
            },
        }
        manager._patch_kcenter_eviction_profiles = (
            lambda candidate_members, support_threshold: {
                0: {"eviction_loss": 0.0},
                1: {"eviction_loss": 0.1},
                "candidate": {"eviction_loss": 0.2},
            }
        )
        manager._candidate_expert_lab_distance_audit = lambda members: {
            "expert_lab_distances": [
                {
                    "l_novelty_threshold": 1.0,
                    "ab_novelty_threshold": 1.0,
                    "l_distance": 1.0,
                    "ab_distance": 1.0,
                }
            ],
        }
        manager._candidate_lab_novelty = lambda evaluation: {
            "lab_novel_to_all_experts": False,
        }
        candidates = [
            {
                **first,
                "step": 3,
                "image_id": 3,
                "current_expert_supports": {0: 0.1, 1: 0.1},
            }
        ]

        victim_id, _ = manager._select_budget_replacement(
            candidates,
            support_threshold=0.5,
            admission_route="photometric_drift_channel_gate",
            candidate_evidence={
                "new_coverage_fraction": 0.5,
                "new_coverage_count": 1,
                "candidate_observation_count": 2,
            },
        )

        self.assertEqual(victim_id, 1)
        evaluation = manager.last_budget_replacement_evaluation
        semantic_losses = [
            profile["total_loss"]
            for profile in evaluation[
                "semantic_replacement_losses"
            ].values()
        ]
        self.assertGreater(
            max(semantic_losses),
            min(semantic_losses),
        )
        self.assertEqual(
            evaluation["coverage_selection_mode"],
            "patch_kcenter_minimum_victim_loss",
        )
        self.assertEqual(
            evaluation["semantic_selection_mode"],
            "minimum_uncovered_channel_lab_loss",
        )

    def test_virtual_swap_rejects_when_all_victims_harm_trusted_history(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            budget_patch_victim_selection_enabled=True,
            budget_virtual_swap_enabled=True,
        )
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        second = manager._quarantine_member(
            step=1,
            image_id=1,
            dino_patch_features=patch_features(0.0, 1.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        manager._admit_quarantined_member(first, 0, admission_route="legacy")
        manager._admit_quarantined_member(second, 1, admission_route="legacy")
        manager._expert_coverage_profiles = lambda: {
            expert_id: {
                "observed_count": 2,
                "covered_count": 2,
                "unique_count": 0,
                "redundant_count": 2,
                "coverage_fraction": 1.0,
                "unique_fraction": 0.0,
                "redundant_fraction": 1.0,
            }
            for expert_id in (0, 1)
        }
        manager._expert_lab_uniqueness_profiles = lambda: {
            expert_id: {
                "has_lab_state": False,
                "nearest_relative_distance": None,
                "photometrically_redundant": False,
            }
            for expert_id in (0, 1)
        }
        manager._candidate_expert_lab_distance_audit = lambda members: {}
        manager._candidate_lab_novelty = lambda evaluation: {
            "lab_novel_to_all_experts": False,
        }
        manager._virtual_swap_profiles = lambda *args, **kwargs: {
            0: {"safe": False, "median_normal_score_shift": 0.2},
            1: {"safe": False, "median_normal_score_shift": 0.3},
        }
        candidates = [
            {
                **first,
                "step": 3,
                "image_id": 3,
                "current_expert_supports": {0: 0.1, 1: 0.1},
            }
        ]

        victim_id, replacement = manager._select_budget_replacement(
            candidates,
            support_threshold=0.5,
            admission_route="photometric_drift_channel_gate",
            candidate_evidence={
                "new_coverage_fraction": 0.5,
                "new_coverage_count": 1,
                "candidate_observation_count": 2,
            },
        )

        self.assertIsNone(victim_id)
        self.assertIsNone(replacement)
        self.assertEqual(
            manager.last_budget_replacement_evaluation["outcome"],
            "virtual_swap_rejected",
        )

    def test_virtual_swap_treats_lab_novelty_as_task_new_coverage(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            budget_virtual_swap_enabled=True,
            photometric_budget_admission_mode="channel_or_lab",
        )
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        second = manager._quarantine_member(
            step=1,
            image_id=1,
            dino_patch_features=patch_features(0.0, 1.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        manager._admit_quarantined_member(first, 0, admission_route="legacy")
        manager._admit_quarantined_member(second, 1, admission_route="legacy")
        manager._candidate_expert_lab_distance_audit = lambda members: {}
        manager._candidate_lab_novelty = lambda evaluation: {
            "lab_novel_to_all_experts": True,
        }
        manager._expert_coverage_profiles = lambda: {
            expert_id: {
                "observed_count": 2,
                "covered_count": 2,
                "unique_count": 0,
                "redundant_count": 2,
                "coverage_fraction": 1.0,
                "unique_fraction": 0.0,
                "redundant_fraction": 1.0,
            }
            for expert_id in (0, 1)
        }
        manager._expert_lab_uniqueness_profiles = lambda: {
            expert_id: {
                "has_lab_state": True,
                "nearest_relative_distance": 0.0,
                "photometrically_redundant": True,
            }
            for expert_id in (0, 1)
        }
        manager._virtual_swap_profiles = lambda *args, **kwargs: {
            expert_id: {
                "safe": True,
                "mean_normal_score_shift": 0.0,
                "normal_score_iqr_increase": 0.0,
                "current_state_support_loss": 0.0,
                "recurrent_state_loss": 0.0,
                "net_task_gain": 1.0,
            }
            for expert_id in (0, 1)
        }

        victim_id, replacement = manager._select_budget_replacement(
            [
                {
                    **first,
                    "step": 2,
                    "image_id": 2,
                    "current_expert_supports": {0: 0.9, 1: 0.9},
                }
            ],
            support_threshold=0.5,
            admission_route="photometric_drift_channel_gate",
            candidate_evidence={
                "new_coverage_fraction": 0.0,
                "new_coverage_count": 0,
                "candidate_observation_count": 1,
            },
        )

        self.assertIsNotNone(victim_id)
        self.assertIsNotNone(replacement)
        self.assertTrue(
            manager.last_budget_replacement_evaluation[
                "task_new_coverage"
            ]
        )

    def test_virtual_swap_rejects_without_trusted_history(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            budget_virtual_swap_enabled=True,
        )
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        second = {
            **first,
            "step": 1,
            "image_id": 1,
            "embedding": manager._image_embedding(
                patch_features(0.0, 1.0)
            ),
        }
        manager._admit_quarantined_member(first, 0, admission_route="legacy")
        manager._admit_quarantined_member(second, 1, admission_route="legacy")
        manager._candidate_expert_lab_distance_audit = lambda members: {}
        manager._candidate_lab_novelty = lambda evaluation: {
            "lab_novel_to_all_experts": False,
        }

        victim_id, replacement = manager._select_budget_replacement(
            [
                {
                    **first,
                    "step": 2,
                    "image_id": 2,
                    "current_expert_supports": {0: 0.1, 1: 0.1},
                }
            ],
            support_threshold=0.5,
            admission_route="photometric_drift_channel_gate",
            candidate_evidence={
                "new_coverage_fraction": 1.0,
                "new_coverage_count": 1,
                "candidate_observation_count": 1,
            },
        )

        self.assertIsNone(victim_id)
        self.assertIsNone(replacement)
        self.assertEqual(
            manager.last_budget_replacement_evaluation["outcome"],
            "virtual_swap_insufficient_history",
        )

    def test_virtual_scoring_support_uses_dino_before_lab(self):
        manager = DynamicExpertManager()
        record = {
            "support_threshold": 0.5,
            "supports": {},
            "embedding": torch.tensor([1.0, 0.0]),
            "patch_prototypes": torch.tensor([[1.0, 0.0]]),
            "lab_feature": np.asarray(
                [10.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64
            ),
        }
        entity = {
            "expert_id": None,
            "embedding": torch.tensor([1.0, 0.0]),
            "patch_prototypes": torch.tensor([[1.0, 0.0]]),
            "lab_center": np.asarray(
                [90.0, 9.0, 9.0, 9.0, 9.0, 9.0], dtype=np.float64
            ),
            "lab_l_scale": 0.01,
            "lab_ab_scale": 0.01,
        }

        support = manager._virtual_entity_support(record, entity)

        self.assertAlmostEqual(support, 1.0)

    def test_legacy_admission_replaces_redundant_expert_at_cap(self):
        manager = DynamicExpertManager(committee_cap=2)
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        second = {
            **first,
            "step": 1,
            "image_id": 1,
            "embedding": manager._image_embedding(
                patch_features(0.0, 1.0)
            ),
        }
        manager._admit_quarantined_member(first, 0, admission_route="legacy")
        manager._admit_quarantined_member(second, 1, admission_route="legacy")
        manager.coverage_history = [
            {
                "step": 2,
                "support_threshold": 0.5,
                "supports": {0: 0.9, 1: 0.9},
            }
        ]
        candidate_members = [
            {
                **first,
                "step": step,
                "image_id": step,
                "embedding": manager._image_embedding(
                    patch_features(-1.0, 0.0)
                ),
                "current_expert_supports": {0: 0.1, 1: 0.1},
            }
            for step in (3, 4)
        ]

        _, expert_id = manager._admit_quarantined_member(
            candidate_members[0],
            step=4,
            cluster_members=candidate_members,
            admission_route="legacy",
            support_threshold=0.5,
        )

        self.assertEqual(expert_id, 2)
        self.assertEqual(len(manager.active_experts), 2)
        self.assertEqual(manager.retired_experts[-1]["expert_id"], 0)
        self.assertEqual(
            manager.last_budget_replacement["admission_route"],
            "legacy",
        )

    def test_legacy_route_reaches_budget_replacement_via_advance(self):
        manager = DynamicExpertManager(
            min_cluster_support=2,
            committee_cap=2,
            base_ttl=10,
            max_ttl=10,
        )
        manager.ms_history.append(1.0)
        manager.support_history.append(0.5)
        first = patch_features(1.0, 0.0)
        second = patch_features(0.0, 1.0)
        third = patch_features(-1.0, 0.0)

        manager.advance(0, 0, first, 0.1, 1.0, {})
        manager.advance(1, 1, first, 0.1, 1.0, {})
        manager.advance(
            2,
            2,
            second,
            0.1,
            1.0,
            {0: 1.0},
            current_expert_supports={0: 0.1},
        )
        manager.advance(
            3,
            3,
            second,
            0.1,
            1.0,
            {0: 1.0},
            current_expert_supports={0: 0.1},
        )
        manager.advance(
            4,
            4,
            third,
            0.1,
            1.0,
            {0: 1.0, 1: 1.0},
            current_expert_supports={0: 0.1, 1: 0.1},
        )
        event = manager.advance(
            5,
            5,
            third,
            0.1,
            1.0,
            {0: 1.0, 1: 1.0},
            current_expert_supports={0: 0.1, 1: 0.1},
        )

        self.assertEqual(event["admitted_expert_id"], 2)
        self.assertEqual(event["budget_replacement"]["admission_route"], "legacy")
        self.assertEqual(len(manager.active_experts), 2)
        self.assertIn(
            event["budget_replacement"]["expert_id"],
            event["deleted_expert_ids"],
        )

    def test_budget_replacement_rejects_candidate_without_new_coverage(self):
        manager = DynamicExpertManager(committee_cap=1)
        member = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        manager._admit_quarantined_member(member, 0, admission_route="legacy")
        manager.coverage_history = [
            {
                "step": 1,
                "support_threshold": 0.5,
                "supports": {0: 0.9},
            }
        ]
        covered_candidate = {
            **member,
            "step": 2,
            "image_id": 2,
            "current_expert_supports": {0: 0.9},
        }

        _, expert_id = manager._admit_quarantined_member(
            covered_candidate,
            step=2,
            cluster_members=[covered_candidate],
            admission_route="legacy",
            support_threshold=0.5,
        )

        self.assertIsNone(expert_id)
        self.assertEqual(len(manager.active_experts), 1)
        self.assertEqual(manager.retired_experts, [])
        self.assertEqual(
            manager.last_budget_replacement_evaluation["outcome"],
            "insufficient_new_coverage",
        )
        self.assertEqual(
            manager.last_budget_replacement_evaluation[
                "new_coverage_fraction"
            ],
            0.0,
        )

    def test_budget_replacement_audits_candidate_to_all_expert_lab_centers(
        self,
    ):
        manager = DynamicExpertManager(
            committee_cap=2,
            cold_start_lab_enabled=True,
        )

        def member(step, lab_l, supports=None):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.8,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [lab_l, 10.0, 0.0, 0.0, 0.0, 0.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports=supports,
            )

        first = member(0, 40.0)
        second = member(1, 60.0)
        manager._admit_quarantined_member(
            first,
            0,
            admission_route="photometric_drift_channel_gate",
        )
        manager._admit_quarantined_member(
            second,
            1,
            admission_route="photometric_drift_channel_gate",
        )
        candidates = [
            member(2, 80.0, {0: 0.9, 1: 0.9}),
            member(3, 82.0, {0: 0.9, 1: 0.9}),
        ]

        victim_id, _ = manager._select_budget_replacement(
            candidates,
            support_threshold=0.5,
            admission_route="photometric_drift_channel_gate",
        )
        audit = manager.last_budget_replacement_evaluation

        self.assertIsNone(victim_id)
        self.assertEqual(audit["outcome"], "insufficient_new_coverage")
        self.assertEqual(audit["candidate_lab_member_count"], 2)
        self.assertEqual(audit["candidate_lab_center"][0], 81.0)
        self.assertEqual(
            [item["expert_id"] for item in audit["expert_lab_distances"]],
            [0, 1],
        )
        self.assertAlmostEqual(
            audit["expert_lab_distances"][0]["total_distance"],
            0.41,
        )
        self.assertAlmostEqual(
            audit["expert_lab_distances"][1]["total_distance"],
            0.21,
        )
        self.assertTrue(
            audit["expert_lab_distances"][0]["l_outside_baseline"]
        )
        self.assertTrue(
            audit["expert_lab_distances"][1]["l_outside_baseline"]
        )
        self.assertFalse(
            audit["expert_lab_distances"][0]["ab_outside_baseline"]
        )
        self.assertEqual(audit["nearest_lab_expert_id"], 1)
        self.assertAlmostEqual(
            audit["nearest_lab_total_distance"],
            0.21,
        )
        self.assertEqual(audit["lab_distance_missing_expert_ids"], [])

    def test_dual_budget_admission_accepts_lab_novel_candidate(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            cold_start_lab_enabled=True,
            photometric_budget_admission_mode="channel_or_lab",
        )

        def member(step, lab_l, supports=None):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.8,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [lab_l, 10.0, 0.0, 0.0, 0.0, 0.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports=supports,
            )

        manager._admit_quarantined_member(
            member(0, 40.0),
            0,
            admission_route="photometric_drift_channel_gate",
        )
        manager._admit_quarantined_member(
            member(1, 41.0),
            1,
            admission_route="photometric_drift_channel_gate",
        )
        manager.coverage_history = [
            {
                "step": 2,
                "support_threshold": 0.5,
                "supports": {0: 0.9, 1: 0.9},
            },
            {
                "step": 3,
                "support_threshold": 0.5,
                "supports": {0: 0.9, 1: 0.1},
            },
        ]
        candidates = [
            member(step, lab_l, {0: 0.9, 1: 0.9})
            for step, lab_l in ((4, 80.0), (5, 82.0))
        ]

        _, admitted_id = manager._admit_quarantined_member(
            candidates[0],
            step=5,
            cluster_members=candidates,
            admission_route="photometric_drift_channel_gate",
            support_threshold=0.5,
        )

        self.assertEqual(
            admitted_id,
            2,
            manager.last_novel_quarantine_event,
        )
        self.assertEqual(
            [expert["expert_id"] for expert in manager.active_experts],
            [0, 2],
        )
        self.assertEqual(manager.retired_experts[-1]["expert_id"], 1)
        evaluation = manager.last_budget_replacement_evaluation
        self.assertEqual(evaluation["candidate_admission_evidence"], "lab")
        self.assertFalse(evaluation["channel_novel"])
        self.assertTrue(evaluation["lab_novel"])
        self.assertTrue(
            evaluation["victim_lab_uniqueness"][
                "photometrically_redundant"
            ]
        )
        self.assertEqual(
            manager.last_budget_replacement["reason"],
            "lab_or_coverage_budget_replacement",
        )

    def test_dual_budget_lab_only_protects_nonredundant_experts(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            cold_start_lab_enabled=True,
            photometric_budget_admission_mode="channel_or_lab",
        )

        def member(step, lab_l, supports=None):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.8,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [lab_l, 10.0, 0.0, 0.0, 0.0, 0.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports=supports,
            )

        manager._admit_quarantined_member(
            member(0, 40.0),
            0,
            admission_route="photometric_drift_channel_gate",
        )
        manager._admit_quarantined_member(
            member(1, 60.0),
            1,
            admission_route="photometric_drift_channel_gate",
        )
        manager.coverage_history = [
            {
                "step": 2,
                "support_threshold": 0.5,
                "supports": {0: 0.9, 1: 0.1},
            },
            {
                "step": 3,
                "support_threshold": 0.5,
                "supports": {0: 0.1, 1: 0.9},
            },
        ]
        candidates = [
            member(step, lab_l, {0: 0.9, 1: 0.9})
            for step, lab_l in ((4, 80.0), (5, 82.0))
        ]

        _, admitted_id = manager._admit_quarantined_member(
            candidates[0],
            step=5,
            cluster_members=candidates,
            admission_route="photometric_drift_channel_gate",
            support_threshold=0.5,
        )

        self.assertIsNone(admitted_id)
        self.assertEqual(
            [expert["expert_id"] for expert in manager.active_experts],
            [0, 1],
        )
        self.assertEqual(manager.retired_experts, [])
        evaluation = manager.last_budget_replacement_evaluation
        self.assertTrue(evaluation["lab_novel"])
        self.assertFalse(evaluation["channel_novel"])
        self.assertEqual(evaluation["outcome"], "no_redundant_victim")

    def test_dual_budget_admission_rejects_covered_lab_redundant_candidate(
        self,
    ):
        manager = DynamicExpertManager(
            committee_cap=1,
            cold_start_lab_enabled=True,
            photometric_budget_admission_mode="channel_or_lab",
        )

        def member(step, lab_l, supports=None):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.8,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [lab_l, 10.0, 0.0, 0.0, 0.0, 0.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports=supports,
            )

        manager._admit_quarantined_member(
            member(0, 40.0),
            0,
            admission_route="photometric_drift_channel_gate",
        )
        candidates = [
            member(step, lab_l, {0: 0.9})
            for step, lab_l in ((1, 40.5), (2, 41.0))
        ]

        _, admitted_id = manager._admit_quarantined_member(
            candidates[0],
            step=2,
            cluster_members=candidates,
            admission_route="photometric_drift_channel_gate",
            support_threshold=0.5,
        )

        self.assertIsNone(admitted_id)
        evaluation = manager.last_budget_replacement_evaluation
        self.assertFalse(evaluation["channel_novel"])
        self.assertFalse(evaluation["lab_novel"])
        self.assertEqual(evaluation["outcome"], "insufficient_new_coverage")

    def test_dual_budget_admission_requires_lab_state_for_every_expert(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            cold_start_lab_enabled=True,
            photometric_budget_admission_mode="channel_or_lab",
        )

        def member(step, lab_l, supports=None):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.8,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [lab_l, 10.0, 0.0, 0.0, 0.0, 0.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports=supports,
            )

        manager._admit_quarantined_member(
            member(0, 40.0),
            0,
            admission_route="photometric_drift_channel_gate",
        )
        manager._admit_quarantined_member(
            member(1, 60.0),
            1,
            admission_route="legacy",
        )
        candidates = [
            member(step, lab_l, {0: 0.9, 1: 0.9})
            for step, lab_l in ((2, 80.0), (3, 82.0))
        ]

        victim_id, _ = manager._select_budget_replacement(
            candidates,
            support_threshold=0.5,
            admission_route="photometric_drift_channel_gate",
        )

        evaluation = manager.last_budget_replacement_evaluation
        self.assertIsNone(victim_id)
        self.assertFalse(evaluation["lab_comparable_to_all_experts"])
        self.assertFalse(evaluation["lab_novel"])
        self.assertEqual(evaluation["lab_distance_missing_expert_ids"], [1])

    def test_lab_confirmed_gate_does_not_wait_for_coverage_loss(self):
        manager = DynamicExpertManager(
            photometric_candidate_enabled=True,
            photometric_gate_mode="lab_confirmed",
        )

        triggered, expert_id = manager._advance_photometric_gate(
            {
                "expert_id": 3,
                "l_ewma": 0.08,
                "ab_ewma": 0.0,
                "l_threshold": 0.02,
                "ab_threshold": 0.02,
                "drift_confirmed": True,
            },
            support_threshold=0.5,
            expert_channel_supports={3: 0.9},
        )

        self.assertTrue(triggered)
        self.assertEqual(expert_id, 3)
        self.assertFalse(manager.last_photometric_gate["coverage_loss"])
        self.assertEqual(
            manager.last_photometric_gate["gate_mode"],
            "lab_confirmed",
        )

    def test_fixed_low_risk_pool_skips_high_risk_and_confirms_at_four(self):
        manager = DynamicExpertManager(
            novel_admission_mode="channel_gate",
            photometric_candidate_enabled=True,
            photometric_candidate_window_mode="fixed_low_risk",
            photometric_fixed_low_risk_members=4,
        )
        features = patch_features(1.0, 0.0)
        common = {
            "dino_patch_features": features,
            "ms_short_score": 0.1,
            "ms_short_ratio": 0.8,
            "channel_support": 0.2,
            "provisional_channel_support": 0.5,
            "memory_patch_ids": [0],
            "lab_feature": torch.tensor(
                [60.0, 2.0, 1.0, 3.0, 2.0, 4.0]
            ),
            "photometric_trigger_expert_id": 0,
            "photometric_trigger_lab_center": [
                40.0, 2.0, 1.0, 3.0, 2.0, 4.0
            ],
            "historical_ms_threshold": 0.5,
        }

        manager._advance_novel_quarantine(
            step=0,
            image_id=0,
            ms_score=0.8,
            trigger_route="photometric_drift",
            **common,
        )
        self.assertIsNone(manager.novel_quarantine)

        admitted_expert_id = None
        for step in range(1, 5):
            _, admitted_expert_id = manager._advance_novel_quarantine(
                step=step,
                image_id=step,
                ms_score=0.1,
                trigger_route=("photometric_drift" if step == 1 else None),
                **common,
            )

        self.assertEqual(admitted_expert_id, 0)
        self.assertEqual(
            manager.last_novel_quarantine_event["decision"],
            "confirmed_channel_gate",
        )
        self.assertEqual(
            manager.last_novel_quarantine_event["low_risk_image_ids"],
            [1, 2, 3, 4],
        )

    def test_dynamic_low_risk_pool_requires_three_joint_members(self):
        manager = DynamicExpertManager(
            novel_admission_mode="channel_gate",
            novel_memory_keep_fraction=0.5,
            photometric_candidate_enabled=True,
            photometric_candidate_window_mode="dynamic_low_risk",
            photometric_dynamic_min_low_risk_members=3,
            photometric_dynamic_max_low_risk_members=6,
            photometric_min_low_risk_members=3,
            photometric_episode_enabled=True,
        )
        manager.photometric_episode_active = True
        manager.photometric_episode_pending = False
        features = patch_features(1.0, 0.0)
        lightness = [60.0, 80.0, 100.0, 80.0, 80.0]

        for step, value in enumerate(lightness):
            _, expert_id = manager._advance_novel_quarantine(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                trigger_route=("photometric_drift" if step == 0 else None),
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [value, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                photometric_trigger_expert_id=0,
                photometric_trigger_lab_center=[
                    40.0, 2.0, 1.0, 3.0, 2.0, 4.0
                ],
                historical_ms_threshold=0.5,
            )
            if step in {2, 3}:
                self.assertEqual(
                    manager.last_novel_quarantine_event["decision"],
                    "collecting_inconsistent",
                )
                self.assertEqual(len(manager.novel_quarantine), step + 1)

        profile = manager.last_novel_quarantine_event["photometric_profile"]
        self.assertEqual(
            manager.last_novel_quarantine_event["decision"],
            "confirmed_channel_gate",
        )
        self.assertEqual(expert_id, 0)
        self.assertEqual(profile["selected_inlier_image_ids"], [1, 3, 4])
        self.assertEqual(profile["selected_inlier_count"], 3)
        self.assertTrue(
            manager.last_novel_quarantine_event["episode_finished"]
        )
        self.assertEqual(
            [member["image_id"] for member in manager.clusters[0]["members"]],
            [1, 3, 4],
        )
        self.assertIsNone(manager.novel_quarantine)
        self.assertFalse(manager.photometric_episode_active)
        self.assertIsNotNone(manager.photometric_handled_lab_center)

    def test_dynamic_low_risk_pool_confirms_two_when_two_are_required(self):
        manager = DynamicExpertManager(
            novel_admission_mode="channel_gate",
            photometric_candidate_enabled=True,
            photometric_candidate_window_mode="dynamic_low_risk",
            photometric_dynamic_min_low_risk_members=2,
            photometric_dynamic_max_low_risk_members=6,
            photometric_min_low_risk_members=2,
            photometric_episode_enabled=True,
        )
        manager.photometric_episode_active = True
        manager.photometric_episode_pending = False
        features = patch_features(1.0, 0.0)

        admitted_id = None
        for step in range(2):
            _, admitted_id = manager._advance_novel_quarantine(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                trigger_route=("photometric_drift" if step == 0 else None),
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [80.0, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                photometric_trigger_expert_id=0,
                photometric_trigger_lab_center=[
                    40.0, 2.0, 1.0, 3.0, 2.0, 4.0
                ],
                historical_ms_threshold=0.5,
            )

        self.assertEqual(admitted_id, 0)
        profile = manager.last_novel_quarantine_event["photometric_profile"]
        self.assertEqual(profile["required_count"], 2)
        self.assertEqual(profile["selected_inlier_count"], 2)

    def test_retired_state_match_allows_two_member_history_reentry(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            cold_start_lab_enabled=True,
            novel_admission_mode="channel_gate",
            photometric_candidate_enabled=True,
            photometric_candidate_window_mode="dynamic_low_risk",
            photometric_dynamic_min_low_risk_members=3,
            photometric_dynamic_max_low_risk_members=6,
            photometric_min_low_risk_members=3,
            photometric_history_reentry_enabled=True,
            photometric_history_reentry_members=2,
            photometric_adaptive_candidate_evidence=True,
            photometric_episode_enabled=True,
        )

        def member(step, value, features):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [value, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports={0: 0.1},
            )

        state_a = member(0, 40.0, patch_features(1.0, 0.0))
        state_b = member(1, 80.0, patch_features(0.0, 1.0))
        state_b["ms_short_score"] = 0.3
        manager._admit_quarantined_member(
            state_a,
            0,
            admission_route="photometric_drift_channel_gate",
        )
        manager._admit_quarantined_member(
            state_b,
            1,
            admission_route="photometric_drift_channel_gate",
        )
        manager._replace_budgeted_expert(1, 2, {})
        manager.photometric_episode_active = True
        manager.photometric_episode_pending = False

        admitted_id = None
        for step, value in ((3, 80.0), (4, 81.0)):
            current = member(step, value, patch_features(0.0, 1.0))
            _, admitted_id = manager._advance_novel_quarantine(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(0.0, 1.0),
                ms_score=current["ms_score"],
                ms_short_score=current["ms_short_score"],
                ms_short_ratio=current["ms_short_ratio"],
                channel_support=current["channel_support"],
                provisional_channel_support=current[
                    "provisional_channel_support"
                ],
                trigger_route=("photometric_drift" if step == 3 else None),
                memory_patch_ids=current["memory_patch_ids"],
                lab_feature=torch.tensor(current["lab_feature"]),
                photometric_trigger_expert_id=0,
                photometric_trigger_lab_center=[
                    40.0, 2.0, 1.0, 3.0, 2.0, 4.0
                ],
                historical_ms_threshold=0.5,
                current_expert_supports={0: 0.1},
                support_threshold=0.5,
            )

        self.assertEqual(admitted_id, 2)
        admitted = next(
            expert
            for expert in manager.active_experts
            if expert["expert_id"] == admitted_id
        )
        self.assertEqual(
            admitted["admission_route"],
            "photometric_drift_history_reentry_channel_gate",
        )
        self.assertEqual(admitted["cluster_id"], 1)
        self.assertEqual(admitted["reentry_from_expert_id"], 1)
        self.assertEqual(
            admitted["reentry_confirmation_member_image_ids"],
            [3, 4],
        )
        self.assertEqual(len(manager.clusters), 2)
        self.assertEqual(
            manager.last_history_reentry_event["decision"],
            "history_restored_free_slot",
        )
        admitted = next(
            expert
            for expert in manager.active_experts
            if expert["expert_id"] == admitted_id
        )
        self.assertEqual(
            admitted["reentry_reference_source"],
            "current_confirmation",
        )
        self.assertIn(admitted["image_id"], {3, 4})
        profile = manager.last_novel_quarantine_event["photometric_profile"]
        self.assertEqual(profile["required_count"], 2)
        self.assertTrue(profile["history_reentry"]["matched"])
        self.assertEqual(
            profile["history_reentry"]["matched_retired_expert_id"],
            1,
        )

    def test_history_window_reentry_uses_two_members_without_mode_switch(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            cold_start_lab_enabled=True,
            novel_admission_mode="channel_gate",
            photometric_candidate_enabled=True,
            photometric_candidate_window_mode="history",
            photometric_quarantine_steps=4,
            photometric_min_low_risk_members=3,
            photometric_history_reentry_enabled=True,
            photometric_history_reentry_members=2,
            photometric_adaptive_candidate_evidence=True,
            photometric_episode_enabled=True,
        )

        def member(step, value, features):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [value, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports={0: 0.1},
            )

        state_a = member(0, 40.0, patch_features(1.0, 0.0))
        state_b = member(1, 80.0, patch_features(0.0, 1.0))
        manager._admit_quarantined_member(
            state_a,
            0,
            admission_route="photometric_drift_channel_gate",
        )
        manager._admit_quarantined_member(
            state_b,
            1,
            admission_route="photometric_drift_channel_gate",
        )
        manager._replace_budgeted_expert(1, 2, {})
        manager.photometric_episode_active = True
        manager.photometric_episode_pending = False

        admitted_id = None
        for step, value in ((3, 80.0), (4, 81.0)):
            current = member(step, value, patch_features(0.0, 1.0))
            _, admitted_id = manager._advance_novel_quarantine(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(0.0, 1.0),
                ms_score=current["ms_score"],
                ms_short_score=current["ms_short_score"],
                ms_short_ratio=current["ms_short_ratio"],
                channel_support=current["channel_support"],
                provisional_channel_support=current[
                    "provisional_channel_support"
                ],
                trigger_route=("photometric_drift" if step == 3 else None),
                memory_patch_ids=current["memory_patch_ids"],
                lab_feature=torch.tensor(current["lab_feature"]),
                photometric_trigger_expert_id=0,
                photometric_trigger_lab_center=[
                    40.0, 2.0, 1.0, 3.0, 2.0, 4.0
                ],
                historical_ms_threshold=0.5,
                current_expert_supports={0: 0.1},
                support_threshold=0.5,
            )

        self.assertEqual(
            admitted_id,
            2,
            manager.last_novel_quarantine_event,
        )
        self.assertEqual(
            manager.last_history_reentry_event["decision"],
            "history_restored_free_slot",
        )
        profile = manager.last_novel_quarantine_event["photometric_profile"]
        self.assertEqual(profile["required_count"], 2)
        self.assertTrue(profile["history_reentry"]["matched"])
        self.assertEqual(
            profile["history_reentry"]["matched_retired_expert_id"],
            1,
        )

    def test_history_window_reentry_audits_no_match(self):
        manager = DynamicExpertManager(
            photometric_history_reentry_enabled=True,
            photometric_history_reentry_members=2,
        )
        candidates = []
        for step, value in ((0, 80.0), (1, 81.0)):
            candidates.append(
                manager._quarantine_member(
                    step=step,
                    image_id=step,
                    dino_patch_features=patch_features(0.0, 1.0),
                    ms_score=0.1,
                    ms_short_score=0.1,
                    ms_short_ratio=0.8,
                    channel_support=0.2,
                    provisional_channel_support=0.5,
                    memory_patch_ids=[0],
                    lab_feature=torch.tensor(
                        [value, 2.0, 1.0, 3.0, 2.0, 4.0]
                    ),
                    historical_ms_threshold=0.5,
                )
            )

        profile, match = manager._history_reentry_candidate_profile(
            candidates,
            trigger_expert_id=0,
            trigger_lab_center=[
                40.0, 2.0, 1.0, 3.0, 2.0, 4.0
            ],
        )

        self.assertTrue(profile["consistent"])
        self.assertFalse(match["matched"])
        self.assertEqual(
            manager.last_history_reentry_event["decision"],
            "history_reentry_no_match",
        )

    def test_passive_history_reentry_requires_post_retirement_evidence(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            cold_start_lab_enabled=True,
            novel_admission_mode="channel_gate",
            photometric_history_reentry_enabled=True,
            photometric_history_reentry_members=2,
        )

        def member(step, value, features):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [value, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
            )

        state_a = member(0, 40.0, patch_features(1.0, 0.0))
        manager._admit_quarantined_member(
            state_a,
            0,
            admission_route="photometric_drift_channel_gate",
        )
        manager._replace_budgeted_expert(0, 2, {})

        candidates, history_match = manager._passive_history_reentry_profile(
            [
                state_a,
                member(3, 40.0, patch_features(1.0, 0.0)),
            ]
        )
        self.assertIsNone(candidates)
        self.assertIsNone(history_match)
        self.assertEqual(
            manager.last_history_reentry_event["reason"],
            "insufficient_post_retirement_evidence",
        )

        candidates, history_match = manager._passive_history_reentry_profile(
            [
                member(3, 40.0, patch_features(1.0, 0.0)),
                member(4, 41.0, patch_features(1.0, 0.0)),
            ]
        )
        self.assertEqual([member["image_id"] for member in candidates], [4, 3])
        self.assertTrue(history_match["matched"])
        self.assertEqual(
            manager.last_history_reentry_event["observer"],
            "passive_history_reentry",
        )

    def test_history_window_reentry_uses_only_anchor_consistent_history(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            cold_start_lab_enabled=True,
            photometric_history_reentry_enabled=True,
            photometric_history_reentry_members=2,
        )

        def member(step, value, features):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [value, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
            )

        state_a = member(0, 40.0, patch_features(1.0, 0.0))
        state_b = member(1, 80.0, patch_features(0.0, 1.0))
        manager._admit_quarantined_member(
            state_a,
            0,
            admission_route="photometric_drift_channel_gate",
        )
        manager._admit_quarantined_member(
            state_b,
            1,
            admission_route="photometric_drift_channel_gate",
        )
        manager._replace_budgeted_expert(1, 2, {})
        manager.photometric_episode_start_step = 10
        candidates = [
            member(8, 40.0, patch_features(1.0, 0.0)),
            member(9, 80.0, patch_features(0.0, 1.0)),
            member(10, 80.0, patch_features(0.0, 1.0)),
        ]

        profile, match = manager._history_reentry_candidate_profile(
            candidates,
            trigger_expert_id=0,
            trigger_lab_center=[
                40.0, 2.0, 1.0, 3.0, 2.0, 4.0
            ],
        )

        self.assertTrue(profile["consistent"])
        self.assertTrue(match["matched"])
        self.assertEqual(match["matched_retired_expert_id"], 1)
        self.assertEqual(
            manager.last_history_reentry_event["candidate_member_count"],
            2,
        )
        self.assertEqual(
            manager.last_history_reentry_event["episode_start_step"],
            10,
        )

    def test_retired_expert_history_keeps_distinct_cluster_lineage(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            cold_start_lab_enabled=True,
            photometric_state_distribution_enabled=True,
        )

        def member(step, value):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [value, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
            )

        manager._admit_quarantined_member(
            member(0, 40.0),
            0,
            admission_route="photometric_drift_channel_gate",
        )
        manager._admit_quarantined_member(
            member(1, 41.0),
            1,
            admission_route="photometric_drift_channel_gate",
        )
        manager._replace_budgeted_expert(0, 2, {})
        manager._sync_historical_states(step=2)

        expert_states = [
            state
            for state in manager.historical_states
            if state.get("source_expert_id") is not None
        ]
        self.assertEqual(len(expert_states), 2)
        self.assertEqual(
            {int(state["cluster_id"]) for state in expert_states},
            {0, 1},
        )
        self.assertEqual(
            {
                state["source_type"]
                for state in expert_states
            },
            {"active", "retired"},
        )

    def test_fragmented_committee_support_is_not_coherent_coverage(self):
        manager = DynamicExpertManager(committee_cap=2)
        manager.active_experts = [
            {"expert_id": 0, "lifecycle_state": "active"},
            {"expert_id": 1, "lifecycle_state": "active"},
        ]
        candidates = [
            {"current_expert_supports": {0: 0.8, 1: 0.1}},
            {"current_expert_supports": {0: 0.1, 1: 0.8}},
        ]

        profile = manager._history_reentry_support_profile(
            candidates,
            support_threshold=0.5,
        )

        self.assertTrue(profile["fully_covered"])
        self.assertFalse(profile["coherently_covered"])
        self.assertEqual(
            profile["expert_support_fractions"],
            {"0": 0.5, "1": 0.5},
        )

    def test_history_reentry_reuses_cluster_and_swaps_at_full_budget(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            cold_start_lab_enabled=True,
            novel_admission_mode="channel_gate",
            photometric_candidate_enabled=True,
            photometric_candidate_window_mode="dynamic_low_risk",
            photometric_dynamic_min_low_risk_members=3,
            photometric_dynamic_max_low_risk_members=6,
            photometric_min_low_risk_members=3,
            photometric_history_reentry_enabled=True,
            photometric_history_reentry_members=2,
            photometric_adaptive_candidate_evidence=True,
            photometric_episode_enabled=True,
        )

        def member(step, value, features, supports):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [value, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports=supports,
            )

        state_a = member(0, 40.0, patch_features(1.0, 0.0), {})
        state_b = member(1, 80.0, patch_features(0.0, 1.0), {})
        state_c = member(2, 120.0, patch_features(-1.0, 0.0), {})
        manager._admit_quarantined_member(
            state_a,
            0,
            admission_route="photometric_drift_channel_gate",
        )
        manager._admit_quarantined_member(
            state_b,
            1,
            admission_route="photometric_drift_channel_gate",
        )
        manager._replace_budgeted_expert(1, 2, {})
        manager._admit_quarantined_member(
            state_c,
            2,
            admission_route="photometric_drift_channel_gate",
        )
        manager.coverage_history = [
            {
                "step": 2,
                "support_threshold": 0.5,
                "supports": {0: 0.8, 2: 0.1},
            }
        ]
        manager.photometric_episode_active = True
        manager.photometric_episode_pending = False

        admitted_id = None
        for step, value in ((3, 80.0), (4, 81.0)):
            current = member(
                step,
                value,
                patch_features(0.0, 1.0),
                {0: 0.1, 2: 0.1},
            )
            _, admitted_id = manager._advance_novel_quarantine(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(0.0, 1.0),
                ms_score=current["ms_score"],
                ms_short_score=current["ms_short_score"],
                ms_short_ratio=current["ms_short_ratio"],
                channel_support=current["channel_support"],
                provisional_channel_support=current[
                    "provisional_channel_support"
                ],
                trigger_route=("photometric_drift" if step == 3 else None),
                memory_patch_ids=current["memory_patch_ids"],
                lab_feature=torch.tensor(current["lab_feature"]),
                photometric_trigger_expert_id=0,
                photometric_trigger_lab_center=[
                    40.0, 2.0, 1.0, 3.0, 2.0, 4.0
                ],
                historical_ms_threshold=0.5,
                current_expert_supports={0: 0.1, 2: 0.1},
                support_threshold=0.5,
            )

        self.assertEqual(admitted_id, 3)
        restored = next(
            expert
            for expert in manager.active_experts
            if expert["expert_id"] == admitted_id
        )
        self.assertEqual(restored["cluster_id"], 1)
        self.assertEqual(restored["reentry_from_expert_id"], 1)
        self.assertEqual(restored["image_id"], state_b["image_id"])
        self.assertEqual(
            restored["reentry_confirmation_member_image_ids"],
            [3, 4],
        )
        self.assertEqual(len(manager.clusters), 3)
        self.assertEqual(len(manager.active_experts), 2)
        self.assertEqual(
            {expert["expert_id"] for expert in manager.active_experts},
            {0, 3},
        )
        self.assertEqual(
            manager.last_history_reentry_event["decision"],
            "history_restored_by_replacement",
        )
        self.assertEqual(
            manager.last_history_reentry_event["victim_expert_id"],
            2,
        )
        self.assertEqual(
            manager.last_budget_replacement["reason"],
            "history_reentry_swap",
        )

    def test_history_reentry_skips_restore_when_current_expert_covers_state(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            cold_start_lab_enabled=True,
            novel_admission_mode="channel_gate",
            photometric_candidate_enabled=True,
            photometric_candidate_window_mode="dynamic_low_risk",
            photometric_dynamic_min_low_risk_members=3,
            photometric_dynamic_max_low_risk_members=6,
            photometric_min_low_risk_members=3,
            photometric_history_reentry_enabled=True,
            photometric_history_reentry_members=2,
            photometric_adaptive_candidate_evidence=True,
            photometric_episode_enabled=True,
        )

        def member(step, value, features, supports):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [value, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports=supports,
            )

        state_a = member(0, 40.0, patch_features(1.0, 0.0), {})
        state_b = member(1, 80.0, patch_features(0.0, 1.0), {})
        manager._admit_quarantined_member(
            state_a,
            0,
            admission_route="photometric_drift_channel_gate",
        )
        manager._admit_quarantined_member(
            state_b,
            1,
            admission_route="photometric_drift_channel_gate",
        )
        manager._replace_budgeted_expert(1, 2, {})
        manager.photometric_episode_active = True
        manager.photometric_episode_pending = False

        admitted_id = None
        for step, value in ((3, 80.0), (4, 81.0)):
            current = member(
                step,
                value,
                patch_features(0.0, 1.0),
                {0: 0.8},
            )
            _, admitted_id = manager._advance_novel_quarantine(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(0.0, 1.0),
                ms_score=current["ms_score"],
                ms_short_score=current["ms_short_score"],
                ms_short_ratio=current["ms_short_ratio"],
                channel_support=current["channel_support"],
                provisional_channel_support=current[
                    "provisional_channel_support"
                ],
                trigger_route=("photometric_drift" if step == 3 else None),
                memory_patch_ids=current["memory_patch_ids"],
                lab_feature=torch.tensor(current["lab_feature"]),
                photometric_trigger_expert_id=0,
                photometric_trigger_lab_center=[
                    40.0, 2.0, 1.0, 3.0, 2.0, 4.0
                ],
                historical_ms_threshold=0.5,
                current_expert_supports={0: 0.8},
                support_threshold=0.5,
            )

        self.assertIsNone(admitted_id)
        self.assertEqual(len(manager.active_experts), 1)
        self.assertEqual(
            manager.last_history_reentry_event["decision"],
            "history_shared_coverage",
        )
        self.assertIsNone(manager.clusters[1]["active_expert_id"])

    def test_continuous_reentry_uses_low_weight_provisional_before_active(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            cold_start_lab_enabled=True,
            novel_admission_mode="channel_gate",
            photometric_history_reentry_enabled=True,
            photometric_history_reentry_members=2,
            photometric_history_reentry_continuous_evidence=True,
            photometric_denstream_support_enabled=True,
        )

        def member(step, value, features):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [value, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports={0: 0.1},
            )

        state_a = member(0, 40.0, patch_features(1.0, 0.0))
        state_b = member(1, 80.0, patch_features(0.0, 1.0))
        manager._admit_quarantined_member(
            state_a, 0, admission_route="photometric_drift_channel_gate"
        )
        manager._admit_quarantined_member(
            state_b, 1, admission_route="photometric_drift_channel_gate"
        )
        manager._replace_budgeted_expert(1, 2, {})
        state_c = member(2, 120.0, patch_features(-1.0, 0.0))
        manager._admit_quarantined_member(
            state_c, 2, admission_route="photometric_drift_channel_gate"
        )
        candidates = [
            member(3, 80.0, patch_features(0.0, 1.0)),
            member(4, 81.0, patch_features(0.0, 1.0)),
        ]
        history_match = manager._historical_state_match(candidates)
        active_before = manager.active_experts_before_step()
        replacement_before = manager.last_budget_replacement

        _, admitted_id = manager._advance_history_reentry_provisional(
            history_match,
            candidates,
            step=4,
            support_threshold=0.5,
            admission_route="history_reentry_observer_channel_gate",
        )

        self.assertIsNone(admitted_id)
        self.assertEqual(manager.active_experts_before_step(), active_before)
        self.assertIs(manager.last_budget_replacement, replacement_before)
        provisional = manager.provisional_expert_before_step()
        self.assertEqual(provisional["state"], "history_reentry_provisional")
        self.assertEqual(provisional["weight"], 0.25)
        self.assertEqual(
            manager.last_history_reentry_event["support_event_count"], 1
        )
        self.assertIn(
            "history_reentry_provisional_created",
            manager.last_history_reentry_event["lifecycle_events"],
        )

    def test_continuous_reentry_matures_on_independent_support_event(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            cold_start_lab_enabled=True,
            novel_admission_mode="channel_gate",
            photometric_history_reentry_enabled=True,
            photometric_history_reentry_members=2,
            photometric_history_reentry_continuous_evidence=True,
            photometric_denstream_support_enabled=True,
        )

        def member(step, value, features):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [value, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports={0: 0.1},
            )

        state_a = member(0, 40.0, patch_features(1.0, 0.0))
        state_b = member(1, 80.0, patch_features(0.0, 1.0))
        manager._admit_quarantined_member(
            state_a, 0, admission_route="photometric_drift_channel_gate"
        )
        manager._admit_quarantined_member(
            state_b, 1, admission_route="photometric_drift_channel_gate"
        )
        manager._replace_budgeted_expert(1, 2, {})
        first = [
            member(3, 80.0, patch_features(0.0, 1.0)),
            member(4, 81.0, patch_features(0.0, 1.0)),
        ]
        history_match = manager._historical_state_match(first)
        manager._advance_history_reentry_provisional(
            history_match,
            first,
            step=4,
            support_threshold=0.5,
            admission_route="history_reentry_observer_channel_gate",
        )
        second = [
            first[-1],
            member(5, 80.5, patch_features(0.0, 1.0)),
        ]

        _, admitted_id = manager._advance_history_reentry_provisional(
            history_match,
            second,
            step=5,
            support_threshold=0.5,
            admission_route="history_reentry_observer_channel_gate",
        )

        self.assertIsNotNone(
            admitted_id,
            (manager.last_history_reentry_event, manager.confirmed_provisional),
        )
        self.assertIsNone(manager.confirmed_provisional)
        restored = next(
            expert
            for expert in manager.active_experts
            if int(expert["expert_id"]) == int(admitted_id)
        )
        self.assertEqual(restored["reentry_from_expert_id"], 1)
        self.assertEqual(
            manager.last_history_reentry_event["decision"],
            "history_restored_free_slot",
        )
        self.assertIn(
            "reentry_activated",
            manager.last_history_reentry_event["lifecycle_events"],
        )
        self.assertIn(
            "history_reentry_provisional_matured",
            manager.last_history_reentry_event["lifecycle_events"],
        )

    def test_continuous_reentry_provisional_decays_and_discards(self):
        manager = DynamicExpertManager(
            novel_admission_mode="channel_gate",
            novel_confirmed_ttl=2,
            photometric_history_reentry_enabled=True,
            photometric_history_reentry_continuous_evidence=True,
            photometric_denstream_support_enabled=True,
        )
        manager.confirmed_provisional = {
            "kind": "history_reentry",
            "members": [],
            "candidates": [],
            "start_step": 4,
            "last_support_step": 4,
            "last_decay_step": 4,
            "ttl": 2,
            "matched_state_id": 3,
            "effective_support": 0.4,
            "support_event_count": 1,
        }

        manager._decay_history_reentry_provisional(step=6)

        self.assertIsNone(manager.confirmed_provisional)
        self.assertEqual(
            manager.last_history_reentry_event["decision"],
            "history_reentry_provisional_discarded",
        )
        self.assertEqual(
            manager.last_history_reentry_event["reason"],
            "maximum_wait_timeout",
        )
        self.assertIn(
            "history_reentry_provisional_rolled_back",
            manager.last_history_reentry_event["lifecycle_events"],
        )
        self.assertIn(
            "history_reentry_provisional_timeout",
            manager.last_history_reentry_event["lifecycle_events"],
        )

    def test_historical_restore_reuses_formal_admission_weight(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            cold_start_lab_enabled=True,
            photometric_history_reentry_enabled=True,
        )
        original = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            memory_patch_ids=[0],
            lab_feature=torch.tensor(
                [50.0, 2.0, 128.0, 3.0, 128.0, 4.0]
            ),
            historical_ms_threshold=0.5,
        )
        _, original_expert_id = manager._admit_quarantined_member(
            original,
            0,
            cluster_members=[
                original,
                {**original, "step": 1, "image_id": 1},
            ],
            admission_route="photometric_drift_channel_gate",
        )
        manager._replace_budgeted_expert(original_expert_id, 1, {})
        candidates = [
            {
                **original,
                "step": step,
                "image_id": step,
            }
            for step in (2, 3)
        ]
        history_match = manager._historical_state_match(candidates)
        manager._prepare_formal_admission = lambda *_args, **_kwargs: (
            True,
            0.25,
            None,
        )

        _, restored_id = manager._activate_historical_state(
            history_match,
            candidates,
            step=3,
            support_threshold=0.5,
            admission_route="history_reentry_observer_channel_gate",
        )
        restored = next(
            expert
            for expert in manager.active_experts
            if int(expert["expert_id"]) == int(restored_id)
        )

        self.assertEqual(restored["scoring_weight"], 0.25)
        self.assertEqual(
            restored["confidence_source"],
            "unified_delta_j_dino_score",
        )

    def test_continuous_reentry_rolls_back_after_activation_rejection(self):
        manager = DynamicExpertManager(
            novel_admission_mode="channel_gate",
            photometric_history_reentry_enabled=True,
            photometric_history_reentry_continuous_evidence=True,
            photometric_denstream_support_enabled=True,
        )
        candidate = manager._quarantine_member(
            step=5,
            image_id=5,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            memory_patch_ids=[0],
            historical_ms_threshold=0.5,
            current_expert_supports={},
        )
        previous_candidate = {
            **candidate,
            "step": 4,
            "image_id": 4,
        }
        manager.confirmed_provisional = {
            "kind": "history_reentry",
            "members": [previous_candidate],
            "candidates": [previous_candidate],
            "start_step": 4,
            "last_support_step": 4,
            "last_decay_step": 4,
            "ttl": 4,
            "matched_state_id": 3,
            "effective_support": 0.6,
            "support_event_count": 1,
        }
        manager.historical_states = [{"state_id": 3}]

        def reject_activation(*args, **kwargs):
            manager.last_history_reentry_event = {
                "decision": "history_reentry_rejected",
                "reason": "virtual_swap_rejected",
            }
            return None, None

        manager._activate_historical_state = reject_activation

        _, admitted_id = manager._advance_history_reentry_provisional(
            {"matched_state_id": 3, "matched": True},
            [previous_candidate, candidate],
            step=5,
            support_threshold=0.5,
            admission_route="history_reentry_observer_channel_gate",
            candidate_evidence={
                "risk_profile": {"acceptable": True},
                "temporary_support_evidence": {"increment": 0.6},
            },
        )

        self.assertIsNone(admitted_id)
        self.assertIsNone(manager.confirmed_provisional)
        self.assertEqual(manager.historical_states, [{"state_id": 3}])
        self.assertIn(
            "history_reentry_provisional_rolled_back",
            manager.last_history_reentry_event["lifecycle_events"],
        )

    def test_history_reentry_prefers_dormant_victim(self):
        manager = DynamicExpertManager(committee_cap=2)
        member_a = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.2,
            provisional_channel_support=0.5,
            memory_patch_ids=[0],
            lab_feature=torch.tensor([40.0, 2.0, 1.0, 3.0, 2.0, 4.0]),
            historical_ms_threshold=0.5,
            current_expert_supports={},
        )
        member_b = manager._quarantine_member(
            step=1,
            image_id=1,
            dino_patch_features=patch_features(0.0, 1.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.2,
            provisional_channel_support=0.5,
            memory_patch_ids=[0],
            lab_feature=torch.tensor([80.0, 2.0, 1.0, 3.0, 2.0, 4.0]),
            historical_ms_threshold=0.5,
            current_expert_supports={},
        )
        manager._admit_quarantined_member(
            member_a,
            0,
            admission_route="photometric_drift_channel_gate",
        )
        manager._admit_quarantined_member(
            member_b,
            1,
            admission_route="photometric_drift_channel_gate",
        )
        manager.active_experts[1]["lifecycle_state"] = "dormant"
        candidate = dict(member_a)
        candidate["current_expert_supports"] = {0: 0.1}

        victim_id, audit = manager._select_history_reentry_victim(
            [candidate],
            support_threshold=0.5,
            admission_route="photometric_drift_history_reentry_channel_gate",
        )

        self.assertEqual(victim_id, 1)
        self.assertEqual(audit["victim_lifecycle_state"], "dormant")

    def test_photometric_only_state_is_matched_and_materialized(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            cold_start_lab_enabled=True,
            photometric_history_reentry_enabled=True,
            photometric_history_reentry_members=2,
            photometric_adaptive_candidate_evidence=True,
            photometric_budget_admission_mode="channel_or_lab",
        )

        def member(step, score=0.1):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=score,
                ms_short_score=score,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [60.0, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports={},
            )

        manager._record_photometric_state([member(0)], step=0)
        candidates = [member(1), member(2)]
        match = manager._historical_state_match(candidates)

        self.assertTrue(match["matched"])
        self.assertEqual(match["matched_source_type"], "photometric_only")
        cluster_id, expert_id = manager._activate_historical_state(
            match,
            candidates,
            step=2,
            support_threshold=0.5,
            admission_route="photometric_drift_history_reentry_channel_gate",
            candidate_evidence={
                "history_match": match,
                "new_coverage_fraction": None,
                "new_coverage_count": 0,
                "candidate_observation_count": 0,
            },
        )

        self.assertEqual(cluster_id, 0)
        self.assertEqual(expert_id, 0)
        self.assertEqual(
            manager.last_history_reentry_event["decision"],
            "history_materialized_free_slot",
        )
        self.assertEqual(len(manager.historical_states), 1)
        self.assertEqual(
            manager.historical_states[0]["active_expert_id"],
            expert_id,
        )
        self.assertTrue(manager.historical_states[0]["has_patch_memory"])

    def test_history_joint_risk_allows_one_mild_member(self):
        manager = DynamicExpertManager(
            cold_start_lab_enabled=True,
            photometric_history_reentry_enabled=True,
            photometric_history_reentry_members=2,
        )

        def member(step, score):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=score,
                ms_short_score=score,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [60.0, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports={},
            )

        manager._record_photometric_state([member(0, 0.1)], step=0)
        candidates = [member(1, 0.1), member(2, 0.6)]
        evidence = manager._candidate_admission_evidence(
            candidates,
            {
                "consistent": True,
                "selected_inlier_image_ids": [1, 2],
            },
            support_threshold=0.5,
        )

        self.assertEqual(evidence["strict_low_risk_count"], 1)
        self.assertTrue(evidence["risk_profile"]["acceptable"])
        self.assertEqual(evidence["decision"], "update_existing_state")

    def test_history_joint_risk_rejects_extreme_member(self):
        manager = DynamicExpertManager(
            cold_start_lab_enabled=True,
            photometric_history_reentry_enabled=True,
            photometric_history_reentry_members=2,
        )

        def member(step, score):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=score,
                ms_short_score=score,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [60.0, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports={},
            )

        manager._record_photometric_state([member(0, 0.1)], step=0)
        candidates = [member(1, 0.1), member(2, 1.1)]
        evidence = manager._candidate_admission_evidence(
            candidates,
            {
                "consistent": True,
                "selected_inlier_image_ids": [1, 2],
            },
            support_threshold=0.5,
        )

        self.assertFalse(evidence["risk_profile"]["acceptable"])
        self.assertEqual(evidence["decision"], "keep_collecting")

    def test_historical_state_registry_is_bounded(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            photometric_history_max=4,
            photometric_history_min=2,
            photometric_state_registry_cap=4,
        )
        for step in range(6):
            member = manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [20.0 * step, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports={},
            )
            manager._record_photometric_state([member], step=step)

        self.assertEqual(len(manager.historical_states), 4)
        self.assertEqual(
            [state["last_seen_step"] for state in manager.historical_states],
            [2, 3, 4, 5],
        )

    def test_state_registry_defaults_to_twice_committee_cap(self):
        manager = DynamicExpertManager(committee_cap=2)

        self.assertEqual(manager.photometric_state_registry_cap, 4)

        with self.assertRaises(ValueError):
            DynamicExpertManager(
                committee_cap=2,
                photometric_state_registry_cap=1,
            )

    def test_state_lifecycle_records_create_match_update_mature_and_merge(self):
        manager = DynamicExpertManager(
            committee_cap=1,
            photometric_state_registry_cap=2,
            cold_start_lab_enabled=True,
            photometric_state_distribution_enabled=True,
            photometric_temporary_state_enabled=True,
            photometric_history_reentry_enabled=True,
        )

        def member(step, value):
            return manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=patch_features(1.0, 0.0),
                ms_score=0.1,
                ms_short_score=0.1,
                ms_short_ratio=0.8,
                channel_support=0.2,
                provisional_channel_support=0.5,
                memory_patch_ids=[0],
                lab_feature=torch.tensor(
                    [value, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                historical_ms_threshold=0.5,
                current_expert_supports={},
            )

        first = manager._record_photometric_state(
            [member(0, 60.0)],
            step=0,
            state_status="temporary",
            source_type="temporary",
        )
        match = manager._historical_state_match(
            [member(1, 60.5), member(2, 60.8)]
        )
        updated = manager._update_matched_historical_state(
            match,
            [member(1, 60.5), member(2, 60.8)],
            step=2,
        )
        manager._record_photometric_state(
            [member(3, 61.0)],
            step=3,
            state_status="temporary",
            source_type="temporary",
        )

        event_types = {
            event["event_type"] for event in manager.state_lifecycle_events
        }
        self.assertEqual(first["state_status"], "temporary")
        self.assertEqual(updated["state_status_after"], "stable")
        self.assertTrue(
            {"create", "match", "update", "mature", "merge"}
            <= event_types
        )
        required_fields = {
            "event_type",
            "step",
            "state_id",
            "matched_or_merged_state_id",
            "source_type_before",
            "source_type_after",
            "status_before",
            "status_after",
            "active_expert_id",
            "support_count",
            "support_event_count",
            "member_count",
            "reason",
        }
        self.assertTrue(
            all(
                required_fields <= set(event)
                for event in manager.state_lifecycle_events
            )
        )
        self.assertEqual(len(manager.historical_states), 1)

    def test_registry_evicts_temporary_before_active_state(self):
        manager = DynamicExpertManager(
            committee_cap=1,
            photometric_state_registry_cap=2,
            photometric_state_distribution_enabled=True,
        )

        def add_state(step, center, embedding, status, active_expert_id=None):
            return manager._upsert_historical_state(
                source_type=(
                    "active"
                    if active_expert_id is not None
                    else "temporary"
                ),
                step=step,
                lab_center=np.asarray(
                    [center, 2.0, 1.0, 3.0, 2.0, 4.0]
                ),
                dino_center=torch.tensor(embedding, dtype=torch.float32),
                support_count=1,
                active_expert_id=active_expert_id,
                state_status=status,
            )

        active = add_state(0, 20.0, [1.0, 0.0], "stable", 7)
        first_temporary = add_state(1, 60.0, [0.0, 1.0], "temporary")
        add_state(2, 90.0, [-1.0, 0.0], "temporary")

        remaining_ids = {
            state["state_id"] for state in manager.historical_states
        }
        evicted = [
            event
            for event in manager.state_lifecycle_events
            if event["event_type"] == "evict"
        ]
        self.assertEqual(len(manager.historical_states), 2)
        self.assertIn(active["state_id"], remaining_ids)
        self.assertNotIn(first_temporary["state_id"], remaining_ids)
        self.assertEqual(evicted[-1]["state_id"], first_temporary["state_id"])

    def test_build_lab_state_accepts_legacy_member_without_lab_feature(self):
        manager = DynamicExpertManager(cold_start_lab_enabled=True)
        members = [{"step": 0, "image_id": 0}]

        self.assertIsNone(manager._build_lab_state(members))

    def test_photometric_handled_state_blocks_repeat_then_rearms(self):
        manager = DynamicExpertManager(
            photometric_candidate_enabled=True,
            photometric_gate_mode="lab_confirmed",
            photometric_episode_enabled=True,
            photometric_exit_confirm_steps=2,
            photometric_local_lab_distance_threshold=0.04,
        )
        manager.active_experts = [{"expert_id": 3}]
        shifted = {
            "expert_id": 3,
            "current_lab_feature": [60.0, 2.0, 1.0, 3.0, 2.0, 4.0],
            "lab_center": [40.0, 2.0, 1.0, 3.0, 2.0, 4.0],
            "l_ewma": 0.08,
            "ab_ewma": 0.0,
            "l_threshold": 0.02,
            "ab_threshold": 0.02,
            "drift_confirmed": True,
        }

        triggered, _ = manager._advance_photometric_gate(
            shifted,
            support_threshold=0.5,
            expert_channel_supports={3: 0.9},
            step=10,
        )
        self.assertTrue(triggered)
        manager._finish_photometric_episode(
            [60.0, 2.0, 1.0, 3.0, 2.0, 4.0]
        )

        triggered, _ = manager._advance_photometric_gate(
            {
                **shifted,
                "current_lab_feature": [61.0, 2.0, 1.0, 3.0, 2.0, 4.0],
            },
            support_threshold=0.5,
            expert_channel_supports={3: 0.9},
            step=11,
        )
        self.assertFalse(triggered)
        self.assertTrue(
            manager.last_photometric_gate["handled_state_blocked"]
        )

        for step, lightness in ((12, 70.0), (13, 71.0)):
            triggered, _ = manager._advance_photometric_gate(
                {
                    **shifted,
                    "current_lab_feature": [
                        lightness, 2.0, 1.0, 3.0, 2.0, 4.0
                    ],
                },
                support_threshold=0.5,
                expert_channel_supports={3: 0.9},
                step=step,
            )

        self.assertTrue(triggered)
        self.assertIsNone(manager.photometric_handled_lab_center)
        self.assertTrue(manager.photometric_episode_active)

    def test_photometric_episode_triggers_once_and_aborts_after_recovery(self):
        manager = DynamicExpertManager(
            photometric_candidate_enabled=True,
            photometric_gate_mode="lab_confirmed",
            photometric_episode_enabled=True,
            photometric_exit_confirm_steps=2,
        )
        manager.active_experts = [{"expert_id": 3}]
        shifted = {
            "expert_id": 3,
            "lab_center": [40.0, 2.0, 1.0, 3.0, 2.0, 4.0],
            "l_ewma": 0.08,
            "ab_ewma": 0.0,
            "l_threshold": 0.02,
            "ab_threshold": 0.02,
            "drift_confirmed": True,
        }
        stable = {
            **shifted,
            "l_ewma": 0.0,
            "drift_confirmed": False,
        }

        triggered, expert_id = manager._advance_photometric_gate(
            shifted,
            support_threshold=0.5,
            expert_channel_supports={3: 0.9},
            step=10,
        )
        self.assertTrue(triggered)
        self.assertEqual(expert_id, 3)
        self.assertTrue(manager.photometric_episode_pending)

        manager.photometric_episode_pending = False
        manager.novel_quarantine = [{"step": 10, "image_id": 10}]
        manager.novel_quarantine_route = "photometric_drift"
        triggered, _ = manager._advance_photometric_gate(
            shifted,
            support_threshold=0.5,
            expert_channel_supports={3: 0.9},
            step=11,
        )
        self.assertFalse(triggered)

        manager._advance_photometric_gate(
            stable,
            support_threshold=0.5,
            expert_channel_supports={3: 0.9},
            step=12,
        )
        self.assertIsNone(manager.photometric_pool_abort_reason)
        manager._advance_photometric_gate(
            stable,
            support_threshold=0.5,
            expert_channel_supports={3: 0.9},
            step=13,
        )
        self.assertEqual(
            manager.photometric_pool_abort_reason,
            "aborted_photometric_recovered",
        )
        manager._abort_photometric_quarantine(
            13,
            manager.photometric_pool_abort_reason,
        )
        self.assertIsNone(manager.novel_quarantine)
        self.assertFalse(manager.photometric_episode_active)
        self.assertEqual(
            manager.last_novel_quarantine_event["decision"],
            "aborted_photometric_recovered",
        )

    def test_minimum_active_expert_retains_last_coverage(self):
        manager = DynamicExpertManager(
            cold_start_mode="adaptive",
            cold_start_min_samples=4,
            cold_start_max_samples=4,
            cold_start_required_members=3,
            base_ttl=1,
            max_ttl=1,
            minimum_active_experts=1,
        )
        features = patch_features(1.0, 0.0)
        for step in range(4):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=None if step == 0 else 0.1,
                channel_support=None if step == 0 else 1.0,
                expert_channel_supports={},
            )

        deleted, events = manager._advance_lifecycle(
            step=4,
            support_threshold=0.5,
            expert_channel_supports={0: 0.0},
        )

        self.assertEqual(deleted, [])
        self.assertEqual(len(manager.active_experts), 1)
        self.assertEqual(manager.active_experts[0]["ttl"], 1)
        self.assertEqual(events[0]["decision"], "coverage_floor_retained")
        self.assertTrue(events[0]["coverage_floor_retained"])

    def test_dormant_lifecycle_keeps_slot_and_reactivates(self):
        manager = DynamicExpertManager(
            cold_start_mode="adaptive",
            cold_start_min_samples=4,
            cold_start_max_samples=4,
            cold_start_required_members=3,
            base_ttl=1,
            max_ttl=1,
            lifecycle_mode="dormant_replace",
        )
        features = patch_features(1.0, 0.0)
        for step in range(4):
            manager.advance(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=None if step == 0 else 0.1,
                channel_support=None if step == 0 else 1.0,
                expert_channel_supports={},
            )

        deleted, events = manager._advance_lifecycle(
            step=4,
            support_threshold=0.5,
            expert_channel_supports={0: 0.0},
            current_expert_supports={0: 0.0},
        )

        self.assertEqual(deleted, [])
        self.assertEqual(manager.active_experts[0]["lifecycle_state"], "dormant")
        self.assertEqual(manager.active_experts_before_step(), [])
        self.assertEqual(len(manager.expert_slots_before_step()), 1)
        self.assertTrue(events[0]["became_dormant_this_step"])

        deleted, events = manager._advance_lifecycle(
            step=5,
            support_threshold=0.5,
            expert_channel_supports={0: 0.0},
            current_expert_supports={0: 0.9},
        )

        self.assertEqual(deleted, [])
        self.assertEqual(manager.active_experts[0]["lifecycle_state"], "active")
        self.assertEqual(len(manager.active_experts_before_step()), 1)
        self.assertTrue(events[0]["reactivated_this_step"])

    def test_budget_replacement_prefers_dormant_slot(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            lifecycle_mode="dormant_replace",
            budget_min_new_coverage_fraction=0.5,
        )
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        second = {
            **first,
            "step": 1,
            "image_id": 1,
            "embedding": manager._image_embedding(
                patch_features(0.0, 1.0)
            ),
        }
        manager._admit_quarantined_member(first, 0, admission_route="legacy")
        manager._admit_quarantined_member(second, 1, admission_route="legacy")
        manager.active_experts[1]["lifecycle_state"] = "dormant"
        manager.coverage_history = [
            {
                "step": 5,
                "support_threshold": 0.5,
                "supports": {0: 0.9, 1: 0.0},
            }
        ]
        candidate_members = [
            {
                **first,
                "step": step,
                "image_id": step,
                "embedding": manager._image_embedding(
                    patch_features(-1.0, 0.0)
                ),
                "current_expert_supports": {0: 0.0, 1: 0.0},
            }
            for step in (2, 3)
        ]

        _, admitted_id = manager._admit_quarantined_member(
            candidate_members[0],
            step=3,
            cluster_members=candidate_members,
            admission_route="photometric_drift_channel_gate",
            support_threshold=0.5,
        )

        self.assertEqual(admitted_id, 2)
        self.assertEqual(
            [expert["expert_id"] for expert in manager.active_experts],
            [0, 2],
        )
        self.assertEqual(manager.retired_experts[-1]["expert_id"], 1)
        self.assertEqual(
            manager.last_budget_replacement["victim_lifecycle_state"],
            "dormant",
        )

    def test_diversity_audit_reuses_coverage_history_without_mutation(self):
        manager = DynamicExpertManager(committee_cap=2)
        manager.active_experts = [
            {
                "expert_id": 0,
                "lifecycle_state": "active",
                "scoring_weight": 1.0,
            },
            {
                "expert_id": 1,
                "lifecycle_state": "active",
                "scoring_weight": 0.25,
            },
        ]
        manager.coverage_history = [
            {
                "step": 2,
                "support_threshold": 0.5,
                "supports": {0: 0.8, 1: 0.7},
            },
            {
                "step": 3,
                "support_threshold": 0.5,
                "supports": {0: 0.8, 1: 0.1},
            },
            {
                "step": 4,
                "support_threshold": 0.5,
                "supports": {0: 0.1, 1: 0.2},
            },
        ]
        history_before = [dict(record) for record in manager.coverage_history]

        audit = manager.diversity_audit_before_step()

        self.assertEqual(audit["comparable_state_count"], 3)
        self.assertEqual(audit["covered_state_count"], 2)
        self.assertEqual(audit["redundantly_covered_state_count"], 1)
        self.assertAlmostEqual(audit["state_coverage_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(audit["active_expert_redundancy"], 0.5)
        self.assertLess(
            audit["effective_redundancy"],
            audit["active_expert_redundancy"],
        )
        self.assertAlmostEqual(
            audit["effective_redundancy"],
            ((0.8 + 0.25 * 0.7 - 0.8) / (0.8 + 0.25 * 0.7)) / 2,
        )
        profile = audit["expert_coverage_profiles"][0]
        self.assertEqual(profile["exclusive_coverage_count"], 1)
        self.assertAlmostEqual(profile["coverage_removal_loss"], 1.0 / 3.0)
        self.assertEqual(audit["redundant_without_benefit_count"], 0)
        self.assertEqual(manager.coverage_history, history_before)

    def test_dino_removal_loss_makes_overlapping_expert_nonredundant(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            budget_patch_coverage_enabled=True,
        )
        first = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        second = manager._quarantine_member(
            step=1,
            image_id=1,
            dino_patch_features=patch_features(0.0, 1.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            historical_ms_threshold=0.5,
        )
        manager._admit_quarantined_member(first, 0, admission_route="legacy")
        manager._admit_quarantined_member(second, 1, admission_route="legacy")
        manager.coverage_history = [
            {
                "step": step,
                "image_id": step,
                "support_threshold": 0.5,
                "supports": {0: 0.9, 1: 0.9},
                "embedding": member["embedding"],
                "patch_prototypes": member["patch_prototypes"],
                "lab_feature": None,
            }
            for step, member in enumerate((first, second), start=2)
        ]

        audit = manager.diversity_audit_before_step()

        self.assertEqual(audit["active_expert_redundancy"], 1.0)
        self.assertEqual(audit["redundant_without_benefit_count"], 0)
        self.assertTrue(
            all(
                profile["dino_coverage_removal_loss"] > 0.0
                for profile in audit["expert_coverage_profiles"].values()
            )
        )

    def test_combined_fitness_denstream_does_not_use_passive_reentry(self):
        manager = DynamicExpertManager(
            photometric_candidate_window_mode="history",
            photometric_fitness_history_enabled=True,
            photometric_fitness_adaptive_evidence_enabled=True,
            photometric_denstream_support_enabled=True,
            photometric_temporary_state_enabled=True,
            budget_unified_delta_j_dino_score_enabled=True,
            photometric_history_reentry_enabled=True,
        )
        manager._passive_history_reentry_profile = lambda _members: self.fail(
            "combined path must let FITNESS select history before DenStream"
        )

        manager._consider_candidate(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=None,
            channel_support=None,
            ms_threshold=None,
            support_threshold=None,
            provisional_channel_support=None,
            novel_ms_threshold=None,
            novel_support_threshold=None,
            ms_short_score=None,
            ms_short_ratio=None,
            memory_patch_ids=[0],
            lab_feature=torch.zeros(6),
            lab_drift_profile={},
        )

    def test_combined_fitness_denstream_still_decays_reentry_provisional(self):
        manager = DynamicExpertManager(
            photometric_candidate_window_mode="history",
            photometric_fitness_history_enabled=True,
            photometric_fitness_adaptive_evidence_enabled=True,
            photometric_denstream_support_enabled=True,
            photometric_temporary_state_enabled=True,
            budget_unified_delta_j_dino_score_enabled=True,
            photometric_history_reentry_enabled=True,
            photometric_history_reentry_continuous_evidence=True,
            novel_confirmed_ttl=5,
        )
        manager.confirmed_provisional = {
            "kind": "history_reentry",
            "members": [],
            "candidates": [],
            "start_step": 0,
            "last_support_step": 0,
            "last_decay_step": 0,
            "ttl": 5,
            "matched_state_id": 3,
            "effective_support": 1.0,
            "support_event_count": 1,
        }

        manager._consider_candidate(
            step=1,
            image_id=1,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=None,
            channel_support=None,
            ms_threshold=None,
            support_threshold=None,
            provisional_channel_support=None,
            novel_ms_threshold=None,
            novel_support_threshold=None,
            ms_short_score=None,
            ms_short_ratio=None,
            memory_patch_ids=[0],
            lab_feature=torch.zeros(6),
            lab_drift_profile={},
        )

        provisional = manager.confirmed_provisional
        self.assertIsNotNone(provisional)
        self.assertEqual(provisional["last_decay_step"], 1)
        self.assertEqual(provisional["ttl"], 4)
        self.assertAlmostEqual(
            provisional["effective_support"],
            manager.photometric_denstream_decay,
        )

    def test_unified_delta_j_dino_redundancy_uses_ratio_once(self):
        manager = DynamicExpertManager(committee_cap=1)
        manager.active_experts = [
            {
                "expert_id": 0,
                "scoring_weight": 1.0,
                "member_reliability": 1.0,
            }
        ]
        manager._coverage_loss_replacement_profiles = lambda *_args: {
            0: {"swap_gain": 0.2, "removal_loss": 0.0}
        }
        manager._expert_coverage_profiles = lambda: {}
        manager._expert_lab_uniqueness_profiles = lambda: {}
        manager._expert_contribution_profiles = lambda *_args: {
            0: {"contribution_score": 0.6}
        }
        manager._candidate_contribution_profile = lambda *_args: {
            "contribution_score": 0.1
        }
        manager._candidate_risk_profile = lambda *_args: {
            "median_risk_ratio": 1.0
        }

        audit = manager._unified_delta_j_shadow(
            [{"step": 1, "member_reliability": 1.0}],
            0.5,
            {},
            {},
            {
                0: {
                    "safe": True,
                    "current_state_support_gain": 0.2,
                    "task_cost": 0.0,
                    "dino_normal_score_shadow": {
                        "available": True,
                        "normal_score_penalty": 0.0,
                    },
                }
            },
        )
        profile = audit["victim_profiles"][0]

        self.assertAlmostEqual(profile["redundancy_ratio"], 0.5)
        self.assertAlmostEqual(
            profile["dino_score_redundancy_penalty"], 0.1
        )

    def test_controlled_contamination_fixture_forces_free_slot_only(self):
        manager = DynamicExpertManager(
            committee_cap=2,
            budget_unified_delta_j_shadow_enabled=True,
            budget_unified_delta_j_enabled=True,
            budget_unified_delta_j_dino_score_enabled=True,
        )
        clean = manager._quarantine_member(
            step=0,
            image_id=0,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            memory_patch_ids=[0],
        )
        manager._admit_quarantined_member(clean, 0, admission_route="legacy")
        contaminated = manager._quarantine_member(
            step=1,
            image_id=1,
            dino_patch_features=patch_features(1.0, 0.0),
            ms_score=0.1,
            ms_short_score=0.1,
            ms_short_ratio=0.8,
            channel_support=0.8,
            provisional_channel_support=0.5,
            memory_patch_ids=[0],
        )

        _, rejected_id = manager._admit_quarantined_member(
            contaminated,
            1,
            admission_route="controlled_contamination",
        )
        self.assertIsNone(rejected_id)
        _, injected_id = manager._admit_quarantined_member(
            contaminated,
            1,
            admission_route="controlled_contamination",
            candidate_evidence={"controlled_contamination": True},
        )

        self.assertIsNotNone(injected_id)
        self.assertEqual(len(manager.active_experts), 2)
        self.assertTrue(
            manager.last_budget_replacement_evaluation[
                "controlled_contamination_forced_admission"
            ]
        )


if __name__ == "__main__":
    unittest.main()
