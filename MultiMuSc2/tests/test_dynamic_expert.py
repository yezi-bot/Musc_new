import unittest

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

    def test_adaptive_evidence_admits_joint_channel_novel_state(self):
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
                "historical_low_risk": False,
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

        self.assertEqual(evidence["decision"], "new_scoring_expert")
        self.assertEqual(evidence["new_coverage_fraction"], 1.0)
        self.assertEqual(evidence["strict_low_risk_count"], 0)

    def test_adaptive_evidence_records_covered_lab_state_only(self):
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

        self.assertEqual(evidence["decision"], "photometric_state_only")
        self.assertEqual(evidence["new_coverage_fraction"], 0.0)

    def test_adaptive_pool_admits_channel_novel_state_after_two_images(self):
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
        for step in range(2):
            _, admitted_id = manager._advance_novel_quarantine(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.8,
                ms_short_score=0.8,
                ms_short_ratio=1.2,
                channel_support=0.1,
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
                current_expert_supports={0: 0.1},
                support_threshold=0.5,
            )

        self.assertEqual(admitted_id, 1)
        self.assertEqual(
            manager.last_novel_quarantine_event["confirmation_reason"],
            "new_scoring_expert",
        )
        self.assertEqual(
            manager.last_novel_quarantine_event["low_risk_count"],
            0,
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
        for step in range(2):
            _, admitted_id = manager._advance_novel_quarantine(
                step=step,
                image_id=step,
                dino_patch_features=features,
                ms_score=0.8,
                ms_short_score=0.8,
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
        self.assertEqual(
            manager.last_novel_quarantine_event["decision"],
            "recorded_photometric_state_only",
        )
        self.assertEqual(len(manager.historical_states), 1)
        self.assertEqual(
            manager.historical_states[-1]["source_type"],
            "photometric_only",
        )
        self.assertEqual(len(manager.active_experts), 1)

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

        self.assertEqual(admitted_id, 2)
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
        self.assertEqual(len(manager.clusters), 2)
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
        self.assertEqual(evidence["decision"], "history_reentry")

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
            [5, 4, 3, 2],
        )

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


if __name__ == "__main__":
    unittest.main()
