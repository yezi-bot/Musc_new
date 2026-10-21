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

    def test_dynamic_low_risk_pool_waits_until_max_before_rejection(self):
        manager = DynamicExpertManager(
            novel_admission_mode="quarantine",
            photometric_candidate_enabled=True,
            photometric_candidate_window_mode="dynamic_low_risk",
            photometric_dynamic_min_low_risk_members=3,
            photometric_dynamic_max_low_risk_members=6,
        )
        features = patch_features(1.0, 0.0)
        lightness = [60.0, 80.0, 100.0, 80.0, 80.0, 80.0]

        for step, value in enumerate(lightness):
            manager._advance_novel_quarantine(
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
            if step == 2:
                self.assertEqual(
                    manager.last_novel_quarantine_event["decision"],
                    "collecting_inconsistent",
                )
                self.assertEqual(len(manager.novel_quarantine), 3)

        self.assertEqual(
            manager.last_novel_quarantine_event["decision"],
            "rejected_photometric_inconsistent",
        )
        self.assertIsNone(manager.novel_quarantine)

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
