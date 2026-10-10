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


if __name__ == "__main__":
    unittest.main()
