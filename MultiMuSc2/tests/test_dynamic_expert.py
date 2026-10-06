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


if __name__ == "__main__":
    unittest.main()
