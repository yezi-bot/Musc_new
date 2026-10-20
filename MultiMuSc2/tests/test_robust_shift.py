import unittest

import torch

from MultiMuSc2.models.modules._DYNAMIC_PIPELINE import (
    _robust_shift_audit_profile,
    _robust_shift_window_audit_profile,
)
from MultiMuSc2.models.modules._ROBUST_SHIFT import (
    pairwise_window_sinkhorn_profile,
    robust_sinkhorn_profile,
    robust_window_sinkhorn_profile,
    uniformly_sample_patch_ids,
)


class RobustShiftTest(unittest.TestCase):
    def setUp(self):
        self.reference = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.patch_ids = torch.arange(4)

    def test_identical_distributions_have_zero_robust_cost(self):
        profile = robust_sinkhorn_profile(
            self.reference,
            self.reference,
            self.patch_ids,
            self.patch_ids,
            grid_size=2,
            position_radius=0,
        )

        self.assertAlmostEqual(profile["sinkhorn_cost"], 0.0, places=6)
        self.assertAlmostEqual(profile["discrete_cost"], 0.0, places=6)
        self.assertAlmostEqual(profile["mutual_fraction"], 1.0, places=6)

    def test_global_shift_increases_robust_cost(self):
        shifted = self.reference + torch.tensor([0.5, 0.0])
        profile = robust_sinkhorn_profile(
            shifted,
            self.reference,
            self.patch_ids,
            self.patch_ids,
            grid_size=2,
            position_radius=0,
        )

        self.assertAlmostEqual(profile["sinkhorn_cost"], 0.5, places=6)
        self.assertAlmostEqual(profile["discrete_cost"], 0.5, places=6)
        self.assertAlmostEqual(profile["mutual_fraction"], 1.0, places=6)

    def test_global_matching_is_not_limited_to_same_position(self):
        profile = robust_sinkhorn_profile(
            self.reference[[3, 2, 1, 0]],
            self.reference,
            self.patch_ids,
            self.patch_ids,
            grid_size=2,
            position_radius=None,
        )

        self.assertAlmostEqual(profile["discrete_cost"], 0.0, places=6)

    def test_mutual_fraction_drops_for_ambiguous_matches(self):
        ambiguous = torch.zeros_like(self.reference)
        profile = robust_sinkhorn_profile(
            ambiguous,
            ambiguous,
            self.patch_ids,
            self.patch_ids,
            grid_size=2,
            position_radius=None,
        )

        self.assertAlmostEqual(profile["mutual_fraction"], 0.25, places=6)

    def test_audit_uses_only_historical_expert(self):
        stream = torch.stack((self.reference, self.reference + 0.1))
        expert = {"expert_id": 3, "image_id": 0}
        profile = _robust_shift_audit_profile(
            stream[1],
            current_step=1,
            active_experts=[expert],
            committee_features=stream,
            grid_size=2,
            max_patches=4,
            epsilon=0.05,
            iterations=20,
        )

        self.assertEqual(profile["nearest_expert_id"], 3)
        self.assertIn("3", profile["expert_profiles"])

        with self.assertRaisesRegex(ValueError, "historical experts"):
            _robust_shift_audit_profile(
                stream[1],
                current_step=1,
                active_experts=[{"expert_id": 4, "image_id": 1}],
                committee_features=stream,
                grid_size=2,
                max_patches=4,
                epsilon=0.05,
                iterations=20,
            )

    def test_uniform_sampling_respects_budget(self):
        ids = uniformly_sample_patch_ids(16, 5, torch.device("cpu"))
        self.assertEqual(ids.numel(), 5)
        self.assertEqual(int(ids[0]), 0)
        self.assertEqual(int(ids[-1]), 15)

    def test_window_profile_aggregates_image_pairs(self):
        target_window = torch.stack(
            (
                self.reference,
                self.reference + 0.1,
                self.reference + 0.2,
            )
        )
        profile = robust_window_sinkhorn_profile(
            target_window,
            self.reference.unsqueeze(0),
            self.patch_ids,
            grid_size=2,
            position_radius=0,
        )

        self.assertEqual(profile["target_image_count"], 3)
        self.assertEqual(profile["reference_image_count"], 1)
        self.assertEqual(profile["pair_count"], 3)
        self.assertGreater(profile["discrete_cost"], 0.0)
        self.assertAlmostEqual(profile["mutual_fraction"], 1.0, places=6)

    def test_pairwise_window_excludes_self_comparisons(self):
        window = torch.stack(
            (
                self.reference,
                self.reference + 0.1,
                self.reference + 0.2,
            )
        )
        profile = pairwise_window_sinkhorn_profile(
            window,
            self.patch_ids,
            grid_size=2,
            position_radius=0,
        )

        self.assertEqual(profile["pair_count"], 3)
        self.assertGreater(profile["discrete_cost_median"], 0.0)
        self.assertGreater(
            profile["discrete_cost_max"],
            profile["discrete_cost_median"],
        )
        self.assertAlmostEqual(
            profile["mutual_fraction_mean"],
            1.0,
            places=6,
        )

    def test_window_audit_uses_selected_history_and_all_experts(self):
        stream = torch.stack(
            (
                self.reference,
                self.reference + 0.1,
                self.reference + 0.2,
                self.reference + 0.3,
            )
        )
        profile = _robust_shift_window_audit_profile(
            current_step=3,
            selected_steps=[3, 1, 2],
            window_steps=[0, 1, 2, 3],
            active_experts=[{"expert_id": 3, "image_id": 0}],
            committee_features=stream,
            grid_size=2,
            max_patches=4,
            position_radius=0,
            epsilon=0.05,
            iterations=20,
        )

        self.assertEqual(profile["state"], "observed")
        self.assertEqual(profile["selected_steps"], [3, 1, 2])
        self.assertEqual(profile["dropped_steps"], [0])
        self.assertEqual(profile["nearest_expert_id"], 3)
        self.assertEqual(profile["within_window_pair_count"], 3)
        self.assertEqual(
            set(profile["within_window_pair_profiles"]),
            {"1:2", "1:3", "2:3"},
        )
        self.assertAlmostEqual(
            profile["within_window_mutual_fraction"],
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            profile["expert_profiles"]["3"]["mutual_fraction"],
            1.0,
            places=6,
        )
        self.assertIn("3", profile["expert_profiles"])

    def test_window_audit_keeps_fixed_initial_expert_reference(self):
        stream = torch.stack(
            (
                self.reference,
                self.reference + 0.1,
                self.reference + 0.2,
                self.reference + 0.3,
            )
        )
        profile = _robust_shift_window_audit_profile(
            current_step=3,
            selected_steps=[2, 3],
            window_steps=[2, 3],
            active_experts=[{"expert_id": 9, "image_id": 2}],
            committee_features=stream,
            grid_size=2,
            max_patches=4,
            position_radius=0,
            epsilon=0.05,
            iterations=20,
            fixed_reference_expert={"expert_id": 3, "image_id": 0},
        )

        self.assertEqual(profile["nearest_expert_id"], 9)
        self.assertEqual(profile["fixed_reference_expert_id"], 3)
        self.assertEqual(profile["fixed_reference_image_id"], 0)
        self.assertGreater(
            profile["fixed_reference_discrete_cost"],
            profile["nearest_discrete_cost"],
        )
        self.assertNotIn("3", profile["expert_profiles"])


if __name__ == "__main__":
    unittest.main()
