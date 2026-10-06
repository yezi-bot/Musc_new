import unittest

import numpy as np
import torch

from MultiMuSc2.models.modules._MSM import (
    MSM2_online,
    aggregate_reference_distances,
    compute_scores_fast,
    safe_topmin_counts,
)


class ConstantFuser:
    def __init__(self, value):
        self.value = value
        self.last_samples = None

    def score_samples(self, samples):
        self.last_samples = samples
        return np.full(samples.shape[0], self.value, dtype=np.float32)


class OnlineMSMTest(unittest.TestCase):
    def test_nearest_thirty_percent_is_safe_for_small_committees(self):
        query = torch.zeros(3, 2)
        for expert_count in range(1, 6):
            references = torch.arange(
                1,
                expert_count + 1,
                dtype=torch.float32,
            ).reshape(expert_count, 1, 1).repeat(1, 3, 2)
            scores = aggregate_reference_distances(query, references)
            legacy_scores = compute_scores_fast(
                torch.cat([query.unsqueeze(0), references], dim=0),
                0,
                "cpu",
            )

            self.assertEqual(scores.shape, (3,))
            self.assertTrue(torch.isfinite(scores).all())
            self.assertTrue(torch.equal(scores, legacy_scores))
            k_min, k_max, keep = safe_topmin_counts(expert_count)
            self.assertGreaterEqual(k_max, 1)
            self.assertGreaterEqual(keep, 1)
            self.assertLessEqual(k_min, k_max - 1)

    def test_fixed_fusion_uses_historical_scales(self):
        current_dino = torch.zeros(2, 2)
        current_clip = torch.zeros(2, 2)
        expert_dino = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]])
        expert_clip = torch.tensor([[[0.0, 4.0], [0.0, 4.0]]])

        patch_score, audit = MSM2_online(
            current_dino,
            current_clip,
            expert_dino,
            expert_clip,
            fusion_mode="fixed",
            dino_scale=2.0,
            clip_scale=4.0,
        )

        self.assertTrue(torch.allclose(patch_score, torch.ones(2)))
        self.assertEqual(audit["reference_count"], 1)
        self.assertIsNone(audit["fuser_score"])

    def test_fuser_receives_weighted_pairs_and_preserves_original_formula(self):
        fuser = ConstantFuser(2.0)
        current = torch.zeros(2, 2)
        expert_dino = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]])
        expert_clip = torch.tensor([[[0.0, 4.0], [0.0, 4.0]]])

        patch_score, audit = MSM2_online(
            current,
            current,
            expert_dino,
            expert_clip,
            detect_fuser=fuser,
            fusion_mode="fuser",
        )

        np.testing.assert_allclose(fuser.last_samples, [[2.0, 2.0], [2.0, 2.0]])
        self.assertTrue(torch.allclose(patch_score, torch.full((2,), 16.0)))
        self.assertEqual(audit["fuser_positive_ratio"], 1.0)
        self.assertEqual(audit["fuser_negative_ratio"], 0.0)

    def test_dino_and_clip_expert_ids_must_be_aligned(self):
        current = torch.zeros(2, 2)
        with self.assertRaisesRegex(ValueError, "expert counts must match"):
            MSM2_online(
                current,
                current,
                torch.zeros(1, 2, 2),
                torch.zeros(2, 2, 2),
                fusion_mode="fixed",
                dino_scale=1.0,
                clip_scale=1.0,
            )

    def test_dino_only_does_not_require_clip_features(self):
        current = torch.zeros(2, 2)
        patch_score, audit = MSM2_online(
            current,
            None,
            torch.ones(1, 2, 2),
            None,
            fusion_mode="dino_only",
        )

        self.assertEqual(patch_score.shape, (2,))
        self.assertIsNone(audit["clip_distance"])


if __name__ == "__main__":
    unittest.main()
