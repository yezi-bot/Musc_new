import unittest

import numpy as np
import torch

from MultiMuSc2.models.modules._MSM import (
    MSM2_online,
    aggregate_reference_distances,
    build_causal_fuser_training_data,
    compute_scores_fast,
    fit_causal_detect_fuser,
    safe_topmin_counts,
)


class ConstantFuser:
    def __init__(self, value):
        self.value = value
        self.last_samples = None

    def score_samples(self, samples):
        self.last_samples = samples
        return np.full(samples.shape[0], self.value, dtype=np.float32)

    def fit(self, samples):
        self.last_samples = samples
        return self


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

    def test_patch_reliability_changes_selection_not_distance_scale(self):
        current = torch.tensor([[0.0, 0.0]])
        references = torch.tensor([[[0.0, 0.0], [0.4, 0.0]]])
        baseline = aggregate_reference_distances(current, references)
        weighted, audit = aggregate_reference_distances(
            current,
            references,
            reference_patch_reliabilities=torch.tensor([[0.1, 1.0]]),
            reliability_penalty=1.0,
            return_reliability_audit=True,
        )

        self.assertGreater(float(weighted), float(baseline))
        self.assertAlmostEqual(float(weighted), 0.4)
        self.assertAlmostEqual(audit["reference_reliability_mean"], 1.0)
        self.assertEqual(audit["raw_reference_weight"], 1)
        self.assertAlmostEqual(audit["selection_changed_fraction"], 1.0)
        self.assertGreater(float(audit["score_shift"]), 0.0)

    def test_all_one_patch_reliability_preserves_baseline(self):
        current = torch.randn(3, 4)
        references = torch.randn(2, 3, 4)
        baseline = aggregate_reference_distances(current, references)
        weighted = aggregate_reference_distances(
            current,
            references,
            reference_patch_reliabilities=torch.ones(2, 3),
            reliability_penalty=2.0,
        )

        self.assertTrue(torch.allclose(weighted, baseline))

    def test_uniform_low_reliability_preserves_baseline(self):
        current = torch.randn(3, 4)
        references = torch.randn(2, 3, 4)
        baseline = aggregate_reference_distances(current, references)
        weighted = aggregate_reference_distances(
            current,
            references,
            reference_patch_reliabilities=torch.full((2, 3), 0.2),
            reliability_penalty=2.0,
        )

        self.assertTrue(torch.allclose(weighted, baseline))

    def test_reliability_audit_groups_actual_selection_by_reference(self):
        current = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
        references = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 1.0]],
                [[10.0, 0.0], [10.0, 1.0]],
            ]
        )
        reliabilities = torch.tensor(
            [[0.5, 0.9], [0.8, 0.9]]
        )

        _, audit = aggregate_reference_distances(
            current,
            references,
            reference_patch_reliabilities=reliabilities,
            reliability_penalty=0.0,
            return_reliability_audit=True,
        )

        self.assertEqual(audit["selected_reference_raw_counts"], [1, 1])
        np.testing.assert_allclose(
            audit["selected_reference_effective_weights"],
            [0.5, 0.8],
        )

    def test_clip_only_returns_clip_distance(self):
        current = torch.zeros(2, 2)
        expert_dino = torch.full((1, 2, 2), 2.0)
        expert_clip = torch.full((1, 2, 2), 4.0)
        patch_score, audit = MSM2_online(
            current,
            current,
            expert_dino,
            expert_clip,
            fusion_mode="clip_only",
        )

        self.assertTrue(torch.equal(patch_score, audit["clip_distance"]))
        self.assertFalse(torch.equal(patch_score, audit["dino_distance"]))

    def test_causal_training_is_leave_one_image_out(self):
        dino = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [4.0, 0.0]],
            ]
        )
        clip = dino * 2.0
        training = build_causal_fuser_training_data(
            dino,
            clip,
            image_ids=[10, 11],
            member_steps=[2, 4],
            current_step=5,
        )

        self.assertTrue(training["available"])
        self.assertEqual(training["train_pairs"].shape, (4, 2))
        self.assertTrue((training["train_pairs"][:, 0] > 0).all())
        self.assertTrue((training["train_pairs"][:, 1] > 0).all())
        self.assertGreater(training["dino_scale"], 0.0)
        self.assertGreater(training["clip_scale"], 0.0)

    def test_causal_training_rejects_current_or_future_members(self):
        features = torch.zeros(2, 2, 2)
        with self.assertRaisesRegex(ValueError, "strictly historical"):
            build_causal_fuser_training_data(
                features,
                features,
                image_ids=[10, 11],
                member_steps=[2, 5],
                current_step=5,
            )

    def test_causal_training_preserves_uniform_reference_reliability(self):
        dino = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [4.0, 0.0]],
            ]
        )
        baseline = build_causal_fuser_training_data(
            dino,
            dino,
            image_ids=[10, 11],
            member_steps=[1, 2],
            current_step=3,
        )
        weighted = build_causal_fuser_training_data(
            dino,
            dino,
            image_ids=[10, 11],
            member_steps=[1, 2],
            current_step=3,
            training_dino_patch_reliabilities=torch.tensor(
                [[1.0, 1.0], [0.25, 0.25]]
            ),
        )

        self.assertTrue(weighted["available"])
        self.assertTrue(
            torch.allclose(
                baseline["train_pairs"],
                weighted["train_pairs"],
            )
        )

    def test_fuser_fit_reports_insufficient_and_degenerate_history(self):
        fuser = ConstantFuser(1.0)
        one_image = torch.ones(1, 2, 2)
        insufficient = fit_causal_detect_fuser(
            fuser,
            one_image,
            one_image,
            image_ids=[10],
            member_steps=[1],
            current_step=2,
        )
        identical = torch.ones(2, 2, 2)
        degenerate = fit_causal_detect_fuser(
            fuser,
            identical,
            identical,
            image_ids=[10, 11],
            member_steps=[1, 2],
            current_step=3,
        )

        self.assertEqual(insufficient["reason"], "insufficient_history")
        self.assertFalse(insufficient["fitted"])
        self.assertEqual(degenerate["reason"], "degenerate_training_data")
        self.assertFalse(degenerate["fitted"])

    def test_fuser_fit_uses_aligned_causal_pairs(self):
        fuser = ConstantFuser(1.0)
        dino = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [4.0, 0.0]],
            ]
        )
        result = fit_causal_detect_fuser(
            fuser,
            dino,
            dino * 2.0,
            image_ids=[10, 11],
            member_steps=[1, 2],
            current_step=3,
        )

        self.assertTrue(result["fitted"])
        np.testing.assert_allclose(fuser.last_samples, result["train_pairs"].numpy())


if __name__ == "__main__":
    unittest.main()
