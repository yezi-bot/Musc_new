import unittest

import torch

from MultiMuSc2.models.modules._DYNAMIC_PIPELINE import (
    _image_msm_score,
    _strict_history_fallback,
    build_dynamic_committee_timeline,
    score_dynamic_msm2_layer,
)
from MultiMuSc2.models.modules._MSM import aggregate_reference_distances


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
