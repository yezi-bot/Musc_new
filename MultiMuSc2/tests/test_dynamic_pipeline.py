import unittest

import torch

from MultiMuSc2.models.modules._DYNAMIC_PIPELINE import (
    _image_msm_score,
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

    def test_all_modes_share_timeline_and_return_finite_scores(self):
        dino = synthetic_features(4)
        clip = dino * 1.7
        timeline = scoring_timeline()

        for mode in ("fuser", "fixed", "dino_only"):
            scores, audits = score_dynamic_msm2_layer(
                dino,
                None if mode == "dino_only" else clip,
                timeline,
                fusion_mode=mode,
            )
            self.assertEqual(scores.shape, (4, 4))
            self.assertTrue(torch.isfinite(scores).all())
            self.assertEqual(audits[0]["fallback_reason"], "first_image_unavailable")

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


if __name__ == "__main__":
    unittest.main()
