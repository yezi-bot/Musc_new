import unittest

import torch
import torch.nn.functional as F

from MultiMuSc2.models.modules._DYNAMIC_DINO_STREAM import (
    DynamicDinoOnlineState,
    chunked_dino_score,
    feature_key,
)
from MultiMuSc2.models.modules._MSM import (
    aggregate_reference_distances,
)


def make_features(step, patch_count=4, feature_dim=6):
    torch.manual_seed(100 + step)
    features = torch.randn(patch_count, feature_dim)
    return F.normalize(features, dim=-1)


def make_state():
    return DynamicDinoOnlineState(
        device="cpu",
        image_size=8,
        feature_layers=[23],
        r_list=[1],
        committee_config={
            "r": 1,
            "dino_layer": 23,
            "position_radius": 1,
            "channel_distance_quantile": 0.7,
            "reliability_alpha": 0.5,
            "channel_ttl": 5,
            "mature_span": 3.0,
            "density_k": 2,
            "ms_quantile": 0.3,
            "support_quantile": 0.7,
            "duplicate_similarity": 0.98,
            "min_cluster_support": 2,
            "committee_cap": 5,
            "base_ttl": 5,
            "max_ttl": 20,
            "ttl_gap_multiplier": 2.0,
        },
        scoring_config={
            "topmin_min": 0.02,
            "topmin_max": 0.3,
            "reference_chunk_size": 2,
        },
    )


class DynamicDinoStreamTest(unittest.TestCase):
    def test_chunked_score_matches_direct_score(self):
        query = make_features(0, patch_count=9)
        references = [
            make_features(step, patch_count=9)
            for step in range(1, 6)
        ]

        expected = aggregate_reference_distances(
            query,
            torch.stack(references),
            topmin_min=0.02,
            topmin_max=0.3,
        )
        actual = chunked_dino_score(
            query=query,
            references=references,
            device="cpu",
            topmin_min=0.02,
            topmin_max=0.3,
            chunk_size=2,
        )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_first_image_is_unavailable(self):
        state = make_state()

        anomaly_map, record = state.process_features(
            {
                feature_key(1, 23): make_features(0),
            }
        )

        self.assertIsNone(anomaly_map)
        self.assertFalse(record["available"])
        self.assertEqual(
            record["fallback_reason"],
            "first_image_unavailable",
        )
        self.assertEqual(
            record["scoring_reference_image_ids"],
            [],
        )

    def test_all_references_are_strictly_historical(self):
        state = make_state()

        for step in range(5):
            state.process_features(
                {
                    feature_key(1, 23):
                        make_features(step),
                }
            )

        for record in state.timeline:
            step = record["step"]

            self.assertTrue(
                all(
                    image_id < step
                    for image_id in
                    record["scoring_reference_image_ids"]
                )
            )
            self.assertTrue(
                all(
                    expert["image_id"] < step
                    for expert in
                    record["active_experts_before"]
                )
            )

    def test_score_precedes_manager_and_memory_update(self):
        state = make_state()
        events = []

        original_score = state._score_current
        original_advance = state.manager.advance
        original_update = state.memory.update

        def tracked_score(*args, **kwargs):
            events.append("score")
            return original_score(*args, **kwargs)

        def tracked_advance(*args, **kwargs):
            events.append("advance")
            return original_advance(*args, **kwargs)

        def tracked_update(*args, **kwargs):
            events.append("memory_update")
            return original_update(*args, **kwargs)

        state._score_current = tracked_score
        state.manager.advance = tracked_advance
        state.memory.update = tracked_update

        state.process_features(
            {
                feature_key(1, 23): make_features(0),
            }
        )

        self.assertEqual(
            events,
            [
                "score",
                "advance",
                "memory_update",
            ],
        )


if __name__ == "__main__":
    unittest.main()