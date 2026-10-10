import unittest

import torch

from MultiMuSc2.models.modules._CHANNEL import ChannelMemory


class ChannelMemoryTest(unittest.TestCase):
    def test_association_profile_separates_provisional_and_mature_channels(self):
        features = torch.eye(4, dtype=torch.float32)
        reliability = torch.ones(4)
        memory = ChannelMemory(
            max_ttl=5,
            mature_span=3.0,
            density_k=1,
            device="cpu",
            position_radius=1,
        )

        memory.update(features, 0, 2, reliability)
        provisional = memory.association_profile(
            features, 2, current_image_id=1
        )
        memory.update(features, 1, 2, reliability)
        still_provisional = memory.association_profile(
            features, 2, current_image_id=2
        )
        memory.update(features, 2, 2, reliability)
        mature = memory.association_profile(features, 2)

        self.assertEqual(provisional["provisional_match_fraction"], 1.0)
        self.assertEqual(
            provisional["recent_provisional_match_fraction"], 1.0
        )
        self.assertEqual(still_provisional["provisional_match_fraction"], 1.0)
        self.assertEqual(
            still_provisional["recent_provisional_match_fraction"], 0.0
        )
        self.assertEqual(mature["mature_match_fraction"], 1.0)
        self.assertEqual(mature["provisional_match_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
