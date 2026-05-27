import unittest

import torch

from utils.cache_lifecycle import queue_aged_blocks


class CacheLifecycleTest(unittest.TestCase):
    def test_sink_and_recent_blocks_are_not_compressed_early(self):
        recent = []
        compressed_starts = []

        for chunk_idx, start in enumerate([0, 4, 8, 12]):
            block = torch.zeros(1, 4, 16, 2, 2)
            aged = queue_aged_blocks(
                recent,
                clean_block=block,
                chunk_idx=chunk_idx,
                current_start_frame=start,
                nsink=8,
                nrecent=4,
            )
            compressed_starts.extend(item["start"] for item in aged)

        self.assertEqual(compressed_starts, [8])
        self.assertEqual([item["start"] for item in recent], [12])

    def test_extra_metadata_is_preserved(self):
        recent = []
        block = torch.zeros(1, 4, 16, 2, 2)
        queue_aged_blocks(recent, block, 0, 8, nsink=8, nrecent=4, extra={"prev": "p0"})
        aged = queue_aged_blocks(recent, block, 1, 12, nsink=8, nrecent=4, extra={"prev": "p1"})

        self.assertEqual(len(aged), 1)
        self.assertEqual(aged[0]["prev"], "p0")


if __name__ == "__main__":
    unittest.main()
