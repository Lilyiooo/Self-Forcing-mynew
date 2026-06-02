import importlib.util
import os
import unittest

import torch


class ListLogger:
    def __init__(self):
        self.records = []

    def log(self, event, **fields):
        self.records.append({"event": event, **fields})


def load_kv_cache_class():
    path = os.path.join(os.path.dirname(__file__), "..", "model", "kv_cache.py")
    spec = importlib.util.spec_from_file_location("kv_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HeterogeneousKVCache


class HeterogeneousKVCacheTest(unittest.TestCase):
    def _make_cache(self, eviction_policy, nmid_tokens=5):
        HeterogeneousKVCache = load_kv_cache_class()
        return HeterogeneousKVCache(
            batch_size=1,
            num_transformer_blocks=1,
            num_heads=2,
            head_dim=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
            Nmid_tokens=nmid_tokens,
            eviction_policy=eviction_policy,
        )

    def _make_topk_cache(self, top_k_blocks=2):
        HeterogeneousKVCache = load_kv_cache_class()
        return HeterogeneousKVCache(
            batch_size=1,
            num_transformer_blocks=1,
            num_heads=2,
            head_dim=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
            Nmid_tokens=5,
            eviction_policy="fifo",
            top_k_enabled=True,
            top_k_blocks=top_k_blocks,
        )

    def test_fifo_eviction_removes_oldest_and_tracks_delta(self):
        cache = self._make_cache("fifo")

        first = torch.ones(2, 1, 3, 2, 4)
        second = torch.ones(2, 1, 3, 2, 4) * 2
        cache.push_mid_block(first, "mid", 0.8, temporal_position=8, n_frames=4)
        cache.push_mid_block(second, "low", 0.1, temporal_position=12, n_frames=4)

        self.assertEqual(len(cache.mid_meta), 1)
        self.assertEqual(cache.mid_meta[0].temporal_position, 12)
        self.assertEqual(cache.rope_delta_frames, 4)
        self.assertTrue(torch.equal(cache.get_mid_kv(0), second))

    def test_density_eviction_does_not_track_rope_delta(self):
        cache = self._make_cache("density")

        first = torch.ones(2, 1, 3, 2, 4)
        second = torch.ones(2, 1, 3, 2, 4) * 2
        cache.push_mid_block(first, "mid", 0.8, temporal_position=8, n_frames=4)
        cache.push_mid_block(second, "low", 0.1, temporal_position=12, n_frames=4)

        self.assertEqual(len(cache.mid_meta), 1)
        self.assertEqual(cache.mid_meta[0].temporal_position, 12)
        self.assertEqual(cache.rope_delta_frames, 0)
        self.assertTrue(torch.equal(cache.get_mid_kv(0), second))

    def test_reset_clears_rope_delta(self):
        cache = self._make_cache("fifo")
        first = torch.ones(2, 1, 3, 2, 4)
        second = torch.ones(2, 1, 3, 2, 4) * 2
        cache.push_mid_block(first, "mid", 0.8, temporal_position=8, n_frames=4)
        cache.push_mid_block(second, "low", 0.1, temporal_position=12, n_frames=4)
        self.assertEqual(cache.rope_delta_frames, 4)

        cache.reset_mid_buffer()

        self.assertEqual(cache.rope_delta_frames, 0)
        self.assertEqual(cache.mid_token_count, 0)
        self.assertEqual(cache.mid_meta, [])

    def test_debug_logger_records_mid_insert_and_eviction(self):
        cache = self._make_cache("fifo")
        logger = ListLogger()
        cache.set_debug_logger(logger)

        first = torch.ones(2, 1, 3, 2, 4)
        second = torch.ones(2, 1, 3, 2, 4) * 2
        cache.push_mid_block(first, "mid", 0.8, temporal_position=8, n_frames=4)
        cache.push_mid_block(second, "mid", 0.7, temporal_position=12, n_frames=4)

        events = [record["event"] for record in logger.records]
        self.assertEqual(events, ["mid_insert", "mid_evict", "mid_insert"])
        eviction = logger.records[1]
        self.assertEqual(eviction["evicted_temporal_position"], 8)
        self.assertEqual(eviction["updated_rope_delta_frames"], 4)

    def test_topk_archive_selects_recent_blocks_without_eviction(self):
        cache = self._make_topk_cache(top_k_blocks=2)

        first = torch.ones(2, 1, 3, 2, 4)
        second = torch.ones(2, 1, 3, 2, 4) * 2
        third = torch.ones(2, 1, 3, 2, 4) * 3
        cache.push_mid_block(first, "mid", 0.8, temporal_position=8, n_frames=4)
        cache.push_mid_block(second, "mid", 0.7, temporal_position=12, n_frames=4)
        cache.push_mid_block(third, "mid", 0.6, temporal_position=16, n_frames=4)

        self.assertEqual(len(cache.mid_meta), 3)
        self.assertEqual(cache.rope_delta_frames, 0)
        selected = cache.get_mid_kv(0)

        self.assertEqual(selected.shape[2], 6)
        self.assertTrue(torch.equal(selected[:, :, :3], second))
        self.assertTrue(torch.equal(selected[:, :, 3:], third))
        self.assertEqual(cache.mid_token_count, 6)


if __name__ == "__main__":
    unittest.main()
