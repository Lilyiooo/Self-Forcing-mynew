import importlib.util
import os
import unittest

import torch


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


if __name__ == "__main__":
    unittest.main()
