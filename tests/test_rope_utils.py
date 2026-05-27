import unittest

import torch

from utils.rope_utils import apply_temporal_rope_shift, apply_temporal_rope_to_unrotated


def make_freqs(max_pos=64, dim=64, theta=10000):
    idx = torch.arange(0, dim, 2).float()
    inv_freq = 1.0 / (theta ** (idx / dim))
    positions = torch.arange(max_pos).float()
    freqs = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


class TemporalRopeShiftTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.k = torch.randn(2, 17, 3, 128)
        self.freqs = make_freqs(dim=128)

    def test_delta_zero_is_identity(self):
        shifted = apply_temporal_rope_shift(self.k, self.freqs, delta=0)
        self.assertTrue(torch.equal(shifted, self.k))

    def test_shift_preserves_norm(self):
        shifted = apply_temporal_rope_shift(self.k, self.freqs, delta=5)
        self.assertTrue(torch.allclose(shifted.norm(dim=-1), self.k.norm(dim=-1), atol=1e-5))

    def test_shift_composes(self):
        shifted_twice = apply_temporal_rope_shift(
            apply_temporal_rope_shift(self.k, self.freqs, delta=3),
            self.freqs,
            delta=7,
        )
        shifted_once = apply_temporal_rope_shift(self.k, self.freqs, delta=10)
        self.assertTrue(torch.allclose(shifted_twice, shifted_once, atol=1e-5))


class InitialTemporalRopeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.k = torch.randn(1, 12, 2, 128)
        self.freqs = make_freqs(max_pos=64, dim=128)

    def test_initial_rope_shape_and_norm(self):
        out = apply_temporal_rope_to_unrotated(
            self.k,
            freqs=self.freqs,
            start_frame=4,
            grid_shape=(2, 2, 3),
            temporal_stride=2,
        )
        self.assertEqual(out.shape, self.k.shape)
        self.assertTrue(torch.allclose(out.norm(dim=-1), self.k.norm(dim=-1), atol=1e-5))

    def test_initial_rope_rejects_grid_token_mismatch(self):
        with self.assertRaises(ValueError):
            apply_temporal_rope_to_unrotated(
                self.k,
                freqs=self.freqs,
                start_frame=4,
                grid_shape=(2, 2, 2),
                temporal_stride=2,
            )

    def test_initial_rope_depends_on_start_frame(self):
        out_a = apply_temporal_rope_to_unrotated(
            self.k,
            freqs=self.freqs,
            start_frame=0,
            grid_shape=(2, 2, 3),
            temporal_stride=2,
        )
        out_b = apply_temporal_rope_to_unrotated(
            self.k,
            freqs=self.freqs,
            start_frame=5,
            grid_shape=(2, 2, 3),
            temporal_stride=2,
        )
        self.assertFalse(torch.allclose(out_a, out_b))


if __name__ == "__main__":
    unittest.main()
