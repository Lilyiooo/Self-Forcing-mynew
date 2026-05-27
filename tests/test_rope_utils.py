import unittest

import torch

from utils.rope_utils import apply_temporal_rope_shift


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


if __name__ == "__main__":
    unittest.main()
