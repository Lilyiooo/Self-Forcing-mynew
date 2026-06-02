import importlib.util
import os
import unittest

import torch


def load_module(relpath, name):
    path = os.path.join(os.path.dirname(__file__), "..", *relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UniformPackForcingTest(unittest.TestCase):
    def test_uniform_thresholds_route_all_scores_to_mid(self):
        density_mod = load_module(("model", "density_estimator.py"), "density_estimator")
        estimator = density_mod.DensityEstimator(
            high_threshold=1.1,
            low_threshold=-0.1,
        )

        for density_score in (0.0, 0.5, 1.0):
            if density_score >= estimator.high_threshold:
                tier = "high"
            elif density_score <= estimator.low_threshold:
                tier = "low"
            else:
                tier = "mid"
            self.assertEqual(tier, "mid")

    def test_mid_tokens_match_current_block_length(self):
        compress_mod = load_module(("model", "compress.py"), "compress")
        head = compress_mod.HRHead32x(in_ch=16, d_model=1536)

        with torch.no_grad():
            tokens_t2 = head(torch.zeros(1, 16, 2, 60, 104)).shape[1]
            tokens_t4 = head(torch.zeros(1, 16, 4, 60, 104)).shape[1]

        self.assertEqual(tokens_t2, 104)
        self.assertEqual(tokens_t4, 208)
        self.assertEqual(
            compress_mod.HeterogeneousCompressor.compressed_grid_shape(None, "mid", (4, 60, 104)),
            (2, 8, 13),
        )


if __name__ == "__main__":
    unittest.main()
