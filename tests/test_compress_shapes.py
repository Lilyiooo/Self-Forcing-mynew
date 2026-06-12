import importlib.util
import os
import unittest

import torch


def load_compressor_class():
    path = os.path.join(os.path.dirname(__file__), "..", "model", "compress.py")
    spec = importlib.util.spec_from_file_location("compress", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HeterogeneousCompressor


class CompressedGridShapeTest(unittest.TestCase):
    def test_grid_shapes_match_hr_head_strides(self):
        HeterogeneousCompressor = load_compressor_class()
        fn = HeterogeneousCompressor.compressed_grid_shape

        self.assertEqual(fn(None, "high", (4, 60, 104)), (2, 30, 52))
        self.assertEqual(fn(None, "mid", (4, 60, 104)), (2, 8, 13))
        self.assertEqual(fn(None, "low", (4, 60, 104)), (2, 4, 7))

    def test_temporal_padding_uses_ceil_division(self):
        HeterogeneousCompressor = load_compressor_class()
        fn = HeterogeneousCompressor.compressed_grid_shape

        self.assertEqual(fn(None, "high", (3, 60, 104))[0], 2)
        self.assertEqual(fn(None, "high", (1, 60, 104))[0], 1)

    def test_project_to_kv_is_layer_specific(self):
        HeterogeneousCompressor = load_compressor_class()
        compressor = HeterogeneousCompressor(
            vae=None,
            d_model=12,
            in_ch=16,
            num_layers=3,
        )
        with torch.no_grad():
            compressor.kv_k_proj[1].weight.copy_(compressor.kv_k_proj[0].weight + 0.1)
            compressor.kv_v_proj[1].weight.copy_(compressor.kv_v_proj[0].weight + 0.1)

        tokens = torch.randn(1, 5, 12)
        kv = compressor.project_to_kv(tokens, num_layers=3, num_heads=3)

        self.assertEqual(len(kv), 3)
        self.assertEqual(kv[0].shape, (2, 1, 5, 3, 4))
        self.assertFalse(torch.allclose(kv[0], kv[1]))

    def test_legacy_shared_kv_checkpoint_expands_to_all_layers(self):
        HeterogeneousCompressor = load_compressor_class()
        compressor = HeterogeneousCompressor(
            vae=None,
            d_model=12,
            in_ch=16,
            num_layers=3,
        )
        state = compressor.state_dict()
        legacy = {
            k: v.clone()
            for k, v in state.items()
            if not k.startswith("kv_k_proj.") and not k.startswith("kv_v_proj.")
        }
        legacy["kv_k_proj.weight"] = torch.randn_like(compressor.kv_k_proj[0].weight)
        legacy["kv_k_proj.bias"] = torch.randn_like(compressor.kv_k_proj[0].bias)
        legacy["kv_v_proj.weight"] = torch.randn_like(compressor.kv_v_proj[0].weight)
        legacy["kv_v_proj.bias"] = torch.randn_like(compressor.kv_v_proj[0].bias)

        compressor.load_state_dict(legacy, strict=True)

        for layer_idx in range(3):
            self.assertTrue(torch.equal(compressor.kv_k_proj[layer_idx].weight, legacy["kv_k_proj.weight"]))
            self.assertTrue(torch.equal(compressor.kv_v_proj[layer_idx].bias, legacy["kv_v_proj.bias"]))


if __name__ == "__main__":
    unittest.main()
