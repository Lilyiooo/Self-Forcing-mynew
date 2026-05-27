import importlib.util
import os
import unittest


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


if __name__ == "__main__":
    unittest.main()
