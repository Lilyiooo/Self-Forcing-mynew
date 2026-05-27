import unittest

import torch


def build_context(full_k, full_v, mid_kv, sink_tokens):
    sink_len = min(sink_tokens, full_k.shape[1])
    sink_k = full_k[:, :sink_len]
    sink_v = full_v[:, :sink_len]
    recent_k = full_k[:, sink_len:]
    recent_v = full_v[:, sink_len:]

    if mid_kv is not None:
        attn_k = torch.cat([sink_k, mid_kv[0], recent_k], dim=1)
        attn_v = torch.cat([sink_v, mid_kv[1], recent_v], dim=1)
    else:
        attn_k = torch.cat([sink_k, recent_k], dim=1)
        attn_v = torch.cat([sink_v, recent_v], dim=1)
    return attn_k, attn_v


class AttentionContextOrderTest(unittest.TestCase):
    def test_context_order_is_sink_mid_recent(self):
        sink_k = torch.ones(1, 2, 1, 1)
        recent_k = torch.ones(1, 3, 1, 1) * 3
        mid_k = torch.ones(1, 4, 1, 1) * 2

        full_k = torch.cat([sink_k, recent_k], dim=1)
        full_v = full_k * 10
        mid_kv = torch.stack([mid_k, mid_k * 10], dim=0)

        attn_k, attn_v = build_context(full_k, full_v, mid_kv, sink_tokens=2)

        self.assertEqual(attn_k.flatten().tolist(), [1, 1, 2, 2, 2, 2, 3, 3, 3])
        self.assertEqual(attn_v.flatten().tolist(), [10, 10, 20, 20, 20, 20, 30, 30, 30])

    def test_context_order_without_mid_is_sink_recent(self):
        sink_k = torch.ones(1, 2, 1, 1)
        recent_k = torch.ones(1, 3, 1, 1) * 3
        full_k = torch.cat([sink_k, recent_k], dim=1)

        attn_k, _ = build_context(full_k, full_k, mid_kv=None, sink_tokens=2)

        self.assertEqual(attn_k.flatten().tolist(), [1, 1, 3, 3, 3])


if __name__ == "__main__":
    unittest.main()
