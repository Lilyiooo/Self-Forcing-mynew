# model/kv_cache.py
"""
Heterogeneous KV Cache with three-zone structure:
  - sink:   first few frames, always retained
  - mid:    compressed historical blocks, managed by token budget
  - recent: sliding window of most recent frames

The mid buffer uses token-budget-based management instead of fixed-size blocks.
Eviction is density-aware: lowest density blocks are evicted first.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class MidBlockMeta:
    """Metadata for one compressed block in the mid buffer."""
    density_level: str       # "high", "mid", "low"
    density_score: float     # raw density score
    n_tokens: int            # number of compressed tokens in this block
    n_frames: int            # number of latent frames represented by this block
    kv_slice: slice          # position in mid_kv_buffer tensor
    temporal_position: int   # absolute frame index when generated (for RoPE adjustment)


class HeterogeneousKVCache:
    """
    Three-zone KV cache with token-budget-based mid buffer management.
    Designed to be compatible with the existing CausalWanSelfAttention KV cache interface.
    """

    def __init__(
        self,
        batch_size: int,
        num_transformer_blocks: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        Nsink: int = 8,            # sink size in latent frames
        Nrecent: int = 4,          # recent window size in latent frames
        Nmid_tokens: int = 5000,   # total token budget for mid buffer
        frame_seq_length: int = 1560,  # tokens per latent frame
        local_attn_size: int = -1,     # local attention window size (-1 = global)
        eviction_policy: str = "density",
    ):
        self.batch_size = batch_size
        self.num_transformer_blocks = num_transformer_blocks
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.Nsink = Nsink
        self.Nrecent = Nrecent
        self.Nmid_tokens = Nmid_tokens
        self.frame_seq_length = frame_seq_length
        self.local_attn_size = local_attn_size
        self.eviction_policy = eviction_policy
        self.rope_delta_frames = 0

        # Total KV cache size for recent + sink zones
        if local_attn_size != -1:
            self.kv_cache_size = local_attn_size * frame_seq_length
        else:
            self.kv_cache_size = 32760

        # Main KV cache (same structure as original, for sink + recent)
        # Per-layer list of dicts with "k", "v", "global_end_index", "local_end_index"
        self.layer_caches: list[dict] = []
        for _ in range(num_transformer_blocks):
            self.layer_caches.append({
                "k": torch.zeros(
                    [batch_size, self.kv_cache_size, num_heads, head_dim],
                    dtype=dtype, device=device),
                "v": torch.zeros(
                    [batch_size, self.kv_cache_size, num_heads, head_dim],
                    dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })

        # Mid buffer metadata (shared across all layers, one list of meta per compressed block)
        self.mid_meta: list[MidBlockMeta] = []

        # Mid buffer tensors (per-layer)
        self.mid_kv_buffers: Optional[list[torch.Tensor]] = None  # lazy init
        self.mid_token_count: int = 0

        # Cross-attention cache (same as original)
        self.crossattn_cache: list[dict] = []
        for _ in range(num_transformer_blocks):
            self.crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })

    # ------------------------------------------------------------------
    # Mid buffer operations
    # ------------------------------------------------------------------

    def _ensure_mid_buffers(self, kv_compressed: torch.Tensor):
        """Lazily initialize mid KV buffers based on the first compressed block."""
        if self.mid_kv_buffers is not None:
            return
        B = kv_compressed.shape[1]
        n_heads = kv_compressed.shape[3]
        head_dim = kv_compressed.shape[4]
        self.mid_kv_buffers = []
        for _ in range(self.num_transformer_blocks):
            self.mid_kv_buffers.append(
                torch.zeros(
                    2, B, self.Nmid_tokens, n_heads, head_dim,
                    dtype=kv_compressed.dtype, device=kv_compressed.device,
                )
            )

    def push_mid_block(
        self,
        kv_compressed: torch.Tensor,   # (2, B, N_new, n_heads, head_dim) per-layer or shared
        density_level: str,
        density_score: float,
        temporal_position: int = 0,
        n_frames: int = 0,
    ) -> None:
        """
        Push a new compressed block into the mid buffer.
        If exceeding token budget, evict lowest-density blocks first, then insert.
        """
        # kv_compressed can be per-layer or a single shared tensor
        # We handle the per-layer case where kv_compressed is a list of per-layer tensors
        if isinstance(kv_compressed, list):
            # Per-layer compressed KV
            N_new = kv_compressed[0].shape[2]
            self._ensure_mid_buffers(kv_compressed[0])

            # Eviction: evict lowest density blocks until enough space
            while self.mid_token_count + N_new > self.Nmid_tokens and self.mid_meta:
                self._evict_one_mid_block()

            # Truncate if single block exceeds budget
            if N_new > self.Nmid_tokens:
                kv_compressed = [kvc[:, :, :self.Nmid_tokens, :, :] for kvc in kv_compressed]
                N_new = self.Nmid_tokens

            # Write into per-layer buffers
            start = self.mid_token_count
            end = start + N_new
            for layer_idx in range(self.num_transformer_blocks):
                self.mid_kv_buffers[layer_idx][:, :, start:end, :, :] = kv_compressed[layer_idx]

            # Update metadata
            self.mid_token_count = end
            self.mid_meta.append(MidBlockMeta(
                density_level=density_level,
                density_score=density_score,
                n_tokens=N_new,
                n_frames=n_frames,
                kv_slice=slice(start, end),
                temporal_position=temporal_position,
            ))
        else:
            # Single shared tensor (broadcast to all layers)
            N_new = kv_compressed.shape[2]
            self._ensure_mid_buffers(kv_compressed)

            while self.mid_token_count + N_new > self.Nmid_tokens and self.mid_meta:
                self._evict_one_mid_block()

            if N_new > self.Nmid_tokens:
                kv_compressed = kv_compressed[:, :, :self.Nmid_tokens, :, :]
                N_new = self.Nmid_tokens

            start = self.mid_token_count
            end = start + N_new
            for layer_idx in range(self.num_transformer_blocks):
                self.mid_kv_buffers[layer_idx][:, :, start:end, :, :] = kv_compressed

            self.mid_token_count = end
            self.mid_meta.append(MidBlockMeta(
                density_level=density_level,
                density_score=density_score,
                n_tokens=N_new,
                n_frames=n_frames,
                kv_slice=slice(start, end),
                temporal_position=temporal_position,
            ))

        assert self.mid_token_count <= self.Nmid_tokens, \
            f"mid buffer overflow: {self.mid_token_count} > {self.Nmid_tokens}"

    def _evict_one_mid_block(self) -> None:
        if self.eviction_policy == "fifo":
            self._evict_mid_block(0)
        elif self.eviction_policy == "density":
            min_idx = min(range(len(self.mid_meta)), key=lambda i: self.mid_meta[i].density_score)
            self._evict_mid_block(min_idx)
        else:
            raise ValueError(f"unknown eviction_policy: {self.eviction_policy}")

    def _evict_lowest_density_block(self) -> None:
        """Evict the block with the lowest density_score (in-place compaction)."""
        if not self.mid_meta:
            return
        min_idx = min(range(len(self.mid_meta)), key=lambda i: self.mid_meta[i].density_score)
        self._evict_mid_block(min_idx)

    def _evict_mid_block(self, evict_idx: int) -> None:
        """Evict one mid block by index (in-place compaction)."""
        if not self.mid_meta:
            return

        evicted = self.mid_meta.pop(evict_idx)
        ev_slice = evicted.kv_slice
        ev_n = evicted.n_tokens
        if self.eviction_policy == "fifo":
            self.rope_delta_frames += evicted.n_frames

        # In-place move: shift tokens after ev_slice forward
        end = self.mid_token_count
        ev_start = ev_slice.start
        ev_end = ev_slice.stop

        if ev_end < end:
            for layer_idx in range(self.num_transformer_blocks):
                self.mid_kv_buffers[layer_idx][:, :, ev_start:end - ev_n, :, :] = \
                    self.mid_kv_buffers[layer_idx][:, :, ev_end:end, :, :].clone()

        self.mid_token_count -= ev_n

        # Update kv_slice offsets for remaining blocks
        for meta in self.mid_meta:
            s = meta.kv_slice
            if s.start >= ev_end:
                meta.kv_slice = slice(s.start - ev_n, s.stop - ev_n)

    def get_mid_kv(self, layer_idx: int = 0) -> Optional[torch.Tensor]:
        """
        Return the current mid buffer KV for a specific layer.
        shape: (2, B, mid_token_count, n_heads, head_dim) or None if empty
        """
        if self.mid_kv_buffers is None or self.mid_token_count == 0:
            return None
        return self.mid_kv_buffers[layer_idx][:, :, :self.mid_token_count, :, :]

    def reset_mid_buffer(self) -> None:
        """Reset mid buffer at the start of each video generation."""
        self.mid_meta = []
        self.mid_token_count = 0
        self.rope_delta_frames = 0
        # Don't zero mid_kv_buffers tensors; next writes will overwrite

    # ------------------------------------------------------------------
    # Compatibility with existing layer_cache interface
    # ------------------------------------------------------------------

    def get_layer_cache(self, layer_idx: int) -> dict:
        """Get the layer cache dict for the given transformer block (sink + recent)."""
        return self.layer_caches[layer_idx]

    def get_crossattn_cache(self, layer_idx: int) -> dict:
        """Get the cross-attention cache for the given transformer block."""
        return self.crossattn_cache[layer_idx]

    def reset(self) -> None:
        """Full reset for a new generation."""
        self.reset_mid_buffer()
        for block_index in range(self.num_transformer_blocks):
            self.crossattn_cache[block_index]["is_init"] = False
            self.layer_caches[block_index]["global_end_index"].fill_(0)
            self.layer_caches[block_index]["local_end_index"].fill_(0)
