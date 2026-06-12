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
    block_id: int = 0         # monotonic compressed block id


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
        top_k_enabled: bool = False,
        top_k_blocks: int = 8,
        mid_archive_capacity_blocks: int = 64,
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
        self.debug_logger = None
        self.top_k_enabled = top_k_enabled
        self.top_k_blocks = top_k_blocks
        self.mid_archive_capacity_blocks = mid_archive_capacity_blocks
        self._next_block_id = 0
        self._last_logged_selection = None

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
        self.mid_archive_kv: list[list[torch.Tensor]] = []

        # Cross-attention cache (same as original)
        self.crossattn_cache: list[dict] = []
        for _ in range(num_transformer_blocks):
            self.crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })

    def set_debug_logger(self, logger) -> None:
        self.debug_logger = logger
        for layer_idx, cache in enumerate(self.layer_caches):
            cache["_debug_logger"] = logger
            cache["_debug_layer_idx"] = layer_idx
            cache["_debug_frame_seq_length"] = self.frame_seq_length

    def _record_debug_event(self, event: str, **fields) -> None:
        if self.debug_logger is not None:
            self.debug_logger.log(event, **fields)

    def _archive_total_tokens(self) -> int:
        return sum(meta.n_tokens for meta in self.mid_meta)

    def _new_meta(
        self,
        density_level: str,
        density_score: float,
        n_tokens: int,
        n_frames: int,
        kv_slice: slice,
        temporal_position: int,
    ) -> MidBlockMeta:
        meta = MidBlockMeta(
            block_id=self._next_block_id,
            density_level=density_level,
            density_score=density_score,
            n_tokens=n_tokens,
            n_frames=n_frames,
            kv_slice=kv_slice,
            temporal_position=temporal_position,
        )
        self._next_block_id += 1
        return meta

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
            if self.top_k_enabled:
                meta = self._new_meta(
                    density_level=density_level,
                    density_score=density_score,
                    n_tokens=N_new,
                    n_frames=n_frames,
                    kv_slice=slice(0, N_new),
                    temporal_position=temporal_position,
                )
                self.mid_meta.append(meta)
                self.mid_archive_kv.append([kvc.detach().clone() for kvc in kv_compressed])
                self._evict_archive_blocks_if_needed()
                self.mid_token_count = self._selected_mid_token_count()
                self._record_debug_event(
                    "mid_archive_insert",
                    block_id=meta.block_id,
                    density_level=density_level,
                    density_score=float(density_score),
                    n_tokens=N_new,
                    n_frames=n_frames,
                    temporal_position=temporal_position,
                    block_start=temporal_position,
                    block_end=temporal_position + n_frames,
                    archive_blocks=len(self.mid_meta),
                    archive_total_tokens=self._archive_total_tokens(),
                    active_mid_blocks=len(self._selected_archive_indices()),
                    active_mid_token_count=self.mid_token_count,
                    rope_delta_frames=self.rope_delta_frames,
                )
                return
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
            self.mid_meta.append(self._new_meta(
                density_level=density_level,
                density_score=density_score,
                n_tokens=N_new,
                n_frames=n_frames,
                kv_slice=slice(start, end),
                temporal_position=temporal_position,
            ))
            self._record_debug_event(
                "mid_insert",
                density_level=density_level,
                density_score=float(density_score),
                n_tokens=N_new,
                n_frames=n_frames,
                temporal_position=temporal_position,
                block_start=temporal_position,
                block_end=temporal_position + n_frames,
                mid_token_count=self.mid_token_count,
                len_mid_meta=len(self.mid_meta),
                rope_delta_frames=self.rope_delta_frames,
            )
        else:
            # Single shared tensor (broadcast to all layers)
            N_new = kv_compressed.shape[2]
            if self.top_k_enabled:
                meta = self._new_meta(
                    density_level=density_level,
                    density_score=density_score,
                    n_tokens=N_new,
                    n_frames=n_frames,
                    kv_slice=slice(0, N_new),
                    temporal_position=temporal_position,
                )
                self.mid_meta.append(meta)
                self.mid_archive_kv.append(
                    [kv_compressed.detach().clone() for _ in range(self.num_transformer_blocks)]
                )
                self._evict_archive_blocks_if_needed()
                self.mid_token_count = self._selected_mid_token_count()
                self._record_debug_event(
                    "mid_archive_insert",
                    block_id=meta.block_id,
                    density_level=density_level,
                    density_score=float(density_score),
                    n_tokens=N_new,
                    n_frames=n_frames,
                    temporal_position=temporal_position,
                    block_start=temporal_position,
                    block_end=temporal_position + n_frames,
                    archive_blocks=len(self.mid_meta),
                    archive_total_tokens=self._archive_total_tokens(),
                    active_mid_blocks=len(self._selected_archive_indices()),
                    active_mid_token_count=self.mid_token_count,
                    rope_delta_frames=self.rope_delta_frames,
                )
                return
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
            self.mid_meta.append(self._new_meta(
                density_level=density_level,
                density_score=density_score,
                n_tokens=N_new,
                n_frames=n_frames,
                kv_slice=slice(start, end),
                temporal_position=temporal_position,
            ))
            self._record_debug_event(
                "mid_insert",
                density_level=density_level,
                density_score=float(density_score),
                n_tokens=N_new,
                n_frames=n_frames,
                temporal_position=temporal_position,
                block_start=temporal_position,
                block_end=temporal_position + n_frames,
                mid_token_count=self.mid_token_count,
                len_mid_meta=len(self.mid_meta),
                rope_delta_frames=self.rope_delta_frames,
            )

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
        updated_rope_delta = self.rope_delta_frames

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

        self._record_debug_event(
            "mid_evict",
            eviction_policy=self.eviction_policy,
            evicted_index=evict_idx,
            evicted_temporal_position=evicted.temporal_position,
            evicted_n_frames=evicted.n_frames,
            evicted_n_tokens=ev_n,
            updated_rope_delta_frames=updated_rope_delta,
            mid_token_count=self.mid_token_count,
            len_mid_meta=len(self.mid_meta),
        )

    def _evict_archive_blocks_if_needed(self) -> None:
        if not self.top_k_enabled:
            return
        if self.mid_archive_capacity_blocks <= 0:
            return
        while len(self.mid_meta) > self.mid_archive_capacity_blocks:
            self._evict_archive_block(0)

    def _evict_archive_block(self, evict_idx: int) -> None:
        if not self.mid_meta:
            return
        evicted = self.mid_meta.pop(evict_idx)
        self.mid_archive_kv.pop(evict_idx)
        if self.eviction_policy == "fifo":
            self.rope_delta_frames += evicted.n_frames
        self._last_logged_selection = None
        self._record_debug_event(
            "mid_archive_evict",
            eviction_policy=self.eviction_policy,
            evicted_block_id=evicted.block_id,
            evicted_temporal_position=evicted.temporal_position,
            evicted_n_frames=evicted.n_frames,
            evicted_n_tokens=evicted.n_tokens,
            updated_rope_delta_frames=self.rope_delta_frames,
            archive_blocks=len(self.mid_meta),
            archive_total_tokens=self._archive_total_tokens(),
        )

    def get_mid_kv(self, layer_idx: int = 0) -> Optional[torch.Tensor]:
        """
        Return the current mid buffer KV for a specific layer.
        shape: (2, B, mid_token_count, n_heads, head_dim) or None if empty
        """
        if self.top_k_enabled:
            selected = self._selected_archive_indices()
            if not selected:
                self.mid_token_count = 0
                return None
            if layer_idx == 0:
                signature = tuple(self.mid_meta[i].block_id for i in selected)
                if signature != self._last_logged_selection:
                    self._last_logged_selection = signature
                    self._record_debug_event(
                        "mid_select",
                        strategy="recency",
                        top_k_blocks=self.top_k_blocks,
                        selected_block_ids=list(signature),
                        selected_temporal_positions=[
                            self.mid_meta[i].temporal_position for i in selected
                        ],
                        selected_n_tokens=[self.mid_meta[i].n_tokens for i in selected],
                        archive_blocks=len(self.mid_meta),
                        archive_total_tokens=self._archive_total_tokens(),
                        active_mid_blocks=len(selected),
                        active_mid_tokens=sum(self.mid_meta[i].n_tokens for i in selected),
                    )
            kv = torch.cat([self.mid_archive_kv[i][layer_idx] for i in selected], dim=2)
            self.mid_token_count = kv.shape[2]
            return kv
        if self.mid_kv_buffers is None or self.mid_token_count == 0:
            return None
        return self.mid_kv_buffers[layer_idx][:, :, :self.mid_token_count, :, :]

    def _selected_archive_indices(self) -> list[int]:
        if not self.mid_meta:
            return []
        start = max(0, len(self.mid_meta) - self.top_k_blocks)
        return list(range(start, len(self.mid_meta)))

    def _selected_mid_token_count(self) -> int:
        return sum(self.mid_meta[i].n_tokens for i in self._selected_archive_indices())

    def reset_mid_buffer(self) -> None:
        """Reset mid buffer at the start of each video generation."""
        self.mid_meta = []
        self.mid_archive_kv = []
        self.mid_token_count = 0
        self.rope_delta_frames = 0
        self._next_block_id = 0
        self._last_logged_selection = None
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
