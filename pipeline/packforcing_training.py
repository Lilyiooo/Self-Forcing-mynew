"""
packforcing_training.py — PackForcing training pipeline with heterogeneous KV cache.

Mirrors SelfForcingTrainingPipeline but uses three-zone cache (sink/mid/recent)
with compression during training, ensuring train-test consistency.

Supports end-to-end differentiable compression: when gradients are enabled,
compressed KV tensors are accumulated in a list (not in-place buffers) so that
DMD loss gradients flow through attention back to the compressor parameters.
"""

from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist
from utils.cache_lifecycle import queue_aged_blocks
from utils.cache_debug import make_cache_debug_logger
from utils.rope_utils import apply_temporal_rope_shift, apply_temporal_rope_to_unrotated


class PackForcingTrainingPipeline:
    def __init__(
        self,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: WanDiffusionWrapper,
        compressor,              # HeterogeneousCompressor instance
        density_estimator,       # DensityEstimator instance
        het_cache_config,        # OmegaConf with Nsink, Nrecent, Nmid_tokens
        num_frame_per_block=4,
        independent_first_frame: bool = False,
        same_step_across_blocks: bool = False,
        last_step_only: bool = False,
        num_max_frames: int = 20,
        context_noise: int = 0,
        enable_differentiable_compression: bool = False,
        top_k_config=None,
        **kwargs
    ):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]
        self._rank = dist.get_rank() if dist.is_initialized() else 0

        # Compressor and density estimator (owned by DMD model, passed in)
        self.compressor = compressor
        self.density_estimator = density_estimator
        self.het_cache_config = het_cache_config

        # Wan model hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False
        self.enable_differentiable_compression = enable_differentiable_compression

        # Read actual model dimensions from generator
        self.model_dim = getattr(generator.model, 'dim', 1536)
        self.model_num_heads = getattr(generator.model, 'num_heads', 12)
        self.model_head_dim = self.model_dim // self.model_num_heads
        self.model_num_layers = getattr(generator.model, 'num_layers', 30)
        self.num_transformer_blocks = self.model_num_layers

        # Cache config
        self.Nsink = getattr(het_cache_config, "Nsink", 8)
        self.Nrecent = getattr(het_cache_config, "Nrecent", 4)
        self.Nmid_tokens = getattr(het_cache_config, "Nmid_tokens", 5000)
        self.top_k_enabled = getattr(
            top_k_config,
            "enabled",
            getattr(het_cache_config, "top_k_enabled", False),
        )
        self.top_k_blocks = getattr(
            top_k_config,
            "top_k_blocks",
            getattr(het_cache_config, "top_k_blocks", 8),
        )
        effective_window = self.Nsink + self.Nrecent + self.num_frame_per_block
        self.effective_window = effective_window

        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length

        # Cache state (re-initialized each call)
        self.kv_cache1 = None
        self.crossattn_cache = None
        self.het_kv_cache = None
        self._recent_clean_blocks = []

        # End-to-end differentiable mid KV storage
        # When grad is enabled, compressed KV tensors are stored here instead of
        # in-place buffer writes, allowing gradients to flow from DMD loss through
        # attention back to compressor parameters.
        self._diff_mid_kv_list: List[torch.Tensor] = []
        self._diff_mode: bool = False
        self._applied_rope_delta_frames = 0
        self.cache_debug_logger = make_cache_debug_logger(het_cache_config)

    def _rank0_print(self, message: str) -> None:
        if self._rank == 0:
            print(message, flush=True)

    # ------------------------------------------------------------------
    # Cache initialization (three-zone heterogeneous cache)
    # ------------------------------------------------------------------

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """Initialize heterogeneous KV cache and set up model attention params."""
        from model.kv_cache import HeterogeneousKVCache

        # Update model attention layers for eviction
        self.generator.model.local_attn_size = self.effective_window
        for block in self.generator.model.blocks:
            block.self_attn.local_attn_size = self.effective_window
            block.self_attn.sink_size = self.Nsink
            block.self_attn.max_attention_size = self.effective_window * self.frame_seq_length

        # Create heterogeneous KV cache
        self.het_kv_cache = HeterogeneousKVCache(
            batch_size=batch_size,
            num_transformer_blocks=self.num_transformer_blocks,
            num_heads=self.model_num_heads,
            head_dim=self.model_head_dim,
            dtype=dtype,
            device=device,
            Nsink=self.Nsink,
            Nrecent=self.Nrecent,
            Nmid_tokens=self.Nmid_tokens,
            frame_seq_length=self.frame_seq_length,
            local_attn_size=self.effective_window,
            eviction_policy=getattr(self.het_cache_config, "eviction_policy", "density"),
            top_k_enabled=self.top_k_enabled,
            top_k_blocks=self.top_k_blocks,
        )
        if self.cache_debug_logger is not None:
            self.het_kv_cache.set_debug_logger(self.cache_debug_logger)

        # Reset density estimator for each new video
        self.density_estimator.reset()
        self._recent_clean_blocks = []
        self._applied_rope_delta_frames = 0

        # Set up backward-compatible aliases
        self.kv_cache1 = [self.het_kv_cache.get_layer_cache(i) for i in range(self.num_transformer_blocks)]
        self.crossattn_cache = [self.het_kv_cache.get_crossattn_cache(i) for i in range(self.num_transformer_blocks)]

    def _log_cache_block_state(self, chunk_idx: int, current_start_frame: int, event: str) -> None:
        if self.cache_debug_logger is None or self.het_kv_cache is None:
            return
        self.cache_debug_logger.log(
            event,
            chunk_idx=chunk_idx,
            current_start_frame=current_start_frame,
            current_time_sec=float(current_start_frame * 4 / 16.0),
            recent_queue_len=len(self._recent_clean_blocks),
            mid_token_count=self.het_kv_cache.mid_token_count,
            len_mid_meta=len(self.het_kv_cache.mid_meta),
            rope_delta_frames=self.het_kv_cache.rope_delta_frames,
            applied_rope_delta_frames=self._applied_rope_delta_frames,
            diff_mode=self._diff_mode,
        )

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """Cross-attention cache is handled by HeterogeneousKVCache. No-op."""
        pass

    # ------------------------------------------------------------------
    # Mid KV injection
    # ------------------------------------------------------------------

    def _get_mid_kv_per_layer(self):
        """Build per-layer mid_kv list from differentiable list or heterogeneous cache."""
        if self._diff_mode and self._diff_mid_kv_list:
            # Differentiable path: concatenate all compressed KV tensors.
            # torch.cat preserves gradient chain back to compressor parameters.
            combined_kv = torch.cat(self._diff_mid_kv_list, dim=2)  # (2, B, total_N, heads, head_dim)
            return [combined_kv] * self.num_transformer_blocks
        # Fallback: use in-place cache buffer (inference or no mid tokens)
        if self.het_kv_cache is None:
            return [None] * self.num_transformer_blocks
        return [self.het_kv_cache.get_mid_kv(i) for i in range(self.num_transformer_blocks)]

    # ------------------------------------------------------------------
    # Block compression
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _project_compressed_to_kv(self, compressed_tokens):
        """
        Project compressed tokens to KV pairs using compressor's learned projections.
        Avoids accessing model's FSDP-sharded attention layers directly.
        """
        return self.compressor.project_to_kv(
            compressed_tokens,
            num_layers=self.num_transformer_blocks,
            num_heads=self.model_num_heads,
        )

    def _apply_mid_temporal_rope(self, kv_compressed_per_layer, density_level, latent_shape, current_start_frame):
        """Apply initial temporal RoPE to compressed mid keys before caching."""
        grid_shape = self.compressor.compressed_grid_shape(density_level, latent_shape)
        freqs = self.generator.model.freqs
        roped = []
        for kv in kv_compressed_per_layer:
            k = apply_temporal_rope_to_unrotated(
                kv[0],
                freqs=freqs,
                start_frame=current_start_frame,
                grid_shape=grid_shape,
                temporal_stride=2,
            )
            roped.append(torch.stack([k, kv[1]], dim=0))
        return roped

    def _apply_sink_rope_correction_if_needed(self):
        """Apply FIFO eviction's temporal shift to already-roped sink keys."""
        if self.het_kv_cache is None:
            return

        total_delta = getattr(self.het_kv_cache, "rope_delta_frames", 0)
        delta = total_delta - self._applied_rope_delta_frames
        if delta <= 0:
            return

        sink_tokens = self.Nsink * self.frame_seq_length
        freqs = self.generator.model.freqs
        for layer_cache in self.kv_cache1:
            local_end = layer_cache["local_end_index"].item()
            n_sink = min(sink_tokens, local_end)
            if n_sink == 0:
                continue
            layer_cache["k"][:, :n_sink] = apply_temporal_rope_shift(
                layer_cache["k"][:, :n_sink],
                freqs=freqs,
                delta=delta,
            )
        self._applied_rope_delta_frames = total_delta
        if self.cache_debug_logger is not None:
            self.cache_debug_logger.log(
                "sink_rope_correction",
                delta=delta,
                rope_delta_frames=total_delta,
                applied_rope_delta_frames=self._applied_rope_delta_frames,
                diff_mode=self._diff_mode,
            )

    @torch.no_grad()
    def _compress_block(self, denoised_pred, chunk_idx, current_start_frame):
        """Compress a finished block and push into mid buffer (no gradient)."""
        if self.het_kv_cache is None or self.compressor is None:
            return

        self._rank0_print(
            f"[PackForcing] compress block chunk={chunk_idx}, start={current_start_frame}"
        )
        z_current = denoised_pred  # (B, T, C, H, W)
        z_compress = z_current.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        # Estimate density
        density_level, density_score, density_info = self.density_estimator(z_compress)

        # Compress in (B, C, T, H, W) format
        compressed_tokens, _ = self.compressor(z_compress, density_level)

        # Project compressed tokens through each layer's k_proj/v_proj
        kv_compressed_per_layer = self._project_compressed_to_kv(compressed_tokens)
        kv_compressed_per_layer = self._apply_mid_temporal_rope(
            kv_compressed_per_layer,
            density_level=density_level,
            latent_shape=z_compress.shape[2:],
            current_start_frame=current_start_frame,
        )

        # Push into mid buffer
        self.het_kv_cache.push_mid_block(
            kv_compressed=kv_compressed_per_layer,
            density_level=density_level,
            density_score=density_score,
            temporal_position=current_start_frame,
            n_frames=denoised_pred.shape[1],
        )
        self._apply_sink_rope_correction_if_needed()
        if self.cache_debug_logger is not None:
            inserted_tokens = self.het_kv_cache.mid_meta[-1].n_tokens if self.het_kv_cache.mid_meta else 0
            self.cache_debug_logger.log(
                "block_compressed",
                chunk_idx=chunk_idx,
                block_start=current_start_frame,
                block_end=current_start_frame + denoised_pred.shape[1],
                density_level=density_level,
                density_score=float(density_score),
                n_tokens=inserted_tokens,
                raw_n_tokens=compressed_tokens.shape[1],
                temporal_position=current_start_frame,
                mid_token_count=self.het_kv_cache.mid_token_count,
                rope_delta_frames=self.het_kv_cache.rope_delta_frames,
                applied_rope_delta_frames=self._applied_rope_delta_frames,
                diff_mode=self._diff_mode,
            )

    def _compress_block_differentiable(self, denoised_pred, chunk_idx, current_start_frame):
        """
        Differentiable compression — keeps gradient chain alive so DMD loss gradients
        can flow through attention back to compressor Conv3D / projection parameters.

        Stores compressed KV in _diff_mid_kv_list (not in-place buffer).
        """
        if self.compressor is None:
            return

        z_current = denoised_pred  # NOT detached — gradient flows through
        z_compress = z_current.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        # Density estimation (routing only, no learnable parameters)
        with torch.no_grad():
            density_level, density_score, density_info = self.density_estimator(z_compress)

        # Compress WITH gradient — HR branch Conv3D is fully differentiable
        compressed_tokens, _ = self.compressor(z_compress, density_level)

        # Project to KV WITH gradient — kv_k_proj / kv_v_proj are learnable
        kv_compressed = self.compressor.project_to_kv(
            compressed_tokens,
            num_layers=self.num_transformer_blocks,
            num_heads=self.model_num_heads,
        )
        kv_compressed = self._apply_mid_temporal_rope(
            kv_compressed,
            density_level=density_level,
            latent_shape=z_compress.shape[2:],
            current_start_frame=current_start_frame,
        )

        # Store in list (differentiable via torch.cat in _get_mid_kv_per_layer)
        self._diff_mid_kv_list.append(kv_compressed[0])  # (2, B, N, heads, head_dim)
        if self.cache_debug_logger is not None:
            self.cache_debug_logger.log(
                "block_compressed_diff_path",
                chunk_idx=chunk_idx,
                block_start=current_start_frame,
                block_end=current_start_frame + denoised_pred.shape[1],
                density_level=density_level,
                density_score=float(density_score),
                n_tokens=compressed_tokens.shape[1],
                temporal_position=current_start_frame,
                diff_mid_blocks=len(self._diff_mid_kv_list),
                diff_mode=self._diff_mode,
            )

    def _queue_clean_block_for_compression(self, clean_block, chunk_idx, current_start_frame):
        """Compress only blocks that have aged out of the recent window."""
        aged_blocks = queue_aged_blocks(
            self._recent_clean_blocks,
            clean_block=clean_block,
            chunk_idx=chunk_idx,
            current_start_frame=current_start_frame,
            nsink=self.Nsink,
            nrecent=self.Nrecent,
        )
        for block in aged_blocks:
            if self._diff_mode and block["latent"].requires_grad:
                self._compress_block_differentiable(
                    denoised_pred=block["latent"],
                    chunk_idx=block["chunk_idx"],
                    current_start_frame=block["start"],
                )
            else:
                latent = block["latent"].detach()
                self._compress_block(
                    denoised_pred=latent,
                    chunk_idx=block["chunk_idx"],
                    current_start_frame=block["start"],
                )
                if self._diff_mode:
                    self._compress_block_differentiable(
                        denoised_pred=latent,
                        chunk_idx=block["chunk_idx"],
                        current_start_frame=block["start"],
                    )

    # ------------------------------------------------------------------
    # Random index sync (same as SelfForcingTrainingPipeline)
    # ------------------------------------------------------------------

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)
        return indices.tolist()

    # ------------------------------------------------------------------
    # Main training rollout
    # ------------------------------------------------------------------

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        """
        Same as SelfForcingTrainingPipeline.inference_with_trajectory() but
        uses three-zone heterogeneous KV cache with compression.

        When torch.is_grad_enabled(), compression is differentiable: compressed KV
        tensors are stored in a list and concatenated via torch.cat, preserving the
        gradient chain from DMD loss through attention back to compressor parameters.
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        self._rank0_print(
            f"[PackForcing] rollout start: frames={num_frames}, "
            f"block={self.num_frame_per_block}, diff={torch.is_grad_enabled() and self.enable_differentiable_compression}"
        )
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 0: Set up differentiable compression mode
        self._diff_mode = torch.is_grad_enabled() and self.enable_differentiable_compression
        self._diff_mid_kv_list = []

        # Step 1: Initialize heterogeneous KV cache
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._log_cache_block_state(
            chunk_idx=-1,
            current_start_frame=0,
            event="cache_initialized",
        )

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    mid_kv_per_layer=self._get_mid_kv_per_layer(),
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 20  # last 20 frames get gradients

        for block_index, current_num_frames in enumerate(all_num_frames):
            self._rank0_print(
                f"[PackForcing] block {block_index + 1}/{len(all_num_frames)} "
                f"start={current_start_frame}, frames={current_num_frames}"
            )
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            is_gradient_block = current_start_frame >= start_gradient_frame_index

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            mid_kv_per_layer=self._get_mid_kv_per_layer(),
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    if not is_gradient_block:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                                mid_kv_per_layer=self._get_mid_kv_per_layer(),
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            mid_kv_per_layer=self._get_mid_kv_per_layer(),
                        )
                    break

            clean_block = denoised_pred

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    mid_kv_per_layer=self._get_mid_kv_per_layer(),
                )

            # Step 3.3a: Queue CLEAN latent and compress only after it ages out
            # of the full-resolution recent window. Sink frames are never
            # compressed into mid.
            if is_gradient_block and self._diff_mode:
                clean_denoised = clean_block
            else:
                clean_denoised = clean_block.detach()
            self._queue_clean_block_for_compression(
                clean_block=clean_denoised,
                chunk_idx=block_index,
                current_start_frame=current_start_frame,
            )
            self._log_cache_block_state(
                chunk_idx=block_index,
                current_start_frame=current_start_frame,
                event="block_generated",
            )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to
