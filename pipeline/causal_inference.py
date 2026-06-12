from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.cache_lifecycle import queue_aged_blocks
from utils.cache_debug import make_cache_debug_logger
from utils.rope_utils import apply_temporal_rope_shift, apply_temporal_rope_to_unrotated

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation

# NOTE: model.* imports are done lazily inside __init__ to avoid circular import:
#   pipeline/__init__ → causal_inference → model/__init__ → base → pipeline (cycle)


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        self.device = device
        # Lazy imports to avoid circular dependency (pipeline ↔ model.base)
        from model.density_estimator import DensityEstimator
        from model.compress import HeterogeneousCompressor
        from model.kv_cache import HeterogeneousKVCache
        self._DensityEstimator = DensityEstimator
        self._HeterogeneousCompressor = HeterogeneousCompressor
        self._HeterogeneousKVCache = HeterogeneousKVCache

        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        # Read actual transformer dimensions from generator model
        self.model_dim = getattr(self.generator.model, 'dim', 1536)
        self.model_num_heads = getattr(self.generator.model, 'num_heads', 12)
        self.model_head_dim = self.model_dim // self.model_num_heads
        self.model_num_layers = getattr(self.generator.model, 'num_layers', 30)
        # Override hardcoded values with actual model config
        self.num_transformer_blocks = self.model_num_layers

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        # Step 3: Initialize heterogeneous KV cache components
        self.heterogeneous_cache_enabled = False
        het_cfg = getattr(args, "heterogeneous_cache", None)
        if het_cfg is not None and getattr(het_cfg, "enabled", False):
            self.heterogeneous_cache_enabled = True
            self.het_cache_config = het_cfg
            top_k_cfg = getattr(args, "top_k", None)
            self.top_k_enabled = getattr(
                top_k_cfg,
                "enabled",
                getattr(het_cfg, "top_k_enabled", False),
            )
            self.top_k_blocks = getattr(
                top_k_cfg,
                "top_k_blocks",
                getattr(het_cfg, "top_k_blocks", 8),
            )
            print(f"Heterogeneous KV cache enabled: Nsink={het_cfg.Nsink}, "
                  f"Nrecent={het_cfg.Nrecent}, Nmid_tokens={het_cfg.Nmid_tokens}")

            # Density estimator
            density_cfg = getattr(args, "density_estimator", None)
            self.density_estimator = self._DensityEstimator(
                high_threshold=getattr(density_cfg, "delta_high", 0.67),
                low_threshold=getattr(density_cfg, "delta_low", 0.33),
                motion_weight=getattr(density_cfg, "motion_weight", 0.6),
                complexity_weight=getattr(density_cfg, "complexity_weight", 0.4),
            )

            # Compressor (will be initialized lazily on first device placement)
            self.compressor = None
            self.het_kv_cache = None
            self._recent_clean_blocks = []
            self._applied_rope_delta_frames = 0
            self.cache_debug_logger = make_cache_debug_logger(args)
        else:
            self.cache_debug_logger = None

    def load_compressor_state_dict(self, state_dict, device=None, dtype=None, strict=True):
        """Load trained heterogeneous compressor weights for validation/inference."""
        if not self.heterogeneous_cache_enabled:
            return False

        if self.compressor is None:
            compressor_cfg = getattr(self.args, "compressor", None)
            in_ch = getattr(compressor_cfg, "in_ch", 16)
            self.compressor = self._HeterogeneousCompressor(
                vae=self.vae,
                d_model=self.model_dim,
                in_ch=in_ch,
                num_layers=self.model_num_layers,
            )

        if device is not None or dtype is not None:
            self.compressor = self.compressor.to(device=device, dtype=dtype)
        self.compressor.load_state_dict(state_dict, strict=strict)
        return True

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache
        if self.heterogeneous_cache_enabled:
            self._initialize_heterogeneous_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device
            )
            self._log_cache_block_state(
                chunk_idx=-1,
                current_start_frame=0,
                event="cache_initialized",
            )
        else:
            if self.kv_cache1 is None:
                self._initialize_kv_cache(
                    batch_size=batch_size,
                    dtype=noise.dtype,
                    device=noise.device
                )
                self._initialize_crossattn_cache(
                    batch_size=batch_size,
                    dtype=noise.dtype,
                    device=noise.device
                )
            else:
                # reset cross attn cache
                for block_index in range(self.num_transformer_blocks):
                    self.crossattn_cache[block_index]["is_init"] = False
                # reset kv cache
                for block_index in range(len(self.kv_cache1)):
                    self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self._run_generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    current_start_frame=current_start_frame,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self._run_generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    current_start_frame=current_start_frame,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        # For heterogeneous cache: track previous chunk latent for density estimation
        z_prev_chunk = None

        for chunk_idx, current_num_frames in enumerate(all_num_frames):
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                print(f"current_timestep: {current_timestep}")
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self._run_generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        current_start_frame=current_start_frame,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self._run_generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        current_start_frame=current_start_frame,
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            clean_denoised = denoised_pred.detach().clone()  # Save clean latent BEFORE cache update

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self._run_generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                current_start_frame=current_start_frame,
            )

            # Step 3.3a: Heterogeneous cache - queue CLEAN latent and compress
            # only after it leaves the full-resolution recent window.
            if self.heterogeneous_cache_enabled:
                self._queue_clean_block_for_compression(
                    clean_block=clean_denoised,
                    chunk_idx=chunk_idx,
                    current_start_frame=current_start_frame,
                    z_prev_chunk=z_prev_chunk,
                )
                z_prev_chunk = clean_denoised
                self._log_cache_block_state(
                    chunk_idx=chunk_idx,
                    current_start_frame=current_start_frame,
                    event="block_generated",
                )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    # ------------------------------------------------------------------
    # Heterogeneous cache methods
    # ------------------------------------------------------------------

    def _initialize_heterogeneous_cache(self, batch_size, dtype, device):
        """Initialize the heterogeneous KV cache (three-zone structure)."""
        het_cfg = self.het_cache_config
        Nsink = getattr(het_cfg, "Nsink", 8)
        Nrecent = getattr(het_cfg, "Nrecent", 4)

        # Compute effective attention window for KV cache eviction:
        # sink frames + recent window + 1 block buffer for current write
        effective_window = Nsink + Nrecent + self.num_frame_per_block

        # Update the model's attention layers to enable eviction with the
        # correct window size and sink retention.  Without this the model
        # uses local_attn_size=-1 which disables eviction entirely, and the
        # fixed-size KV cache overflows for long videos (>21 latent frames).
        self.generator.model.local_attn_size = effective_window
        for block in self.generator.model.blocks:
            block.self_attn.local_attn_size = effective_window
            block.self_attn.sink_size = Nsink
            block.self_attn.max_attention_size = effective_window * self.frame_seq_length
        self.local_attn_size = effective_window

        # Initialize the compressor (lazy, on first call)
        if self.compressor is None:
            compressor_cfg = getattr(self.args, "compressor", None)
            in_ch = getattr(compressor_cfg, "in_ch", 16)
            # Use actual model hidden dim as d_model for KV compatibility
            d_model = self.model_dim
            self.compressor = self._HeterogeneousCompressor(
                vae=self.vae,
                d_model=d_model,
                in_ch=in_ch,
                num_layers=self.model_num_layers,
            ).to(device=device, dtype=dtype)
        else:
            self.compressor = self.compressor.to(device=device, dtype=dtype)

        # Initialize the heterogeneous KV cache
        self.het_kv_cache = self._HeterogeneousKVCache(
            batch_size=batch_size,
            num_transformer_blocks=self.num_transformer_blocks,
            num_heads=self.model_num_heads,
            head_dim=self.model_head_dim,
            dtype=dtype,
            device=device,
            Nsink=Nsink,
            Nrecent=Nrecent,
            Nmid_tokens=getattr(het_cfg, "Nmid_tokens", 5000),
            frame_seq_length=self.frame_seq_length,
            local_attn_size=effective_window,
            eviction_policy=getattr(het_cfg, "eviction_policy", "density"),
            top_k_enabled=self.top_k_enabled,
            top_k_blocks=self.top_k_blocks,
            mid_archive_capacity_blocks=getattr(het_cfg, "mid_archive_capacity_blocks", 64),
        )
        if self.cache_debug_logger is not None:
            self.het_kv_cache.set_debug_logger(self.cache_debug_logger)

        # Reset density estimator
        self.density_estimator.reset_stats()
        self._recent_clean_blocks = []
        self._applied_rope_delta_frames = 0

        # Set up the old-style caches as aliases for backward compatibility
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
        )

    def _run_generator(self, noisy_image_or_video, conditional_dict, timestep, current_start_frame):
        """Run generator with or without heterogeneous cache."""
        if self.heterogeneous_cache_enabled and self.het_kv_cache is not None:
            # Build per-layer mid_kv list
            mid_kv_per_layer = []
            for layer_idx in range(self.num_transformer_blocks):
                mid_kv = self.het_kv_cache.get_mid_kv(layer_idx)
                mid_kv_per_layer.append(mid_kv)

            return self.generator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                mid_kv_per_layer=mid_kv_per_layer,
            )
        else:
            return self.generator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

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

        sink_tokens = getattr(self.het_cache_config, "Nsink", 8) * self.frame_seq_length
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
            )

    @torch.no_grad()
    def _maybe_compress_block(self, denoised_pred, chunk_idx, current_start_frame, z_prev_chunk):
        """
        Compress a finished block and push it into the mid buffer.
        Only starts compressing after the recent window is full.
        """
        if self.het_kv_cache is None or self.compressor is None:
            return

        # Current chunk latent (for density estimation)
        z_current = denoised_pred  # (B, T, C, H, W)
        z_compress = z_current.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        # Estimate density (v2: single call, no dependency on z_prev or LR complexity)
        density_level, density_score, density_info = self.density_estimator(z_compress)

        # Compress: need latent in (B, C, T, H, W) format for compressor
        compressed_tokens, _complexity_from_lr = self.compressor(z_compress, density_level)

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

        B, N, D = compressed_tokens.shape
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
                raw_n_tokens=N,
                temporal_position=current_start_frame,
                mid_token_count=self.het_kv_cache.mid_token_count,
                rope_delta_frames=self.het_kv_cache.rope_delta_frames,
                applied_rope_delta_frames=self._applied_rope_delta_frames,
            )
        print(f"  [HetCache] chunk {chunk_idx}: density={density_score:.3f} ({density_level}), "
              f"tokens={N}, mid_total={self.het_kv_cache.mid_token_count}/{self.het_kv_cache.Nmid_tokens}")

        # Optional density logging (used by experiment_A_density_vis)
        if hasattr(self, '_density_log') and self._density_log is not None:
            self._density_log.append({
                "block_idx": int(chunk_idx),
                "time_sec": float(chunk_idx * self.num_frame_per_block * 4 / 16.0),
                "motion_score": density_info["raw_motion"],
                "complexity_score": density_info["raw_complexity"],
                "norm_motion": density_info["norm_motion"],
                "norm_complexity": density_info["norm_complexity"],
                "density_score": float(density_score),
                "tier": density_level,
                "tokens_allocated": int(N),
            })

    def _queue_clean_block_for_compression(self, clean_block, chunk_idx, current_start_frame, z_prev_chunk=None):
        """Keep sink/recent full-res; compress only aged historical blocks."""
        if self.het_kv_cache is None or self.compressor is None:
            return

        Nsink = getattr(self.het_cache_config, "Nsink", 8)
        Nrecent = getattr(self.het_cache_config, "Nrecent", 4)
        aged_blocks = queue_aged_blocks(
            self._recent_clean_blocks,
            clean_block=clean_block,
            chunk_idx=chunk_idx,
            current_start_frame=current_start_frame,
            nsink=Nsink,
            nrecent=Nrecent,
            extra={"prev": z_prev_chunk},
        )
        for block in aged_blocks:
            self._maybe_compress_block(
                denoised_pred=block["latent"],
                chunk_idx=block["chunk_idx"],
                current_start_frame=block["start"],
                z_prev_chunk=block["prev"],
            )
