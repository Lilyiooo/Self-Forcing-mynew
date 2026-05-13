"""
packforcing_training.py — PackForcing training pipeline with heterogeneous KV cache.

Mirrors SelfForcingTrainingPipeline but uses three-zone cache (sink/mid/recent)
with compression during training, ensuring train-test consistency.
"""

from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


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
        **kwargs
    ):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

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
        )

        # Reset density estimator for each new video
        self.density_estimator.reset()

        # Set up backward-compatible aliases
        self.kv_cache1 = [self.het_kv_cache.get_layer_cache(i) for i in range(self.num_transformer_blocks)]
        self.crossattn_cache = [self.het_kv_cache.get_crossattn_cache(i) for i in range(self.num_transformer_blocks)]

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """Cross-attention cache is handled by HeterogeneousKVCache. No-op."""
        pass

    # ------------------------------------------------------------------
    # Mid KV injection
    # ------------------------------------------------------------------

    def _get_mid_kv_per_layer(self):
        """Build per-layer mid_kv list from heterogeneous cache."""
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

    @torch.no_grad()
    def _compress_block(self, denoised_pred, chunk_idx, current_start_frame):
        """Compress a finished block and push into mid buffer."""
        if self.het_kv_cache is None or self.compressor is None:
            return

        z_current = denoised_pred  # (B, T, C, H, W)

        # Estimate density
        density_level, density_score, density_info = self.density_estimator(z_current)

        # Compress: need (B, C, T, H, W) format
        z_compress = z_current.permute(0, 2, 1, 3, 4)
        compressed_tokens, _ = self.compressor(z_compress, density_level)

        # Project compressed tokens through each layer's k_proj/v_proj
        kv_compressed_per_layer = self._project_compressed_to_kv(compressed_tokens)

        # Push into mid buffer
        self.het_kv_cache.push_mid_block(
            kv_compressed=kv_compressed_per_layer,
            density_level=density_level,
            density_score=density_score,
            temporal_position=current_start_frame,
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
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
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

        # Step 1: Initialize heterogeneous KV cache
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
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
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

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
                    if current_start_frame < start_gradient_frame_index:
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

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            clean_denoised = denoised_pred.detach()  # Save clean latent BEFORE adding noise

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

            # Step 3.3a: Compress CLEAN latent (not the noised version)
            with torch.no_grad():
                self._compress_block(
                    denoised_pred=clean_denoised,
                    chunk_idx=block_index,
                    current_start_frame=current_start_frame,
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
