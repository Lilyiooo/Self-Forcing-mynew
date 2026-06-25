import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD
import torch
import torch.nn.functional as F
import wandb
import time
import os
from einops import rearrange

# Heterogeneous cache imports
from model.density_estimator import DensityEstimator
from model.compress import HeterogeneousCompressor
from utils.rope_utils import apply_temporal_rope_to_unrotated


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        # Load compressor checkpoint (for end-to-end training resume)
        if getattr(config, "compressor_ckpt", False):
            print(f"Loading compressor checkpoint from {config.compressor_ckpt}")
            ckpt = torch.load(config.compressor_ckpt, map_location="cpu")
            compressor_sd = ckpt.get("compressor", ckpt)
            # Will be loaded after compressor is initialized (step 8)
            self._pending_compressor_sd = compressor_sd
        else:
            self._pending_compressor_sd = None

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

        # Step 8: Initialize heterogeneous cache components (if enabled)
        self.heterogeneous_cache_enabled = getattr(
            self.model, 'heterogeneous_cache_enabled', False)
        if self.heterogeneous_cache_enabled:
            # Use compressor created by DMD model (avoids duplicate creation)
            self.compressor = getattr(self.model, 'compressor', None)
            if self.compressor is not None:
                self.compressor = self.compressor.to(device=self.device, dtype=self.dtype)

            compressor_train_cfg = getattr(config, "compressor_training", None)
            self.lambda_recon = getattr(compressor_train_cfg, "lambda_recon", 0.01) if compressor_train_cfg else 0.01
            self.recon_warmup_steps = getattr(compressor_train_cfg, "warmup_steps", 100) if compressor_train_cfg else 100

            # End-to-end compressor training: separate optimizer with its own LR
            self.end_to_end_compressor = getattr(compressor_train_cfg, "end_to_end", False) if compressor_train_cfg else False
            if self.end_to_end_compressor and self.compressor is not None:
                compressor_lr = getattr(compressor_train_cfg, "compressor_lr", 1e-4)
                self.compressor_optimizer = torch.optim.AdamW(
                    [p for p in self.compressor.parameters() if p.requires_grad],
                    lr=compressor_lr,
                    betas=(0.0, 0.999),
                    weight_decay=0.01,
                )
                self.compressor_grad_clip = getattr(compressor_train_cfg, "compressor_grad_clip", 1.0)
                print(f"End-to-end compressor training enabled: lr={compressor_lr}, "
                      f"grad_clip={self.compressor_grad_clip}")
            else:
                self.compressor_optimizer = None
                self.compressor_grad_clip = 1.0

            print(f"Heterogeneous cache training enabled: lambda_recon={self.lambda_recon}, "
                  f"warmup_steps={self.recon_warmup_steps}, end_to_end={self.end_to_end_compressor}")
        else:
            self.compressor = None
            self.compressor_optimizer = None
            self.end_to_end_compressor = False

        # Load compressor state dict from checkpoint (after initialization)
        if self._pending_compressor_sd is not None and self.compressor is not None:
            self.compressor.load_state_dict(self._pending_compressor_sd, strict=True)
            print("Compressor checkpoint loaded successfully.")
            self._pending_compressor_sd = None

    def pretrain_compressor(self, num_steps=100):
        """
        Pretrain HR compression heads via autoencoder reconstruction.
        - Uses random latent input (no generator needed, saves huge compute).
        - Each HR head encodes, its symmetric decoder reconstructs.
        - Loss = MSE(reconstructed, original) — real information-preserving objective.
        - LR branch is skipped entirely (no VAE calls during pretrain).
        - Decoder weights are discarded after pretrain; only HR heads are kept.
        - Runs in float32 for gradient precision (bfloat16 is too imprecise for pretrain).
        """
        if not self.heterogeneous_cache_enabled or self.compressor is None:
            return

        if self.is_main_process:
            print("Starting compressor pretrain (autoencoder reconstruction)...")

        # Freeze backbone
        for param in self.model.generator.parameters():
            param.requires_grad_(False)

        # Only train HR heads + decoders (skip LR branch and proj)
        for name, param in self.compressor.named_parameters():
            if "hr_" in name or "decoder_" in name or "kv_" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        # Temporarily cast trainable modules to float32 for stable gradients
        pretrain_modules = [
            self.compressor.hr_high, self.compressor.hr_mid, self.compressor.hr_low,
            self.compressor.decoder_high, self.compressor.decoder_mid, self.compressor.decoder_low,
            self.compressor.kv_k_proj, self.compressor.kv_v_proj,
        ]
        for m in pretrain_modules:
            m.float()

        compressor_train_cfg = getattr(self.config, "compressor_training", None) or OmegaConf.create({})
        optimizer = torch.optim.AdamW(
            [p for p in self.compressor.parameters() if p.requires_grad],
            lr=getattr(compressor_train_cfg, "pretrain_lr", 1e-3),
        )

        # Latent shape config (defaults match Wan2.1-T2V at 480p)
        compressor_cfg = getattr(self.config, "compressor", None) or OmegaConf.create({})
        in_ch = getattr(compressor_cfg, "in_ch", 16)
        pretrain_batch_size = getattr(compressor_train_cfg, "pretrain_batch_size", 2)
        latent_T = getattr(
            compressor_train_cfg, "latent_T",
            getattr(self.config, "num_frame_per_block", 3),
        )
        latent_H = getattr(compressor_train_cfg, "latent_H", 60)
        latent_W = getattr(compressor_train_cfg, "latent_W", 104)
        target_shape = (latent_T, latent_H, latent_W)

        if self.is_main_process:
            print(f"  latent shape: (B={pretrain_batch_size}, C={in_ch}, T={latent_T}, H={latent_H}, W={latent_W})")
            print(f"  lr={optimizer.param_groups[0]['lr']}, steps={num_steps}")

        # HR head + decoder pairs
        head_decoder_pairs = [
            ("8x",   self.compressor.hr_high, self.compressor.decoder_high),
            ("32x",  self.compressor.hr_mid,  self.compressor.decoder_mid),
            ("128x", self.compressor.hr_low,  self.compressor.decoder_low),
        ]

        for step_idx in range(num_steps):
            # Random latent in float32 (matches actual latent distribution ~ N(0,1))
            z_input = torch.randn(
                pretrain_batch_size, in_ch, *target_shape,
                device=self.device, dtype=torch.float32,
            )

            total_loss = torch.tensor(0.0, device=self.device)
            for name, hr_head, decoder in head_decoder_pairs:
                compressed = hr_head(z_input)                          # (B, N, D)
                reconstructed = decoder(compressed, target_shape)      # (B, C, T, H, W)
                loss = F.mse_loss(reconstructed, z_input)
                total_loss = total_loss + loss

                # KV projection warmup: ensure kv_k_proj/kv_v_proj produce
                # non-degenerate output (not all zeros / all same)
                B_c, N_c, D_c = compressed.shape
                kv_reg = torch.tensor(0.0, device=self.device)
                for k_proj, v_proj in zip(self.compressor.kv_k_proj, self.compressor.kv_v_proj):
                    k_out = k_proj(compressed)
                    v_out = v_proj(compressed)
                    kv_reg = kv_reg + (
                        (k_out.var(dim=-1).mean() - 1.0).pow(2)
                        + (v_out.var(dim=-1).mean() - 1.0).pow(2)
                    )
                # Regularize: every per-layer KV projection should produce
                # non-degenerate output. This is not KV distillation yet.
                total_loss = total_loss + 0.01 * kv_reg / max(1, len(self.compressor.kv_k_proj))

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.compressor.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            if step_idx % 10 == 0 and self.is_main_process:
                print(f"  Compressor pretrain step {step_idx}/{num_steps}, loss={total_loss.item():.4f}")

        # Cast HR heads back to training dtype
        for m in pretrain_modules:
            if m is not None:
                m.to(dtype=self.dtype)

        # Restore trainable states
        for param in self.model.generator.parameters():
            param.requires_grad_(True)
        for param in self.compressor.parameters():
            param.requires_grad_(True)

        # Discard decoders only if not doing joint PackForcing training
        # (PackForcing needs decoders for reconstruction auxiliary loss)
        if not self.heterogeneous_cache_enabled:
            self.compressor.decoder_high = None
            self.compressor.decoder_mid = None
            self.compressor.decoder_low = None
            if self.is_main_process:
                print("Compressor pretraining complete. Decoders discarded.")
        else:
            if self.is_main_process:
                print("Compressor pretraining complete. Decoders kept for joint training.")

    def _unwrap_generator(self):
        return self.model.generator.module if hasattr(self.model.generator, "module") else self.model.generator

    def _pool_teacher_kv_to_grid(self, teacher_kv, full_grid_shape, target_grid_shape):
        """Pool full-resolution per-layer KV to the compressed token grid."""
        batch_size, num_tokens, num_heads, head_dim = teacher_kv.shape
        frames, height, width = full_grid_shape
        assert num_tokens == frames * height * width, (
            f"teacher token mismatch: {num_tokens} != {full_grid_shape}"
        )
        x = teacher_kv.reshape(batch_size, frames, height, width, num_heads, head_dim)
        x = x.permute(0, 4, 5, 1, 2, 3).reshape(
            batch_size, num_heads * head_dim, frames, height, width
        )
        x = F.adaptive_avg_pool3d(x.float(), output_size=target_grid_shape)
        out_t, out_h, out_w = target_grid_shape
        x = x.reshape(batch_size, num_heads, head_dim, out_t, out_h, out_w)
        return x.permute(0, 3, 4, 5, 1, 2).reshape(
            batch_size, out_t * out_h * out_w, num_heads, head_dim
        ).to(teacher_kv.dtype)

    def _set_kv_capture(self, capture_list):
        generator = self._unwrap_generator()
        for block in generator.model.blocks:
            block.self_attn._kv_capture_list = capture_list

    def _clear_kv_capture(self):
        generator = self._unwrap_generator()
        for block in generator.model.blocks:
            if hasattr(block.self_attn, "_kv_capture_list"):
                delattr(block.self_attn, "_kv_capture_list")

    def _set_attn_distill_capture(self, capture_list):
        generator = self._unwrap_generator()
        for block in generator.model.blocks:
            block.self_attn._attn_distill_capture_list = capture_list

    def _clear_attn_distill_capture(self):
        generator = self._unwrap_generator()
        for block in generator.model.blocks:
            if hasattr(block.self_attn, "_attn_distill_capture_list"):
                delattr(block.self_attn, "_attn_distill_capture_list")

    def _set_attn_context_capture(self, capture_list):
        generator = self._unwrap_generator()
        for block in generator.model.blocks:
            block.self_attn._attn_context_capture_list = capture_list

    def _clear_attn_context_capture(self):
        generator = self._unwrap_generator()
        for block in generator.model.blocks:
            if hasattr(block.self_attn, "_attn_context_capture_list"):
                delattr(block.self_attn, "_attn_context_capture_list")

    def _select_distill_layers(self, used_layers: int, max_layers: int):
        if max_layers <= 0 or max_layers >= used_layers:
            return list(range(used_layers))
        if max_layers == 1:
            return [used_layers - 1]
        return torch.linspace(0, used_layers - 1, steps=max_layers).round().long().unique().tolist()

    def _sample_spatial_balanced_query_indices(
        self,
        frames: int,
        height: int,
        width: int,
        num_query_tokens: int,
        device,
        grid_h: int = 4,
        grid_w: int = 4,
    ):
        total_tokens = int(frames) * int(height) * int(width)
        if num_query_tokens <= 0 or total_tokens <= num_query_tokens:
            return torch.arange(total_tokens, device=device, dtype=torch.long)

        indices = []
        grid_h = max(1, int(grid_h))
        grid_w = max(1, int(grid_w))
        height = int(height)
        width = int(width)
        for t in range(int(frames)):
            frame_offset = t * height * width
            for gh in range(grid_h):
                h0 = (gh * height) // grid_h
                h1 = ((gh + 1) * height) // grid_h
                if h1 <= h0:
                    continue
                for gw in range(grid_w):
                    w0 = (gw * width) // grid_w
                    w1 = ((gw + 1) * width) // grid_w
                    if w1 <= w0:
                        continue
                    h = (h0 + h1 - 1) // 2
                    w = (w0 + w1 - 1) // 2
                    indices.append(frame_offset + h * width + w)

        if len(indices) < num_query_tokens:
            indices.extend(
                torch.linspace(
                    0,
                    total_tokens - 1,
                    steps=num_query_tokens,
                    device=device,
                ).round().long().tolist()
            )

        indices = torch.tensor(indices, device=device, dtype=torch.long).unique(sorted=True)
        if indices.numel() > num_query_tokens:
            keep = torch.linspace(
                0,
                indices.numel() - 1,
                steps=num_query_tokens,
                device=device,
            ).round().long()
            indices = indices.index_select(0, keep)
        return indices

    def _attention_output_distill_loss(
        self,
        query,
        teacher_k,
        teacher_v,
        student_k,
        student_v,
        num_query_tokens: int,
        query_grid_shape=None,
        query_sampling: str = "uniform",
        query_grid_h: int = 4,
        query_grid_w: int = 4,
        spatial_ratio: float = 0.5,
    ):
        """Match Attn(Q, compressed K/V) to Attn(Q, full K/V) on sampled queries."""
        num_tokens = query.shape[1]
        if num_tokens > num_query_tokens:
            if query_sampling == "mixed" and query_grid_shape is not None:
                num_spatial = int(round(num_query_tokens * float(spatial_ratio)))
                num_spatial = min(max(0, num_spatial), num_query_tokens)
                spatial_indices = self._sample_spatial_balanced_query_indices(
                    frames=query_grid_shape[0],
                    height=query_grid_shape[1],
                    width=query_grid_shape[2],
                    num_query_tokens=max(1, num_spatial),
                    device=query.device,
                    grid_h=query_grid_h,
                    grid_w=query_grid_w,
                ) if num_spatial > 0 else torch.empty(0, device=query.device, dtype=torch.long)
                num_uniform = max(0, num_query_tokens - spatial_indices.numel())
                if num_uniform > 0:
                    uniform_indices = torch.linspace(
                        0, num_tokens - 1, steps=num_uniform,
                        device=query.device,
                    ).round().long()
                    indices = torch.cat([spatial_indices, uniform_indices]).unique(sorted=True)
                else:
                    indices = spatial_indices
                if indices.numel() > num_query_tokens:
                    keep = torch.linspace(
                        0, indices.numel() - 1, steps=num_query_tokens,
                        device=query.device,
                    ).round().long()
                    indices = indices.index_select(0, keep)
            elif query_sampling == "spatial_balanced" and query_grid_shape is not None:
                indices = self._sample_spatial_balanced_query_indices(
                    frames=query_grid_shape[0],
                    height=query_grid_shape[1],
                    width=query_grid_shape[2],
                    num_query_tokens=num_query_tokens,
                    device=query.device,
                    grid_h=query_grid_h,
                    grid_w=query_grid_w,
                )
            else:
                indices = torch.linspace(
                    0, num_tokens - 1, steps=num_query_tokens,
                    device=query.device,
                ).round().long()
            query = query.index_select(1, indices)

        q = query.detach().transpose(1, 2).float()
        teacher_k_t = teacher_k.detach().transpose(1, 2).float()
        teacher_v_t = teacher_v.detach().transpose(1, 2).float()
        student_k_t = student_k.transpose(1, 2).float()
        student_v_t = student_v.transpose(1, 2).float()

        with torch.no_grad():
            teacher_out = F.scaled_dot_product_attention(
                q, teacher_k_t, teacher_v_t, dropout_p=0.0, is_causal=False
            )
        student_out = F.scaled_dot_product_attention(
            q, student_k_t, student_v_t, dropout_p=0.0, is_causal=False
        )
        teacher_out = teacher_out.transpose(1, 2)
        student_out = student_out.transpose(1, 2)
        loss = F.mse_loss(student_out, teacher_out)
        loss = loss + (1 - F.cosine_similarity(
            student_out.flatten(2),
            teacher_out.flatten(2),
            dim=-1,
        ).mean())
        return loss

    def _context_replacement_attn_loss(
        self,
        query,
        teacher_context,
        student_k,
        student_v,
        target_start_token: int,
        target_num_tokens: int,
        num_query_tokens: int,
        query_grid_shape=None,
        query_sampling: str = "uniform",
        query_grid_h: int = 4,
        query_grid_w: int = 4,
        spatial_ratio: float = 0.5,
    ):
        """Match future attention when only the target block is compressed."""
        context_k = teacher_context["k"]
        context_v = teacher_context["v"]
        context_start = int(teacher_context.get("context_start", 0))
        target_start = int(target_start_token) - context_start
        target_end = target_start + int(target_num_tokens)
        if target_start < 0 or target_end > context_k.shape[1]:
            return None

        student_context_k = torch.cat(
            [
                context_k[:, :target_start].detach(),
                student_k,
                context_k[:, target_end:].detach(),
            ],
            dim=1,
        )
        student_context_v = torch.cat(
            [
                context_v[:, :target_start].detach(),
                student_v,
                context_v[:, target_end:].detach(),
            ],
            dim=1,
        )
        return self._attention_output_distill_loss(
            query=query,
            teacher_k=context_k,
            teacher_v=context_v,
            student_k=student_context_k,
            student_v=student_context_v,
            num_query_tokens=num_query_tokens,
            query_grid_shape=query_grid_shape,
            query_sampling=query_sampling,
            query_grid_h=query_grid_h,
            query_grid_w=query_grid_w,
            spatial_ratio=spatial_ratio,
        )

    def _multi_context_replacement_attn_loss(
        self,
        query,
        teacher_context,
        replacements,
        num_query_tokens: int,
        query_grid_shape=None,
        query_sampling: str = "uniform",
        query_grid_h: int = 4,
        query_grid_w: int = 4,
        spatial_ratio: float = 0.5,
    ):
        """Match future attention when multiple context spans are compressed."""
        if not replacements:
            return None
        context_k = teacher_context["k"]
        context_v = teacher_context["v"]
        context_start = int(teacher_context.get("context_start", 0))

        spans = []
        for item in replacements:
            target_start = int(item["target_start_token"]) - context_start
            target_end = target_start + int(item["target_num_tokens"])
            if target_start < 0 or target_end > context_k.shape[1] or target_start >= target_end:
                return None
            spans.append((target_start, target_end, item["student_k"], item["student_v"]))
        spans.sort(key=lambda x: x[0])
        for prev, curr in zip(spans, spans[1:]):
            if curr[0] < prev[1]:
                return None

        k_parts = []
        v_parts = []
        cursor = 0
        for target_start, target_end, student_k, student_v in spans:
            k_parts.append(context_k[:, cursor:target_start].detach())
            v_parts.append(context_v[:, cursor:target_start].detach())
            k_parts.append(student_k)
            v_parts.append(student_v)
            cursor = target_end
        k_parts.append(context_k[:, cursor:].detach())
        v_parts.append(context_v[:, cursor:].detach())

        return self._attention_output_distill_loss(
            query=query,
            teacher_k=context_k,
            teacher_v=context_v,
            student_k=torch.cat(k_parts, dim=1),
            student_v=torch.cat(v_parts, dim=1),
            num_query_tokens=num_query_tokens,
            query_grid_shape=query_grid_shape,
            query_sampling=query_sampling,
            query_grid_h=query_grid_h,
            query_grid_w=query_grid_w,
            spatial_ratio=spatial_ratio,
        )

    def _load_kv_distill_prompts(self, prompt_path: str):
        if not prompt_path:
            return []
        try:
            with open(prompt_path, encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            if self.is_main_process:
                print(f"[KV distill] Prompt path not found: {prompt_path}; falling back to empty prompts.")
            return []

    def _make_ar_rollout_caches(
        self,
        batch_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        kv_cache = []
        crossattn_cache = []
        for _ in range(num_layers):
            kv_cache.append({
                "k": torch.zeros(
                    [batch_size, num_tokens, num_heads, head_dim],
                    device=device, dtype=dtype,
                ),
                "v": torch.zeros(
                    [batch_size, num_tokens, num_heads, head_dim],
                    device=device, dtype=dtype,
                ),
                "global_end_index": torch.tensor([0], device=device, dtype=torch.long),
                "local_end_index": torch.tensor([0], device=device, dtype=torch.long),
            })
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, num_heads, head_dim], device=device, dtype=dtype),
                "v": torch.zeros([batch_size, 512, num_heads, head_dim], device=device, dtype=dtype),
                "is_init": False,
            })
        return kv_cache, crossattn_cache

    @torch.no_grad()
    def _sample_ar_rollout_distill_block(
        self,
        conditional_dict,
        batch_size: int,
        in_ch: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        denoise_timestep: int,
        rollout_blocks: int,
        target_block_index: int,
        future_gap_blocks: int,
        generator_model,
        target_block_indices=None,
    ):
        """Generate short AR cache history and return a past block plus future-query captures."""
        frame_seq_length = (latent_h // 2) * (latent_w // 2)
        cache_tokens = max(1, rollout_blocks * latent_t * frame_seq_length)
        kv_cache, crossattn_cache = self._make_ar_rollout_caches(
            batch_size=batch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            num_tokens=cache_tokens,
            dtype=self.dtype,
            device=self.device,
        )

        if target_block_indices is None:
            target_block_indices = [int(target_block_index)]
        target_block_indices = sorted(set(int(i) for i in target_block_indices))
        target_block_set = set(target_block_indices)
        primary_target_index = int(target_block_indices[-1])

        target_blocks = {}
        target_kvs = {}
        target_attns = {}
        future_attn = None
        future_context = None
        current_start_frame = 0
        future_block_index = min(
            rollout_blocks - 1,
            primary_target_index + max(1, int(future_gap_blocks)),
        )

        for block_idx in range(rollout_blocks):
            noisy_z = torch.randn(
                batch_size, latent_t, in_ch, latent_h, latent_w,
                device=self.device, dtype=self.dtype,
            )
            denoise_t = torch.full(
                (batch_size, latent_t),
                int(denoise_timestep),
                device=self.device,
                dtype=torch.int64,
            )
            captured_future_attn = []
            captured_future_context = []
            if block_idx == future_block_index:
                self._set_attn_distill_capture(captured_future_attn)
                self._set_attn_context_capture(captured_future_context)
            try:
                _, pred_x0 = self.model.generator(
                    noisy_image_or_video=noisy_z,
                    conditional_dict=conditional_dict,
                    timestep=denoise_t,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * frame_seq_length,
                )
            finally:
                self._clear_attn_distill_capture()
                self._clear_attn_context_capture()

            if block_idx == future_block_index:
                future_attn = captured_future_attn
                future_context = captured_future_context

            context_t = torch.zeros(
                batch_size, latent_t, device=self.device, dtype=torch.int64
            )
            captured_kv = []
            captured_attn = []
            if block_idx in target_block_set:
                self._set_kv_capture(captured_kv)
                self._set_attn_distill_capture(captured_attn)
            try:
                self.model.generator(
                    noisy_image_or_video=pred_x0,
                    conditional_dict=conditional_dict,
                    timestep=context_t,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * frame_seq_length,
                )
            finally:
                self._clear_kv_capture()
                self._clear_attn_distill_capture()

            if block_idx in target_block_set:
                target_blocks[block_idx] = pred_x0.detach()
                target_kvs[block_idx] = captured_kv
                target_attns[block_idx] = captured_attn

            current_start_frame += latent_t

        if not target_blocks:
            raise RuntimeError("AR rollout distill failed to capture target block KV.")
        if future_attn is None or not future_attn:
            future_attn = target_attns[target_block_indices[-1]]

        del kv_cache, crossattn_cache
        ordered_indices = [idx for idx in target_block_indices if idx in target_blocks and target_kvs.get(idx)]
        if not ordered_indices:
            raise RuntimeError("AR rollout distill failed to capture any non-empty target KV.")
        return (
            [
                target_blocks[idx].permute(0, 2, 1, 3, 4).contiguous()
                for idx in ordered_indices
            ],
            [target_kvs[idx] for idx in ordered_indices],
            [target_attns[idx] for idx in ordered_indices],
            future_attn,
            future_context,
            [idx * latent_t for idx in ordered_indices],
        )

    def pretrain_compressor_kv_distill(self, num_steps=0):
        """Distill compressed per-layer KV toward pooled full-resolution teacher KV."""
        if num_steps <= 0 or not self.heterogeneous_cache_enabled or self.compressor is None:
            return

        if self.is_main_process:
            print("Starting compressor KV distillation pretrain...")

        generator = self._unwrap_generator()
        generator_model = generator.model
        num_layers = getattr(generator_model, "num_layers", len(generator_model.blocks))
        num_heads = getattr(generator_model, "num_heads", 12)
        head_dim = getattr(generator_model, "dim", 1536) // num_heads

        for param in self.model.generator.parameters():
            param.requires_grad_(False)
        for param in self.model.text_encoder.parameters():
            param.requires_grad_(False)
        for name, param in self.compressor.named_parameters():
            param.requires_grad_(False if ".vae." in name or "decoder_" in name else True)

        compressor_train_cfg = getattr(self.config, "compressor_training", None) or OmegaConf.create({})
        optimizer = torch.optim.AdamW(
            [p for p in self.compressor.parameters() if p.requires_grad],
            lr=getattr(compressor_train_cfg, "kv_distill_lr", 1e-4),
        )
        batch_size = getattr(compressor_train_cfg, "kv_distill_batch_size", 1)
        latent_t = getattr(
            compressor_train_cfg,
            "kv_distill_latent_T",
            getattr(self.config, "num_frame_per_block", 4),
        )
        latent_h = getattr(compressor_train_cfg, "kv_distill_latent_H", 60)
        latent_w = getattr(compressor_train_cfg, "kv_distill_latent_W", 104)
        in_ch = getattr(getattr(self.config, "compressor", None), "in_ch", 16)
        density_level = getattr(compressor_train_cfg, "kv_distill_density_level", "mid")
        k_weight = getattr(compressor_train_cfg, "kv_distill_k_weight", 1.0)
        v_weight = getattr(compressor_train_cfg, "kv_distill_v_weight", 1.0)
        cosine_weight = getattr(compressor_train_cfg, "kv_distill_cosine_weight", 0.1)
        attn_output_weight = getattr(compressor_train_cfg, "kv_distill_attn_output_weight", 0.0)
        context_weight = getattr(compressor_train_cfg, "kv_distill_context_weight", attn_output_weight)
        attn_query_tokens = getattr(compressor_train_cfg, "kv_distill_attn_query_tokens", 128)
        attn_max_layers = getattr(compressor_train_cfg, "kv_distill_attn_max_layers", 8)
        attn_context_replacement = getattr(
            compressor_train_cfg, "kv_distill_attn_context_replacement", False
        )
        query_sampling = getattr(compressor_train_cfg, "kv_distill_query_sampling", "uniform")
        query_grid_h = getattr(compressor_train_cfg, "kv_distill_query_grid_h", 4)
        query_grid_w = getattr(compressor_train_cfg, "kv_distill_query_grid_w", 4)
        spatial_ratio = getattr(compressor_train_cfg, "kv_distill_spatial_ratio", 0.5)
        latent_source = getattr(compressor_train_cfg, "kv_distill_latent_source", "random")
        denoise_timestep = getattr(compressor_train_cfg, "kv_distill_denoise_timestep", 750)
        use_real_prompts = getattr(compressor_train_cfg, "kv_distill_use_real_prompts", False)
        prompt_path = getattr(compressor_train_cfg, "kv_distill_prompt_path", getattr(self.config, "data_path", ""))
        prompt_pool = self._load_kv_distill_prompts(prompt_path) if use_real_prompts else []
        rollout_blocks = getattr(compressor_train_cfg, "kv_distill_rollout_blocks", 4)
        random_target = getattr(compressor_train_cfg, "kv_distill_random_target", False)
        num_replaced_blocks = max(1, int(getattr(
            compressor_train_cfg, "kv_distill_num_replaced_blocks", 1
        )))
        multi_target_prob = float(getattr(
            compressor_train_cfg, "kv_distill_multi_target_prob", 1.0
        ))
        replace_min_gap_blocks = max(1, int(getattr(
            compressor_train_cfg, "kv_distill_replace_min_gap_blocks", 1
        )))
        replace_max_gap_blocks = max(replace_min_gap_blocks, int(getattr(
            compressor_train_cfg, "kv_distill_replace_max_gap_blocks", 3
        )))
        target_block_index = getattr(
            compressor_train_cfg,
            "kv_distill_target_block_index",
            max(0, getattr(getattr(self.config, "heterogeneous_cache", None), "Nsink", 8) // max(1, latent_t)),
        )
        nsink = getattr(
            getattr(self.config, "heterogeneous_cache", None),
            "Nsink",
            0,
        )
        nrecent = getattr(
            getattr(self.config, "heterogeneous_cache", None),
            "Nrecent",
            latent_t,
        )
        latent_t_int = max(1, int(latent_t))
        default_future_gap_blocks = max(
            1,
            (int(nrecent) + latent_t_int - 1) // latent_t_int + 1,
        )
        future_gap_blocks = getattr(
            compressor_train_cfg,
            "kv_distill_future_gap_blocks",
            default_future_gap_blocks,
        )
        future_gap_min = getattr(compressor_train_cfg, "kv_distill_future_gap_min", future_gap_blocks)
        future_gap_max = getattr(compressor_train_cfg, "kv_distill_future_gap_max", future_gap_blocks)
        if latent_source == "ar_rollout":
            rollout_blocks = max(2, int(rollout_blocks))
            future_gap_blocks = max(1, int(future_gap_blocks))
            future_gap_min = max(1, int(future_gap_min))
            future_gap_max = max(future_gap_min, int(future_gap_max))
            target_block_index = max(0, int(target_block_index))
            rollout_blocks = max(
                rollout_blocks,
                target_block_index + future_gap_max + 1,
            )
            target_block_index = min(target_block_index, rollout_blocks - 2)
            future_block_index = min(
                rollout_blocks - 1,
                target_block_index + future_gap_blocks,
            )
        else:
            future_gap_blocks = max(1, int(future_gap_blocks))
            future_block_index = None
        global_rank = dist.get_rank() if dist.is_initialized() else 0

        if self.is_main_process:
            print(
                f"  KV distill latent shape: (B={batch_size}, C={in_ch}, "
                f"T={latent_t}, H={latent_h}, W={latent_w}), steps={num_steps}"
            )
            print(
                f"  KV distill source={latent_source}, "
                f"attn_output_weight={attn_output_weight}, "
                f"context_weight={context_weight}, "
                f"attn_context_replacement={attn_context_replacement}, "
                f"query_sampling={query_sampling}, query_grid={query_grid_h}x{query_grid_w}, "
                f"spatial_ratio={spatial_ratio}, "
                f"k_weight={k_weight}, v_weight={v_weight}, "
                f"use_real_prompts={use_real_prompts}, prompt_pool={len(prompt_pool)}, "
                f"rollout_blocks={rollout_blocks}, target_block_index={target_block_index}, "
                f"random_target={random_target}, future_gap_min={future_gap_min}, "
                f"future_gap_max={future_gap_max}, future_gap_blocks={future_gap_blocks}, "
                f"future_block_index={future_block_index}, "
                f"num_replaced_blocks={num_replaced_blocks}, "
                f"multi_target_prob={multi_target_prob}, "
                f"replace_min_gap_blocks={replace_min_gap_blocks}, "
                f"replace_max_gap_blocks={replace_max_gap_blocks}"
            )

        self.compressor.train()
        for step_idx in range(num_steps):
            with torch.no_grad():
                if prompt_pool:
                    text_prompts = [
                        prompt_pool[(step_idx * batch_size + i + global_rank) % len(prompt_pool)]
                        for i in range(batch_size)
                    ]
                else:
                    text_prompts = [""] * batch_size
                conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
                if latent_source == "random":
                    z_c_first = torch.randn(
                        batch_size, in_ch, latent_t, latent_h, latent_w,
                        device=self.device, dtype=self.dtype,
                    )
                    z_btc = z_c_first.permute(0, 2, 1, 3, 4).contiguous()
                    rope_start_frame = 0
                    captured_kv = []
                    captured_attn = []
                    captured_query_attn = captured_attn
                    captured_query_context = None
                    target_blocks = [z_c_first]
                    captured_kvs = [captured_kv]
                    captured_attns = [captured_attn]
                    rope_start_frames = [rope_start_frame]
                elif latent_source == "denoised":
                    noisy_z_btc = torch.randn(
                        batch_size, latent_t, in_ch, latent_h, latent_w,
                        device=self.device, dtype=self.dtype,
                    )
                    denoise_t = torch.full(
                        (batch_size, latent_t),
                        int(denoise_timestep),
                        device=self.device,
                        dtype=torch.int64,
                    )
                    _, pred_x0 = self.model.generator(
                        noisy_image_or_video=noisy_z_btc,
                        conditional_dict=conditional_dict,
                        timestep=denoise_t,
                    )
                    z_btc = pred_x0.detach()
                    z_c_first = z_btc.permute(0, 2, 1, 3, 4).contiguous()
                    del noisy_z_btc, pred_x0
                    rope_start_frame = 0
                    captured_kv = []
                    captured_attn = []
                    captured_query_attn = captured_attn
                    captured_query_context = None
                    target_blocks = [z_c_first]
                    captured_kvs = [captured_kv]
                    captured_attns = [captured_attn]
                    rope_start_frames = [rope_start_frame]
                elif latent_source == "ar_rollout":
                    effective_future_gap_blocks = future_gap_blocks
                    effective_target_block_index = target_block_index
                    effective_target_block_indices = [target_block_index]
                    if random_target:
                        if future_gap_max > future_gap_min:
                            effective_future_gap_blocks = int(torch.randint(
                                low=future_gap_min,
                                high=future_gap_max + 1,
                                size=(1,),
                                device=self.device,
                            ).item())
                        else:
                            effective_future_gap_blocks = future_gap_min
                        max_target_block = max(0, rollout_blocks - effective_future_gap_blocks - 1)
                        sink_blocks = max(0, int(nsink) // max(1, int(latent_t)))
                        min_target_block = min(sink_blocks, max_target_block)
                        if max_target_block > min_target_block:
                            effective_target_block_index = int(torch.randint(
                                low=min_target_block,
                                high=max_target_block + 1,
                                size=(1,),
                                device=self.device,
                            ).item())
                        else:
                            effective_target_block_index = min_target_block
                    primary_target_block = int(effective_target_block_index)
                    future_block_for_targets = min(
                        rollout_blocks - 1,
                        primary_target_block + int(effective_future_gap_blocks),
                    )
                    candidate_targets = list(range(0, max(0, future_block_for_targets)))
                    if random_target and candidate_targets:
                        sink_blocks = max(0, int(nsink) // max(1, int(latent_t)))
                        candidate_targets = [
                            idx for idx in candidate_targets
                            if idx >= min(sink_blocks, max(candidate_targets))
                        ] or candidate_targets
                    use_multi_target = (
                        num_replaced_blocks > 1
                        and candidate_targets
                        and float(torch.rand((), device=self.device).item()) < multi_target_prob
                    )
                    effective_target_block_indices = [primary_target_block]
                    if use_multi_target:
                        valid_extra = [
                            idx for idx in candidate_targets
                            if idx < primary_target_block
                            and primary_target_block - idx >= replace_min_gap_blocks
                            and primary_target_block - idx <= replace_max_gap_blocks
                        ]
                        if len(valid_extra) < num_replaced_blocks - 1:
                            valid_extra = [
                                idx for idx in candidate_targets
                                if idx != primary_target_block
                                and abs(idx - primary_target_block) >= replace_min_gap_blocks
                            ]
                        if len(valid_extra) < num_replaced_blocks - 1:
                            valid_extra = [idx for idx in candidate_targets if idx != primary_target_block]
                        if valid_extra:
                            perm = torch.randperm(len(valid_extra), device=self.device)
                            for perm_idx in perm[: max(0, num_replaced_blocks - 1)].detach().cpu().tolist():
                                effective_target_block_indices.append(valid_extra[int(perm_idx)])
                    effective_target_block_indices = sorted(set(effective_target_block_indices))
                    (
                        target_blocks,
                        captured_kvs,
                        captured_attns,
                        captured_query_attn,
                        captured_query_context,
                        rope_start_frames,
                    ) = self._sample_ar_rollout_distill_block(
                        conditional_dict=conditional_dict,
                        batch_size=batch_size,
                        in_ch=in_ch,
                        latent_t=latent_t,
                        latent_h=latent_h,
                        latent_w=latent_w,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        denoise_timestep=int(denoise_timestep),
                        rollout_blocks=rollout_blocks,
                        target_block_index=effective_target_block_index,
                        future_gap_blocks=effective_future_gap_blocks,
                        generator_model=generator_model,
                        target_block_indices=effective_target_block_indices,
                    )
                    z_c_first = torch.cat(target_blocks, dim=0)
                    z_btc = None
                else:
                    raise ValueError(
                        f"Unsupported kv_distill_latent_source={latent_source!r}; "
                        "expected 'random', 'denoised', or 'ar_rollout'."
                    )

                if latent_source != "ar_rollout":
                    timestep = torch.zeros(
                        batch_size, latent_t, device=self.device, dtype=torch.int64
                    )
                    self._set_kv_capture(captured_kv)
                    if attn_output_weight > 0:
                        self._set_attn_distill_capture(captured_attn)
                    try:
                        self.model.generator(
                            noisy_image_or_video=z_btc,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                        )
                    finally:
                        self._clear_kv_capture()
                        self._clear_attn_distill_capture()
                    target_blocks = [z_c_first]
                    captured_kvs = [captured_kv]
                    captured_attns = [captured_attn]
                    rope_start_frames = [rope_start_frame]

            compressed_tokens, _ = self.compressor(z_c_first, density_level)
            student_kv = self.compressor.project_to_kv(
                compressed_tokens,
                num_layers=num_layers,
                num_heads=num_heads,
            )
            target_grid = self.compressor.compressed_grid_shape(
                density_level, (latent_t, latent_h, latent_w)
            )
            num_targets = max(1, len(captured_kvs))
            student_kv_by_target = []
            student_roped_kv_by_target = []
            for target_idx in range(num_targets):
                start = target_idx * batch_size
                end = start + batch_size
                per_target_kv = [
                    torch.stack([kv[0][start:end], kv[1][start:end]], dim=0)
                    for kv in student_kv
                ]
                student_kv_by_target.append(per_target_kv)
                target_rope_start = int(rope_start_frames[target_idx])
                student_roped_kv_by_target.append([
                    torch.stack([
                        apply_temporal_rope_to_unrotated(
                            kv[0],
                            freqs=generator_model.freqs,
                            start_frame=target_rope_start,
                            grid_shape=target_grid,
                            temporal_stride=2,
                        ),
                        kv[1],
                    ], dim=0)
                    for kv in per_target_kv
                ])

            full_grid = (latent_t, latent_h // 2, latent_w // 2)
            loss = torch.tensor(0.0, device=self.device)
            used_layers = min(
                min(len(kv_items) for kv_items in captured_kvs),
                num_layers,
            )
            attn_layer_ids = set(self._select_distill_layers(used_layers, attn_max_layers))
            context_replacement_used = 0
            context_replacement_fallback = 0
            attn_loss_skipped = 0
            for layer_idx in range(used_layers):
                layer_loss = torch.tensor(0.0, device=self.device)
                valid_targets_for_layer = 0
                for target_idx in range(num_targets):
                    if layer_idx >= len(captured_kvs[target_idx]):
                        continue
                    teacher_k, teacher_v = captured_kvs[target_idx][layer_idx]
                    target_k = self._pool_teacher_kv_to_grid(teacher_k, full_grid, target_grid)
                    target_v = self._pool_teacher_kv_to_grid(teacher_v, full_grid, target_grid)
                    pred_k = student_kv_by_target[target_idx][layer_idx][0]
                    pred_v = student_kv_by_target[target_idx][layer_idx][1]
                    target_loss = k_weight * F.mse_loss(pred_k.float(), target_k.float())
                    target_loss = target_loss + v_weight * F.mse_loss(pred_v.float(), target_v.float())
                    target_loss = target_loss + cosine_weight * (
                        1 - F.cosine_similarity(pred_k.float(), target_k.float(), dim=-1).mean()
                        + 1 - F.cosine_similarity(pred_v.float(), target_v.float(), dim=-1).mean()
                    )
                    layer_loss = layer_loss + target_loss
                    valid_targets_for_layer += 1
                layer_loss = layer_loss / max(1, valid_targets_for_layer)

                if (
                    (attn_output_weight > 0 or context_weight > 0)
                    and layer_idx in attn_layer_ids
                    and captured_attns
                    and layer_idx < len(captured_attns[0])
                    and layer_idx < len(captured_query_attn)
                ):
                    teacher_q = captured_query_attn[layer_idx][0]
                    _, teacher_roped_k, teacher_attn_v = captured_attns[0][layer_idx]
                    attn_loss = None
                    attn_loss_weight = attn_output_weight
                    attempted_context_replacement = False
                    if (
                        context_weight > 0
                        and attn_context_replacement
                        and captured_query_context is not None
                        and layer_idx < len(captured_query_context)
                    ):
                        attempted_context_replacement = True
                        replacements = []
                        target_num_tokens = latent_t * (latent_h // 2) * (latent_w // 2)
                        for target_idx in range(num_targets):
                            if layer_idx >= len(student_roped_kv_by_target[target_idx]):
                                continue
                            replacements.append({
                                "student_k": student_roped_kv_by_target[target_idx][layer_idx][0],
                                "student_v": student_roped_kv_by_target[target_idx][layer_idx][1],
                                "target_start_token": int(rope_start_frames[target_idx]) * (latent_h // 2) * (latent_w // 2),
                                "target_num_tokens": target_num_tokens,
                            })
                        attn_loss = self._multi_context_replacement_attn_loss(
                            query=teacher_q,
                            teacher_context=captured_query_context[layer_idx],
                            replacements=replacements,
                            num_query_tokens=attn_query_tokens,
                            query_grid_shape=full_grid,
                            query_sampling=query_sampling,
                            query_grid_h=query_grid_h,
                            query_grid_w=query_grid_w,
                            spatial_ratio=spatial_ratio,
                        )
                        if attn_loss is not None:
                            attn_loss_weight = context_weight
                            context_replacement_used += 1
                    if attn_loss is None and attn_output_weight > 0:
                        attn_loss = self._attention_output_distill_loss(
                            teacher_q,
                            teacher_roped_k,
                            teacher_attn_v,
                            student_roped_kv_by_target[0][layer_idx][0],
                            student_roped_kv_by_target[0][layer_idx][1],
                            num_query_tokens=attn_query_tokens,
                            query_grid_shape=full_grid,
                            query_sampling=query_sampling,
                            query_grid_h=query_grid_h,
                            query_grid_w=query_grid_w,
                            spatial_ratio=spatial_ratio,
                        )
                        if attempted_context_replacement:
                            context_replacement_fallback += 1
                    if attn_loss is not None:
                        layer_loss = layer_loss + attn_loss_weight * attn_loss
                    else:
                        attn_loss_skipped += 1
                loss = loss + layer_loss
            loss = loss / max(1, used_layers)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.compressor.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            if step_idx % 10 == 0 and self.is_main_process:
                print(
                    f"  KV distill step {step_idx}/{num_steps}, loss={loss.item():.6f}, "
                    f"targets={num_targets}, target_starts={rope_start_frames}, "
                    f"context_used={context_replacement_used}, "
                    f"context_fallback={context_replacement_fallback}, "
                    f"attn_skipped={attn_loss_skipped}"
                )

        for param in self.model.generator.parameters():
            param.requires_grad_(True)
        for param in self.model.text_encoder.parameters():
            param.requires_grad_(True)
        for param in self.compressor.parameters():
            param.requires_grad_(True)
        self.compressor.to(dtype=self.dtype)
        gc.collect()
        torch.cuda.empty_cache()

    def _compute_reconstruction_loss(self, pred_image):
        """
        Compute reconstruction auxiliary loss for the compressor.
        Uses decoders to reconstruct from compressed tokens and compares with original.
        pred_image: (B, T, C, H, W) denoised prediction from generator
        """
        if not self.heterogeneous_cache_enabled or self.compressor is None:
            return torch.tensor(0.0, device=self.device)

        z_input = pred_image.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        target_shape = z_input.shape[2:]

        total_loss = torch.tensor(0.0, device=self.device)
        for level, decoder in [("high", self.compressor.decoder_high),
                               ("mid", self.compressor.decoder_mid),
                               ("low", self.compressor.decoder_low)]:
            if decoder is None:
                continue
            compressed, _ = self.compressor(z_input, level)
            reconstructed = decoder(compressed, target_shape)
            total_loss = total_loss + F.mse_loss(reconstructed, z_input)

        return total_loss

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        # Save compressor state dict for end-to-end training
        if self.compressor is not None:
            state_dict["compressor"] = self.compressor.state_dict()

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            # Zero compressor gradients before forward (end-to-end path)
            if self.compressor_optimizer is not None:
                self.compressor_optimizer.zero_grad(set_to_none=True)

            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )

            # DMD loss + compressor reconstruction auxiliary loss
            total_loss = generator_loss

            # Add compressor recon loss if heterogeneous cache enabled
            if self.heterogeneous_cache_enabled and self.compressor is not None:
                # Get pred_image from generator_log_dict for recon loss
                if "pred_image" in generator_log_dict:
                    if self.step >= self.recon_warmup_steps:
                        lambda_r = self.lambda_recon
                    else:
                        # Linear warmup
                        lambda_r = self.lambda_recon * (self.step / max(1, self.recon_warmup_steps))
                    if lambda_r > 0:
                        recon_loss = self._compute_reconstruction_loss(generator_log_dict["pred_image"])
                        total_loss = total_loss + lambda_r * recon_loss
                        generator_log_dict["recon_loss"] = recon_loss
                    generator_log_dict["effective_lambda"] = lambda_r

            total_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            # Step compressor optimizer (end-to-end gradient from DMD loss
            # flows through attention → compressed KV → compressor parameters)
            if self.compressor_optimizer is not None:
                compressor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.compressor.parameters() if p.requires_grad],
                    self.compressor_grad_clip,
                )
                self.compressor_optimizer.step()
                generator_log_dict["compressor_grad_norm"] = compressor_grad_norm
                if self.is_main_process:
                    print(f"compressor_grad_norm: {compressor_grad_norm.item():.6f}")

            generator_log_dict.update({"generator_loss": total_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    @torch.no_grad()
    def validate(self):
        """Run inference on a few prompts and save videos for monitoring."""
        from pipeline import CausalInferencePipeline
        import imageio

        # Non-rank-0 processes wait at barrier until rank 0 finishes
        if not self.is_main_process:
            dist.barrier()
            return

        print(f"[Validation] Starting at step {self.step}...")

        # Load validation prompts
        prompt_path = getattr(self.config, "prompt_path", "prompts/validation_60s.txt")
        with open(prompt_path, "r") as f:
            all_prompts = [line.strip() for line in f.readlines() if line.strip()]
        num_val_prompts = min(getattr(self.config, "num_val_prompts", 2), len(all_prompts))
        val_prompts = all_prompts[:num_val_prompts]

        # Video length: 60s at 16fps ≈ 960 pixel frames
        # latent_T = (960-1)/4 + 1 ≈ 240 → 957 frames = 59.8s
        # num_blocks = 240 / num_frame_per_block(3) = 80
        val_num_latent_frames = getattr(self.config, "val_num_latent_frames", 240)

        # Load checkpoint
        checkpoint_path = os.path.join(
            self.output_path, f"checkpoint_model_{self.step:06d}", "model.pt")
        if not os.path.exists(checkpoint_path):
            print(f"[Validation] Checkpoint not found: {checkpoint_path}, skipping.")
            dist.barrier()
            return

        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Create inference pipeline (fresh non-FSDP models)
        pipeline = CausalInferencePipeline(args=self.config, device=self.device)

        # Load generator weights (prefer EMA if available)
        use_ema = "generator_ema" in state_dict and self.generator_ema is not None
        key = "generator_ema" if use_ema else "generator"
        print(f"[Validation] Loading {key} weights...")
        # Strip FSDP/checkpoint prefixes (EMA state_dict may keep them)
        rename = lambda n: (
            n.replace("_fsdp_wrapped_module.", "")
             .replace("_checkpoint_wrapped_module.", "")
             .replace("_orig_mod.", "")
        )
        clean_sd = {rename(k): v for k, v in state_dict[key].items()}
        pipeline.generator.load_state_dict(clean_sd, strict=True)

        if "compressor" in state_dict and getattr(
                getattr(self.config, "heterogeneous_cache", None), "enabled", False):
            print("[Validation] Loading compressor weights...")
            pipeline.load_compressor_state_dict(
                state_dict["compressor"],
                device=self.device,
                dtype=torch.bfloat16,
                strict=True,
            )
        del state_dict

        pipeline = pipeline.to(dtype=torch.bfloat16)
        pipeline.generator.to(device=self.device)
        pipeline.text_encoder.to(device=self.device)
        pipeline.vae.to(device=self.device)

        # Output directory
        val_dir = os.path.join(self.output_path, f"validation_step_{self.step:06d}")
        os.makedirs(val_dir, exist_ok=True)

        for i, prompt in enumerate(val_prompts):
            noise = torch.randn(
                [1, val_num_latent_frames, 16, 60, 104],
                device=self.device,
                dtype=torch.bfloat16,
            )
            video, _ = pipeline.inference(
                noise=noise,
                text_prompts=[prompt],
                return_latents=True,
            )
            current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
            video_uint8 = (255.0 * current_video).clamp(0, 255).to(torch.uint8)
            safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in prompt[:60])
            output_path = os.path.join(val_dir, f"{i}_{safe_name}.mp4")
            writer = imageio.get_writer(output_path, fps=16, codec="libx264", quality=5)
            for frame in video_uint8[0].numpy():
                writer.append_data(frame)
            writer.close()
            print(f"[Validation] Saved {output_path}")

            # Clear KV cache between prompts
            if pipeline.heterogeneous_cache_enabled and pipeline.het_kv_cache is not None:
                pipeline.het_kv_cache.reset()
                pipeline.kv_cache1 = [pipeline.het_kv_cache.get_layer_cache(i)
                                      for i in range(pipeline.num_transformer_blocks)]
                pipeline.crossattn_cache = [pipeline.het_kv_cache.get_crossattn_cache(i)
                                            for i in range(pipeline.num_transformer_blocks)]
            else:
                pipeline.kv_cache1 = None
                pipeline.crossattn_cache = None

        # Free memory
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[Validation] Done. Videos saved to {val_dir}")

        dist.barrier()

    def train(self):
        start_step = self.step

        # Phase 2: Pretrain compressor HR heads (if heterogeneous cache enabled)
        if self.heterogeneous_cache_enabled and self.step == 0:
            compressor_cfg = getattr(self.config, "compressor_training", None)
            pretrain_steps = getattr(compressor_cfg, "pretrain_epochs", 5) * 20  # approximate
            self.pretrain_compressor(num_steps=pretrain_steps)
            kv_distill_steps = getattr(compressor_cfg, "kv_distill_steps", 0)
            self.pretrain_compressor_kv_distill(num_steps=kv_distill_steps)

        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            max_train_steps = getattr(self.config, "max_train_steps", None)
            reached_max_train_steps = max_train_steps is not None and self.step >= max_train_steps

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            should_save = (
                (not self.config.no_save)
                and (self.step - start_step) > 0
                and (self.step % self.config.log_iters == 0 or reached_max_train_steps)
            )
            if should_save:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()
                if not reached_max_train_steps and getattr(self.config, "validate_on_save", True):
                    self.validate()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )
                    if "recon_loss" in generator_log_dict:
                        wandb_loss_dict["recon_loss"] = generator_log_dict["recon_loss"].mean().item()
                        wandb_loss_dict["effective_lambda"] = generator_log_dict["effective_lambda"]
                    if "compressor_grad_norm" in generator_log_dict:
                        wandb_loss_dict["compressor_grad_norm"] = generator_log_dict["compressor_grad_norm"].mean().item()

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            if reached_max_train_steps:
                if self.is_main_process:
                    print(f"[Debug] Reached max_train_steps={max_train_steps}, exiting.")
                break
