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
                k_out = self.compressor.kv_k_proj(compressed)
                v_out = self.compressor.kv_v_proj(compressed)
                # Regularize: output should have unit variance per feature dim
                total_loss = total_loss + 0.01 * (
                    (k_out.var(dim=-1).mean() - 1.0).pow(2)
                    + (v_out.var(dim=-1).mean() - 1.0).pow(2)
                )

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
                if not reached_max_train_steps:
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
