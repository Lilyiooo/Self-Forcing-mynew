"""
experiment_A_density_vis.py — Experiment A: Information Density Distribution Visualization

Generates 30s videos for 3 types of prompts (mixed/action/static),
records per-block density scores, and saves JSON results for plotting.

Usage:
    torchrun --nproc_per_node=1 experiment_A_density_vis.py \
        --config_path configs/heterogeneous_cache.yaml \
        --checkpoint_path outputs/xxx/checkpoint_model_XXXXXX/model.pt \
        --output_dir results/density_exp

    # Then plot:
    python plot_density.py --results_dir results/density_exp --output_dir figures/
"""

import argparse
import json
import os
import gc
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from einops import rearrange
import imageio

from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from demo_utils.memory import gpu, get_cuda_free_memory_gb

# ── Prompt definitions (3 categories × 2-3 prompts) ──────────────────

PROMPTS = {
    # A: Mixed (dynamic + static) — should show clear peaks/valleys
    "mixed_run_1": "A person runs at full speed across a field, then stops and stands still watching the sunset.",
    "mixed_run_2": "A car accelerates through a winding mountain road, then parks and the driver steps out to look at the view.",
    "mixed_run_3": "A dancer performs an energetic routine, then takes a bow and holds a still pose.",

    # B: Sustained high-dynamic — density should stay high throughout
    "action_run_1": "A skateboarder performs continuous tricks and jumps in a skate park.",
    "action_run_2": "Fast-paced street traffic with cars and motorcycles constantly moving.",

    # C: Sustained low-dynamic — density should stay low throughout
    "static_run_1": "A still life of flowers on a table in soft morning light, gentle breeze makes petals barely move.",
    "static_run_2": "A mountain lake at dawn, completely calm water reflecting the sky.",
}

# ── Video parameters ──────────────────────────────────────────────────
# 30s at 16fps = 480 pixel frames
# latent_T = (480-1)/4 + 1 ≈ 120 → (120-1)*4+1 = 477 frames = 29.8s
# blocks = 120 / num_frame_per_block(3) = 40
NUM_LATENT_FRAMES = 120
FPS = 16


def main():
    parser = argparse.ArgumentParser(description="Experiment A: Density Visualization")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/density_exp")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Distributed setup
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda")
        local_rank = 0

    is_main = (local_rank == 0)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed + local_rank)
    torch.set_grad_enabled(False)

    # Load config
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    # Ensure heterogeneous cache is enabled (required for density estimation)
    if not getattr(config, "heterogeneous_cache", None) or not config.heterogeneous_cache.enabled:
        print("[ERROR] This experiment requires heterogeneous_cache.enabled=true in config.")
        return

    # Initialize pipeline
    pipeline = CausalInferencePipeline(args=config, device=device)

    # Load checkpoint
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    key = "generator_ema" if "generator_ema" in state_dict else "generator"
    print(f"[ExperimentA] Loading {key} weights...")
    rename = lambda n: (
        n.replace("_fsdp_wrapped_module.", "")
         .replace("_checkpoint_wrapped_module.", "")
         .replace("_orig_mod.", "")
    )
    clean_sd = {rename(k): v for k, v in state_dict[key].items()}
    pipeline.generator.load_state_dict(clean_sd, strict=True)
    del state_dict

    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device=device)
    pipeline.text_encoder.to(device=device)
    pipeline.vae.to(device=device)

    print(f"[ExperimentA] Pipeline ready. Running {len(PROMPTS)} prompts, "
          f"{NUM_LATENT_FRAMES} latent frames (~30s @ {FPS}fps)")

    # ── Run each prompt ───────────────────────────────────────────────
    for prompt_id, prompt_text in PROMPTS.items():
        if not is_main:
            if dist.is_initialized():
                dist.barrier()
            continue

        print(f"\n{'='*60}")
        print(f"[{prompt_id}] {prompt_text[:80]}...")
        print(f"{'='*60}")

        # Set up density logging
        pipeline._density_log = []

        # Generate video
        noise = torch.randn(
            [1, NUM_LATENT_FRAMES, 16, 60, 104],
            device=device,
            dtype=torch.bfloat16,
        )

        try:
            video, latents = pipeline.inference(
                noise=noise,
                text_prompts=[prompt_text],
                return_latents=True,
            )
        except Exception as e:
            print(f"[{prompt_id}] Inference failed: {e}")
            pipeline._density_log = None
            # Reset caches
            pipeline.kv_cache1 = None
            pipeline.crossattn_cache = None
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # Save density log
        json_path = os.path.join(args.output_dir, f"{prompt_id}_density.json")
        with open(json_path, "w") as f:
            json.dump(pipeline._density_log, f, indent=2)
        print(f"[{prompt_id}] Saved {len(pipeline._density_log)} blocks → {json_path}")

        # Save video
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        video_uint8 = (255.0 * current_video).clamp(0, 255).to(torch.uint8)
        video_path = os.path.join(args.output_dir, f"{prompt_id}_video.mp4")
        writer = imageio.get_writer(video_path, fps=FPS, codec="libx264", quality=5)
        for frame in video_uint8[0].numpy():
            writer.append_data(frame)
        writer.close()
        print(f"[{prompt_id}] Saved video → {video_path}")

        # Print quick stats
        if pipeline._density_log:
            from collections import Counter
            tiers = [d["tier"] for d in pipeline._density_log]
            cnt = Counter(tiers)
            total = len(tiers)
            import numpy as np
            density_arr = [d["density_score"] for d in pipeline._density_log]
            print(f"  Total blocks: {total}")
            print(f"  High (8×):  {cnt.get('high',0):3d} ({100*cnt.get('high',0)/total:.1f}%)")
            print(f"  Mid  (32×): {cnt.get('mid',0):3d} ({100*cnt.get('mid',0)/total:.1f}%)")
            print(f"  Low  (128×):{cnt.get('low',0):3d} ({100*cnt.get('low',0)/total:.1f}%)")
            print(f"  Density — mean:{np.mean(density_arr):.3f}  std:{np.std(density_arr):.3f}  "
                  f"min:{np.min(density_arr):.3f}  max:{np.max(density_arr):.3f}")

        # Clean up for next prompt
        pipeline._density_log = None
        if pipeline.heterogeneous_cache_enabled and pipeline.het_kv_cache is not None:
            pipeline.het_kv_cache.reset()
            pipeline.kv_cache1 = [pipeline.het_kv_cache.get_layer_cache(i)
                                  for i in range(pipeline.num_transformer_blocks)]
            pipeline.crossattn_cache = [pipeline.het_kv_cache.get_crossattn_cache(i)
                                        for i in range(pipeline.num_transformer_blocks)]
        else:
            pipeline.kv_cache1 = None
            pipeline.crossattn_cache = None
        del noise, video, latents, current_video, video_uint8
        gc.collect()
        torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.barrier()

    print(f"\n[ExperimentA] Done. Results in {args.output_dir}/")
    print(f"[ExperimentA] Next step: python plot_density.py --results_dir {args.output_dir} --output_dir figures/")


if __name__ == "__main__":
    main()
