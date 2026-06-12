#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/inspire/ssd/project/video-generation/public/wangqiqi/Self-Forcing-mynew"

cd "${REPO_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

torchrun --standalone --nproc_per_node=8 train.py \
    --config_path configs/packforcing_mid_eviction_750.yaml \
    --logdir logs/debug/train_packforcing_mid_eviction_perlayerkv_750 \
    --disable-wandb \
    --no_visualize

python inference.py \
    --config_path configs/packforcing_mid_eviction_750.yaml \
    --checkpoint_path logs/debug/train_packforcing_mid_eviction_perlayerkv_750/checkpoint_model_000750/model.pt \
    --data_path prompts/validation_60s.txt \
    --output_folder logs/debug/infer_packforcing_mid_eviction_perlayerkv_750 \
    --num_output_frames 240

python /inspire/ssd/project/video-generation/public/wangqiqi/occupy.py -m 95
