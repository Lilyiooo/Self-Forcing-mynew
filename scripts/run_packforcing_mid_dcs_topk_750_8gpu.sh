#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/inspire/ssd/project/video-generation/public/wangqiqi/Self-Forcing-mynew"

cd "${REPO_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

torchrun --standalone --nproc_per_node=8 train.py \
    --config_path configs/packforcing_mid_dcs_topk_750.yaml \
    --logdir logs/debug/train_packforcing_mid_dcs_topk_750 \
    --disable-wandb \
    --no_visualize

python /inspire/ssd/project/video-generation/public/wangqiqi/occupy.py -m 95
