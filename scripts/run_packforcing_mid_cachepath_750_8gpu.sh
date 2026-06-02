#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/inspire/ssd/project/video-generation/public/wangqiqi/Self-Forcing-mynew"

cd "${REPO_ROOT}"

torchrun --standalone --nproc_per_node=8 train.py \
    --config_path configs/packforcing_mid_cachepath.yaml \
    --logdir logs/debug/train_packforcing_mid_cachepath_750 \
    --disable-wandb \
    --no_visualize



python /inspire/ssd/project/video-generation/public/wangqiqi/occupy.py -m 95
