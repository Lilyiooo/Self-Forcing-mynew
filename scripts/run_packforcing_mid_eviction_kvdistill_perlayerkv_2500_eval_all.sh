#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/inspire/ssd/project/video-generation/public/wangqiqi/Self-Forcing-mynew"
RESULT_ROOT="/inspire/qb-ilm/project/video-generation/public/wangqiqi/Self-Forcing-packforcing/mideviction_kvdistill_perlayerkv_2500"
TRAIN_LOGDIR="${RESULT_ROOT}/train"
CONFIG_PATH="configs/packforcing_mid_eviction_kvdistill_2500.yaml"

cd "${REPO_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

torchrun --standalone --nproc_per_node=8 train.py \
    --config_path "${CONFIG_PATH}" \
    --logdir "${TRAIN_LOGDIR}" \
    --disable-wandb \
    --no_visualize

mkdir -p "${RESULT_ROOT}"

for step in $(seq 250 250 2500); do
    step_padded="$(printf "%06d" "${step}")"
    checkpoint_path="${TRAIN_LOGDIR}/checkpoint_model_${step_padded}/model.pt"
    output_folder="${RESULT_ROOT}/step_${step_padded}"

    python inference.py \
        --config_path "${CONFIG_PATH}" \
        --checkpoint_path "${checkpoint_path}" \
        --data_path prompts/validation_60s.txt \
        --output_folder "${output_folder}" \
        --num_output_frames 240
done

python /inspire/ssd/project/video-generation/public/wangqiqi/occupy.py -m 95