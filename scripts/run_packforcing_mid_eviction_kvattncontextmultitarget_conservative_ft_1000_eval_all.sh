#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/inspire/qb-ilm/project/exploration-topic/wangqiqi-CZXS25210124/Self-Forcing-mynew"
RESULT_ROOT="/inspire/qb-ilm/project/exploration-topic/wangqiqi-CZXS25210124/Self-Forcing-mynew_result/mideviction_kvattncontextmultitarget_conservative_ft_1000"
TRAIN_LOGDIR="${RESULT_ROOT}/train"
LOG_DIR="${RESULT_ROOT}/logs"
CONFIG_PATH="configs/packforcing_mid_eviction_kvattncontextmultitarget_conservative_ft_1000.yaml"

: "${COMPRESSOR_CKPT:?Set COMPRESSOR_CKPT=/path/to/V2/checkpoint_model_xxxxxx/model.pt}"

cd "${REPO_ROOT}"

mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

torchrun --standalone --nproc_per_node=8 train.py \
    --config_path "${CONFIG_PATH}" \
    --logdir "${TRAIN_LOGDIR}" \
    --disable-wandb \
    --no_visualize 2>&1 | tee "${LOG_DIR}/run.log"

for step in $(seq 250 250 1000); do
    step_padded="$(printf "%06d" "${step}")"
    checkpoint_path="${TRAIN_LOGDIR}/checkpoint_model_${step_padded}/model.pt"
    output_folder="${RESULT_ROOT}/step_${step_padded}"

    python inference.py \
        --config_path "${CONFIG_PATH}" \
        --checkpoint_path "${checkpoint_path}" \
        --data_path "${REPO_ROOT}/prompts/validation_60s.txt" \
        --output_folder "${output_folder}" \
        --num_output_frames 240 2>&1 | tee "${LOG_DIR}/infer_step_${step_padded}.log"
done
