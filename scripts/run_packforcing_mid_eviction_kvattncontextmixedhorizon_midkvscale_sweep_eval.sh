#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/inspire/qb-ilm/project/exploration-topic/wangqiqi-CZXS25210124/Self-Forcing-mynew"
RESULT_ROOT="/inspire/qb-ilm/project/exploration-topic/wangqiqi-CZXS25210124/Self-Forcing-mynew_result/mideviction_kvattncontextmixedhorizon_midkvscale_sweep"
LOG_DIR="${RESULT_ROOT}/logs"
CONFIG_PATH="configs/packforcing_mid_eviction_kvattncontextmixedhorizon_2500.yaml"

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH=/path/to/mixedhorizon/checkpoint_model_xxxxxx/model.pt}"

cd "${REPO_ROOT}"

mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

declare -a SWEEP_PAIRS=(
    "0.50 0.50"
    "0.50 0.75"
    "0.50 1.00"
    "0.75 0.75"
    "0.75 1.00"
)

for pair in "${SWEEP_PAIRS[@]}"; do
    read -r mid_k_scale mid_v_scale <<< "${pair}"
    tag="k${mid_k_scale}_v${mid_v_scale}"
    output_folder="${RESULT_ROOT}/${tag}"

    python inference.py \
        --config_path "${CONFIG_PATH}" \
        --checkpoint_path "${CHECKPOINT_PATH}" \
        --data_path "${REPO_ROOT}/prompts/validation_60s.txt" \
        --output_folder "${output_folder}" \
        --num_output_frames 240 \
        --mid_k_scale "${mid_k_scale}" \
        --mid_v_scale "${mid_v_scale}" 2>&1 | tee "${LOG_DIR}/infer_${tag}.log"
done
