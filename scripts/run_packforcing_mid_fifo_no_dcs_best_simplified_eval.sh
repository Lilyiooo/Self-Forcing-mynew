#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/inspire/qb-ilm/project/exploration-topic/wangqiqi-CZXS25210124/Self-Forcing-mynew"
RESULT_ROOT="/inspire/qb-ilm/project/exploration-topic/wangqiqi-CZXS25210124/Self-Forcing-mynew_result/fifo_no_dcs_best_simplified"
CONFIG_PATH="configs/packforcing_mid_eviction_kvattncontextmixedhorizon_2500.yaml"

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH=/path/to/best_simplified_compressor/checkpoint_model_xxxxxx/model.pt}"

cd "${REPO_ROOT}"

mkdir -p "${RESULT_ROOT}"

python inference.py \
    --config_path "${CONFIG_PATH}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --data_path "${REPO_ROOT}/prompts/validation_60s.txt" \
    --output_folder "${RESULT_ROOT}" \
    --num_output_frames 240
