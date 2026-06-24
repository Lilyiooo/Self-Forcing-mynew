#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/inspire/qb-ilm/project/exploration-topic/wangqiqi-CZXS25210124/Self-Forcing-mynew"
RESULT_ROOT="/inspire/qb-ilm/project/exploration-topic/wangqiqi-CZXS25210124/Self-Forcing-mynew_result/dcs_queryaffinity_top4"
CONFIG_PATH="configs/packforcing_mid_dcs_queryaffinity_top4_eval.yaml"

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH=/path/to/best_simplified_compressor/checkpoint_model_xxxxxx/model.pt}"

cd "${REPO_ROOT}"

mkdir -p "${RESULT_ROOT}"

python inference.py \
    --config_path "${CONFIG_PATH}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --data_path "${REPO_ROOT}/prompts/validation_60s.txt" \
    --output_folder "${RESULT_ROOT}" \
    --num_output_frames 240
