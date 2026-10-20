#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export GPUS="${GPUS:-8}"
export MASTER_PORT="${MASTER_PORT:-32026}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
export GRADIENT_ACC="${GRADIENT_ACC:-1}"
export LORA_R="${LORA_R:-64}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../../runs/camera-kev-04b-lora-goutu}"

exec bash "${SCRIPT_DIR}/train_camera_kev_lora.sh" "$@"
