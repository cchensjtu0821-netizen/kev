#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export GPUS="${GPUS:-4}"
export MASTER_PORT="${MASTER_PORT:-32036}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export GRADIENT_ACC="${GRADIENT_ACC:-1}"
export LORA_R="${LORA_R:-128}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../../runs/camera-kev-04b-lora-seka}"

exec bash "${SCRIPT_DIR}/train_camera_kev_lora.sh" "$@"
