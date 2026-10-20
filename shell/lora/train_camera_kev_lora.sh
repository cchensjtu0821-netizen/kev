#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

# Dense CameraAgent 0.4B, attention LoRA, and a new Kev pointer head.
# CameraModel's MoE routing and generative packing settings do not apply here.
camera_start=base
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/srv/workspace/Kirin_AI_Workspace/AIC_I/l00883213/camera_agent/hisi_dinov3_llm022/released_model/vlm_0p4_384_full_vb46k_general_domain_v3_0808}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/runs/camera-kev-04b-lora}"
TRAIN_META="${TRAIN_META:-${META_PATH:-}}"
EPOCHS="${EPOCHS:-${NUM_TRAIN_EPOCHS:-3}}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
GRADIENT_ACC="${GRADIENT_ACC:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
HEAD_LR="${HEAD_LR:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
MASTER_PORT="${MASTER_PORT:-32026}"
LORA_R="${LORA_R:-64}"
if ! [[ "${LORA_R}" =~ ^[1-9][0-9]*$ ]]; then
  echo "LORA_R must be a positive integer, got: ${LORA_R}" >&2
  exit 2
fi

source "${PROJECT_DIR}/shell/camera_train_common.sh" "$@"
