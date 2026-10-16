#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# Point these at the machine that holds the CameraAgent code, weights and images.
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/srv/workspace/Kirin_AI_Workspace/AIC_I/l00883213/camera_agent/hisi_dinov3_llm022/released_model/vlm_0p4_384_full_vb46k_general_domain_v3_0808}"
TRAIN_META="${TRAIN_META:-}"
VAL_META="${VAL_META:-}"
TRAIN_DATA="${TRAIN_DATA:-}"
VAL_DATA="${VAL_DATA:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/runs/camera-kev-04b}"
CAMERA_ENV_BIN="${CAMERA_ENV_BIN:-/srv/workspace/Kirin_AI_Workspace/AIC_I/c00932551/condaenv/MoE_CameraModel/bin}"
CAMERA_CODE_DIR="${CAMERA_CODE_DIR:-}"
TORCHRUN="${TORCHRUN:-${CAMERA_ENV_BIN}/torchrun}"

export PATH="${CAMERA_ENV_BIN}:${PATH}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${CAMERA_CODE_DIR}" ]]; then
  if [[ ! -d "${CAMERA_CODE_DIR}" ]]; then
    echo "CameraAgent source directory not found: ${CAMERA_CODE_DIR}" >&2
    exit 2
  fi
  export PYTHONPATH="${CAMERA_CODE_DIR}:${PYTHONPATH}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-32006}"
EPOCHS="${EPOCHS:-3}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
GRADIENT_ACC="${GRADIENT_ACC:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
MAX_LENGTH="${MAX_LENGTH:-2560}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"
HEAD_DIM="${HEAD_DIM:-256}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
HEAD_LR="${HEAD_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_STEPS="${SAVE_STEPS:-2500}"
MAX_STEPS="${MAX_STEPS:-0}"
SEED="${SEED:-17}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-false}"
FREEZE_LLM="${FREEZE_LLM:-false}"
FREEZE_MLP="${FREEZE_MLP:-false}"

freeze_args=()
add_freeze_flag() {
  local name="$1" value="${!1}" flag="$2"
  case "${value,,}" in
    1|true) freeze_args+=("${flag}") ;;
    0|false) ;;
    *) echo "${name} must be true/false or 1/0, got: ${value}" >&2; exit 2 ;;
  esac
}
add_freeze_flag FREEZE_BACKBONE --freeze-backbone
add_freeze_flag FREEZE_LLM --freeze-llm
add_freeze_flag FREEZE_MLP --freeze-mlp

train_args=()
val_args=()
if [[ -n "${TRAIN_META}" && -z "${TRAIN_DATA}" ]]; then
  train_args=(--train-meta "${TRAIN_META}")
elif [[ -n "${TRAIN_DATA}" && -z "${TRAIN_META}" ]]; then
  train_args=(--train-data "${TRAIN_DATA}")
else
  echo "Set exactly one of TRAIN_META or TRAIN_DATA" >&2
  exit 2
fi
if [[ -z "${VAL_META}" && -z "${VAL_DATA}" ]]; then
  val_args=()
elif [[ -n "${VAL_META}" && -z "${VAL_DATA}" ]]; then
  val_args=(--val-meta "${VAL_META}")
elif [[ -n "${VAL_DATA}" && -z "${VAL_META}" ]]; then
  val_args=(--val-data "${VAL_DATA}")
else
  echo "Set at most one of VAL_META or VAL_DATA" >&2
  exit 2
fi

if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "CameraAgent checkpoint directory not found: ${MODEL_NAME_OR_PATH}" >&2
  exit 2
fi

"${TORCHRUN}" --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 \
  --nproc-per-node="${GPUS}" --master-port="${MASTER_PORT}" -m kev.camera.train \
  "${train_args[@]}" "${val_args[@]}" \
  --model "${MODEL_NAME_OR_PATH}" --out "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" --batch "${PER_DEVICE_BATCH_SIZE}" --accum "${GRADIENT_ACC}" \
  --num-workers "${DATALOADER_NUM_WORKERS}" \
  --max-length "${MAX_LENGTH}" --image-size "${IMAGE_SIZE}" --head-dim "${HEAD_DIM}" \
  --lr "${LEARNING_RATE}" --head-lr "${HEAD_LR}" \
  --weight-decay "${WEIGHT_DECAY}" --warmup-ratio "${WARMUP_RATIO}" \
  --save-every-steps "${SAVE_STEPS}" --max-steps "${MAX_STEPS}" --seed "${SEED}" \
  "${freeze_args[@]}" "$@"
