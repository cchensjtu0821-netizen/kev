#!/usr/bin/env bash
set -euo pipefail

TRAIN_META="${TRAIN_META:-}"
VAL_META="${VAL_META:-}"
TRAIN_DATA="${TRAIN_DATA:-}"
VAL_DATA="${VAL_DATA:-}"
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
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
HEAD_LR="${HEAD_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_STEPS="${SAVE_STEPS:-2500}"
MAX_STEPS="${MAX_STEPS:-0}"
STOP_AFTER="${STOP_AFTER:-0}"
RESUME="${RESUME:-}"
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

source_args=()
if [[ -n "${RESUME}" ]]; then
  source_args=(--resume "${RESUME}")
elif [[ "${camera_start}" == "base" ]]; then
  source_args=(--model "${MODEL_NAME_OR_PATH}")
  if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
    echo "CameraAgent checkpoint directory not found: ${MODEL_NAME_OR_PATH}" >&2
    exit 2
  fi
else
  if [[ -z "${CAMERA_CHECKPOINT:-}" || ! -d "${CAMERA_CHECKPOINT}" ]]; then
    echo "Set CAMERA_CHECKPOINT to a saved Camera Kev checkpoint directory, or set RESUME" >&2
    exit 2
  fi
  source_args=(--checkpoint "${CAMERA_CHECKPOINT}")
fi

shape_args=()
if [[ -n "${MAX_LENGTH:-}" ]]; then shape_args+=(--max-length "${MAX_LENGTH}"); fi
if [[ -n "${IMAGE_SIZE:-}" ]]; then shape_args+=(--image-size "${IMAGE_SIZE}"); fi
if [[ -n "${HEAD_DIM:-}" ]]; then shape_args+=(--head-dim "${HEAD_DIM}"); fi
if [[ -n "${LORA_R:-}" ]]; then shape_args+=(--lora-r "${LORA_R}"); fi
if [[ -n "${QUESTION_LAYOUT:-}" ]]; then shape_args+=(--question-layout "${QUESTION_LAYOUT}"); fi

"${TORCHRUN}" --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 \
  --nproc-per-node="${GPUS}" --master-port="${MASTER_PORT}" -m kev.camera.train \
  "${train_args[@]}" "${val_args[@]}" \
  "${source_args[@]}" --out "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" --batch "${PER_DEVICE_BATCH_SIZE}" --accum "${GRADIENT_ACC}" \
  --num-workers "${DATALOADER_NUM_WORKERS}" \
  "${shape_args[@]}" \
  --lr "${LEARNING_RATE}" --head-lr "${HEAD_LR}" \
  --weight-decay "${WEIGHT_DECAY}" --warmup-ratio "${WARMUP_RATIO}" \
  --save-every-steps "${SAVE_STEPS}" --max-steps "${MAX_STEPS}" \
  --stop-after "${STOP_AFTER}" --seed "${SEED}" \
  "${freeze_args[@]}" "$@"
