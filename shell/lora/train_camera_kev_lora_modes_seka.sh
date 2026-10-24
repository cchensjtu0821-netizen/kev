#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QUESTION_LAYOUT="${QUESTION_LAYOUT:-all}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../../runs/camera-kev-04b-lora-modes-seka}"

exec bash "${SCRIPT_DIR}/train_camera_kev_lora_seka.sh" "$@"
