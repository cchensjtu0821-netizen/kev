#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# Start a new run from both the CameraAgent weights and the trained Kev pointer head.
camera_start=checkpoint
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/runs/camera-kev-continued}"
source "${PROJECT_DIR}/shell/camera_train_common.sh" "$@"
