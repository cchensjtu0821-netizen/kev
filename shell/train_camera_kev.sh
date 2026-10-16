#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# Start from the original CameraAgent 0.4B and initialize a new Kev pointer head.
camera_start=base
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/srv/workspace/Kirin_AI_Workspace/AIC_I/l00883213/camera_agent/hisi_dinov3_llm022/released_model/vlm_0p4_384_full_vb46k_general_domain_v3_0808}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/runs/camera-kev-04b}"
source "${PROJECT_DIR}/shell/camera_train_common.sh" "$@"
