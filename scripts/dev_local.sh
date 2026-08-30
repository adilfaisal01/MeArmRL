#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Local Docker debugging wrapper for MeArmRL.
#
# Runs the shared MeArmRL image on your own GPU machine, bind-mounting your
# local source edits so they take effect without rebuilding the image.
#
# Usage:
#   bash scripts/dev_local.sh --task=Template-Mearmrl-v0 --num_envs=64 --headless
#   bash scripts/dev_local.sh --shell          # drop into a bash in the container
#
# Requires: Docker with the NVIDIA Container Toolkit, an NVIDIA GPU.

set -euo pipefail

IMAGE="${MEARMRL_IMAGE:-ghcr.io/adilfaisal01/mearmrl:0.1.0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pull the image if not present locally.
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "[dev_local] Pulling ${IMAGE} ..."
    docker pull "${IMAGE}"
fi

# --- Command ---------------------------------------------------------------
if [ "${1:-}" = "--shell" ]; then
    CMD=(bash)
else
    CMD=(/workspace/isaaclab/isaaclab.sh -p scripts/skrl/train.py "$@")
fi

# --- Run --------------------------------------------------------------------
exec docker run --rm --gpus all \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e XDG_RUNTIME_DIR=/tmp/xdg \
    -e OMNI_KIT_ALLOW_ROOT=1 \
    -e ISAACLAB_PATH=/workspace/isaaclab \
    -e PYTHONPATH=/workspace/isaaclab/source \
    -v "${REPO_ROOT}/source/MeArmRL:/workspace/isaaclab/source/MeArmRL" \
    -v "${REPO_ROOT}/logs:/logs" \
    -w /workspace/isaaclab \
    "${IMAGE}" \
    "${CMD[@]}"
