#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Local Docker debugging wrapper for MeArmRL.
#
# Runs the locally-built MeArmRL image on your own GPU machine, bind-mounting
# your local source edits so they take effect without rebuilding the image.
#
# Usage:
#   bash scripts/dev_local.sh --task=Template-Mearmrl-v0 --num_envs=64 --headless
#   bash scripts/dev_local.sh --shell          # drop into a bash in the container
#
# Prerequisites (one-time):
#   docker login nvcr.io                                        # free NGC API key
#   docker build -t mearmrl:local -f docker/Dockerfile .        # ~15-20 min first build
#
# Requires: Docker with the NVIDIA Container Toolkit, an NVIDIA GPU.

set -euo pipefail

IMAGE="${MEARMRL_IMAGE:-mearmrl:local}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "[dev_local] ERROR: image '${IMAGE}' not found." >&2
    echo "[dev_local] Build it first:" >&2
    echo "[dev_local]   docker login nvcr.io && docker build -t mearmrl:local -f docker/Dockerfile ." >&2
    exit 1
fi

# --- Command ---------------------------------------------------------------
if [ "${1:-}" = "--shell" ]; then
    CMD=(bash)
else
    CMD=(/IsaacLab/isaaclab.sh -p scripts/skrl/train.py "$@")
fi

# --- Run --------------------------------------------------------------------
exec docker run --rm --gpus all \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e XDG_RUNTIME_DIR=/tmp/xdg \
    -e OMNI_KIT_ALLOW_ROOT=1 \
    -e ISAACLAB_PATH=/IsaacLab \
    -e PYTHONPATH=/IsaacLab/source:/MeArmRL/source \
    -v "${REPO_ROOT}/source/MeArmRL:/MeArmRL/source/MeArmRL" \
    -v "${REPO_ROOT}/logs:/MeArmRL/logs" \
    -v "${REPO_ROOT}/logs:/logs" \
    -w /MeArmRL \
    "${IMAGE}" \
    "${CMD[@]}"