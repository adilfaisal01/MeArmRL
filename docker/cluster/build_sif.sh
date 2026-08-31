#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Build the MeArmRL Singularity/Apptainer SIF from the locally-built Docker image.
#
# NOTE: requires Docker (reads the local daemon). For a Docker-free SIF build,
# use `docker/cluster/apptainer-standalone.def` instead — it builds the full
# image from nvcr.io with Apptainer only.
#
# Usage:
#   bash docker/cluster/build_sif.sh <TAG> [OUT_DIR]
#
# Examples:
#   # On the login node (image must already be built: docker build -t mearmrl:local ...):
#   bash docker/cluster/build_sif.sh local "$HOME"     # writes /home/<user>/mearmrl_local.sif
#   bash docker/cluster/build_sif.sh local             # writes docker/cluster/exports/mearmrl_local.sif
#
# Prerequisites (one-time before this script):
#   docker login nvcr.io && docker build -t mearmrl:local -f docker/Dockerfile .

set -euo pipefail

TAG="${1:?Usage: build_sif.sh <TAG> [OUT_DIR]}"
OUT_DIR="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/exports" && pwd)}"
SRC_IMAGE="mearmrl:${TAG}"

mkdir -p "${OUT_DIR}"
SIF="${OUT_DIR}/mearmrl_${TAG}.sif"

# Build from the local Docker daemon (no registry round-trip).
if ! docker image inspect "${SRC_IMAGE}" >/dev/null 2>&1; then
    echo "[build_sif] ERROR: local image '${SRC_IMAGE}' not found." >&2
    echo "[build_sif] Build it first:" >&2
    echo "[build_sif]   docker login nvcr.io && docker build -t mearmrl:${TAG} -f docker/Dockerfile ." >&2
    exit 1
fi
echo "[build_sif] Building from local Docker daemon: ${SRC_IMAGE}"
apptainer build "${SIF}" "docker-daemon://${SRC_IMAGE}"

echo "[build_sif] SIF written to: ${SIF}"