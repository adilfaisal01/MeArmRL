#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Build the MeArmRL Singularity/Apptainer SIF from the Docker image.
#
# Usage:
#   bash docker/cluster/build_sif.sh <TAG> [REGISTRY]
#
# Examples:
#   # From the local Docker daemon (no registry round-trip):
#   bash docker/cluster/build_sif.sh 0.1.0
#
#   # From the registry:
#   bash docker/cluster/build_sif.sh 0.1.0 ghcr.io/adilfaisal01
#
# The SIF is written to docker/cluster/exports/ (gitignored).

set -euo pipefail

TAG="${1:?Usage: build_sif.sh <TAG> [REGISTRY]}"
REGISTRY="${2:-ghcr.io/adilfaisal01}"

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/exports" && pwd)"
mkdir -p "${OUT_DIR}"
SIF="${OUT_DIR}/mearmrl_${TAG}.sif"

# Prefer the local Docker daemon; fall back to the registry.
if docker image inspect "${REGISTRY}/mearmrl:${TAG}" >/dev/null 2>&1; then
    echo "[build_sif] Building from local Docker daemon: ${REGISTRY}/mearmrl:${TAG}"
    apptainer build "${SIF}" "docker-daemon://${REGISTRY}/mearmrl:${TAG}"
else
    echo "[build_sif] Building from registry: ${REGISTRY}/mearmrl:${TAG}"
    apptainer build "${SIF}" "docker://${REGISTRY}/mearmrl:${TAG}"
fi

echo "[build_sif] SIF written to: ${SIF}"
