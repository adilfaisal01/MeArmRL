#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Test the MeArmRL containerization pipeline.
#
# Stages:
#   1. Static checks (bash -n on all shell scripts)
#   2. Build the MeArmRL image (requires isaac-lab-base to exist)
#   3. Smoke test: list registered tasks inside the container
#   4. (Optional) zero-agent run with a few envs to verify the env loads
#
# Usage:
#   bash scripts/test_container.sh            # static checks + build + list_envs
#   bash scripts/test_container.sh --full     # also runs zero_agent smoke test
#   bash scripts/test_container.sh --skip-build  # only static checks + smoke test
#
# Requires: Docker, an NVIDIA GPU (for the smoke test), and the isaac-lab-base
# image built from /mnt/E/IsaacLab (see docker/README.md).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${MEARMRL_IMAGE:-ghcr.io/adilfaisal01/mearmrl:0.1.0}"
BASE_IMAGE="${MEARMRL_BASE_IMAGE:-isaac-lab-base}"
TASK="${TASK:-Template-Mearmrl-v0}"

FULL=0
SKIP_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --full) FULL=1 ;;
        --skip-build) SKIP_BUILD=1 ;;
        *) echo "[test] Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

echo "=== [1/4] Static checks ==="
for f in \
    docker/entrypoint.sh \
    docker/cluster/build_sif.sh \
    slurm/env.sh \
    slurm/train.sbatch \
    slurm/train-multinode.sbatch \
    scripts/dev_local.sh \
    scripts/test_container.sh; do
    bash -n "${REPO_ROOT}/${f}"
    echo "  OK: ${f}"
done

echo "=== [2/4] Build the MeArmRL image ==="
if [ "${SKIP_BUILD}" -eq 1 ]; then
    echo "  Skipped (--skip-build)."
else
    if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
        echo "  ERROR: base image '${BASE_IMAGE}' not found." >&2
        echo "  Build it first: cd /mnt/E/IsaacLab/docker && python3 container.py build base" >&2
        exit 1
    fi
    docker build -t "${IMAGE}" -f "${REPO_ROOT}/docker/Dockerfile" "${REPO_ROOT}"
    echo "  Built: ${IMAGE}"
fi

echo "=== [3/4] Smoke test: list registered tasks ==="
docker run --rm \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e XDG_RUNTIME_DIR=/tmp/xdg \
    -e OMNI_KIT_ALLOW_ROOT=1 \
    -e ISAACLAB_PATH=/workspace/isaaclab \
    -e PYTHONPATH=/workspace/isaaclab/source \
    -w /workspace/isaaclab \
    "${IMAGE}" \
    /workspace/isaaclab/isaaclab.sh -p scripts/list_envs.py --keyword "${TASK}" 2>&1 | tail -20

echo "=== [4/4] Zero-agent smoke test ==="
if [ "${FULL}" -eq 1 ]; then
    docker run --rm --gpus all \
        -e ACCEPT_EULA=Y \
        -e PRIVACY_CONSENT=Y \
        -e XDG_RUNTIME_DIR=/tmp/xdg \
        -e OMNI_KIT_ALLOW_ROOT=1 \
        -e ISAACLAB_PATH=/workspace/isaaclab \
        -e PYTHONPATH=/workspace/isaaclab/source \
        -w /workspace/isaaclab \
        "${IMAGE}" \
        /workspace/isaaclab/isaaclab.sh -p scripts/zero_agent.py --task "${TASK}" --num_envs 4 --headless 2>&1 | tail -20
else
    echo "  Skipped (pass --full to run the zero-agent smoke test; requires a GPU)."
fi

echo "=== Done ==="
