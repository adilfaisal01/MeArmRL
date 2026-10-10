#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Set up a CPU-only Python environment for the Week-1 lab and run its checks.
#
# No GPU, no Isaac Sim, no Docker. Works on Linux, macOS, and Windows (Git-Bash
# or WSL). This is the "start coding" path: you can read, edit, and test the
# reward math before you ever touch a GPU box.
#
# Usage:
#   bash scripts/lab_setup.sh
#   LAB_VENV=/somewhere/else bash scripts/lab_setup.sh   # custom venv location
#
# Training itself still needs the GPU container -- see docker/README.md.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${LAB_VENV:-${REPO_ROOT}/.venv-lab}"
PYTHON="${PYTHON:-python3}"

echo "=== [1/3] Creating virtualenv at ${VENV} ==="
if [ ! -d "${VENV}" ]; then
    "${PYTHON}" -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

echo "=== [2/3] Installing CPU dependencies (torch, pyyaml, pytest) ==="
python -m pip install --upgrade pip
case "$(uname -s)" in
    Linux)
        # The default PyPI torch wheel for Linux pulls in ~2.5 GB of CUDA libraries
        # that a CPU-only lab environment never uses. The CPU index is a fraction
        # of that. macOS and Windows wheels are already CPU/MPS builds.
        python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
        ;;
    *)
        python -m pip install torch
        ;;
esac
python -m pip install pyyaml pytest

echo "=== [3/3] Running lab checks ==="
set +e
python "${REPO_ROOT}/scripts/check_week01.py" --static
static_rc=$?
set -e
echo "    (static exit: ${static_rc}; a fresh checkout reports 'none of the student files were modified')"
python -m pytest "${REPO_ROOT}/labs/week01_intro_isaaclab/harness" -q

echo
echo "Setup complete. Activate the environment with:"
echo "    source ${VENV}/bin/activate"
echo "Then explore the rewards with:"
echo "    python labs/week01_intro_isaaclab/harness/playground.py"
