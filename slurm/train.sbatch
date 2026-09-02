#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Single-node MeArmRL training job (pyxis/enroot).
#
# Submit from the directory containing your checkout of the repo:
#   cd ~/MeArmRL && sbatch slurm/train.sbatch
#
# Prerequisites (one-time on the login node):
#   docker login nvcr.io                                     # free NGC API key
#   docker build -t mearmrl:local -f docker/Dockerfile .
#   bash docker/cluster/build_sif.sh local "$HOME"           # produces a SIF
#
# The script bind-mounts:
#   - your source checkout  -> /MeArmRL/source/MeArmRL (edits take effect)
#   - your per-user logs    -> /logs
#   - your per-user cache   -> Isaac Sim cache dirs (shader/ComputeCache persist)

#SBATCH --job-name=mearmrl-train
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

set -euo pipefail

# --- Task / training args (override per job) --------------------------------
TASK="${TASK:-Mearmrl-Reach-v0}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-4800}"
SEED="${SEED:-42}"
CHECKPOINT="${CHECKPOINT:-}"

# --- SLURM-derived env + per-user scratch setup -----------------------------
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# --- Container image ---------------------------------------------------------
# Default: the SIF built by `bash docker/cluster/build_sif.sh local "$HOME"`.
# Override with: MEARMRL_IMAGE=/full/path/to/mearmrl.sif sbatch slurm/train.sbatch
IMAGE="${MEARMRL_IMAGE:-${HOME}/mearmrl_local.sif}"
if [ ! -f "${IMAGE}" ]; then
    echo "[train.sbatch] ERROR: SIF '${IMAGE}' not found." >&2
    echo "[train.sbatch] Build it first on the login node:" >&2
    echo "[train.sbatch]   docker login nvcr.io && docker build -t mearmrl:local -f docker/Dockerfile ." >&2
    echo "[train.sbatch]   bash docker/cluster/build_sif.sh local \"\$HOME\"" >&2
    exit 1
fi

# --- Container mounts --------------------------------------------------------
# Student source (editable install path) + per-user logs + per-user cache.
# Cache paths follow the Isaac Lab docker layout.
MOUNTS=(
    "${SLURM_SUBMIT_DIR}/source/MeArmRL:/MeArmRL/source/MeArmRL"
    "${MEARMRL_JOB_LOGS}:/logs"
    "${MEARMRL_CACHE_ROOT}/kit:/isaac-sim/kit/cache"
    "${MEARMRL_CACHE_ROOT}/ov:/root/.cache/ov"
    "${MEARMRL_CACHE_ROOT}/pip:/root/.cache/pip"
    "${MEARMRL_CACHE_ROOT}/glcache:/root/.cache/nvidia/GLCache"
    "${MEARMRL_CACHE_ROOT}/computecache:/root/.nv/ComputeCache"
    "${MEARMRL_CACHE_ROOT}/ov-logs:/root/.nvidia-omniverse/logs"
    "${MEARMRL_CACHE_ROOT}/ov-data:/root/.local/share/ov/data"
    "${MEARMRL_CACHE_ROOT}/documents:/root/Documents"
)
MOUNT_ARGS=()
for m in "${MOUNTS[@]}"; do
    MOUNT_ARGS+=(--container-mounts="${m}")
done

# --- Build the training command ----------------------------------------------
TRAIN_ARGS=(--task="${TASK}" --num_envs="${NUM_ENVS}" --max_iterations="${MAX_ITERATIONS}" --seed="${SEED}" --headless)
if [ -n "${CHECKPOINT}" ]; then
    TRAIN_ARGS+=(--checkpoint="${CHECKPOINT}")
fi

# --- Launch ------------------------------------------------------------------
# --container-remap-root --container-user=0: the container's root user is
# remapped to the host user, so the container can write to your per-user
# scratch dirs without any chown.
srun \
    --container-image="${IMAGE}" \
    --container-remap-root \
    --container-user=0 \
    "${MOUNT_ARGS[@]}" \
    --container-workdir=/MeArmRL \
    /IsaacLab/isaaclab.sh -p scripts/skrl/train.py "${TRAIN_ARGS[@]}"
