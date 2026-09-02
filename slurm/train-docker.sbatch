#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Single-node MeArmRL training job for clusters with NATIVE Docker on the
# compute nodes (no pyxis/enroot). For pyxis/enroot clusters use train.sbatch.
#
# Submit from the directory containing your checkout of the repo:
#   MEARMRL_BACKEND=docker sbatch slurm/train-docker.sbatch
#
# Image distribution (pick one, see slurm/README.md "Native Docker"):
#   - Self-hosted registry on the login node:
#       MEARMRL_IMAGE=login-node:5000/mearmrl:local sbatch slurm/train-docker.sbatch
#   - docker save/load via shared filesystem:
#       docker save mearmrl:local | gzip > /scratch/$USER/mearmrl.tar.gz
#     then load on the node before running (see slurm/README.md).
#
# Notes:
#   - No --container-remap-root equivalent: the container runs as root.
#     env.sh detects MEARMRL_BACKEND=docker and makes scratch dirs world-writable.
#   - For rootless Docker/Podman clusters, adapt the launch line to
#     `podman run --userns=keep-id` to keep host-uid file ownership.
#   - Multi-node is not provided; adapt from train-multinode.sbatch if needed.

#SBATCH --job-name=mearmrl-train-docker
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
# MEARMRL_BACKEND=docker makes env.sh chmod 777 the scratch dirs.
export MEARMRL_BACKEND=docker
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# --- Container image ---------------------------------------------------------
IMAGE="${MEARMRL_IMAGE:-mearmrl:local}"

# --- Container mounts --------------------------------------------------------
MOUNT_ARGS=()
for m in \
    "${SLURM_SUBMIT_DIR}/source/MeArmRL:/MeArmRL/source/MeArmRL" \
    "${MEARMRL_JOB_LOGS}:/logs" \
    "${MEARMRL_CACHE_ROOT}/kit:/isaac-sim/kit/cache" \
    "${MEARMRL_CACHE_ROOT}/ov:/root/.cache/ov" \
    "${MEARMRL_CACHE_ROOT}/glcache:/root/.cache/nvidia/GLCache" \
    "${MEARMRL_CACHE_ROOT}/computecache:/root/.nv/ComputeCache" \
    "${MEARMRL_CACHE_ROOT}/ov-logs:/root/.nvidia-omniverse/logs" \
    "${MEARMRL_CACHE_ROOT}/ov-data:/root/.local/share/ov/data"; do
    MOUNT_ARGS+=(-v "${m}")
done

# --- Build the training command ----------------------------------------------
TRAIN_ARGS=(--task="${TASK}" --num_envs="${NUM_ENVS}" --max_iterations="${MAX_ITERATIONS}" --seed="${SEED}" --headless)
if [ -n "${CHECKPOINT}" ]; then
    TRAIN_ARGS+=(--checkpoint="${CHECKPOINT}")
fi

# --- Launch -------------------------------------------------------------------
srun docker run --rm --gpus all \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e XDG_RUNTIME_DIR=/tmp/xdg \
    -e OMNI_KIT_ALLOW_ROOT=1 \
    -e ISAACLAB_PATH=/IsaacLab \
    -e PYTHONPATH=/IsaacLab/source:/MeArmRL/source \
    "${MOUNT_ARGS[@]}" \
    -w /MeArmRL \
    "${IMAGE}" \
    /IsaacLab/isaaclab.sh -p scripts/skrl/train.py "${TRAIN_ARGS[@]}"