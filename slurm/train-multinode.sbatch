#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Multi-node MeArmRL training job (pyxis/enroot + torchrun + NCCL).
#
# Submit from the directory containing your checkout of the repo:
#   cd ~/MeArmRL && sbatch slurm/train-multinode.sbatch
#
# Override node/GPU counts per job:
#   sbatch --nodes=4 --gpus-per-node=1 slurm/train-multinode.sbatch
#
# The script bind-mounts:
#   - your source checkout  -> /opt/MeArmRL/source/MeArmRL (edits take effect)
#   - your per-user logs    -> /logs
#   - your per-user cache   -> /isaac-sim/.cache (shader/ComputeCache persist)

#SBATCH --job-name=mearmrl-train-mn
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

set -euo pipefail

# --- Task / training args (override per job) --------------------------------
TASK="${TASK:-Template-Mearmrl-v0}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"
SEED="${SEED:-42}"
CHECKPOINT="${CHECKPOINT:-}"

# --- SLURM-derived env + per-user scratch setup -----------------------------
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# --- Container image ---------------------------------------------------------
IMAGE="ghcr.io/<org>/mearmrl:0.1.0"

# --- Container mounts --------------------------------------------------------
MOUNTS=(
    "${SLURM_SUBMIT_DIR}/source/MeArmRL:/opt/MeArmRL/source/MeArmRL"
    "${MEARMRL_JOB_LOGS}:/logs"
    "${MEARMRL_CACHE_ROOT}/.cache:/isaac-sim/.cache"
    "${MEARMRL_CACHE_ROOT}/computecache:/isaac-sim/.nv/ComputeCache"
    "${MEARMRL_CACHE_ROOT}/ov-logs:/isaac-sim/.nvidia-omniverse/logs"
    "${MEARMRL_CACHE_ROOT}/ov-config:/isaac-sim/.nvidia-omniverse/config"
    "${MEARMRL_CACHE_ROOT}/ov-data:/isaac-sim/.local/share/ov/data"
    "${MEARMRL_CACHE_ROOT}/ov-pkg:/isaac-sim/.local/share/ov/pkg"
    "${MEARMRL_CACHE_ROOT}/hub:/var/cache/hub"
)
MOUNT_ARGS=()
for m in "${MOUNTS[@]}"; do
    MOUNT_ARGS+=(--container-mounts="${m}")
done

# --- Build the torchrun command ----------------------------------------------
TRAIN_ARGS=(--task="${TASK}" --num_envs="${NUM_ENVS}" --max_iterations="${MAX_ITERATIONS}" --seed="${SEED}" --distributed --headless)
if [ -n "${CHECKPOINT}" ]; then
    TRAIN_ARGS+=(--checkpoint="${CHECKPOINT}")
fi

# --- Launch (torchrun rendezvous on MASTER_ADDR:MASTER_PORT) -----------------
# --container-remap-root --container-user=0: the container's root user is
# remapped to the host user, so the container (which runs as 1234:1234 by
# default) can write to your per-user scratch dirs without any chown.
srun \
    --container-image="${IMAGE}" \
    --container-remap-root \
    --container-user=0 \
    "${MOUNT_ARGS[@]}" \
    --container-workdir=/opt/MeArmRL \
    torchrun \
        --nnodes="${SLURM_NNODES}" \
        --nproc_per_node="${SLURM_GPUS_PER_NODE}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --node_rank="${NODE_RANK}" \
        scripts/skrl/train.py "${TRAIN_ARGS[@]}"
