# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SLURM-derived environment for distributed MeArmRL training.
#
# Source this from an sbatch script. It derives MASTER_ADDR, MASTER_PORT,
# WORLD_SIZE, and RANK from the SLURM environment so the same script works
# for single- and multi-node jobs.

# --- Distributed training env ---------------------------------------------
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export MASTER_PORT="${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}"
export WORLD_SIZE="$((SLURM_NNODES * SLURM_GPUS_PER_NODE))"
export NODE_RANK="${SLURM_NODEID}"

# --- NCCL (defaults are set in the container entrypoint; overridable here) --
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^docker0,lo}"

# --- Per-user scratch paths (strict isolation) -----------------------------
export MEARMRL_LOGS_ROOT="/scratch/${USER}/mearmrl-logs"
export MEARMRL_CACHE_ROOT="/scratch/${USER}/isaac-sim-cache"
export MEARMRL_JOB_LOGS="${MEARMRL_LOGS_ROOT}/jobs/${SLURM_JOB_ID}"

# --- One-time setup (idempotent) -------------------------------------------
# Create per-user scratch dirs. Group read/execute bits let teammates/TA
# inspect runs. No chown needed: the sbatch scripts run the container with
# --container-remap-root --container-user=0, so the container writes as the
# host user and owns these dirs naturally.
mkdir -p "${MEARMRL_LOGS_ROOT}/jobs" "${MEARMRL_CACHE_ROOT}"
chmod 750 "${MEARMRL_LOGS_ROOT}" "${MEARMRL_CACHE_ROOT}"
