#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Container entrypoint for MeArmRL.
#
# Responsibilities:
#   1. Ensure XDG_RUNTIME_DIR exists (required for EGL/offscreen rendering).
#   2. Export NCCL defaults for multi-node DDP -- only if the user has not
#      already set them, so SLURM jobs can override.
#   3. Guard against an unwritable /logs bind-mount (clear error, not silent
#      corruption).
#   4. exec the requested command.

set -euo pipefail

# --- XDG runtime dir -------------------------------------------------------
mkdir -p "${XDG_RUNTIME_DIR:-/tmp/xdg}"

# --- NCCL defaults (overridable) -------------------------------------------
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^docker0,lo}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

# --- Logs writability guard ------------------------------------------------
# train.py writes to the cwd-relative logs/ (= /MeArmRL/logs), so that is where
# runs must persist; /logs is also bind-mounted on HPC. Fail loudly instead of
# silently losing every checkpoint to the container's ephemeral layer.
for logs_dir in /MeArmRL/logs /logs; do
    if [ -d "${logs_dir}" ] && [ ! -w "${logs_dir}" ]; then
        echo "[entrypoint] ERROR: ${logs_dir} is not writable by the container user." >&2
        echo "[entrypoint] Fix on the HPC host (no root needed):" >&2
        echo "[entrypoint]   srun --container-remap-root --container-user=0 ..." >&2
        echo "[entrypoint] (remaps the container root to your host user, so the" >&2
        echo "[entrypoint]  container writes to your own scratch dirs naturally)." >&2
        exit 1
    fi
done

# --- Run -------------------------------------------------------------------
exec "$@"
