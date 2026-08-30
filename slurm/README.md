# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MeArmRL on HPC (SLURM + pyxis/enroot)

This document is the student-facing guide. It covers the three development
workflows, the one-time per-user setup, and how to submit single- and
multi-node jobs.

## Prerequisites

- A SLURM account on the cluster with GPU access.
- The cluster runs **pyxis/enroot** (or Apptainer/Singularity as a fallback).
- The shared image `ghcr.io/adilfaisal01/mearmrl:<tag>` is available (maintainer-pushed).

## One-time per-user setup

Run once on the HPC login node:

```bash
# 1. Clone the repo into your home directory
git clone <your-fork-or-the-shared-repo-url> ~/MeArmRL
cd ~/MeArmRL

# 2. Create your per-user scratch dirs
mkdir -p /scratch/$USER/mearmrl-logs/jobs /scratch/$USER/isaac-sim-cache
chmod 750 /scratch/$USER/mearmrl-logs /scratch/$USER/isaac-sim-cache
```

No `chown` is needed: the sbatch scripts run the container with
`--container-remap-root --container-user=0`, which remaps the container's root
user to your host user. The container then writes to your own scratch dirs
naturally. `slurm/env.sh` re-runs the `mkdir`/`chmod` idempotently on every
job.

## Development workflows

### 1. Local Docker for fast iteration (recommended for debugging)

Run the same frozen image on your own GPU machine to debug reward functions
before spending HPC hours:

```bash
bash scripts/dev_local.sh --task=Template-Mearmrl-v0 --num_envs=64 --headless
```

Your local `source/MeArmRL` edits are bind-mounted into the container, so
changes take effect immediately -- no rebuild. Logs land in `./logs/`.

### 2. Local edit -> git sync -> HPC submit (pure git)

1. Edit reward functions / env config in `source/MeArmRL/MeArmRL/tasks/manager_based/mearmrl/` on your laptop.
2. Commit and push to your fork/branch.
3. On the HPC login node, pull and submit:

   ```bash
   cd ~/MeArmRL && git pull && sbatch slurm/train.sbatch
   ```

   The sbatch script uses `$SLURM_SUBMIT_DIR` as the bind-mount source, so
   whatever checkout you submit from is what runs.

### 3. VS Code Remote-SSH (feels local, files live on HPC)

Open `~/MeArmRL` on the HPC via VS Code Remote-SSH. Edit in your local editor
window; files save directly to HPC. Submit from the integrated terminal.

## Submitting jobs

### Single node

```bash
cd ~/MeArmRL && sbatch slurm/train.sbatch
```

Override training args per job:

```bash
TASK=Template-Mearmrl-v0 NUM_ENVS=128 MAX_ITERATIONS=2000 sbatch slurm/train.sbatch
```

### Multi-node (torchrun + NCCL)

```bash
cd ~/MeArmRL && sbatch --nodes=4 --gpus-per-node=1 slurm/train-multinode.sbatch
```

`slurm/env.sh` derives `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, and `RANK`
from the SLURM environment, so the same script works for 1 or N nodes.

## Where your work lives

| Artifact | Location | Notes |
|---|---|---|
| Logs / checkpoints | `/scratch/$USER/mearmrl-logs/jobs/<SLURM_JOB_ID>/` | Job-ID-prefixed; cannot collide with other jobs or users |
| Isaac Sim shader/ComputeCache | `/scratch/$USER/isaac-sim-cache/` | Persists across jobs; first run is slow, later runs are fast |
| Your source edits | `~/MeArmRL/source/MeArmRL/` | Bind-mounted into the container at runtime |

## Reading a teammate's run

Per-user dirs are `chmod 750` (group read/execute), so teammates in the same
group can inspect but not modify each other's runs:

```bash
ls /scratch/<teammate>/mearmrl-logs/jobs/
```

To resume from a teammate's checkpoint, copy it into your own logs dir first
(or pass `--checkpoint` with the full path -- the container will warn if it is
outside your own `/logs`).

## Troubleshooting

- **`/logs is not writable`**: the container user cannot write your scratch
  dir. Make sure the sbatch scripts run with `--container-remap-root
  --container-user=0` (they do by default).
- **Slow first run**: expected -- shader/ComputeCache warm-up. Subsequent runs
  are 3-5x faster.
- **Multi-node hangs**: check `NCCL_DEBUG=WARN` output in the `.err` file.
  Confirm InfiniBand (`NCCL_IB_HCA`) and that `--exclusive` is set.
- **No GPU found**: ensure `--gpus-per-node` matches the partition and that
  pyxis/enroot is configured with the NVIDIA runtime.
