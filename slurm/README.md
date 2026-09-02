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
- The cluster runs **pyxis/enroot** (see `slurm/train.sbatch`) or has **native
  Docker** on the compute nodes (see `slurm/train-docker.sbatch` — details in
  "Choosing a cluster backend" below).
- A free [NGC account](https://ngc.nvidia.com/) and API key (needed to pull
  the `nvcr.io/nvidia/isaac-sim` base image).

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

No `chown` is needed on pyxis clusters: the sbatch scripts run the container
with `--container-remap-root --container-user=0`, which remaps the container's
root user to your host user. `slurm/env.sh` re-runs the `mkdir`/`chmod`
idempotently on every job.

Then get the image onto the cluster — **pick one of the three paths below**
depending on what is installed on the login node and your uplink speed. All
three produce the same thing: `~/mearmrl_local.sif`, visible on every node
via the shared home filesystem. On pyxis clusters pyxis only needs the SIF
file to *run* — the login node does not need Docker or Apptainer installed.

| Login node has… | Recommended path |
|---|---|
| Docker + good bandwidth | **C** (build on login node) |
| Apptainer but no Docker | **A** (build SIF on laptop, upload) |
| Neither Docker nor Apptainer | **A** (build SIF on laptop, upload) |
| Slow login-node bandwidth to nvcr.io | **A** or **B** (offload the heavy pull to your laptop) |

Regardless of path, pyxis clusters have no `chown` to worry about: the sbatch
scripts run the container with `--container-remap-root --container-user=0`,
which remaps the container's root user to your host user. `slurm/env.sh`
re-runs the `mkdir`/`chmod` idempotently on every job.

### Path A — build the SIF on your laptop, upload to the HPC

Universal fallback: works on any cluster with pyxis/enroot, because the login
node only needs the SIF file to run it. Requires Docker + Apptainer on your
laptop (or just Apptainer, using the standalone definition).

```bash
# On your laptop, from the repo root.

# With Docker + Apptainer:
docker login nvcr.io
docker build -t mearmrl:local -f docker/Dockerfile .
bash docker/cluster/build_sif.sh local "$HOME"      # writes ~/mearmrl_local.sif

# OR without Docker (Apptainer only, standalone definition):
export APPTAINER_DOCKER_USERNAME='$oauthtoken'
export APPTAINER_DOCKER_PASSWORD='<NGC API key>'
apptainer build ~/mearmrl_local.sif docker/cluster/apptainer-standalone.def

# Upload (~11 GB; rsync resumes if the connection drops):
rsync -avP ~/mearmrl_local.sif <user>@hpc-login:~/
```

On the HPC, submit as usual — `train.sbatch` defaults `MEARMRL_IMAGE` to
`~/mearmrl_local.sif`.

### Path B — build Docker on your laptop, upload the tarball, convert on the login node

Use if your laptop has Docker but you'd rather not install Apptainer locally.
**Requires Docker on the login node** (to `docker load` the tarball) — if the
login node has no Docker, use Path A instead.

```bash
# On your laptop:
docker login nvcr.io
docker build -t mearmrl:local -f docker/Dockerfile .
docker save mearmrl:local | gzip > mearmrl.tar.gz   # ~11 GB

# Upload:
rsync -avP mearmrl.tar.gz <user>@hpc-login:/scratch/$USER/

# On the login node:
cd ~/MeArmRL && docker load < /scratch/$USER/mearmrl.tar.gz
bash docker/cluster/build_sif.sh local "$HOME"
```

### Path C — build everything on the login node

Simplest when the login node has Docker and good bandwidth to nvcr.io. No
upload at all — the pull happens over the cluster's fast connection.

```bash
# On the login node, from the repo root:
docker login nvcr.io
docker build -t mearmrl:local -f docker/Dockerfile .    # ~15-20 min first build
bash docker/cluster/build_sif.sh local "$HOME"          # writes ~/mearmrl_local.sif
```

## Choosing a cluster backend

Your cluster uses one of two container runtimes. Pick the matching sbatch:

| Backend | sbatch | Image format | Distribution |
|---|---|---|---|
| pyxis/enroot (most academic clusters) | `slurm/train.sbatch`, `slurm/train-multinode.sbatch` | `.sif` file | Build SIF on the login node; all nodes read it from the shared home FS |
| Native Docker on compute nodes | `slurm/train-docker.sbatch` | Docker image | Self-hosted registry **or** `docker save/load` (below) |

### Native Docker: getting the image onto the compute nodes

Each compute node's Docker daemon needs the image. Two options:

**Option A — self-hosted registry on the login node** (fast, one pull per node):

```bash
# On the login node:
docker run -d -p 5000:5000 --restart=always --name registry registry:2
docker tag mearmrl:local <login-node-hostname>:5000/mearmrl:local
docker push <login-node-hostname>:5000/mearmrl:local
```

Requires the login node to expose port 5000 to compute nodes. Submit with:

```bash
MEARMRL_IMAGE=<login-node-hostname>:5000/mearmrl:local sbatch slurm/train-docker.sbatch
```

**Option B — `docker save`/`load` via the shared filesystem** (no registry):

```bash
# On the login node:
docker save mearmrl:local | gzip > /scratch/$USER/mearmrl.tar.gz

# On a compute node (once, or as a preamble step in the job):
docker load < /scratch/$USER/mearmrl.tar.gz
```

No registry to run, but the ~11 GB tarball eats scratch space and every node
pays a one-time load.

**Permissions**: native Docker has no `--container-remap-root`, so the
container runs as real root. `train-docker.sbatch` sets
`MEARMRL_BACKEND=docker`, and `slurm/env.sh` then makes your scratch dirs
world-writable (`chmod 777`) so the container can write checkpoints. If your
cluster offers rootless Docker or Podman, prefer
`podman run --userns=keep-id` to keep host-uid file ownership instead.
Multi-node training is not provided for the Docker path — adapt
`train-multinode.sbatch` if your cluster needs it.

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

Override training args per job (defaults: `Mearmrl-Reach-v0`, 1024 envs,
4800 iterations; the cartpole template task is available as a known-good
control via `TASK=Template-Mearmrl-v0`):

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
