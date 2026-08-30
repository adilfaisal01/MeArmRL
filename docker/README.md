# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MeArmRL container image for HPC deployment (Docker + Singularity/Apptainer).

This image is a **shared, frozen baseline**. It contains Isaac Sim 6.0.0,
Isaac Lab, the RL stack (skrl, gymnasium, hydra), and an editable install of
the MeArmRL extension. Students **never build or push** this image; they
bind-mount their own checkout of `source/MeArmRL` over
`/opt/MeArmRL/source/MeArmRL` at runtime so their reward-function edits take
effect without a rebuild.

## Maintainer: build & push

```bash
# 1. Build (pin the Isaac Lab tag/commit compatible with Isaac Sim 6.0.0)
docker build -t ghcr.io/<org>/mearmrl:0.1.0 \
    --build-arg ISAAC_LAB_REF=<pinned-isaaclab-tag-or-commit> \
    -f docker/Dockerfile .

# 2. Push to GitHub Container Registry
docker push ghcr.io/<org>/mearmrl:0.1.0
```

Notes:

- The base image `nvcr.io/nvidia/isaac-sim:6.0.0` runs as non-root user
  `1234:1234`. The Dockerfile keeps that user.
- `ACCEPT_EULA=Y` and `PRIVACY_CONSENT=Y` are baked in. Set
  `PRIVACY_CONSENT` to `N` if you want to opt out of telemetry.
- The editable install of MeArmRL means the bind-mounted source at
  `/opt/MeArmRL/source/MeArmRL` overrides the baked-in copy at runtime.

## Build the Singularity/Apptainer SIF

On the HPC login node (see `docker/cluster/build_sif.sh`):

```bash
# From the local Docker daemon (no registry round-trip):
bash docker/cluster/build_sif.sh 0.1.0

# Or from the registry:
apptainer build mearmrl_0.1.0.sif docker://ghcr.io/<org>/mearmrl:0.1.0
```

The SIF is read-only by Apptainer default, so students cannot modify it.

## Local development (students)

Students can run the same image on their own GPU machine to debug reward
functions before spending HPC hours:

```bash
bash scripts/dev_local.sh --task=Template-Mearmrl-v0 --num_envs=64 --headless
```

See `slurm/README.md` for the full student workflow (local edit -> git sync ->
HPC submit, VS Code Remote-SSH, and local Docker debugging).
