# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MeArmRL container image for HPC deployment (Docker + Singularity/Apptainer).

This image is a **shared, frozen baseline**. It contains Isaac Sim 5.1.0,
Isaac Lab 2.3.2, the RL stack (skrl, gymnasium, hydra), and an editable
install of the MeArmRL extension. Students **never build or push** this image;
they bind-mount their own checkout of `source/MeArmRL` over
`/workspace/isaaclab/source/MeArmRL` at runtime so their reward-function edits
take effect without a rebuild.

## Maintainer: build & push

The image is built in two stages:

### 1. Build the Isaac Lab base image (from the local Isaac Lab checkout)

The local checkout at `/mnt/E/IsaacLab` (Isaac Lab 2.3.2, commit
`b0542fe2d`) ships its own docker tooling. Build the base image with:

```bash
cd /mnt/E/IsaacLab/docker
python3 container.py build base
```

This produces the `isaac-lab-base` image (Isaac Sim 5.1.0 + Isaac Lab 2.3.2).
The Isaac Sim version is pinned in `/mnt/E/IsaacLab/docker/.env.base`
(`ISAACSIM_VERSION=5.1.0`).

### 2. Build the MeArmRL image on top

```bash
docker build -t ghcr.io/adilfaisal01/mearmrl:0.1.0 -f docker/Dockerfile .
docker push ghcr.io/adilfaisal01/mearmrl:0.1.0
```

Notes:

- The base image runs as root (Isaac Lab's default). The MeArmRL layer keeps
  that user.
- `ACCEPT_EULA=Y` and `PRIVACY_CONSENT=Y` are baked in. Set
  `PRIVACY_CONSENT` to `N` if you want to opt out of telemetry.
- The editable install of MeArmRL means the bind-mounted source at
  `/workspace/isaaclab/source/MeArmRL` overrides the baked-in copy at runtime.

## Build the Singularity/Apptainer SIF

On the HPC login node (see `docker/cluster/build_sif.sh`):

```bash
# From the local Docker daemon (no registry round-trip):
bash docker/cluster/build_sif.sh 0.1.0

# Or from the registry:
apptainer build mearmrl_0.1.0.sif docker://ghcr.io/adilfaisal01/mearmrl:0.1.0
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
