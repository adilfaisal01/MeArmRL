# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MeArmRL container image (Docker + Compose + Singularity/Apptainer).

The image is **built locally by whoever needs it** — there is no shared
registry and no maintainer push step. It contains Isaac Sim 5.1.0, Isaac Lab
2.3.2 (cloned from GitHub and installed into the Isaac Sim Python), the RL
stack (skrl, gymnasium, hydra), and an editable install of the MeArmRL
extension. Students bind-mount their own checkout of `source/MeArmRL` over
`/MeArmRL/source/MeArmRL` at runtime, so reward-function edits take effect
without a rebuild.

Container layout:

| Path | Contents |
|---|---|
| `/isaac-sim` | NVIDIA Isaac Sim runtime (from the base image) |
| `/IsaacLab` | Isaac Lab 2.3.2 (cloned, pinned tag `v2.3.2`) |
| `/MeArmRL` | this repo: extension, scripts, entrypoint |

## Prerequisites (one-time)

- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- A free [NGC account](https://ngc.nvidia.com/) and API key, then:

```bash
docker login nvcr.io
```

## Build (single command)

```bash
docker build -t mearmrl:local -f docker/Dockerfile .
```

The first build pulls the ~10.5 GB Isaac Sim base from nvcr.io and runs
`isaaclab.sh --install` (torch + Isaac Lab extensions), so expect **15-20
minutes and an ~11 GB image**. Later rebuilds are fast (Docker layer cache);
only the layer containing your code changes.

## Run (two options)

### Option 1: Docker Compose (recommended)

```bash
cd docker

# Headless training (default service). Configure with env vars, either inline
# or via docker/.env (copy .env.example):
docker compose up

# One-off training that removes the container when it exits:
docker compose run --rm mearmrl

# Configure a run (defaults: Mearmrl-Reach-Student-v0, 1024 envs, seed 42, 4800 iters):
TASK=Mearmrl-Reach-Student-v0 NUM_ENVS=512 MAX_ITERATIONS=2000 docker compose up

# Interactive shell (bind-mounts your source automatically)
docker compose --profile dev run --rm dev

# One-off command inside the dev container
docker compose --profile dev run --rm dev \
    /IsaacLab/isaaclab.sh -p scripts/list_envs.py
```

Named volumes persist the Isaac Sim caches (shaders, GL, ComputeCache) across
runs, so only the first run pays the warm-up cost. `docker compose up` leaves
the exited container around — run `docker compose down` to clean up, or use
`docker compose run --rm mearmrl` for a self-cleaning one-off.

### Option 2: the `dev_local.sh` wrapper

```bash
bash scripts/dev_local.sh --task=Template-Mearmrl-v0 --num_envs=64 --headless
bash scripts/dev_local.sh --shell
```

Minimal and ephemeral (no cache volumes); use whichever suits you.

## GUI / X11 (optional, not enabled by default)

To launch Isaac Sim's viewer from the container, add X11 mounts to your
`docker run` / compose invocation and allow local connections on the host:

```bash
xhost +   # on the host (restrict with 'xhost +local:' if you prefer)
docker compose --profile dev run --rm \
    -e DISPLAY -v "$HOME/.Xauthority:/root/.Xauthority" \
    -v /tmp/.X11-unix:/tmp/.X11-unix dev
```

This works because the base image is the full `nvcr.io/nvidia/isaac-sim`
runtime. (Note: NVIDIA's pre-built `nvcr.io/nvidia/isaac-lab` image is
headless-only and would not support this.)

## Build the Singularity/Apptainer SIF (HPC)

On the cluster login node, after `docker build`:

```bash
bash docker/cluster/build_sif.sh local "$HOME"   # writes ~/mearmrl_local.sif
```

This requires Docker (it reads the local daemon image). The SIF is read-only
by Apptainer default, so students cannot modify it.

### Build the SIF without Docker

`docker/cluster/apptainer-standalone.def` builds the full image directly from
`nvcr.io` with Apptainer only — no Docker at any point:

```bash
export APPTAINER_DOCKER_USERNAME='$oauthtoken'
export APPTAINER_DOCKER_PASSWORD='<NGC API key>'
apptainer build ~/mearmrl_local.sif docker/cluster/apptainer-standalone.def
```

Run from the repo root (the `%files` section pulls your `source/MeArmRL` and
`scripts` into the image). The definition mirrors `docker/Dockerfile` — keep
them in sync when upgrading. See `slurm/README.md` for the three
delivery paths (laptop build + upload / docker save + convert / login-node
build).

## Testing

```bash
# Static checks + build + list registered tasks
bash scripts/test_container.sh

# Also run a zero-agent smoke test (requires an NVIDIA GPU)
bash scripts/test_container.sh --full

# Skip the build (if the image is already built)
bash scripts/test_container.sh --skip-build
```

**A GPU is required even for `list_envs`.** `list_envs.py` launches
`AppLauncher(headless=True)`, which boots Isaac Sim, and Isaac Sim hard-requires
an NVIDIA driver (`libcuda.so.1`) even in headless mode — on a GPU-less
machine, the PhysX fabric extension fails during app startup and the task
table never prints. The pipeline's no-GPU value is the static checks + the
build itself; anything that instantiates the sim app belongs on a GPU box.

## Build gotchas (historical)

Four non-obvious fixes are baked into the Dockerfile. Re-introducing any of
these mistakes will produce a broken or failing build:

1. **`USER root`** — the `nvcr.io/nvidia/isaac-sim:5.x` images default to a
   non-root user (`isaac-sim`, uid 1234). Without `USER root`, `apt-get`
   fails with exit 100 (`Permission denied` on `/var/lib/apt/lists`).
2. **`ENV TERM=xterm`** — the base image sets `TERM=dumb`, which breaks
   `tput` inside `isaaclab.sh` (`'ansi+tabs': unknown terminal type`).
3. **`flatdict==4.0.1` pre-install with `--no-build-isolation`** — flatdict
   is sdist-only and its `setup.py` imports `pkg_resources`, which pip's
   isolated build env doesn't have (latest setuptools ≥82 removed it). The
   pinned `setuptools<82` + `--no-build-isolation` pair fixes it.
4. **Explicit core-isaaclab install after `--install`** — `isaaclab.sh
   --install` uses `find -exec` whose exit status comes from the LAST
   invocation, silently swallowing an early failure (this is how the core
   package can go missing while the build "succeeds"). The explicit install
   makes any such failure fail the build.

## Pinning

The Isaac Sim version and Isaac Lab tag are pinned in `docker/Dockerfile`
(`ISAACSIM_VERSION` build arg defaults to `5.1.0`; Isaac Lab tag `v2.3.2`).
Bump both together when upgrading — Isaac Lab releases track specific Isaac
Sim versions.

Notes:

- The base image and the MeArmRL layer run as root (usual for these containers).
- `ACCEPT_EULA=Y` and `PRIVACY_CONSENT=Y` are baked in. Set `PRIVACY_CONSENT`
  to `N` if you want to opt out of telemetry.
- The editable install of MeArmRL means the bind-mounted source at
  `/MeArmRL/source/MeArmRL` overrides the baked-in copy at runtime.