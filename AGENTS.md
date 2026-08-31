# AGENTS.md

Guidance for working in this repo. Read `README.md`, `docker/README.md`, and `slurm/README.md` for full context; this file captures the non-obvious parts.

## What this is

Isaac Lab 2.3.2 extension template for a MeArm robot arm, on Isaac Sim 5.1.0. The container stack is **built locally by students/maintainers alike** — there is no shared registry or maintainer push step. `docker/Dockerfile` starts from `nvcr.io/nvidia/isaac-sim:5.1.0`, clones Isaac Lab tag `v2.3.2` from GitHub, installs it into the Isaac Sim Python, then adds the MeArmRL extension and the RL stack. Students bind-mount `source/MeArmRL` over `/MeArmRL/source/MeArmRL` at runtime, so reward/env edits take effect without rebuilding.

## Running code

- The Python entrypoint is `isaaclab.sh -p` (the Isaac Sim Python), **not** `python` or bare `pip` — the container has two Pythons (Isaac Sim's at `/isaac-sim/python.sh`, apt's at `/usr/bin/python3`) and only the former has Isaac Lab. Inside the container: `/IsaacLab/isaaclab.sh -p scripts/skrl/train.py ...`.
- Build the image: `docker login nvcr.io` (free NGC API key, one-time) then `docker build -t mearmrl:local -f docker/Dockerfile .`.
- Local dev, two equivalent routes: `bash scripts/dev_local.sh --task=Template-Mearmrl-v0 --num_envs=64 --headless` (minimal) or `docker compose --profile dev run --rm mearmrl` from `docker/` (persistent Isaac Sim cache volumes). `--shell` drops into a bash.
- Container layout: `/isaac-sim` (Isaac Sim runtime), `/IsaacLab` (Isaac Lab clone), `/MeArmRL` (this repo). The workspace root inside the container is `/MeArmRL`, not the old `/workspace/isaaclab`.
- Install in editable mode: `/IsaacLab/isaaclab.sh -p -m pip install -e source/MeArmRL`.

## Registered tasks

`Template-Mearmrl-v0` and `Mearmrl-Reach-v0` are registered in `source/MeArmRL/MeArmRL/tasks/manager_based/mearmrl/__init__.py`. `scripts/list_envs.py` hardcodes the `Template-` keyword filter — if you rename a task, update that filter.

## Container image (student-built, no registry)

- No GHCR/push workflow exists anymore; the image is built with a single `docker build` on whatever machine needs it. Pin `ISAACSIM_VERSION` (default `5.1.0`) and the Isaac Lab tag (`v2.3.2`) together when upgrading.
- `docker/docker-compose.yaml` defines two profiles: `dev` (interactive bash, bind-mounted source) and `train` (headless training), plus named volumes for the Isaac Sim caches.
- X11/GUI is documented in `docker/README.md` but not enabled by default (the base image supports it).
- BuildKit is required for the pip cache mount in the Dockerfile (`RUN --mount=type=cache`); Docker 23+ has it by default, older versions need `DOCKER_BUILDKIT=1`.

## Container test pipeline

`bash scripts/test_container.sh` (static `bash -n` checks + build + `list_envs`; no GPU needed). `--full` adds a zero-agent GPU smoke test; `--skip-build` reuses the existing image. There is **no `pytest` suite** and `tests/` is gitignored — don't look for one.

## SLURM / HPC

Three deployment paths (see `slurm/README.md` "Choosing a cluster backend"):

- pyxis/enroot (most academic clusters): get `~/mearmrl_local.sif` onto the cluster via one of THREE delivery paths (see `slurm/README.md`): (A) build the SIF on a laptop (`build_sif.sh` with Docker, or `docker/cluster/apptainer-standalone.def` with Apptainer only) and `rsync` it up; (B) `docker save` tarball uploaded, converted on the login node (needs login-node Docker); (C) build Docker image + SIF directly on the login node. Then `sbatch slurm/train.sbatch` (single) / `slurm/train-multinode.sbatch` (torchrun + NCCL). The sbatch scripts default `IMAGE` to `$HOME/mearmrl_local.sif` and fail loudly if it's missing; override with `MEARMRL_IMAGE=`.
- native Docker: `MEARMRL_BACKEND=docker sbatch slurm/train-docker.sbatch`; image reaches compute nodes via a self-hosted registry on the login node or `docker save/load` (both documented). Single-node only; runs as root (no remap-root), with `env.sh` loosening scratch perms to `777` when `MEARMRL_BACKEND=docker`.
- Per-job overrides via env: `TASK= NUM_ENVS= MAX_ITERATIONS= SEED= CHECKPOINT= MEARMRL_IMAGE= sbatch slurm/train.sbatch`.

- Submit from your checkout: the sbatch scripts use `$SLURM_SUBMIT_DIR` as the bind-mount source.
- Per-user scratch (`/scratch/$USER/mearmrl-logs`, `/scratch/$USER/isaac-sim-cache`) is created idempotently by `slurm/env.sh`. The pyxis scripts use `--container-remap-root --container-user=0`, which remaps container root to the host user — **never `chown`** scratch dirs.

## Lint / format

- ruff + ruff-format via pre-commit: `pre-commit run --all-files`. Config: line-length 120, target py310, custom isort sections (`omniverse-extensions`, `isaaclab*`, then first-party). `__init__.py` ignores `F401`.
- Pyright is configured in `pyproject.toml` (`reportMissingImports = "none"`) but is **disabled in pre-commit** (VPN hang) — don't re-enable it.

## Known broken hook

The `insert-license` pre-commit hook references `.github/LICENSE_HEADER.txt`, but **`.github/` does not exist** in this repo (no `LICENSE` file is tracked either). `pre-commit run --all-files` will fail that hook until `.github/LICENSE_HEADER.txt` is added or the hook block is removed. The Apache-2.0 mimic hook (`source/isaaclab_mimic/...`) also targets paths that don't exist here.

## Repo quirks

- `tests/` is gitignored. All `*.usd*` files are gitignored **except** `source/MeArmRL/MeArmRL/robot_description/mate_connectors/isaac/assets/mate_connectors.usd` (deliberately committed).
- Top-level `.vscode/` is tracked (tasks/templates/`setup_vscode.py`); per-user `.vscode/settings.json` is not.
- `source/MeArmRL/MeArmRL.egg-info/` is an editable-install artifact (gitignored).