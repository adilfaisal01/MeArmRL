# AGENTS.md

Guidance for working in this repo. Read `README.md`, `docker/README.md`, and `slurm/README.md` for full context; this file captures the non-obvious parts.

## What this is

Isaac Lab 2.3.2 extension template for a MeArm robot arm, on Isaac Sim 5.1.0. The runtime is a **shared, frozen Docker image** `ghcr.io/adilfaisal01/mearmrl:0.1.0` (maintainer-built; students never build/push it). Students bind-mount `source/MeArmRL` over `/workspace/isaaclab/source/MeArmRL` at runtime, so reward/env edits take effect without rebuilding.

## Running code

- The Python entrypoint is `isaaclab.sh -p` (the Isaac Sim Python), **not** `python`, unless Isaac Lab is installed in a venv. Inside the container it's `/workspace/isaaclab/isaaclab.sh -p scripts/skrl/train.py ...`.
- Local dev: `bash scripts/dev_local.sh --task=Template-Mearmrl-v0 --num_envs=64 --headless` (bind-mounts `source/MeArmRL` + `logs/`, requires GPU + NVIDIA Container Toolkit). `--shell` drops into a bash.
- Install in editable mode: `python -m pip install -e source/MeArmRL` (the package lives under `source/`, not the repo root).

## Registered tasks

`Template-Mearmrl-v0` and `Mearmrl-Reach-v0` are registered in `source/MeArmRL/MeArmRL/tasks/manager_based/mearmrl/__init__.py`. `scripts/list_envs.py` hardcodes the `Template-` keyword filter — if you rename a task, update that filter.

## Container image (maintainer only)

- The Dockerfile layers on `isaac-lab-base`, built from the local Isaac Lab checkout at `/mnt/E/IsaacLab`: `cd /mnt/E/IsaacLab/docker && python3 container.py build base`. That base must exist locally before `docker build`.
- Image is ~11 GB (Isaac Sim itself dominates — user code is ~1.5 MB). GHCR push needs a PAT with `write:packages`; the package **must be public** (private free tier is 500 MB, public is unlimited). Registry is GHCR, not Docker Hub.
- The extension is installed editable, so a bind-mounted `source/MeArmRL` overrides the baked-in copy.

## Container test pipeline

`bash scripts/test_container.sh` (static `bash -n` checks + build + `list_envs`; no GPU needed). `--full` adds a zero-agent GPU smoke test; `--skip-build` reuses the existing image. There is **no `pytest` suite** and `tests/` is gitignored — don't look for one.

## SLURM / HPC (pyxis/enroot)

- Single node: `sbatch slurm/train.sbatch`. Multi-node: `sbatch --nodes=N --gpus-per-node=1 slurm/train-multinode.sbatch` (torchrun + NCCL).
- Per-job overrides via env: `TASK= NUM_ENVS= MAX_ITERATIONS= SEED= CHECKPOINT= sbatch slurm/train.sbatch`.
- Submit from your checkout: the sbatch scripts use `$SLURM_SUBMIT_DIR` as the bind-mount source.
- Per-user scratch (`/scratch/$USER/mearmrl-logs`, `/scratch/$USER/isaac-sim-cache`) is created idempotently by `slurm/env.sh`. The container runs with `--container-remap-root --container-user=0`, which remaps container root to the host user — **never `chown`** scratch dirs.

## Lint / format

- ruff + ruff-format via pre-commit: `pre-commit run --all-files`. Config: line-length 120, target py310, custom isort sections (`omniverse-extensions`, `isaaclab*`, then first-party). `__init__.py` ignores `F401`.
- Pyright is configured in `pyproject.toml` (`reportMissingImports = "none"`) but is **disabled in pre-commit** (VPN hang) — don't re-enable it.

## Known broken hook

The `insert-license` pre-commit hook references `.github/LICENSE_HEADER.txt`, but **`.github/` does not exist** in this repo (no `LICENSE` file is tracked either). `pre-commit run --all-files` will fail that hook until `.github/LICENSE_HEADER.txt` is added or the hook block is removed. The Apache-2.0 mimic hook (`source/isaaclab_mimic/...`) also targets paths that don't exist here.

## Repo quirks

- `tests/` is gitignored. All `*.usd*` files are gitignored **except** `source/MeArmRL/MeArmRL/robot_description/mate_connectors/isaac/assets/mate_connectors.usd` (deliberately committed).
- Top-level `.vscode/` is tracked (tasks/templates/`setup_vscode.py`); per-user `.vscode/settings.json` is not.
- `source/MeArmRL/MeArmRL.egg-info/` is an editable-install artifact (gitignored).
