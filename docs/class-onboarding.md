# Onboarding a mixed-OS class: the GPU-free inner loop

**Status:** design + shipped Tier 1/2. Tier 3 (distribution) is a decision, not yet implemented.
**Audience:** course maintainers.
**Related:** `docker/README.md` (GPU container), `slurm/README.md` (HPC), `labs/week01_intro_isaaclab/harness/README.md`.

## TL;DR

Isaac Sim needs Linux + an NVIDIA GPU + CUDA — that is not something packaging can
change. So instead of trying to make *everything* run everywhere, split the class
into two loops:

* a **GPU-free inner loop** (edit → reason → test reward code) that runs on any
  laptop, including macOS; and
* a **GPU outer loop** (actually training) that runs on whatever GPU access the
  class has — container, HPC, or cloud.

Tier 1/2 of this plan ship the inner loop. Tier 3 is about where the outer loop
lives.

## 1. The constraint that cannot be engineered away

Isaac Sim 5.1 requires an NVIDIA driver (`libcuda.so.1`) **even headless** — the
PhysX/RTX engine will not start without it. The OS matrix is therefore really a
hardware matrix:

| Student machine | Runs Isaac Sim locally? | Path |
| --- | --- | --- |
| Linux + NVIDIA dGPU | yes | current Docker/HPC path |
| Windows 10/11 + NVIDIA dGPU | yes, via WSL2 | WSL2 + Windows CUDA driver + Docker Desktop (WSL2 backend) |
| Windows with no dGPU | no | cloud / classroom GPU |
| **macOS (Intel or Apple Silicon)** | **no** | **no CUDA, no NVIDIA — impossible locally** |
| Cloud / cluster GPU | yes | the existing SLURM paths |

The goal is therefore **not** "run on macOS". It is: make the Linux+GPU path one
command, make Windows painless, and give macOS/no-GPU students real coding time
plus a route to a GPU for the one run that needs it.

## 2. Two loops

| | Inner loop (shipped) | Outer loop |
| --- | --- | --- |
| Question | "Is my reward correct? What will it do?" | "Does the policy train?" |
| Needs | `torch` | Isaac Sim, NVIDIA GPU |
| Runs on | any OS, any laptop | container / HPC / cloud |
| Time | milliseconds | minutes |
| Entry point | `scripts/lab_setup.sh` | `docker compose up`, `sbatch`, cloud recipe |

Most of the Week-1 learning (read a reward, change a weight, write a new term,
predict the effect) lives in the inner loop.

## 3. Tier 1 — the GPU-free inner loop (shipped)

`labs/week01_intro_isaaclab/harness/` runs the lab's **real** reward code on CPU.
It needs only `torch`, `pyyaml`, `pytest`.

```bash
bash scripts/lab_setup.sh          # Linux / macOS / Git-Bash: venv + deps + checks
```

```powershell
scripts\lab_setup.cmd              # Windows (batch wrapper avoids ExecutionPolicy)
```

```bash
python -m pytest labs/week01_intro_isaaclab/harness -q
python labs/week01_intro_isaaclab/harness/playground.py
```

### How it stays honest

Rather than reimplementing the rewards (which would drift), the harness loads the
production sources:

* **`mdp/rewards.py`** is now importable with only `torch`. The four
  `isaaclab.utils.math` helpers it uses are tried first from Isaac Lab and fall
  back to `mdp/math_compat.py`, pure-torch copies taken verbatim from Isaac Lab
  `v2.3.2`. `test_math_parity.py` asserts the two agree on random inputs; it only
  runs where Isaac Lab is installed, so drift is caught in-container.
* **`reach_student_cfg.py`** is parsed and its top-level functions executed against
  a fake env, injecting `torch` and the math helpers. The student still edits
  exactly the two lab files — there is no third copy to keep in sync.
* **`FakeEnv`** implements only the slice of the Isaac Lab API the rewards touch:
  `env.scene[name].data.{root_pos_w, root_quat_w, body_pos_w, body_quat_w,
  joint_pos, joint_vel}`, `env.command_manager.get_command(...)`, and
  `env.episode_length_buf`.

### What it catches (verified)

Running the harness against deliberately-broken copies of `reach_student_cfg.py`:

| Variant | stationary | closing | retreat | reset step | verdict |
| --- | --- | --- | --- | --- | --- |
| correct | 0.000 | +0.600 | -0.600 | 0.000 | OK |
| sign flipped | 0.000 | -0.600 | +0.600 | 0.000 | flagged |
| no reset guard | 0.000 | +0.600 | -0.600 | **+0.950** | flagged |
| untouched skeleton | 0.000 | 0.000 | 0.000 | 0.000 | skipped |

The student tests **gate behaviorally** (does the term return a nonzero bonus for
a real gap?), not by scanning for `TODO` markers.

## 4. Tier 2 — cross-OS fixes (shipped)

1. **Line endings.** `.gitattributes` now forces LF for `*.sh`, `*.ps1`,
   `*.sbatch`, `Dockerfile`, `docker/**`, `*.py`, `*.yaml`, `*.toml`, `*.json`,
   `*.md`, `*.txt`. Without this, Git for Windows (`core.autocrlf=true`) checks
   those out as CRLF, and the container fails with a confusing
   `exec format error` / `: not found`. (Existing checkouts may need a one-time
   `git add --renormalize .`.)
2. **Static lab checks without the simulator.** `scripts/check_week01.py --static`
   runs with no Isaac Sim, no GPU, no container: git allowlist, skrl YAML, and AST
   checks on the reward functions (reward-term wiring, plus advisory warnings for
   per-step footguns like `.item()` or Python loops). The full in-container check
   is unchanged.
3. **An OS-aware `doctor` script.** `python scripts/doctor.py` detects the platform
   (Linux / WSL2 / macOS / Windows), then checks the NVIDIA driver, Docker CLI and
   daemon, the NVIDIA container runtime, the built image, free disk space, NGC
   auth, and the CPU harness -- printing the exact fix command for whatever is
   missing. On macOS it exits `2` with cloud/CPU guidance, because the GPU path is
   impossible there. `--json` makes it easy to survey a whole class; `--gpu-smoke`
   additionally runs `nvidia-smi` inside the built image.
4. **Docker Desktop networking.** `network_mode: host` was removed from
   `docker/docker-compose.yaml` (Linux-only; Docker Desktop rejects it) -- the
   services use the default bridge network, and nothing here needed host
   networking. Add it back per service only if you need it on Linux. `doctor.py`
   no longer warns about it.
5. **Repo-wide yamllint policy.** `.yamllint` raises the line limit to 126 so the
   126-column copyright header (which must stay verbatim) and the repo's ~120-column
   style pass; `document-start` is disabled to match existing files.
6. **Ignored venv.** `.venv-lab/` is gitignored.
7. **Windows entry point.** `scripts\lab_setup.cmd` runs the setup via PowerShell
   with `-ExecutionPolicy Bypass` -- a fresh Windows machine blocks `.ps1` files
   ("running scripts is disabled"), and the `.ps1` previously also called
   `Activate.ps1`, which is policy-blocked too. It now calls the venv interpreter
   directly and prefers the `py -3` launcher. `doctor.py` prints an OS-specific
   Python install command when the interpreter is too old, and the CPU harness
   installs `numpy` (torch warned about its absence).

## 5. Tier 3 — decided: per-student build + university HPC cluster

**Decision (2026-10):** students **build the image on their own machine** and run
**training on the university-owned supercomputing cluster** via
Singularity/Apptainer + SLURM. No course image registry is used, so the NGC
redistribution question below is moot: every student pulls the nvcr.io base
themselves, which the licence permits.

Consequences:

- The build (15–20 min, ~11 GB, NGC login) happens once per machine and needs **no
  GPU**, so it can be done on a laptop even though training cannot run there.
- Training uses the existing machinery: `docker build` -> SIF
  (`bash docker/cluster/build_sif.sh local "$HOME"`, or the Docker-free
  `docker/cluster/apptainer-standalone.def`) -> `sbatch slurm/train.sbatch`.
  `train.sbatch` bind-mounts the student's `source/MeArmRL`, so **lab config edits
  take effect without rebuilding the SIF**.
- **Results persist and are retrievable.** The job dir is mounted at
  `/MeArmRL/logs` (where `train.py` writes -- mounting only `/logs` silently lost
  every checkpoint to the container's ephemeral layer until 2026-10).
  `scripts/pull_results.sh` rsyncs a run back to `logs/remote/<JOB_ID>/` and prints
  the `play.py` command; `VIDEO=1` on a cluster job records videos.
- **macOS / no local GPU:** the inner loop runs locally (CPU harness); training
  goes to the cluster. Building the SIF on Apple Silicon means building an amd64
  image under emulation -- build on a lab Linux machine or the login node instead.

Options considered (A + cluster chosen):

| Option | Student friction | Cross-OS | Notes |
| --- | --- | --- | --- |
| **A. Per-student build (chosen)** | 15–20 min build + NGC login + Docker + GPU | all three | license-clean; slowest ramp |
| B. Prebuilt image in a registry | `docker pull` only | all three | biggest win, but the licence restricts redistribution |
| C. Offline tarball (`docker save`) | `docker load`, no network | all three | same licence question; ~5 GB via the LMS |
| D. Managed cloud GPU | click a template | any OS incl. macOS | Brev.dev / RunPod / Lambda; not used (university owns a cluster) |
| E. One classroom GPU box | SSH / remote desktop | any OS incl. macOS | not used (university owns a cluster) |

**Licensing note (why not a registry).** A derived image built from
`nvcr.io/nvidia/isaac-sim` still contains NVIDIA's layers; NGC images are under
NVIDIA's Software License Agreement and redistribution is generally restricted.
Keeping the build per-student avoids the question entirely.

## 6. Verification

```bash
python scripts/doctor.py                               # platform preflight (see exit codes above)
python -m pytest labs/week01_intro_isaaclab/harness -q  # 8 passed, 6 skipped on a clean tree
python scripts/check_week01.py --static                # static lab checks
bash -n scripts/lab_setup.sh                           # shell syntax
```

On a clean checkout the harness reports `8 passed, 6 skipped` (the 5 student tests
skip until the shaping term is implemented; the parity test skips without Isaac
Lab). `check_week01.py --static` reports "none of the student files were modified"
until the student starts work — expected.

## 7. Decisions & open items

Resolved: distribution = **per-student build, no registry** (§5); training =
**university HPC cluster, Singularity + SLURM** (§5); Week-1 skeleton = **two
genuine blanks, option (b)** (`reach_progress_bonus`); plus the doctor,
`network_mode`, and yamllint items in §4.

Still open:

- **Optional tiny CPU devcontainer** for a fully reproducible inner loop,
  deliberately separate from the GPU image.
