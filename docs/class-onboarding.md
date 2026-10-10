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
scripts\lab_setup.ps1              # Windows PowerShell
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
3. **Ignored venv.** `.venv-lab/` is gitignored.

### Still recommended (not done)

* **`network_mode: host` in `docker/docker-compose.yaml` is Linux-only** and makes
  `docker compose up` fail on Docker Desktop (macOS *and* Windows). It is not
  needed for headless training; move it behind an override file.
* **An OS-aware `doctor` script** that checks driver / toolkit / GPU visibility /
  NGC auth and prints the exact next command, refusing clearly on macOS.

## 5. Tier 3 — where the outer loop lives (decision)

| Option | Student friction | Cross-OS | Notes |
| --- | --- | --- | --- |
| **A. Per-student build** (today) | 15–20 min build + NGC login + Docker + GPU | all three | license-clean; slowest ramp |
| **B. Prebuilt image in a registry** | `docker pull` only | all three | biggest win; **licensing**: see below |
| **C. Offline tarball** (`docker save`) | `docker load`, no network | all three | same licensing question; ~5 GB via the LMS |
| **D. Managed cloud GPU** | click a template | any OS **incl. macOS** | Brev.dev / RunPod / Lambda / department server — the only route for Macs |
| **E. One classroom GPU box** | SSH / remote desktop | any OS incl. macOS | often the most reliable for a mixed class |

**Licensing caveat (verify, not legal advice).** A derived image built from
`nvcr.io/nvidia/isaac-sim` still contains NVIDIA's layers. NGC images are under
NVIDIA's Software License Agreement, and redistribution is generally restricted.
A **private** course registry (GHCR private, or a university registry) accessible
to enrolled students is the pragmatic middle ground; public redistribution should
be cleared first. Option A stays entirely license-clean.

## 6. Verification

```bash
bash -n scripts/lab_setup.sh scripts/check_week01.py   # static shell checks (sh only)
python -m pytest labs/week01_intro_isaaclab/harness -q  # 8 passed, 6 skipped on a clean tree
python scripts/check_week01.py --static                # static lab checks
```

On a clean checkout the harness reports `8 passed, 6 skipped` (the 5 student tests
skip until the shaping term is implemented; the parity test skips without Isaac
Lab). `check_week01.py --static` reports "none of the student files were modified"
until the student starts work — expected.

## 7. Open decisions

1. **Distribution channel** (Tier 3): registry vs per-student build vs tarball, and
   whether the licence permits a private registry.
2. **macOS / no-GPU students**: is "inner loop locally, training on cloud" the
   intended split, or should a cloud recipe be scripted into the repo?
3. **Week-1 skeleton**: blanks 2 and 3 of `reach_progress_bonus` are *already
   functionally correct*, yet carry `TODO(week01)` markers. That is confusing for
   students and defeats comment-based tooling. Either make the blanks genuinely
   empty, or relabel them "verify you understand why this is right". Not changed
   here because it alters the assignment itself.
4. **Add a `network_mode: host` override** (see §4) and the `doctor` script.
5. **Optional tiny CPU devcontainer** for a fully reproducible inner loop,
   deliberately separate from the GPU image.
