# Week-1 lab harness (CPU only — no GPU, no Isaac Sim, no container)

This directory lets you **write and test reward code on any laptop**, before you
ever touch a GPU. It needs only `torch`, `pyyaml`, and `pytest`, and it runs on
Linux, Windows, and macOS (including Apple Silicon).

The heavy `mearmrl` container is still required to *train* — nothing here
replaces Isaac Sim. But the whole "read the reward, change a weight, write a new
term" loop, which is where most of the learning happens, does not need a
simulator.

## Quick start

```bash
bash scripts/lab_setup.sh          # Linux / macOS / Git-Bash
```

```powershell
scripts\lab_setup.ps1              # Windows PowerShell
```

Either one creates a local `.venv-lab`, installs the CPU dependencies, runs the
static lab checks, and runs these tests. Or do it by hand:

```bash
python -m pip install -r labs/week01_intro_isaaclab/harness/requirements.txt
python -m pytest labs/week01_intro_isaaclab/harness -q
python labs/week01_intro_isaaclab/harness/playground.py
```

## What is here

| File | Purpose |
| --- | --- |
| `labkit.py` | The fake env and the loaders that import the **real** reward code. |
| `test_stock_rewards.py` | Shape/boundary tests for the baseline tracking reward terms (Part 1). |
| `test_student_rewards.py` | Behavioral tests for `reach_progress_bonus`, including the reset trap. Skips while the term still returns zero for every gap. |
| `test_math_parity.py` | Confirms the CPU math fallbacks match Isaac Lab exactly. Only runs in the container. |
| `playground.py` | Prints/plots reward-vs-error curves and your shaping term along a scripted trajectory. |

## How it works (and why it is trustworthy)

Rather than re-implementing the rewards, the harness loads the **actual
production source**:

* `mdp/rewards.py` is imported directly. It is written so that on a laptop —
  where `isaaclab` is absent — it falls back to `mdp/math_compat.py`, pure-torch
  copies of the four `isaaclab.utils.math` helpers it uses. In the container it
  uses Isaac Lab's own implementations. `test_math_parity.py` keeps the two in
  lockstep.
* `reach_student_cfg.py` is parsed and its top-level functions are executed
  against a fake env. **You still edit exactly the two lab files** — there is no
  third copy to keep in sync.

The fake env (`labkit.FakeEnv`) implements only the slice of the Isaac Lab API
the reward terms touch: `env.scene[name].data.{root_pos_w, root_quat_w,
body_pos_w, body_quat_w, joint_pos, joint_vel}`, `env.command_manager.
get_command(...)`, and `env.episode_length_buf`.

## What is *not* covered

Anything that needs the physics engine: whether the arm can physically reach a
target, contact/mimic-joint behavior, actuator gains, and the training run
itself. Those still require the GPU container — see `docker/README.md` and
`slurm/README.md`.
