# Week 1 Lab — Reward Engineering with Isaac Lab

**Estimated time:** 2-4 hours (including training waits)
**Deliverables:** answers to the questions below + one modified reward config + one training run (log directory)

> **Golden rule: you edit exactly two files — `reach_student_cfg.py` and
> `skrl_ppo_student_cfg.yaml`.**
> The robot, scene, sensors, actions, observations, commands and terminations
> are inherited from the base reach environment and are not yours to change.
> `scripts/check_week01.py` verifies this before you submit.

## Learning goals

By the end of this lab you can:

1. Run an Isaac Lab RL environment inside the course container.
2. Read a `RewardsCfg` and explain what each reward term does, its sign, and
   how per-step values become the logged episode return.
3. Run controlled one-variable reward experiments and predict/observe their
   effect on training (reward hacking, jitter, sparse signal).
4. Tune a PPO hyperparameter in the skrl config and see its effect on the
   reward curve.
5. Write a new reward function, register it as a `RewTerm`, and train a PPO
   policy with skrl to compare against a baseline.

## Setup

Everything runs inside the course Docker image (see `docker/README.md` for the
one-time build). From the repo's `docker/` directory:

```bash
docker compose --profile dev run --rm dev        # interactive shell
# inside the container:
/IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task=Mearmrl-Reach-v0 --num_envs=4 --headless
```

Success looks like: ~2-3 min of startup (shader warm-up on first run), then
observation/action space printouts and a clean stepping loop. `Ctrl+C` to stop.

> The arm task drives 4 joints (`revolute_base`, `revolute_left`,
> `revolute_right`, `rev_end_effector1`); the other 10 linkage joints are
> driven by PhysX mimic constraints. Articulation = 14 joints total.

## Part 1 — Anatomy of a reward (30 min)

Open `reach_student_cfg.py` and the reward implementations in
`source/MeArmRL/MeArmRL/tasks/manager_based/mearmrl/mdp/rewards.py`. The
baseline `RewardsCfg` (inherited from `mearmrl_reach_env_cfg.py`) has five
terms:

| Term | Function | Weight | Sign |
|---|---|---|---|
| `end_effector_position_tracking` | `position_command_error` | `-0.2` | penalty |
| `end_effector_position_tracking_fine_grained` | `position_command_error_tanh` | `+0.1` (std `0.1`) | bonus |
| `end_effector_orientation_tracking` | `orientation_command_error` | `-0.1` | penalty |
| `action_rate` | `action_rate_l2` | `-0.0001` | penalty |
| `joint_vel` | `joint_vel_l2` | `-0.0001` | penalty |

Answer:

**Q1.** Which terms are penalties and which are bonuses, and why does Isaac Lab
use a negative weight for a penalty instead of a positive weight for a
"not-failing" reward? What would happen to the policy if you flipped the sign
of `end_effector_position_tracking`?

**Q2.** Each term returns a per-env, per-step scalar. How do these become the
"reward" you see in the training logs? (Hint: `RewTerm` weight, then the
episode return summed over steps.) Why does a shaping bonus like the tanh term
matter when the L2 penalty already pulls the arm toward the target?

**Q3.** `CurriculumCfg` quietly ramps two penalty weights at step 4500
(`action_rate` → `-0.005`, `joint_vel` → `-0.001`). Why would the task want
smooth, slow motion *later* in training but not from the start?

## Part 2 — One-variable calibration (45 min)

In `StudentRewardsCfg` (rewards) or `skrl_ppo_student_cfg.yaml` (PPO
hyperparameters), change **one** thing per run and record the result. Use the
same fixed budget for each run:

```bash
/IsaacLab/isaaclab.sh -p scripts/skrl/train.py --task=Mearmrl-Reach-Student-v0 \
    --num_envs=1024 --headless --seed=42 --max_iterations=2000
```

| Run | Change | Predicted effect | Observed effect |
|---|---|---|---|
| A | `end_effector_position_tracking` weight `-0.2` → `-2.0` | | |
| B | `action_rate` and `joint_vel` weights → `0.0` | | |
| C | `end_effector_position_tracking_fine_grained` std `0.1` → `0.02` | | |
| D | `agent.learning_rate` `5.0e-04` → `5.0e-03` (in `skrl_ppo_student_cfg.yaml`) | | |

Before each run, write your prediction in the table. After each run, note what
the reward curve actually did. Watch for: reward hacking (a term dominating
everything), jitter (no smoothness penalty), a stalled curve (signal too
sparse to learn from), and an unstable curve (learning rate too high). Commit
your changes on a branch named `week01-<yourname>`.

## Part 3 — Write a new reward term (45 min)

The file `reach_student_cfg.py` contains a `reach_progress_bonus` skeleton with
three blanks. Fill them in so the function rewards the *decrease* in
end-effector distance to the target since the previous step:

1. **BLANK 1** — progress = previous distance − current distance (positive when
   the arm gets closer).
2. **BLANK 2** — zero out progress on the first step of a new episode, where the
   cached distance still belongs to the previous episode. A freshly reset env
   has `env.episode_length_buf == 1` at reward time.
3. **BLANK 3** — remember the current distance for the next step.

Then register it by uncommenting the example `RewTerm` in `StudentRewardsCfg`
(choose your own weight). Verify everything is wired up:

```bash
/IsaacLab/isaaclab.sh -p scripts/check_week01.py --smoke
```

This compares your config against the base env (only `rewards`/`curriculum`
may differ), checks that every curriculum term still targets an existing
reward term, and runs a short zero-action smoke test to catch shape/`None`
bugs in your function. It must exit 0.

> Alternative if you want a different shape: a *success bonus* that returns
> `+1.0` when the distance is below `0.02` m and `0.0` otherwise. Same
> registration, different function body.

## Part 4 — Train, compare, verify (45 min, mostly waiting)

Train your modified config with the fixed budget:

```bash
/IsaacLab/isaaclab.sh -p scripts/skrl/train.py --task=Mearmrl-Reach-Student-v0 \
    --num_envs=1024 --headless --seed=42 --max_iterations=4800
```

> **Running on the cluster (optional).** If your GPU is on an HPC cluster, the
> same training runs via SLURM — see `slurm/README.md` for the full guide
> (image delivery, per-user setup, troubleshooting). The short version, from
> your checkout on the login node:
>
> ```bash
> TASK=Mearmrl-Reach-Student-v0 sbatch slurm/train.sbatch
> ```
>
> Logs/checkpoints land in `/scratch/$USER/mearmrl-logs/jobs/<JOB_ID>/` instead
> of `logs/skrl/...`; the sbatch defaults (`Mearmrl-Reach-v0`, 1024 envs, 4800
> iterations) match this lab's budget, so only `TASK` needs overriding.

Checkpoints land in `logs/skrl/mearmrl-reach/<timestamp>/checkpoints/`
(`agent_*.pt`, `best_agent.pt`); metrics go to the TensorBoard event file in
the same run directory (open with `tensorboard --logdir logs/skrl`).

Compare against the baseline run (same budget, `--task=Mearmrl-Reach-v0`).
Record: mean total reward at the start vs. the end, and which reward term
dominates the improvement (or the penalty). One paragraph — graded on your
reasoning, not on the final reward number.

Verify the policy loads and steps:

```bash
/IsaacLab/isaaclab.sh -p scripts/skrl/play.py --task=Mearmrl-Reach-Student-v0 \
    --num_envs=8 --checkpoint=/MeArmRL/logs/skrl/mearmrl-reach/<timestamp>/checkpoints/agent_4800.pt \
    --headless
```

(With X11 mounts and without `--headless` you can watch the arm in a viewport.)

## Submission checklist

- [ ] Branch `week01-<yourname>` with your changes to `reach_student_cfg.py`
      and `skrl_ppo_student_cfg.yaml`
- [ ] Answers to Q1-Q3 (a few sentences each)
- [ ] Part 2 results table (predicted vs. observed for runs A, B, C, D)
- [ ] Your `reach_progress_bonus` function + registered `RewTerm`
- [ ] `scripts/check_week01.py` exits 0 (paste the output) — this includes the
      git-diff allowlist check: only `reach_student_cfg.py` and
      `skrl_ppo_student_cfg.yaml` may be modified (add your answers file with
      `--allow <path>` if it is tracked)
- [ ] Your training `logs/skrl/mearmrl-reach/<timestamp>/` directory (params +
      event file; checkpoints can be excluded if large)
- [ ] One paragraph interpreting your reward curve vs. the baseline

## Troubleshooting

- **`/IsaacLab/isaaclab.sh: no such file`** — you are on the host, not in the
  container. Prefix with `docker compose --profile dev run --rm dev`.
- **`libcuda.so.1: cannot open shared object file`** — no GPU passed through.
  The container needs `--gpus all` (compose handles it); the driver must work
  on the host (`nvidia-smi`).
- **`check_week01.py` reports a `[FROZEN]` section** — you edited something
  outside the two student files. Revert it (`git checkout -- <file>`).
- **`[CURRICULUM]` error** — you renamed/removed a reward term that a
  curriculum term targets. Keep the term name or update the curriculum.
- **`[SKRL]` error** — `skrl_ppo_student_cfg.yaml` is malformed or you changed
  `agent.experiment.directory`. Fix the YAML or revert that line.
- **`[GIT]` error** — the diff against `master` shows files other than the two
  student files, or you never modified them. Commit your work on a branch and
  revert any edits outside the student files.
- **Training crashes at ~step 4500** — a curriculum term points at a reward
  term that no longer exists. Run `check_week01.py` to find it.
- **Everything else**: see `docker/README.md` and the README's arm section.
