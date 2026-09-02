# Week 1 Lab — Introduction to Isaac Lab: the MATE Arm

**Estimated time:** 2-4 hours (including training waits)
**Deliverables:** answers to the questions below + one modified reward config + one training run (log directory)

## Learning goals

By the end of this lab you can:

1. Run an Isaac Lab RL environment inside the course container.
2. Explain how a `ManagerBasedRLEnv` is assembled: scene, actions,
   observations, commands, rewards, terminations, curriculum.
3. Change the task (reward shaping / command sampling) and observe the effect
   on training.
4. Train a PPO policy with skrl and read the logged metrics.

## Setup

Everything runs inside the course Docker image (see `docker/README.md` for the
one-time build). From the repo's `docker/` directory:

```bash
docker compose --profile dev run --rm mearmrl     # interactive shell
# inside the container:
/IsaacLab/isaaclab.sh -p scripts/zero_agent.py --task=Mearmrl-Reach-v0 --num_envs=4 --headless
```

Success looks like: ~2-3 min of startup (shader warm-up on first run), then
observation/action space printouts and a clean stepping loop. `Ctrl+C` to stop.

> The arm task drives 4 joints (`revolute_base`, `revolute_left`,
> `revolute_right`, `rev_end_effector1`); the other 10 linkage joints are
> driven by PhysX mimic constraints. Articulation = 14 joints total.

## Part 1 — Find your way around (30 min)

The env config lives in
`source/MeArmRL/MeArmRL/tasks/manager_based/mearmrl/mearmrl_reach_env_cfg.py`.
Read it with the questions below in mind, then answer:

**Q1.** The `MearmrlReachSceneCfg` spawns one `robot` articulation. Where does
the robot USD path come from, and why does the config resolve it relative to
its own file location? (Hint: `mate_connectors_config.py`, `_USD_PATH`.)

**Q2.** Draw (on paper or ASCII) the data flow for one `env.step()`:
action tensor → action term → joint targets → physics → observations/rewards.
Name the manager that owns each stage.

**Q3.** `CommandsCfg` resamples a target pose every 4 s. Which observation
term carries that target to the policy, and what shape does the final
observation vector have? (Count the terms in `ObservationsCfg`; joint counts
are 4 actuated + 1 action repeat + 7 pose values.)

## Part 2 — Shape the task (45 min)

Open `mearmrl_reach_env_cfg.py` and make **one** of the following changes
(your choice; note the original values first):

| Option | Change | What to watch |
|---|---|---|
| A | Change `end_effector_position_tracking` weight from `-0.2` to `-2.0` | Does the policy prioritize reaching over smoothness? |
| B | Tighten the pose command ranges to `pos_x=(0.1, 0.2)`, `pos_z=(0.2, 0.3)` | Does a smaller workspace make learning faster? |
| C | Shorten `episode_length_s` from `12.0` to `6.0` | Effect on episode reward variance and reset frequency |

Commit your change on a branch named `week01-<yourname>`.

## Part 3 — Train and read the curve (45 min, mostly waiting)

```bash
/IsaacLab/isaaclab.sh -p scripts/skrl/train.py --task=Mearmrl-Reach-v0 \
    --num_envs=1024 --headless --seed=42 --max_iterations=4800
```

Checkpoints land in `logs/skrl/mearmrl-reach/<timestamp>/checkpoints/`
(`agent_*.pt`, `best_agent.pt`); metrics go to the TensorBoard event file in
the same run directory (open with `tensorboard --logdir logs/skrl`).

Record: mean total reward at the start vs. the end, and which reward term
dominates the improvement (or the penalty). One paragraph.

## Part 4 — Verify the policy (15 min)

```bash
/IsaacLab/isaaclab.sh -p scripts/skrl/play.py --task=Mearmrl-Reach-v0 \
    --num_envs=8 --checkpoint=/MeArmRL/logs/skrl/mearmrl-reach/<timestamp>/checkpoints/agent_4800.pt \
    --headless
```

Confirm it loads and steps without errors. (With X11 mounts and without
`--headless` you can watch the arm in a viewport.)

## Submission checklist

- [ ] Branch `week01-<yourname>` with your config modification
- [ ] Answers to Q1-Q3 (a few sentences each; the ASCII diagram for Q2)
- [ ] Your training `logs/skrl/mearmrl-reach/<timestamp>/` directory (params +
      event file; checkpoints can be excluded if large)
- [ ] One paragraph interpreting your reward curve

## Troubleshooting

- **`/IsaacLab/isaaclab.sh: no such file`** — you are on the host, not in the
  container. Prefix with `docker compose --profile dev run --rm mearmrl`.
- **`libcuda.so.1: cannot open shared object file`** — no GPU passed through.
  The container needs `--gpus all` (compose handles it); the driver must work
  on the host (`nvidia-smi`).
- **Everything else**: see `docker/README.md` and the README's arm section.