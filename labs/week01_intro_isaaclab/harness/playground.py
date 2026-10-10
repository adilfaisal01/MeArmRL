# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward playground: explore the Week-1 rewards on CPU. No GPU, no container.

    python labs/week01_intro_isaaclab/harness/playground.py
    python labs/week01_intro_isaaclab/harness/playground.py --plot   # needs matplotlib

Use this to *predict before you train*: see how each baseline term scores as a
function of tracking error, and watch your own shaping term tick along a
scripted trajectory. Learning goal 3 of the lab ("predict, then observe") is a
lot cheaper here than on a GPU.
"""

from __future__ import annotations

import argparse

import torch
from labkit import Cfg, load_reward_module, load_student_module, make_env

ROBOT = Cfg("robot", body=0)


def stock_table() -> tuple[list[float], list[float], list[float]]:
    """Return (distances, position penalty, tanh bonus) for the baseline tracking terms."""
    rewards = load_reward_module()
    distances = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    penalties, bonuses = [], []
    for d in distances:
        env = make_env(ee_pos=[(d, 0.0, 0.0)], target=[(0.0, 0.0, 0.0)])
        penalties.append(rewards.position_command_error(env, "ee_pose", ROBOT).item())
        bonuses.append(rewards.position_command_error_tanh(env, 0.1, "ee_pose", ROBOT).item())
    return distances, penalties, bonuses


def scripted_student_rollout() -> list[tuple[int, float, float]]:
    """Run the student's shaping term along a trajectory, including one episode reset.

    Returns (step, distance_to_target, bonus) rows. The ee starts 1 m away, closes
    at 0.2 m/step, and at step 3 the episode resets and the ee spawns near the
    target -- the point where a missing first-step guard shows up as a spike.
    """
    namespace = load_student_module()
    fn = namespace.get("reach_progress_bonus")
    if fn is None:
        return []

    env = make_env(ee_pos=[(1.0, 0.0, 0.0)], target=[(0.0, 0.0, 0.0)], episode_length=5)
    rows = []
    script = [(5, 1.0), (6, 0.8), (7, 0.6), (1, 0.05), (2, 0.0)]
    for step, x in script:
        env.episode_length_buf = torch.tensor([step])
        env.scene["robot"].data.body_pos_w[0, 0, 0] = x
        bonus = fn(env, command_name="ee_pose", asset_cfg=ROBOT).item()
        rows.append((step, x, bonus))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Also render the curves (needs matplotlib).")
    args = parser.parse_args()

    print("Baseline tracking terms (target at the origin, root at world origin)\n")
    print(f"  {'dist (m)':>9} {'-0.2*||e||':>12} {'+0.1*tanh(e/0.1)':>19}")
    distances, penalties, bonuses = stock_table()
    for d, p, b in zip(distances, penalties, bonuses, strict=True):
        print(f"  {d:>9.2f} {-0.2 * p:>12.4f} {0.1 * b:>19.4f}")

    print("\nYour shaping term along a scripted trajectory (a reset happens at step 1):")
    rows = scripted_student_rollout()
    if not rows:
        print("  (no reach_progress_bonus() found -- add one in reach_student_cfg.py)")
    else:
        print(f"  {'step':>4} {'dist (m)':>9} {'bonus':>9}")
        for step, dist, bonus in rows:
            flag = ""
            if step == 1 and abs(bonus) > 1e-6:
                flag = "  <-- reset spike: add the first-step guard"
            print(f"  {step:>4} {dist:>9.2f} {bonus:>9.3f}{flag}")
        if all(abs(bonus) < 1e-9 for _, _, bonus in rows):
            print("  note: your term returns zero everywhere -- the TODO(week01) blanks look unfilled.")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n[plot] matplotlib is not installed; run `pip install matplotlib` to enable --plot.")
            return 0
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(distances, [-0.2 * p for p in penalties], "o-", label="position penalty (weighted)")
        ax[0].plot(distances, [0.1 * b for b in bonuses], "s-", label="tanh bonus (weighted)")
        ax[0].set(xlabel="tracking error (m)", ylabel="reward per step", title="Baseline terms")
        ax[0].legend()
        ax[1].plot(distances, bonuses, "s-", color="tab:green")
        ax[1].set(xlabel="tracking error (m)", ylabel="tanh bonus (unweighted)", title="Kernel shape (std=0.1)")
        fig.tight_layout()
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
