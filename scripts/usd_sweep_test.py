# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted joint sweep for the MATE connectors arm USD.

Drives each of the 4 actuated joints through its range sequentially while
holding the others, printing per-joint position readbacks. The mimic-driven
linkage joints (rev_boom*, rev_rigging*, rev_effector*, rev_end_effector2)
are printed too, so four-bar tracking can be verified numerically (headless)
or visually (GUI). Works with any IsaacLab env; write-to-file for diagnostics.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted joint sweep for MATE connectors arm.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import MeArmRL.tasks  # noqa: F401

TASK = "Mearmrl-Reach-v0"
ACTUATED = ["revolute_base", "revolute_left", "revolute_right", "rev_end_effector1"]
# Joints that should move WITH their masters (mimic-driven four-bar linkages).
MIMIC_EXPECTATIONS = {
    "revolute_right": ["rev_boom2", "rev_boom3", "rev_rigging_twin_bottom"],
    "revolute_left": ["rev_boom_trinagle", "rev_rigging1", "rev_rigging3", "rev_rigging4", "rev_effector1", "rev_effector2"],
    "rev_end_effector1": ["rev_end_effector2"],
}


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene.articulations["robot"]
    joint_names = [n for n in robot.joint_names]
    print(f"[SWEEP] joints in articulation: {joint_names}")

    # Sweep each actuated joint: hold mid position, then command a ramp across
    # its range while others stay at zero. Targets are applied as position
    # actions; the JointPositionActionCfg scale is 1.0, so actions are the
    # target positions in rad relative to default (default is all-zero).
    steps_per_phase = 120  # 2 s at 60 Hz sim
    plan = []
    for joint_idx, name in enumerate(ACTUATED):
        plan.append(("hold-zero", None))
        plan.append(("sweep-up", joint_idx))
        plan.append(("sweep-down", joint_idx))

    results = []
    phase_i = 0
    step_in_phase = 0
    for phase, joint_idx in plan:
        for _ in range(steps_per_phase):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                if phase == "sweep-up":
                    actions[0, joint_idx] = 1.0
                elif phase == "sweep-down":
                    actions[0, joint_idx] = -1.0
                env.step(actions)
            step_in_phase += 1

        with torch.inference_mode():
            pos = robot.data.joint_pos[0].cpu().numpy()
        readback = {n: round(float(pos[i]), 3) for i, n in enumerate(joint_names)}
        results.append((phase, ACTUATED[joint_idx] if joint_idx is not None else None, readback))
        print(f"[SWEEP] after {phase} ({ACTUATED[joint_idx] if joint_idx is not None else 'zero'}): {readback}")

    # Numeric check: after the final sweep-down of the last joint, all should
    # be near zero-ish or bounded; the real check is mimic tracking.
    lines = []
    for phase, driver, readback in results:
        if driver is None:
            continue
        followers = MIMIC_EXPECTATIONS.get(driver, [])
        for f in followers:
            if f in readback:
                lines.append(f"{driver}={readback[driver]} -> {f}={readback[f]}")
    print("[SWEEP] mimic tracking (driver -> follower):")
    for line in lines:
        print(f"[SWEEP]   {line}")

    with open("/tmp/opencode/sweep_results.txt", "w") as fh:
        fh.write("\n".join(f"[SWEEP] {line}" for line in lines) + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()