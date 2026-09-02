# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Claw joint probe for the MATE connectors arm.

Drives rev_end_effector1 to targets WITHIN its URDF limits and reads back
position/velocity to distinguish torque-limited from mechanically-blocked.
Repeats the same targets with boosted effort limit to test the 1.8 N*m cap.
Writes results to /tmp/opencode/claw_probe.txt.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Claw joint probe.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--boost_effort", type=float, default=None, help="Override actuator effort_limit")
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
CLAW_IDX = 3  # index of rev_end_effector1 in the action space


def run_phase(env, robot, label, claw_target, steps, out):
    for _ in range(steps):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[0, CLAW_IDX] = claw_target
            env.step(actions)
    with torch.inference_mode():
        pos = robot.data.joint_pos[0].cpu()
        vel = robot.data.joint_vel[0].cpu()
    claw = float(pos[-2])  # rev_end_effector1 is second-to-last joint
    claw2 = float(pos[-1])
    lines = [
        f"{label}: claw={claw:.4f} claw2={claw2:.4f} vel={float(vel[-2]):.4f}",
    ]
    for line in lines:
        print(f"[CLAW] {line}")
    out.extend(lines)


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    if args_cli.boost_effort is not None:
        env_cfg.scene.robot.actuators["arm"].effort_limit = args_cli.boost_effort
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()
    robot = env.unwrapped.scene.articulations["robot"]
    names = list(robot.joint_names)
    print(f"[CLAW] joints: {names}")

    out = []
    hold = 120  # 2 s per phase at 60 Hz
    # Phase 1: rest
    run_phase(env, robot, "rest (target 0)", 0.0, hold, out)
    # Phase 2: claw to +0.4 (inside upper limit 0.419)
    run_phase(env, robot, "target +0.4", 0.4, hold, out)
    # Phase 3: claw to -0.5 (inside lower limit -0.576)
    run_phase(env, robot, "target -0.5", -0.5, hold, out)
    # Phase 4: back to 0
    run_phase(env, robot, "target 0 again", 0.0, hold, out)

    with open("/tmp/opencode/claw_probe.txt", "a") as fh:
        tag = f"boost={args_cli.boost_effort}" if args_cli.boost_effort else "default-effort"
        fh.write(f"--- {tag} ---\n" + "\n".join(out) + "\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()