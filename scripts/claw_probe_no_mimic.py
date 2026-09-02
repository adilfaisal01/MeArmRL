# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Claw probe variant: test whether the claw mimic constraint blocks actuation.

Runs the Mearmrl-Reach-v0 env against a USD COPY with the PhysxMimicJointAPI
removed from rev_end_effector2 (claw-left). If the claw master
(rev_end_effector1) then tracks commands, the mimic constraint is the blocker.
Resets the env before each phase so episode timeouts don't pollute readbacks.
Writes results to /tmp/opencode/claw_no_mimic.txt.
"""

import argparse
import os
import shutil

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Claw probe with mimic removed from claw-left.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# --- Prepare a mimic-free USD copy BEFORE the env loads it ---
SRC_USD = "/MeArmRL/source/MeArmRL/MeArmRL/robot_description/mate_connectors/isaac/assets/mate_connectors.usd"
MOD_USD = "/tmp/opencode/claw_no_mimic.usd"

from pxr import Usd

shutil.copyfile(SRC_USD, MOD_USD)
stage = Usd.Stage.Open(MOD_USD)
claw2 = stage.GetPrimAtPath("/new_mate_connectors_assem/joints/rev_end_effector2")
removed = []
for schema in list(claw2.GetAppliedSchemas()):
    if "PhysxMimicJointAPI" in schema:
        inst = schema.split(":", 1)[1] if ":" in schema else ""
        claw2.RemoveAPI("PhysxMimicJointAPI", inst)
        rel = claw2.GetRelationship(f"physxMimicJoint:{inst}:referenceJoint")
        if rel:
            rel.ClearTargets(True)
            rel.SetTargets([])
        removed.append(f"PhysxMimicJointAPI:{inst}")
stage.GetRootLayer().Save()
print(f"[CLAW2] removed mimic APIs from rev_end_effector2: {removed}")

# --- Now boot the env against the modified asset ---
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import MeArmRL.tasks  # noqa: F401

TASK = "Mearmrl-Reach-v0"
CLAW_IDX = 3  # index of rev_end_effector1 in the action space


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env_cfg.scene.robot.spawn.usd_path = MOD_USD
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()
    robot = env.unwrapped.scene.articulations["robot"]

    out = []

    out_fh = open("/tmp/opencode/claw_no_mimic.txt", "w")

    def run_phase(label, claw_target, steps=100):
        for _ in range(steps):
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[0, CLAW_IDX] = claw_target
            env.step(actions)
        pos = robot.data.joint_pos[0].cpu()
        vel = robot.data.joint_vel[0].cpu()
        names = list(robot.joint_names)
        readback = {n: round(float(pos[i]), 3) for i, n in enumerate(names)}
        claw = float(pos[-2])
        claw2 = float(pos[-1])
        line = f"{label}: claw={claw:.4f} claw2={claw2:.4f} vel={float(vel[-2]):.4f}"
        out_fh.write(line + "\n" + f"  all: {readback}\n")
        out_fh.flush()

    run_phase("rest (target 0)", 0.0)
    run_phase("target +0.35", 0.35)
    run_phase("target -0.5", -0.5)
    run_phase("target 0 again", 0.0)

    out_fh.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()