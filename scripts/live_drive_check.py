# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read the LIVE joint drive properties after IsaacLab env init."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--boost_effort", type=float, default=None)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import MeArmRL.tasks  # noqa: F401

TASK = "Mearmrl-Reach-v0"


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    if args_cli.boost_effort is not None:
        env_cfg.scene.robot.actuators["arm"].effort_limit_sim = args_cli.boost_effort
        env_cfg.scene.robot.actuators["arm"].effort_limit = args_cli.boost_effort
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene.articulations["robot"]
    lines = [f"joint_names: {list(robot.joint_names)}"]
    act = robot.actuators.get("arm")
    lines.append(f"actuator cfg: {act.__class__.__name__} effort_limit={act.effort_limit} effort_limit_sim={getattr(act, 'effort_limit_sim', None)} stiffness={act.stiffness} damping={act.damping}")

    # Live USD prim drive readback
    from pxr import Usd, UsdPhysics

    from isaaclab.sim import find_first_matching_prim  # noqa: F401

    prim = find_first_matching_prim("/World/envs/env_0/Robot")
    stage = prim.GetStage()
    ACTUATED = ["revolute_base", "revolute_left", "revolute_right", "rev_end_effector1"]
    for child in Usd.PrimRange(prim):
        if child.GetTypeName() == "PhysicsRevoluteJoint" and child.GetName() in ACTUATED:
            drive = UsdPhysics.DriveAPI.Get(child, "angular")
            if drive:
                lines.append(
                    f"LIVE {child.GetName()}: stiffness={drive.GetStiffnessAttr().Get()} damping={drive.GetDampingAttr().Get()} maxForce={drive.GetMaxForceAttr().Get() if drive.GetMaxForceAttr() else '<unauthored>'}"
                )
            else:
                lines.append(f"LIVE {child.GetName()}: <no DriveAPI>")

    # PhysX-side truth: what gains does the simulation actually hold?
    names = list(robot.joint_names)
    ks = robot.data.joint_stiffness[0].cpu().tolist()
    ds = robot.data.joint_damping[0].cpu().tolist()
    ef = robot.data.joint_effort_limit[0].cpu().tolist() if hasattr(robot.data, "joint_effort_limit") else None
    for i, n in enumerate(names):
        limit = f" effort_limit={ef[i]:.2f}" if ef is not None else ""
        lines.append(f"PHYSX {n}: stiffness={ks[i]:.1f} damping={ds[i]:.2f}{limit}")

    with open("/tmp/opencode/live_drive.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()