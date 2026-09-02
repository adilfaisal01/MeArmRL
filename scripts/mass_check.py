# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Report live body masses of the articulation (density override check)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=1)
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
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()
    robot = env.unwrapped.scene.articulations["robot"]

    masses = robot.root_physx_view.get_link_materials() if hasattr(robot.root_physx_view, "get_link_materials") else None
    lines = []
    if masses is not None:
        for i, name in enumerate(list(robot.body_names)):
            lines.append(f"{name}: density={float(masses[i][0]):.1f}")
    else:
        # fallback: read masses from data
        lines.append(f"body_names: {list(robot.body_names)}")
        lines.append(f"total mass: {float(robot.root_physx_view.get_link_energies().sum()) if hasattr(robot.root_physx_view, 'get_link_energies') else 'n/a'}")

    # masses via usd readback of the live stage
    from pxr import Usd, UsdPhysics

    from isaaclab.sim import find_first_matching_prim

    prim = find_first_matching_prim("/World/envs/env_0/Robot")
    stage = prim.GetStage()
    total = 0.0
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.MassAPI):
            m = UsdPhysics.MassAPI(child).GetMassAttr().Get()
            d = UsdPhysics.MassAPI(child).GetDensityAttr().Get()
            if m is not None:
                total += m
                lines.append(f"MASS {child.GetName()}: {m:.4f} kg (density={d})")
    lines.append(f"TOTAL: {total:.4f} kg")

    with open("/tmp/opencode/mass_check.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()