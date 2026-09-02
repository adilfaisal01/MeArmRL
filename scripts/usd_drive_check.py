# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare joint drive properties of the actuated joints in the USD asset."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--usd", type=str, default=None)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics

if args.usd is None:
    args.usd = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "source/MeArmRL/MeArmRL/robot_description/mate_connectors/isaac/assets/mate_connectors.usd",
    )

stage = Usd.Stage.Open(os.path.abspath(args.usd))
ACTUATED = ["revolute_base", "revolute_left", "revolute_right", "rev_end_effector1"]
lines = []
for prim in stage.Traverse():
    if prim.GetTypeName() != "PhysicsRevoluteJoint":
        continue
    name = prim.GetName()
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    props = {}
    if drive:
        for attr_name in ("stiffness", "damping", "maxForce"):
            attr = getattr(drive, f"Get{attr_name.capitalize()}Attr", lambda: None)()
            props[attr_name] = attr.Get() if attr else "<unauthored>"
    else:
        props = "<no DriveAPI:angular>"
    friction = prim.GetAttribute("physxJoint:jointFriction").Get() if prim.GetAttribute("physxJoint:jointFriction") else "<none>"
    mark = "*" if name in ACTUATED else " "
    lines.append(f"{mark} {name}: drive={props} friction={friction}")

with open("/tmp/opencode/drive_check.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")

simulation_app.close()