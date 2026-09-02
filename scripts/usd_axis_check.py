# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check mimic referenceJointAxis vs the reference joint's actual axis."""

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

axes = {}
for prim in stage.Traverse():
    if prim.GetTypeName() == "PhysicsRevoluteJoint":
        axis = prim.GetAttribute("physics:axis").Get()
        axes[prim.GetName()] = str(axis)

lines = []
for prim in stage.Traverse():
    if prim.GetTypeName() != "PhysicsRevoluteJoint":
        continue
    for schema in prim.GetAppliedSchemas():
        if "PhysxMimicJointAPI" not in schema:
            continue
        inst = schema.split(":", 1)[1] if ":" in schema else ""
        rel = prim.GetRelationship(f"physxMimicJoint:{inst}:referenceJoint")
        ref = str(rel.GetTargets()[0]).split("/")[-1] if rel and rel.GetTargets() else "<none>"
        axis_attr = prim.GetAttribute(f"physxMimicJoint:{inst}:referenceJointAxis")
        m_axis = str(axis_attr.Get()) if axis_attr else "<unauthored>"
        gearing = prim.GetAttribute(f"physxMimicJoint:{inst}:gearing")
        g = gearing.Get() if gearing else "<unauthored>"
        ref_axis = axes.get(ref, "<none>")
        ok = "OK " if m_axis == ref_axis else "BAD"
        lines.append(f"[{ok}] {prim.GetName()}: mimic ref={ref} refJointAxis={m_axis} refJointActualAxis={ref_axis} gearing={g}")

with open("/tmp/opencode/axis_check.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")

simulation_app.close()