# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone USD inspection: dump bodies, joints, and their relationships.

Boots SimulationApp (headless), then opens the USD stage with pxr.Usd and
prints every link/rigid body, every physics joint (type, body0/body1 rels,
mimic API presence), and the articulation root. Read-only diagnostic.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect a USD asset: bodies, joints, relationships.")
parser.add_argument("--usd", type=str, default=None, help="Path to USD file (default: bundled mate_connectors.usd)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.usd is None:
    args_cli.usd = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "source/MeArmRL/MeArmRL/robot_description/mate_connectors/isaac/assets/mate_connectors.usd",
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

JOINT_TYPES = [
    "PhysicsRevoluteJoint",
    "PhysicsPrismaticJoint",
    "PhysicsFixedJoint",
    "PhysicsSphericalJoint",
    "PhysicsDistanceJoint",
    "PhysicsJoint",
]


def get_targets(prim: Usd.Prim, rel_name: str) -> list[str]:
    rel = rel_name
    attr = UsdPhysics.Joint(prim).GetBody0Rel() if rel == "body0" else UsdPhysics.Joint(prim).GetBody1Rel()
    if not attr:
        return []
    return [str(t.path) for t in attr.GetTargets()]


def main():
    usd_path = os.path.abspath(args_cli.usd)
    print(f"\n{'=' * 80}\nInspecting: {usd_path}\n{'=' * 80}")
    if not os.path.isfile(usd_path):
        print(f"[ERROR] File not found: {usd_path}")
        return

    stage = Usd.Stage.Open(usd_path)
    print(f"Stage opened: rootLayer={stage.GetRootLayer().identifier}")
    print(f"Default prim: {stage.GetDefaultPrim().GetPath() if stage.HasDefaultPrim() else '<none>'}")
    print(f"Meters per unit: {UsdGeom.GetStageMetersPerUnit(stage)}")

    bodies: list[Usd.Prim] = []
    joints: list[Usd.Prim] = []
    articulation_roots: list[Usd.Prim] = []
    other: list[Usd.Prim] = []

    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        if UsdPhysics.ArticulationRootAPI(prim):
            articulation_roots.append(prim)
        if UsdPhysics.RigidBodyAPI(prim):
            bodies.append(prim)
        elif type_name in JOINT_TYPES or type_name.endswith("Joint"):
            joints.append(prim)
        elif type_name in ("Xform", "Scope"):
            other.append(prim)

    print(f"\n--- Articulation roots ({len(articulation_roots)}) ---")
    for p in articulation_roots:
        print(f"  {p.GetPath()}  (type={p.GetTypeName()})")

    print(f"\n--- Rigid bodies ({len(bodies)}) ---")
    for p in bodies:
        mass_api = UsdPhysics.MassAPI(p)
        mass = mass_api.GetMassAttr().Get()
        print(f"  {p.GetPath()}  (type={p.GetTypeName()}, mass={mass})")

    print(f"\n--- Joints ({len(joints)}) ---")
    for p in joints:
        joint = UsdPhysics.Joint(p)
        body0 = [str(t) for t in joint.GetBody0Rel().GetTargets()] if joint.GetBody0Rel() else []
        body1 = [str(t) for t in joint.GetBody1Rel().GetTargets()] if joint.GetBody1Rel() else []
        collision = UsdPhysics.Joint(p).GetCollisionEnabledAttr().Get()
        has_mimic = p.HasAPI(PhysxSchema.PhysxMimicJointAPI)
        limits = (
            UsdPhysics.RevoluteJoint(p).GetLowerLimitAttr().Get(),
            UsdPhysics.RevoluteJoint(p).GetUpperLimitAttr().Get(),
        ) if type_name in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint") else None
        flag = "OK " if (body0 and body1) else "BAD"
        print(f"  [{flag}] {p.Ge~~tPath()}  (type={p.GetTypeName()})")
        print(f"        body0={body0 or '<MISSING>'}")
        print(f"        body1={body1 or '<MISSING>'}")
        if p.GetTypeName() == "PhysicsRevoluteJoint":
            print(f"        limits={limits}")
        for schema in p.GetAppliedSchemas():
            if "PhysxMimicJointAPI" in schema:
                inst = schema.split(":", 1)[1] if ":" in schema else ""
                for pn in sorted(p.GetPropertyNames()):
                    if "imic" not in pn:
                        continue
                    attr = p.GetAttribute(pn)
                    rel = p.GetRelationship(pn)
                    if attr:
                        print(f"        mimic[{inst}] {pn} = {attr.Get()}")
                    elif rel:
                        print(f"        mimic[{inst}] {pn} -> {[str(t) for t in rel.GetTargets()]}")

    mesh_count = sum(1 for p in stage.Traverse() if p.GetTypeName() == "Mesh")
    mat_count = sum(1 for p in stage.Traverse() if UsdShade.Material(p))
    summary_lines = []
    summary_lines.append(f"  Meshes: {mesh_count}, Materials: {mat_count}")
    summary_lines.append(f"  Bodies: {len(bodies)}, Joints: {len(joints)}, Articulation roots: {len(articulation_roots)}")
    bad = [p.GetPath() for p in joints if not (UsdPhysics.Joint(p).GetBody0Rel() and UsdPhysics.Joint(p).GetBody0Rel().GetTargets()) or not (UsdPhysics.Joint(p).GetBody1Rel() and UsdPhysics.Joint(p).GetBody1Rel().GetTargets())]
    summary_lines.append(f"  Joints with missing body rels: {len(bad)}")
    for path in bad:
        summary_lines.append(f"    - {path}")
    with open("/tmp/opencode/inspect_summary.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()