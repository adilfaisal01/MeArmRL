#!/usr/bin/env python3
"""Convert the cleaned MATE connectors URDF to a USD asset for IsaacLab.

Runs inside Isaac Sim's Python (``isaaclab.sh -p``) and uses the
``isaacsim.asset.importer.urdf`` extension (Isaac Sim 5.1 API:
``_urdf.ImportConfig`` + ``URDFParseAndImportFile`` kit command).

The URDF references meshes as ``package://mate_connectors_assem/meshes/*.stl``.
The 5.1 importer resolves ``package://`` URIs through the ``ROS_PACKAGE_PATH``
environment variable, so we create a temp "ROS workspace" containing a
``mate_connectors_assem`` symlink to this package dir.

Usage (from the repo root, inside the container):
    /IsaacLab/isaaclab.sh -p source/MeArmRL/MeArmRL/robot_description/mate_connectors/isaac/convert_urdf_to_usd.py

Output:
    <package>/isaac/assets/mate_connectors.usd  (flattened, self-contained)
"""
import argparse
import os
import sys
import tempfile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert MATE connectors URDF to USD (Isaac Sim 5.1 API).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import omni.kit.app
import omni.kit.commands

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)

from isaacsim.asset.importer.urdf import _urdf  # noqa: E402

# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(PKG_DIR, "urdf", "mate_connectors_clean.urdf")
USD_DIR = os.path.join(PKG_DIR, "isaac", "assets")
USD_PATH = os.path.join(USD_DIR, "mate_connectors.usd")

ROS_PACKAGE_NAME = "mate_connectors_assem"


def main() -> int:
    os.makedirs(USD_DIR, exist_ok=True)

    # The importer looks up package://<name>/... via ROS_PACKAGE_PATH, i.e. it
    # expects a directory <entry>/<name>/ containing the meshes. Create a temp
    # workspace with a symlink named after the ROS package.
    ros_ws = tempfile.mkdtemp(prefix="mate_ros_ws_")
    os.symlink(PKG_DIR, os.path.join(ros_ws, ROS_PACKAGE_NAME))
    os.environ["ROS_PACKAGE_PATH"] = ros_ws
    print(f"ROS_PACKAGE_PATH={ros_ws} ({ROS_PACKAGE_NAME} -> {PKG_DIR})")

    config = _urdf.ImportConfig()
    # Keep fixed joints as separate PhysicsFixedJoint prims. Merging collapses
    # the base chain into the root Xform and breaks the articulation.
    config.set_merge_fixed_joints(False)
    # The arm is table-mounted; fix the base link to the world.
    config.set_fix_base(True)
    # The URDF has no <collision> tags; generate collision from visuals.
    config.set_collision_from_visuals(True)
    # Convex decomposition keeps the STL collision meshes PhysX-friendly.
    config.set_convex_decomp(True)
    config.set_self_collision(False)
    # URDF <mimic> tags -> PhysxMimicJointAPI (four-bar linkages).
    config.set_parse_mimic(True)
    # Regularize link density so tiny Onshape masses don't destabilize sim.
    config.set_density(1000.0)
    # Import inertia tensors from the URDF.
    config.set_import_inertia_tensor(True)
    # Position-drive defaults (IsaacLab's ImplicitActuatorCfg overrides gains).
    config.set_default_drive_type(_urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION)

    # Import into a temp dir first, then flatten into the committed asset.
    tmp_dir = tempfile.mkdtemp(prefix="mate_urdf_")
    tmp_usd = os.path.join(tmp_dir, "mate_connectors.usd")

    result = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=URDF_PATH,
        import_config=config,
        dest_path=tmp_usd,
    )
    print(f"Imported USD: {result}")

    from pxr import Sdf, Usd, UsdPhysics

    # Parse URDF joint limits and <mimic> references once.
    import xml.etree.ElementTree as ET

    urdf_limits = {}
    urdf_mimics = {}
    for joint in ET.parse(URDF_PATH).getroot().findall("joint"):
        name = joint.get("name")
        lim = joint.find("limit")
        if lim is not None and lim.get("lower") is not None and lim.get("upper") is not None:
            urdf_limits[name] = (float(lim.get("lower")), float(lim.get("upper")))  # pyright: ignore[reportArgumentType]
        mimic = joint.find("mimic")
        if mimic is not None:
            urdf_mimics[name] = mimic.get("joint")

    # Work on the flattened stage: a single layer, so post-import repairs
    # author correctly (the importer writes a layer-stack where editing the
    # composed stage targets the wrong layer). Flatten to a temp FILE first,
    # then reopen it file-backed — anonymous layers have bitten us before.
    stage = Usd.Stage.Open(tmp_usd)
    if stage is None:
        raise RuntimeError(f"Failed to open imported stage: {tmp_usd}")
    tmp_flat = os.path.join(tmp_dir, "flattened.usd")
    if not stage.Flatten().Export(tmp_flat):
        raise RuntimeError(f"Flatten export failed: {tmp_flat}")
    flat_stage = Usd.Stage.Open(tmp_flat)

    # 1) Restore URDF joint limits on revolute joints (the importer does not
    #    author them; PhysX requires finite limits on mimic joints).
    import math

    limits_restored = 0
    joint_paths = {}
    for prim in flat_stage.Traverse():
        if prim.GetTypeName().endswith("Joint"):
            joint_paths[prim.GetName()] = prim.GetPath()
        rj = UsdPhysics.RevoluteJoint(prim)
        if not rj or prim.GetName() not in urdf_limits:
            continue
        lo, hi = urdf_limits[prim.GetName()]
        # USD revolute-joint limits are in degrees; URDF uses radians.
        rj.GetLowerLimitAttr().Set(math.degrees(lo))
        rj.GetUpperLimitAttr().Set(math.degrees(hi))
        limits_restored += 1

    # 2) Repair empty PhysxMimicJointAPI referenceJoint relationships.
    mimics_repaired = []
    for prim in flat_stage.Traverse():
        for schema in prim.GetAppliedSchemas():
            if "PhysxMimicJointAPI" not in schema:
                continue
            inst = schema.split(":", 1)[1] if ":" in schema else ""
            rel_name = f"physxMimicJoint:{inst}:referenceJoint"
            rel = prim.GetRelationship(rel_name)
            if rel is None:
                rel = prim.CreateRelationship(rel_name)
            if rel.GetTargets():
                continue
            ref_name = urdf_mimics.get(prim.GetName())
            if ref_name and ref_name in joint_paths:
                rel.SetTargets([Sdf.Path(joint_paths[ref_name])])
                mimics_repaired.append(f"{prim.GetName()} -> {ref_name}")

    report = [f"Limits restored on {limits_restored} revolute joints", f"Mimic refs repaired: {mimics_repaired}"]
    with open("/tmp/opencode/convert_repairs.txt", "w") as f:
        f.write("\n".join(report) + "\n")

    # IsaacLab references the asset as <defaultPrim>; without it the reference
    # resolves to nothing ("Unresolved reference prim path").
    if not flat_stage.HasDefaultPrim():
        root_children = list(flat_stage.GetPseudoRoot().GetAllChildren())
        asset_root = [p for p in root_children if not str(p.GetPath()).startswith("/Flattened_Prototype")]
        if len(asset_root) == 1:
            flat_stage.SetDefaultPrim(asset_root[0])
            print(f"Set default prim: {asset_root[0].GetPath()}")
        else:
            raise RuntimeError(f"Expected exactly one asset root, got: {[str(p.GetPath()) for p in root_children]}")
    if not flat_stage.GetRootLayer().Export(USD_PATH):
        raise RuntimeError(f"Final export failed: {USD_PATH}")

    # Read back the exported file and verify the repairs actually landed.
    check_stage = Usd.Stage.Open(USD_PATH)
    boom2 = check_stage.GetPrimAtPath("/new_mate_connectors_assem/joints/rev_boom2")
    rj2 = UsdPhysics.RevoluteJoint(boom2)
    rel2 = boom2.GetRelationship("physxMimicJoint:rotZ:referenceJoint")
    report.append(
        "VERIFY rev_boom2: limits=({},{}) mimic_ref={}".format(
            rj2.GetLowerLimitAttr().Get(), rj2.GetUpperLimitAttr().Get(), [str(t) for t in rel2.GetTargets()] if rel2 else None
        )
    )
    report.append(f"Flattened USD: {USD_PATH} ({os.path.getsize(USD_PATH)} bytes)")
    with open("/tmp/opencode/convert_repairs.txt", "w") as f:
        f.write("\n".join(report) + "\n")

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())