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

    from pxr import Usd

    stage = Usd.Stage.Open(tmp_usd)
    if stage is None:
        raise RuntimeError(f"Failed to open imported stage: {tmp_usd}")
    # IsaacLab references the asset as <defaultPrim>; without it the reference
    # resolves to nothing ("Unresolved reference prim path").
    if not stage.HasDefaultPrim():
        root_children = list(stage.GetPseudoRoot().GetAllChildren())
        asset_root = [p for p in root_children if str(p.GetPath()) != "/World"]
        if len(asset_root) == 1:
            stage.SetDefaultPrim(asset_root[0])
            print(f"Set default prim: {asset_root[0].GetPath()}")
        else:
            raise RuntimeError(f"Expected exactly one root prim, got: {[str(p.GetPath()) for p in root_children]}")
    stage.Flatten().Export(USD_PATH)
    print(f"Flattened USD: {USD_PATH} ({os.path.getsize(USD_PATH)} bytes)")

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())