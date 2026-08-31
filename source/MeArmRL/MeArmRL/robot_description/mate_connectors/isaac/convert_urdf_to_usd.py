#!/usr/bin/env python3
"""Convert the cleaned MATE connectors URDF to a USD asset for IsaacLab.

Runs inside Isaac Sim's Python (``python.sh``) and uses the official
``isaacsim.asset.importer.urdf`` extension. The importer natively supports
``<mimic>`` joints (see the extension's ``test_mimic.urdf``), so the
four-bar linkages stay coherent under PhysX.

Usage (from the Isaac Sim install root):
    ./python.sh /path/to/ros2_ws/src/mate_connectors_assem/isaac/convert_urdf_to_usd.py

Output:
    <package>/isaac/assets/mate_connectors.usd
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Paths (edit these if your workspace lives elsewhere).
# ---------------------------------------------------------------------------
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(PKG_DIR, "urdf", "mate_connectors_clean.urdf")
USD_DIR = os.path.join(PKG_DIR, "isaac", "assets")
USD_PATH = os.path.join(USD_DIR, "mate_connectors.usd")

# ROS package name that owns the meshes (used to resolve package:// URLs).
ROS_PACKAGE_NAME = "mate_connectors_assem"


def main() -> None:
    # Omniverse/Isaac Sim imports must happen AFTER SimulationApp is created.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})

    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

    os.makedirs(USD_DIR, exist_ok=True)

    # The importer writes a layer-stack directory; we import into a temp dir
    # and then flatten the composed stage into a single self-contained USD.
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="mate_urdf_")
    tmp_usd = os.path.join(tmp_dir, "mate_connectors.usd")

    config = URDFImporterConfig()
    config.urdf_path = URDF_PATH
    config.usd_path = tmp_usd

    # Keep fixed joints as separate PhysicsFixedJoint prims (matches the
    # documented URDF importer example). Merging them collapses the base
    # chain into the root Xform, which breaks the articulation.
    config.merge_fixed_joints = False
    # The arm is table-mounted; fix the base link to the world.
    config.fix_base = True
    # Only the 4 actuated joints get drives; mimics are locked by PhysX.
    config.make_default_joints_drive = False
    # The URDF has no <collision> tags; generate them from the visuals.
    config.collision_from_visuals = True
    # Convex decomposition keeps the STL collision meshes PhysX-friendly.
    config.collision_type = "convexDecomposition"
    config.allow_self_collision = False
    # Resolve package://mate_connectors_assem/meshes/*.stl.
    config.ros_package_paths = [{"package": ROS_PACKAGE_NAME, "path": PKG_DIR}]
    # Regularize link density so tiny Onshape masses don't destabilize sim.
    config.link_density = 1000.0
    # Joint drive type: implicit (PD) drives, matching the IsaacLab config.
    config.joint_drive_type = "implicit"
    config.joint_target_type = "position"

    importer = URDFImporter(config)
    result = importer.import_urdf()
    print(f"Imported USD: {result}")

    # Flatten the composed stage into a single file for IsaacLab.
    from pxr import Usd

    stage = Usd.Stage.Open(result)
    if stage is None:
        raise RuntimeError(f"Failed to open imported stage: {result}")
    stage.Flatten().Export(USD_PATH)
    print(f"Flattened USD: {USD_PATH}")

    simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
