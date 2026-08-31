#!/usr/bin/env python3
"""Generate a cleaned URDF for Isaac Sim / IsaacLab import.

Reads the Onshape-exported URDF and produces ``mate_connectors_clean.urdf``
with the following transformations:

1. Removes Onshape loop-closure anchor links (``closing_*``,
   ``effector_wrist_side_single__1__loop_closure``) and their fixed joints.
2. Converts ``revolute_base`` from ``continuous`` to ``revolute`` with
   +/-90 deg limits (SG90 pan-base assumption).
3. Converts the closed-loop ``continuous`` joints to ``revolute`` joints
   with ``<mimic>`` tags so PhysX keeps the four-bar linkages coherent.
   The mimic relationships are taken from ``left_four_bar_kinematics.py``.
4. Regularizes inertias (mass floor 1e-4 kg, diagonal inertia floor 1e-8).
5. Keeps ``package://`` mesh references (resolved at import time via the
   importer's ``ros_package_paths`` option).

Usage:
    python3 generate_clean_urdf.py [input.urdf] [output.urdf]
"""
import math
import sys
import xml.etree.ElementTree as ET

SRC = "urdf/dummy_mate_connectors_assembly.urdf"
DST = "urdf/mate_connectors_clean.urdf"

# Links that are pure Onshape loop-closure anchors (zero mass, no visual).
DUMMY_LINKS = {
    "closing_gripper_mate_connectro1",
    "closing_gripper_mate_connectro2",
    "closing_rev_boom1_1",
    "closing_rev_boom1_2",
    "closing_rev_rigging2_1",
    "closing_rev_rigging2_2",
    "effector_wrist_side_single__1__loop_closure",
    # Empty root link: the importer maps it to the robot Xform, which makes
    # every joint's body0 point at the Xform instead of the parent rigid
    # body. Removing it makes base_table_bottom the tree root (like UR10).
    "root",
}

# Joints whose child is a dummy link (removed together with the link).
DUMMY_JOINTS = {
    "closing_gripper_mate_connectro1",
    "closing_gripper_mate_connectro2",
    "closing_rev_boom1_1",
    "closing_rev_boom1_2",
    "closing_rev_rigging2_1",
    "closing_rev_rigging2_2",
    "fastened_2_loop_closure",
    # Connects the empty root link to the base; removed with it.
    "fixed_node_to_root_joint",
}

# Base joint: continuous -> revolute with SG90 pan-base limits.
BASE_JOINT = "revolute_base"
BASE_LIMITS = (-1.5708, 1.5708)

# Closed-loop joints -> (source joint, multiplier, offset).
# Derived from left_four_bar_kinematics.py. Where the solver couples two
# inputs (e.g. rev_rigging4 = -theta - phi) the dominant source is used and
# the approximation is documented in the README.
MIMIC_MAP = {
    "rev_rigging4": ("revolute_left", -1.0, 0.0),
    "rev_rigging3": ("revolute_left", -1.0, 0.0),
    "rev_rigging1": ("revolute_left", 1.0, 0.0),
    "rev_effector1": ("revolute_left", -1.0, 0.0),
    "rev_effector2": ("revolute_left", -1.0, 0.0),
    "rev_rigging_twin_bottom": ("revolute_right", -1.0, 0.0),
    "rev_boom_trinagle": ("revolute_left", -1.0, 0.0),
    "rev_boom3": ("revolute_right", 1.0, 0.0),
    "rev_boom2": ("revolute_right", 1.0, 0.0),
    "rev_end_effector2": ("rev_end_effector1", -1.0, 0.0),
}

# Source joint limits (lower, upper) used to size the mimic joint limits.
SOURCE_LIMITS = {
    "revolute_left": (-1.07998, 0.141749),
    "revolute_right": (-1.12601, 0.706583),
    "rev_end_effector1": (-0.575959, 0.418879),
}

MASS_FLOOR = 1e-4
INERTIA_FLOOR = 1e-8


def regularize_inertial(inertial):
    """Floor mass and diagonal inertia so PhysX accepts the link."""
    mass_el = inertial.find("mass")
    if mass_el is None:
        return
    mass = float(mass_el.get("value", "0"))
    if mass < MASS_FLOOR:
        mass_el.set("value", f"{MASS_FLOOR:.6e}")

    inertia_el = inertial.find("inertia")
    if inertia_el is None:
        return
    diag = [float(inertia_el.get(k, "0")) for k in ("ixx", "iyy", "izz")]
    if max(diag) < INERTIA_FLOOR:
        for k in ("ixx", "iyy", "izz"):
            inertia_el.set(k, f"{INERTIA_FLOOR:.6e}")


def scaled_limits(source, multiplier):
    """Map a source joint's limits through the mimic relationship."""
    lower, upper = SOURCE_LIMITS[source]
    if multiplier < 0:
        lower, upper = -upper, -lower
    return lower, upper


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else SRC
    dst = sys.argv[2] if len(sys.argv) > 2 else DST

    tree = ET.parse(src)
    root = tree.getroot()

    # 1. Remove dummy links and their joints.
    for link in list(root.findall("link")):
        if link.get("name") in DUMMY_LINKS:
            root.remove(link)
    for joint in list(root.findall("joint")):
        if joint.get("name") in DUMMY_JOINTS:
            root.remove(joint)

    # 2. Collect source joint limits for mimic sizing.
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name in SOURCE_LIMITS:
            continue
        limit = joint.find("limit")
        if limit is not None and name in ("revolute_left", "revolute_right", "rev_end_effector1"):
            SOURCE_LIMITS[name] = (
                float(limit.get("lower", "0")),
                float(limit.get("upper", "0")),
            )

    # 3. Convert base joint to revolute with limits.
    for joint in root.findall("joint"):
        if joint.get("name") != BASE_JOINT:
            continue
        joint.set("type", "revolute")
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        limit.set("lower", f"{BASE_LIMITS[0]}")
        limit.set("upper", f"{BASE_LIMITS[1]}")
        limit.set("effort", "1")
        limit.set("velocity", "1")

    # 4. Convert loop-closing joints to revolute mimics.
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in MIMIC_MAP:
            continue
        source, multiplier, offset = MIMIC_MAP[name]
        joint.set("type", "revolute")
        lower, upper = scaled_limits(source, multiplier)
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        limit.set("lower", f"{lower}")
        limit.set("upper", f"{upper}")
        limit.set("effort", "1")
        limit.set("velocity", "1")
        mimic = joint.find("mimic")
        if mimic is None:
            mimic = ET.SubElement(joint, "mimic")
        mimic.set("joint", source)
        mimic.set("multiplier", f"{multiplier}")
        mimic.set("offset", f"{offset}")

    # 5. Regularize inertias.
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            regularize_inertial(inertial)

    # Serialize with a readable header.
    ET.indent(tree, space="  ")
    header = (
        '<?xml version="1.0" ?>\n'
        "<!-- Cleaned URDF for Isaac Sim / IsaacLab import.\n"
        "     Generated from dummy_mate_connectors_assembly.urdf by\n"
        "     isaac/generate_clean_urdf.py. Closed-loop joints are driven\n"
        "     via <mimic>; see isaac/README.md for the approximation notes. -->\n"
    )
    body = ET.tostring(root, encoding="unicode")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(header + body + "\n")

    print(f"Wrote {dst}")
    print(f"  links: {len(root.findall('link'))}")
    print(f"  joints: {len(root.findall('joint'))}")
    mimics = [j.get("name") for j in root.findall("joint") if j.find("mimic") is not None]
    print(f"  mimic joints: {len(mimics)} -> {mimics}")


if __name__ == "__main__":
    main()
