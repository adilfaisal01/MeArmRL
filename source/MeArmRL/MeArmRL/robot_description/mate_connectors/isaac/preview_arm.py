#!/usr/bin/env python3
"""Render a 3D preview of the MATE connectors arm from the cleaned URDF.

Loads the URDF, computes the fixed-joint transform tree (zero pose for the
revolute joints), and draws every STL mesh with matplotlib. This is a quick
visual sanity check that runs WITHOUT Isaac Sim.

Usage:
    python3 preview_arm.py [output.png]
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(PKG_DIR, "urdf", "mate_connectors_clean.urdf")
MESH_DIR = os.path.join(PKG_DIR, "meshes")


def rpy_to_matrix(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def parse_origin(el):
    if el is None:
        return np.eye(4)
    xyz = np.array([float(v) for v in el.get("xyz", "0 0 0").split()])
    rpy = np.array([float(v) for v in el.get("rpy", "0 0 0").split()])
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "mate_connectors_preview.png"

    tree = ET.parse(URDF_PATH)
    root = tree.getroot()

    # Build parent->child transform map (fixed joints only; revolute at zero).
    transforms = {}
    for j in root.findall("joint"):
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        transforms[child] = (parent, parse_origin(j.find("origin")))

    # Compute world transform for every link.
    world = {}
    for link in root.findall("link"):
        name = link.get("name")
        T = np.eye(4)
        cur = name
        while cur in transforms:
            p, Tj = transforms[cur]
            T = Tj @ T
            cur = p
        world[name] = T

    # Load meshes.
    meshes = []
    for link in root.findall("link"):
        name = link.get("name")
        for vis in link.findall("visual"):
            mesh_el = vis.find("geometry/mesh")
            if mesh_el is None:
                continue
            fn = mesh_el.get("filename")
            base = os.path.basename(fn)
            path = os.path.join(MESH_DIR, base)
            if not os.path.exists(path):
                print(f"  WARN: missing mesh {path}")
                continue
            m = trimesh.load(path)
            scale = [float(v) for v in mesh_el.get("scale", "1 1 1").split()]
            if scale != [1.0, 1.0, 1.0]:
                m = m.copy()
                m.apply_scale(scale)
            m.apply_transform(world[name] @ parse_origin(vis.find("origin")))
            meshes.append(m)

    if not meshes:
        print("No meshes loaded; nothing to render.")
        return 1

    combined = trimesh.util.concatenate(meshes)
    print(f"Loaded {len(meshes)} meshes, {len(combined.vertices)} vertices total")

    # Render with matplotlib.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    verts = combined.vertices
    faces = combined.faces
    tri = verts[faces]
    pc = Poly3DCollection(tri, alpha=0.85, facecolor="0.75", edgecolor="0.3", linewidth=0.1)
    ax.add_collection3d(pc)

    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    c = (lo + hi) / 2
    r = (hi - lo).max() / 2 + 0.02
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("MATE Connectors Arm (zero pose, from cleaned URDF)")
    ax.view_init(elev=25, azim=-60)

    fig.savefig(out, dpi=150)
    print(f"Saved preview: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
