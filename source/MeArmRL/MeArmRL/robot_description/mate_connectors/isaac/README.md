# MATE Connectors Arm — Isaac Sim / IsaacLab Conversion

This directory contains everything needed to take the RViz-only URDF
(`urdf/dummy_mate_connectors_assembly.urdf`) and bring it into Isaac Sim
and IsaacLab as a simulated, actuated robot.

## Files

| File | Purpose |
|---|---|
| `urdf/mate_connectors_clean.urdf` | Cleaned URDF (generated, do not hand-edit) |
| `generate_clean_urdf.py` | Regenerates the cleaned URDF from the Onshape export |
| `convert_urdf_to_usd.py` | Isaac Sim script: URDF -> USD |
| `mate_connectors_config.py` | IsaacLab `ArticulationCfg` |
| `assets/mate_connectors.usd` | Flattened USD (output of the conversion) |

## Verified in this workspace

The full pipeline was run and verified with Isaac Sim 6.0 + IsaacLab 0.54.4
in the `robot-isaac` pyenv:

- `convert_urdf_to_usd.py` produces a flattened, self-contained USD.
- The USD loads via `isaacsim.core.experimental.utils.stage.open_stage` with
  1 articulation root, 14 revolute joints, and 10 PhysX mimic joints.
- `MATE_CONNECTORS_CFG` spawns in IsaacLab: 14 joints found, 4 actuated
  (`revolute_base`, `revolute_left`, `revolute_right`, `rev_end_effector1`),
  and the sim steps without errors.

## What was cleaned up

1. **Loop-closure anchors removed.** Onshape emits zero-mass "closing_*" and
   `effector_wrist_side_single__1__loop_closure` links used only to close
   kinematic loops. These and their fixed joints are deleted.
2. **Empty `root` link removed.** The Onshape export has an empty `root` link
   at the top of the tree. If kept, the importer maps it to the robot Xform
   and every joint's `body0` points at the Xform instead of a rigid body,
   which breaks the articulation (IsaacLab finds 0 joints). Removing it makes
   `base_table_bottom` the tree root, matching the UR10 reference layout.
3. **`revolute_base` got limits.** It was `continuous` (unlimited). Assigned
   ±90° (±1.5708 rad) as an SG90 pan-base assumption.
4. **Closed loops -> mimic joints.** The two four-bar linkages and the
   gripper linkage are driven by `revolute_left`, `revolute_right`, and
   `rev_end_effector1`. The dependent joints are now `revolute` with
   `<mimic>` tags so PhysX keeps them coherent. Isaac Sim's URDF importer
   natively supports `<mimic>`.
5. **Inertia regularization.** Onshape masses as low as `1e-9 kg` are floored
   to `1e-4 kg` and diagonal inertias to `1e-8` so PhysX accepts every link.

## Actuated DOFs (4)

| Joint | Servo | Limits (rad) |
|---|---|---|
| `revolute_base` | base SG90 | -1.5708 .. 1.5708 |
| `revolute_left` | left boom SG90 | -1.07998 .. 0.141749 |
| `revolute_right` | right boom SG90 | -1.12601 .. 0.706583 |
| `rev_end_effector1` | gripper SG90 | -0.575959 .. 0.418879 |

Everything else moves via mimic.

## Mimic approximation notes

The mimic relationships are taken from `scripts/left_four_bar_kinematics.py`.
That node solves the loops exactly; the URDF mimic tags are a static
approximation:

- `rev_rigging4`, `rev_rigging3`, `rev_rigging1`, `rev_effector1`,
  `rev_effector2`, `rev_boom_trinagle` mimic `revolute_left`.
- `rev_rigging_twin_bottom`, `rev_boom3`, `rev_boom2` mimic `revolute_right`.
- `rev_end_effector2` mimics `rev_end_effector1` (multiplier -1).

The solver couples some joints to *both* `revolute_left` and
`revolute_right` (e.g. `rev_rigging4 = -theta - phi`). A single-source mimic
cannot express that, so the cleaned URDF uses the dominant source. For
simulation this is acceptable; if you need exact kinematics, drive the
mimic joints from a controller instead (see "Exact kinematics" below).

## Workflow

### 1. Convert URDF -> USD (Isaac Sim)

From the Isaac Sim install root (or with the `robot-isaac` pyenv active):

```bash
python3 <repo>/source/MeArmRL/MeArmRL/robot_description/mate_connectors/isaac/convert_urdf_to_usd.py
```

This writes `isaac/assets/mate_connectors.usd` (flattened, self-contained).
The importer resolves the `package://mate_connectors_assem/meshes/*.stl`
references via `ros_package_paths` (the script points it at the package dir
containing `meshes/`), keeps fixed joints as separate prims
(`merge_fixed_joints=False` — required so the base chain stays rigid bodies),
fixes the base, and generates convex-decomposed collision meshes from the
visuals. The script then flattens the importer's layer-stack output into a
single USD file for IsaacLab.

> If your Isaac Sim version predates the `isaacsim.asset.importer.urdf`
> module name, use `omni.isaac.urdf` instead (same config fields).

### 2. Use in IsaacLab

Point `mate_connectors_config.py` at the USD and spawn it:

```python
from mate_connectors_config import MATE_CONNECTORS_CFG

scene.articulations["mate_connectors"] = MATE_CONNECTORS_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot"
)
```

The 4 actuated joints use implicit PD drives (stiffness 100, damping 10,
effort 1.8 N·m). Tune with Isaac Sim's Gain Tuner extension.

### 3. Verify

- Open the USD in Isaac Sim and confirm the arm moves when you drive
  `revolute_base` / `revolute_left` / `revolute_right` / `rev_end_effector1`.
- Check the mimic joints track their sources without drift.
- If links jitter, raise `solver_position_iteration_count` or lower
  `link_density`.

## Exact kinematics (optional)

For exact four-bar kinematics, keep the mimic joints but drive them from a
controller that implements the solver equations from
`left_four_bar_kinematics.py` (they are closed-form for the left loop and
gripper; the boom loop needs the Newton-Raphson solve). The joint names in
the USD match the URDF, so the solver can be ported directly.
