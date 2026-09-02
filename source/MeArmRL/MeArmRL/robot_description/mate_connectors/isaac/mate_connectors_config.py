"""IsaacLab Articulation configuration for the MATE connectors robotic arm.

This config is generated for the cleaned URDF
(``urdf/mate_connectors_clean.urdf``). It assumes the USD produced by
``convert_urdf_to_usd.py`` is placed at ``assets/mate_connectors.usd``.

Actuated DOFs (4):
    revolute_base, revolute_left, revolute_right, rev_end_effector1

All other movable joints are driven by ``<mimic>`` relationships baked into
the URDF, so they are NOT actuated here.

Usage (inside an IsaacLab environment):
    from mate_connectors_config import MATE_CONNECTORS_CFG
    scene.articulations["mate_connectors"] = MATE_CONNECTORS_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
"""
from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass

# Absolute path to the flattened USD asset, resolved from this file's
# location so it works regardless of the current working directory.
_USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "mate_connectors.usd")

# ---------------------------------------------------------------------------
# Joint limits (rad) taken from the cleaned URDF.
# ---------------------------------------------------------------------------
_JOINT_LIMITS = {
    "revolute_base": (-1.5708, 1.5708),
    "revolute_left": (-1.07998, 0.141749),
    "revolute_right": (-1.12601, 0.706583),
    "rev_end_effector1": (-0.575959, 0.418879),
}

# Actuated joint names in the order they appear in the USD articulation.
ACTUATED_JOINT_NAMES = [
    "revolute_base",
    "revolute_left",
    "revolute_right",
    "rev_end_effector1",
]

# Default joint positions (rad). Zero pose matches the URDF home pose.
DEFAULT_JOINT_POS = {name: 0.0 for name in ACTUATED_JOINT_NAMES}

# SG90-class servos: ~1.8 kg*cm stall torque at ~0.1 m arm -> ~1.8 N*m.
# Effort is capped conservatively; tune with the Gain Tuner extension.
_JOINT_EFFORT = 1.8
_JOINT_VELOCITY = 1.0  # rad/s (SG90 ~ 0.1 s/60 deg)


@configclass
class MateConnectorsCfg(ArticulationCfg):
    """Configuration for the MATE connectors robotic arm."""

    # -- Initial state ---------------------------------------------------------
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=DEFAULT_JOINT_POS,
        joint_vel={name: 0.0 for name in ACTUATED_JOINT_NAMES},
    )

    # -- Actuators -------------------------------------------------------------
    # Implicit (PD) drives on the 4 actuated joints. The mimic joints are
    # locked by PhysX and need no drive.
    # NOTE: gains are ~50x the datasheet-ish values because the PhysX implicit
    # drive behaves ~2500x weaker than configured on these gram-scale links
    # (measured steady-state droop); retune with the Gain Tuner extension.
    actuators = {
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ACTUATED_JOINT_NAMES,
            effort_limit=100.0,
            velocity_limit=_JOINT_VELOCITY,
            stiffness={
                name: 5000.0 for name in ACTUATED_JOINT_NAMES
            },
            damping={
                name: 200.0 for name in ACTUATED_JOINT_NAMES
            },
        ),
    }

    # -- Articulation ----------------------------------------------------------
    # Set by the caller via .replace() with the environment prim path.
    prim_path = "{ENV_REGEX_NS}/Robot"
    spawn = sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # Gram-scale links need many solver iterations; the defaults (8/0)
            # leave the drives effectively unable to hold pose against gravity.
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=4,
        ),
    )


MATE_CONNECTORS_CFG = MateConnectorsCfg()
