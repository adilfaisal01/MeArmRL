# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shape/boundary tests for the lab's production reward terms -- on CPU, no GPU.

These load the real ``mdp/rewards.py`` and drive it through a fake env, so they
verify the code students are asked to reason about in Part 1 of the lab.
"""

from __future__ import annotations

import math

import pytest
import torch
from labkit import Cfg, load_reward_module, make_env

mdp_rewards = load_reward_module()
ROBOT = Cfg("robot", body=0)
JOINTS = Cfg("robot", joints=[0])

# A 90-degree rotation about z, in Isaac Lab's (w, x, y, z) convention.
QUAT_Z90 = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))


def test_position_error_is_zero_at_target():
    env = make_env(ee_pos=[(0.2, 0.0, 0.3)], target=[(0.2, 0.0, 0.3)])
    err = mdp_rewards.position_command_error(env, "ee_pose", ROBOT)
    assert err.item() == pytest.approx(0.0, abs=1e-6)


def test_position_error_grows_with_distance():
    distances = [0.0, 0.1, 0.5, 1.0, 2.0]
    errors = [
        mdp_rewards.position_command_error(
            make_env(ee_pos=[(d, 0.0, 0.0)], target=[(0.0, 0.0, 0.0)]), "ee_pose", ROBOT
        ).item()
        for d in distances
    ]
    assert errors == sorted(errors)  # monotonically increasing penalty
    assert errors[0] == pytest.approx(0.0, abs=1e-6)


def test_tanh_bonus_is_bounded_and_peaks_at_target():
    at_target = mdp_rewards.position_command_error_tanh(
        make_env(ee_pos=[(0.0, 0.0, 0.0)], target=[(0.0, 0.0, 0.0)]), 0.1, "ee_pose", ROBOT
    ).item()
    far = mdp_rewards.position_command_error_tanh(
        make_env(ee_pos=[(5.0, 0.0, 0.0)], target=[(0.0, 0.0, 0.0)]), 0.1, "ee_pose", ROBOT
    ).item()
    assert at_target == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= far < at_target  # bonus decays to zero with distance


def test_orientation_error_is_zero_for_matching_orientation():
    env = make_env(ee_pos=[(0, 0, 0)], target=[(0, 0, 0)], ee_quat=[QUAT_Z90])
    env.command_manager.get_command("ee_pose")[:, 3:7] = torch.tensor([QUAT_Z90], dtype=torch.float32)
    assert mdp_rewards.orientation_command_error(env, "ee_pose", ROBOT).item() == pytest.approx(0.0, abs=1e-6)


def test_orientation_error_reports_the_rotation_angle():
    # ee at identity, commanded 90 deg about z -> angular error is pi/2 rad.
    env = make_env(ee_pos=[(0, 0, 0)], target=[(0, 0, 0)], ee_quat=[(1.0, 0.0, 0.0, 0.0)])
    env.command_manager.get_command("ee_pose")[:, 3:7] = torch.tensor([QUAT_Z90], dtype=torch.float32)
    assert mdp_rewards.orientation_command_error(env, "ee_pose", ROBOT).item() == pytest.approx(math.pi / 2, abs=1e-5)


def test_command_is_expressed_in_the_robot_body_frame():
    # Root rotated +90 deg about z; a command of +x in the body frame is +y in the world.
    env = make_env(
        ee_pos=[(0.0, 1.0, 0.0)],
        target=[(1.0, 0.0, 0.0)],
        root_quat=[QUAT_Z90],
    )
    assert mdp_rewards.position_command_error(env, "ee_pose", ROBOT).item() == pytest.approx(0.0, abs=1e-5)


def test_joint_pos_target_l2_sums_squared_deviation():
    env = make_env(ee_pos=[(0, 0, 0)], joint_pos=[(0.1, -0.2)])
    both_joints = Cfg("robot", joints=[0, 1])
    value = mdp_rewards.joint_pos_target_l2(env, target=0.0, asset_cfg=both_joints).item()
    assert value == pytest.approx(0.1**2 + 0.2**2, abs=1e-6)


def test_joint_pos_target_l2_wraps_angles():
    env = make_env(ee_pos=[(0, 0, 0)], joint_pos=[(3.0 * math.pi,)])
    value = mdp_rewards.joint_pos_target_l2(env, target=0.0, asset_cfg=JOINTS).item()
    assert value == pytest.approx(math.pi**2, abs=1e-5)  # wrapped to pi, not 3*pi
