# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral tests for the student's own reward term -- on CPU, no GPU.

These skip until you implement the three ``TODO(week01)`` blanks in
``reach_student_cfg.py``, then run your actual function against the fake env.
They are here to catch the two classic bugs *before* you spend GPU time.
"""

from __future__ import annotations

import pytest
import torch
from labkit import Cfg, FakeEnv, load_student_module, make_env, student_reward_is_implemented

ROBOT = Cfg("robot", body=0)


@pytest.fixture(scope="module")
def progress_bonus():
    namespace = load_student_module()
    fn = namespace.get("reach_progress_bonus")
    if fn is None:
        pytest.skip("no reach_progress_bonus() found in reach_student_cfg.py")
    if not student_reward_is_implemented(fn):
        pytest.skip("reach_progress_bonus returns zero for every gap -- fill in the TODO(week01) blanks")
    return fn


def _ee_x(env: FakeEnv) -> torch.Tensor:
    return env.scene["robot"].data.body_pos_w[0, 0, 0]


def test_stationary_end_effector_earns_no_bonus(progress_bonus):
    env = make_env(ee_pos=[(1.0, 0.0, 0.0)], target=[(0.0, 0.0, 0.0)])
    progress_bonus(env, command_name="ee_pose", asset_cfg=ROBOT)  # seed the cache
    held = progress_bonus(env, command_name="ee_pose", asset_cfg=ROBOT)
    assert held.item() == pytest.approx(0.0, abs=1e-6)


def test_closing_the_gap_earns_a_positive_bonus(progress_bonus):
    env = make_env(ee_pos=[(1.0, 0.0, 0.0)], target=[(0.0, 0.0, 0.0)])
    progress_bonus(env, command_name="ee_pose", asset_cfg=ROBOT)
    _ee_x(env)[...] = 0.6  # moved 0.4 m closer
    reward = progress_bonus(env, command_name="ee_pose", asset_cfg=ROBOT)
    assert reward.item() == pytest.approx(0.4, abs=1e-6)


def test_retreating_is_penalised(progress_bonus):
    env = make_env(ee_pos=[(0.5, 0.0, 0.0)], target=[(0.0, 0.0, 0.0)])
    progress_bonus(env, command_name="ee_pose", asset_cfg=ROBOT)
    _ee_x(env)[...] = 0.9  # moved 0.4 m away from the target
    reward = progress_bonus(env, command_name="ee_pose", asset_cfg=ROBOT)
    assert reward.item() == pytest.approx(-0.4, abs=1e-6)


def test_no_spike_when_a_new_episode_starts(progress_bonus):
    """The reset trap: a stale cached distance must not leak in as a free bonus."""
    env = make_env(ee_pos=[(1.0, 0.0, 0.0)], target=[(0.0, 0.0, 0.0)], episode_length=5)
    progress_bonus(env, command_name="ee_pose", asset_cfg=ROBOT)  # caches distance ~1.0

    # Episode ends and resets: episode_length_buf goes back to 1 and the ee spawns
    # near the target. The cached 1.0 belongs to the *previous* episode and must be
    # discarded rather than reported as a 1.0 reward jump.
    env.episode_length_buf = torch.tensor([1])
    _ee_x(env)[...] = 0.05
    reward = progress_bonus(env, command_name="ee_pose", asset_cfg=ROBOT)
    assert reward.item() == pytest.approx(0.0, abs=1e-6), (
        "bonus spiked on the first step of a new episode -- add the first-step guard"
    )


def test_bonus_reflects_only_the_latest_step(progress_bonus):
    env = make_env(ee_pos=[(1.0, 0.0, 0.0)], target=[(0.0, 0.0, 0.0)])
    progress_bonus(env, command_name="ee_pose", asset_cfg=ROBOT)
    rewards = []
    for x in (0.8, 0.6, 0.4):  # 0.2 m per step
        _ee_x(env)[...] = x
        rewards.append(progress_bonus(env, command_name="ee_pose", asset_cfg=ROBOT).item())
    assert rewards == pytest.approx([0.2, 0.2, 0.2], abs=1e-6)
