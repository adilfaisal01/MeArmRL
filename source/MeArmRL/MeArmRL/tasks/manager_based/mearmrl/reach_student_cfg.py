# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""STUDENT-EDITABLE config for the Week 1 reward-engineering lab.

This is the ONLY file you should modify for the lab. The robot, scene,
sensors, actions, observations, commands and terminations are inherited from
the base reach environment and are NOT yours to change.

What you CAN change here:

* the reward weights / parameters in ``StudentRewardsCfg``,
* the curriculum in ``StudentCurriculumCfg``,
* add your own reward function (see the ``reach_progress_bonus`` skeleton
  below) and register it as a new ``RewTerm`` in ``StudentRewardsCfg``.

Run ``scripts/check_week01.py`` before submitting to verify you did not touch
anything outside this file's reward/curriculum surface.
"""

import torch

from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: F401  (used when you uncomment the example term)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms

from . import mdp  # noqa: F401  (used when you add your own reward terms)
from .mearmrl_reach_env_cfg import CurriculumCfg, MearmrlReachEnvCfg, RewardsCfg

# End-effector body used for reach tracking (same as the base env).
_EE_BODY = "effector_claw_right"


def reach_progress_bonus(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward the decrease in end-effector distance to the commanded target.

    This is a *shaping* bonus: it rewards getting closer to the target each
    step, not just being close. Fill in the three blanks (marked ``TODO``).

    Reference: ``position_command_error_tanh`` in ``mdp/rewards.py`` shows how
    to fetch the command and compute the current distance.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)

    # cache the previous distance on the env (per-env state, survives across steps)
    if not hasattr(env, "_reach_prev_dist"):
        env._reach_prev_dist = distance.clone()

    # BLANK 1: progress = previous distance - current distance (positive = getting closer)
    progress = torch.zeros_like(distance)  # TODO(week01): replace this line

    # BLANK 2: zero out progress on the first step of a new episode, where the
    # cached distance still belongs to the previous episode.
    # Hint: a freshly reset env has env.episode_length_buf == 1 at reward time.
    first_step = env.episode_length_buf == 1
    progress = torch.where(first_step, torch.zeros_like(progress), progress)  # TODO(week01): replace this line

    # BLANK 3: remember the current distance for the next step
    env._reach_prev_dist = distance.clone()  # TODO(week01): replace this line

    return progress


@configclass
class StudentRewardsCfg(RewardsCfg):
    """Reward terms for the student task.

    Inherits the 5 baseline terms from ``RewardsCfg``. Tune weights/params
    here, or add your own terms (see the commented example below).
    """

    # Example of registering your own term (uncomment after writing the function):
    # reach_progress = RewTerm(
    #     func=reach_progress_bonus,
    #     weight=0.5,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=_EE_BODY), "command_name": "ee_pose"},
    # )


@configclass
class StudentCurriculumCfg(CurriculumCfg):
    """Curriculum terms for the student task (editable).

    Inherits the baseline penalty ramps. If you rename or remove a reward term
    that a curriculum term targets, ``scripts/check_week01.py`` will flag it.
    """


@configclass
class MearmrlReachStudentEnvCfg(MearmrlReachEnvCfg):
    """Student variant of the reach environment.

    Only ``rewards`` and ``curriculum`` are overridable; every other section is
    inherited frozen from ``MearmrlReachEnvCfg``.
    """

    rewards: StudentRewardsCfg = StudentRewardsCfg()
    curriculum: StudentCurriculumCfg = StudentCurriculumCfg()
