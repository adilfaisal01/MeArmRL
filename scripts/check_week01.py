# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Week 1 lab checker: verify the student config only touches rewards/curriculum.

Run inside the container:

    /IsaacLab/isaaclab.sh -p scripts/check_week01.py
    /IsaacLab/isaaclab.sh -p scripts/check_week01.py --smoke   # + short zero-action run

Exit code 0 = OK, 1 = problems found.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Week 1 lab config checker.")
parser.add_argument("--smoke", action="store_true", default=False, help="Also run a short zero-action smoke test.")
parser.add_argument("--num_steps", type=int, default=10, help="Number of steps for the smoke test.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments for the smoke test.")
parser.add_argument(
    "--base-branch", type=str, default="master", help="Git branch to diff against for the file-allowlist check."
)
parser.add_argument(
    "--allow",
    action="append",
    default=[],
    help="Extra file path (relative to repo root) allowed in the git diff. Repeatable.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import dataclasses
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import MeArmRL.tasks  # noqa: F401
import torch

from isaaclab.managers import CurriculumTermCfg as CurrTerm

from isaaclab_tasks.utils import parse_env_cfg

# Sections that must be identical to the base env.
FROZEN_SECTIONS = ("scene", "observations", "actions", "commands", "terminations", "events", "sim", "viewer")
# Scalar settings that must match the base env.
FROZEN_SETTINGS = ("episode_length_s", "decimation")
# The only file students may modify (relative to the repo root).
STUDENT_FILE = "source/MeArmRL/MeArmRL/tasks/manager_based/mearmrl/reach_student_cfg.py"
# Repo root (two levels up from this script).
REPO_ROOT = Path(__file__).resolve().parent.parent


def stable_repr(obj):
    """Process-stable, JSON-safe representation of a config object."""
    if callable(obj) and not isinstance(obj, type):
        return f"<func {obj.__module__}.{obj.__name__}>"
    if isinstance(obj, dict):
        return {str(k): stable_repr(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [stable_repr(v) for v in obj]
    if dataclasses.is_dataclass(obj):
        return {f.name: stable_repr(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    return f"<{type(obj).__module__}.{type(obj).__name__}>"


def check_git_diff(problems):
    """Verify the git diff only touches the student config file."""
    allowed = {STUDENT_FILE, *args_cli.allow}
    result = subprocess.run(
        ["git", "diff", "--name-only", args_cli.base_branch],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        problems.append(
            f"[GIT] could not diff against '{args_cli.base_branch}' ({result.stderr.strip()}). "
            "Commit your work on a branch and make sure the base branch exists."
        )
        return
    changed = [line for line in result.stdout.splitlines() if line.strip()]
    outside = [path for path in changed if path not in allowed]
    if outside:
        problems.append(f"[GIT] files outside the student config were modified: {outside}")
    if STUDENT_FILE not in changed:
        problems.append(f"[GIT] '{STUDENT_FILE}' was not modified — did you do the assignment?")


def main():
    """Check the student config against the base reach environment."""
    problems = []

    base_cfg = parse_env_cfg("Mearmrl-Reach-v0", device=args_cli.device, num_envs=None)
    student_cfg = parse_env_cfg("Mearmrl-Reach-Student-v0", device=args_cli.device, num_envs=None)

    # 1) frozen sections must match the base env
    for section in FROZEN_SECTIONS:
        if stable_repr(getattr(base_cfg, section)) != stable_repr(getattr(student_cfg, section)):
            problems.append(f"[FROZEN] section '{section}' differs from the base env. Revert it.")

    # 2) scalar settings must match the base env
    for setting in FROZEN_SETTINGS:
        if getattr(base_cfg, setting) != getattr(student_cfg, setting):
            problems.append(f"[FROZEN] setting '{setting}' differs from the base env. Revert it.")

    # 3) curriculum terms must only target reward terms that exist
    for field in dataclasses.fields(student_cfg.curriculum):
        term = getattr(student_cfg.curriculum, field.name)
        if not isinstance(term, CurrTerm):
            continue
        target = term.params.get("term_name")
        if target is not None and not hasattr(student_cfg.rewards, target):
            problems.append(
                f"[CURRICULUM] term '{field.name}' targets reward term '{target}', "
                "which does not exist in StudentRewardsCfg."
            )

    # 4) git diff must only touch the student config file
    check_git_diff(problems)

    # 5) optional smoke test: run the student env with zero actions
    if args_cli.smoke:
        smoke_cfg = parse_env_cfg("Mearmrl-Reach-Student-v0", device=args_cli.device, num_envs=args_cli.num_envs)
        env = gym.make("Mearmrl-Reach-Student-v0", cfg=smoke_cfg)
        env.reset()
        with torch.inference_mode():
            for _ in range(args_cli.num_steps):
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)
                if not torch.isfinite(env.unwrapped.reward_buf).all():
                    problems.append("[SMOKE] non-finite reward detected during the zero-action run.")
                    break
        env.close()
        if not any("[SMOKE]" in p for p in problems):
            print(f"[SMOKE] ran {args_cli.num_steps} zero-action steps on Mearmrl-Reach-Student-v0 OK")

    # report
    if problems:
        print("check_week01.py FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("check_week01.py OK: only rewards/curriculum differ from the base env.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()
