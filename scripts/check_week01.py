# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Week 1 lab checker: verify the student config only touches rewards/curriculum.

Two modes:

``--static``
    No Isaac Sim, no GPU, no container. Checks the git allowlist, the skrl YAML,
    and (via AST) the reward-function wiring in ``reach_student_cfg.py``. Runs on
    any laptop in well under a second; use it constantly while you iterate.

full (default, inside the container)
    Adds the authoritative checks against the base environment (frozen sections
    and settings, curriculum targets) and an optional zero-action smoke test.

Usage::

    python scripts/check_week01.py --static                                  # laptop
    /IsaacLab/isaaclab.sh -p scripts/check_week01.py                        # container
    /IsaacLab/isaaclab.sh -p scripts/check_week01.py --smoke                # + GPU

Exit code 0 = OK, 1 = problems found. Warnings never fail the check.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import subprocess
import sys
from pathlib import Path

# Sections that must be identical to the base env.
FROZEN_SECTIONS = ("scene", "observations", "actions", "commands", "terminations", "events", "sim", "viewer")
# Scalar settings that must match the base env.
FROZEN_SETTINGS = ("episode_length_s", "decimation")
# The only files students may modify (relative to the repo root).
STUDENT_FILES = {
    "source/MeArmRL/MeArmRL/tasks/manager_based/mearmrl/reach_student_cfg.py",
    "source/MeArmRL/MeArmRL/tasks/manager_based/mearmrl/agents/skrl_ppo_student_cfg.yaml",
}
# Repo root (two levels up from this script).
REPO_ROOT = Path(__file__).resolve().parent.parent
STUDENT_CFG = REPO_ROOT / "source/MeArmRL/MeArmRL/tasks/manager_based/mearmrl/reach_student_cfg.py"
SKRL_CFG = REPO_ROOT / "source/MeArmRL/MeArmRL/tasks/manager_based/mearmrl/agents/skrl_ppo_student_cfg.yaml"


##
# Shared checks (no Isaac Lab, no GPU)
##


def report(problems: list[str], warnings: list[str], ok_message: str) -> int:
    """Print warnings/problems and return the process exit code."""
    for warning in warnings:
        print(f"WARNING: {warning}")
    if problems:
        print("check_week01.py FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    suffix = f" ({len(warnings)} warning(s))" if warnings else ""
    print(f"{ok_message}{suffix}")
    return 0


def check_git_diff(problems: list[str], base_branch: str, allow: list[str]) -> None:
    """Verify the git diff only touches the student config files."""
    allowed = {*STUDENT_FILES, *allow}
    result = subprocess.run(
        ["git", "diff", "--name-only", base_branch],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        problems.append(
            f"[GIT] could not diff against '{base_branch}' ({result.stderr.strip()}). "
            "Commit your work on a branch and make sure the base branch exists."
        )
        return
    changed = [line for line in result.stdout.splitlines() if line.strip()]
    outside = [path for path in changed if path not in allowed]
    if outside:
        problems.append(f"[GIT] files outside the student config were modified: {outside}")
    if not any(path in changed for path in STUDENT_FILES):
        problems.append("[GIT] none of the student files were modified — did you do the assignment?")


def check_skrl_cfg(problems: list[str]) -> None:
    """Validate the student skrl config parses and keeps the log directory."""
    import yaml

    try:
        with open(SKRL_CFG, encoding="utf-8") as f:
            cfg = yaml.full_load(f)
    except Exception as e:  # noqa: BLE001 - report any parse failure
        problems.append(f"[SKRL] could not parse skrl_ppo_student_cfg.yaml: {e}")
        return
    if not isinstance(cfg, dict):
        problems.append("[SKRL] skrl_ppo_student_cfg.yaml must be a YAML mapping.")
        return
    for key in ("models", "memory", "agent", "trainer"):
        if key not in cfg:
            problems.append(f"[SKRL] skrl_ppo_student_cfg.yaml is missing the top-level '{key}' key.")
    if cfg.get("agent", {}).get("experiment", {}).get("directory") != "mearmrl-reach":
        problems.append("[SKRL] keep agent.experiment.directory = 'mearmrl-reach' so logs land where the lab expects.")


def _call_name(node: ast.AST) -> str | None:
    """Best-effort name of a call target (``foo``, ``mdp.foo`` -> ``foo``)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


# Methods that force a device->host sync if called inside a reward term, which
# would stall the whole vectorized step. Advisory only.
_SYNC_METHODS = {"item", "numpy", "cpu", "tolist"}


def check_student_cfg_static(problems: list[str], warnings: list[str]) -> None:
    """AST checks on reach_student_cfg.py: reward wiring + per-step footguns."""
    if not STUDENT_CFG.exists():
        problems.append(f"[AST] student config not found: {STUDENT_CFG}")
        return
    try:
        tree = ast.parse(STUDENT_CFG.read_text(encoding="utf-8"), filename=str(STUDENT_CFG))
    except SyntaxError as e:
        problems.append(f"[AST] {STUDENT_CFG.name} has a syntax error: {e}")
        return

    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    # Every RewTerm(func=<name defined here>) must actually exist, and carry a weight.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _call_name(node.func) == "RewTerm"):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        func = kwargs.get("func")
        if func is None:
            problems.append(f"[AST] line {node.lineno}: RewTerm(...) is missing func=.")
            continue
        name = _call_name(func)
        if "weight" not in kwargs:
            problems.append(f"[AST] line {node.lineno}: RewTerm({name}) is missing weight=.")
        if isinstance(func, ast.Name) and func.id not in defined:
            problems.append(
                f"[AST] line {node.lineno}: reward function '{func.id}' is not defined in {STUDENT_CFG.name}."
            )

    # Per-step footguns inside the student's own reward functions.
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.GeneratorExp)):
                warnings.append(
                    f"[AST] {node.name}:{sub.lineno}: Python-level loop/comprehension in a reward term runs "
                    "per env and per step — vectorize it with torch instead."
                )
            if isinstance(sub, ast.Call) and _call_name(sub.func) in _SYNC_METHODS:
                warnings.append(
                    f"[AST] {node.name}:{sub.lineno}: .{_call_name(sub.func)}() inside a reward term forces a "
                    "host sync every step."
                )


##
# --static mode (laptop: no Isaac Sim, no GPU)
##


def build_static_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_week01.py --static",
        description="Week 1 lab static checks (no Isaac Sim, no GPU).",
    )
    parser.add_argument("--base-branch", type=str, default="master", help="Git branch to diff against.")
    parser.add_argument("--allow", action="append", default=[], help="Extra path allowed in the git diff. Repeatable.")
    return parser


def run_static(argv: list[str]) -> int:
    args = build_static_parser().parse_args(argv)
    problems: list[str] = []
    warnings: list[str] = []
    check_git_diff(problems, args.base_branch, args.allow)
    check_skrl_cfg(problems)
    check_student_cfg_static(problems, warnings)
    ok = "check_week01.py --static OK: git allowlist, skrl config, and reward wiring look good."
    return report(problems, warnings, ok)


##
# Full mode (container: needs Isaac Sim + a GPU)
##


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


def build_full_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Week 1 lab config checker.")
    parser.add_argument("--smoke", action="store_true", default=False, help="Also run a short zero-action smoke test.")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of steps for the smoke test.")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of environments for the smoke test.")
    parser.add_argument(
        "--base-branch", type=str, default="master", help="Git branch to diff against for the file-allowlist check."
    )
    parser.add_argument(
        "--allow", action="append", default=[], help="Extra file path (relative to repo root) allowed in the git diff."
    )
    return parser


def run_full(argv: list[str]) -> int:
    """Boot the simulator first, then run the full checks (imports Isaac Lab)."""
    from isaaclab.app import AppLauncher

    parser = build_full_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args(argv)

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        return _run_full_checks(args_cli)
    finally:
        simulation_app.close()


def _run_full_checks(args_cli) -> int:
    import gymnasium as gym
    import MeArmRL.tasks  # noqa: F401
    import torch

    from isaaclab.managers import CurriculumTermCfg as CurrTerm

    from isaaclab_tasks.utils import parse_env_cfg

    problems: list[str] = []
    warnings: list[str] = []

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

    # 4) git diff must only touch the student config files
    check_git_diff(problems, args_cli.base_branch, args_cli.allow)

    # 5) student skrl config must parse and keep the log directory
    check_skrl_cfg(problems)

    # 6) static AST checks on the student reward functions
    check_student_cfg_static(problems, warnings)

    # 7) optional smoke test: run the student env with zero actions
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

    return report(problems, warnings, "check_week01.py OK: only rewards/curriculum differ from the base env.")


if __name__ == "__main__":
    _argv = sys.argv[1:]
    if "--static" in _argv:
        sys.exit(run_static([arg for arg in _argv if arg != "--static"]))
    sys.exit(run_full(_argv))
