# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free scaffolding for the Week-1 lab: a fake env + loaders for the real reward code.

This module lets the lab's reward functions run on a plain CPU Python with only
``torch`` installed -- no Isaac Sim, no container, no GPU. It is deliberately
small: it fakes exactly the slice of the Isaac Lab API that the reward terms
touch, and loads the **real** source files so tests exercise production code,
not a copy.

Two loaders:

* :func:`load_reward_module` -- imports ``mdp/rewards.py`` without importing the
  ``mdp`` package ``__init__`` (which would pull in ``isaaclab``). It registers a
  synthetic package so the module's relative imports resolve normally.
* :func:`load_student_module` -- parses the student's ``reach_student_cfg.py``
  and executes only its top-level function definitions and simple assignments,
  injecting ``torch`` and the pure math helpers. This keeps the lab's
  "edit only these files" contract while making the student's own source
  testable on CPU.

Nothing here is used by the container build or the training run.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import torch

# --- paths (this file is <repo>/labs/week01_intro_isaaclab/harness/labkit.py) ---
REPO_ROOT = Path(__file__).resolve().parents[3]
MDP_DIR = REPO_ROOT / "source/MeArmRL/MeArmRL/tasks/manager_based/mearmrl/mdp"
STUDENT_CFG = REPO_ROOT / "source/MeArmRL/MeArmRL/tasks/manager_based/mearmrl/reach_student_cfg.py"
SKRL_CFG = REPO_ROOT / "source/MeArmRL/MeArmRL/tasks/manager_based/mearmrl/agents/skrl_ppo_student_cfg.yaml"

_MATH_NAMES = ("combine_frame_transforms", "quat_error_magnitude", "quat_mul", "wrap_to_pi")
_PKG = "_mearmrl_mdp"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _mdp_package() -> types.ModuleType:
    """Register a synthetic package for ``mdp/`` without running its ``__init__``.

    ``mdp/__init__.py`` does ``from isaaclab.envs.mdp import *``, which fails on a
    laptop. Registering the package object by hand lets us load its submodules
    (and their relative imports) while skipping that line entirely.
    """
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(MDP_DIR)]  # type: ignore[attr-defined]
        pkg.__package__ = _PKG
        sys.modules[_PKG] = pkg
    return sys.modules[_PKG]


def load_mdp_module(name: str = "math_compat"):
    """Import a module from ``mdp/`` by path, bypassing the package ``__init__``."""
    _mdp_package()
    full = f"{_PKG}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, MDP_DIR / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {MDP_DIR / f'{name}.py'}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = _PKG
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


def load_reward_module():
    """Return the lab's real ``mdp/rewards.py`` module (pure-torch importable)."""
    load_mdp_module("math_compat")  # so rewards.py's relative fallback resolves
    return load_mdp_module("rewards")


class _StubSceneEntityCfg:
    """Minimal stand-in so student assignments referencing ``SceneEntityCfg`` run."""

    def __init__(self, name: str, body_names=None, joint_names=None):
        self.name = name
        self.body_names = body_names
        self.joint_names = joint_names


def _identity_quat_batch(n: int) -> torch.Tensor:
    """Return an ``(n, 4)`` batch of identity quaternions in Isaac Lab's (w, x, y, z) order."""
    return torch.tensor([IDENTITY_QUAT], dtype=torch.float32).expand(n, -1).contiguous()


def _exec_segment(segment: str, filename: Path, namespace: dict) -> None:
    # `from __future__ import annotations` makes the function annotations lazy, so a
    # signature mentioning isaaclab types imports cleanly without those types existing.
    src = "from __future__ import annotations\n" + segment
    # Compile only to attach the real filename, so a traceback points at
    # reach_student_cfg.py. `segment` comes from a file already inside this repo.
    code = compile(src, str(filename), "exec")  # pi-lens-ignore: no-compile-call
    exec(code, namespace)  # noqa: S102 - trusted in-repo source


def load_student_module(path: Path | None = None) -> dict:
    """Execute the student config's top-level functions/assignments in a CPU namespace.

    Imports and class definitions are skipped (they need Isaac Lab). Everything the
    student's reward functions actually use -- ``torch`` and the pure math helpers --
    is injected, so ``ns["reach_progress_bonus"]`` is the student's real function.
    """
    path = Path(path) if path is not None else STUDENT_CFG
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise RuntimeError(f"could not parse {path}: {exc}") from exc

    math_mod = load_mdp_module("math_compat")
    namespace: dict = {
        "__name__": "_student_cfg",
        "__builtins__": __builtins__,
        "torch": torch,
        "SceneEntityCfg": _StubSceneEntityCfg,
    }
    for helper in _MATH_NAMES:
        namespace[helper] = getattr(math_mod, helper)

    functions: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            segment = ast.get_source_segment(source, node)
            if segment is None:  # pragma: no cover - defensive
                continue
            _exec_segment(segment, path, namespace)
            functions.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # Best-effort: a constant like `_EE_BODY = "effector_claw_right"` is useful;
            # anything that needs Isaac Lab is skipped rather than aborting the load.
            segment = ast.get_source_segment(source, node)
            if segment is None:  # pragma: no cover - defensive
                continue
            try:
                _exec_segment(segment, path, namespace)
            except Exception:  # noqa: BLE001 - optional module-level constants
                continue
    namespace["__student_functions__"] = functions
    return namespace


def student_source(path: Path | None = None) -> str:
    """Return the raw text of the student config."""
    return (Path(path) if path is not None else STUDENT_CFG).read_text(encoding="utf-8")


def student_reward_is_implemented(fn, *, epsilon: float = 1e-9) -> bool:
    """True once the student's shaping term gives a nonzero bonus for a real gap.

    Detection is behavioral, not comment-based: two of the three ``TODO(week01)``
    markers sit on lines that are already correct (the reset guard and the cache
    update), so scanning for those comments would never let a finished lab through.
    """
    env = make_env(ee_pos=[(1.0, 0.0, 0.0)], target=[(0.0, 0.0, 0.0)])
    fn(env, command_name="ee_pose", asset_cfg=Cfg())  # seed the cached distance
    env.scene["robot"].data.body_pos_w[0, 0, 0] = 0.5  # 0.5 m closer
    return abs(fn(env, command_name="ee_pose", asset_cfg=Cfg()).item()) > epsilon


# ---------------------------------------------------------------------------
# Fake env: exactly the Isaac Lab surface the reward terms touch
# ---------------------------------------------------------------------------
IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)  # (w, x, y, z), Isaac Lab convention


class _Data:
    """Per-asset tensors the reward terms read (declared so static checkers see them)."""

    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor

    def __init__(self, **fields: torch.Tensor):
        self.__dict__.update(fields)


class _Asset:
    def __init__(self, data: _Data):
        self.data = data


class _Scene:
    def __init__(self, assets: dict[str, _Asset]):
        self._assets = assets

    def __getitem__(self, name: str) -> _Asset:
        return self._assets[name]


class _CommandManager:
    def __init__(self, commands: dict[str, torch.Tensor]):
        self._commands = commands

    def get_command(self, name: str) -> torch.Tensor:
        return self._commands[name]


class FakeEnv:
    """Stands in for ``ManagerBasedRLEnv`` for the reward terms used in the lab."""

    def __init__(self, num_envs: int, device: str = "cpu"):
        self.num_envs = num_envs
        self.device = device
        self.scene = _Scene({})
        self.command_manager = _CommandManager({})
        self.episode_length_buf = torch.ones(num_envs, dtype=torch.long)

    def add_asset(self, name: str, **fields: torch.Tensor) -> None:
        self.scene._assets[name] = _Asset(_Data(**fields))

    def set_command(self, name: str, command: torch.Tensor) -> None:
        self.command_manager._commands[name] = command


class Cfg:
    """Stand-in for ``SceneEntityCfg`` (``.name``, ``.body_ids``, ``.joint_ids``)."""

    def __init__(self, name: str = "robot", body: int = 0, joints: list[int] | None = None):
        self.name = name
        self.body_ids = [body]
        self.joint_ids = joints if joints is not None else [0]


def _as_2d(value, width: int | None, num_envs: int | None = None) -> torch.Tensor:
    t = torch.as_tensor(value, dtype=torch.float32)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    if width is not None and t.shape[-1] != width:
        raise ValueError(f"expected last dim {width}, got {t.shape}")
    if num_envs is not None and t.shape[0] == 1 and num_envs > 1:
        t = t.expand(num_envs, -1).contiguous()
    return t


def make_env(
    ee_pos,
    target=None,
    *,
    ee_quat=None,
    root_pos=None,
    root_quat=None,
    joint_pos=None,
    joint_vel=None,
    episode_length: int = 5,
    command_name: str = "ee_pose",
    asset_name: str = "robot",
) -> FakeEnv:
    """Build a FakeEnv for the reach task.

    Args:
        ee_pos: End-effector positions, shape (N, 3) (or (3,) broadcast to all envs).
        target: Commanded end-effector positions, shape (N, 3) (or (3,)). Defaults to the origin.
        ee_quat: End-effector orientations (w, x, y, z), shape (N, 4). Defaults to identity.
        root_pos/root_quat: Robot root pose. Defaults to origin / identity.
        joint_pos/joint_vel: Joint states, shape (N, J). Default per (N, 1) zeros.
        episode_length: Initial ``env.episode_length_buf`` value.
    """
    ee = _as_2d(ee_pos, 3)
    n = ee.shape[0]
    target_pos = _as_2d(target, 3, n) if target is not None else torch.zeros(n, 3)

    command = torch.zeros(n, 7)
    command[:, :3] = target_pos
    command[:, 3] = 1.0  # identity quaternion (w, x, y, z)

    joint_pos = _as_2d(joint_pos, None, n) if joint_pos is not None else torch.zeros(n, 1)
    joint_vel = _as_2d(joint_vel, None, n) if joint_vel is not None else torch.zeros(n, 1)
    ee_quat = _as_2d(ee_quat, 4, n) if ee_quat is not None else _identity_quat_batch(n)

    env = FakeEnv(n)
    env.add_asset(
        asset_name,
        root_pos_w=_as_2d(root_pos, 3, n) if root_pos is not None else torch.zeros(n, 3),
        root_quat_w=_as_2d(root_quat, 4, n) if root_quat is not None else _identity_quat_batch(n),
        body_pos_w=ee.unsqueeze(1),
        body_quat_w=ee_quat.unsqueeze(1),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
    )
    env.set_command(command_name, command)
    env.episode_length_buf = torch.full((n,), episode_length, dtype=torch.long)
    return env
