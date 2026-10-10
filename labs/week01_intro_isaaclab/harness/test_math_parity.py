# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guard against drift between mdp/math_compat.py and the real Isaac Lab math.

On a laptop (no Isaac Lab) this skips: the CPU fallbacks are validated where the
authoritative implementation exists. Inside the container, set
``MEARMRL_REQUIRE_PARITY=1`` (``scripts/test_container.sh`` does) to turn a
missing ImportError into a failure instead of a silent skip.
"""

# isaaclab exists only inside the container; this module is built to skip (or, with
# MEARMRL_REQUIRE_PARITY=1, fail) when it is absent, so its absence is not an error.
# pyright: reportMissingImports=false

from __future__ import annotations

import os

import pytest
import torch

# In CI and on a laptop an absent Isaac Lab is expected; in the container it is required.
_REQUIRE_PARITY = os.environ.get("MEARMRL_REQUIRE_PARITY") == "1"
try:
    from isaaclab.utils.math import combine_frame_transforms as ref_combine  # noqa: E402
    from isaaclab.utils.math import quat_error_magnitude as ref_qem  # noqa: E402
    from isaaclab.utils.math import quat_mul as ref_qmul  # noqa: E402
    from isaaclab.utils.math import wrap_to_pi as ref_wrap  # noqa: E402
except ImportError:
    if _REQUIRE_PARITY:
        raise
    pytest.skip("Isaac Lab not installed; parity is checked in-container", allow_module_level=True)

from labkit import load_mdp_module  # noqa: E402

compat = load_mdp_module("math_compat")


def _random_quat(n: int) -> torch.Tensor:
    q = torch.randn(n, 4)
    return q / q.norm(dim=-1, keepdim=True)


def test_wrap_to_pi_matches():
    angles = torch.linspace(-4 * torch.pi, 4 * torch.pi, 257)
    assert torch.allclose(compat.wrap_to_pi(angles), ref_wrap(angles), atol=1e-6)


def test_quat_mul_matches():
    torch.manual_seed(0)
    q1, q2 = _random_quat(64), _random_quat(64)
    assert torch.allclose(compat.quat_mul(q1, q2), ref_qmul(q1, q2), atol=1e-6)


def test_quat_error_magnitude_matches():
    torch.manual_seed(1)
    q1, q2 = _random_quat(64), _random_quat(64)
    assert torch.allclose(compat.quat_error_magnitude(q1, q2), ref_qem(q1, q2), atol=1e-6)


def test_combine_frame_transforms_matches():
    torch.manual_seed(2)
    t01, q01, t12, q12 = torch.randn(64, 3), _random_quat(64), torch.randn(64, 3), _random_quat(64)
    got_t, got_q = compat.combine_frame_transforms(t01, q01, t12, q12)
    ref_t, ref_q = ref_combine(t01, q01, t12, q12)
    assert torch.allclose(got_t, ref_t, atol=1e-6)
    assert torch.allclose(got_q, ref_q, atol=1e-6)
