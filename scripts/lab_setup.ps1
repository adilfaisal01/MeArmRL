# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Set up a CPU-only Python environment for the Week-1 lab and run its checks.
# No GPU, no Isaac Sim, no Docker.
#
# Usage:
#   scripts\lab_setup.ps1

$ErrorActionPreference = "Stop"
# Let native commands (python, pytest) return non-zero without throwing, so we
# can report their exit codes ourselves.
$PSNativeCommandUseErrorActionPreference = $false

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Venv = if ($env:LAB_VENV) { $env:LAB_VENV } else { Join-Path $RepoRoot ".venv-lab" }

Write-Host "=== [1/3] Creating virtualenv at $Venv ==="
if (-not (Test-Path $Venv)) {
    python -m venv $Venv
}
& "$Venv\Scripts\Activate.ps1"

Write-Host "=== [2/3] Installing CPU dependencies (torch, pyyaml, pytest) ==="
python -m pip install --upgrade pip
python -m pip install torch pyyaml pytest

Write-Host "=== [3/3] Running lab checks ==="
python "$RepoRoot\scripts\check_week01.py" --static
$staticRc = $LASTEXITCODE
Write-Host "    (static exit: $staticRc; a fresh checkout reports 'none of the student files were modified')"
python -m pytest "$RepoRoot\labs\week01_intro_isaaclab\harness" -q

Write-Host ""
Write-Host "Setup complete. Activate the environment with:"
Write-Host "    $Venv\Scripts\Activate.ps1"
Write-Host "Then explore the rewards with:"
Write-Host "    python labs\week01_intro_isaaclab\harness\playground.py"
