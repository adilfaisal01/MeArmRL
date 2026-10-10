# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Set up a CPU-only Python environment for the Week-1 lab and run its checks.
# No GPU, no Isaac Sim, no Docker.
#
# Usage (Windows):
#   scripts\lab_setup.cmd                 <- recommended: avoids the ExecutionPolicy
#   powershell -ExecutionPolicy Bypass -File scripts\lab_setup.ps1

$ErrorActionPreference = "Stop"
# Let native commands (python, pytest) return non-zero without throwing, so we
# can report their exit codes ourselves.
$PSNativeCommandUseErrorActionPreference = $false

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Venv = if ($env:LAB_VENV) { $env:LAB_VENV } else { Join-Path $RepoRoot ".venv-lab" }

# Prefer the `py` launcher (installed by python.org builds, and on PATH even when
# `python` is the Microsoft Store stub); fall back to `python`.
$BaseExe = "python"
$BaseArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    $BaseExe = "py"
    $BaseArgs = @("-3")
} elseif (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.10+ not found. Install from https://www.python.org/downloads/windows/ and tick 'Add python.exe to PATH', then re-run."
}

Write-Host "=== [1/3] Creating virtualenv at $Venv ==="
if (-not (Test-Path $Venv)) {
    & $BaseExe @BaseArgs -m venv $Venv
}
# Call the venv interpreter directly instead of running Activate.ps1: the default
# Windows ExecutionPolicy ("Restricted") blocks venv activation with
# "running scripts is disabled on this system".
$Py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "virtualenv interpreter not found at $Py"
}

Write-Host "=== [2/3] Installing CPU dependencies (torch, numpy, pyyaml, pytest) ==="
& $Py -m pip install --upgrade pip
& $Py -m pip install torch numpy pyyaml pytest

Write-Host "=== [3/3] Running lab checks ==="
& $Py "$RepoRoot\scripts\check_week01.py" --static
$staticRc = $LASTEXITCODE
Write-Host "    (static exit: $staticRc; a fresh checkout reports 'none of the student files were modified')"
& $Py -m pytest "$RepoRoot\labs\week01_intro_isaaclab\harness" -q

Write-Host ""
Write-Host "Setup complete. Run things with the venv interpreter:"
Write-Host "    $Py labs\week01_intro_isaaclab\harness\playground.py"
Write-Host "    $Py -m pytest labs\week01_intro_isaaclab\harness -q"
Write-Host "To activate it instead (may need: Set-ExecutionPolicy -Scope Process RemoteSigned):"
Write-Host "    . $Venv\Scripts\Activate.ps1"
