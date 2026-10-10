@echo off
REM Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
REM All rights reserved.
REM
REM SPDX-License-Identifier: BSD-3-Clause
REM
REM Windows entry point for the CPU lab setup (Week-1 harness).
REM
REM A fresh Windows machine has PowerShell's ExecutionPolicy set to "Restricted",
REM which blocks .ps1 files ("running scripts is disabled on this system"). Batch
REM files are not policy-gated, so this wrapper runs the PowerShell script with
REM -ExecutionPolicy Bypass for this one invocation only.
REM
REM Usage:
REM   scripts\lab_setup.cmd

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0lab_setup.ps1" %*
exit /b %ERRORLEVEL%
