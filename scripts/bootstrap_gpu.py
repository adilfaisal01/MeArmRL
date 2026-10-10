#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-command GPU setup: download Isaac Sim + Isaac Lab by building the container.

You do **not** need to read docker/README.md first -- this script runs the same
checks the doctor does, walks you through the single interactive step (the NGC
login), builds ``mearmrl:local`` (which pulls Isaac Sim 5.1.0 from nvcr.io and
clones Isaac Lab v2.3.2 into it), and then smoke-tests that the GPU reaches the
container.

    python scripts/bootstrap_gpu.py

It is safe to re-run: the build is layered, an existing image is reused, and it
stops early with a plain-English reason if this machine cannot do the GPU path.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import doctor  # noqa: E402  (deliberately after the sys.path insert; same directory)

IMAGE = os.environ.get("MEARMRL_IMAGE", "mearmrl:local")
NGC_KEY_URL = "https://org.ngc.nvidia.com/setup/api-key"

ENV_FLAGS = [
    "-e",
    "ACCEPT_EULA=Y",
    "-e",
    "PRIVACY_CONSENT=Y",
    "-e",
    "XDG_RUNTIME_DIR=/tmp/xdg",
    "-e",
    "OMNI_KIT_ALLOW_ROOT=1",
    "-e",
    "ISAACLAB_PATH=/IsaacLab",
    "-e",
    "PYTHONPATH=/IsaacLab/source:/MeArmRL/source",
]


def banner(text: str) -> None:
    print(f"\n{'=' * 68}\n  {text}\n{'=' * 68}")


def show(checks) -> None:
    for check in checks:
        print(f"  {doctor.SYMBOL[check.status]} {check.title}")
        if check.detail:
            print(f"         {check.detail}")


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the MeArmRL GPU container (Isaac Sim + Isaac Lab).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--yes", action="store_true", help="Do not ask before the (long) build.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the image exists already.")
    parser.add_argument("--skip-verify", action="store_true", help="Do not smoke-test the GPU afterwards.")
    args = parser.parse_args(argv)

    banner("MeArmRL setup (GPU path)")
    print("  Builds the container with Isaac Sim + Isaac Lab in it.")
    print("  First run: ~15-20 minutes and ~11 GB. Needs an NVIDIA GPU + driver.")
    print("  Re-running is cheap -- the image is cached. You can stop with Ctrl+C.")

    # --- 1. checks ------------------------------------------------------------------
    banner("[1/4] Checking this machine")
    platform_info = doctor.detect_platform()
    checks = (
        doctor.check_platform(platform_info)
        + doctor.check_gpu()
        + doctor.check_docker()
        + doctor.check_image()
        + doctor.check_ngc_auth()
    )
    show(checks)

    if platform_info["group"] == "macos":
        banner("STOP: the GPU path cannot run on macOS")
        print("  Isaac Sim needs an NVIDIA GPU, and macOS has none.")
        print("  Do the CPU half of the lab here, and train on the university cluster:")
        print("    bash scripts/lab_setup.sh")
        print("  Details: docs/class-onboarding.md")
        return 2

    ngc = next(c for c in checks if c.key == "ngc_auth")
    blocking = [c for c in checks if c.status == doctor.FAIL and c.key != "ngc_auth"]
    if blocking:
        banner("STOP: fix these first, then re-run this script")
        for check in blocking:
            print(f"  {check.title}: {check.detail}")
            for line in check.fix:
                print(f"    -> {line}")
        print("\n  Then re-run:  python scripts/bootstrap_gpu.py")
        return 1

    image = next(c for c in checks if c.key == "image")
    have_image = image.status == doctor.OK

    # --- 2. NGC login ---------------------------------------------------------------
    banner("[2/4] NGC login (one-time)")
    if ngc.status == doctor.OK:
        print("  Already logged in to nvcr.io.")
    else:
        print("  The Isaac Sim base image is hosted on nvcr.io and needs a free NGC account.")
        print(f"  1. Create/copy an API key:  {NGC_KEY_URL}")
        print("  2. At the prompt below type EXACTLY:")
        print("        Username: $oauthtoken          <- literally this, not your email")
        print("        Password: <paste your API key>")
        if not sys.stdin.isatty():
            print("\n  (Not an interactive terminal.) Run this yourself, then re-run:")
            print("    docker login nvcr.io")
            return 1
        if run(["docker", "login", "nvcr.io"]) != 0:
            print("\n  Login failed. Nothing else can work without it. Re-run when ready.")
            return 1

    # --- 3. build -------------------------------------------------------------------
    banner("[3/4] Building the image")
    if have_image and not args.force:
        print(f"  {IMAGE} already exists -- skipping the build (use --force to rebuild).")
    else:
        print(f"  Downloading Isaac Sim 5.1.0 and cloning Isaac Lab v2.3.2 into {IMAGE}.")
        print("  This is the slow part -- please leave it running. Safe to re-run if it stops.")
        if not args.yes and sys.stdin.isatty():
            answer = input("\n  Start the build now? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("  Cancelled. Re-run when ready.")
                return 1
        if run(["docker", "build", "-t", IMAGE, "-f", "docker/Dockerfile", "."]) != 0:
            print("\n  Build failed -- the reason is in the last lines above.")
            print("  Guide: docker/README.md (section 'Build gotchas').")
            return 1
        print(f"\n  Built {IMAGE}.")

    # --- 4. verify ------------------------------------------------------------------
    banner("[4/4] Smoke test: can the container use the GPU?")
    if args.skip_verify:
        print("  Skipped (--skip-verify).")
    else:
        rc = run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                *ENV_FLAGS,
                IMAGE,
                "/IsaacLab/isaaclab.sh",
                "-p",
                "scripts/list_envs.py",
                "--keyword",
                "Mearmrl",
            ]
        )
        if rc != 0:
            print("\n  The container ran, but the GPU smoke test failed.")
            print("  Usual causes: the NVIDIA driver, or the container toolkit.")
            print("  Diagnose with:  python scripts/doctor.py --gpu-smoke")
            return 1

    banner("Done -- you are ready to train")
    print("  Train locally (headless):")
    print("    cd docker && docker compose up")
    print("  Interactive shell inside the container:")
    print("    docker compose --profile dev run --rm dev")
    print("  Or train on the university cluster instead (no local GPU needed):")
    print("    TASK=Mearmrl-Reach-Student-v0 sbatch slurm/train.sbatch")
    print("  More: docker/README.md · slurm/README.md · docs/class-onboarding.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
