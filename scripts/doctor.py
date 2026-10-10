#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OS-aware preflight for the MeArmRL lab: can this machine run what?

It reports, for your actual operating system, what is missing and the exact
command to fix it -- instead of making students decode a raw Docker error. It
never needs the GPU, the container, or a network connection (the optional
``--gpu-smoke`` test does start a container).

Pure standard library, Python 3.8+, so it runs the same on Linux, WSL2, macOS
and Windows.

Usage::

    python scripts/doctor.py
    python scripts/doctor.py --json        # machine-readable (handy for a class survey)
    python scripts/doctor.py --gpu-smoke   # also run `nvidia-smi` inside the built image

Exit codes::

    0  ready for the GPU training path
    1  fixable problem(s) -- the reported commands will fix them
    2  the GPU path is not possible on this machine (e.g. macOS); use the CPU
       harness now and a cloud/classroom GPU for training
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE = os.environ.get("MEARMRL_IMAGE", "mearmrl:local")

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
_SYMBOL = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[SKIP]"}


class Check:
    """One diagnostic result."""

    def __init__(self, key: str, status: str, title: str, detail: str = "", fix: list[str] | None = None):
        self.key = key
        self.status = status
        self.title = title
        self.detail = detail
        self.fix = fix or []

    def as_dict(self) -> dict:
        return {"key": self.key, "status": self.status, "title": self.title, "detail": self.detail, "fix": self.fix}


def run(cmd: list[str], timeout: int = 20) -> tuple[int, str, str]:
    """Run a command, returning (returncode, stdout, stderr). Never raises."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except FileNotFoundError:
        return 127, "", "command not found"
    except subprocess.TimeoutExpired:
        return 124, "", "timed out"
    except OSError as exc:  # pragma: no cover - defensive
        return 1, "", str(exc)


def detect_platform() -> dict:
    """Classify the host into the groups the guidance branches on."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    release = platform.release().lower()
    is_wsl = system == "linux" and ("microsoft" in release or "WSL_DISTRO_NAME" in os.environ)
    if system == "darwin":
        group, label = "macos", "macOS"
    elif system == "windows":
        group, label = "windows", "Windows (native)"
    elif is_wsl:
        group, label = "wsl2", "Windows (WSL2)"
    elif system == "linux":
        group, label = "linux", "Linux (native)"
    else:
        group, label = "unknown", f"{platform.system() or 'unknown'} ({machine})"
    return {"system": system, "machine": machine, "is_wsl": is_wsl, "group": group, "label": label}


##
# Individual checks
##


def check_platform(p: dict) -> list[Check]:
    detail = f"{p['label']}, {p['machine']}"
    if p["group"] == "macos":
        note = "Isaac Sim needs CUDA; macOS has no NVIDIA GPU support."
        return [Check("platform", WARN, f"Platform: {detail}", note)]
    return [Check("platform", OK, f"Platform: {detail}")]


def _python_install_hint() -> str:
    """An OS-appropriate command for installing Python, for the too-old case."""
    hints = {
        "darwin": "brew install python@3.12   (or https://www.python.org/downloads/macos/)",
        "windows": "winget install Python.Python.3.12   (or python.org; tick 'Add python.exe to PATH')",
    }
    return hints.get(platform.system().lower(), "sudo apt install python3 python3-venv   (Debian/Ubuntu)")


def check_python() -> list[Check]:
    v = platform.python_version()
    # Doctor runs *before* setup, so it must work on an older interpreter in order to
    # report that it is too old -- hence this intentionally "obsolete" version guard.
    if sys.version_info < (3, 10):  # noqa: UP036
        return [
            Check(
                "python",
                FAIL,
                f"Python: {v}",
                "The lab harness needs Python 3.10+.",
                [_python_install_hint(), "Then re-run: python scripts/doctor.py"],
            )
        ]
    return [Check("python", OK, f"Python: {v}")]


def check_gpu() -> list[Check]:
    checks: list[Check] = []
    if shutil.which("nvidia-smi") is None:
        checks.append(
            Check(
                "gpu_driver",
                FAIL,
                "NVIDIA driver (nvidia-smi)",
                "nvidia-smi was not found on PATH.",
                [
                    "Linux: install your distro's NVIDIA driver package, then reboot.",
                    "Windows: install the Windows NVIDIA driver (NOT one inside WSL), then reboot.",
                    "macOS: not supported -- use the CPU harness and a cloud/classroom GPU.",
                ],
            )
        )
        return checks

    rc, out, err = run(["nvidia-smi", "-L"], timeout=15)
    if rc != 0:
        checks.append(
            Check(
                "gpu_driver",
                FAIL,
                "NVIDIA driver (nvidia-smi)",
                (err or out or "nvidia-smi failed.").splitlines()[0],
                ["Install/repair the NVIDIA driver and reboot, then run `nvidia-smi -L`."],
            )
        )
        return checks

    devices = [line for line in out.splitlines() if line.strip()]
    summary = f"{len(devices)} GPU(s): " + "; ".join(devices[:3])
    checks.append(Check("gpu_driver", OK, "NVIDIA driver (nvidia-smi)", summary))
    return checks


def check_docker() -> list[Check]:
    checks: list[Check] = []
    if shutil.which("docker") is None:
        checks.append(
            Check(
                "docker_cli",
                FAIL,
                "Docker CLI",
                "docker was not found on PATH.",
                [
                    "Linux: https://docs.docker.com/engine/install/ then add yourself to the docker group.",
                    "Windows/macOS: install Docker Desktop with the WSL2 backend (Windows).",
                ],
            )
        )
        return checks

    rc, out, _ = run(["docker", "--version"], timeout=15)
    checks.append(Check("docker_cli", OK if rc == 0 else FAIL, "Docker CLI", out or "unknown"))

    rc, out, err = run(["docker", "info", "--format", "{{.ServerVersion}}|{{.OperatingSystem}}"], timeout=20)
    if rc != 0:
        checks.append(
            Check(
                "docker_daemon",
                FAIL,
                "Docker daemon",
                (err or "cannot reach the Docker daemon").splitlines()[0],
                ["Start Docker: `sudo systemctl start docker` (Linux) or launch Docker Desktop."],
            )
        )
        return checks

    version, _, os_name = out.partition("|")
    checks.append(Check("docker_daemon", OK, "Docker daemon", f"server {version} on {os_name}"))

    # NVIDIA container runtime (needed for --gpus all)
    rc, out, _ = run(["docker", "info", "--format", "{{range $k, $v := .Runtimes}}{{$k}} {{end}}"], timeout=20)
    runtimes = out.split() if rc == 0 else []
    if "nvidia" in runtimes:
        checks.append(Check("docker_nvidia_runtime", OK, "NVIDIA container runtime", "nvidia runtime is configured"))
    else:
        checks.append(
            Check(
                "docker_nvidia_runtime",
                FAIL,
                "NVIDIA container runtime",
                "the nvidia runtime is not registered with Docker.",
                [
                    "Install the NVIDIA Container Toolkit, then:",
                    "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker",
                ],
            )
        )

    if "docker desktop" in os_name.lower():
        checks.append(
            Check(
                "docker_desktop",
                WARN,
                "Docker Desktop",
                "docker-compose.yaml sets network_mode: host, which Docker Desktop does not support.",
                ["Use `docker run` (see docker/README.md), or an override file that drops network_mode."],
            )
        )
    return checks


def check_image() -> list[Check]:
    checks: list[Check] = []
    if shutil.which("docker") is None:
        return [Check("image", SKIP, f"Image {IMAGE}", "docker not available")]
    rc, out, _ = run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], timeout=20)
    if rc == 0 and out:
        checks.append(Check("image", OK, f"Image {IMAGE}", f"present ({out[:19]})"))
    else:
        checks.append(
            Check(
                "image",
                WARN,
                f"Image {IMAGE}",
                "not built yet.",
                ["docker login nvcr.io && docker build -t mearmrl:local -f docker/Dockerfile ."],
            )
        )

    rc, out, _ = run(["docker", "info", "--format", "{{.DockerRootDir}}"], timeout=20)
    root = Path(out) if rc == 0 and out else None
    if root is not None and root.exists():
        free_gb = shutil.disk_usage(root).free / 1e9
        status = OK if free_gb >= 20 else (WARN if free_gb >= 10 else FAIL)
        fix = None
        if status != OK:
            fix = [
                "Free up space, or move the Docker data-root to a larger disk.",
                "  docker info --format '{{.DockerRootDir}}'",
            ]
        checks.append(
            Check(
                "disk",
                status,
                "Free disk space",
                f"{free_gb:.0f} GB free at {root} (build needs ~20 GB)",
                fix,
            )
        )
    else:
        checks.append(Check("disk", SKIP, "Free disk space", "Docker data-root not visible from here"))
    return checks


def check_ngc_auth() -> list[Check]:
    cfg = Path.home() / ".docker" / "config.json"
    if not cfg.exists():
        return [
            Check(
                "ngc_auth",
                FAIL,
                "NGC authentication",
                f"{cfg} not found.",
                ["docker login nvcr.io   # free NGC account + API key"],
            )
        ]
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [Check("ngc_auth", WARN, "NGC authentication", f"could not read {cfg}: {exc}")]

    if any("nvcr.io" in key for key in data.get("auths", {})):
        return [Check("ngc_auth", OK, "NGC authentication", "nvcr.io credentials found")]
    if data.get("credsStore") or data.get("credHelpers"):
        return [
            Check(
                "ngc_auth",
                WARN,
                "NGC authentication",
                "credentials are kept in an OS credential store, so they cannot be verified from here.",
                ["If `docker pull nvcr.io/nvidia/isaac-sim:5.1.0` fails, run `docker login nvcr.io`."],
            )
        ]
    return [
        Check(
            "ngc_auth",
            FAIL,
            "NGC authentication",
            "no nvcr.io entry in ~/.docker/config.json.",
            ["docker login nvcr.io   # free NGC account + API key"],
        )
    ]


def check_cpu_path() -> list[Check]:
    venv = REPO_ROOT / ".venv-lab"
    exe = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not exe.exists():
        return [
            Check(
                "cpu_path",
                WARN,
                "CPU lab environment",
                ".venv-lab not created yet.",
                ["bash scripts/lab_setup.sh          # or: scripts\\lab_setup.ps1"],
            )
        ]
    rc, out, _ = run([str(exe), "-c", "import torch; print(torch.__version__)"], timeout=60)
    status = OK if rc == 0 else WARN
    return [Check("cpu_path", status, "CPU lab environment", f"torch {out}" if rc == 0 else "torch not importable")]


##
# Reporting
##


def _first_fixes(checks: list[Check]) -> list[str]:
    fixes: list[str] = []
    for check in checks:
        if check.status == FAIL:
            fixes.append(f"# {check.title}")
            fixes.extend(f"  {line}" for line in check.fix)
    return fixes


def compute_verdict(p: dict, checks: list[Check]) -> dict:
    """Return (exit_code, headline, next_steps)."""
    if p["group"] == "macos":
        return {
            "code": 2,
            "headline": "The GPU path is not possible on macOS -- no CUDA/NVIDIA support.",
            "next": [
                "Start coding now with the CPU harness (no GPU needed):",
                "  bash scripts/lab_setup.sh",
                "  python labs/week01_intro_isaaclab/harness/playground.py",
                "For training, use a cloud GPU or your class's GPU box -- see docs/class-onboarding.md.",
            ],
        }

    if p["group"] == "windows":
        return {
            "code": 1,
            "headline": "Windows cannot pass an NVIDIA GPU to a Linux container directly.",
            "next": [
                "Install WSL2 + the Windows NVIDIA driver, then run Docker inside WSL2:",
                "  wsl --install",
                "  (then re-run this script from inside WSL)",
                "Meanwhile the CPU harness works natively: bash scripts/lab_setup.sh",
            ],
        }

    blockers = [c for c in checks if c.status == FAIL]
    if not blockers:
        return {
            "code": 0,
            "headline": "Ready for the GPU training path.",
            "next": [
                "docker login nvcr.io   # once, if you have not already",
                "docker build -t mearmrl:local -f docker/Dockerfile .",
                "cd docker && docker compose up",
            ],
        }
    return {
        "code": 1,
        "headline": f"Not ready yet -- {len(blockers)} fixable problem(s) above.",
        "next": _first_fixes(checks),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MeArmRL preflight: can this machine run the lab?")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text.")
    parser.add_argument("--gpu-smoke", action="store_true", help="Also run nvidia-smi inside the built image.")
    args = parser.parse_args(argv)

    p = detect_platform()
    checks: list[Check] = []
    checks += check_platform(p)
    checks += check_python()
    checks += check_gpu()
    checks += check_docker()
    checks += check_image()
    checks += check_ngc_auth()
    checks += check_cpu_path()

    if args.gpu_smoke:
        if shutil.which("docker") and run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"])[0] == 0:
            rc, out, err = run(["docker", "run", "--rm", "--gpus", "all", IMAGE, "nvidia-smi", "-L"], timeout=180)
            checks.append(
                Check(
                    "gpu_smoke",
                    OK if rc == 0 else FAIL,
                    "GPU inside the container",
                    out.splitlines()[0] if rc == 0 and out else (err or "failed").splitlines()[0],
                )
            )
        else:
            checks.append(Check("gpu_smoke", SKIP, "GPU inside the container", f"{IMAGE} not built"))

    verdict = compute_verdict(p, checks)

    if args.json:
        json.dump(
            {"platform": p, "checks": [c.as_dict() for c in checks], "verdict": verdict},
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return verdict["code"]

    print("MeArmRL doctor -- can this machine run the lab?\n")
    for check in checks:
        print(f"  {_SYMBOL[check.status]} {check.title}")
        if check.detail:
            print(f"         {check.detail}")
        for line in check.fix:
            print(f"         -> {line}")
    print(f"\nVerdict: {verdict['headline']}")
    for line in verdict["next"]:
        print(f"  {line}")
    return verdict["code"]


if __name__ == "__main__":
    raise SystemExit(main())
