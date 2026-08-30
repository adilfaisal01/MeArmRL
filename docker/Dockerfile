# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MeArmRL container image.
#
# Layers the MeArmRL extension on top of the `isaac-lab-base` image, which is
# built from the local Isaac Lab checkout (/mnt/E/IsaacLab, Isaac Lab 2.3.2)
# using its own docker tooling:
#
#   cd /mnt/E/IsaacLab/docker && python3 container.py build base
#
# That produces the `isaac-lab-base` image (Isaac Sim 5.1.0 + Isaac Lab 2.3.2).
# This Dockerfile then adds the MeArmRL extension and the RL stack.
#
# The image is a SHARED, FROZEN baseline. Students never build or push it.
# Students bind-mount their own checkout of `source/MeArmRL` over
# `/workspace/isaaclab/source/MeArmRL` at runtime, so their reward-function
# edits take effect without rebuilding the image.
#
# Build (maintainer only):
#   docker build -t ghcr.io/adilfaisal01/mearmrl:<tag> -f docker/Dockerfile .

ARG ISAAC_LAB_BASE_IMAGE=isaac-lab-base
FROM ${ISAAC_LAB_BASE_IMAGE}

# ---------------------------------------------------------------------------
# RL stack on top of the Isaac Sim Python
# ---------------------------------------------------------------------------
# The isaac-lab-base image installs Isaac Lab extensions into the Isaac Sim
# Python. We add skrl (the RL library used by scripts/skrl/train.py) plus a
# few helpers. `isaaclab.sh -p` resolves to the Isaac Sim python.
RUN /workspace/isaaclab/isaaclab.sh -p -m pip install --no-cache-dir \
        "skrl>=2.0.0" \
        "gymnasium" \
        "hydra-core" \
        "omegaconf" \
        "packaging" \
        "psutil" \
        "toml"

# ---------------------------------------------------------------------------
# MeArmRL extension (editable install so the bind-mounted source wins at runtime)
# ---------------------------------------------------------------------------
COPY --chown=root:root source/MeArmRL /workspace/isaaclab/source/MeArmRL
COPY --chown=root:root scripts /workspace/isaaclab/scripts
COPY --chown=root:root docker/entrypoint.sh /workspace/isaaclab/docker/entrypoint.sh

WORKDIR /workspace/isaaclab

RUN /workspace/isaaclab/isaaclab.sh -p -m pip install --no-cache-dir -e source/MeArmRL

# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------
ENV OMNI_KIT_ALLOW_ROOT=1 \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    XDG_RUNTIME_DIR=/tmp/xdg \
    ISAACLAB_PATH=/workspace/isaaclab \
    PYTHONPATH=/workspace/isaaclab/source:${PYTHONPATH}

ENTRYPOINT ["/workspace/isaaclab/docker/entrypoint.sh"]
