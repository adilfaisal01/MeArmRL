# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MeArmRL container image.
#
# Self-contained build: layers the MeArmRL extension and the RL stack on top of
# the NVIDIA Isaac Sim runtime, cloning Isaac Lab from GitHub (pinned tag). No
# local isaac-lab-base pre-build and no registry push is required -- students
# build this image themselves with a single command:
#
#   docker login nvcr.io   # one-time, free NGC account + API key
#   docker build -t mearmrl:local -f docker/Dockerfile .
#
# Container layout:
#   /isaac-sim    NVIDIA Isaac Sim runtime (from the base image)
#   /IsaacLab     Isaac Lab 2.3.2, cloned from GitHub and installed into the
#                 Isaac Sim Python
#   /MeArmRL      this repo (extension + scripts + entrypoint), editable-installed
#
# The image is built locally by whoever needs it. Students bind-mount their own
# checkout of `source/MeArmRL` over `/MeArmRL/source/MeArmRL` at runtime, so
# reward/env edits take effect without rebuilding the image.
#
# Build (anyone with Docker + an NVIDIA GPU):
#   docker build -t mearmrl:local -f docker/Dockerfile .

ARG ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim
ARG ISAACSIM_VERSION=5.1.0
FROM ${ISAACSIM_BASE_IMAGE}:${ISAACSIM_VERSION}

ENV DOCKER_USER_HOME=/root
ENV DEBIAN_FRONTEND=noninteractive

# The isaac-sim:5.x images default to a non-root user (isaac-sim, uid 1234).
# All build steps here need root (apt, git clone into /, pip into /isaac-sim).
USER root

WORKDIR /

# ---------------------------------------------------------------------------
# System utilities (cmake/build-essential are needed by `isaaclab.sh --install`)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        cmake \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Isaac Lab (deps layer, changes rarely)
# ---------------------------------------------------------------------------
RUN git clone --depth=1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git /IsaacLab

ENV ISAACSIM_PATH=/isaac-sim
ENV ISAACLAB_PATH=/IsaacLab
ENV PYTHONPATH=${ISAACLAB_PATH}/source:${PYTHONPATH}

WORKDIR /IsaacLab

# Link the Isaac Sim runtime so isaaclab.sh finds _isaac_sim/python.sh
RUN ln -s ${ISAACSIM_PATH} _isaac_sim

# Upgrade the Isaac Sim Python's pip (not the system Python's)
RUN /IsaacLab/_isaac_sim/kit/python/bin/python3 -m pip install --no-cache-dir \
        "pip<25.0" "packaging<24.0"

# Install Isaac Lab + its dependencies into the Isaac Sim Python
# (torch, isaacsim-rl, all Isaac Lab extensions)
RUN --mount=type=cache,target=/root/.cache/pip \
    TERM=xterm ./isaaclab.sh --install

# ---------------------------------------------------------------------------
# RL stack on top of the Isaac Sim Python
# ---------------------------------------------------------------------------
# `isaaclab.sh -p` resolves to the Isaac Sim Python. Always use it (never bare
# `pip`, which would target the apt Python).
RUN /IsaacLab/isaaclab.sh -p -m pip install --no-cache-dir \
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
COPY source/MeArmRL /MeArmRL/source/MeArmRL
COPY scripts /MeArmRL/scripts
COPY docker/entrypoint.sh /MeArmRL/docker/entrypoint.sh

WORKDIR /MeArmRL

RUN /IsaacLab/isaaclab.sh -p -m pip install --no-cache-dir -e /MeArmRL/source/MeArmRL

# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------
ENV OMNI_KIT_ALLOW_ROOT=1 \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    XDG_RUNTIME_DIR=/tmp/xdg \
    ISAACLAB_PATH=/IsaacLab \
    PYTHONPATH=/IsaacLab/source:/MeArmRL/source:${PYTHONPATH}

ENTRYPOINT ["/MeArmRL/docker/entrypoint.sh"]