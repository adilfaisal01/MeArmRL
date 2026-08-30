# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MeArmRL container image.
#
# Builds on the official NVIDIA Isaac Sim 6.0.0 container and layers in:
#   - Isaac Lab (pinned commit, cloned at build time)
#   - RL stack (skrl, gymnasium, hydra, ...)
#   - the MeArmRL extension (editable install)
#
# The image is a SHARED, FROZEN baseline. Students never build or push it.
# Students bind-mount their own checkout of `source/MeArmRL` over
# `/opt/MeArmRL/source/MeArmRL` at runtime, so their reward-function edits
# take effect without rebuilding the image.
#
# Build (maintainer only):
#   docker build -t ghcr.io/<org>/mearmrl:<tag> \
#       --build-arg ISAAC_LAB_REF=<pinned-isaaclab-tag-or-commit> \
#       -f docker/Dockerfile .
#
# The base image runs as a non-root user with UID/GID 1234:1234. We keep that
# user so NVIDIA's tested cache-volume layout and ownership work unchanged.

ARG ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.0
FROM ${ISAAC_SIM_IMAGE}

# ---------------------------------------------------------------------------
# System packages (as root, then drop back to the image's default user 1234)
# ---------------------------------------------------------------------------
USER root

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        htop \
        libegl1 \
        libglvnd0 \
        libgomp1 \
        openssh-client \
        rdma-core \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Isaac Lab (pinned, cloned at build time)
# ---------------------------------------------------------------------------
ARG ISAAC_LAB_REF=main
ARG ISAAC_LAB_REPO=https://github.com/isaac-sim/IsaacLab.git

# Works for tags, branches, and commit SHAs.
RUN git clone --depth 1 ${ISAAC_LAB_REPO} /opt/IsaacLab \
    && cd /opt/IsaacLab \
    && git fetch --depth 1 origin ${ISAAC_LAB_REF} \
    && git checkout FETCH_HEAD

RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
        -e /opt/IsaacLab/source/isaaclab \
        -e /opt/IsaacLab/source/isaaclab_assets \
        -e /opt/IsaacLab/source/isaaclab_tasks \
        -e /opt/IsaacLab/source/isaaclab_rl

# ---------------------------------------------------------------------------
# RL stack on top of the Isaac Sim Python
# ---------------------------------------------------------------------------
RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
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
COPY --chown=1234:1234 source/MeArmRL /opt/MeArmRL/source/MeArmRL
COPY --chown=1234:1234 scripts /opt/MeArmRL/scripts
COPY --chown=1234:1234 docker/entrypoint.sh /opt/MeArmRL/docker/entrypoint.sh

WORKDIR /opt/MeArmRL

RUN /isaac-sim/python.sh -m pip install --no-cache-dir -e source/MeArmRL

# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------
ENV OMNI_KIT_ALLOW_ROOT=1 \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    XDG_RUNTIME_DIR=/tmp/xdg \
    PYTHONPATH=/opt/IsaacLab/source/isaaclab:/opt/MeArmRL/source:${PYTHONPATH}

# Restore the image's default non-root user (1234:1234)
USER 1234

ENTRYPOINT ["/opt/MeArmRL/docker/entrypoint.sh"]
