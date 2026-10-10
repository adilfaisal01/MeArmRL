#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Pull training artifacts (checkpoints, params, TensorBoard events, videos) from
# the university cluster to this machine.
#
#   MEARMRL_REMOTE=user@hpc-login.university.edu bash scripts/pull_results.sh
#   MEARMRL_REMOTE=... bash scripts/pull_results.sh 12345678   # a specific job
#   MEARMRL_REMOTE=... bash scripts/pull_results.sh --all      # every job
#   MEARMRL_REMOTE=... bash scripts/pull_results.sh --list     # list remote jobs
#
# Artifacts land under logs/remote/<JOB_ID>/, which the compose container sees as
# /MeArmRL/logs/remote/<JOB_ID>. Set MEARMRL_REMOTE once (or export it in your
# shell profile). On Windows, run this from Git-Bash or WSL (needs rsync + ssh).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REMOTE="${MEARMRL_REMOTE:-}"
MODE="latest"
JOB_ID=""
WEIGHTS_ONLY=0
DRY_RUN=0

usage() {
    cat >&2 <<'EOF'
Pull MeArmRL training artifacts from the cluster to this machine.

Usage:
  MEARMRL_REMOTE=user@host bash scripts/pull_results.sh [JOB_ID] [options]

Options:
  --all        pull every job
  --weights    pull only checkpoints/ and params/ (skip videos, TensorBoard)
  --list       list remote jobs (newest first) and exit
  --dry-run    print the rsync commands without running them
  -h, --help   show this help

Environment:
  MEARMRL_REMOTE       ssh target, e.g. user@hpc-login.university.edu  (required)
  MEARMRL_REMOTE_LOGS  remote logs root (default /scratch/<remote-user>/mearmrl-logs)
  MEARMRL_DEST         local destination (default <repo>/logs/remote)
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --all) MODE="all" ;;
        --list) MODE="list" ;;
        --weights) WEIGHTS_ONLY=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h | --help) usage; exit 0 ;;
        -*) echo "pull_results: unknown option '$1'" >&2; usage; exit 2 ;;
        *) JOB_ID="$1"; MODE="job" ;;
    esac
    shift
done

if [ -z "${REMOTE}" ]; then
    echo "pull_results: MEARMRL_REMOTE is not set." >&2
    echo "  MEARMRL_REMOTE=user@hpc-login.university.edu bash scripts/pull_results.sh" >&2
    exit 2
fi

for tool in rsync ssh; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "pull_results: '${tool}' not found on PATH (Windows: use Git-Bash or WSL)." >&2
        exit 2
    fi
done

# The remote user is the part before '@'; fall back to the local user.
if [[ "${REMOTE}" == *"@"* ]]; then
    REMOTE_USER="${REMOTE%%@*}"
else
    REMOTE_USER="${USER:-$(id -un)}"
fi
REMOTE_LOGS="${MEARMRL_REMOTE_LOGS:-/scratch/${REMOTE_USER}/mearmrl-logs}"
REMOTE_JOBS="${REMOTE_LOGS}/jobs"
DEST="${MEARMRL_DEST:-${REPO_ROOT}/logs/remote}"

SSH_OPTS=(-o ConnectTimeout=10)
RSYNC_SSH="ssh -o ConnectTimeout=10"

run() {
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "+ $*"
    else
        "$@"
    fi
}

remote_jobs() {
    # The remote path is interpolated locally on purpose -- it depends on the
    # remote user parsed from MEARMRL_REMOTE.
    # shellcheck disable=SC2029
    ssh "${SSH_OPTS[@]}" "${REMOTE}" "ls -1t ${REMOTE_JOBS} 2>/dev/null || true"
}

case "${MODE}" in
    list)
        echo "Jobs in ${REMOTE}:${REMOTE_JOBS} (newest first):"
        jobs="$(remote_jobs)"
        if [ -z "${jobs}" ]; then echo "  (none)"; else echo "${jobs}" | sed 's/^/  /'; fi
        exit 0
        ;;
    latest)
        JOB_ID="$(remote_jobs | head -n1)"
        if [ -z "${JOB_ID}" ]; then
            echo "pull_results: no jobs found under ${REMOTE}:${REMOTE_JOBS}" >&2
            exit 1
        fi
        echo "Latest job: ${JOB_ID}"
        ;;
esac

if [ "${MODE}" = "all" ]; then
    echo "Pulling all jobs -> ${DEST}/"
    run mkdir -p "${DEST}"
    run rsync -avP -e "${RSYNC_SSH}" "${REMOTE}:${REMOTE_JOBS}/" "${DEST}/"
    echo "Done."
    exit 0
fi

SRC="${REMOTE}:${REMOTE_JOBS}/${JOB_ID}/"
OUT="${DEST}/${JOB_ID}/"
echo "Pulling job ${JOB_ID} -> ${OUT}"
run mkdir -p "${DEST}"
if [ "${WEIGHTS_ONLY}" -eq 1 ]; then
    run rsync -avP -e "${RSYNC_SSH}" \
        --include='*/' --include='checkpoints/**' --include='params/**' \
        --exclude='*' "${SRC}" "${OUT}"
else
    run rsync -avP -e "${RSYNC_SSH}" "${SRC}" "${OUT}"
fi

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "(dry run: nothing was copied)"
    exit 0
fi

# --- Summarise what was pulled + how to play it ------------------------------
echo
checkpoints="$(find "${OUT}" -name '*.pt' 2>/dev/null | sort || true)"
videos="$(find "${OUT}" -name '*.mp4' 2>/dev/null | sort || true)"

if [ -n "${videos}" ]; then
    echo "Videos:"
    echo "${videos}" | sed "s|^${REPO_ROOT}/||" | sed 's/^/  /'
fi

if [ -z "${checkpoints}" ]; then
    echo "No .pt checkpoints found under ${OUT}."
    echo "If the job ran before the /MeArmRL/logs mount fix, its results were not persisted."
    exit 0
fi

echo "Checkpoints:"
echo "${checkpoints}" | sed "s|^${REPO_ROOT}/||" | sed 's/^/  /'

best="$(echo "${checkpoints}" | grep -m1 'best_agent\.pt' || true)"
if [ -z "${best}" ]; then best="$(echo "${checkpoints}" | tail -n1)"; fi
relative="${best#"${DEST}/${JOB_ID}/"}"
echo
echo "Play it (inside the container; ./logs is mounted at /MeArmRL/logs):"
echo "  /IsaacLab/isaaclab.sh -p scripts/skrl/play.py \\"
echo "      --task=Mearmrl-Reach-Student-v0 --num_envs=8 \\"
echo "      --checkpoint=/MeArmRL/logs/remote/${JOB_ID}/${relative} --headless"
