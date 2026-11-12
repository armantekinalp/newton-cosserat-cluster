#!/bin/bash
# Build a benchmark venv tarball on a DelftBlue LOGIN node without touching the slow /scratch venv.
#
# Everything (uv cache, venv, tarball) is built on the login node's local /tmp, then written to
# /scratch as ONE file, the access pattern BeeGFS handles well. Jobs copy that file to the
# compute node's /tmp and unpack it there (see storage_probe.sbatch).
#
# The venv omits torch and the packages only torch needs (triton, nvidia-cusparselt): the
# benchmark never imports them (torch is only used by the Isaac Lab adapter). newton-cosserat
# itself stays an editable install pointing at this checkout's src/, so source edits on /scratch
# take effect without rebuilding; rebuild only after uv.lock changes.
#
# Run on a login node (needs internet and ~12 GB free in /tmp); NC_REPO is the checkout
# (default: ../newton-cosserat next to this repository):
#   bash delftblue/diagnostics/build_venv.sh
# Override the output with OUTPUT=/scratch/$USER/venv.tar.zst (default).

set -euo pipefail

NC_REPO=$(cd "${NC_REPO:-$(dirname "${BASH_SOURCE[0]}")/../../../newton-cosserat}" && pwd)
cd "${NC_REPO}"
if [[ ! -f pyproject.toml || ! -f uv.lock ]]; then
    echo "NC_REPO=${NC_REPO} is not a newton-cosserat checkout." >&2
    exit 1
fi
export PATH="$HOME/.local/bin:$PATH"

BUILD=/tmp/${USER}/newton-cosserat-venv-build
OUTPUT=${OUTPUT:-/scratch/$USER/venv.tar.zst}
mkdir -p "${BUILD}"
trap 'rm -rf "${BUILD}"' EXIT

echo "Free space in /tmp: $(df -h /tmp | awk 'NR == 2 {print $4}') (need ~12 GB)"
SECONDS=0
stamp() { printf '[%4ds] %s\n' "${SECONDS}" "$1"; }

# Local cache and venv; copy instead of hard-linking so the venv is self-contained.
UV_CACHE_DIR=${BUILD}/uv-cache UV_PROJECT_ENVIRONMENT=${BUILD}/.venv UV_LINK_MODE=copy \
    uv sync --locked --extra cuda13 --extra viz --compile-bytecode \
    --no-install-package torch --no-install-package triton --no-install-package nvidia-cusparselt-cu13
stamp "venv built: $(du -sh "${BUILD}/.venv" | cut -f1)"

# Fail now rather than in the job if the venv is incomplete.
"${BUILD}/.venv/bin/python" -c "import jax, warp, newton, soromox, newton_cosserat.solvers, matplotlib"
stamp "import check passed"

# Record which lock file the tarball matches; jobs warn when uv.lock is newer.
sha256sum uv.lock > "${BUILD}/.venv/uv.lock.sha256"
if command -v zstd >/dev/null; then
    tar -C "${BUILD}" -I 'zstd -T4 -3' -cf "${BUILD}/venv.tar" .venv
else
    [[ "${OUTPUT}" == *.zst ]] && OUTPUT=${OUTPUT%.zst}
    tar -C "${BUILD}" -cf "${BUILD}/venv.tar" .venv
fi
stamp "tarball on local disk: $(du -h "${BUILD}/venv.tar" | cut -f1)"

# One sequential write to /scratch; write then rename so a job never sees a half-written file.
cp "${BUILD}/venv.tar" "${OUTPUT}.partial"
mv "${OUTPUT}.partial" "${OUTPUT}"
stamp "written to ${OUTPUT}"
if [[ "${OUTPUT}" == *.zst ]]; then
    other=${OUTPUT%.zst}
else
    other=${OUTPUT}.zst
fi
if [[ -f "${other}" ]]; then
    echo "Note: ${other} also exists; the probe prefers venv.tar.zst. Delete whichever is stale."
fi
