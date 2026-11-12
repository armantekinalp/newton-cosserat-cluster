#!/usr/bin/env bash
# Build the DelftBlue Apptainer image on a LOGIN node (needs internet and --fakeroot).
#
#   bash delftblue/build_sif.sh
#   OUTPUT=/scratch/$USER/containers/other.sif bash delftblue/build_sif.sh
# The DAIC image (CUDA 12 JAX; DAIC cannot build images itself) is built the same way:
#   DEF=daic/newton-cosserat.def OUTPUT=/scratch/$USER/containers/newton-cosserat-daic.sif \
#       bash delftblue/build_sif.sh
#
# NC_REPO is the newton-cosserat checkout whose pyproject.toml, uv.lock and src/ go into the
# image (default: ../newton-cosserat next to this repository). The build runs from there,
# because the %files paths in the .def are relative to it.
#
# The build and its temporary files stay on the login node's local /tmp; only the
# finished .sif (one large file, which BeeGFS handles well) is written to /scratch.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEF=$(realpath "${DEF:-${HERE}/newton-cosserat.def}")
NC_REPO=$(cd "${NC_REPO:-${HERE}/../../newton-cosserat}" && pwd)
if [[ ! -f "${NC_REPO}/pyproject.toml" || ! -f "${NC_REPO}/uv.lock" ]]; then
    echo "NC_REPO=${NC_REPO} is not a newton-cosserat checkout." >&2
    exit 1
fi
cd "${NC_REPO}"

OUTPUT=${OUTPUT:-/scratch/${USER}/containers/newton-cosserat.sif}
BUILD=/tmp/${USER}/newton-cosserat-sif-build
export APPTAINER_TMPDIR=${BUILD}/tmp APPTAINER_CACHEDIR=${BUILD}/cache
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}" "$(dirname "${OUTPUT}")"
trap 'rm -rf "${BUILD}"' EXIT

echo "Free space in /tmp: $(df -h /tmp | awk 'NR == 2 {print $4}') (need ~15 GB)"
SECONDS=0
echo "Building ${DEF} from ${NC_REPO} ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git')) -> ${OUTPUT}"
apptainer build --fakeroot "${BUILD}/newton-cosserat.sif" "${DEF}"
echo "[${SECONDS}s] image built: $(du -h "${BUILD}/newton-cosserat.sif" | cut -f1)"

# Write then rename so a running job never sees a half-written image.
cp "${BUILD}/newton-cosserat.sif" "${OUTPUT}.partial"
mv "${OUTPUT}.partial" "${OUTPUT}"
echo "[${SECONDS}s] written to ${OUTPUT}"
