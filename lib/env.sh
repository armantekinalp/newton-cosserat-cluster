# Shared part of delftblue/env.sh and daic/env.sh; source those, not this file.
#
# The cluster file sets, before sourcing this one:
#   NC_CLUSTER       delftblue | daic
#   NC_BINDS         host paths mounted into the image besides the checkout (space-separated)
#   NC_CACHE_TAR     default for the cache tarball kept across jobs
#   NC_REBUILD_HINT  command that rebuilds the image, printed when it is stale
#
# User variables:
#   NC_SIF        path to the .sif image (required)
#   NC_REPO       newton-cosserat checkout mounted at /workspace/newton-cosserat
#                 (default: ../newton-cosserat next to this repository)
#   NC_CACHE      Warp/JAX/matplotlib caches on node-local disk
#                 (default: /tmp/$USER/newton-cosserat-cache; jobs use a per-job directory)
#   NC_CACHE_TAR  the same caches kept across jobs as ONE file. The caches are ~1200 small
#                 files; copying them one by one from BeeGFS took ~160 s, the tarball < 1 s.

if [[ -z "${NC_SIF:-}" || ! -f "${NC_SIF}" ]]; then
    echo "Set NC_SIF to the newton-cosserat .sif image (currently '${NC_SIF:-}')." >&2
    return 1
fi
NC_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NC_REPO=$(cd "${NC_REPO:-${NC_TOOLS}/../newton-cosserat}" 2>/dev/null && pwd) || NC_REPO=
if [[ ! -f "${NC_REPO}/pyproject.toml" ]]; then
    echo "Set NC_REPO to the newton-cosserat checkout (default: ${NC_TOOLS%/*}/newton-cosserat)." >&2
    return 1
fi
NC_CACHE=${NC_CACHE:-/tmp/${USER}/newton-cosserat-cache}
# Exported so `srun bash -c 'source .../env.sh'` in a job sees the same values.
export NC_CLUSTER NC_SIF NC_TOOLS NC_REPO NC_CACHE NC_CACHE_TAR
mkdir -p "${NC_CACHE}"/{warp,jax,matplotlib,pycache}

# Unpack the caches saved by an earlier job, if any.
nc_restore_caches() {
    if [[ -f "${NC_CACHE_TAR}" ]]; then
        tar -C "${NC_CACHE}" -xf "${NC_CACHE_TAR}" || echo "WARNING: could not unpack ${NC_CACHE_TAR}" >&2
    fi
}

# Save the caches for later jobs. Concurrent jobs: the last writer wins, which only
# costs recompilation of whatever the other job added.
nc_save_caches() {
    mkdir -p "$(dirname "${NC_CACHE_TAR}")"
    local partial=${NC_CACHE_TAR}.partial.$$
    if tar -C "${NC_CACHE}" -cf "${partial}" warp jax matplotlib; then
        mv "${partial}" "${NC_CACHE_TAR}"
    else
        rm -f "${partial}"
        echo "WARNING: could not save caches to ${NC_CACHE_TAR}" >&2
    fi
}

# Copy the image to node-local disk and point NC_SIF at the copy (one sequential read
# instead of random reads from the shared filesystem during start-up).
nc_stage_image() {
    # Per-job name: two jobs on one node must not delete each other's copy.
    local local_sif=/tmp/${USER}/${SLURM_JOB_ID:-interactive}-$(basename "${NC_SIF}")
    if [[ "${NC_SIF}" != "${local_sif}" ]]; then
        mkdir -p "/tmp/${USER}"
        cp "${NC_SIF}" "${local_sif}"
        export NC_SIF=${local_sif}
    fi
}

# Warn when the image was built from a different uv.lock than the checkout's.
nc_check_lock() {
    local image_sha repo_sha
    image_sha=$(apptainer exec "${NC_SIF}" cat /opt/uv.lock.sha256 2>/dev/null || true)
    repo_sha=$(sha256sum "${NC_REPO}/uv.lock" | cut -d' ' -f1)
    if [[ "${image_sha}" != "${repo_sha}" ]]; then
        echo "WARNING: ${NC_SIF} was built from a different uv.lock; rebuild with: ${NC_REBUILD_HINT}" >&2
    fi
}

# --cleanenv keeps host modules (LD_LIBRARY_PATH, PYTHONPATH, CUDA modules) out of the
# container; JAX and Warp use the CUDA libraries bundled in their wheels, the driver
# comes from the host through --nv. Do not `module load cuda`.
nc_exec() {
    local binds=() path
    for path in ${NC_BINDS}; do binds+=(--bind "${path}"); done
    apptainer exec --nv --cleanenv "${binds[@]}" \
        --bind "${NC_REPO}:/workspace/newton-cosserat" \
        --pwd /workspace/newton-cosserat \
        --env "WARP_CACHE_PATH=${NC_CACHE}/warp" \
        --env "JAX_COMPILATION_CACHE_DIR=${NC_CACHE}/jax" \
        --env "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0" \
        --env "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0" \
        --env "MPLCONFIGDIR=${NC_CACHE}/matplotlib" \
        --env "PYTHONPYCACHEPREFIX=${NC_CACHE}/pycache" \
        "${NC_SIF}" "$@"
}
