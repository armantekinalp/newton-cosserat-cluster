#!/bin/bash
# Test running the contact-scaling benchmark from node-local /tmp instead of /scratch.
#
# The startup probe showed ~440 s per Python process just loading jax/warp/newton from the
# /scratch venv (BeeGFS, spinning disks), equally slow on a second run, and a 6.6 GB
# `cp -a .venv /tmp` that did not finish in 15 min. Following the DelftBlue I/O guide
# (https://doc.dhpc.tudelft.nl/delftblue/Optimise-Your-IO/), this job stages the venv as a
# single tarball onto the node-local SSD (/tmp), runs from there, and measures each step.
# /home lives on the same BeeGFS as /scratch, so it is not tested separately.
#
# One-time preparation on a LOGIN node (and again after uv.lock changes): build a fresh, slimmer
# venv on the login node's local disk and write it to /scratch as one tarball. Tarring the
# existing /scratch venv is as slow as importing from it.
#   bash delftblue/diagnostics/build_venv.sh
#
# Submit from the newton-cosserat-cluster root (NC_REPO defaults to ../newton-cosserat):
#   sbatch delftblue/diagnostics/storage_probe.sbatch
# Overrides (--export=ALL,NAME=value): VENV_TAR, CASE_MODES ("point hydroelastic"),
#   DD_MB (2048), STEP_TIMEOUT (900 s per step).
# Output: slurm-storage-probe-<jobid>.out
#
#SBATCH --job-name=storage-probe
#SBATCH --account=innovation
#SBATCH --partition=gpu-a100
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out

# No -e: every measurement should run even if an earlier one fails or times out.
set -uo pipefail

cd "${SLURM_SUBMIT_DIR}"
if [[ ! -d delftblue/diagnostics ]]; then
    echo "Submit this script from the newton-cosserat-cluster repository root." >&2
    exit 1
fi
NC_TOOLS=${PWD}
NC_REPO=$(cd "${NC_REPO:-${NC_TOOLS}/../newton-cosserat}" && pwd)
cd "${NC_REPO}"
REPO=${PWD}

export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/$USER/.cache/uv}
export PATH="$HOME/.local/bin:$PATH"
export UV_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
export MPLCONFIGDIR=${MPLCONFIGDIR:-/scratch/$USER/.cache/matplotlib}

STEP_TIMEOUT=${STEP_TIMEOUT:-900}       # seconds per step; a timed-out step is reported, not fatal
DD_MB=${DD_MB:-2048}                    # dd test-file size
CASE_MODES=${CASE_MODES:-"point hydroelastic"}
VENV_TAR=${VENV_TAR:-}
if [[ -z "${VENV_TAR}" ]]; then
    for candidate in /scratch/$USER/venv.tar.zst /scratch/$USER/venv.tar; do
        [[ -f "${candidate}" ]] && VENV_TAR=${candidate} && break
    done
fi
SCRATCH_JAX_CACHE=${JAX_COMPILATION_CACHE_DIR:-/scratch/$USER/.cache/jax}
SCRATCH_WARP_CACHE=${WARP_CACHE_PATH:-/scratch/$USER/.cache/warp}
SCRATCH_VENV=${REPO}/.venv

# Per-job node-local directory, removed on exit, timeout or cancel. /tmp is NOT cleaned
# automatically on DelftBlue and is shared with other jobs on the node (FAQ).
LOCAL=/tmp/${USER}/${SLURM_JOB_ID}
SCRATCH_TEST=/scratch/${USER}/storage-probe-${SLURM_JOB_ID}
mkdir -p "${LOCAL}" "${SCRATCH_TEST}"
clean_up() { rm -rf "${LOCAL}" "${SCRATCH_TEST}" && echo "Cleaned up ${LOCAL} and ${SCRATCH_TEST}"; }
trap clean_up EXIT
trap 'exit 143' TERM INT  # turn scancel / time-limit signals into a normal exit so EXIT runs

section() { printf '\n===== %s  [%s]\n' "$1" "$(date '+%F %T')"; }
now() { date +%s.%N; }
elapsed() { awk "BEGIN{printf \"%.1f\", $2 - $1}"; }
# timed LABEL CMD...: run under a timeout, report wall time and exit code.
timed() {
    local label=$1; shift
    local t0 t1 rc
    t0=$(now)
    timeout "${STEP_TIMEOUT}" "$@"
    rc=$?
    t1=$(now)
    printf '>>> %-46s %8s s  (exit %d%s)\n' "${label}" "$(elapsed "${t0}" "${t1}")" "${rc}" \
        "$( ((rc == 124)) && echo ', TIMED OUT')"
    return "${rc}"
}
# dd_line LABEL dd-args...: print dd's throughput summary line.
dd_line() {
    local label=$1; shift
    local out rc
    out=$(timeout "${STEP_TIMEOUT}" dd "$@" 2>&1)
    rc=$?
    printf '  %-40s %s%s\n' "${label}" "$(echo "${out}" | tail -1)" "$( ((rc == 124)) && echo '  TIMED OUT')"
    return "${rc}"
}
# Write a test file, then read it back bypassing the cache (cold) and normally twice (cached).
throughput() {
    local name=$1 dir=$2 mb=$3
    local f=${dir}/ddtest.bin
    echo "${name}: ${dir} (${mb} MiB)"
    dd_line "write (fdatasync)" if=/dev/zero of="${f}" bs=16M count=$((mb / 16)) conv=fdatasync
    dd_line "read, O_DIRECT (uncached)" if="${f}" of=/dev/null bs=16M iflag=direct \
        || dd_line "read (O_DIRECT unsupported)" if="${f}" of=/dev/null bs=16M
    dd_line "read, buffered, pass 1" if="${f}" of=/dev/null bs=16M
    dd_line "read, buffered, pass 2" if="${f}" of=/dev/null bs=16M
    rm -f "${f}"
}

PROBE=$(cat <<'EOF'
import sys, time
t0 = t = time.time()
def step(name):
    global t
    now = time.time(); print(f"  {name:<16}{now - t:8.1f} s", flush=True); t = now
import numpy;           step("import numpy")
import jax;             step("import jax")
import warp as wp;      wp.config.log_level = wp.LOG_WARNING; step("import warp")
wp.init();              step("wp.init")
jax.devices();          step("jax.devices")
import newton, soromox; step("newton+soromox")
import newton_cosserat.solvers; step("newton_cosserat")
print(f"  {'total':<16}{time.time() - t0:8.1f} s")
print(f"  (prefix {sys.prefix}; interpreter/stdlib from {sys.base_prefix})")
EOF
)

# ---------------------------------------------------------------------------------------------
section "0. node, disks, mounts"
echo "job ${SLURM_JOB_ID} on $(hostname); local dir ${LOCAL}"
lsblk -d -o NAME,ROTA,SIZE,MODEL,TRAN 2>/dev/null
findmnt -no TARGET,SOURCE,FSTYPE,OPTIONS /tmp
findmnt -no TARGET,SOURCE,FSTYPE,OPTIONS /scratch
echo "BeeGFS client tuning:"
grep -hiE 'tuneFileCacheType|tuneFileCacheBufSize' /proc/fs/beegfs/*/config 2>/dev/null | sort -u \
    || echo "  (not readable)"
free -g | head -2
df -h /tmp /scratch 2>/dev/null
echo "Other users' data in /tmp: $(du -sh /tmp 2>/dev/null | cut -f1) used"

# ---------------------------------------------------------------------------------------------
section "1. stage the venv tarball onto /tmp and start Python from it  (the key test)"
LOCAL_PY=""
if [[ -n "${VENV_TAR}" && -f "${VENV_TAR}" ]]; then
    TAR_MB=$(( $(stat -c %s "${VENV_TAR}") / 1048576 ))
    echo "tarball: ${VENV_TAR} (${TAR_MB} MiB, modified $(date -r "${VENV_TAR}" '+%F %T'))"
    if [[ "${VENV_TAR}" == *.zst ]] && ! command -v zstd >/dev/null; then
        echo "ERROR: zstd is not available on $(hostname); rebuild an uncompressed venv.tar." >&2
    fi
    t0=$(now)
    timed "copy tarball /scratch -> /tmp" cp "${VENV_TAR}" "${LOCAL}/"
    t1=$(now)
    awk "BEGIN{printf \"  => %.0f MB/s sequential read from /scratch\n\", ${TAR_MB} * 1.048576 / ($t1 - $t0)}"
    LOCAL_TAR=${LOCAL}/$(basename "${VENV_TAR}")
    case "${VENV_TAR}" in
        *.zst) timed "extract on /tmp (zstd)" tar -I zstd -C "${LOCAL}" -xf "${LOCAL_TAR}" ;;
        *) timed "extract on /tmp" tar -C "${LOCAL}" -xf "${LOCAL_TAR}" ;;
    esac
    rm -f "${LOCAL_TAR}"
    if [[ -x "${LOCAL}/.venv/bin/python" ]]; then
        LOCAL_PY=${LOCAL}/.venv/bin/python
        if [[ -f "${LOCAL}/.venv/uv.lock.sha256" ]]; then
            (cd "${REPO}" && sha256sum --quiet -c "${LOCAL}/.venv/uv.lock.sha256") \
                && echo "venv matches the current uv.lock" \
                || echo "WARNING: uv.lock changed since the tarball was built; rebuild it."
        fi
        echo "Run 1 (cold):"
        timed "startup from /tmp venv, run 1" "${LOCAL_PY}" -c "${PROBE}"
        echo "Run 2 (page cache):"
        timed "startup from /tmp venv, run 2" "${LOCAL_PY}" -c "${PROBE}"
    else
        echo "Extraction did not produce ${LOCAL}/.venv/bin/python; skipping the tests that need it."
    fi
else
    echo "No tarball found (looked for /scratch/$USER/venv.tar.zst and venv.tar)."
    echo "Create one on the login node (see the header of this script) and resubmit."
fi

# ---------------------------------------------------------------------------------------------
section "2. stage JAX / Warp kernel caches onto /tmp, and write them back as one tarball"
mkdir -p "${LOCAL}/cache"
for cache in "${SCRATCH_JAX_CACHE}" "${SCRATCH_WARP_CACHE}"; do
    name=$(basename "${cache}")
    if [[ -d "${cache}" ]]; then
        echo "${cache}: $(du -sh "${cache}" | cut -f1), $(find "${cache}" -type f | wc -l) files"
        timed "copy ${name} cache (file by file) -> /tmp" cp -a "${cache}" "${LOCAL}/cache/${name}"
    else
        echo "${cache}: not present; starting empty"
        mkdir -p "${LOCAL}/cache/${name}"
    fi
done
# What the benchmark job would do at the end: one sequential write instead of many small files.
timed "tar caches /tmp -> /scratch (single file)" tar -C "${LOCAL}" -cf "${SCRATCH_TEST}/caches.tar" cache
ls -lh "${SCRATCH_TEST}/caches.tar" 2>/dev/null | awk '{print "  caches.tar: " $5}'
timed "read caches.tar back /scratch -> /tmp" tar -C "${LOCAL}" -xf "${SCRATCH_TEST}/caches.tar" --one-top-level=cache-roundtrip

# ---------------------------------------------------------------------------------------------
section "3. real benchmark cases: venv and caches on /tmp, source and results on /scratch"
# Same settings as contact_scaling_venv.sbatch defaults, one CUDA world per mode. Point mode crashed in
# job 1279398 with EBUSY on the XLA autotune cache on /scratch; here that cache is on /tmp.
if [[ -n "${LOCAL_PY}" ]]; then
    for mode in ${CASE_MODES}; do
        out=${LOCAL}/results/${mode}/results.json
        timed "benchmark ${mode}, 1 CUDA world (parent+worker)" env \
            JAX_COMPILATION_CACHE_DIR="${LOCAL}/cache/$(basename "${SCRATCH_JAX_CACHE}")" \
            JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0 \
            WARP_CACHE_PATH="${LOCAL}/cache/$(basename "${SCRATCH_WARP_CACHE}")" \
            "${LOCAL_PY}" -m tools.benchmarks.contact_scaling --contact-mode "${mode}" --devices cuda:0 \
            --cuda-worlds 1 --output "${out}" --hydroelastic-sectors 16 --hydroelastic-rings 2 \
            --hydroelastic-resolution 0.01 --environment-resolution 0.005 --steps 100 --repeats 2
        # Worker logs are on /tmp and would be deleted: show the tail of each one.
        for log in "${LOCAL}/results/${mode}"/results-cases/*.log; do
            [[ -f "${log}" ]] && { echo "  --- tail of $(basename "${log}")"; tail -8 "${log}" | sed 's/^/  /'; }
        done
    done
    mkdir -p "${REPO}/artifacts/storage-probe/${SLURM_JOB_ID}"
    cp -r "${LOCAL}/results" "${REPO}/artifacts/storage-probe/${SLURM_JOB_ID}/" 2>/dev/null \
        && echo "Results copied to artifacts/storage-probe/${SLURM_JOB_ID}/results"
else
    echo "No /tmp venv (section 1 failed or was skipped); skipped."
fi

# ---------------------------------------------------------------------------------------------
section "4. diagnostics: raw throughput and the /scratch venv (slow if /scratch is slow)"
echo "Largest packages in the /scratch venv:"
du -sm "${SCRATCH_VENV}"/lib/python3*/site-packages/* 2>/dev/null | sort -n | tail -8 \
    | awk '{printf "  %6d MiB  %s\n", $1, $2}'
throughput "/tmp (node-local SSD)" "${LOCAL}" "${DD_MB}"
throughput "/scratch (BeeGFS)" "${SCRATCH_TEST}" "${DD_MB}"
if [[ -x /usr/bin/time ]]; then
    echo "import from the /scratch venv, CPU vs wall time (wall >> user+sys with many major faults = I/O wait):"
    timeout "${STEP_TIMEOUT}" /usr/bin/time -v "${SCRATCH_VENV}/bin/python" -c \
        "import jax, warp as wp; wp.config.log_level = wp.LOG_WARNING; wp.init(); jax.devices()" \
        2> "${LOCAL}/time.txt"
    grep -E 'Elapsed|User time|System time|Percent of CPU|Major|File system inputs' "${LOCAL}/time.txt" \
        | sed 's/^\s*/  /'
fi

section "done"
