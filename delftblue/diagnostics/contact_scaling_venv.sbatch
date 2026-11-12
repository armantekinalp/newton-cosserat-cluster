#!/bin/bash
# Contact-scaling benchmark on a DelftBlue GPU node.
#
# One-time setup on a LOGIN node (compute nodes may not have internet access):
#   export UV_CACHE_DIR=/scratch/$USER/.cache/uv      # home quota is small
#   curl -LsSf https://astral.sh/uv/install.sh | sh  # if uv is not installed yet
#   cd /scratch/$USER/newton-cosserat   # the NC_REPO checkout
#   uv sync --locked --extra cuda13 --extra viz --compile-bytecode
#   # Precompiled .pyc files and a warm font cache avoid minutes of first-import work
#   # on the parallel filesystem inside every job.
#   uv run --no-sync python -c "import jax, warp, matplotlib.pyplot"
#
# Submit from the newton-cosserat-cluster root (NC_REPO defaults to ../newton-cosserat):
#   sbatch --account=research-<faculty>-<department> --mail-user=<you>@tudelft.nl \
#       delftblue/diagnostics/contact_scaling_venv.sbatch
# Optional overrides (exported into the job):
#   see the "Benchmark parameters" block below for every variable and its default, e.g.
#   sbatch --export=ALL,MODES=hydroelastic,DEVICES=cuda:0,CUDA_WORLDS="1 64 1024" ... contact_scaling_venv.sbatch
#
#SBATCH --job-name=contact-scaling
#SBATCH --account=innovation
#SBATCH --partition=gpu-a100
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out
# Email on start, end and failure; the end/fail mail includes the job's exit code.
# Pass the address when submitting: sbatch --mail-user=<you>@tudelft.nl ...
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ---- Benchmark parameters (each can be overridden with sbatch --export=ALL,NAME=value) ----
MODES=${MODES:-"point hydroelastic"}          # contact modes, run one after another
DEVICES=${DEVICES:-"cuda:0"}                  # add "cpu" for CPU numbers (single-threaded, slow to compile)
CPU_WORLDS=${CPU_WORLDS:-"1"}                 # environment counts swept on the CPU
CUDA_WORLDS=${CUDA_WORLDS:-"1 2"}  # ... and on the GPU
# Hydroelastic discretization (ignored in point mode).
HYDROELASTIC_SECTORS=${HYDROELASTIC_SECTORS:-16}       # rod perimeter subdivisions, >= 4
HYDROELASTIC_RINGS=${HYDROELASTIC_RINGS:-2}            # rod radial subdivisions
HYDROELASTIC_RESOLUTION=${HYDROELASTIC_RESOLUTION:-0.01}  # rod axial spacing [m]
ENVIRONMENT_RESOLUTION=${ENVIRONMENT_RESOLUTION:-0.005}   # curved environment shapes [m]
# Any other Config field, e.g. "--steps 500 --repeats 3 --stiffness 2e5".
# No colon: an explicitly empty EXTRA_ARGS="" runs the benchmark defaults.
EXTRA_ARGS=${EXTRA_ARGS-"--steps 100 --repeats 2"}
# --------------------------------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR}"
if [[ ! -d delftblue/diagnostics ]]; then
    echo "Submit this script from the newton-cosserat-cluster repository root." >&2
    exit 1
fi
NC_TOOLS=${PWD}
NC_REPO=$(cd "${NC_REPO:-${NC_TOOLS}/../newton-cosserat}" && pwd)
cd "${NC_REPO}"

export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/$USER/.cache/uv}
export PATH="$HOME/.local/bin:$PATH"
# The environment was synced on the login node; never resolve or download here.
export UV_OFFLINE=1
# Keep JAX/Warp kernel caches off the small home quota.
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-/scratch/$USER/.cache/warp}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Every sweep case runs in a fresh worker; persist JAX compilations across cases and jobs.
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/scratch/$USER/.cache/jax}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export MPLCONFIGDIR=${MPLCONFIGDIR:-/scratch/$USER/.cache/matplotlib}
# JAX/Warp use the CUDA libraries bundled with the pip wheels; do not `module load cuda`,
# which can shadow them through LD_LIBRARY_PATH.

RUN_DIR="artifacts/contact-scaling/delftblue-${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

JOB_T0=$(date +%s)
LAST_T=${JOB_T0}
# Print wall-clock time, time since the previous stamp and since the job started;
# also appended to ${RUN_DIR}/timings.txt.
stamp() {
    local now
    now=$(date +%s)
    printf '[%s] %-45s +%6ds  (total %ds)\n' "$(date '+%F %T')" "$1" \
        $((now - LAST_T)) $((now - JOB_T0)) | tee -a "${RUN_DIR}/timings.txt"
    LAST_T=${now}
}
stamp "setup"

{
    echo "job ${SLURM_JOB_ID} on $(hostname) at $(date -Is)"
    echo "newton-cosserat $(git rev-parse HEAD 2>/dev/null || echo unknown) $(git status --porcelain 2>/dev/null | wc -l) dirty files"
    echo "newton-cosserat-cluster $(git -C "${NC_TOOLS}" rev-parse HEAD 2>/dev/null || echo unknown)"
    lscpu | grep -E "Model name|^CPU\(s\)|MHz" || true
    nvidia-smi
} | tee "${RUN_DIR}/environment.txt"

uv run --no-sync python -c "import jax, warp as wp; wp.init(); print('JAX devices:', jax.devices())" \
    | tee -a "${RUN_DIR}/environment.txt"
stamp "environment probe (JAX/Warp init)"

# Sample GPU utilisation in the background for the whole job.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 5 \
    > "${RUN_DIR}/gpu-usage.csv" &
SMI_PID=$!
trap 'kill ${SMI_PID} 2>/dev/null || true' EXIT

status=0
outputs=()
for mode in ${MODES}; do
    output="${RUN_DIR}/${mode}/results.json"
    args=(
        --contact-mode "${mode}" --devices ${DEVICES} --output "${output}"
        --cpu-worlds ${CPU_WORLDS} --cuda-worlds ${CUDA_WORLDS}
        --hydroelastic-sectors "${HYDROELASTIC_SECTORS}" --hydroelastic-rings "${HYDROELASTIC_RINGS}"
        --hydroelastic-resolution "${HYDROELASTIC_RESOLUTION}" --environment-resolution "${ENVIRONMENT_RESOLUTION}"
    )
    echo "=== ${mode}: ${args[*]} ${EXTRA_ARGS}"
    # A failed case should not stop the remaining modes; the sweep still writes results.
    mode_status=0
    srun --ntasks=1 uv run --no-sync python -m tools.benchmarks.contact_scaling "${args[@]}" ${EXTRA_ARGS} \
        || mode_status=$?
    echo "=== ${mode} finished with exit code ${mode_status}"
    stamp "benchmark ${mode} (exit ${mode_status})"
    ((mode_status)) && status=${mode_status}
    [[ -f "${output}" ]] && outputs+=("${output}")
done

if ((${#outputs[@]})); then
    uv run --no-sync python -m tools.benchmarks.contact_scaling.plot "${outputs[@]}" \
        --output "${RUN_DIR}/contact-scaling.png" || echo "Plotting failed (needs the viz extra)." >&2
    stamp "plotting"
fi

stamp "job finished"
echo "Results in ${NC_REPO}/${RUN_DIR}"
exit "${status}"
