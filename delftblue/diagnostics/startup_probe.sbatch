#!/bin/bash
# Diagnose the ~7 min Python/JAX/Warp start-up seen in contact-scaling jobs on DelftBlue.
#
# Submit from the newton-cosserat-cluster root (same environment as contact_scaling_venv.sbatch):
#   sbatch delftblue/diagnostics/startup_probe.sbatch
# Output: slurm-startup-probe-<jobid>.out plus $NC_REPO/artifacts/startup-probe/<jobid>/.
# Sections 7-10 compare the CUDA 13 JAX stack against CPU-only JAX and, if
# /scratch/$USER/jax-cu12 exists (see section 10), a CUDA 12 JAX build.
#
#SBATCH --job-name=startup-probe
#SBATCH --account=innovation
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out

# No -e: every probe should run even if an earlier one fails or times out.
set -uo pipefail

cd "${SLURM_SUBMIT_DIR}"
if [[ ! -d delftblue/diagnostics ]]; then
    echo "Submit this script from the newton-cosserat-cluster repository root." >&2
    exit 1
fi
NC_TOOLS=${PWD}
NC_REPO=$(cd "${NC_REPO:-${NC_TOOLS}/../newton-cosserat}" && pwd)
cd "${NC_REPO}"

# Same environment as contact_scaling_venv.sbatch.
export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/$USER/.cache/uv}
export PATH="$HOME/.local/bin:$PATH"
export UV_OFFLINE=1
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-/scratch/$USER/.cache/warp}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/scratch/$USER/.cache/jax}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/scratch/$USER/.cache/matplotlib}

OUT="artifacts/startup-probe/${SLURM_JOB_ID}"
mkdir -p "${OUT}"
STEP_TIMEOUT=${STEP_TIMEOUT:-900}  # seconds; a hung step is killed and reported

section() { printf '\n===== %s  [%s]\n' "$1" "$(date '+%F %T')"; }
# Run a command under a timeout and report its wall time and exit code.
timed() {
    local label=$1; shift
    local t0 t1 rc
    t0=$(date +%s.%N)
    timeout "${STEP_TIMEOUT}" "$@"
    rc=$?
    t1=$(date +%s.%N)
    printf '>>> %-40s %8.1f s  (exit %d%s)\n' "${label}" "$(awk "BEGIN{print ${t1} - ${t0}}")" "${rc}" \
        "$( ((rc == 124)) && echo ', TIMED OUT')"
}
# Prefix each line with a wall-clock timestamp (portable replacement for `ts`).
stamp_lines() { while IFS= read -r line; do printf '%s %s\n' "$(date +%T.%N | cut -c1-12)" "${line}"; done; }

section "node"
echo "job ${SLURM_JOB_ID} on $(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "venv: $(uv run --no-sync python -c 'import sys; print(sys.prefix)' 2>/dev/null)"
df -hT . "${UV_CACHE_DIR}" "${TMPDIR:-/tmp}" 2>/dev/null
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

section "1. bare interpreter through uv"
timed "uv run python -c pass" uv run --no-sync python -c "pass"
VENV_PY=$(uv run --no-sync python -c 'import sys; print(sys.executable)')
timed "venv python -c pass (no uv)" "${VENV_PY}" -c "pass"

section "2. import breakdown (-X importtime)"
timed "import jax, warp (importtime)" uv run --no-sync python -X importtime -c "import jax, warp" \
    2> "${OUT}/importtime.txt"
echo "Slowest cumulative imports (us):"
grep '^import time:' "${OUT}/importtime.txt" | sort -t'|' -k2 -n | tail -20

PROBE=$(cat <<'EOF'
import time
t0 = t = time.time()
def step(name):
    global t
    now = time.time(); print(f"  {name:<16}{now - t:8.1f} s", flush=True); t = now
import jax;             step("import jax")
import warp as wp;      step("import warp")
wp.init();              step("wp.init")
print("  ", jax.devices(), flush=True); step("jax.devices")
import newton, soromox; step("newton+soromox")
import newton_cosserat.solvers; step("newton_cosserat")
print(f"  {'total':<16}{time.time() - t0:8.1f} s")
EOF
)

section "3a. init steps, run 1"
timed "init probe run 1" uv run --no-sync python -c "${PROBE}"
section "3b. init steps, run 2 (warm page cache)"
timed "init probe run 2" uv run --no-sync python -c "${PROBE}"

section "4. jax.devices() with verbose XLA/PJRT logging"
TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=pjrt_client=3,gpu_client=3,cuda_executor=2 \
    timeout "${STEP_TIMEOUT}" uv run --no-sync python -c "import jax; print(jax.devices())" 2>&1 \
    | stamp_lines > "${OUT}/jax_devices_verbose.log"
echo "(full log: ${OUT}/jax_devices_verbose.log; last 40 lines)"
tail -40 "${OUT}/jax_devices_verbose.log"

section "5. syscall summary for import + init (if strace exists)"
if command -v strace >/dev/null; then
    timeout "${STEP_TIMEOUT}" strace -f -c -o "${OUT}/strace_summary.txt" \
        "${VENV_PY}" -c "import jax, warp as wp; wp.init(); jax.devices()"
    head -30 "${OUT}/strace_summary.txt"
else
    echo "strace not available; skipped"
fi

section "6. same probe with the venv copied to node-local disk"
LOCAL=${TMPDIR:-/tmp}/probe-venv-${SLURM_JOB_ID}
VENV_DIR=$(dirname "$(dirname "${VENV_PY}")")
timed "copy venv to local disk" cp -a "${VENV_DIR}" "${LOCAL}"
# The copied venv still resolves the editable install; PYTHONPATH is a fallback.
timed "init probe from local venv" env PYTHONPATH="${PWD}/src:${PWD}" "${LOCAL}/bin/python" -c "${PROBE}"
rm -rf "${LOCAL}"

# ---- CUDA version checks: is the CUDA 13 JAX stack the slow part? ----

section "7. CUDA stack: driver, bundled libraries, what actually gets loaded"
nvidia-smi --query-gpu=driver_version,compute_cap --format=csv
uv run --no-sync python - <<'EOF' 2>&1
import importlib.metadata as md, jax, warp as wp
for name in ("jax", "jaxlib", "jax-cuda13-plugin", "jax-cuda13-pjrt", "jax-cuda12-plugin", "warp-lang",
             "nvidia-cuda-runtime", "nvidia-cuda-nvcc", "nvidia-cudnn-cu13", "nvidia-nvjitlink"):
    try: print(f"  {name:<22}{md.version(name)}")
    except md.PackageNotFoundError: pass
wp.init()  # its banner reports Warp's CUDA toolkit and the driver version
jax.devices()
jax.print_environment_info()
# Which CUDA shared libraries were really mapped (bundled wheels vs. system / module paths)?
libs = sorted({l.split()[-1] for l in open("/proc/self/maps") if ".so" in l and
               any(k in l for k in ("cuda", "cublas", "cudnn", "nvrtc", "nvJitLink", "cusparse", "cusolver", "cufft"))})
print("Loaded CUDA libraries:"); print("\n".join("  " + l for l in libs))
EOF

section "8. JAX backend init: CPU-only vs CUDA 13"
timed "JAX_PLATFORMS=cpu  import+devices" env JAX_PLATFORMS=cpu \
    uv run --no-sync python -c "import jax; print(jax.devices())"
timed "JAX_PLATFORMS=cuda import+devices" env JAX_PLATFORMS=cuda \
    uv run --no-sync python -c "import jax; print(jax.devices())"
timed "Warp only: import+init" uv run --no-sync python -c "import warp as wp; wp.init()"

COMPILE=$(cat <<'EOF'
import time, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
print("  device:", jax.devices()[0])
@jax.jit
def f(x):
    for _ in range(20):
        x = jnp.tanh(x @ x.T) + jnp.sin(x)
    return jnp.linalg.solve(x @ x.T + jnp.eye(x.shape[0]), x).sum()
x = jnp.ones((256, 256))
t = time.time(); f(x).block_until_ready(); print(f"  first call (compile) {time.time() - t:8.2f} s")
t = time.time(); f(x).block_until_ready(); print(f"  second call (cached) {time.time() - t:8.4f} s")
EOF
)
section "9. JAX JIT compile speed (no persistent cache), CPU vs CUDA 13"
# Slow ptxas / driver PTX JIT would show up here, i.e. in the benchmark's build phase.
timed "compile probe, cpu" env JAX_PLATFORMS=cpu JAX_COMPILATION_CACHE_DIR= \
    uv run --no-sync python -c "${COMPILE}"
timed "compile probe, cuda13" env JAX_PLATFORMS=cuda JAX_COMPILATION_CACHE_DIR= \
    uv run --no-sync python -c "${COMPILE}"

section "10. Same tests with a CUDA 12 JAX venv (optional)"
# Create it once on the LOGIN node, same JAX version as the project venv:
#   JV=$(uv run --no-sync python -c 'import jax; print(jax.__version__)')
#   uv venv /scratch/$USER/jax-cu12 --python "$(uv run --no-sync python -c 'import sys; print(sys.executable)')"
#   uv pip install --python /scratch/$USER/jax-cu12 "jax[cuda12]==$JV"
CU12_VENV=${CU12_VENV:-/scratch/$USER/jax-cu12}
if [[ -x "${CU12_VENV}/bin/python" ]]; then
    timed "cuda12: import+devices" env JAX_PLATFORMS=cuda \
        "${CU12_VENV}/bin/python" -c "import jax; print(jax.__version__, jax.devices())"
    timed "compile probe, cuda12" env JAX_PLATFORMS=cuda JAX_COMPILATION_CACHE_DIR= \
        "${CU12_VENV}/bin/python" -c "${COMPILE}"
else
    echo "No CUDA 12 venv at ${CU12_VENV}; skipped (see comment above to create one)."
fi

section "done"
echo "Artifacts in ${OUT}"
