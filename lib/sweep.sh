# Contact-scaling sweep inside the Apptainer image; the body of delftblue/contact_scaling.sbatch
# and daic/contact_scaling.sbatch. Those set NC_CLUSTER, RUN_TAG and the benchmark parameters
# (MODES, DEVICES, CPU_WORLDS, CUDA_WORLDS, HYDROELASTIC_*, ENVIRONMENT_RESOLUTION, EXTRA_ARGS),
# then source this file from the newton-cosserat-cluster root.

# Per-job cache directory on node-local disk, removed when the job exits (its contents are
# saved to NC_CACHE_TAR after every contact mode). /tmp is not cleaned automatically.
export NC_CACHE=/tmp/${USER}/newton-cosserat-cache-${SLURM_JOB_ID}

# Defines nc_stage_image, nc_check_lock, nc_restore_caches, nc_save_caches, nc_exec, NC_REPO, NC_TOOLS.
source "${NC_CLUSTER}/env.sh"
cd "${NC_REPO}"

RUN_DIR="artifacts/contact-scaling/${RUN_TAG}-${SLURM_JOB_ID}"
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

image=${NC_SIF}
nc_stage_image
stamp "copy image to node-local /tmp"
nc_check_lock
nc_restore_caches
stamp "restore Warp/JAX caches"

{
    echo "job ${SLURM_JOB_ID} on $(hostname) at $(date -Is)"
    echo "image ${image} ($(stat -c '%y' "${image}")), staged at ${NC_SIF}"
    echo "newton-cosserat $(git rev-parse HEAD 2>/dev/null || echo unknown) $(git status --porcelain 2>/dev/null | wc -l) dirty files"
    echo "newton-cosserat-cluster $(git -C "${NC_TOOLS}" rev-parse HEAD 2>/dev/null || echo unknown)"
    lscpu | grep -E "Model name|^CPU\(s\)|MHz" || true
    nvidia-smi
} | tee "${RUN_DIR}/environment.txt"

nc_exec python -c "import jax, warp as wp; wp.init(); print('JAX devices:', jax.devices())" \
    | tee -a "${RUN_DIR}/environment.txt"
nc_exec cat /opt/venv-freeze.txt > "${RUN_DIR}/container-packages.txt"
stamp "environment probe (JAX/Warp init)"

# Sample GPU utilisation in the background for the whole job.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 5 \
    > "${RUN_DIR}/gpu-usage.csv" &
SMI_PID=$!
# On exit (also on failure or scancel): remove the staged image copy (never the original)
# and this job's cache directory.
clean_up() {
    kill "${SMI_PID}" 2>/dev/null || true
    [[ "${NC_SIF}" != "${image}" ]] && rm -f "${NC_SIF}"
    rm -rf "${NC_CACHE}"
}
trap clean_up EXIT
trap 'exit 143' TERM INT  # turn scancel / time-limit signals into a normal exit so EXIT runs

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
    srun --ntasks=1 bash -c 'source "${NC_TOOLS}/${NC_CLUSTER}/env.sh" && nc_exec "$@"' _ \
        python -m tools.benchmarks.contact_scaling "${args[@]}" ${EXTRA_ARGS} \
        || mode_status=$?
    echo "=== ${mode} finished with exit code ${mode_status}"
    stamp "benchmark ${mode} (exit ${mode_status})"
    # Save after every mode so a job hitting its time limit still keeps compiled kernels.
    nc_save_caches
    ((mode_status)) && status=${mode_status}
    [[ -f "${output}" ]] && outputs+=("${output}")
done

if ((${#outputs[@]})); then
    nc_exec python -m tools.benchmarks.contact_scaling.plot "${outputs[@]}" \
        --output "${RUN_DIR}/contact-scaling.png" || echo "Plotting failed." >&2
    stamp "plotting"
fi

stamp "job finished"
echo "Results in ${NC_REPO}/${RUN_DIR}"
exit "${status}"
