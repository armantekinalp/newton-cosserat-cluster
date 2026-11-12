#!/bin/bash
# Contact-scaling benchmark on a DelftBlue GPU node inside the newton-cosserat Apptainer image.
#
# Same sweep as diagnostics/contact_scaling_venv.sbatch, but Python and every dependency come from one .sif file
# copied to the node-local SSD, instead of the /scratch venv whose thousands of small files
# made each worker process take minutes to start.
#
# One-time setup on a LOGIN node (and again after uv.lock changes):
#   bash delftblue/build_sif.sh      # -> /scratch/$USER/containers/newton-cosserat.sif
#
# Submit from the newton-cosserat-cluster root (NC_REPO defaults to ../newton-cosserat):
#   sbatch delftblue/contact_scaling.sbatch
# Optional overrides (exported into the job), see "Benchmark parameters" below, e.g.
#   sbatch --export=ALL,MODES=hydroelastic,CUDA_WORLDS="1 64 1024" delftblue/contact_scaling.sbatch
#
#SBATCH --job-name=contact-scaling-apptainer
#SBATCH --account=innovation
#SBATCH --partition=gpu-a100
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ---- Benchmark parameters (each can be overridden with sbatch --export=ALL,NAME=value) ----
NC_SIF=${NC_SIF:-/scratch/${USER}/containers/newton-cosserat.sif}
MODES=${MODES:-"point hydroelastic"}          # contact modes, run one after another
DEVICES=${DEVICES:-"cuda:0"}                  # add "cpu" for CPU numbers (single-threaded, slow to compile)
CPU_WORLDS=${CPU_WORLDS:-"1"}                 # environment counts swept on the CPU
CUDA_WORLDS=${CUDA_WORLDS:-"1 2 4 8 16 32 64 128 256 512 1024"}  # ... and on the GPU
# Hydroelastic discretization (ignored in point mode).
HYDROELASTIC_SECTORS=${HYDROELASTIC_SECTORS:-16}       # rod perimeter subdivisions, >= 4
HYDROELASTIC_RINGS=${HYDROELASTIC_RINGS:-2}            # rod radial subdivisions
HYDROELASTIC_RESOLUTION=${HYDROELASTIC_RESOLUTION:-0.01}  # rod axial spacing [m]
ENVIRONMENT_RESOLUTION=${ENVIRONMENT_RESOLUTION:-0.005}   # curved environment shapes [m]
# Any other Config field, e.g. "--steps 500 --repeats 3 --stiffness 2e5". Empty runs the
# benchmark defaults (1000 steps, 5 timed repeats); "--steps 100 --repeats 2" is a quick check.
# (No colon in ${EXTRA_ARGS-...}: an explicitly empty value must not fall back to the default.)
EXTRA_ARGS=${EXTRA_ARGS-""}
# --------------------------------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR}"
if [[ ! -f lib/sweep.sh ]]; then
    echo "Submit this script from the newton-cosserat-cluster repository root." >&2
    exit 1
fi
NC_CLUSTER=delftblue
RUN_TAG=delftblue-apptainer   # results in $NC_REPO/artifacts/contact-scaling/$RUN_TAG-<jobid>/
source lib/sweep.sh
