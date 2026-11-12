#!/bin/bash
# Contact-scaling benchmark on a DAIC GPU node inside the newton-cosserat Apptainer image.
#
# One-time setup: build the image and convert it to a .sif (daic/README.md),
# then run daic/check.sh once on a GPU node.
#
# Submit from the newton-cosserat-cluster root (NC_REPO defaults to ../newton-cosserat):
#   sbatch --export=ALL,NC_SIF=/tudelft.net/staff-umbrella/<project>/containers/newton-cosserat-daic.sif \
#       daic/contact_scaling.sbatch
# Optional overrides (exported into the job), see "Benchmark parameters" below, e.g.
#   sbatch --export=ALL,NC_SIF=...,MODES=hydroelastic,CUDA_WORLDS="1 64 1024" daic/contact_scaling.sbatch
# GPU type: the default is an A40; use --gres=gpu:l40:1 on the sbatch command line for an L40.
# The default QoS `short` allows up to 4 h; pass --qos=medium --time=... for longer sweeps.
#
#SBATCH --job-name=contact-scaling
#SBATCH --account=me-cor
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ---- Benchmark parameters (each can be overridden with sbatch --export=ALL,NAME=value) ----
MODES=${MODES:-"point hydroelastic"}          # contact modes, run one after another
DEVICES=${DEVICES:-"cuda:0"}                  # add "cpu" for CPU numbers (single-threaded, slow to compile)
CPU_WORLDS=${CPU_WORLDS:-"1"}                 # environment counts swept on the CPU
CUDA_WORLDS=${CUDA_WORLDS:-"1 2 4 8 16 32 64 128 256 512 1024"}  # ... and on the GPU
# Hydroelastic discretization (ignored in point mode).
HYDROELASTIC_SECTORS=${HYDROELASTIC_SECTORS:-16}       # rod perimeter subdivisions, >= 4
HYDROELASTIC_RINGS=${HYDROELASTIC_RINGS:-2}            # rod radial subdivisions
HYDROELASTIC_RESOLUTION=${HYDROELASTIC_RESOLUTION:-0.01}  # rod axial spacing [m]
ENVIRONMENT_RESOLUTION=${ENVIRONMENT_RESOLUTION:-0.005}   # curved environment shapes [m]
# Any other Config field, e.g. "--steps 500 --repeats 3 --stiffness 2e5".
EXTRA_ARGS=${EXTRA_ARGS:-""}
# --------------------------------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR}"
if [[ ! -f lib/sweep.sh ]]; then
    echo "Submit this script from the newton-cosserat-cluster repository root." >&2
    exit 1
fi
NC_CLUSTER=daic
RUN_TAG=daic   # results in $NC_REPO/artifacts/contact-scaling/$RUN_TAG-<jobid>/
source lib/sweep.sh
