#!/usr/bin/env bash
# Smoke test for the DAIC image. Run on a GPU node:
#   sinteractive --gres=gpu:a40:1 --cpus-per-task=4 --mem=16G --time=00:45:00
#   NC_SIF=/path/to/newton-cosserat-daic.sif daic/check.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

nvidia-smi --query-gpu=name,driver_version --format=csv

echo "=== JAX / Warp devices"
nc_exec python - <<'EOF'
import jax
import warp as wp

wp.init()
print("JAX", jax.__version__, jax.devices())
print("Warp", wp.config.version, "CUDA devices:", wp.get_cuda_device_count())
assert any(d.platform == "gpu" for d in jax.devices()), "JAX does not see the GPU"
assert wp.get_cuda_device_count() > 0, "Warp does not see the GPU"
EOF

echo "=== one short contact-scaling case per contact mode"
out=/tmp/${USER}/newton-cosserat-smoke
for mode in point hydroelastic; do
    nc_exec python -m tools.benchmarks.contact_scaling --contact-mode "${mode}" \
        --devices cuda:0 --cuda-worlds 1 --steps 100 --repeats 1 --output "${out}/${mode}/results.json"
done
echo "Smoke test passed; results in ${out}"
