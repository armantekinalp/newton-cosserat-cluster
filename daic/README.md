# DAIC Apptainer image

A dependency image for running newton-cosserat, in particular the
contact-scaling benchmark (`tools/benchmarks/contact_scaling.md` in newton-cosserat), on the
TU Delft [DAIC](https://daic.tudelft.nl/) cluster.

| File | Purpose |
| --- | --- |
| `newton-cosserat.def` | Image recipe: Python 3.13, locked dependencies plus the `viz` extra, CUDA 12 JAX |
| `env.sh` | `nc_exec`: run a command in the image with GPU, checkout and caches mounted |
| `check.sh` | Smoke test on a GPU node |
| `contact_scaling.sbatch` | Contact-scaling sweep job |

## What is in the image

- Python 3.13 and every package from `uv.lock` for the core dependencies and the
  `viz` extra (matplotlib for result plots). No dev tools, examples or Isaac Lab.
- **Deviation from the lock file:** DAIC's NVIDIA driver (550, CUDA ≤ 12.4) cannot
  run the CUDA 13 JAX plugin of the `cuda13` extra, which needs driver ≥ 580. The image
  installs `jax[cuda12]` at the locked JAX version instead; the build fails if that
  changes any locked package. Warp's locked wheel is already a CUDA 12 build. The
  `cuda13` extra's PyTorch is not installed; the benchmark does not use it.
- `/opt/venv-freeze.txt` lists every installed version (each job copies it to
  `container-packages.txt`); `/opt/uv.lock.sha256` records the lock file the image was
  built from, and jobs warn if the checkout's `uv.lock` differs.
- newton-cosserat is installed in editable mode at `/workspace/newton-cosserat/src`.
  `nc_exec` bind-mounts your checkout there, so code changes need no rebuild. Rebuild
  only when `uv.lock` changes.

## 1. Build the image (DelftBlue login node)

DAIC disables user namespaces, so the image cannot be built there. DelftBlue login nodes
support `apptainer build --fakeroot`; from `newton-cosserat-cluster` on DelftBlue:

```bash
DEF=daic/newton-cosserat.def OUTPUT=/scratch/$USER/containers/newton-cosserat-daic.sif \
    bash delftblue/build_sif.sh
scp /scratch/$USER/containers/newton-cosserat-daic.sif \
    <netid>@login1.daic.tudelft.nl:/tudelft.net/staff-umbrella/<project>/containers/
```

On DAIC, keep newton-cosserat and newton-cosserat-cluster side by side in project storage.

## 2. Smoke test (DAIC GPU node)

From `newton-cosserat-cluster` on DAIC:

```bash
sinteractive --gres=gpu:a40:1 --cpus-per-task=4 --mem=16G --time=00:45:00
export NC_SIF=/tudelft.net/staff-umbrella/<project>/containers/newton-cosserat-daic.sif
daic/check.sh
```

It checks that JAX and Warp both see the GPU, then runs one short point and one short
hydroelastic case. JAX may warn that the driver is older than its bundled `ptxas` and
disable parallel compilation; that is expected with driver 550.

## 3. Run the benchmark

```bash
sbatch --export=ALL,NC_SIF=$NC_SIF daic/contact_scaling.sbatch
```

The job copies the image to node-local `/tmp`, restores the Warp/JAX caches from
`newton-cosserat-caches.tar` next to the image (`NC_CACHE_TAR`) and saves them back after
each contact mode. Results go to `artifacts/contact-scaling/daic-<jobid>/` in the
newton-cosserat checkout. The parameters at the top of `contact_scaling.sbatch` (modes,
world counts, hydroelastic resolution, extra arguments) can be overridden through
`--export`. The defaults are account `me-cor`, partition `general`, QoS `short` (≤ 4 h)
and one A40.

For interactive work: `source daic/env.sh` and prefix commands with `nc_exec`.
