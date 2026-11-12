# DelftBlue Apptainer image

A dependency image for running newton-cosserat, in particular the
contact-scaling benchmark (`tools/benchmarks/contact_scaling.md` in newton-cosserat), on the
TU Delft DelftBlue cluster.

Why: a venv on `/scratch` (BeeGFS) consists of thousands of small files, and loading
them made every Python process spend 4–8 minutes importing JAX, Warp and Newton. A
`.sif` image is a single compressed file; jobs copy it to the node-local SSD, so
start-up reads nothing from BeeGFS.

| File | Purpose |
| --- | --- |
| `newton-cosserat.def` | Image recipe: Python 3.13, locked dependencies with the `cuda13` and `viz` extras, no torch |
| `build_sif.sh` | Build the image on a login node with `--fakeroot` and copy it to `/scratch` |
| `env.sh` | DelftBlue settings for `../lib/env.sh`: `nc_exec`, `nc_stage_image`, `nc_check_lock`, cache tarball |
| `contact_scaling.sbatch` | Contact-scaling sweep job |
| `diagnostics/` | Earlier uv-venv job and the start-up / storage probes that motivated the image |

## What is in the image

- Python 3.13 and every package from `uv.lock` for the core dependencies and the
  `cuda13` and `viz` extras. Unlike the [DAIC image](../daic/README.md), the CUDA 13
  JAX plugin is used unchanged, because DelftBlue's driver (615) supports it. PyTorch,
  Triton and cuSPARSELt are not installed; the benchmark does not use them.
- `/opt/venv-freeze.txt` lists every installed version (each job copies it to
  `container-packages.txt`); `/opt/uv.lock.sha256` records the lock file the image was
  built from, and jobs warn if the checkout's `uv.lock` differs.
- newton-cosserat is installed in editable mode at `/workspace/newton-cosserat/src`.
  `nc_exec` bind-mounts your checkout there, so code changes need no rebuild. Rebuild
  only when `uv.lock` changes.

## 1. Build the image (login node)

The build needs internet access, so it runs on a login node. Temporary files stay on
the login node's `/tmp` (about 15 GB); only the finished image is written to `/scratch`:

```bash
cd /scratch/$USER/newton-cosserat-cluster   # next to /scratch/$USER/newton-cosserat
bash delftblue/build_sif.sh   # -> /scratch/$USER/containers/newton-cosserat.sif
```

## 2. Run the benchmark

```bash
sbatch delftblue/contact_scaling.sbatch
```

The job copies the image to `/tmp`, records how long that and the JAX/Warp start-up
take in `timings.txt`, and writes results to
`artifacts/contact-scaling/delftblue-apptainer-<jobid>/` in the newton-cosserat checkout. Parameters at the top of the
script can be overridden through `--export`, including `NC_SIF` for another image.

For interactive work on a GPU node:

```bash
export NC_SIF=/scratch/$USER/containers/newton-cosserat.sif
source delftblue/env.sh
nc_stage_image   # optional: copy the image to /tmp first
nc_exec python -c "import jax, warp as wp; wp.init(); print(jax.devices())"
```

Warp and JAX caches live on node-local `/tmp` while a job runs. The job unpacks them
from `/scratch/$USER/.cache/newton-cosserat-caches.tar` at the start and writes them back
as that single tarball after each contact mode (`nc_restore_caches` / `nc_save_caches`).
Copying the ~1200 small cache files individually from BeeGFS took about 160 s; the
tarball takes under a second.
