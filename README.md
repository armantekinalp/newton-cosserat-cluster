# newton-cosserat-cluster

Container images and Slurm jobs for running
[newton-cosserat](https://github.com/armantekinalp/newton-cosserat) on the TU Delft clusters (DelftBlue and DAIC).
Kept separate so the main repository stays free of site-specific setup.

## Layout

```text
<workdir>/
├── newton-cosserat/            # the code (NC_REPO)
└── newton-cosserat-cluster/    # this repository
    ├── lib/                    # shared: env.sh (nc_exec, image staging, caches), sweep.sh (job body)
    ├── delftblue/              # DelftBlue image recipe, build_sif.sh (builds both images), sweep job
    │   └── diagnostics/        # uv-venv job and start-up/storage probes
    └── daic/                   # DAIC image recipe, smoke test, sweep job
```

Scripts find the code through `NC_REPO`, which defaults to `../newton-cosserat`. Set it
if your checkout lives elsewhere. Run every command from this repository's root.

| Cluster | GPU driver | JAX backend | Image built on | Guide |
| --- | --- | --- | --- | --- |
| DelftBlue | 615 (CUDA 13) | locked `cuda13` extra | DelftBlue login node (`--fakeroot`) | [delftblue/README.md](delftblue/README.md) |
| DAIC | 550 (CUDA ≤ 12.4) | `jax[cuda12]` at the locked version | DelftBlue login node (DAIC cannot build) | [daic/README.md](daic/README.md) |

Both images contain only the dependencies. The checkout is bind-mounted at run time, so
code changes need no rebuild. Rebuild only when `uv.lock` changes; jobs warn
when the image is stale.

## Quick start (DelftBlue)

```bash
cd /scratch/$USER
git clone https://github.com/armantekinalp/newton-cosserat.git newton-cosserat
git clone <newton-cosserat-cluster-url> newton-cosserat-cluster
cd newton-cosserat-cluster

bash delftblue/build_sif.sh                 # once per uv.lock, on a login node (~15 GB in /tmp)
sbatch delftblue/contact_scaling.sbatch     # results in ../newton-cosserat/artifacts/contact-scaling/
```

Override sweep parameters with `--export`, e.g.
`sbatch --export=ALL,MODES=hydroelastic,CUDA_WORLDS="1 64 1024" delftblue/contact_scaling.sbatch`.
Each job script lists its parameters at the top.

## Quick start (DAIC)

```bash
# on DelftBlue: build the DAIC image and copy it over
DEF=daic/newton-cosserat.def OUTPUT=/scratch/$USER/containers/newton-cosserat-daic.sif bash delftblue/build_sif.sh
scp /scratch/$USER/containers/newton-cosserat-daic.sif <netid>@login1.daic.tudelft.nl:/tudelft.net/staff-umbrella/<project>/containers/

# on DAIC, with both checkouts side by side
export NC_SIF=/tudelft.net/staff-umbrella/<project>/containers/newton-cosserat-daic.sif
daic/check.sh                                              # on a GPU node (sinteractive)
sbatch --export=ALL,NC_SIF=$NC_SIF daic/contact_scaling.sbatch
```

## Interactive use

```bash
export NC_SIF=...            # path to the .sif
source delftblue/env.sh      # or daic/env.sh
nc_exec python -m examples.gvs.free_space --device cuda:0
```

`nc_exec` runs a command in the image with the GPU (`--nv`), the checkout and the
Warp/JAX caches mounted. Do not `module load cuda`; the wheels bring their own CUDA libraries.
