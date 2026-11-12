# Settings for running newton-cosserat inside the DelftBlue Apptainer image:
#   export NC_SIF=/scratch/$USER/containers/newton-cosserat.sif
#   source delftblue/env.sh
#   nc_exec python -c "import jax; print(jax.devices())"
# Variables and functions: see lib/env.sh.

NC_CLUSTER=delftblue
NC_BINDS="/scratch /tmp"
NC_CACHE_TAR=${NC_CACHE_TAR:-/scratch/${USER}/.cache/newton-cosserat-caches.tar}
NC_REBUILD_HINT="bash delftblue/build_sif.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh" || return 1
