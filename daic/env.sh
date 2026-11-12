# Settings for running newton-cosserat inside the DAIC Apptainer image:
#   export NC_SIF=/tudelft.net/staff-umbrella/<project>/containers/newton-cosserat-daic.sif
#   source daic/env.sh
#   nc_exec python -c "import jax; print(jax.devices())"
# Variables and functions: see lib/env.sh. Caches stay on node-local /tmp while running,
# because staff-umbrella/bulk are Windows shares and $HOME has a small quota; between jobs
# they are kept as one tarball next to the image.

NC_CLUSTER=daic
NC_BINDS="/tudelft.net /tmp"
NC_CACHE_TAR=${NC_CACHE_TAR:-$(dirname "${NC_SIF:-.}")/newton-cosserat-caches.tar}
NC_REBUILD_HINT="DEF=daic/newton-cosserat.def OUTPUT=... bash delftblue/build_sif.sh (on DelftBlue)"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh" || return 1
