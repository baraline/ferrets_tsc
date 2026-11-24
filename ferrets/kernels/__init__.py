"""numba kernels only: nothing here imports the transform, classification or
estimator layers. Importing it checks the numba cache against the kernel
sources (helpers/nbcache.py)."""
from ..helpers import nbcache as _nbcache

_nbcache.check()
