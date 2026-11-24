"""Tables of the term frequency by count c: sublinear (1 + ln c) or raw."""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _fill_tf(out):
    """out[c] = 1 + ln c, computed in float64 (out[0] unused)."""
    out[0] = 0.0
    for c in range(1, out.shape[0]):
        out[c] = 1.0 + np.log(np.float64(c))


@njit(cache=True, nogil=True)
def _fill_tf_store(out):
    """The batch design's float32 sublinear tf, float32(log(float32 c)) + 1
    (out[0] = 0)."""
    out[0] = np.float32(0.0)
    for c in range(1, out.shape[0]):
        out[c] = np.float32(np.log(np.float32(c))) + np.float32(1.0)


@njit(cache=True, nogil=True)
def _fill_count(out):
    """out[c] = c (the raw count; out[0] unused), for a float32 or float64
    out."""
    out[0] = 0.0
    for c in range(1, out.shape[0]):
        out[c] = c
