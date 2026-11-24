"""The non-finite count of an input collection."""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, nogil=True)
def _count_nonfinite(X, n_blocks):
    """Number of non-finite values of X (n, C, m), summed over n_blocks row
    blocks."""
    n, C, m = X.shape
    bad = np.zeros(n_blocks, dtype=np.int64)
    for b in prange(n_blocks):
        lo = b * n // n_blocks
        hi = (b + 1) * n // n_blocks
        k = 0
        for i in range(lo, hi):
            for c in range(C):
                for t in range(m):
                    if not np.isfinite(X[i, c, t]):
                        k += 1
        bad[b] = k
    return bad.sum()
