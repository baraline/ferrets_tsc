"""Sparse helpers of the Gram (dual) ridge route."""

import numpy as np
from numba import njit, prange


@njit(inline="always")
def _lower_bound(ids, lo, hi, j0):
    """First e in [lo, hi) with ids[e] >= j0 (ids ascending in a row)."""
    while lo < hi:
        mid = (lo + hi) >> 1
        if np.int64(ids[mid]) < j0:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True, parallel=True, nogil=True)
def _densify_rows(ptr, ids, vals, c0, c1, T, o0):
    """T[r, o0:o0 + c1 - c0] = row r's entries with column in [c0, c1),
    zeros elsewhere: a column segment of a row-major slab. One row per
    task, so tasks write disjoint rows."""
    s = c1 - c0
    for r in prange(ptr.shape[0] - 1):
        for j in range(s):
            T[r, o0 + j] = 0.0
        e = _lower_bound(ids, ptr[r], ptr[r + 1], c0)
        end = ptr[r + 1]
        while e < end:
            j = np.int64(ids[e])
            if j >= c1:
                break
            T[r, o0 + j - c0] = vals[e]
            e += 1


@njit(cache=True, parallel=True, nogil=True)
def _rows_dot_acc(ptr, ids, vals, vec, acc):
    """acc[r] += val * vec[id] over the row's entries in entry order, so
    feeding the blocks one after another gives the whole row's sparse dot in
    a fixed order."""
    for r in prange(ptr.shape[0] - 1):
        s = acc[r]
        for e in range(ptr[r], ptr[r + 1]):
            s += vals[e] * vec[ids[e]]
        acc[r] = s


@njit(cache=True, nogil=True)
def _mirror_upper(K):
    n = K.shape[0]
    for i in range(n):
        for j in range(i):
            K[i, j] = K[j, i]


@njit(cache=True, parallel=True, nogil=True)
def _dual_to_primal(ptr, ids, vals, C, bounds, out):
    """out[j, k] = sum_r X[r, j] C[r, k], rows ascending inside every column.
    Parallel over the column ranges of ``bounds``; each scans every row from
    a binary-searched start, so no transpose is built and a column's sum
    does not depend on how the columns are split."""
    n = ptr.shape[0] - 1
    n_heads = C.shape[1]
    for g in prange(bounds.shape[0] - 1):
        j0 = bounds[g]
        j1 = bounds[g + 1]
        for j in range(j0, j1):
            for k in range(n_heads):
                out[j, k] = 0.0
        for r in range(n):
            e = _lower_bound(ids, ptr[r], ptr[r + 1], j0)
            end = ptr[r + 1]
            while e < end:
                j = np.int64(ids[e])
                if j >= j1:
                    break
                v = vals[e]
                for k in range(n_heads):
                    out[j, k] += v * C[r, k]
                e += 1
