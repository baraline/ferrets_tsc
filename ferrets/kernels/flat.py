"""Products of the flat design used by the LSQR route."""

import numpy as np
from numba import njit, prange

from ..kernels.gram import _lower_bound


@njit(cache=True, parallel=True, nogil=True)
def _fd_colmean(ptr, ids, vals, off, rows, rp, rj0, rj1, scale32, out32):
    """Float32 column mean of the row subset ``rows``, in row order: each
    value is scaled by scale32 (float32(1/n)) and summed in float32, as
    scipy does. Parallel over the column ranges (rp, rj0, rj1) (block,
    first column, end column); each range finds every row's first entry
    >= j0 by binary search, so a column's sum does not depend on the
    ranges."""
    for g in prange(rp.shape[0]):
        p = rp[g]
        j0 = rj0[g]
        j1 = rj1[g]
        base = off[p]
        for j in range(j0, j1):
            out32[base + j] = np.float32(0)
        for i in range(rows.shape[0]):
            r = rows[i]
            lo = ptr[p, r]
            hi = ptr[p, r + 1]
            e = _lower_bound(ids, lo, hi, j0) if j0 > 0 else lo
            while e < hi:
                j = np.int64(ids[e])
                if j >= j1:
                    break
                cc = base + j
                out32[cc] = out32[cc] + np.float32(vals[e] * scale32)
                e += 1


@njit(cache=True, parallel=True, nogil=True)
def _fd_colmean_acc(ptr, ids, vals, off, rows, rscale32, rp, rj0, rj1, out32):
    """_fd_colmean's arithmetic with a per-row scale and without the zeroing
    (the caller zeroes out32): out32[col] += float32(val * rscale32[i]) over
    the rows in order. With rscale32[i] = w_i / sum w it is the weighted
    column mean of the class-weighted ridge. Parallel over column ranges as
    in _fd_colmean, so a column's sum does not depend on the ranges."""
    for g in prange(rp.shape[0]):
        p = rp[g]
        j0 = rj0[g]
        j1 = rj1[g]
        base = off[p]
        for i in range(rows.shape[0]):
            r = rows[i]
            lo = ptr[p, r]
            hi = ptr[p, r + 1]
            e = _lower_bound(ids, lo, hi, j0) if j0 > 0 else lo
            sc = rscale32[i]
            while e < hi:
                j = np.int64(ids[e])
                if j >= j1:
                    break
                cc = base + j
                out32[cc] = out32[cc] + np.float32(vals[e] * sc)
                e += 1


@njit(cache=True, parallel=True, nogil=True)
def _fd_matvec(ptr, ids, vals, off, rows, V, out, n_blocks):
    """out[i, g] = X[rows[i]] . V[:, g], the blocks in pseudo-channel order
    and each row's entries in order, so every column's sum is the sparse row
    dot, carried across the blocks. Parallel over row blocks; inside one,
    block-outer: all its rows against block p before block p + 1, so the
    slice of V that block p reads stays in cache across the rows. One
    column: a register sum; more: the columns of an entry together (one
    cache line of V; the caller pads the columns to a multiple of 4)."""
    P = ptr.shape[0]
    G = V.shape[1]
    nr = rows.shape[0]
    for b in prange(n_blocks):
        lo = b * nr // n_blocks
        hi = (b + 1) * nr // n_blocks
        acc = np.zeros((hi - lo, G))
        for p in range(P):
            base = off[p]
            for i in range(lo, hi):
                r = rows[i]
                q = i - lo
                e0 = ptr[p, r]
                e1 = ptr[p, r + 1]
                if G >= 2:
                    for e in range(e0, e1):
                        a = vals[e]
                        c = base + np.int64(ids[e])
                        for g in range(G):
                            acc[q, g] += a * V[c, g]
                else:
                    s = acc[q, 0]
                    for e in range(e0, e1):
                        s += vals[e] * V[base + np.int64(ids[e]), 0]
                    acc[q, 0] = s
        for i in range(lo, hi):
            for g in range(G):
                out[i, g] = acc[i - lo, g]


@njit(cache=True, parallel=True, nogil=True)
def _fd_rmatvec(ptr, ids, vals, off, rows, U, gs, sp, sj0, sj1, sub, out):
    """out[off[p] + j, g] = sum_i X_p[rows[i], j] U[i, g], rows ascending.
    Used when the row-parts scratch of _fd_rmatvec_parts would not fit
    (config.RMV_SCRATCH_MB). Parallel over groups, one per thread; group k
    holds the column spans gs[k]:gs[k + 1] of (sp, sj0, sj1). A span finds
    each row's first entry >= j0 by binary search once, then walks
    sub-ranges of ``sub`` columns with a per-row cursor, so the sums being
    accumulated (sub x G) stay in cache. No scratch and no reduction pass;
    a column's sum does not depend on the spans or the threads."""
    nr = rows.shape[0]
    G = U.shape[1]
    for k in prange(gs.shape[0] - 1):
        cur = np.empty(nr, np.int64)
        for q in range(gs[k], gs[k + 1]):
            p = sp[q]
            j0 = sj0[q]
            j1 = sj1[q]
            base = off[p]
            for i in range(nr):
                r = rows[i]
                lo = ptr[p, r]
                cur[i] = (_lower_bound(ids, lo, ptr[p, r + 1], j0) if j0 > 0
                          else lo)
            for s0 in range(j0, j1, sub):
                s1 = min(j1, s0 + sub)
                ob = base + s0
                for jj in range(s1 - s0):
                    for g in range(G):
                        out[ob + jj, g] = 0.0
                for i in range(nr):
                    r = rows[i]
                    e = cur[i]
                    end = ptr[p, r + 1]
                    while e < end:
                        jj = np.int64(ids[e]) - s0
                        if jj >= s1 - s0:
                            break
                        a = vals[e]
                        for g in range(G):
                            out[ob + jj, g] += a * U[i, g]
                        e += 1
                    cur[i] = e


@njit(cache=True, parallel=True, nogil=True)
def _fd_rmatvec_parts(ptr, ids, vals, off, rows, U, bands, n_parts, scratch,
                      out):
    """X_S^T U: per band of consecutive blocks, n_parts fixed row partitions
    each sum their rows ascending into their own scratch (one sequential
    pass over each row's entries), then the partitions are added in order.
    A column's sum depends on n_parts, never on the thread count.
    scratch: (n_parts, columns of the widest band, G) float64, overwritten."""
    nr = rows.shape[0]
    G = U.shape[1]
    for b in range(bands.shape[0] - 1):
        p0 = bands[b]
        p1 = bands[b + 1]
        c0 = off[p0]
        width = off[p1] - c0
        for g in prange(n_parts):
            for j in range(width):
                for k in range(G):
                    scratch[g, j, k] = 0.0
            lo = g * nr // n_parts
            hi = (g + 1) * nr // n_parts
            for p in range(p0, p1):
                bp = off[p] - c0
                for i in range(lo, hi):
                    r = rows[i]
                    for e in range(ptr[p, r], ptr[p, r + 1]):
                        a = vals[e]
                        j = bp + np.int64(ids[e])
                        for k in range(G):
                            scratch[g, j, k] += a * U[i, k]
        for j in prange(width):
            for k in range(G):
                s = scratch[0, j, k]
                for g in range(1, n_parts):
                    s += scratch[g, j, k]
                out[c0 + j, k] = s
