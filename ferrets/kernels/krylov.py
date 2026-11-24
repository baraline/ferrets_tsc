"""Golub-Kahan step kernels of the lockstep LSQR."""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, nogil=True)
def _gk_orth(Vw, c, Vv, g, beta, B, kk, vn):
    """One Golub-Kahan right step of column g, fused: vn = Vw[:, c] - beta
    Vv[:, g], re-orthogonalised against the basis rows B[:kk] by one
    classical Gram-Schmidt pass (vn -= B^T (B vn)). Returns ||vn||.
    Parallel over fixed chunks of the F entries; the dot products are summed
    chunk by chunk in order, so they do not depend on the thread count."""
    F = vn.shape[0]
    clen = 4096
    nch = (F + clen - 1) // clen
    part = np.empty((max(1, nch), kk + 1))
    for q in prange(nch):
        f0 = q * clen
        f1 = min(F, f0 + clen)
        for f in range(f0, f1):
            vn[f] = Vw[f, c] - beta * Vv[f, g]
        for k in range(kk):
            s = 0.0
            for f in range(f0, f1):
                s += B[k, f] * vn[f]
            part[q, k] = s
    h = np.empty(max(1, kk))
    for k in range(kk):
        s = 0.0
        for q in range(nch):
            s += part[q, k]
        h[k] = s
    for q in prange(nch):
        f0 = q * clen
        f1 = min(F, f0 + clen)
        proj = np.zeros(f1 - f0)
        for k in range(kk):
            hk = h[k]
            for f in range(f0, f1):
                proj[f - f0] += B[k, f] * hk
        s = 0.0
        for f in range(f0, f1):
            x = vn[f] - proj[f - f0]
            vn[f] = x
            s += x * x
        part[q, kk] = s
    ss = 0.0
    for q in range(nch):
        ss += part[q, kk]
    return np.sqrt(ss)


@njit(cache=True, parallel=True, nogil=True)
def _gk_store(vn, al, Brow, Vv, g):
    """v = vn / al into the basis row and column g of Vv."""
    for f in prange(vn.shape[0]):
        v = vn[f] / al
        Brow[f] = v
        Vv[f, g] = v
