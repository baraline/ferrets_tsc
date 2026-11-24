"""Averaged least-squares SGD kernels of the streaming head."""

import numpy as np
from numba import njit, prange

from ..kernels.team import _barrier


@njit(cache=True, parallel=True, nogil=True, error_model="numpy")
def _asgd_pass(rs, re, ids, vals, order, wt, Y, WA, b, ab, seg, t_start,
               sgd, n_parts, pbuf, T, flags):
    """One pass of averaged squared-loss SGD over the rows ``order`` of a CSR
    chunk, every head (class) at every row. It follows sklearn's _plain_sgd
    with squared_error loss, l2 penalty, constant eta, average=1 and the
    sparse intercept decay (``sgd``: _sgd_params). A non-empty ``wt`` scales
    row r's update by wt[r] (sample_weight); an empty one leaves the
    arithmetic unweighted.

    WA is class-minor (F, K', 2): weights W in [..., 0], the averaging
    accumulator AW in [..., 1] (averaged weights = (AW + a W) / b), so a
    nonzero touches one contiguous run for all heads. wscale and the averaging
    scalars depend only on the step, so every head shares seg = (t0, e0) (e0
    = 1: the segment began with wscale c; 0: with a reset). A row's dot
    products are summed over npt = min(n_parts, max(1, nz // 512)) contiguous
    parts of its entries, each ascending, the parts added in order: npt is set
    by the row's length, never by the thread count, so the floats do not
    depend on T.

    One parallel region: prange(T) runs one iteration per thread of a team
    (kernels.team); the threads walk the rows together and meet at spin
    barriers. Per row:
      A. the npt * K' cells (part g, head k), g-major, are cut into T runs;
         a thread sums part g of its heads. Barrier.
      B. every thread computes every head's step (the same floats), and a
         wscale reset, cut by features. Barrier.
      C. if any head moves, the row's entries are cut into T runs and one
         thread updates every head of an entry. Barrier.
    flags: T * 8 int64, zeroed by the caller; pbuf: (n_parts, >= K') float64.
    """
    F = WA.shape[0]
    K = WA.shape[1]
    n = order.shape[0]
    weighted = wt.shape[0] > 0
    eta = sgd[0]
    decay = sgd[1]
    maxdl = sgd[2]
    reset = sgd[3]
    lnc = sgd[4]
    one_m_c = -np.expm1(lnc)
    for tid in prange(T):
        bl = b.copy()
        abl = ab.copy()
        c1 = np.empty(K)
        c2 = np.empty(K)
        acc = np.empty(K)
        t0 = seg[0]
        e0 = seg[1]
        gen = np.int64(0)
        f0 = F * tid // T
        f1 = F * (tid + 1) // T
        for ii in range(n):
            r = order[ii]
            t = t_start + ii
            lo = rs[r]
            nz = re[r] - lo
            npt = min(n_parts, max(1, nz // 512))
            ncell = npt * K
            c = ncell * tid // T                 # A: part g of heads ka..kz
            cz = ncell * (tid + 1) // T
            while c < cz:
                g = c // K
                ka = c - g * K
                kz = min(K, cz - g * K)
                a = lo + nz * g // npt
                z = lo + nz * (g + 1) // npt
                for k in range(ka, kz):
                    acc[k] = 0.0
                for e in range(a, z):
                    j = ids[e]
                    v = vals[e]
                    for k in range(ka, kz):
                        acc[k] += WA[j, k, 0] * v
                for k in range(ka, kz):
                    pbuf[g, k] = acc[k]
                c = g * K + kz
            gen += 1
            _barrier(flags, tid, T, gen)
            ws_prev = np.exp(float(e0 + t - 1 - t0) * lnc)
            ws = np.exp(float(e0 + t - t0) * lnc)
            if ws < reset:
                # sklearn's reset_wscale, with the averaging scalars as of
                # the end of step t - 1
                a_p, inv_b = _avg_scalars(t0, e0, t - 1, lnc, one_m_c)
                for j in range(f0, f1):
                    for k in range(K):
                        wj = WA[j, k, 0]
                        WA[j, k, 1] = (WA[j, k, 1] + a_p * wj) * inv_b
                        WA[j, k, 0] = wj * ws
                t0 = t
                e0 = 0
                ws = 1.0
                gen += 1
                _barrier(flags, tid, T, gen)
            a_prev, _ = _avg_scalars(t0, e0, t - 1, lnc, one_m_c)
            any_upd = False
            for k in range(K):                   # B: every head's step
                sk = 0.0
                for g in range(npt):
                    sk += pbuf[g, k]
                # the dot used the scale before this step's pre-reset scale
                p = sk * ws_prev + bl[k]
                dl = p - Y[r, k]
                if dl > maxdl:
                    dl = maxdl
                elif dl < -maxdl:
                    dl = -maxdl
                upd = -eta * dl
                if weighted:
                    upd = upd * wt[r]
                c1[k] = upd / ws
                c2[k] = a_prev * (-upd / ws)
                if upd != 0.0:
                    any_upd = True
                bl[k] += upd * decay
                abl[k] += (bl[k] - abl[k]) / float(t)
            if any_upd:                          # C: entries, every head
                for e in range(lo + nz * tid // T, lo + nz * (tid + 1) // T):
                    j = ids[e]
                    v = vals[e]
                    for k in range(K):
                        WA[j, k, 0] = WA[j, k, 0] + v * c1[k]
                        WA[j, k, 1] = WA[j, k, 1] + v * c2[k]
            gen += 1
            _barrier(flags, tid, T, gen)
        if tid == 0:
            for k in range(K):
                b[k] = bl[k]
                ab[k] = abl[k]
            seg[0] = t0
            seg[1] = e0


@njit(inline="always")
def _avg_scalars(t0, e0, t_last, lnc, one_m_c):
    """(a, 1 / b) of sklearn's final reset_wscale after step t_last for the
    segment (t0, e0): the averaged weights are (AW + a W) / b."""
    t_ref = float(t0 - 1) if t0 > 1 else 1.0
    if t_last >= t0:
        a = (np.exp(float(e0) * lnc) / t_ref
             * (-np.expm1(float(t_last - t0 + 1) * lnc)) / one_m_c)
        bb = float(t_last) / t_ref
    else:
        a = 0.0
        bb = 1.0
    return a, 1.0 / bb


@njit(cache=True, parallel=True, nogil=True)
def _asgd_finish(WA, seg, t_last, sgd, out):
    """The averaged weights after step t_last into out (F, K')."""
    lnc = sgd[4]
    one_m_c = -np.expm1(lnc)
    a, inv_b = _avg_scalars(seg[0], seg[1], t_last, lnc, one_m_c)
    for j in prange(WA.shape[0]):
        for k in range(WA.shape[1]):
            out[j, k] = (WA[j, k, 1] + a * WA[j, k, 0]) * inv_b
