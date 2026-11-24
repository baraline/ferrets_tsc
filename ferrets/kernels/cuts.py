"""Cut-set samplers: the Fourier word's medians and the level word's
quantiles of sampled windows, both by in-place selection."""

import numpy as np
from numba import njit, prange

from ..constants import EPS_STD, LEVEL_CUTS


@njit(inline="always")
def _select(a, lo, hi, kpos):
    """Hoare selection of order statistic kpos in a[lo:hi+1] (in place)."""
    while hi > lo:
        mid = (lo + hi) >> 1
        x = a[lo]
        y = a[mid]
        z = a[hi]
        if x > y:
            x, y = y, x
        if y > z:
            y = z
            if x > y:
                y = x
        pivot = y
        i = lo
        j = hi
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                tmp = a[i]
                a[i] = a[j]
                a[j] = tmp
                i += 1
                j -= 1
        if kpos <= j:
            hi = j
        elif kpos >= i:
            lo = i
        else:
            return


@njit(inline="always")
def _min_of(a, first, stop):
    """min(a[first:stop]) for stop > first."""
    low = a[first]
    for t in range(first + 1, stop):
        if a[t] < low:
            low = a[t]
    return low


@njit(inline="always")
def _lerp(lo, hi, gam):
    """numpy's linear interpolation between the order statistics lo <= hi at
    fraction gam, in its t >= 0.5 form."""
    dba = hi - lo
    val = lo + dba * gam
    if gam >= 0.5:
        val = hi - dba * (1.0 - gam)
    return val


@njit(inline="always")
def _median_inplace(a, k):
    """np.quantile(a[:k], 0.5) ('linear'), selecting in place: numpy's index
    plan (previous = floor((k-1)/2), next = previous + 1, both the last index
    at k = 1; gamma = 0, 0.5 or 1) and its lerp. After the selection of the
    lower order statistic everything right of it is >=, so the upper one is
    the minimum of that part."""
    if k == 1:
        pp = 0
        gam = 1.0
    else:
        pp = (k - 1) // 2
        gam = 0.0 if (k - 1) % 2 == 0 else 0.5
    _select(a, 0, k - 1, pp)
    lo = a[pp]
    if k == 1:
        hi_ = lo
    else:
        hi_ = _min_of(a, pp + 1, k)
    return _lerp(lo, hi_, gam)


@njit(inline="always")
def _gather_block(X, ch, is_diff, window, first, nb, d_rows, d_starts,
                  d_dil, buf):
    """The nb sampled windows of draw entries first .. first + nb - 1 into
    buf[t, q] (q the sample): X[row, ch, start + t d], or on a difference
    pseudo-channel the first difference at that point with the last increment
    repeated."""
    m = X.shape[2]
    for q in range(nb):
        di = first + q
        row = d_rows[di]
        start = d_starts[di]
        d = d_dil[di]
        if is_diff:
            for t in range(window):
                jj = start + t * d
                if jj < m - 1:
                    buf[t, q] = X[row, ch, jj + 1] - X[row, ch, jj]
                else:
                    buf[t, q] = X[row, ch, m - 1] - X[row, ch, m - 2]
        else:
            for t in range(window):
                buf[t, q] = X[row, ch, start + t * d]


@njit(inline="always")
def _window_mean_std(buf, window, nb, mean, std):
    """mean[q] and the population standard deviation std[q] of the nb windows
    buf[:window, q]: two passes, each sample adding its points t-ascending
    from 0.0 (the loop over q is the inner one, so the block size does not
    change a sample's result)."""
    fw = float(window)
    for q in range(nb):
        mean[q] = 0.0
        std[q] = 0.0
    for t in range(window):
        for q in range(nb):
            mean[q] += buf[t, q]
    for q in range(nb):
        mean[q] = mean[q] / fw
    for t in range(window):
        for q in range(nb):
            dv = buf[t, q] - mean[q]
            std[q] += dv * dv
    for q in range(nb):
        std[q] = np.sqrt(std[q] / fw)


@njit(inline="always")
def _quantile_pos(k, j):
    """numpy's 'linear' index plan for the quantile j / 16 of k >= 2 values:
    the virtual index (k - 1) j / 16 as its floor and fraction. The 16 is
    LEVEL_CUTS + 1, the bin count of the level word."""
    vi = (k - 1) * (j / 16.0)
    lo = np.int64(np.floor(vi))
    return lo, vi - lo


@njit(inline="always")
def _level_targets(k, tg):
    """The order statistics the 15 quantiles read from below: the distinct
    lo of _quantile_pos over j = 1..15 (non-decreasing in j), ascending, into
    tg; their count. Every lo is at most k - 2, so lo + 1 exists."""
    nt = 0
    for j in range(1, LEVEL_CUTS + 1):
        lo = _quantile_pos(k, j)[0]
        if nt == 0 or lo != tg[nt - 1]:
            tg[nt] = lo
            nt += 1
    return nt


@njit(inline="always")
def _multiselect(a, k, tg, nt, stk):
    """Select the order statistics tg[:nt] (sorted, distinct, below k) of
    a[:k] in place by bisecting the target list: the middle target is
    selected over the current range (the Hoare _select), then the targets
    left of it over the part left of it and those right of it over the part
    right of it, with an explicit stack of (lo, hi, first target, last
    target) in stk. Afterwards a[tg[i]] is the tg[i]-th order statistic,
    everything left of it <= and everything right of it >=, and the elements
    between two targets lie between their values."""
    stk[0, 0] = 0
    stk[0, 1] = k - 1
    stk[0, 2] = 0
    stk[0, 3] = nt - 1
    top = 1
    while top > 0:
        top -= 1
        lo = stk[top, 0]
        hi = stk[top, 1]
        t0 = stk[top, 2]
        t1 = stk[top, 3]
        tm = (t0 + t1) >> 1
        p = tg[tm]
        if t0 < tm:
            stk[top, 0] = lo
            stk[top, 1] = p - 1
            stk[top, 2] = t0
            stk[top, 3] = tm - 1
            top += 1
        if tm < t1:
            stk[top, 0] = p + 1
            stk[top, 1] = hi
            stk[top, 2] = tm + 1
            stk[top, 3] = t1
            top += 1
        _select(a, lo, hi, p)     # last: the inlined body reassigns lo, hi


@njit(inline="always")
def _quantiles16_select(a, k, out, off, tg, stk):
    """np.quantile(a[:k], j / 16) for j = 1..15 ('linear', numpy's lerp with
    its t >= 0.5 form) into out[off:off + 15], by selection: the lower
    order statistic a[lo] of each cut is selected in place (_multiselect),
    the upper one is the minimum of a[lo + 1 .. next target] (the next target
    is k - 1 after the last), the (lo + 1)-th order statistic, so the lerp
    sees the values a sorted array would hold. a[:k] ends partially ordered;
    a[k:] is never read; tg is an int64 scratch of 15 or more, stk an int64
    (16, 4) one."""
    if k == 1:
        for j in range(1, LEVEL_CUTS + 1):
            out[off + j - 1] = a[0]
    else:
        nt = _level_targets(k, tg)
        _multiselect(a, k, tg, nt, stk)
        ti = 0
        for j in range(1, LEVEL_CUTS + 1):
            lo, gam = _quantile_pos(k, j)
            while tg[ti] < lo:
                ti += 1
            nxt = k - 1
            if ti + 1 < nt:
                nxt = tg[ti + 1]
            out[off + j - 1] = _lerp(a[lo], _min_of(a, lo + 1, nxt + 1), gam)


@njit(inline="always")
def _flat_std_cuts(out, off):
    """Set every std cut out[off:off + 15] (ascending) at most EPS_STD to
    0.0, so none lies in (0, EPS_STD]: the flat-window rule of
    kernels.level._level_part. They are a prefix and stay below any window
    std above EPS_STD, so a non-flat window has as many cuts below it as
    before. The mean cuts are not touched."""
    for j in range(LEVEL_CUTS):
        if out[off + j] <= EPS_STD:
            out[off + j] = 0.0


@njit(cache=True, parallel=True, nogil=True, error_model="numpy")
def _fourier_cut_jobs(X, grp_ptr, j_ch, j_diff, j_w, j_k, j_draw, j_tab, j_res,
                      d_rows, d_starts, d_dil, c_all, s_all, kmax, res, j_lev,
                      lres):
    """Every job of a batch: the normalised DFT parts (re_1, im_1, ...,
    re_4, im_4) of its sampled windows, then the median of each part into
    res[j_res[j]]. Parallel over groups of jobs; each group reuses one
    scratch set across its jobs.

    Per sample: the window is gathered (_gather_block), mean and std
    (_window_mean_std), direct DFT against the (t, k) tables, flat windows
    (std <= EPS_STD) zero. Samples run in blocks of 64 laid out t-major;
    every sample's sums keep their t-ascending order.

    A job with j_lev[j] >= 0 also carries the level cut job of row j_lev[j]
    of lres: the windows' means and stds computed here are kept, and their
    15 linear quantiles j / 16 go to lres[j_lev[j], :15] and [15:30], as
    _level_cut_jobs computes them."""
    B = 64                  # fixed; not config.LEVEL_SAMPLER_BLOCK
    max_w = 1
    for j in range(j_w.shape[0]):
        if j_w[j] > max_w:
            max_w = j_w[j]
    lmax = 1
    for j in range(j_lev.shape[0]):
        if j_lev[j] >= 0:
            lmax = kmax
    for gi in prange(grp_ptr.shape[0] - 1):
        lanes = np.empty((8, kmax))
        buf = np.empty((max_w, B))
        acc = np.empty((8, B))
        mean = np.empty(B)
        std = np.empty(B)
        mu = np.empty(lmax)
        sd = np.empty(lmax)
        tg = np.empty(LEVEL_CUTS, dtype=np.int64)
        stk = np.empty((16, 4), dtype=np.int64)
        for j in range(grp_ptr[gi], grp_ptr[gi + 1]):
            kj = j_k[j]
            window = j_w[j]
            tb = j_tab[j]
            lv = j_lev[j]
            fw = float(window)
            for s0 in range(0, kj, B):
                nb = min(B, kj - s0)
                _gather_block(X, j_ch[j], j_diff[j] != 0, window,
                              j_draw[j] + s0, nb, d_rows, d_starts, d_dil,
                              buf)
                _window_mean_std(buf, window, nb, mean, std)
                for c in range(8):
                    for q in range(nb):
                        acc[c, q] = 0.0
                for t in range(window):
                    cb = tb + 4 * t
                    k0 = c_all[cb]
                    k1 = c_all[cb + 1]
                    k2 = c_all[cb + 2]
                    k3 = c_all[cb + 3]
                    n0 = s_all[cb]
                    n1 = s_all[cb + 1]
                    n2 = s_all[cb + 2]
                    n3 = s_all[cb + 3]
                    for q in range(nb):
                        v = buf[t, q]
                        acc[0, q] += v * k0
                        acc[1, q] -= v * n0
                        acc[2, q] += v * k1
                        acc[3, q] -= v * n1
                        acc[4, q] += v * k2
                        acc[5, q] -= v * n2
                        acc[6, q] += v * k3
                        acc[7, q] -= v * n3
                for q in range(nb):
                    if lv >= 0:
                        mu[s0 + q] = mean[q]
                        sd[s0 + q] = std[q]
                    if std[q] <= EPS_STD:
                        for c in range(8):
                            lanes[c, s0 + q] = 0.0
                    else:
                        inv = 1.0 / (fw * std[q])
                        for c in range(8):
                            lanes[c, s0 + q] = acc[c, q] * inv
            for c in range(8):
                res[j_res[j], c] = _median_inplace(lanes[c], kj)
            if lv >= 0:
                _quantiles16_select(mu, kj, lres[lv], 0, tg, stk)
                _quantiles16_select(sd, kj, lres[lv], LEVEL_CUTS, tg, stk)
                _flat_std_cuts(lres[lv], LEVEL_CUTS)


@njit(cache=True, parallel=True, nogil=True, error_model="numpy")
def _level_cut_jobs(X, grp_ptr, j_ch, j_diff, j_w, j_k, j_draw, j_res,
                    d_rows, d_starts, d_dil, kmax, blk, res):
    """Every job of a batch: each sampled window's mean and population
    standard deviation, then the 15 linear quantiles j / 16 of each set into
    res[j_res[j], :15] (means) and res[j_res[j], 15:30] (stds). Parallel
    over groups of jobs. The sums are _fourier_cut_jobs' (_gather_block,
    _window_mean_std), for blocks of blk >= 1 samples
    (config.LEVEL_SAMPLER_BLOCK); blk does not change the result. The
    quantiles come by in-place multi-selection of the order-statistic pairs
    they read (_quantiles16_select), not by sorting; the std cuts at most
    EPS_STD then become 0 (_flat_std_cuts)."""
    B = blk
    max_w = 1
    for j in range(j_w.shape[0]):
        if j_w[j] > max_w:
            max_w = j_w[j]
    for gi in prange(grp_ptr.shape[0] - 1):
        mu = np.empty(kmax)
        sd = np.empty(kmax)
        buf = np.empty((max_w, B))
        mean = np.empty(B)
        std = np.empty(B)
        tg = np.empty(LEVEL_CUTS, dtype=np.int64)
        stk = np.empty((16, 4), dtype=np.int64)
        for j in range(grp_ptr[gi], grp_ptr[gi + 1]):
            kj = j_k[j]
            window = j_w[j]
            for s0 in range(0, kj, B):
                nb = min(B, kj - s0)
                _gather_block(X, j_ch[j], j_diff[j] != 0, window,
                              j_draw[j] + s0, nb, d_rows, d_starts, d_dil,
                              buf)
                _window_mean_std(buf, window, nb, mean, std)
                for q in range(nb):
                    mu[s0 + q] = mean[q]
                    sd[s0 + q] = std[q]
            _quantiles16_select(mu, kj, res[j_res[j]], 0, tg, stk)
            _quantiles16_select(sd, kj, res[j_res[j]], LEVEL_CUTS, tg, stk)
            _flat_std_cuts(res[j_res[j]], LEVEL_CUTS)
