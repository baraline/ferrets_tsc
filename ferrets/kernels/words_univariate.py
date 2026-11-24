"""The univariate keying kernel (C = 1 and the per-channel layout): PAA and
Fourier words in one pass."""

import numpy as np
from numba import njit, prange

from ..constants import (_KIND_PAA, EPS_STD, KEYSPACE, LEVEL_BITS, MASK_WORDS,
                         N_COEF, WORD_LEN)
from ..kernels.emit import _emit_pair
from ..kernels.level import _level_part
from ..kernels.window import _gather, _moments


@njit(inline="always")
def _paa_keys(c, L, paa, n_w, std, pool, key):
    """The 8-bit PAA word of every window into key[:n_w]: a bit is 1 where
    the segment sum (paa samples) exceeds an eighth of the window's segment
    total, first segment most significant; flat windows key 0. The threshold
    is the sum of the window's eight segment sums, added left to right, times
    0.125 (a power of two: exact), so a bit is a function of the window's own
    values and identical windows get one key wherever they sit in a chain.
    Unrolled: the eight bits of a window come from one pass. pool is
    scratch."""
    n_pool = L - paa + 1
    for t in range(n_pool):
        pool[t] = c[t]
    for k in range(1, paa):
        for t in range(n_pool):
            pool[t] += c[t + k]
    p1 = paa
    p2 = 2 * paa
    p3 = 3 * paa
    p4 = 4 * paa
    p5 = 5 * paa
    p6 = 6 * paa
    p7 = 7 * paa
    for t in range(n_w):
        tot = pool[t]
        tot += pool[t + p1]
        tot += pool[t + p2]
        tot += pool[t + p3]
        tot += pool[t + p4]
        tot += pool[t + p5]
        tot += pool[t + p6]
        tot += pool[t + p7]
        th = tot * 0.125
        k = (np.int64(pool[t] > th) * 128 + np.int64(pool[t + p1] > th) * 64
             + np.int64(pool[t + p2] > th) * 32
             + np.int64(pool[t + p3] > th) * 16
             + np.int64(pool[t + p4] > th) * 8
             + np.int64(pool[t + p5] > th) * 4
             + np.int64(pool[t + p6] > th) * 2
             + np.int64(pool[t + p7] > th))
        if std[t] <= EPS_STD:
            k = 0
        key[t] = k


@njit(inline="always")
def _fourier_keys(c, w, n_w, std, ckx, skx, cuts, inv_scale, mr, mi, key,
                  flat_key):
    """The 8-bit Fourier word of every window into key[:n_w]; flat windows
    take flat_key. The modulated prefix sums (mr, mi) run over the whole
    chain, two coefficients per pass; then one pass per window rotates each
    coefficient back by the window start's table entry (ckx, skx), scales it
    by 1 / (w std) and compares its real and imaginary parts with the 8 cuts
    (re_1, im_1, ..., re_4, im_4, most significant first). The key is built
    in a register; inv_scale is scratch."""
    for t in range(n_w):
        inv_scale[t] = 1.0 / (w * std[t])
    L = n_w + w - 1
    k = 0
    while k < 4:
        a0 = 0.0
        b0 = 0.0
        a1 = 0.0
        b1 = 0.0
        mr[k, 0] = 0.0
        mi[k, 0] = 0.0
        mr[k + 1, 0] = 0.0
        mi[k + 1, 0] = 0.0
        for t in range(L):
            ct = c[t]
            a0 += ct * ckx[k, t]
            b0 -= ct * skx[k, t]
            a1 += ct * ckx[k + 1, t]
            b1 -= ct * skx[k + 1, t]
            mr[k, t + 1] = a0
            mi[k, t + 1] = b0
            mr[k + 1, t + 1] = a1
            mi[k + 1, t + 1] = b1
        k += 2
    c0r = cuts[0]
    c0i = cuts[1]
    c1r = cuts[2]
    c1i = cuts[3]
    c2r = cuts[4]
    c2i = cuts[5]
    c3r = cuts[6]
    c3i = cuts[7]
    for t in range(n_w):
        isc = inv_scale[t]
        kk = np.int64(0)
        a = mr[0, t + w] - mr[0, t]
        b = mi[0, t + w] - mi[0, t]
        cr = ckx[0, t]
        sr = skx[0, t]
        kk = kk * 4 + np.int64((a * cr - b * sr) * isc > c0r) * 2 \
            + np.int64((a * sr + b * cr) * isc > c0i)
        a = mr[1, t + w] - mr[1, t]
        b = mi[1, t + w] - mi[1, t]
        cr = ckx[1, t]
        sr = skx[1, t]
        kk = kk * 4 + np.int64((a * cr - b * sr) * isc > c1r) * 2 \
            + np.int64((a * sr + b * cr) * isc > c1i)
        a = mr[2, t + w] - mr[2, t]
        b = mi[2, t + w] - mi[2, t]
        cr = ckx[2, t]
        sr = skx[2, t]
        kk = kk * 4 + np.int64((a * cr - b * sr) * isc > c2r) * 2 \
            + np.int64((a * sr + b * cr) * isc > c2i)
        a = mr[3, t + w] - mr[3, t]
        b = mi[3, t + w] - mi[3, t]
        cr = ckx[3, t]
        sr = skx[3, t]
        kk = kk * 4 + np.int64((a * cr - b * sr) * isc > c3r) * 2 \
            + np.int64((a * sr + b * cr) * isc > c3i)
        key[t] = kk
    for t in range(n_w):
        if std[t] <= EPS_STD:
            key[t] = flat_key


@njit(inline="always")
def _key_chain(kind, c, L, w, n_w, std, pool, inv_scale, mr, mi, key,
               ckx, skx, cuts, flat_key):
    """The word of every window of one residue chain: the PAA word when kind
    is _KIND_PAA, else the Fourier word, into key[:n_w]."""
    if kind == _KIND_PAA:
        _paa_keys(c, L, w // WORD_LEN, n_w, std, pool, key)
    else:
        _fourier_keys(c, w, n_w, std, ckx, skx, cuts, inv_scale, mr, mi, key,
                      flat_key)


@njit(inline="always")
def _key_pair(X, r_x, ch, is_diff, kind, w, d, own, lr, cuts, flat_key, cos_t,
              sin_t, l_cuts, c, S, S2, mean, std, flag, pool, inv_scale, mr,
              mi, key, key_j, kl):
    """Row r_x of X, channel ch, one (view, dilation) pair of window w and
    dilation d: every residue chain gathered and keyed (the word of ``kind``)
    into key_j in window-start order and, when ``own``, its level part (cut
    set lr of l_cuts) into kl. The rest are scratch."""
    m = X.shape[2]
    for r0 in range(d):
        L = (m - r0 + d - 1) // d
        if L < w:
            continue
        _gather(X, r_x, ch, is_diff, r0, d, L, c)
        n_w = _moments(c, L, w, S, S2, mean, std, flag)
        if own:
            for t in range(n_w):
                kl[r0 + t * d] = _level_part(l_cuts, lr, mean[t], std[t],
                                             LEVEL_BITS, LEVEL_BITS)
        _key_chain(kind, c, L, w, n_w, std, pool, inv_scale, mr, mi, key,
                   cos_t, sin_t, cuts, flat_key)
        for t in range(n_w):
            key_j[r0 + t * d] = key[t]


@njit(cache=True, parallel=True, nogil=True, error_model="numpy")
def _key_univariate(X, rows, n_blocks, v_kind, v_win, v_ch, v_diff, v_dptr,
                    dils, v_cuts, v_flat, d_col, d_tab, own_level, l_row,
                    l_cuts, v_woff, v_trig, cos_tab, sin_tab, v0, v1, mode, nr,
                    row_tile, head_col, hc_none, tf_tab, idf, W, out_ids,
                    out_cnt, out_nnz, acc, sq):
    """Views v0..v1-1 over the rows ``rows`` of X (n, C, m), parallel over row
    blocks (scratch is allocated once per block).

    Per (row, view, dilation): every residue chain is keyed, the keys laid
    out in window-start order, numerosity-reduced when nr != 0 (a window
    whose key equals the previous start's is dropped), counted and read out
    in ascending key order.
    - Mode 0 (_MODE_GLOBAL): (local column, count) pairs of the view's
      pseudo-channel go to out_ids / out_cnt row i from out_nnz[i]; column
      ids start at d_col[di].
    - Mode 1 (_MODE_SCORE): the row is scored directly. head_col[d_tab[di] +
      key] is the key's column (hc_none: dropped); with w = tf_tab[count]
      idf[g], g = v_woff[v] + column: acc[i] += w W[g], sq[i] += w^2.
    With own_level[di] set, the level block (cut set l_cuts[l_row[v]])
    follows the word block at column d_col[di] + 256.

    Row tiles (row_tile rows, < 1 acts as 1; the loop runs views, slots, then
    the rows of a tile) only reorder work across rows: a row still receives
    its blocks in view, slot, word-then-level order, so every float addition
    keeps its operands and its order and rows never interact. The same holds
    in _key_families. ablation.kernels._key_univariate_bits copies this loop:
    keep them in step."""
    n_rows = rows.shape[0]
    m = X.shape[2]
    n_heads = W.shape[1]
    tile = max(row_tile, np.int64(1))
    for b in prange(n_blocks):
        lo = b * n_rows // n_blocks
        hi = (b + 1) * n_rows // n_blocks
        c = np.empty(m)
        S = np.empty(m + 1)
        S2 = np.empty(m + 1)
        mean = np.empty(m)
        std = np.empty(m)
        pool = np.empty(m)
        inv_scale = np.empty(m)
        mr = np.empty((N_COEF, m + 1))
        mi = np.empty((N_COEF, m + 1))
        flag = np.empty(m, np.int64)
        key = np.empty(m, np.int64)
        key_j = np.empty(m, np.int64)
        kl = np.empty(m, np.int64)
        hist = np.zeros(KEYSPACE, np.int64)
        bm = np.zeros(MASK_WORDS, np.uint64)
        blk_acc = np.zeros(n_heads)
        for i0 in range(lo, hi, tile):
            i1 = min(i0 + tile, hi)
            for v in range(v0, v1):
                w = v_win[v]
                ch = v_ch[v]
                is_diff = v_diff[v] != 0
                trig = v_trig[v]
                lr = l_row[v]
                for di in range(v_dptr[v], v_dptr[v + 1]):
                    d = dils[di]
                    n_windows = m - (w - 1) * d
                    if n_windows <= 0:
                        continue
                    own = own_level[di] != 0
                    for i in range(i0, i1):
                        _key_pair(X, rows[i], ch, is_diff, v_kind[v], w, d,
                                  own, lr, v_cuts[v], v_flat[v], cos_tab[trig],
                                  sin_tab[trig], l_cuts, c, S, S2, mean, std,
                                  flag, pool, inv_scale, mr, mi, key, key_j,
                                  kl)
                        out_nnz[i] = _emit_pair(
                            key_j, kl, own, n_windows, hist, bm, mode, nr, i,
                            out_nnz[i], d_col[di], d_tab[di], v_woff[v],
                            head_col, hc_none, tf_tab, idf, W, out_ids,
                            out_cnt, blk_acc, n_heads, acc, sq)
