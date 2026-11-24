"""The keying kernel for C >= 2: channel-wise words and the cross-channel
words, with the level word."""

import numpy as np
from numba import njit, prange

from ..constants import (_KIND_PAA, _MODE_SCORE, EPS_STD, KEYSPACE, MASK_WORDS,
                         N_COEF, WORD_LEN)
from ..kernels.emit import _emit
from ..kernels.level import _level_part
from ..kernels.window import _gather, _moments


@njit(inline="always")
def _paa_part(c, L, seg, nb, n_w, std, pool, part):
    """nb-bit PAA word of every window into part[:n_w]: a bit is 1 where the
    segment sum (seg samples) exceeds the window's segment total over nb,
    first segment most significant; flat windows give 0. The total is the nb
    segment sums added left to right and nb is a power of two, so the
    threshold is exact and a function of the window's own values only (the
    same word as words_univariate._paa_keys at nb = 8). pool is scratch."""
    n_pool = L - seg + 1
    for t in range(n_pool):
        pool[t] = c[t]
    for k in range(1, seg):
        for t in range(n_pool):
            pool[t] += c[t + k]
    for t in range(n_w):
        tot = pool[t]
        for s in range(1, nb):
            tot += pool[t + s * seg]
        th = tot / float(nb)
        k = np.int64(0)
        for s in range(nb):
            k = k * 2 + np.int64(pool[t + s * seg] > th)
        if std[t] <= EPS_STD:
            k = 0
        part[t] = k


@njit(inline="always")
def _fourier_part(c, w, n_w, std, ckx, skx, cuts, off, nb, inv_scale, mr, mi,
                  part):
    """nb-bit Fourier word of every window into part[:n_w]: the first nb
    normalised parts (re_1, im_1, re_2, ...), most significant first,
    against cuts[off:off + nb], with the prefix-sum and rotation arithmetic
    of _fourier_keys. A flat window takes the bits of the parts' flat values
    (a bit is set where its cut is below 0). ckx, skx are the trig tables
    of the window length; inv_scale, mr and mi are scratch."""
    n_coef = (nb + 1) // 2
    for t in range(n_w):
        inv_scale[t] = 1.0 / (w * std[t])
    L = n_w + w - 1
    for k in range(n_coef):
        a0 = 0.0
        b0 = 0.0
        mr[k, 0] = 0.0
        mi[k, 0] = 0.0
        for t in range(L):
            ct = c[t]
            a0 += ct * ckx[k, t]
            b0 -= ct * skx[k, t]
            mr[k, t + 1] = a0
            mi[k, t + 1] = b0
    flat = np.int64(0)
    for q in range(nb):
        flat = flat * 2 + np.int64(cuts[off + q] < 0)
    for t in range(n_w):
        if std[t] <= EPS_STD:
            part[t] = flat
            continue
        isc = inv_scale[t]
        kk = np.int64(0)
        for q in range(nb):
            k = q // 2
            a = mr[k, t + w] - mr[k, t]
            b = mi[k, t + w] - mi[k, t]
            cr = ckx[k, t]
            sr = skx[k, t]
            if q % 2 == 0:
                val = (a * cr - b * sr) * isc
            else:
                val = (a * sr + b * cr) * isc
            kk = kk * 2 + np.int64(val > cuts[off + q])
        part[t] = kk


@njit(cache=True, parallel=True, nogil=True, error_model="numpy")
def _key_families(X, rows, n_blocks, v_kind, v_win, v_cptr, v_chs, v_diff,
                  v_dptr, dils, v_cuts, d_col, d_tab, own_level, l_row,
                  l_cuts, v_woff, v_trig, cos_tab, sin_tab, w_ptr, w_ref,
                  n_wslots, cross, v0, v1, mode, nr, row_tile, head_col,
                  hc_none, tf_tab, idf, W, out_ids, out_cnt, out_nnz, acc,
                  sq):
    """Family views v0..v1-1 over the rows ``rows`` of X (n, C, m), parallel
    over row blocks. Modes, row tiles and the order guarantee are those of
    _key_univariate.

    Per (row, view, dilation) with k channel slots and b = 8 // k bits per
    channel: every channel's 8-bit word per window start (kw[slot, j]) and,
    with the coarse word (bit 0 of cross), its b-bit part (kc[slot, j]). The
    blocks [k channel words, coarse, change flags, agreement flags (bit 1 of
    cross)] go through _emit at column d_col[di] + block * 256 (mode 0) or
    head_col offset d_tab[di] + block * 256 (mode 1). With own_level[di] set,
    the level block follows: slot s's b-bit part against cut set
    l_cuts[l_row[v_cptr[v] + s]], slot 0 most significant. v_cuts[v] holds 8
    cuts per channel slot.

    Word table (n_wslots 0: off): w_ref[w_ptr[di] + s] numbers slot s's
    channel word when two or more (family, dilation) pairs of a side key it
    identically (-1: shared with nothing, or a number >= n_wslots, the
    table's capacity). wtab[q, word, start] and stamp[q, word] (the row that
    filled it) hold the words keyed for the q-th row of a tile; a pair copies
    a word its row already keyed instead of keying it. The gather and the
    moments still run for the level word and the PAA coarse part; the
    Fourier coarse part is kw >> (8 - b). kw, kc, kl, part are uint8 (k * b
    <= 8), kj is int64, the type _emit reads."""
    n_rows = rows.shape[0]
    m = X.shape[2]
    n_heads = W.shape[1]
    tile = max(row_tile, np.int64(1))
    for blk_i in prange(n_blocks):
        lo = blk_i * n_rows // n_blocks
        hi = (blk_i + 1) * n_rows // n_blocks
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
        part = np.empty(m, np.uint8)
        kw = np.empty((8, m), np.uint8)
        kc = np.empty((8, m), np.uint8)
        kj = np.empty(m, np.int64)
        kl = np.empty(m, np.uint8)
        n_tab = max(n_wslots, np.int64(1))
        n_q = max(min(tile, hi - lo), np.int64(1))
        wtab = np.empty((n_q, n_tab, m), np.uint8)
        stamp = np.full((n_q, n_tab), -1, np.int64)
        wslot = np.empty(8, np.int64)
        have = np.zeros(8, np.bool_)
        hist = np.zeros(KEYSPACE, np.int64)
        bm = np.zeros(MASK_WORDS, np.uint64)
        blk_acc = np.zeros(n_heads)
        for i0 in range(lo, hi, tile):
            i1 = min(i0 + tile, hi)
            for v in range(v0, v1):
                w = v_win[v]
                is_diff = v_diff[v] != 0
                kind = v_kind[v]
                trig = v_trig[v]
                c0 = v_cptr[v]
                k = v_cptr[v + 1] - c0
                has_c = (cross & 1) != 0 and k > 1
                has_f = (cross & 2) != 0 and k > 1
                paa_coarse = has_c and kind == _KIND_PAA
                ncross = np.int64(has_c) + 2 * np.int64(has_f)
                nblk = k + ncross
                b = WORD_LEN // k
                nb2 = b if b >= 2 else 2
                hm = (b + 1) // 2           # the level part's mean bits
                hs = b // 2                 # ... and std bits
                for di in range(v_dptr[v], v_dptr[v + 1]):
                    d = dils[di]
                    n_windows = m - (w - 1) * d
                    if n_windows <= 0:
                        continue
                    nlev = own_level[di]
                    for si in range(k):
                        s = np.int64(-1)
                        if n_wslots > 0:
                            s = w_ref[w_ptr[di] + si]
                        if s >= n_wslots:
                            s = np.int64(-1)
                        wslot[si] = s
                    for i in range(i0, i1):
                        q = i - i0
                        r_x = rows[i]
                        n_out = out_nnz[i]
                        for si in range(k):
                            s = wslot[si]
                            got = False
                            if s >= 0:
                                got = stamp[q, s] == i
                            have[si] = got
                            if got:
                                for j in range(n_windows):
                                    kw[si, j] = wtab[q, s, j]
                        for r0 in range(d):
                            L = (m - r0 + d - 1) // d
                            if L < w:
                                continue
                            n_w = L - w + 1
                            for si in range(k):
                                if have[si] and nlev == 0 and not paa_coarse:
                                    continue
                                _gather(X, r_x, v_chs[c0 + si], is_diff, r0,
                                        d, L, c)
                                _moments(c, L, w, S, S2, mean, std, flag)
                                if nlev != 0:
                                    lr = l_row[c0 + si]
                                    for t in range(n_w):
                                        pt = _level_part(l_cuts, lr, mean[t],
                                                         std[t], hm, hs)
                                        jj = r0 + t * d
                                        if si == 0:
                                            kl[jj] = pt
                                        else:
                                            lv = np.int64(kl[jj]) << b
                                            kl[jj] = np.uint8(lv | pt)
                                if not have[si]:
                                    if kind == _KIND_PAA:
                                        _paa_part(c, L, w // WORD_LEN,
                                                  WORD_LEN, n_w, std, pool,
                                                  part)
                                    else:
                                        _fourier_part(c, w, n_w, std,
                                                      cos_tab[trig],
                                                      sin_tab[trig], v_cuts[v],
                                                      si * WORD_LEN, WORD_LEN,
                                                      inv_scale, mr, mi, part)
                                    for t in range(n_w):
                                        kw[si, r0 + t * d] = part[t]
                                if paa_coarse:
                                    _paa_part(c, L, w // nb2, nb2, n_w,
                                              std, pool, part)
                                    for t in range(n_w):
                                        kc[si, r0 + t * d] = np.uint8(
                                            np.int64(part[t]) >> (nb2 - b))
                        for si in range(k):
                            s = wslot[si]
                            if s >= 0 and not have[si]:
                                for j in range(n_windows):
                                    wtab[q, s, j] = kw[si, j]
                                stamp[q, s] = i
                        if has_c and kind != _KIND_PAA:
                            for si in range(k):
                                for j in range(n_windows):
                                    kc[si, j] = np.uint8(
                                        np.int64(kw[si, j]) >> (WORD_LEN - b))
                        bsq_all = 0.0
                        for bi in range(nblk + nlev):
                            if bi == nblk:                # the level word
                                for j in range(n_windows):
                                    kj[j] = np.int64(kl[j])
                            elif bi < k:
                                for j in range(n_windows):
                                    kj[j] = np.int64(kw[bi, j])
                            elif has_c and bi == k:      # coarsened words
                                for j in range(n_windows):
                                    key = np.int64(0)
                                    for si in range(k):
                                        key = (key << b) + np.int64(kc[si, j])
                                    kj[j] = key
                            elif bi == k + np.int64(has_c):   # change flags
                                for j in range(n_windows):
                                    key = np.int64(0)
                                    for si in range(k):
                                        ch = np.int64(0)
                                        if j >= d:
                                            ch = np.int64(kw[si, j]
                                                          != kw[si, j - d])
                                        key = key * 2 + ch
                                    kj[j] = key
                            else:                         # agreement flags
                                for j in range(n_windows):
                                    key = np.int64(0)
                                    for si in range(1, k):
                                        key = key * 2 + np.int64(
                                            kw[si, j] == kw[0, j])
                                    kj[j] = key
                            col0 = d_col[di] + bi * KEYSPACE
                            n_out, bsq = _emit(
                                kj, n_windows, hist, bm, mode, nr, i, n_out,
                                col0, d_tab[di] + bi * KEYSPACE, v_woff[v],
                                head_col, hc_none, tf_tab, idf, W, out_ids,
                                out_cnt, blk_acc, n_heads)
                            if mode == _MODE_SCORE:
                                sq[i] += bsq
                                for h in range(n_heads):
                                    acc[i, h] += blk_acc[h]
                        out_nnz[i] = n_out
