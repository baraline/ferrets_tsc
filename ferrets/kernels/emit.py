"""Numerosity reduction, the 256-key histogram and block emission."""

import numpy as np
from numba import njit
from numba.cpython.unsafe.numbers import trailing_zeros

from ..constants import _MODE_GLOBAL, _MODE_SCORE, KEYSPACE, MASK_WORDS


@njit(inline="always")
def _emit(kj_arr, n_windows, hist, bm, mode, nr, i, n_out, col0, v_tab,
          v_woff, head_col, hc_none, tf_tab, idf, W, out_ids, out_cnt,
          blk_acc, n_heads):
    """Emit one block of keys (kj_arr[:n_windows], each in 0..255).

    Counts the keys in hist (consecutive repeats count once when nr != 0),
    then walks the distinct keys in ascending order, clearing hist and bm
    (the 256-bit occupancy mask) as it goes, so both are zero on return.
    Mode 0 appends (col0 + key, count) at out_ids/out_cnt[i, n_out:]. Mode 1
    looks the key up in head_col (offset v_tab) and skips hc_none; otherwise,
    with g = v_woff + hc and wt = tf_tab[count] * idf[g], it adds wt ** 2 to
    the sum of squares and wt * W[g] to blk_acc (zeroed first). Returns
    (n_out, sum of squares); the sum is 0 in mode 0."""
    IT = out_ids.dtype.type
    CT = out_cnt.dtype.type
    prev = np.int64(-1)             # keys are >= 0: never equal
    for j in range(n_windows):
        kj = kj_arr[j]
        if kj != prev:
            if hist[kj] == 0:
                bm[kj >> 6] |= np.uint64(1) << np.uint64(kj & 63)
            hist[kj] += 1
        if nr != 0:
            prev = kj
    bsq = 0.0
    if mode == _MODE_SCORE:
        for h in range(n_heads):
            blk_acc[h] = 0.0
    for wd in range(MASK_WORDS):
        bits = bm[wd]
        while bits != np.uint64(0):
            kk = wd * 64 + np.int64(trailing_zeros(bits))
            bits &= bits - np.uint64(1)
            cnt = hist[kk]
            hist[kk] = 0
            if mode == _MODE_GLOBAL:
                out_ids[i, n_out] = IT(col0 + kk)
                out_cnt[i, n_out] = CT(cnt)
                n_out += 1
            else:
                hc = np.int64(head_col[v_tab + kk])
                if hc != hc_none:
                    g = v_woff + hc
                    wt = tf_tab[cnt] * idf[g]
                    bsq += wt * wt
                    for h in range(n_heads):
                        blk_acc[h] += wt * W[g, h]
        bm[wd] = np.uint64(0)
    return n_out, bsq


@njit(inline="always")
def _emit_pair(key_j, kl, own, n_windows, hist, bm, mode, nr, i, n_out, col0,
               tab0, woff, head_col, hc_none, tf_tab, idf, W, out_ids, out_cnt,
               blk_acc, n_heads, acc, sq):
    """Row i's blocks of one (view, dilation) pair through _emit: the word
    block of key_j at column col0 (head_col offset tab0), then, when ``own``,
    the level block of kl one key block later. In mode 1 each block's squared
    norm and head sums are added to sq[i] and acc[i]. Returns n_out."""
    n_out, bsq = _emit(key_j, n_windows, hist, bm, mode, nr, i, n_out, col0,
                       tab0, woff, head_col, hc_none, tf_tab, idf, W, out_ids,
                       out_cnt, blk_acc, n_heads)
    if mode == _MODE_SCORE:
        sq[i] += bsq
        for h in range(n_heads):
            acc[i, h] += blk_acc[h]
    if own:
        n_out, bsq = _emit(kl, n_windows, hist, bm, mode, nr, i, n_out,
                           col0 + KEYSPACE, tab0 + KEYSPACE, woff, head_col,
                           hc_none, tf_tab, idf, W, out_ids, out_cnt, blk_acc,
                           n_heads)
        if mode == _MODE_SCORE:
            sq[i] += bsq
            for h in range(n_heads):
                acc[i, h] += blk_acc[h]
    return n_out
