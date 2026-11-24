"""The univariate counting kernel with a word length argument."""

import numpy as np
from numba import njit, prange

from ..constants import (_KIND_PAA, KEYSPACE, LEVEL_BITS, MASK_WORDS,
                         N_COEF)
from ..kernels.emit import _emit_pair
from ..kernels.level import _level_part
from ..kernels.window import _gather, _moments
from ..kernels.words_family import _fourier_part, _paa_part


@njit(cache=True, parallel=True, nogil=True, error_model="numpy")
def _key_univariate_bits(X, rows, n_blocks, v_kind, v_win, v_ch, v_diff,
                         v_dptr, dils, v_cuts, v_flat, d_col, d_tab,
                         own_level, l_row, l_cuts, v_woff, v_trig, cos_tab,
                         sin_tab, v0, v1, mode, nr, row_tile, head_col,
                         hc_none, tf_tab, idf, W, out_ids, out_cnt, out_nnz,
                         acc, sq, nbits):
    """kernels.words_univariate._key_univariate with the word length nbits as
    an argument, statement for statement except the keying call.

    Each window is keyed into an nbits-bit word by kernels.words_family's
    _paa_part (nbits segments of w // nbits values) or _fourier_part (the first
    nbits Fourier parts against v_cuts[v][:nbits]; flat windows take the
    parts' flat bits). v_flat is unused. The PAA threshold is the window's
    segment total over the nbits segments. Keys stay below 2^nbits <= 256,
    so they fit the 256-column word block, and only occurring columns are
    kept: layout and head are the model's. The level block of an owned slot
    and the row tiling are _key_univariate's, whatever nbits. At nbits = 8 it
    keys exactly as _key_univariate (pinned by a test)."""
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
                kind = v_kind[v]
                trig = v_trig[v]
                lr = l_row[v]
                for di in range(v_dptr[v], v_dptr[v + 1]):
                    d = dils[di]
                    n_windows = m - (w - 1) * d
                    if n_windows <= 0:
                        continue
                    own = own_level[di] != 0
                    for i in range(i0, i1):
                        r_x = rows[i]
                        n_out = out_nnz[i]
                        for r0 in range(d):
                            L = (m - r0 + d - 1) // d
                            if L < w:
                                continue
                            _gather(X, r_x, ch, is_diff, r0, d, L, c)
                            n_w = _moments(c, L, w, S, S2, mean, std, flag)
                            if own:
                                for t in range(n_w):
                                    kl[r0 + t * d] = _level_part(
                                        l_cuts, lr, mean[t], std[t],
                                        LEVEL_BITS, LEVEL_BITS)
                            if kind == _KIND_PAA:
                                _paa_part(c, L, w // nbits, nbits, n_w, std,
                                          pool, key)
                            else:
                                _fourier_part(c, w, n_w, std, cos_tab[trig],
                                              sin_tab[trig], v_cuts[v], 0,
                                              nbits, inv_scale, mr, mi, key)
                            for t in range(n_w):
                                key_j[r0 + t * d] = key[t]
                        out_nnz[i] = _emit_pair(
                            key_j, kl, own, n_windows, hist, bm, mode, nr, i,
                            n_out, d_col[di], d_tab[di], v_woff[v], head_col,
                            hc_none, tf_tab, idf, W, out_ids, out_cnt,
                            blk_acc, n_heads, acc, sq)
