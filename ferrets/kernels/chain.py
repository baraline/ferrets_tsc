"""The univariate keying kernel, chain-major: the (view, dilation) pairs that
read the same residue chains are keyed chain by chain, their blocks stored,
then handed to every row in plan order."""


import numpy as np
from numba import njit, prange

from ..constants import (_MODE_GLOBAL, _MODE_SCORE, KEYSPACE, LEVEL_BITS,
                         MASK_WORDS, N_COEF)
from ..kernels.emit import _emit, _emit_pair
from ..kernels.level import _level_part
from ..kernels.window import _chain_part, _gather, _window_part
from ..kernels.words_univariate import _key_chain, _key_pair


@njit(cache=True, parallel=True, nogil=True, error_model="numpy")
def _key_univariate_chains(X, rows, n_blocks, v_kind, v_win, v_ch, v_diff,
                           v_dptr, dils, v_cuts, v_flat, d_col, d_tab,
                           own_level, l_row, l_cuts, v_woff, v_trig, cos_tab,
                           sin_tab, mode, nr, row_tile, head_col, hc_none,
                           tf_tab, idf, W, out_ids, out_cnt, out_nnz, acc, sq,
                           g_v0, g_v1, g_sd0, g_sd1, sd_d, sd_p0, sd_p1,
                           sd_wmin, sp_v, sp_di, d0, di_slot, di_off, n_store,
                           st_total, max_pd):
    """words_univariate._key_univariate's output, byte for byte, keyed chain by
    chain. The schedule (transform.chains.chain_pairs) cuts the call's views
    into groups (one channel and side each) and lists, per group, the stored
    dilations (read by two or more pairs) and their pairs (sp_v, sp_di, by
    window); every other pair is a fallback pair. Per row tile and group:

      1. keying, per stored dilation d and row: each residue chain r0 that
         holds the dilation's smallest window is gathered and its mean and
         centred prefix sums built once (_chain_part). The pairs read it in
         window order: window moments once per window length
         (_window_part), the level part, and the PAA or Fourier keys, into
         ks[row in tile, pair, word or level, start].
      2. emission of the dilation's pairs over the tile's rows (_emit) into
         the result store, per slot of the call (read at di - d0): mode 0
         the (id, count) entries (offset di_off, count st_n), mode 1 the
         head sums and squared norms (slot di_slot).
      3. the plan-order pass: per view, dilation slot and row, a stored pair
         copies its entries at the row's cursor (mode 0) or adds its sums to
         acc and sq (mode 1); a fallback pair runs _key_pair at its place.

    A row receives its blocks in _key_univariate's order with the same
    operands, so every output equals _key_univariate's whatever the tile: a
    chain's keys depend on the chain and the window only. Each block owns
    its rows lo..hi, so blocks race on nothing."""
    n_rows = rows.shape[0]
    m = X.shape[2]
    n_heads = W.shape[1]
    tile = max(row_tile, np.int64(1))
    n_groups = g_v0.shape[0]
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
        nq = max(min(tile, hi - lo), np.int64(1))
        ks = np.empty((nq, max(max_pd, np.int64(1)), 2, m), np.uint8)
        n_slot = max(n_store, np.int64(1))
        if mode == _MODE_GLOBAL:
            st_ids = np.empty((nq, max(st_total, np.int64(1))),
                              dtype=out_ids.dtype)
            st_cnt = np.empty((nq, max(st_total, np.int64(1))),
                              dtype=out_cnt.dtype)
            st_acc = np.empty((1, 1, 2, 1))
            st_sq = np.empty((1, 1, 2))
        else:
            st_ids = np.empty((1, 1), dtype=out_ids.dtype)
            st_cnt = np.empty((1, 1), dtype=out_cnt.dtype)
            st_acc = np.empty((nq, n_slot, 2, max(n_heads, 1)))
            st_sq = np.empty((nq, n_slot, 2))
        st_n = np.zeros((nq, n_slot), np.int64)
        for i0 in range(lo, hi, tile):
            i1 = min(i0 + tile, hi)
            for g in range(n_groups):
                for sdi in range(g_sd0[g], g_sd1[g]):
                    d = sd_d[sdi]
                    p0 = sd_p0[sdi]
                    p1 = sd_p1[sdi]
                    vf = sp_v[p0]
                    ch = v_ch[vf]
                    is_diff = v_diff[vf] != 0
                    # keying: one chain build per residue, read by every
                    # pair of the dilation
                    for i in range(i0, i1):
                        q = i - i0
                        for r0 in range(d):
                            L = (m - r0 + d - 1) // d
                            if L < sd_wmin[sdi]:
                                continue
                            _gather(X, rows[i], ch, is_diff, r0, d, L, c)
                            cm = _chain_part(c, L, S, S2)
                            w_cur = np.int64(-1)
                            n_w = np.int64(0)
                            for pi in range(p0, p1):
                                v = sp_v[pi]
                                w = v_win[v]
                                if L < w:
                                    continue
                                if w != w_cur:
                                    n_w = _window_part(c, L, w, cm, S, S2,
                                                       mean, std, flag)
                                    w_cur = w
                                lp = pi - p0
                                if own_level[sp_di[pi]] != 0:
                                    lr = l_row[v]
                                    for t in range(n_w):
                                        ks[q, lp, 1, r0 + t * d] = np.uint8(
                                            _level_part(l_cuts, lr, mean[t],
                                                        std[t], LEVEL_BITS,
                                                        LEVEL_BITS))
                                trig = v_trig[v]
                                _key_chain(v_kind[v], c, L, w, n_w, std,
                                           pool, inv_scale, mr, mi, key,
                                           cos_tab[trig], sin_tab[trig],
                                           v_cuts[v], v_flat[v])
                                for t in range(n_w):
                                    ks[q, lp, 0, r0 + t * d] = np.uint8(
                                        key[t])
                    # emission into the store, pair by pair over the rows
                    for pi in range(p0, p1):
                        v = sp_v[pi]
                        di = sp_di[pi]
                        s = di_slot[di - d0]
                        o = di_off[di - d0]
                        n_windows = m - (v_win[v] - 1) * d
                        lp = pi - p0
                        for i in range(i0, i1):
                            q = i - i0
                            n_out, bsq = _emit(
                                ks[q, lp, 0], n_windows, hist, bm, mode, nr,
                                q, o, d_col[di], d_tab[di], v_woff[v],
                                head_col, hc_none, tf_tab, idf, W, st_ids,
                                st_cnt, blk_acc, n_heads)
                            if mode == _MODE_SCORE:
                                st_sq[q, s, 0] = bsq
                                for h in range(n_heads):
                                    st_acc[q, s, 0, h] = blk_acc[h]
                            if own_level[di] != 0:
                                n_out, bsq = _emit(
                                    ks[q, lp, 1], n_windows, hist, bm, mode,
                                    nr, q, n_out, d_col[di] + KEYSPACE,
                                    d_tab[di] + KEYSPACE, v_woff[v], head_col,
                                    hc_none, tf_tab, idf, W, st_ids, st_cnt,
                                    blk_acc, n_heads)
                                if mode == _MODE_SCORE:
                                    st_sq[q, s, 1] = bsq
                                    for h in range(n_heads):
                                        st_acc[q, s, 1, h] = blk_acc[h]
                            st_n[q, s] = n_out - o
                # the plan-order pass
                for v in range(g_v0[g], g_v1[g]):
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
                        s = di_slot[di - d0]
                        for i in range(i0, i1):
                            q = i - i0
                            n_out = out_nnz[i]
                            if s >= 0:
                                if mode == _MODE_GLOBAL:
                                    o = di_off[di - d0]
                                    for e in range(st_n[q, s]):
                                        out_ids[i, n_out] = st_ids[q, o + e]
                                        out_cnt[i, n_out] = st_cnt[q, o + e]
                                        n_out += 1
                                else:
                                    sq[i] += st_sq[q, s, 0]
                                    for h in range(n_heads):
                                        acc[i, h] += st_acc[q, s, 0, h]
                                    if own:
                                        sq[i] += st_sq[q, s, 1]
                                        for h in range(n_heads):
                                            acc[i, h] += st_acc[q, s, 1, h]
                                out_nnz[i] = n_out
                                continue
                            # a fallback pair: _key_univariate's body
                            _key_pair(X, rows[i], ch, is_diff, v_kind[v], w, d,
                                      own, lr, v_cuts[v], v_flat[v],
                                      cos_tab[trig], sin_tab[trig], l_cuts, c,
                                      S, S2, mean, std, flag, pool, inv_scale,
                                      mr, mi, key, key_j, kl)
                            out_nnz[i] = _emit_pair(
                                key_j, kl, own, n_windows, hist, bm, mode, nr,
                                i, n_out, d_col[di], d_tab[di], v_woff[v],
                                head_col, hc_none, tf_tab, idf, W, out_ids,
                                out_cnt, blk_acc, n_heads, acc, sq)
