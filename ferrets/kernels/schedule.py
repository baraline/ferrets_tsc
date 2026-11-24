"""The chain-major schedule of one univariate keying call, in numba
(transform.chains wraps it into ChainPairs and describes it): which (view,
dilation) pairs share their residue chains, where their results are stored,
and the row tile. Every array is int64."""

import numpy as np
from numba import njit

from ..constants import KEYSPACE, _MODE_GLOBAL, _MODE_SCORE


@njit(cache=True)
def _groups(ch, diff, v0, v1):
    """The runs of consecutive views in [v0, v1) with one channel and side:
    (first views, ends, count); the arrays are longer than the count."""
    n = max(v1 - v0, 0)
    g_v0 = np.empty(n, np.int64)
    g_v1 = np.empty(n, np.int64)
    n_g = 0
    v = v0
    while v < v1:
        e = v + 1
        while e < v1 and ch[e] == ch[v] and diff[e] == diff[v]:
            e += 1
        g_v0[n_g] = v
        g_v1[n_g] = e
        n_g += 1
        v = e
    return g_v0, g_v1, n_g


@njit(cache=True)
def _sorted_pairs(win, kind, dptr, dils, m, g_v0, g_v1, n_g, n_slot):
    """The (view, dilation slot) pairs of every group that have at least one
    window (m - (w - 1) d > 0), each group's pairs contiguous and sorted by
    (dilation, window, kind, view), ties by slot: their dilation, view and
    slot, each group's first pair (gp0, n_g + 1 entries) and the pair
    count; the arrays are longer than the count."""
    pd = np.empty(n_slot, np.int64)
    pv = np.empty(n_slot, np.int64)
    pdi = np.empty(n_slot, np.int64)
    gp0 = np.empty(n_g + 1, np.int64)
    n_p = 0
    for g in range(n_g):
        gp0[g] = n_p
        a = n_p
        for vv in range(g_v0[g], g_v1[g]):
            w = win[vv]
            for di in range(dptr[vv], dptr[vv + 1]):
                d = dils[di]
                if m - (w - 1) * d > 0:
                    pd[n_p] = d
                    pv[n_p] = vv
                    pdi[n_p] = di
                    n_p += 1
        cnt = n_p - a
        if cnt > 1:
            nv = g_v1[g] - g_v0[g] + 1
            key = np.empty(cnt, np.int64)
            for j in range(cnt):
                vv = pv[a + j]
                key[j] = (((pd[a + j] * (m + 1) + win[vv]) * 2
                           + kind[vv]) * nv + (vv - g_v0[g]))
            order = np.argsort(key, kind="mergesort")   # ties: by slot
            td = pd[a:n_p].copy()
            tv = pv[a:n_p].copy()
            tdi = pdi[a:n_p].copy()
            for j in range(cnt):
                pd[a + j] = td[order[j]]
                pv[a + j] = tv[order[j]]
                pdi[a + j] = tdi[order[j]]
    gp0[n_g] = n_p
    return pd, pv, pdi, gp0, n_p


@njit(cache=True)
def _has_level(own_level):
    """1 when the pair has a level block, else 0."""
    return 1 if own_level != 0 else 0


@njit(inline="always")
def _pair_entries(own_level, win, m, d):
    """The most entries a pair of window win at dilation d stores per row:
    one key block, plus one for a level block."""
    return (1 + _has_level(own_level)) * min(KEYSPACE, m - (win - 1) * d)


@njit(cache=True)
def _by_size(size, d):
    """The order in which a group's shared dilations enter the store: the
    larger size (pair count) first, ties by the smaller dilation (the
    dilations are distinct)."""
    n = len(d)
    if n == 0:
        return np.empty(0, np.int64)
    top = max(np.max(size), 0) + 1
    span = np.max(d) + 1
    key = np.empty(n, np.int64)
    for t in range(n):
        key[t] = (top - size[t]) * span + d[t]
    return np.argsort(key)


@njit(cache=True)
def _tile(mode, row_tile, key_bytes, per_row):
    """Rows per tile: 1 outside score mode, else the largest number up to
    row_tile whose key scratch (per_row bytes a row) fits key_bytes, at
    least 1."""
    if mode != _MODE_SCORE:
        return np.int64(1)
    # the full tile is compared in bytes first: a quotient past 2**63 casts
    # to INT64_MIN, and an infinite budget floor-divides to nan
    per_row = max(1, per_row)
    if key_bytes >= row_tile * per_row:
        return np.int64(row_tile)
    return np.int64(max(1, int(key_bytes // per_row)))


@njit(cache=True)
def _chain_schedule(kind, win, ch, diff, dptr, dils, own_level, v0, v1, m,
                    mode, row_tile, n_heads, id_size, cnt_size, key_bytes,
                    store_bytes):
    """The schedule of transform.chains.chain_pairs, which describes the
    outputs: g_v0, g_v1, g_sd0, g_sd1, sd_d, sd_p0, sd_p1, sd_wmin, sp_v,
    sp_di, di_slot, di_off, n_store, st_total, the most pairs at a stored
    dilation and the tile. key_bytes and store_bytes are the budgets in
    bytes; the per-group arrays and the stored-dilation / stored-pair arrays
    are cut to their lengths, di_slot and di_off have one entry per slot of
    the call (-1 where the pair is not stored)."""
    d0 = dptr[v0]
    n_slot = dptr[v1] - d0
    g_v0, g_v1, n_g = _groups(ch, diff, v0, v1)
    pd, pv, pdi, gp0, n_p = _sorted_pairs(win, kind, dptr, dils, m, g_v0,
                                          g_v1, n_g, n_slot)
    # shared dilations: runs of one dilation with two or more pairs whose
    # keys at one row fit the key scratch
    sh_d = np.empty(n_p, np.int64)
    sh_a = np.empty(n_p, np.int64)
    sh_n = np.empty(n_p, np.int64)
    gs0 = np.empty(n_g + 1, np.int64)
    n_sh = 0
    max_pd = 0
    for g in range(n_g):
        gs0[g] = n_sh
        j = gp0[g]
        while j < gp0[g + 1]:
            e = j + 1
            while e < gp0[g + 1] and pd[e] == pd[j]:
                e += 1
            c = e - j
            if c >= 2 and c * 2 * m <= key_bytes:
                sh_d[n_sh] = pd[j]
                sh_a[n_sh] = j
                sh_n[n_sh] = c
                n_sh += 1
                max_pd = max(max_pd, c)
            j = e
    gs0[n_g] = n_sh
    tile = _tile(mode, row_tile, key_bytes, max_pd * 2 * m)
    g_sd0 = np.empty(n_g, np.int64)
    g_sd1 = np.empty(n_g, np.int64)
    sd_d = np.empty(n_sh, np.int64)
    sd_p0 = np.empty(n_sh, np.int64)
    sd_p1 = np.empty(n_sh, np.int64)
    sd_wmin = np.empty(n_sh, np.int64)
    sp_v = np.empty(n_p, np.int64)
    sp_di = np.empty(n_p, np.int64)
    di_slot = np.full(n_slot, -1, np.int64)
    di_off = np.full(n_slot, -1, np.int64)
    n_sd = 0
    n_sp = 0
    n_store = 0
    st_total = 0
    stored_pd = 0
    for g in range(n_g):
        g_sd0[g] = n_sd
        ns = gs0[g + 1] - gs0[g]
        order = _by_size(sh_n[gs0[g]:gs0[g + 1]], sh_d[gs0[g]:gs0[g + 1]])
        chosen = np.zeros(ns, np.bool_)
        used = 0
        for t in order:
            s = gs0[g] + t
            if mode == _MODE_GLOBAL:
                caps = 0
                for j in range(sh_a[s], sh_a[s] + sh_n[s]):
                    caps += _pair_entries(own_level[pdi[j]], win[pv[j]], m,
                                          sh_d[s])
                cost = caps * (id_size + cnt_size) * tile
            else:
                cost = sh_n[s] * 2 * (n_heads + 1) * 8 * tile
            if used + cost <= store_bytes:
                used += cost
                chosen[t] = True
        slots = 0
        total = 0
        for t in range(ns):
            if not chosen[t]:
                continue
            s = gs0[g] + t
            sd_d[n_sd] = sh_d[s]
            sd_p0[n_sd] = n_sp
            sd_wmin[n_sd] = win[pv[sh_a[s]]]
            for j in range(sh_a[s], sh_a[s] + sh_n[s]):
                sp_v[n_sp] = pv[j]
                sp_di[n_sp] = pdi[j]
                di_slot[pdi[j] - d0] = slots
                di_off[pdi[j] - d0] = total
                slots += 1
                if mode == _MODE_GLOBAL:
                    total += _pair_entries(own_level[pdi[j]], win[pv[j]], m,
                                           sh_d[s])
                n_sp += 1
            sd_p1[n_sd] = n_sp
            stored_pd = max(stored_pd, sh_n[s])
            n_sd += 1
        g_sd1[g] = n_sd
        n_store = max(n_store, slots)
        st_total = max(st_total, total)
    return (g_v0[:n_g], g_v1[:n_g], g_sd0, g_sd1, sd_d[:n_sd],
            sd_p0[:n_sd], sd_p1[:n_sd], sd_wmin[:n_sd], sp_v[:n_sp],
            sp_di[:n_sp], di_slot, di_off, n_store, st_total, stored_pd,
            tile)

