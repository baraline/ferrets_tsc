"""The chain-major keying schedule of one univariate keying call: which
(view, dilation) pairs share their residue chains, where their results are
stored, and the row tile (kernels.chain._key_univariate_chains reads it). It is
built in numba (kernels.schedule), one per pseudo-channel in a per-channel
fit and one over every view in its predict."""

from collections import namedtuple

import numpy as np

from ..kernels.schedule import _chain_schedule

ChainPairs = namedtuple("ChainPairs", [
    "g_v0", "g_v1",            # groups: views [g_v0, g_v1) of one channel
    "g_sd0", "g_sd1",          # ... and their stored dilations
    "sd_d", "sd_p0", "sd_p1", "sd_wmin",  # stored dilation: d, its pairs
    "sp_v", "sp_di",           # stored pair: view, dilation slot
    "d0",                      # the call's first dilation slot, dptr[v0]
    "di_slot", "di_off",       # per slot di of the call, at di - d0: store
                               # slot, entry offset (-1: a fallback pair)
    "n_store", "st_total", "max_pd", "tile"])


def chain_pairs(pl, v0, v1, m, mode, row_tile, n_heads, id_size, cnt_size,
                key_mb, store_mb):
    """The schedule of a keying call over views [v0, v1) of plan pl, or None
    when it stores no dilation.

    A group is a run of views with one channel and side (a pseudo-channel's
    views, contiguous in plan order); groups never share a chain. Within a
    group the pairs (view, dilation slot) with windows are grouped by
    dilation, and a dilation read by two or more pairs is shared if its key
    scratch at one row (pairs x 2 x m bytes) fits key_mb. The result store
    takes the shared dilations whole, most pairs first (ties: the smaller
    dilation), while its bytes fit store_mb, restarting at each group: per
    pair kernels.schedule._pair_entries x (id_size + cnt_size) in mode 0, two
    blocks of n_heads + 1 doubles in mode 1, times the tile's rows. Every
    other pair is a fallback pair, keyed on its own. Stored dilations run
    ascending, their pairs by (window, kind, view, slot), so pairs of one
    window length share the window moments.

    The tile is 1 in mode 0; in mode 1 the largest up to row_tile whose key
    scratch fits key_mb, at least 1. Rows never interact, so the output
    depends on neither budget. di_slot and di_off cover the call's slots
    [dptr[v0], dptr[v1]) only, indexed from d0 = dptr[v0]: a fit keeps every
    pseudo-channel's schedule."""
    out = _chain_schedule(
        pl["kind"], pl["win"], pl["ch"], pl["diff"], pl["dptr"], pl["dils"],
        pl["own_level"], int(v0), int(v1), int(m), int(mode), int(row_tile),
        int(n_heads), int(id_size), int(cnt_size), float(key_mb * 1e6),
        float(store_mb * 1e6))
    if len(out[4]) == 0:
        return None
    arrays, (n_store, st_total, max_pd, tile) = out[:12], out[12:]
    return ChainPairs(
        *(a.copy() for a in arrays[:10]), np.int64(pl["dptr"][v0]),
        arrays[10], arrays[11], np.int64(n_store), np.int64(st_total),
        np.int64(max_pd), np.int64(tile))

