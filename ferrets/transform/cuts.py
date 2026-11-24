"""The Fourier cut tables and the level word's cut sets, sampled from
training windows."""

import numba
import numpy as np

from .. import config
from ..constants import (LEVEL_CUTS, TIE_FOURIER, TIE_MEAN, TIE_STD,
                         WORD_LEN)
from ..helpers.split import _split
from ..kernels.cuts import _level_cut_jobs, _fourier_cut_jobs
from ..kernels.rng import _draw_keys, _mt_stream


def _sampler_tables(window):
    """The sampler's cos and sin tables, flat in (t, k) with t < window and
    k < 4: cos / sin(2 pi ((k+1) t mod w) / w) read from numpy's cos and sin
    over arange(w), with the entries at quarter turns rounded to exact 0 and
    +-1."""
    a = np.arange(window)
    two_pi = 2.0 * np.pi
    cos_t = np.cos(two_pi * a.astype(np.float64) / window)
    sin_t = np.sin(two_pi * a.astype(np.float64) / window)
    for t in range(window):
        if (4 * t) % window == 0:
            cos_t[t] = np.round(cos_t[t])
            sin_t[t] = np.round(sin_t[t])
    idx = ((np.arange(4)[:, None] + 1) * a[None, :]) % window
    return (np.ascontiguousarray(cos_t[idx].T.ravel()),
            np.ascontiguousarray(sin_t[idx].T.ravel()))


def _cut_batches(X, jobs, m, seed):
    """The sampler's draws for jobs (ch, is_diff, window, dils). The draw
    depends only on the key (window, dils, per-dilation count) and every key
    replays a fresh RandomState(seed); jobs are sorted by key and batched
    under SAMPLER_BATCH_MB, each batch's distinct keys drawn once, in
    parallel. Yields (batch, draw_off, grp_ptr, d_rows, d_starts, d_dil):
    batch a list of (key, samples, job index, ch, is_diff), draw_off each
    key's first sample in the d_* buffers (reused by the next batch: use them
    before advancing), grp_ptr the batch cut into groups of consecutive jobs
    of about equal work (samples x window)."""
    n = X.shape[0]
    ready = []            # (key, samples, job index, ch, is_diff)
    for ji, (ch, is_diff, w, dils) in enumerate(jobs):
        per_dil = max(1, config.FOURIER_MAX_WINDOWS // max(1, len(dils)))
        k = 0
        for d in dils:
            tot = n * (m - (w - 1) * d)
            k += tot if per_dil >= tot else per_dil
        ready.append(((w, tuple(dils), per_dil), k, ji, ch, is_diff))
    ready.sort(key=lambda r: (r[0], r[2]))
    stream = _mt_stream(np.int64(seed), np.int64(1 << 18))
    cap = max(max(r[1] for r in ready),
              int(config.SAMPLER_BATCH_MB * 1e6 // 24))
    d_rows = np.empty(cap, dtype=np.int64)       # reused by every batch
    d_starts = np.empty(cap, dtype=np.int64)
    d_dil = np.empty(cap, dtype=np.int64)
    n_grp_target = 4 * max(1, numba.get_num_threads())
    start = 0
    while start < len(ready):
        # consecutive jobs whose distinct draws fit the buffers
        stop, used, draw_off = start, 0, {}
        while stop < len(ready):
            key, k = ready[stop][0], ready[stop][1]
            if key not in draw_off:
                if stop > start and used + k > cap:
                    break
                draw_off[key] = used
                used += k
            stop += 1
        batch = ready[start:stop]
        keys = list(draw_off)
        k_dptr = np.zeros(len(keys) + 1, dtype=np.int64)
        np.cumsum([len(key[1]) for key in keys], out=k_dptr[1:])
        done = np.zeros(len(keys), dtype=np.uint8)
        args = (np.asarray([key[0] for key in keys], dtype=np.int64),
                np.asarray([d for key in keys for d in key[1]],
                           dtype=np.int64),
                k_dptr, np.asarray([key[2] for key in keys], dtype=np.int64),
                np.asarray([draw_off[key] for key in keys] + [used],
                           dtype=np.int64),
                d_rows, d_starts, d_dil, done)
        while True:
            _draw_keys(stream, np.int64(n), np.int64(m), *args)
            if done.all():
                break
            stream = _mt_stream(np.int64(seed), np.int64(2 * len(stream)))
        nb_ = len(batch)
        _, _, j_w, j_k, _, _ = _job_arrays(batch, draw_off)
        # groups of consecutive jobs of about equal work (samples x window)
        cost = np.cumsum(j_k * j_w)
        n_grp = int(min(nb_, n_grp_target))
        cuts = np.searchsorted(cost, _split(cost[-1], n_grp)[1:-1],
                               side="right")
        grp_ptr = np.unique(np.concatenate([[0], cuts, [nb_]])).astype(
            np.int64)
        yield batch, draw_off, grp_ptr, d_rows, d_starts, d_dil
        start = stop


def _job_arrays(batch, draw_off):
    """A batch's per-job arrays: channel, difference flag, window, sample
    count, first sample in the draw buffers, job index."""
    i64 = np.int64
    return (np.array([r[3] for r in batch], dtype=i64),
            np.array([int(r[4]) for r in batch], dtype=i64),
            np.array([r[0][0] for r in batch], dtype=i64),
            np.array([r[1] for r in batch], dtype=i64),
            np.array([draw_off[r[0]] for r in batch], dtype=i64),
            np.array([r[2] for r in batch], dtype=i64))


def _fit_fourier_cuts(X, jobs, m, seed, lev=None, lcuts=None):
    """The (8,) cut table of every job (ch, is_diff, window, dils): the
    training median of each normalised Fourier part over at most
    FOURIER_MAX_WINDOWS sampled windows (_cut_batches' draws). With lev (one
    entry per job: a row of lcuts, or -1), a job also fills that row of
    lcuts with the level cut set of the level job that shares its draw and
    data (_fit_cuts)."""
    tables = np.empty((len(jobs), WORD_LEN))
    if lev is None:
        lev = [-1] * len(jobs)
        lcuts = np.empty((0, 2 * LEVEL_CUTS))
    if not jobs:
        return tables
    tab_off, c_l, s_l = {}, [], []
    n_tab = 0
    for (_, _, w, _) in jobs:
        if w not in tab_off:
            cc, ss = _sampler_tables(w)
            tab_off[w] = n_tab
            c_l.append(cc)
            s_l.append(ss)
            n_tab += 4 * w
    c_all = np.concatenate(c_l)
    s_all = np.concatenate(s_l)
    for batch, draw_off, grp_ptr, d_rows, d_starts, d_dil in _cut_batches(
            X, jobs, m, seed):
        j_ch, j_diff, j_w, j_k, j_draw, j_res = _job_arrays(batch, draw_off)
        j_tab = np.array([tab_off[w] for w in j_w.tolist()], dtype=np.int64)
        j_lev = np.array([lev[ji] for ji in j_res.tolist()], dtype=np.int64)
        _fourier_cut_jobs(X, grp_ptr, j_ch, j_diff, j_w, j_k, j_draw, j_tab,
                          j_res, d_rows, d_starts, d_dil, c_all, s_all,
                          np.int64(int(j_k.max())), tables, j_lev, lcuts)
    return tables


def _fit_cuts(X, fourier_jobs, level_jobs, m, seed):
    """(the Fourier cut tables of fourier_jobs, the level cut sets of
    level_jobs), each job (ch, is_diff, window, dils), equal bit for bit to
    what _fit_fourier_cuts and _fit_level_cuts compute apart, before the tie
    margins (_fit_tables adds them). Shared work:

    * DEDUP_FOURIER_JOBS: a Fourier job equal to an earlier one is computed
      once and its table copied;
    * FUSE_LEVEL_CUTS: a level job equal to a Fourier job (same channel,
      side, window and dilation set, so the same draw and windows) is carried
      by that job in _fourier_cut_jobs, which computes those windows' means
      and deviations anyway.

    A job's draw replays its key's stream whatever else is batched with it."""
    lcuts = np.empty((len(level_jobs), 2 * LEVEL_CUTS))
    if config.DEDUP_FOURIER_JOBS:
        first = {}
        for job in fourier_jobs:
            first.setdefault(job, len(first))
        run = list(first)
        back = np.array([first[job] for job in fourier_jobs], dtype=np.int64)
    else:
        run, back = list(fourier_jobs), None
    lev = [-1] * len(run)
    alone = list(range(len(level_jobs)))
    if config.FUSE_LEVEL_CUTS and run and level_jobs:
        at = {}
        for i, job in enumerate(run):
            at.setdefault(job, i)
        alone = []
        for li, job in enumerate(level_jobs):
            i = at.get(tuple(job), -1)
            if i >= 0 and lev[i] < 0:
                lev[i] = li
            else:
                alone.append(li)
    tables = _fit_fourier_cuts(X, run, m, seed, lev, lcuts)
    if back is not None:
        tables = tables[back]
    if alone:
        lcuts[alone] = _fit_level_cuts(X, [level_jobs[i] for i in alone], m,
                                       seed)
    return tables, lcuts


def _fit_tables(X, fourier_jobs, level_jobs, m, seed):
    """The stored cut tables: _fit_cuts's sampled tables plus the tie margins
    (_tie_margins)."""
    return _tie_margins(*_fit_cuts(X, fourier_jobs, level_jobs, m, seed))


def _tie_margins(tables, lcuts):
    """The sampled tables with the tie margins of constants (TIE_FOURIER,
    TIE_MEAN, TIE_STD) added, so every reader (plan, flat key, word sharing,
    kernels, decoder) sees the stored tables. The mean cuts' spread is taken
    per cut set (monotone in c_j: the sorted order and equal cuts are kept);
    std cuts equal to 0 (_flat_std_cuts) stay 0. Returns new arrays
    (tables, lcuts); the arguments are not changed."""
    tables = tables + TIE_FOURIER
    lcuts = lcuts.copy()
    if len(lcuts):
        mc = lcuts[:, :LEVEL_CUTS]
        lcuts[:, :LEVEL_CUTS] = mc + TIE_MEAN * (
            np.abs(mc) + (mc[:, -1:] - mc[:, :1]))
        sc = lcuts[:, LEVEL_CUTS:]
        lcuts[:, LEVEL_CUTS:] = np.where(sc > 0.0, sc * (1.0 + TIE_STD), sc)
    return tables, lcuts


def _fit_level_cuts(X, jobs, m, seed):
    """The level word's cut sets, one (30,) row per job (ch, is_diff, window,
    D): the 15 linear quantiles j / 16 of the sampled windows' means, then of
    their population standard deviations, over the Fourier sampler's draw for
    the key (window, D, per-dilation count), the std cuts at most EPS_STD set
    to 0 (a flat window's std is 0: kernels.level); (0, 30) without jobs."""
    tables = np.empty((len(jobs), 2 * LEVEL_CUTS))
    if not jobs:
        return tables
    for batch, draw_off, grp_ptr, d_rows, d_starts, d_dil in _cut_batches(
            X, jobs, m, seed):
        j_ch, j_diff, j_w, j_k, j_draw, j_res = _job_arrays(batch, draw_off)
        _level_cut_jobs(X, grp_ptr, j_ch, j_diff, j_w, j_k, j_draw, j_res,
                        d_rows, d_starts, d_dil, np.int64(int(j_k.max())),
                        np.int64(max(1, int(config.LEVEL_SAMPLER_BLOCK))),
                        tables)
    return tables
