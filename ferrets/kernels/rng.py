"""numpy's legacy MT19937 RandomState streams replayed in numba: the view
draw and the sampler keys."""

import numpy as np
from numba import njit, prange


# MT19937 and masked bounded integers, as in numpy's legacy RandomState
# (its streams are frozen by NEP 19). The view draw and the Fourier sampler
# are defined by those streams; these kernels reproduce them bit for bit.

_MT_N = 624


@njit(cache=True, nogil=True)
def _mt_seed(seed):
    """RandomState(seed)'s initial MT19937 state (624 words)."""
    key = np.empty(_MT_N, np.uint64)
    sd = np.uint64(seed) & np.uint64(0xffffffff)
    for pos in range(_MT_N):
        key[pos] = sd
        sd = (np.uint64(1812433253) * (sd ^ (sd >> np.uint64(30)))
              + np.uint64(pos + 1)) & np.uint64(0xffffffff)
    return key


@njit(cache=True, nogil=True)
def _mt_twist(key):
    upper = np.uint64(0x80000000)
    lower = np.uint64(0x7fffffff)
    mat = np.uint64(0x9908b0df)
    one = np.uint64(1)
    for i in range(_MT_N):
        y = (key[i] & upper) | (key[(i + 1) % _MT_N] & lower)
        v = key[(i + 397) % _MT_N] ^ (y >> one)
        if y & one:
            v ^= mat
        key[i] = v


@njit(inline="always")
def _mt_next(key, pos):
    """(next 32-bit output, new position) of an MT19937 state."""
    if pos == _MT_N:
        _mt_twist(key)
        pos = 0
    y = key[pos]
    y ^= (y >> np.uint64(11))
    y ^= (y << np.uint64(7)) & np.uint64(0x9d2c5680)
    y ^= (y << np.uint64(15)) & np.uint64(0xefc60000)
    y ^= (y >> np.uint64(18))
    return np.int64(y & np.uint64(0xffffffff)), pos + 1


@njit(cache=True, nogil=True)
def _mt_stream(seed, n_words):
    """The first n_words 32-bit outputs of RandomState(seed)."""
    key = _mt_seed(seed)
    out = np.empty(n_words, np.int64)
    pos = _MT_N
    for i in range(n_words):
        out[i], pos = _mt_next(key, pos)
    return out


@njit(inline="always")
def _mask_of(rng):
    mask = rng
    mask |= mask >> 1
    mask |= mask >> 2
    mask |= mask >> 4
    mask |= mask >> 8
    mask |= mask >> 16
    return mask


@njit(cache=True, nogil=True)
def _draw_views_nb(seeds, n_views, n_win, hi, src, widx, u):
    """The view draw: per pseudo-channel p, RandomState(seeds[p]) makes n_views
    times the calls randint(2), randint(n_win) and uniform(0, hi[window]),
    value for value, into src, widx and u (each [p, view]). Bounded integers
    use masked rejection on 32-bit outputs; uniform(0, h) is
    h * ((a >> 5) * 2^26 + (b >> 6)) / 2^53 from two outputs a, b."""
    rng_w = n_win - 1
    mask_w = _mask_of(rng_w)
    for p in range(seeds.shape[0]):
        key = _mt_seed(seeds[p])
        pos = _MT_N
        for i in range(n_views):
            v, pos = _mt_next(key, pos)
            src[p, i] = v & 1
            wi = 0
            if rng_w > 0:
                while True:
                    v, pos = _mt_next(key, pos)
                    wi = v & mask_w
                    if wi <= rng_w:
                        break
            widx[p, i] = wi
            a, pos = _mt_next(key, pos)
            b, pos = _mt_next(key, pos)
            dbl = ((a >> 5) * 67108864.0 + (b >> 6)) / 9007199254740992.0
            u[p, i] = 0.0 + hi[wi] * dbl


@njit(cache=True, nogil=True)
def _draw_key(stream, n, m, window, dils, per_dil, rows, starts, dil):
    """One sampler configuration's draw from RandomState(seed)'s output
    stream: per dilation every window, or randint(0, n, per_dil) rows then
    randint(0, n_windows, per_dil) starts (masked rejection), written to
    rows / starts / dil. Returns the samples written, or -1 when the stream
    ran out (the caller extends it)."""
    q = 0
    at = 0
    for di in range(dils.shape[0]):
        d = dils[di]
        n_windows = m - (window - 1) * d
        total = n * n_windows
        if per_dil >= total:
            for f in range(total):
                rows[at + f] = f // n_windows
                starts[at + f] = f % n_windows
                dil[at + f] = d
            at += total
            continue
        for part in range(2):
            rng = (n - 1) if part == 0 else (n_windows - 1)
            mask = _mask_of(rng)
            for i in range(per_dil):
                val = 0
                if rng > 0:
                    while True:
                        if q >= stream.shape[0]:
                            return -1
                        val = stream[q] & mask
                        q += 1
                        if val <= rng:
                            break
                if part == 0:
                    rows[at + i] = val
                else:
                    starts[at + i] = val
                    dil[at + i] = d
        at += per_dil
    return at


@njit(cache=True, parallel=True, nogil=True)
def _draw_keys(stream, n, m, k_w, k_dils, k_dptr, k_per, k_off, rows, starts,
               dil, done):
    """Every sampler key's draw (_draw_key: each key replays the same output
    stream with its own rejection pattern), reordered by row into rows /
    starts / dil[k_off[q]:k_off[q+1]]; parallel over keys, each writing its
    own slice.

    The reorder is a stable sort by row (a lane's median does not depend on
    its samples' order; row order keeps each series in cache across its
    samples): an LSD radix sort on 11-bit digits of the row, one counting
    pass per digit of n - 1, O(k) per pass with a 2,049-entry count array
    whatever n. Keys with done[q] set are skipped; done[q] is set once the
    stream sufficed (the caller extends the stream and calls again)."""
    for q in prange(k_w.shape[0]):
        if done[q] != 0:
            continue
        a = k_off[q]
        k = k_off[q + 1] - a
        r_ = np.empty(k, np.int64)
        s_ = np.empty(k, np.int64)
        d_ = np.empty(k, np.int64)
        if _draw_key(stream, n, m, k_w[q], k_dils[k_dptr[q]:k_dptr[q + 1]],
                     k_per[q], r_, s_, d_) < 0:
            continue
        done[q] = 1
        order = np.arange(k)
        tmp = np.empty(k, np.int64)
        cnt = np.empty(2049, np.int64)
        shift = 0
        while True:                    # one stable counting pass per digit
            cnt[:] = 0
            for i in range(k):
                cnt[((r_[order[i]] >> shift) & 2047) + 1] += 1
            for b in range(2048):
                cnt[b + 1] += cnt[b]
            for i in range(k):
                o = order[i]
                g = (r_[o] >> shift) & 2047
                tmp[cnt[g]] = o
                cnt[g] += 1
            order, tmp = tmp, order
            shift += 11
            if (n - 1) >> shift <= 0:
                break
        for i in range(k):
            o = order[i]
            rows[a + i] = r_[o]
            starts[a + i] = s_[o]
            dil[a + i] = d_[o]
