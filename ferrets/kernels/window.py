"""Dilated window gather, window moments with the two-pass rescue, and the
Fourier trig tables."""

import numpy as np
from numba import njit

from ..constants import N_COEF, RESCUE_CHAIN, RESCUE_REL


@njit(inline="always")
def _gather(X, i, ch, is_diff, r0, d, L, c):
    """The L values r0, r0 + d, ... of channel ``ch`` of row ``i`` of X
    (n, C, m) into c: the series, or its first difference (the last
    increment repeated) when ``is_diff``."""
    m = X.shape[2]
    if is_diff:
        for t in range(L - 1):
            j = r0 + t * d
            c[t] = X[i, ch, j + 1] - X[i, ch, j]
        j = r0 + (L - 1) * d
        if j < m - 1:
            c[L - 1] = X[i, ch, j + 1] - X[i, ch, j]
        else:
            c[L - 1] = X[i, ch, m - 1] - X[i, ch, m - 2]
    else:
        for t in range(L):
            c[t] = X[i, ch, r0 + t * d]


@njit(inline="always")
def _moments(c, L, w, S, S2, mean, std, flag):
    """Mean and std of every window of length w of the chain c (length L);
    returns the window count. Prefix sums over the chain centred on its
    mean give every window's one-pass moments; a window whose one-pass
    variance is suspect (tiny against the chain's energy or against the
    window's own) is recomputed in two passes. S, S2 (length L + 1) and mean,
    std, flag (one per window) are caller scratch.

    It is _chain_part (what depends on the chain only) followed by
    _window_part, which the chain-major kernel calls separately to build a
    chain once and read it at several window lengths."""
    cm = _chain_part(c, L, S, S2)
    return _window_part(c, L, w, cm, S, S2, mean, std, flag)


@njit(inline="always")
def _chain_part(c, L, S, S2):
    """The chain mean cm and the prefix sums S, S2 of the chain centred on
    it (S[0] = S2[0] = 0); returns cm."""
    cm = 0.0
    for t in range(L):
        cm += c[t]
    cm /= L
    S[0] = 0.0
    S2[0] = 0.0
    for t in range(L):
        v = c[t] - cm
        S[t + 1] = S[t] + v
        S2[t + 1] = S2[t] + v * v
    return cm


@njit(inline="always")
def _window_part(c, L, w, cm, S, S2, mean, std, flag):
    """Mean, std and the suspect windows' two-pass recompute for the windows
    of length w of a built chain (cm, S, S2 are only read); returns the
    window count. A window is suspect when its one-pass variance is at most
    RESCUE_CHAIN times the chain's mean energy S2[L] / L or at most
    RESCUE_REL times the window's own mean energy; flag marks them."""
    n_w = L - w + 1
    inv_w = 1.0 / w
    thr_c = RESCUE_CHAIN * (S2[L] / L)
    n_bad = 0
    for t in range(n_w):
        s = S[t + w] - S[t]
        q = S2[t + w] - S2[t]
        mu = s * inv_w
        e = q * inv_w
        mean[t] = cm + mu
        vv = max(e - mu * mu, 0.0)
        bad = np.int64((vv <= thr_c) | (vv <= RESCUE_REL * e))
        flag[t] = bad
        n_bad += bad
        std[t] = np.sqrt(vv)
    if n_bad > 0:
        for t in range(n_w):
            if flag[t] != 0:
                tot = 0.0
                for k in range(w):
                    tot += c[t + k]
                mm = tot / w
                dev = 0.0
                for k in range(w):
                    dv = c[t + k] - mm
                    dev += dv * dv
                mean[t] = mm
                std[t] = np.sqrt(dev / w)
    return n_w


@njit(cache=True, nogil=True)
def _fill_trig(windows, cos_tab, sin_tab):
    """cos_tab[s, k, t] = cos(2 pi (k+1) (t mod w_s) / w_s), exact at quarter
    turns (so the Nyquist part of window 8 is exactly 0), tiled over the
    series length; k = 0..N_COEF - 1, one table row per window length
    windows[s]."""
    m = cos_tab.shape[2]
    two_pi = 2.0 * np.pi
    for s in range(windows.shape[0]):
        w = windows[s]
        for k in range(N_COEF):
            for u in range(w):
                r = ((k + 1) * u) % w
                if 4 * r == 0:
                    cs = 1.0
                    sn = 0.0
                elif 4 * r == w:
                    cs = 0.0
                    sn = 1.0
                elif 2 * r == w:
                    cs = -1.0
                    sn = 0.0
                elif 4 * r == 3 * w:
                    cs = 0.0
                    sn = -1.0
                else:
                    ang = two_pi * r / w
                    cs = np.cos(ang)
                    sn = np.sin(ang)
                for t in range(u, m, w):
                    cos_tab[s, k, t] = cs
                    sin_tab[s, k, t] = sn
