"""Reference key scans in numpy that follow the kernels' arithmetic.

Sums are sequential (np.add.accumulate, never numpy's pairwise sum) and the
trigonometric tables are the kernel's own, so keys equal the kernels' bit
for bit."""

import numpy as np

from ..constants import (EPS_STD, KEYSPACE, LEVEL_CUTS, N_COEF, RESCUE_CHAIN,
                         RESCUE_REL, WORD_LEN)
from ..kernels.window import _fill_trig


ALPHABET = 2


def channel_series(S, is_diff):
    """(n, m) series of one input channel, or its first difference with the
    last increment repeated (the kernel's gather): D[j] = x[j+1] - x[j] for
    j < m - 1, D[m-1] = x[m-1] - x[m-2]."""
    S = np.asarray(S, dtype=np.float64)
    if not is_diff:
        return S
    D = np.empty_like(S)
    D[:, :-1] = S[:, 1:] - S[:, :-1]
    D[:, -1] = S[:, -1] - S[:, -2]
    return D


def trig_tables(window, m):
    """The kernel's (cos, sin) tables of one window, each (N_COEF, m), built by
    kernels.window._fill_trig so the reference uses the same values."""
    cos_t = np.zeros((1, N_COEF, m))
    sin_t = np.zeros((1, N_COEF, m))
    _fill_trig(np.asarray([int(window)], dtype=np.int64), cos_t, sin_t)
    return cos_t[0], sin_t[0]


def _chain_moments(ch, Ls, w):
    """Window means and stds of every chain, as kernels.window._moments
    computes them, vectorised over series and chains. ch: (n, R, Lmax)
    chains, zero-padded past each chain's length Ls[r]. Returns (mean, std,
    valid), each (n, R, nwm) with nwm = Lmax - w + 1 (valid is (R, nwm):
    valid[r, t] = t < Ls[r] - w + 1). Sums accumulate sequentially, as in the
    kernel; windows of very small variance are recomputed in two passes."""
    n, R, Lmax = ch.shape
    rr = np.arange(R)
    acc = np.add.accumulate(ch, axis=2)               # sequential sums
    cm = acc[:, rr, Ls - 1] / Ls                      # (n, R)
    v = ch - cm[:, :, None]
    S = np.zeros((n, R, Lmax + 1))
    S2 = np.zeros((n, R, Lmax + 1))
    np.add.accumulate(v, axis=2, out=S[:, :, 1:])
    np.add.accumulate(v * v, axis=2, out=S2[:, :, 1:])
    nwm = Lmax - w + 1
    inv_w = 1.0 / w
    thr_c = RESCUE_CHAIN * (S2[:, rr, Ls] / Ls)    # (n, R)
    s = S[:, :, w:w + nwm] - S[:, :, :nwm]
    q = S2[:, :, w:w + nwm] - S2[:, :, :nwm]
    mu = s * inv_w
    e = q * inv_w
    mean = cm[:, :, None] + mu
    vv = np.maximum(e - mu * mu, 0.0)
    valid = np.arange(nwm)[None, :] < (Ls - w + 1)[:, None]      # (R, nwm)
    bad = ((vv <= thr_c[:, :, None]) | (vv <= RESCUE_REL * e)) \
        & valid[None, :, :]
    std = np.sqrt(vv)
    if bad.any():
        # the two-pass recompute of the suspect windows, sequential sums
        bi, br, bt = np.nonzero(bad)
        W = ch[bi[:, None], br[:, None], bt[:, None] + np.arange(w)[None, :]]
        tot = np.add.accumulate(W, axis=1)[:, -1]
        mm = tot / w
        dv = W - mm[:, None]
        dev = np.add.accumulate(dv * dv, axis=1)[:, -1]
        mean[bi, br, bt] = mm
        std[bi, br, bt] = np.sqrt(dev / w)
    return mean, std, valid


def _chains(S, w, d):
    """The kernel's residue chains of (n, m) series S at dilation d: (ch,
    Ls), ch (n, R, Lmax) zero-padded past each chain's length Ls[r], R =
    min(d, n_windows) the chains holding a window (all of length >= w)."""
    n, m = S.shape
    n_windows = m - (w - 1) * d
    Lmax = -(-m // d)
    R = min(d, n_windows)
    pad = np.zeros((n, Lmax * d))
    pad[:, :m] = S
    ch = np.ascontiguousarray(pad.reshape(n, Lmax, d).transpose(0, 2, 1)
                              [:, :R, :])
    return ch, (m - np.arange(R) + d - 1) // d


def _window_order(K, m, w, d):
    """Per-chain values K (n, R, nwm) in window-start order (n, n_windows):
    start j = t d + r0 is chain r0's t-th window."""
    n, R = K.shape[0], K.shape[1]
    n_windows = m - (w - 1) * d
    if R < d:                       # n_windows < d: one window per chain
        return np.ascontiguousarray(K[:, :, 0])
    return np.ascontiguousarray(
        K.transpose(0, 2, 1).reshape(n, -1)[:, :n_windows])


def _paa_chain_keys(ch, w, std):
    """PAA keys as kernels.words_univariate._paa_keys: bit b = (sum of segment
    b) > an eighth of the window's segment total, most significant first;
    flat windows key 0. The total adds the eight segment sums in segment
    order (elementwise IEEE additions, the kernel's scalar chain) and the
    eighth is a multiplication by 0.125, which is exact."""
    paa = w // WORD_LEN
    Lmax = ch.shape[2]
    nwm = std.shape[2]
    npm = Lmax - paa + 1
    pool = ch[:, :, :npm].copy()
    for k in range(1, paa):
        pool += ch[:, :, k:k + npm]
    tot = pool[:, :, 0:nwm].copy()
    for b in range(1, WORD_LEN):
        tot = tot + pool[:, :, b * paa:b * paa + nwm]
    th = tot * 0.125
    key = np.zeros(std.shape, dtype=np.int64)
    for b in range(WORD_LEN):
        key = key * 2 + (pool[:, :, b * paa:b * paa + nwm] > th)
    key[std <= EPS_STD] = 0
    return key


def _fourier_chain_keys(ch, w, std, cuts, flat_key, cos_w, sin_w):
    """Fourier keys as kernels.words_univariate._fourier_keys: modulated prefix
    sums over the chain, rotated back by the window start's table entry,
    scaled by 1 / (w std), parts re_1, im_1, ..., re_4, im_4 against their
    cuts (most significant first); flat windows take flat_key."""
    n, R, Lmax = ch.shape
    nwm = std.shape[2]
    cuts = np.asarray(cuts, dtype=np.float64)
    key = np.zeros(std.shape, dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        isc = 1.0 / (w * std)
        for k in range(N_COEF):
            ck = cos_w[k, :Lmax]
            sk = sin_w[k, :Lmax]
            mr = np.zeros((n, R, Lmax + 1))
            mi = np.zeros((n, R, Lmax + 1))
            np.add.accumulate(ch * ck, axis=2, out=mr[:, :, 1:])
            # the kernel's b -= c sin: accumulating -(c sin) is the same sum
            # (x - y == x + (-y) in IEEE arithmetic; signed zeros aside,
            # which no comparison below can see)
            np.add.accumulate(-(ch * sk), axis=2, out=mi[:, :, 1:])
            a = mr[:, :, w:w + nwm] - mr[:, :, :nwm]
            b = mi[:, :, w:w + nwm] - mi[:, :, :nwm]
            cr = ck[:nwm]
            sr = sk[:nwm]
            re = (a * cr - b * sr) * isc > cuts[2 * k]
            im = (a * sr + b * cr) * isc > cuts[2 * k + 1]
            key = key * 4 + re * 2 + im
    key[std <= EPS_STD] = int(flat_key)
    return key


def reference_keys(S, kind, window, dilation, cuts=None, flat_key=0,
                   trig=None):
    """Key of every window of every series (window-start order, no
    numerosity reduction), in the kernel's arithmetic: (n, n_windows) int64,
    n_windows = m - (window - 1) x dilation (0 columns when the window does
    not fit). S: (n, m) float64 series of the pseudo-channel (already
    differenced for a difference channel, see ``channel_series``); kind
    "paa" | "fourier"; cuts (8,) and flat_key for "fourier"; trig = (cos,
    sin) of ``trig_tables(window, m)`` (computed when None)."""
    S = np.asarray(S, dtype=np.float64)
    if S.ndim == 1:
        S = S[None, :]
    n, m = S.shape
    w, d = int(window), int(dilation)
    if m - (w - 1) * d <= 0:
        return np.empty((n, 0), dtype=np.int64)
    ch, Ls = _chains(S, w, d)
    mean, std, _ = _chain_moments(ch, Ls, w)
    if kind == "paa":
        K = _paa_chain_keys(ch, w, std)
    elif kind == "fourier":
        if trig is None:
            trig = trig_tables(w, m)
        K = _fourier_chain_keys(ch, w, std, cuts, flat_key, trig[0], trig[1])
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return _window_order(K, m, w, d)


def reference_level_keys(series, window, dilation, cut_rows):
    """The level word of every window (window-start order, no numerosity
    reduction) in the arithmetic of kernels.level on the moments of
    kernels.window._moments. Returns (n, n_windows) int64 (0 columns when the
    window does not fit).

    series: the k (n, m) float64 series of a family's channels in order, each
    already differenced on the difference side (see ``channel_series``);
    k = 1 for a univariate view. cut_rows: their (30,) cut sets (15 mean
    cuts, then 15 std cuts, each sorted). A channel's part takes b = 8 // k
    bits: its mean bin's top ceil(b / 2) bits above its std bin's top
    floor(b / 2), a bin being the number of cuts strictly below the value;
    channel 0 is most significant. A window whose standard deviation is at
    most EPS_STD is flat and takes standard deviation 0, as in
    kernels.level._level_part."""
    k = len(series)
    b = WORD_LEN // k
    hm, hs = (b + 1) // 2, b // 2
    w, d = int(window), int(dilation)
    key = None
    for S, cuts in zip(series, cut_rows):
        S = np.asarray(S, dtype=np.float64)
        if S.ndim == 1:
            S = S[None, :]
        n, m = S.shape
        if m - (w - 1) * d <= 0:
            return np.empty((n, 0), dtype=np.int64)
        ch, Ls = _chains(S, w, d)
        mean, std, _ = _chain_moments(ch, Ls, w)
        cuts = np.asarray(cuts, dtype=np.float64)
        mb = np.searchsorted(cuts[:LEVEL_CUTS], _window_order(mean, m, w, d),
                             side="left")
        sd = _window_order(std, m, w, d)
        sb = np.searchsorted(cuts[LEVEL_CUTS:],
                             np.where(sd <= EPS_STD, 0.0, sd), side="left")
        part = ((mb >> (4 - hm)) << hs) | (sb >> (4 - hs))
        key = part if key is None else (key << b) | part
    return key.astype(np.int64)


def nr_counts(keys, keyspace=KEYSPACE, nr=True):
    """The model's bag of one (view, dilation): counts per key, (n, keyspace)
    int64. With ``nr`` a window whose key equals the previous window start's
    is dropped first (numerosity reduction); otherwise every window counts,
    the model's default."""
    keys = np.asarray(keys, dtype=np.int64)
    n, nw = keys.shape
    out = np.zeros((n, keyspace), dtype=np.int64)
    if nw == 0:
        return out
    kept = np.ones((n, nw), dtype=bool)
    if nr:
        kept[:, 1:] = keys[:, 1:] != keys[:, :-1]
    flat = (np.arange(n)[:, None] * keyspace + keys)[kept]
    out.ravel()[:] = np.bincount(flat, minlength=n * keyspace)
    return out


class ReferenceView:
    """One flat view of a fitted model (a dictionary: source, window, input
    channel, series or difference, its dilations), with the reference scan.
    ``keys_one(x, d)``: keys of one series (x (m,) for a univariate model,
    (C, m) otherwise); ``keys_many(X, d)``: of a collection ((n, m) or
    (n, C, m)). ``level_keys_one`` / ``level_keys_many``: the level word of
    every window instead (k = 1: the mean bin above the std bin among the
    view's cut set ``level_cuts``, (30,); None for a model fitted without
    the level word, which then raises)."""

    def __init__(self, kind, window, channel, is_diff, n_channels, cuts,
                 flat_key, level_cuts=None):
        self.kind = kind
        self.window = int(window)
        self.input_channel = int(channel)
        self.is_diff = bool(is_diff)
        self.n_channels = int(n_channels)
        self.cuts = np.asarray(cuts, dtype=np.float64).copy()
        self.flat_key = int(flat_key)
        self.level_cuts = (None if level_cuts is None else
                           np.asarray(level_cuts, dtype=np.float64).copy())
        self.word_len = WORD_LEN
        self.alphabet = ALPHABET
        self.sub = (self.window // WORD_LEN if kind == "paa" else N_COEF)
        self._trig = {}

    def _series(self, X3):
        if X3.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got "
                             f"{X3.shape[1]}")
        return channel_series(X3[:, self.input_channel, :], self.is_diff)

    def _tables(self, m):
        if self.kind != "fourier":
            return None
        t = self._trig.get(m)
        if t is None:
            t = self._trig[m] = trig_tables(self.window, m)
        return t

    def _many(self, X):
        """The (n, m) series of this view's pseudo-channel in a collection."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            if self.n_channels != 1:
                raise ValueError("a 2-D collection is univariate; this model "
                                 f"has {self.n_channels} channels")
            X = X[:, None, :]
        if X.ndim != 3:
            raise ValueError(f"X must be (n, m) or (n, C, m), got {X.shape}")
        return self._series(X)

    def _one(self, x):
        """The (1, m) series of this view's pseudo-channel in one series."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            if self.n_channels != 1:
                raise ValueError("a 1-D series is univariate; this model has "
                                 f"{self.n_channels} channels")
            X3 = x[None, None, :]
        elif x.ndim == 2:
            X3 = x[None, :, :]
        else:
            raise ValueError(f"x must be (m,) or (C, m), got {x.shape}")
        return self._series(X3)

    def _keys(self, S, dilation):
        return reference_keys(S, self.kind, self.window, dilation, self.cuts,
                              self.flat_key, self._tables(S.shape[1]))

    def _level_keys(self, S, dilation):
        if self.level_cuts is None:
            raise ValueError("this model was fitted without the level word")
        return reference_level_keys([S], self.window, dilation,
                                    [self.level_cuts])

    def keys_many(self, X, dilation):
        return self._keys(self._many(X), dilation)

    def keys_one(self, x, dilation):
        return self._keys(self._one(x), dilation)[0]

    def level_keys_many(self, X, dilation):
        return self._level_keys(self._many(X), dilation)

    def level_keys_one(self, x, dilation):
        return self._level_keys(self._one(x), dilation)[0]
