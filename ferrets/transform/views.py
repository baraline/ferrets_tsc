"""The random draw of views (C = 1, per-channel) and families (C >= 2)."""

import numpy as np

from ..constants import _CROSS, _FAMILY_KS, _SOURCES, _SUBSET_SALT
from ..helpers.validation import _check_windows
from ..kernels.rng import _draw_views_nb

# the switches' public values and their internal forms
CROSS_CHANNEL_WORDS = {"all": _CROSS["all"], "coarse": _CROSS["coarse"],
                       "flags": _CROSS["flags"], "none": _CROSS["none"]}
CHANNEL_LAYOUTS = ("family", "per_channel")
WORD_SOURCES = (("paa", "fourier"), ("paa",), ("fourier",))
DILATIONS = ("log_uniform", "one")


def _draw_views_py(seed, P, m, windows, n_dictionaries, sources):
    """The per-pseudo-channel draw in pure Python, for any subset of sources.

    Returns, per pseudo-channel p, a tuple of (source, window, dilations)
    sorted by (source, window), dilations ascending. With both sources it
    equals the numba draw; with one source every draw comes from it
    (randint(1) consumes no random state, so the stream is the two-source
    stream without the source draw)."""
    fit = [int(w) for w in windows if int(w) <= m]
    if not fit:
        raise ValueError(f"no window fits series of length {m} (windows "
                         f"{tuple(int(w) for w in windows)})")
    windows = fit
    out = []
    for p in range(P):
        rng = np.random.RandomState((int(seed) * 7919 + p) % (1 << 32))
        draws = {}
        for _ in range(int(n_dictionaries)):
            source = sources[rng.randint(len(sources))]
            # every window is a multiple of the word length, so the
            # PAA pool (windows divisible by 8) is the whole list
            window = windows[rng.randint(len(windows))]
            max_d = max(1, (m - 1) // (window - 1))
            d = int(2 ** rng.uniform(0.0, np.log2(max_d + 1)))
            d = max(1, min(d, max_d))
            draws.setdefault((source, window), set()).add(d)
        out.append(tuple((s, w, tuple(sorted(ds)))
                         for (s, w), ds in sorted(draws.items())))
    return out


def _dilation_one(views):
    """``views`` with every view's dilations replaced by (1,).

    A view is (source, window, dilations), or (source, window, dilations,
    channels) for a family. The views of one list differ in (source,
    window[, channels]) already, so nothing merges."""
    return [tuple(v[:2] + ((1,),) + tuple(v[3:]) for v in vp) for vp in views]


def _draw_families(seed, C, m, windows, n_views, sources=_SOURCES, sides=2):
    """The family layout's draw, one tuple of families per side.

    Per side p < sides (0 the series, 1 the first difference), n_views
    times: a source, a window and one log-uniform dilation from
    RandomState(seed * 7919 + p) (the univariate draw's statements), and k
    channels without replacement (k in _FAMILY_KS, at most C; 1 at C = 1)
    from RandomState(seed * 7919 + _SUBSET_SALT + p). Families sharing
    (source, window, channels) merge their dilations. A family is
    (source, window, dilations, channels), sorted by (source, window,
    channels), channels sorted."""
    fit = [int(w) for w in windows if int(w) <= m]
    if not fit:
        raise ValueError(f"no window fits series of length {m} (windows "
                         f"{tuple(int(w) for w in windows)})")
    ks = [k for k in _FAMILY_KS if k <= int(C)] or [1]
    out = []
    for p in range(sides):
        rng = np.random.RandomState((int(seed) * 7919 + p) % (1 << 32))
        srng = np.random.RandomState(
            (int(seed) * 7919 + _SUBSET_SALT + p) % (1 << 32))
        draws = {}
        for _ in range(int(n_views)):
            source = sources[rng.randint(len(sources))]
            window = fit[rng.randint(len(fit))]
            max_d = max(1, (m - 1) // (window - 1))
            d = int(2 ** rng.uniform(0.0, np.log2(max_d + 1)))
            d = max(1, min(d, max_d))
            k = ks[srng.randint(len(ks))]
            chs = tuple(sorted(int(c) for c in
                               srng.choice(int(C), k, replace=False)))
            draws.setdefault((source, window, chs), set()).add(d)
        out.append(tuple((s, w, tuple(sorted(ds)), chs)
                         for (s, w, chs), ds in sorted(draws.items())))
    return out


class ViewsMixin:
    """The seed, the layout hooks and the random draw.

    Reads: seed_."""

    def _resolve_seed(self):
        """The fit's seed, in [0, 2**32): numpy's RandomState and
        train_test_split accept exactly that range, and the view draw and the
        sampler reduce modulo 2**32 anyway."""
        rs = self.random_state
        if isinstance(rs, (int, np.integer)):
            return int(rs) % (1 << 32)
        rng = rs if isinstance(rs, np.random.RandomState) else \
            np.random.RandomState()
        return int(rng.randint(0, 1 << 32, dtype=np.int64))

    def _uses_families(self, C):
        """Whether a C-channel input takes the family layout: C >= 2 under
        channel_layout="family" (under "per_channel" every channel and its
        difference is a pseudo-channel of its own). It chooses the draw;
        everything after the draw follows the views' format."""
        return int(C) >= 2 and self.channel_layout == "family"

    def _cross_code(self):
        """The cross-channel words as the kernel's code (bit 0 the coarse
        word, bit 1 the change and agreement flags). fit stores it in the
        plan; predict reads the plan."""
        return CROSS_CHANNEL_WORDS[self.cross_channel_words]

    def _n_pseudo(self, C):
        """The pseudo-channel count: the sides (the series, and its first
        difference unless use_first_difference is False) in the family
        layout, else every channel of every side (p < C the series)."""
        sides = 2 if self.use_first_difference else 1
        return sides if self._uses_families(C) else sides * C

    def _draw_views(self, P, m):
        """The random draw. C >= 2 under the family layout: per side the
        family draw (``_draw_families``). Otherwise per pseudo-channel: R
        times source, window and one log-uniform dilation from
        RandomState(seed * 7919 + p), by the numba replay with both word
        sources and by the Python draw with one. Duplicates collapse; sorted
        by (source, window). dilation="one" sets every view's dilations to
        (1,) after the draw. An unfitted model (no ``n_channels_``) draws
        per pseudo-channel."""
        C = getattr(self, "n_channels_", 1)
        windows = _check_windows(self.windows)
        sources = tuple(str(s) for s in self.word_sources)
        if self._uses_families(C):
            views = _draw_families(self.seed_, C, m, windows,
                                   self.n_dictionaries, sources=sources,
                                   sides=P)
        elif sources == _SOURCES:
            views = self._draw_univariate(P, m)
        else:
            views = _draw_views_py(self.seed_, P, m, windows,
                                   self.n_dictionaries, sources)
        return views if self.dilation == "log_uniform" else \
            _dilation_one(views)

    def _draw_univariate(self, P, m):
        """The random draw with both word sources, per pseudo-channel.

        R times source, window and one log-uniform dilation from
        RandomState(seed * 7919 + p): the stream replayed by _draw_views_nb,
        with d = int(2 ** u) evaluated in Python so it matches
        _draw_views_py."""
        windows = [w for w in _check_windows(self.windows) if w <= m]
        if not windows:
            raise ValueError(f"no window fits series of length {m} (windows "
                             f"{tuple(self.windows)})")
        seed = self.seed_
        R = int(self.n_dictionaries)
        max_ds = [max(1, (m - 1) // (w - 1)) for w in windows]
        hi = np.array([float(np.log2(md + 1)) for md in max_ds])
        seeds = np.array([(seed * 7919 + p) % (1 << 32) for p in range(P)],
                         dtype=np.int64)
        src = np.empty((P, R), dtype=np.int64)
        widx = np.empty((P, R), dtype=np.int64)
        u = np.empty((P, R), dtype=np.float64)
        _draw_views_nb(seeds, np.int64(R), np.int64(len(windows)), hi, src,
                       widx, u)
        ds = np.array([int(2 ** x) for x in u.ravel().tolist()],
                      dtype=np.int64).reshape(P, R)
        md = np.asarray(max_ds, dtype=np.int64)[widx]
        ds = np.maximum(1, np.minimum(ds, md))
        wv = np.asarray(windows, dtype=np.int64)[widx]
        # sort key = the order of _draw_views_py: "fourier" (src 1) before
        # "paa" (src 0), then the window, then the dilations ascending
        big_d = int(md.max()) + 2
        big_w = int(wv.max()) + 1
        code = ((np.arange(P, dtype=np.int64)[:, None] * 2 + (1 - src))
                * big_w + wv) * big_d + ds
        uq = np.unique(code.ravel())
        dd = uq % big_d
        rest = uq // big_d
        ww = rest % big_w
        rest = rest // big_w
        so = rest % 2
        pp = rest // 2
        out = [[] for _ in range(P)]
        cur = None
        for p_, s_, w_, d_ in zip(pp.tolist(), so.tolist(), ww.tolist(),
                                  dd.tolist()):
            k = (p_, s_, w_)
            if k != cur:
                out[p_].append(["fourier" if s_ == 0 else "paa", w_, []])
                cur = k
            out[p_][-1][2].append(d_)
        return [tuple((a_, b_, tuple(c_)) for a_, b_, c_ in vp) for vp in out]
