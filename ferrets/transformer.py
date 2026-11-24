"""FERRETS: the label-free transform of series into bags of exact words."""

import time

import numpy as np
from aeon.transformations.collection.base import BaseCollectionTransformer

from .constants import _FAMILY_CUTS, WORD_LEN, head_col_none
from .helpers.threads import _resolve_threads, _threads
from .helpers.validation import (_as_collection, _check_windows, _check_X,
                                 _is_int)
from .transform.chunks import _stream_chunks
from .transform.cuts import _fit_tables
from .transform.encoding import _tf_tables
from .transform.features import FeaturesMixin
from .transform.plan import _is_family, _level_jobs, PlanMixin
from .transform.run import TransformMixin
from .transform.views import (CHANNEL_LAYOUTS, CROSS_CHANNEL_WORDS, DILATIONS,
                              WORD_SOURCES, ViewsMixin)

__all__ = ["FERRETS"]


# tables a fit builds for its counting passes and drops when it ends
_FIT_SCRATCH = ("_tf_store", "_tf_tab", "_id_bt", "_cnt_bt", "_tbuf",
                "_chain_cache")


class FERRETS(ViewsMixin, PlanMixin, TransformMixin, FeaturesMixin,
              BaseCollectionTransformer):
    """Fast Exact Random Radix Encodings of Time Series.

    Turns each series into a sparse bag of exact 8-bit words. Random
    dictionaries, each a (word source, window length, dilation) triple, key
    every window of the series and of its first difference into one of 256
    words, a binary PAA word or a binary Fourier word; a level word adds the
    bins of the window's mean and standard deviation. A word is a column of
    the output by arithmetic: no hash map, and the width is bounded in advance.

    Rows hold sublinear term frequencies at unit l2 norm; columns that fewer
    than ``min_doc_freq`` training series reach are dropped. Multivariate input
    keys random channel subsets, plus cross-channel words. No labels are used.

    Parameters
    ----------
    n_dictionaries : int, default=100
        Draws of (word source, window, dilation) per series and per first
        difference (univariate) or of channel families per side (multivariate).
        Repeated draws collapse, so this bounds the number of dictionaries.
    windows : tuple of int, default=(8, 16, 24, 32, 48, 64)
        Window lengths to draw from, multiples of 8. Windows longer than the
        series are skipped.
    word_sources : tuple of str, default=("paa", "fourier")
        Word sources to draw from: ("paa", "fourier"), ("paa",) or
        ("fourier",).
    use_first_difference : bool, default=True
        Also key the first difference of each channel.
    dilation : {"log_uniform", "one"}, default="log_uniform"
        Draw each dictionary's dilation log-uniformly, or use dilation 1.
    level_word : bool, default=True
        Also key each window's level word: the bins of its mean and standard
        deviation among 15 training quantile cuts each. A flat window takes
        standard deviation 0.
    cross_channel_words : {"all", "coarse", "flags", "none"}, default="all"
        Multivariate input, "family" layout: the cross-channel words keyed
        beside the channel-wise words (a coarse word, change flags and
        agreement flags). Must be "all" under the "per_channel" layout.
    channel_layout : {"family", "per_channel"}, default="family"
        Multivariate input: families of 2, 4 or 8 channels keyed under one
        dictionary ("family"), or each channel keyed alone ("per_channel").
    min_doc_freq : int, default=1
        Keep a column that at least this many training series contain.
    use_idf : bool, default=False
        Weight columns by the smooth idf, ln((1 + n) / (1 + df)) + 1.
    numerosity_reduction : bool, default=False
        Drop a window whose key equals that of the window before it.
    sublinear_tf : bool, default=True
        Weight a count c by 1 + ln(c) instead of c.
    n_jobs : int, default=1
        Threads used by fit and transform: None or an int >= 1; -1 uses
        every numba thread.
    random_state : int, RandomState or None, default=0
        Seed of the random draws. An int is reduced modulo 2**32; None draws
        fresh entropy.

    Attributes
    ----------
    n_channels_ : int
        Number of channels of the training series.
    n_timepoints_ : int
        Length of the training series.
    seed_ : int
        The seed of the draws, resolved from ``random_state`` at fit.
    views_ : list
        The drawn dictionaries, one tuple per pseudo-channel: the series and
        its first difference (univariate), each channel and its difference
        ("per_channel"), or each side ("family").
    fourier_cuts_ : list
        The cut tables of the Fourier words, 8 cuts per word: one
        (n_fourier_views, 8) array per pseudo-channel (univariate and
        "per_channel"), or one (64,) array per Fourier view ("family": 8
        cuts per channel slot, the rest zero).
    level_cuts_ : np.ndarray of shape (n_level_sets, 30)
        Per level-word cut set, 15 mean cuts then 15 standard deviation cuts.
    n_features_ : int
        Number of kept columns, the width of the output.
    feature_offsets_ : np.ndarray
        First kept column of each pseudo-channel's (or side's) span, then
        ``n_features_``.
    idf_ : np.ndarray of shape (n_features_,)
        Column weights, all 1 without ``use_idf``.

    See Also
    --------
    FERRETSClassifier : A ridge or streaming head on the same transform.

    Notes
    -----
    The transform is described in docs/model.md. ``fit_transform`` holds every
    encoded block of the training set (about 2 bytes per encoded entry) beside
    the output; ``fit`` followed by ``transform`` keeps the transform's memory
    bound.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from ferrets import FERRETS
    >>> X, _ = make_example_3d_numpy(n_cases=10, n_timepoints=64,
    ...                              random_state=0)
    >>> Xt = FERRETS(n_dictionaries=10).fit_transform(X)
    >>> Xt.shape[0]
    10
    """

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "dictionary",
    }

    _check_X = _check_X     # aeon's input check without its variance floor

    def __init__(self, n_dictionaries=100, windows=(8, 16, 24, 32, 48, 64),
                 word_sources=("paa", "fourier"), use_first_difference=True,
                 dilation="log_uniform", level_word=True,
                 cross_channel_words="all", channel_layout="family",
                 min_doc_freq=1, use_idf=False, numerosity_reduction=False,
                 sublinear_tf=True, n_jobs=1, random_state=0):
        self.n_dictionaries = n_dictionaries
        self.windows = windows
        self.word_sources = word_sources
        self.use_first_difference = use_first_difference
        self.dilation = dilation
        self.level_word = level_word
        self.cross_channel_words = cross_channel_words
        self.channel_layout = channel_layout
        self.min_doc_freq = min_doc_freq
        self.use_idf = use_idf
        self.numerosity_reduction = numerosity_reduction
        self.sublinear_tf = sublinear_tf
        self.n_jobs = n_jobs
        self.random_state = random_state
        super().__init__()

    def _check_params(self):
        for name in ("n_dictionaries", "min_doc_freq"):
            v = getattr(self, name)
            if not _is_int(v) or v < 1:
                raise ValueError(f"{name} must be an int >= 1, got {v!r}")
        _check_windows(self.windows)
        for name in ("use_idf", "numerosity_reduction",
                     "use_first_difference", "sublinear_tf", "level_word"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be a bool")
        _resolve_threads(self.n_jobs)
        rs = self.random_state
        if not (rs is None or isinstance(rs, np.random.RandomState)
                or _is_int(rs)):
            raise ValueError(f"random_state must be an int, None or a "
                             f"RandomState, got {rs!r}")
        v = self.cross_channel_words
        if not (isinstance(v, str) and v in CROSS_CHANNEL_WORDS):
            raise ValueError(f"cross_channel_words must be one of "
                             f"{tuple(CROSS_CHANNEL_WORDS)}, got {v!r}")
        v = self.channel_layout
        if not (isinstance(v, str) and v in CHANNEL_LAYOUTS):
            raise ValueError(f"channel_layout must be one of "
                             f"{CHANNEL_LAYOUTS}, got {v!r}")
        if v == "per_channel" and self.cross_channel_words != "all":
            raise ValueError("cross_channel_words selects the family "
                             "layout's cross-channel words; "
                             "channel_layout='per_channel' has none")
        try:
            src = tuple(str(s) for s in self.word_sources)
        except TypeError:
            src = None
        if src not in WORD_SOURCES:
            raise ValueError(f"word_sources must be one of "
                             f"{WORD_SOURCES}, got "
                             f"{self.word_sources!r}")
        v = self.dilation
        if not (isinstance(v, str) and v in DILATIONS):
            raise ValueError(f"dilation must be one of {DILATIONS}, got "
                             f"{v!r}")

    def _fit(self, X, y=None):
        """Draw the dictionaries and cut tables, then run one counting pass
        over the training series that fixes the kept columns and the idf."""
        self._n_jobs = _resolve_threads(self.n_jobs)
        with _threads(self._n_jobs):
            with self._fit_design(_as_collection(X)):
                pass
        return self

    def _fit_transform(self, X, y=None):
        """``_fit`` then ``_transform`` of the same series, keying them once:
        the rows are decoded from the counting pass's encoded design."""
        self._n_jobs = _resolve_threads(self.n_jobs)
        with _threads(self._n_jobs):
            with self._fit_design(_as_collection(X), keep_all=True) as design:
                return self._design_rows(design)

    def _fit_design(self, X, stats=None, keep_all=False, ram_mb=0.0,
                    metadata=None):
        """Fit the transform on the training collection X (validated float64)
        and return the encoded training design the batch heads learn from:
        the dictionaries, cuts and plan, then the counting pass that fixes
        the kept columns and the idf. The encoded blocks stay in RAM up to
        ram_mb (all of them when keep_all); the design encodes the others
        again on demand. ``stats`` gets the fit's timings and sizes.

        The design is a context manager: leaving its block drops the fit-time
        tables and, when no error left it and ``metadata`` (the data
        description of the classifier driving the fit) is given, declares the
        transformer fitted. An error before the return drops the tables
        too. The design is used only as ``with ... as design`` (the exit
        ends the fit)."""
        tt = time.perf_counter
        stats = {} if stats is None else stats
        try:
            trig = self._fit_dictionaries(X, stats)
            t0 = tt()
            design = self._batch_design(X, trig, stats, keep_all=keep_all,
                                        ram_mb=ram_mb)
            stats["t_pass1"] = tt() - t0
            stats["nnz"] = design.nnz
        except BaseException:
            self._drop_scratch()
            raise
        design.metadata = metadata
        return design

    def _fit_chunks(self, X, stats=None, metadata=None):
        """Fit the transform on the training collection X (validated float64)
        and return its encoded training set as chunks for the streaming head:
        the dictionaries, cuts and plan, then one pass that encodes every
        chunk into a cache (in RAM, on disk, or none: the chunks are then
        encoded again on every visit), counts the document frequencies and
        fixes the kept columns and the idf. The pruned entries, at most one
        per pruned column, ride along in the cache and are dropped at decode.
        ``stats`` gets the fit's timings and sizes.

        The chunks are a context manager like the design of ``_fit_design``:
        leaving the block removes the cache and the fit-time tables and,
        when no error left it and ``metadata`` is given, declares the
        transformer fitted. An error before the return does the same. The
        chunks are used only as ``with ... as chunks`` (the exit ends the
        fit)."""
        stats = {} if stats is None else stats
        try:
            trig = self._fit_dictionaries(X, stats)
            chunks = _stream_chunks(self, X, trig, stats)
        except BaseException:
            self._drop_scratch()
            raise
        chunks.metadata = metadata
        return chunks

    def _drop_scratch(self):
        """Remove the fit-time tables (``_FIT_SCRATCH``)."""
        for name in _FIT_SCRATCH:
            self.__dict__.pop(name, None)

    def _fit_dictionaries(self, X, stats=None):
        """Everything before the first counting pass: the seed, the views,
        the cut tables, the plan and the fit-time tables. Sets seed_,
        n_channels_, n_timepoints_, views_, fourier_cuts_, level_cuts_, _plan
        and _head_col (no column kept yet). Returns the trigonometric
        tables."""
        tt = time.perf_counter
        t0 = tt()
        stats = {} if stats is None else stats
        self._check_params()
        _, C, m = X.shape
        if C == 0:
            raise ValueError("X has no channels")
        if m < WORD_LEN:
            raise ValueError(f"series length {m} is shorter than the smallest "
                             f"window ({WORD_LEN})")
        self.seed_ = self._resolve_seed()
        self.n_channels_ = int(C)          # _draw_views reads it
        self.n_timepoints_ = int(m)
        P = self._n_pseudo(C)
        views = self._draw_views(P, m)
        self.views_ = views
        stats["t_draw"] = tt() - t0
        t1 = tt()
        self._fit_cuts_and_plan(X, views)
        stats["t_sampler"] = tt() - t1
        t2 = tt()
        trig = self._prepare_tables(m)
        stats["t_prepare"] = tt() - t2
        return trig

    def _fit_cuts_and_plan(self, X, views):
        """Fourier cut tables and level cut sets (computed together where
        they share work; _fit_tables adds the tie margins), then the plan.
        Sets level_cuts_, fourier_cuts_ and _plan."""
        _, C, m = X.shape
        P = len(views)
        ljobs = ([j[1:] for j in _level_jobs(views, C)] if self.level_word
                 else [])
        if _is_family(views):
            # one cut job per (channel, side, window, dilations) in family
            # order; a family's cut table holds 8 cuts per channel slot
            jobs = [(c, p == 1, w, ds) for p in range(P)
                    for (s, w, ds, cs) in views[p] if s == "fourier"
                    for c in cs]
            tab, lcuts = _fit_tables(X, jobs, ljobs, m, self.seed_)
            self.level_cuts_ = lcuts
            cuts, at = [], 0
            for p in range(P):
                for (s, w, ds, cs) in views[p]:
                    if s != "fourier":
                        continue
                    cv = np.zeros(_FAMILY_CUTS)
                    for si in range(len(cs)):
                        cv[si * WORD_LEN:(si + 1) * WORD_LEN] = tab[at]
                        at += 1
                    cuts.append(cv)
            self._plan = self._build_plan(views, cuts, m, C, lcuts)
            self.fourier_cuts_ = cuts
        else:
            jobs = [(p % C, p >= C, w, ds) for p in range(P)
                    for (s, w, ds) in views[p] if s == "fourier"]
            cuts, lcuts = _fit_tables(X, jobs, ljobs, m, self.seed_)
            self.level_cuts_ = lcuts
            self._plan = self._build_plan(views, cuts, m, C, lcuts)
            per, at = [], 0
            for p in range(P):
                k = sum(1 for (s, _, _) in views[p] if s == "fourier")
                per.append(cuts[at:at + k].copy())
                at += k
            self.fourier_cuts_ = per

    def _prepare_tables(self, m):
        """The fit-time tables for series of length m: the trigonometric
        tables (returned), the term-frequency table, the buffer types and the
        empty head-column table."""
        pl = self._plan
        trig = self._trig(m)
        pl["sublinear_tf"] = bool(self.sublinear_tf)   # scoring reads them
        pl["numerosity_reduction"] = bool(self.numerosity_reduction)
        # fit scratch, see _FIT_SCRATCH
        self._tf_store, self._tf_tab = _tf_tables(m, pl["sublinear_tf"])
        self._set_buffer_types(m)
        self._head_col = np.full(
            pl["tab_total"], head_col_none(self._id_bt), dtype=self._id_bt)
        return trig

    def _mark_fitted(self, metadata):
        """Declare a transformer fitted by a classifier: ``metadata`` is the
        data description aeon's checks compare new input against."""
        self.metadata_ = dict(metadata)
        self.is_fitted = True

    def _transform(self, X, y=None):
        """The weighted rows of X as a float64 scipy CSR matrix of shape
        (n_cases, n_features_)."""
        with _threads(_resolve_threads(self.n_jobs)):
            try:
                return self._rows(_as_collection(X))
            finally:
                # the keying schedule does not outlive the call
                self.__dict__.pop("_chain_cache", None)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {"n_dictionaries": 10}
