"""FERRETSClassifier: a ridge or streaming SGD head on the transform."""

import time

import numpy as np
from aeon.classification.base import BaseClassifier
from scipy.special import expit

from . import config
from .classification.common import _decide, _targets
from .classification.gram import GramMixin
from .classification.lsqr import LsqrMixin
from .classification.prior import _prior_lambda
from .classification.stream import _stream_weights, StreamMixin
from .classification.weights import (_check_batch_class_weight,
                                     _ridge_class_weights)
from .helpers.threads import _resolve_threads, _threads
from .helpers.validation import _as_collection, _check_X
from .transformer import FERRETS

__all__ = ["FERRETSClassifier"]


class FERRETSClassifier(GramMixin, LsqrMixin, StreamMixin, BaseClassifier):
    """Time series classifier on the FERRETS symbolic transform.

    Random dictionaries key the series and its first difference into sparse
    bags of exact 8-bit words (see ``FERRETS``), and a linear head scores the
    unit-norm rows. The batch head is an exact ridge regression, solved in the
    dual with generalised cross-validation for small training sets and by LSQR
    with a hold-out above. The streaming head is an averaged least-squares SGD
    over a compact chunk cache, for designs that do not fit in memory.

    Both heads weight the classes by the square root of their inverse
    frequency, and the batch head shifts its decision by a prior correction.
    ``predict_proba`` is a monotone logistic squash of the decision, not a
    calibrated probability.

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
    use_streaming : bool, default=False
        False fits the batch ridge head; True the streaming head, whose chunk
        cache is in RAM when small, else on disk, else recomputed each epoch.
    alphas : tuple of float, default=(1e-3, 1e-2, 1e-1, 1.0, 10.0)
        Ridge penalties the batch head chooses from.
    prior_shift : bool, default=True
        Batch head: shift the decision by lambda times the mean target, with
        lambda chosen on out-of-sample decisions of the training set by the
        one-standard-error rule. The streaming head is never shifted.
    batch_class_weight : {"sqrt", None}, default="sqrt"
        Batch head: weight each row of the ridge loss by its class's inverse
        frequency to the power one half (mean 1), in the exact leave-one-out
        and in the LSQR hold-out. None, or exactly balanced training classes,
        leaves the ridge unweighted.
    stream_class_weight : {"sqrt", None}, default="sqrt"
        Streaming head: weight each row update by its class's inverse
        frequency to the power one half (largest weight 1). None, or exactly
        balanced training classes, leaves the updates unweighted.
    design_ram_mb : float, default=8192.0
        RAM budget of the batch Gram route's encoded design blocks; blocks
        past it are recomputed when needed.
    n_jobs : int, default=1
        Threads used by fit and prediction: None or an int >= 1; -1 uses
        every numba thread.
    random_state : int, RandomState or None, default=0
        Seed of the random draws. An int is reduced modulo 2**32; None draws
        fresh entropy.

    Attributes
    ----------
    transformer_ : FERRETS
        The fitted transform; its attributes describe the dictionaries and
        the kept columns.
    coef_ : np.ndarray of shape (n_features, n_outputs)
        Head weights, float32; one output for two classes, else one per class.
        The transpose of scikit-learn's ``coef_``.
    intercept_ : np.ndarray of shape (n_outputs,)
        Head intercept, the prior shift included.
    solver_ : {"gcv", "lsqr", "asgd"}
        The route that fitted the head.
    alpha_ : float or None
        The ridge penalty the batch head chose (None for the streaming head).
    alpha_scores_ : tuple of float or None
        Selection score of each alpha: minus the mean leave-one-out squared
        error (Gram route) or the hold-out accuracy (LSQR route). None for the
        streaming head and for a single alpha on the LSQR route.
    cv_score_ : float or None
        The chosen alpha's leave-one-out accuracy (Gram route) or hold-out
        accuracy (LSQR route); None where alpha_scores_ is None.
    n_epochs_ : int or None
        Epochs of the streaming head (None for the batch head).
    prior_lambda_ : float
        The prior shift's lambda (0 when not shifted).
    prior_scores_ : tuple of float or None
        The equal-weight score of each lambda of the shift's grid (None when
        the shift is off or no out-of-sample decisions exist).
    batch_class_weight_ : np.ndarray or None
        Per-class row weights of the batch ridge (None: unweighted).
    stream_class_weight_ : np.ndarray or None
        Per-class update weights of the streaming head (None: unweighted).
    fit_stats_ : dict
        Timings and sizes of the last fit.
    classes_ : np.ndarray
        Class labels.
    n_classes_ : int
        Number of classes.

    See Also
    --------
    FERRETS : The transform alone.

    Notes
    -----
    The model is described in docs/model.md. ``decision_function`` returns the
    scores that ``predict`` and ``predict_proba`` are taken from.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from ferrets import FERRETSClassifier
    >>> X, y = make_example_3d_numpy(n_cases=20, n_timepoints=64,
    ...                              random_state=0)
    >>> clf = FERRETSClassifier(n_dictionaries=20).fit(X, y)
    >>> clf.predict(X).shape
    (20,)
    """

    _tags = {
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
                 sublinear_tf=True, use_streaming=False,
                 alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0), prior_shift=True,
                 batch_class_weight="sqrt", stream_class_weight="sqrt",
                 design_ram_mb=8192.0, n_jobs=1, random_state=0):
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
        self.use_streaming = use_streaming
        self.alphas = alphas
        self.prior_shift = prior_shift
        self.batch_class_weight = batch_class_weight
        self.stream_class_weight = stream_class_weight
        self.design_ram_mb = design_ram_mb
        self.n_jobs = n_jobs
        self.random_state = random_state
        super().__init__()

    def _check_params(self):
        """Validate the head's parameters (the transform's are validated by
        the transformer). Returns the ridge grid as a tuple of floats."""
        try:
            al = tuple(float(a) for a in self.alphas)
        except TypeError:
            raise ValueError(f"alphas must be a sequence, got {self.alphas!r}")
        if not al or not all(np.isfinite(a) and a > 0 for a in al):
            raise ValueError(f"alphas must be finite and > 0, got "
                             f"{self.alphas!r}")
        if not isinstance(self.use_streaming, (bool, np.bool_)):
            raise ValueError("use_streaming must be a bool")
        if not isinstance(self.prior_shift, (bool, np.bool_)):
            raise ValueError("prior_shift must be a bool")
        v = self.stream_class_weight
        if not (v is None or (isinstance(v, str) and v == "sqrt")):
            raise ValueError(f"stream_class_weight must be 'sqrt' or None, "
                             f"got {v!r}")
        _check_batch_class_weight(self.batch_class_weight)
        v = self.design_ram_mb
        if isinstance(v, (bool, np.bool_)) or not isinstance(
                v, (int, float, np.integer, np.floating)) or not v > 0:
            raise ValueError(f"design_ram_mb must be a number > 0, got {v!r}")
        return al

    def _make_transformer(self):
        """The (unfitted) transform built from the transform parameters."""
        return FERRETS(
            n_dictionaries=self.n_dictionaries, windows=self.windows,
            word_sources=self.word_sources,
            use_first_difference=self.use_first_difference,
            dilation=self.dilation, level_word=self.level_word,
            cross_channel_words=self.cross_channel_words,
            channel_layout=self.channel_layout,
            min_doc_freq=self.min_doc_freq, use_idf=self.use_idf,
            numerosity_reduction=self.numerosity_reduction,
            sublinear_tf=self.sublinear_tf, n_jobs=self.n_jobs,
            random_state=self.random_state)

    def _fit(self, X, y):
        alphas = self._check_params()
        self._n_jobs = _resolve_threads(self.n_jobs)
        y_idx = np.unique(y, return_inverse=True)[1].astype(np.int64).ravel()
        with _threads(self._n_jobs) as blas_limited:
            X = _as_collection(X)
            self._fit_model(X, y_idx, alphas)
        self.fit_stats_["blas_limited"] = blas_limited
        return self

    def _fit_model(self, X, y_idx, alphas):
        """Fit the transformer, then the head on what it returns: the encoded
        design (batch) or the chunks (streaming). X is a validated float64
        collection, y_idx the class indices."""
        tt = time.perf_counter
        t0 = tt()
        stats = {}
        K = self.n_classes_
        tr = self._make_transformer()
        tr._check_params()          # an invalid transform parameter raises
        self.transformer_ = tr      # before the classifier holds anything
        self.stream_class_weight_ = None
        self.batch_class_weight_ = None
        if self.use_streaming:
            cw = _stream_weights(y_idx, K, self.stream_class_weight)
            self.stream_class_weight_ = cw
            stats["stream_class_weight"] = (None if cw is None
                                            else cw.tolist())
            with tr._fit_chunks(X, stats, metadata=self.metadata_) as chunks:
                self._fit_stream(chunks, y_idx, K, stats)
                # no out-of-sample decisions: the streaming head is not
                # shifted
                self._apply_prior(None, None, None, stats)
        else:
            bw = _ridge_class_weights(y_idx, K, self.batch_class_weight)
            self.batch_class_weight_ = bw
            stats["batch_class_weight"] = (None if bw is None
                                           else bw.tolist())
            with tr._fit_design(X, stats,
                                keep_all=self._batch_keep_all(X.shape[0]),
                                ram_mb=self.design_ram_mb,
                                metadata=self.metadata_) as design:
                self._fit_batch(design, y_idx, K, alphas, stats)
        stats["t_fit"] = tt() - t0
        self.fit_stats_ = stats

    def _batch_keep_all(self, n):
        """Whether the batch head of n training series needs every encoded
        block of the design held: the LSQR route flattens them all."""
        return n > config.DUAL_MAX_N

    def _fit_batch(self, design, y_idx, K, alphas, stats):
        """The batch head on the encoded design: the Gram route (up to
        config.DUAL_MAX_N training series) or the LSQR route."""
        Y = _targets(y_idx, K)
        if design.n > config.DUAL_MAX_N:
            self._fit_lsqr(y_idx, Y, alphas, design, stats)
        else:
            self._fit_gram(y_idx, Y, alphas, design, stats)
        stats["recomputed"] = int(design.recomputed)
        self.n_epochs_ = None

    def _apply_prior(self, dec, y_oos, ybar, stats):
        """Choose the prior shift's lambda on out-of-sample decisions dec of
        rows labelled y_oos (None: none exist, lambda 0) and fold it into
        intercept_."""
        lam, sc = 0.0, None
        if self.prior_shift and dec is not None and len(y_oos):
            lam, sc = _prior_lambda(np.asarray(dec, dtype=np.float64),
                                    np.asarray(y_oos), ybar,
                                    self.n_classes_)
        self.prior_lambda_ = float(lam)
        self.prior_scores_ = sc
        if lam:
            self.intercept_ = np.asarray(self.intercept_, dtype=np.float64) \
                - lam * np.asarray(ybar, dtype=np.float64)
        stats["prior_lambda"] = float(lam)

    def _decision(self, X):
        """(n, K') scores of a collection X of the fitted shape."""
        with _threads(_resolve_threads(self.n_jobs)):
            X = _as_collection(X)
            d = self.transformer_._score(X, self.coef_)
        return d + self.intercept_[None, :]

    def decision_function(self, X):
        """Scores of the head: (n,) for two classes (positive: the second
        class in ``classes_``), else (n, n_classes). Equal to
        ``transformer_.transform(X) @ coef_ + intercept_`` up to rounding."""
        self._check_is_fitted()
        X = self._preprocess_collection(X, store_metadata=False)
        self._check_shape(X)
        d = self._decision(X)
        return d[:, 0] if d.shape[1] == 1 else d

    def _predict(self, X):
        return self.classes_[_decide(self._decision(X))]

    def _predict_proba(self, X):
        """One-vs-rest logistic squash of the decision (as sklearn's
        _predict_proba_lr): a monotone map of the scores whose argmax is the
        prediction, not a calibrated probability."""
        prob = expit(self._decision(X))
        if prob.shape[1] == 1:
            p = prob[:, 0]
            return np.stack([1 - p, p], axis=1)
        prob /= prob.sum(axis=1)[:, None]
        return prob

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [{"n_dictionaries": 10},
                {"n_dictionaries": 10, "use_streaming": True}]
