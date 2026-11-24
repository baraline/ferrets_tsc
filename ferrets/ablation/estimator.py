"""FERRETSAblation: FERRETSClassifier with the ablation's extra switches."""

import math
import time

import numpy as np

from ..ablation.heads import AblationHeadsMixin
from ..ablation.transformer import WORD_BITS, AblationTransformer
from ..classifier import FERRETSClassifier
from ..helpers.validation import _is_int


HEADS = ("default", "logistic")


class FERRETSAblation(AblationHeadsMixin, FERRETSClassifier):
    """FERRETSClassifier with switches that are not part of the model.

    Meant for ablation studies: each switch replaces one component of the
    default model so that its effect can be measured.

    Parameters
    ----------
    head : {"default", "logistic"}, default="default"
        "default" is the classifier's own head (the ridge, or the streaming
        least-squares head under ``use_streaming``). "logistic" is a batch
        logistic regression trained by SGD.
    prune_frac : float in (0, 1], default=1.0
        Fraction of each pseudo-channel's (or side's) views kept. The model
        is fitted, its views ranked by the l2 norm of their head weights, and
        refitted on the highest ranked ones with the same seed.
    ridge_cv : None or int >= 2, default=None
        None keeps the classifier's own choice of alpha. An int selects alpha
        by that many folds of cross-validation (batch ridge head only).
    word_bits : {8, 4, 2}, default=8
        Bits per PAA and Fourier word. Fewer than 8 needs the univariate
        kernel (a univariate model, or ``channel_layout="per_channel"``).

    Attributes
    ----------
    views_all_ : list
        Pruned fit only: the views of the first fit.
    view_norms_ : list of np.ndarray
        Pruned fit only: the head-weight norm of each of those views, per
        pseudo-channel.

    Notes
    -----
    The other parameters and the attributes are those of FERRETSClassifier;
    ``solver_`` is "logistic_sgd" or "lsqr_cv<k>" under the two alternative
    heads, and a pruned fit adds ``fit_stats_["first_fit"]``.
    ``head="logistic"`` and ``ridge_cv`` refuse the class weights of a batch
    fit (ValueError): pass ``batch_class_weight=None`` with them."""

    def __init__(self, n_dictionaries=100, windows=(8, 16, 24, 32, 48, 64),
                 word_sources=("paa", "fourier"), use_first_difference=True,
                 dilation="log_uniform", level_word=True,
                 cross_channel_words="all", channel_layout="family",
                 min_doc_freq=1, use_idf=False, numerosity_reduction=False,
                 sublinear_tf=True, use_streaming=False,
                 alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0), prior_shift=True,
                 batch_class_weight="sqrt", stream_class_weight="sqrt",
                 design_ram_mb=8192.0, head="default", prune_frac=1.0,
                 ridge_cv=None, word_bits=8, n_jobs=1, random_state=0):
        self.head = head
        self.prune_frac = prune_frac
        self.ridge_cv = ridge_cv
        self.word_bits = word_bits
        super().__init__(
            n_dictionaries=n_dictionaries, windows=windows,
            word_sources=word_sources,
            use_first_difference=use_first_difference, dilation=dilation,
            level_word=level_word, cross_channel_words=cross_channel_words,
            channel_layout=channel_layout, min_doc_freq=min_doc_freq,
            use_idf=use_idf, numerosity_reduction=numerosity_reduction,
            sublinear_tf=sublinear_tf, use_streaming=use_streaming,
            alphas=alphas, prior_shift=prior_shift,
            batch_class_weight=batch_class_weight,
            stream_class_weight=stream_class_weight,
            design_ram_mb=design_ram_mb, n_jobs=n_jobs,
            random_state=random_state)

    def _check_params(self):
        al = super()._check_params()
        if self.head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got "
                             f"{self.head!r}")
        pf = self.prune_frac
        if isinstance(pf, (bool, np.bool_)) or not isinstance(
                pf, (int, float, np.integer, np.floating)) \
                or not 0.0 < float(pf) <= 1.0:
            raise ValueError(f"prune_frac must be in (0, 1], got {pf!r}")
        cv = self.ridge_cv
        if cv is not None and (not _is_int(cv) or int(cv) < 2):
            raise ValueError(f"ridge_cv must be None or an int >= 2, got "
                             f"{cv!r}")
        if self.use_streaming and self.head != "default":
            raise ValueError(f"head={self.head!r} is a batch head; "
                             "use_streaming=True fits the streaming "
                             "least-squares head")
        if self.use_streaming and cv is not None:
            raise ValueError("ridge_cv applies to the batch ridge head; "
                             "use_streaming=True has none")
        if self.head != "default" and cv is not None:
            raise ValueError("ridge_cv applies to the ridge head, not to "
                             f"head={self.head!r}")
        if (self.batch_class_weight is not None and not self.use_streaming
                and (self.head != "default" or cv is not None)):
            raise ValueError("batch_class_weight weights the default ridge "
                             "head; head='logistic' and ridge_cv have "
                             "their own solvers (pass batch_class_weight="
                             "None)")
        wb = self.word_bits
        if isinstance(wb, (bool, np.bool_)) or not _is_int(wb) \
                or int(wb) not in WORD_BITS:
            raise ValueError(f"word_bits must be one of {WORD_BITS}, got "
                             f"{wb!r}")
        return al

    _force_bits_kernel = False      # test-only: 8 bits through the bits kernel

    def _make_transformer(self):
        """AblationTransformer with the word length and, for a pruned refit,
        the fixed views and seed."""
        tr = AblationTransformer(
            n_dictionaries=self.n_dictionaries, windows=self.windows,
            word_sources=self.word_sources,
            use_first_difference=self.use_first_difference,
            dilation=self.dilation, level_word=self.level_word,
            cross_channel_words=self.cross_channel_words,
            channel_layout=self.channel_layout,
            min_doc_freq=self.min_doc_freq, use_idf=self.use_idf,
            numerosity_reduction=self.numerosity_reduction,
            sublinear_tf=self.sublinear_tf, n_jobs=self.n_jobs,
            random_state=self.random_state, word_bits=self.word_bits)
        tr._force_bits_kernel = self._force_bits_kernel
        fixed = getattr(self, "_fixed", None)
        if fixed is not None:
            tr._fixed_views, tr._fixed_seed = fixed
        return tr

    def _fit_model(self, X, y_idx, alphas):
        """Fit; with prune_frac < 1, keep the best views of each
        pseudo-channel by head-weight norm and refit on them."""
        if float(self.prune_frac) >= 1.0:
            super()._fit_model(X, y_idx, alphas)
            return
        t0 = time.perf_counter()
        super()._fit_model(X, y_idx, alphas)
        first_stats = self.fit_stats_
        views_all = self.transformer_.views_
        norms = self._view_norms()
        kept = []
        for vp, sc in zip(views_all, norms):
            k = max(1, int(math.ceil(float(self.prune_frac) * len(vp))))
            keep = np.sort(np.argsort(-sc)[:k])
            kept.append(tuple(vp[i] for i in keep))
        self._fixed = (kept, self.transformer_.seed_)
        try:
            super()._fit_model(X, y_idx, alphas)
        finally:
            del self._fixed
        self.views_all_ = views_all
        self.view_norms_ = norms
        self.fit_stats_["first_fit"] = first_stats
        self.fit_stats_["t_fit_total"] = time.perf_counter() - t0

    def _view_norms(self):
        """Per pseudo-channel, the l2 norm of each view's head weights (its
        kept columns, every class)."""
        return self.transformer_._view_norms(self.coef_)
