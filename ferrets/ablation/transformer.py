"""The ablation's transform: FERRETS with a selectable word length and, for a
pruned refit, a fixed set of views and seed."""

import numba
import numpy as np

from ..ablation.kernels import _key_univariate_bits
from ..constants import (_MODE_SCORE, _NO_ACC, _NO_HC, _NO_IDF, _NO_IDS,
                         _NO_VEC, _NO_W, head_col_none)
from ..transform.plan import _view_span
from ..transformer import FERRETS


WORD_BITS = (8, 4, 2)       # bits per word, the shipped 8 first


class AblationTransformer(FERRETS):
    """FERRETS with the ablation's ``word_bits`` switch (8, 4 or 2 bits per
    PAA and Fourier word; the univariate kernel only). A pruned refit sets
    ``_fixed_views`` and ``_fixed_seed`` on it before fitting."""

    _force_bits_kernel = False      # test-only: 8 bits through the bits kernel

    def __init__(self, n_dictionaries=100, windows=(8, 16, 24, 32, 48, 64),
                 word_sources=("paa", "fourier"), use_first_difference=True,
                 dilation="log_uniform", level_word=True,
                 cross_channel_words="all", channel_layout="family",
                 min_doc_freq=1, use_idf=False, numerosity_reduction=False,
                 sublinear_tf=True, word_bits=8, n_jobs=1, random_state=0):
        self.word_bits = word_bits
        super().__init__(
            n_dictionaries=n_dictionaries, windows=windows,
            word_sources=word_sources,
            use_first_difference=use_first_difference, dilation=dilation,
            level_word=level_word, cross_channel_words=cross_channel_words,
            channel_layout=channel_layout, min_doc_freq=min_doc_freq,
            use_idf=use_idf, numerosity_reduction=numerosity_reduction,
            sublinear_tf=sublinear_tf, n_jobs=n_jobs,
            random_state=random_state)

    def _build_plan(self, views, cuts, m, C, lcuts=None):
        pl = super()._build_plan(views, cuts, m, C, lcuts)
        pl["word_bits"] = int(self.word_bits)       # read by _run
        return pl

    def _run(self, X, rows, v0, v1, mode, trig, head_col=None, tf_tab=None,
             idf=None, W=None, bufs=None, acc=None, sq=None):
        """FERRETS._run, with the bits kernel for 4- and 2-bit words (and for
        8 bits under _force_bits_kernel)."""
        pl = self._plan
        nbits = int(pl["word_bits"])
        if "cross" in pl:
            if nbits != WORD_BITS[0]:
                raise ValueError("word_bits applies to the univariate "
                                 "kernel; the default channel layout "
                                 "keys 8-bit words")
            return super()._run(X, rows, v0, v1, mode, trig, head_col,
                                tf_tab, idf, W, bufs, acc, sq)
        if nbits == WORD_BITS[0] and not self._force_bits_kernel:
            return super()._run(X, rows, v0, v1, mode, trig, head_col,
                                tf_tab, idf, W, bufs, acc, sq)
        # the argument preparation of FERRETS._run, then the bits kernel
        n_rows = len(rows)
        nb = int(max(1, min(n_rows, 4 * numba.get_num_threads())))
        if mode == _MODE_SCORE:
            ids = cnt = _NO_IDS
            nnz = np.zeros(n_rows, dtype=np.int64)
        else:
            ids, cnt, nnz = bufs
            acc, sq, tf_tab, idf, W = (_NO_ACC, _NO_VEC, _NO_VEC,
                                       _NO_IDF, _NO_W)
        if head_col is None:
            head_col = _NO_HC
        hc_none = head_col_none(head_col.dtype)
        nr = np.int64(1 if pl["numerosity_reduction"] else 0)
        tile = np.int64(self._row_tile(mode))
        _key_univariate_bits(X, rows, np.int64(nb), pl["kind"], pl["win"],
                             pl["ch"], pl["diff"], pl["dptr"], pl["dils"],
                             pl["cuts"], pl["flat"], pl["dcol"], pl["dtab"],
                             pl["own_level"], pl["lrow"], pl["lcuts"],
                             pl["woff"], pl["trig"], trig[0], trig[1],
                             np.int64(v0), np.int64(v1), np.int64(mode), nr,
                             tile, head_col, np.int64(hc_none), tf_tab, idf,
                             W, ids, cnt, nnz, acc, sq, np.int64(nbits))
        return nnz

    def _resolve_seed(self):
        """A pruned refit reuses the first fit's seed (random_state None or
        a RandomState would otherwise draw a new one)."""
        seed = getattr(self, "_fixed_seed", None)
        return seed if seed is not None else super()._resolve_seed()

    def _draw_views(self, P, m):
        """A pruned refit's fixed views, else FERRETS's draw."""
        fixed = getattr(self, "_fixed_views", None)
        if fixed is not None:
            if len(fixed) != P:
                raise RuntimeError(f"{len(fixed)} fixed view lists for {P} "
                                   "pseudo-channels")
            return fixed
        return super()._draw_views(P, m)

    def _view_norms(self, coef):
        """Per pseudo-channel, the l2 norm of each view's head weights (its
        kept columns, every class) of a head with weights coef."""
        pl = self._plan
        hc = self._head_col
        none = head_col_none(hc.dtype)
        coef = np.asarray(coef, dtype=np.float64)
        out = []
        for p in range(pl["P"]):
            v0, v1 = int(pl["pv_ptr"][p]), int(pl["pv_ptr"][p + 1])
            base = int(self.feature_offsets_[p])
            sc = np.empty(v1 - v0)
            for v in range(v0, v1):
                t0, t1 = _view_span(pl, v)
                seg = hc[t0:t1]
                seg = seg.astype(np.int64)
                rows = seg[seg != none] + base
                sc[v - v0] = np.sqrt(np.sum(coef[rows] ** 2))
            out.append(sc)
        return out
