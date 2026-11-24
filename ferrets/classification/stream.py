"""The streaming head: averaged least-squares SGD epochs over the chunks
the transformer encoded."""

import time

import numpy as np

from .. import config
from ..classification.common import _targets
from ..constants import _NO_WT
from ..helpers.threads import _team_size
from ..kernels.asgd import _asgd_finish, _asgd_pass


def _sgd_params():
    """The streaming head's scalars, read by the SGD kernels at call time:
    (eta, intercept decay, max |dloss|, wscale reset, ln(1 - eta alpha))."""
    lnc = np.log1p(-(np.float64(config.SQ_ETA0) * np.float64(config.SQ_ALPHA)))
    return np.array([config.SQ_ETA0, config.INTERCEPT_DECAY,
                     config.MAX_DLOSS, config.WSCALE_RESET, lnc],
                    dtype=np.float64)


def _stream_weights(y_idx, K, rule):
    """The streaming head's per-class update weights (K,), or None when
    unweighted: rule None, or exactly balanced training classes. "sqrt":
    sklearn's "balanced" n / (K n_c) to the power 0.5, normalised so the
    largest weight is 1: a weight above about 2 makes the constant step
    diverge on unit-norm rows, so the majority classes' steps shrink
    instead."""
    if rule is None:
        return None
    counts = np.bincount(y_idx, minlength=K).astype(np.float64)
    if counts.min() == counts.max():
        return None
    w = (len(y_idx) / (K * counts)) ** 0.5
    return w / w.max()


class StreamMixin:
    """The streaming head.

    Sets: alpha_, alpha_scores_, coef_, cv_score_, intercept_, n_epochs_,
    solver_.
    Reads: stream_class_weight_, transformer_."""

    def _fit_stream(self, chunks, y_idx, K, stats):
        """The SQ_EPOCHS passes over the chunks the transformer encoded
        (see _stream_epochs); stats gets t_epochs."""
        t2 = time.perf_counter()
        self._stream_epochs(chunks, y_idx, K, stats)
        stats["t_epochs"] = time.perf_counter() - t2

    def _stream_epochs(self, chunks, y_idx, K, stats):
        """The SQ_EPOCHS passes. RNG protocol: RandomState(seed + 1) draws
        every epoch's chunk order first, then one row permutation per chunk
        in processing order. A chunk's pass is one team kernel on every
        thread (_asgd_pass); the next chunk is then decoded, on every thread,
        into the same buffer. A chunk that follows itself (an epoch ends on
        the chunk the next one starts with) and a cache of one chunk, which
        fits that buffer, are decoded once. The results depend on none of
        it (nor on the cache mode or the thread count). Row updates are
        weighted by stream_class_weight_ when it is not None."""
        F = chunks.n_features
        cw = self.stream_class_weight_   # per-class update weights or None
        Kp = 1 if K == 2 else K
        sgd = _sgd_params()
        # _asgd_pass state: WA weights and their running average (F, Kp, 2);
        # b, ab the intercepts and their average; seg = (t0, e0) the
        # averaging segment all heads share; pbuf partial dot products and
        # flags the team's barrier words (scratch); t the 1-based step.
        WA = np.zeros((F, Kp, 2), dtype=np.float32)
        seg = np.ones(2, dtype=np.int64)
        pbuf = np.zeros((config.ASGD_PARTS, max(Kp, 16)))
        team = _team_size()
        flags = np.zeros(8 * team, dtype=np.int64)
        b = np.zeros(Kp)
        ab = np.zeros(Kp)
        t = 1
        n_ch = chunks.n_chunks
        rng = np.random.RandomState((self.transformer_.seed_ + 1) % (1 << 32))
        seq = np.concatenate([rng.permutation(n_ch)
                              for _ in range(config.SQ_EPOCHS)])
        resident = chunks.resident
        stats["team_threads"] = int(team)
        stats["decode_resident"] = bool(resident)
        feed = chunks.feed(seq)
        tt = time.perf_counter
        t_heads = t_decode = 0.0
        try:
            feed.prefetch(0)
            ch, arrs = feed.fetch(int(seq[0]))
            feed.prefetch(1)
            td = tt()
            cur = feed.decode(int(seq[0]), ch, arrs)
            t_decode += tt() - td
            del arrs
            for j in range(len(seq)):
                nr = len(ch.rows)
                perm = rng.permutation(nr).astype(np.int64)
                y_rows = y_idx[ch.rows]
                Yc = _targets(y_rows, K)
                wt = _NO_WT if cw is None else cw[y_rows]
                row_start, row_end, ids, vals = cur
                th = tt()
                flags[:] = 0
                _asgd_pass(row_start, row_end, ids, vals, perm, wt, Yc, WA, b,
                           ab, seg, np.int64(t), sgd,
                           np.int64(config.ASGD_PARTS), pbuf, np.int64(team),
                           flags)
                t_heads += tt() - th
                t += nr
                if j + 1 < len(seq) and not resident:
                    again = feed.repeats(j + 1)
                    if not again:
                        k = int(seq[j + 1])
                        ch, arrs = feed.fetch(k)
                    feed.prefetch(j + 2)
                    if not again:
                        td = tt()
                        cur = feed.decode(k, ch, arrs)
                        t_decode += tt() - td
                        del arrs
        finally:
            feed.close()
        stats["t_heads"] = t_heads
        stats["t_decode"] = t_decode
        stats["decode_buffers"] = int(feed.buf is not None)
        coef = np.empty((F, Kp), dtype=np.float32)
        _asgd_finish(WA, seg, np.int64(t - 1), sgd, coef)
        self.coef_ = coef
        self.intercept_ = ab.copy()
        self.alpha_ = None
        self.alpha_scores_ = None
        self.cv_score_ = None
        self.solver_ = "asgd"
        self.n_epochs_ = config.SQ_EPOCHS
