"""FERRETSDecoder: a fitted model's columns read back as words."""

import numpy as np

from ..constants import (_KIND_PAA, _MODE_GLOBAL, KEYSPACE, LEVEL_CUTS,
                         WORD_LEN, head_col_none)
from ..decode.reference import ALPHABET, nr_counts, ReferenceView
from ..helpers.threads import _resolve_threads, _threads
from ..helpers.validation import _as_collection
from ..transform.run import _buffer_types


class FERRETSDecoder:
    """Column decoder and exact reference scans of a fitted model.

    Reads a fitted FERRETSClassifier column by column: a global column is a
    (view, dilation, key) word, either a PAA or Fourier word or a level word.
    Univariate models and models with ``channel_layout="per_channel"`` are
    supported; a multichannel model of the "family" layout is refused
    (ValueError). It serves interpretability studies and checks of the
    model's keys against a plain numpy scan.

    Parameters
    ----------
    model : FERRETSClassifier
        A fitted classifier.
    X_train : ignored

    Attributes
    ----------
    specs, dils, keyspace, views, channel, pseudo, input_channel : list
        One entry per flat view: (name, window, sub, alphabet); its
        dilations; its key space; its ReferenceView (reference key scans);
        0 for the series or 1 for its difference; its pseudo-channel; its
        input channel.
    view_off : ndarray of int64
        Global column offset of each view, one past the end last.
    keep_cols : ndarray of int64
        Global column of every head column.
    slot_off, slot_level : list
        Per view, per dilation (a slot): its global start, and whether it
        owns a level block. A slot is its word block, then its level block
        when the model has the level word and the slot owns one, so
        ``view_off[v] + di * 256`` is a slot's start only without the level
        word: use ``slot_start(v, di)``. ``block_start`` and ``block_of``
        map the 256-column blocks.

    ``decode`` and ``encode`` map a column to a word and back; ``positions``,
    ``contains`` and ``reference_counts`` scan with the reference arithmetic;
    ``model_counts`` and ``exact_contains`` read the model's own kernel."""

    def __init__(self, model, X_train=None):
        tr = getattr(model, "transformer_", None)
        if (not hasattr(model, "coef_") or tr is None
                or getattr(tr, "_plan", None) is None):
            raise ValueError("FERRETSDecoder needs a fitted "
                             "ferrets.FERRETSClassifier")
        if "cptr" in tr._plan:
            raise ValueError(
                "FERRETSDecoder decodes the per-pseudo-channel layout (a "
                "univariate model, or ferrets.FERRETSClassifier("
                "channel_layout='per_channel')); this model uses the default "
                "channel layout with cross-channel words")
        self.model = model
        self.transformer = tr
        pl = tr._plan
        level = bool(pl["level"])
        self.n_channels = int(tr.n_channels_)
        self.n_timepoints = int(tr.n_timepoints_)
        V = int(pl["kind"].shape[0])
        P = int(pl["P"])
        self.n_pseudo = P
        self.specs, self.dils, self.keyspace, self.views = [], [], [], []
        self.channel, self.pseudo, self.input_channel = [], [], []
        self.local_col = []
        self.slot_off, self.slot_level = [], []
        pv = pl["pv_ptr"]
        for p in range(P):
            for v in range(int(pv[p]), int(pv[p + 1])):
                kind = "paa" if int(pl["kind"][v]) == _KIND_PAA else "fourier"
                w = int(pl["win"][v])
                view = ReferenceView(kind, w, int(pl["ch"][v]),
                                     int(pl["diff"][v]), self.n_channels,
                                     pl["cuts"][v], int(pl["flat"][v]),
                                     pl["lcuts"][int(pl["lrow"][v])]
                                     if level else None)
                self.specs.append((f"{kind}{w}", w, view.sub, ALPHABET))
                self.dils.append([int(d) for d in
                                  pl["dils"][pl["dptr"][v]:pl["dptr"][v + 1]]])
                self.keyspace.append(KEYSPACE)
                self.views.append(view)
                self.channel.append(int(pl["diff"][v]))
                self.input_channel.append(int(pl["ch"][v]))
                self.pseudo.append(p)
                self.local_col.append(int(pl["col"][v]))
                d0, d1 = int(pl["dptr"][v]), int(pl["dptr"][v + 1])
                self.slot_off.append([int(pl["dtab"][di])
                                      for di in range(d0, d1)])
                self.slot_level.append([bool(pl["own_level"][di])
                                        for di in range(d0, d1)])
        if len(self.views) != V:
            raise ValueError("decoder/model view count mismatch")
        self.has_level = level
        self.view_off = np.append(np.asarray(pl["tab"], dtype=np.int64),
                                  np.int64(pl["tab_total"]))
        # the 256-column blocks in global order: per view, per slot, the word
        # block, then the slot's level block when it owns one
        bs, bv, bd, bl = [], [], [], []
        for v in range(V):
            for di, (g0, lv) in enumerate(zip(self.slot_off[v],
                                              self.slot_level[v])):
                for is_level in ((False, True) if lv else (False,)):
                    bs.append(g0 + KEYSPACE * is_level)
                    bv.append(v)
                    bd.append(di)
                    bl.append(is_level)
        self.block_start = np.asarray(bs, dtype=np.int64)
        self.block_view = np.asarray(bv, dtype=np.int64)
        self.block_slot = np.asarray(bd, dtype=np.int64)
        self.block_level = np.asarray(bl, dtype=bool)
        if (not np.array_equal(self.block_start,
                               KEYSPACE * np.arange(len(bs), dtype=np.int64))
                or KEYSPACE * len(bs) != int(pl["tab_total"])
                or [s[0] for s in self.slot_off]
                != self.view_off[:-1].tolist()):
            raise ValueError("decoder/model layout mismatch: the global "
                             "column space is not the views' slots in order")
        # the kept columns, in head order, from the head-column table
        hc = tr._head_col
        self._none = head_col_none(hc.dtype)
        off = np.asarray(tr.feature_offsets_, dtype=np.int64)
        keep = []
        for p in range(P):
            base = int(pl["pc_tab"][p])
            seg = hc[base:base + int(pl["n_cols"][p])].astype(np.int64)
            k = np.flatnonzero(seg != self._none)
            if not np.array_equal(seg[k], np.arange(len(k))) or \
                    len(k) != int(off[p + 1] - off[p]):
                raise ValueError(f"head-column table of pseudo-channel {p} "
                                 "does not match feature_offsets_")
            keep.append(base + k)
        self.keep_cols = (np.concatenate(keep) if keep
                          else np.empty(0, dtype=np.int64))
        if len(self.keep_cols) != int(tr.n_features_):
            raise ValueError("decoder/model vocabulary size mismatch")
        self._pc_of_view = np.asarray(self.pseudo, dtype=np.int64)

    def view_of(self, g):
        return int(np.searchsorted(self.view_off, int(g), side="right") - 1)

    def head_column(self, g):
        """Head column of global column g, or None when g was pruned."""
        v = self.view_of(g)
        local = int(self.transformer._head_col[int(g)])
        if local == self._none:
            return None
        return int(self.transformer.feature_offsets_[self.pseudo[v]]) + local

    def slot_start(self, v, di):
        """Global column of view v's di-th dilation slot: its word block;
        the slot's level block, when it owns one, follows at + 256."""
        return self.slot_off[v][di]

    def block_of(self, g):
        """(view, dilation index, is_level, key) of global column g."""
        g = int(g)
        if not 0 <= g < int(self.view_off[-1]):
            raise ValueError(f"global column {g} out of range")
        b = int(np.searchsorted(self.block_start, g, side="right") - 1)
        return (int(self.block_view[b]), int(self.block_slot[b]),
                bool(self.block_level[b]), int(g) - int(self.block_start[b]))

    @staticmethod
    def _bin_range(cuts, j):
        """[lo, hi] of bin j among 15 sorted cuts (the values with exactly j
        cuts strictly below: (cuts[j-1], cuts[j]]); None at an open end."""
        return [float(cuts[j - 1]) if j > 0 else None,
                float(cuts[j]) if j < LEVEL_CUTS else None]

    def decode_global(self, g):
        g = int(g)
        v, di, is_level, key = self.block_of(g)
        name, window, sub, alphabet = self.specs[v]
        view = self.views[v]
        d = int(self.dils[v][di])
        out = dict(view=name, view_index=v, dilation=d, key=int(key),
                   pattern=format(int(key), f"0{WORD_LEN}b"),
                   window=window, span=(window - 1) * d + 1,
                   source="level" if is_level else view.kind,
                   channel=("series", "difference")[self.channel[v]],
                   input_channel=int(self.input_channel[v]),
                   pseudo_channel=int(self.pseudo[v]), global_col=g,
                   head_col=self.head_column(g))
        if is_level:            # k = 1: 4 mean-bin bits above 4 std-bin bits
            mb, sb = key >> 4, key & 15
            cuts = view.level_cuts
            out.update(mean_bin=int(mb), std_bin=int(sb),
                       mean_range=self._bin_range(cuts[:LEVEL_CUTS], mb),
                       std_range=self._bin_range(cuts[LEVEL_CUTS:], sb))
        return out

    def decode(self, head_col):
        return self.decode_global(int(self.keep_cols[int(head_col)]))

    def encode(self, word):
        """Global column of a decoded word (the inverse of decode_global). A
        level word is counted in one level block per (pseudo-channel,
        window, dilation), the first view's keying it: a level word named
        by another view encodes to that owner's column."""
        v = int(word["view_index"])
        di = self.dils[v].index(int(word["dilation"]))
        key = int(word["key"])
        if not 0 <= key < KEYSPACE:
            raise ValueError(f"key {key} outside 0..{KEYSPACE - 1}")
        if word.get("source") != "level":
            return self.slot_start(v, di) + key
        if not self.has_level:
            raise ValueError("a level word, but the model has no level word")
        v, di = self._level_owner(v, di)
        return self.slot_start(v, di) + KEYSPACE + key

    def _level_owner(self, v, di):
        """(view, dilation index) of the slot whose level block counts the
        level words of view v's slot di: the first view of its
        pseudo-channel keying its window and dilation."""
        p, w, d = self.pseudo[v], self.specs[v][1], self.dils[v][di]
        for u in range(v + 1):
            if self.pseudo[u] == p and self.specs[u][1] == w \
                    and d in self.dils[u]:
                k = self.dils[u].index(d)
                if not self.slot_level[u][k]:
                    raise ValueError(f"view {u}, dilation {d}: the first "
                                     "slot keying it owns no level block")
                return u, k

    def block_keys_one(self, x, v, dilation, level=False):
        """Every window's key of one series in view v's word block at
        ``dilation`` (with level: the slot's level block), by the reference
        scan."""
        view = self.views[v]
        return (view.level_keys_one if level else view.keys_one)(x, dilation)

    def block_keys_many(self, X, v, dilation, level=False):
        view = self.views[v]
        return (view.level_keys_many if level else view.keys_many)(X,
                                                                   dilation)

    def positions(self, x, word):
        """Window starts (ascending) of every occurrence of ``word`` in the
        series x (all occurrences: no numerosity reduction)."""
        keys = self.block_keys_one(x, word["view_index"], word["dilation"],
                                   word.get("source") == "level")
        return np.flatnonzero(keys == word["key"]).tolist()

    def contains(self, X, word):
        """Presence of ``word`` in each series of X, by the reference scan
        (one series at a time)."""
        return np.array([len(self.positions(x, word)) > 0 for x in X])

    def reference_counts(self, X, v):
        """(n, view width) counts of flat view v recomputed by the reference
        scan and the model's numerosity reduction, if any. The columns are the
        view's global columns from view_off[v], slot by slot: the word block,
        then the level block when the slot owns one."""
        view = self.views[v]
        nr = bool(self.transformer._plan["numerosity_reduction"])   # fit's
        out = []
        for d, lv in zip(self.dils[v], self.slot_level[v]):
            out.append(nr_counts(view.keys_many(X, d), nr=nr))
            if lv:
                out.append(nr_counts(view.level_keys_many(X, d), nr=nr))
        return np.concatenate(out, axis=1)

    def _collection(self, X):
        X3 = _as_collection(X)
        if X3.shape[1] != self.n_channels:
            raise ValueError(f"fitted on {self.n_channels} channels, got "
                             f"{X3.shape[1]}")
        if X3.shape[2] != self.n_timepoints:
            raise ValueError(f"series length {X3.shape[2]} differs from fit "
                             f"({self.n_timepoints})")
        return X3

    def model_counts(self, X, v, chunk_rows=4096):
        """(n, view width) counts of flat view v from the model's own kernel
        (FERRETS._run in _MODE_GLOBAL): the counts its bags are built from."""
        X3 = self._collection(X)
        model = self.model
        tr = self.transformer
        pl = tr._plan
        n, _, m = X3.shape
        width = int(self.view_off[v + 1] - self.view_off[v])
        out = np.zeros((n, width), dtype=np.int64)
        if n == 0:
            return out
        w = self.views[v].window
        rb = max(1, sum((1 + lv) * min(m - (w - 1) * d, KEYSPACE)
                        for d, lv in zip(self.dils[v], self.slot_level[v])
                        if m - (w - 1) * d > 0))
        id_dt, cnt_dt = _buffer_types(pl, m)
        c0 = self.local_col[v]
        with _threads(_resolve_threads(model.n_jobs)):
            trig = tr._trig(m)
            try:
                for s in range(0, n, chunk_rows):
                    e = min(n, s + chunk_rows)
                    ids = np.empty((e - s, rb), dtype=id_dt)
                    cnt = np.empty((e - s, rb), dtype=cnt_dt)
                    nnz = np.zeros(e - s, dtype=np.int64)
                    tr._run(X3, np.arange(s, e, dtype=np.int64), v, v + 1,
                           _MODE_GLOBAL, trig, bufs=(ids, cnt, nnz))
                    mask = np.arange(rb)[None, :] < nnz[:, None]
                    rows = np.nonzero(mask)[0] + s
                    out[rows, ids[mask].astype(np.int64) - c0] = cnt[mask]
            finally:
                # the keying schedule is the call's: none stays on the model
                tr.__dict__.pop("_chain_cache", None)
        return out

    def exact_contains(self, X, words):
        """Presence of each word (decoded dicts) in each series, read from
        the model's own counts: (n, len(words)) bool. A word's column is
        ``encode(word)`` (a level word: its owner's)."""
        X3 = self._collection(X)
        res = np.zeros((X3.shape[0], len(words)), dtype=bool)
        cols = [self.encode(wd) for wd in words]
        by_view = {}
        for j, g in enumerate(cols):
            by_view.setdefault(self.view_of(g), []).append(j)
        for v, js in by_view.items():
            counts = self.model_counts(X3, v)
            for j in js:
                res[:, j] = counts[:, cols[j] - int(self.view_off[v])] > 0
        return res
