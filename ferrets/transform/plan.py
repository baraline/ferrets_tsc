"""The flat view table and column layout of a fit.

Block p: one pseudo-channel of the per-channel layout (univariate: the series,
then its first difference) or one side of the family layout. The plan's P
blocks each own a span of columns; pv_ptr, pc_tab, n_cols, row_bound and
every "block p" of the package index them."""

import numpy as np

from .. import config
from ..constants import (_FAMILY_CUTS, _KIND_FOURIER, _KIND_PAA, KEYSPACE,
                         LEVEL_CUTS, WORD_LEN)


def _family_nblk(k, cross):
    """256-column blocks per (family, dilation) of k channels: one per
    channel, plus the cross-channel words of the code ``cross`` (bit 0 one
    coarse word, bit 1 two flag words) when k > 1."""
    return k + ((cross & 1) + 2 * ((cross >> 1) & 1) if k > 1 else 0)


def _is_family(views):
    """Whether the views are families ((source, window, dilations,
    channels) per side) rather than per-pseudo-channel (source, window,
    dilations) configurations."""
    lens = {len(v) for vp in views for v in vp}
    if len(lens) > 1:
        raise RuntimeError("views mix the family and per-pseudo-channel "
                           "formats")
    return lens == {4}


def _view_span(pl, v):
    """[start, end) of view v's global (unpruned) columns: its slots start
    at dtab[di], each (blocks + own_level[di]) x 256 wide, the last one ending
    the view."""
    d0, d1 = int(pl["dptr"][v]), int(pl["dptr"][v + 1])
    nblk = (_family_nblk(int(pl["cptr"][v + 1] - pl["cptr"][v]),
                         int(pl["cross"])) if "cross" in pl else 1)
    last = d1 - 1
    return (int(pl["dtab"][d0]),
            int(pl["dtab"][last])
            + (nblk + int(pl["own_level"][last])) * KEYSPACE)


def _level_jobs(views, C):
    """The level word's cut sets of a fit's views: one per (side, channel,
    window) some view keys, its sample over the union D of the
    dilations of every view keying that channel on that side at that window,
    whatever its source or channel set. A list of (key, ch, is_diff, window,
    D) sorted by key = (p, ch, window), p the side (family views) or the
    pseudo-channel (ch = p % C, the difference when p >= C); an entry's index
    is its row of level_cuts_."""
    fam = _is_family(views)
    union = {}
    for p, vp in enumerate(views):
        for v in vp:
            w = int(v[1])
            for c in (v[3] if fam else (p % C,)):
                union.setdefault((p, int(c), w), set()).update(
                    int(d) for d in v[2])
    return [((p, c, w), c, (p == 1) if fam else (p >= C), w,
             tuple(sorted(union[(p, c, w)])))
            for (p, c, w) in sorted(union)]


def _lcuts(lcuts, level):
    """The plan's level cut table: a copy of lcuts (J, 30) with the level
    word (level_cuts_ stays the model's record, not the table predict
    reads), else an empty (0, 30) table the kernels never read."""
    if not level:
        return np.zeros((0, 2 * LEVEL_CUTS))
    return np.array(lcuts, dtype=np.float64, order="C", copy=True)


def _check_lcuts(lcuts, views, C):
    """lcuts must be one (15 mean, 15 std) row per level job of the views,
    in _level_jobs order."""
    J = len(_level_jobs(views, C))
    shape = None if lcuts is None else np.shape(lcuts)
    if shape != (J, 2 * LEVEL_CUTS):
        raise ValueError(f"the level word needs the level cut sets of the "
                         f"views, shape ({J}, {2 * LEVEL_CUTS}); got "
                         f"{'none' if shape is None else shape}")


def _word_slots(kind, win, cptr, chs, dptr, dils, vcut, pv_ptr, m):
    """The family plan's channel-word numbers: (wptr, wref, n_wslots).

    A pair is one (dilation slot, channel slot); wptr[di] is the number of
    pairs before slot di (so slot di owns wptr[di] .. wptr[di + 1] - 1) and
    wref[wptr[di] + s] the number of its channel slot s's word, or -1. Two
    pairs key the same word when they agree on the side, the channel, the
    source, the window, the dilation and, for a Fourier word, the 8 cut
    values of the channel's slot bit for bit (a cut row depends on the
    family's whole dilation tuple, so equal windows and dilations are not
    enough). A word is numbered when two or more pairs key it, most used
    first (ties by first use in plan order), at most int(WORD_TABLE_MB x 1e6
    // m) of them, so a block's table of one-byte words stays within the
    budget; a slot with no window (m - (w - 1) d <= 0) is never numbered.
    WORD_TABLE_MB 0 numbers nothing."""
    cap = max(0, int(config.WORD_TABLE_MB * 1e6 // max(int(m), 1)))
    keys, wptr = [], [0]
    for p in range(len(pv_ptr) - 1):
        for v in range(pv_ptr[p], pv_ptr[p + 1]):
            w = win[v]
            fourier = kind[v] == _KIND_FOURIER
            for di in range(dptr[v], dptr[v + 1]):
                live = m - (w - 1) * dils[di] > 0
                for si in range(cptr[v + 1] - cptr[v]):
                    cut = (vcut[v][si * WORD_LEN:(si + 1) * WORD_LEN].tobytes()
                           if fourier else b"")
                    keys.append((p, chs[cptr[v] + si], kind[v], w, dils[di],
                                 cut) if live else None)
                wptr.append(len(keys))
    uses = {}
    for j, key in enumerate(keys):
        if key is not None:
            uses.setdefault(key, []).append(j)
    shared = sorted((k for k, u in uses.items() if len(u) >= 2),
                    key=lambda k: (-len(uses[k]), uses[k][0]))[:cap]
    slot = {k: s for s, k in enumerate(shared)}
    return (np.asarray(wptr, np.int64),
            np.asarray([slot.get(k, -1) for k in keys], np.int64),
            len(shared))


class _PlanLists:
    """The per-view and per-slot lists both plan layouts fill the same way
    (one entry per view, per (view, dilation) slot and per side), and the
    plan arrays made of them."""

    def __init__(self, m, level):
        self.m = m
        self.level = level
        self.kind, self.win, self.ch, self.diff = [], [], [], []
        self.dptr, self.dils = [0], []
        self.vcut, self.vcol, self.vtab, self.trig = [], [], [], []
        self.pv_ptr = [0]
        self.dcol, self.dtab, self.own_level, self.lrow = [], [], [], []
        self.n_cols, self.row_bound, self.pc_tab = [], [], []
        # running totals: tab, columns of the finished blocks; col, columns
        # of the current block; rb, its row entry bound
        self.tab = 0
        self.col = self.rb = 0
        self.seen = set()

    def begin_side(self):
        self.pc_tab.append(self.tab)
        self.col = self.rb = 0
        self.seen = set()

    def end_side(self):
        self.tab += self.col
        self.n_cols.append(self.col)
        self.row_bound.append(self.rb)
        self.pv_ptr.append(len(self.kind))

    def add_view(self, s, w, ds, ch, diff, cut, trig):
        """A view of source s ("paa" or "fourier"), window w and dilations
        ds, with its first channel, its difference flag, its cut row and its
        trigonometric-table slot."""
        self.kind.append(_KIND_PAA if s == "paa" else _KIND_FOURIER)
        self.win.append(w)
        self.ch.append(ch)
        self.diff.append(diff)
        self.dils.extend(ds)
        self.dptr.append(len(self.dils))
        self.vcut.append(cut)
        self.vcol.append(self.col)
        self.vtab.append(self.tab + self.col)
        self.trig.append(trig)

    def add_slots(self, w, ds, nblk, key):
        """The dilation slots of the view just added: each owns nblk blocks
        of KEYSPACE columns, plus a level block (own_level = 1) when it is the
        first slot of its block keying (key, w, d). key says what shares a
        level block: () in the per-channel layout (any view of the block with
        the same window and dilation), the channel tuple in the family
        layout."""
        for d in ds:
            own = int(self.level and (key, w, d) not in self.seen)
            self.seen.add((key, w, d))
            self.dcol.append(self.col)
            self.dtab.append(self.tab + self.col)
            self.own_level.append(own)
            self.col += KEYSPACE * (nblk + own)
            nw = self.m - (w - 1) * d
            if nw > 0:
                self.rb += (nblk + own) * min(nw, KEYSPACE)

    def arrays(self, wins, cut_width, lcuts):
        """The plan's arrays for V views, S dilation slots and P blocks:
        per-view, per-slot and per-block tables, the level cut table, and
        the flags and counts that size them. The layout adds its own, fit
        and the ablation add more. The keys are listed in docs/model.md,
        section "The plan"."""
        i64 = np.int64
        return dict(
            kind=np.asarray(self.kind, i64), win=np.asarray(self.win, i64),
            ch=np.asarray(self.ch, i64), diff=np.asarray(self.diff, i64),
            dptr=np.asarray(self.dptr, i64), dils=np.asarray(self.dils, i64),
            cuts=np.ascontiguousarray(np.asarray(self.vcut, dtype=np.float64)
                                      .reshape(-1, cut_width)),
            col=np.asarray(self.vcol, i64), tab=np.asarray(self.vtab, i64),
            dcol=np.asarray(self.dcol, i64), dtab=np.asarray(self.dtab, i64),
            own_level=np.asarray(self.own_level, i64),
            lrow=np.asarray(self.lrow, i64),
            lcuts=_lcuts(lcuts, self.level), level=self.level,
            woff=np.zeros(len(self.kind), i64),     # set by fit
            trig=np.asarray(self.trig, i64),
            pv_ptr=np.asarray(self.pv_ptr, i64),
            n_cols=np.asarray(self.n_cols, i64),
            pc_tab=np.asarray(self.pc_tab, i64),
            row_bound=np.asarray(self.row_bound, i64),
            windows=np.asarray(wins, i64), tab_total=int(self.tab),
            P=len(self.pv_ptr) - 1)


class PlanMixin:
    """The flat plan of a fit: views, cut tables, columns."""

    def _build_plan(self, views, cuts, m, C, lcuts=None):
        """The flat view table over every block p. With the level word
        (lcuts: its cut sets, the rows of _level_jobs), the first view in plan
        order keying a (block, window, dilation) owns its level block, the
        last block of that slot (own_level[di] = 1); lrow[v] is view v's cut
        set. Level off, lcuts is ignored."""
        level = bool(self.level_word)
        if level:
            _check_lcuts(lcuts, views, C)
        if _is_family(views):
            return self._build_family_plan(views, cuts, m, C, lcuts, level)
        lrows = ({j[0]: r for r, j in enumerate(_level_jobs(views, C))}
                 if level else {})
        wins = sorted({w for vp in views for (_, w, _) in vp})
        slot = {w: i for i, w in enumerate(wins)}
        lists = _PlanLists(m, level)
        flat = []
        at_fourier = 0
        for p in range(len(views)):
            lists.begin_side()
            for (s, w, ds) in views[p]:
                if s == "fourier":
                    cut = cuts[at_fourier]
                    at_fourier += 1
                    fk = 0
                    for b in range(WORD_LEN):
                        fk = fk * 2 + int(cut[b] < 0)
                else:
                    cut = np.zeros(WORD_LEN)
                    fk = 0
                flat.append(fk)
                lists.add_view(s, w, ds, p % C, 1 if p >= C else 0, cut,
                               slot[w])
                lists.lrow.append(lrows.get((p, p % C, w), -1))
                lists.add_slots(w, ds, 1, ())
            lists.end_side()
        return dict(lists.arrays(wins, WORD_LEN, lcuts),
                    flat=np.asarray(flat, np.int64))

    def _build_family_plan(self, views, cuts, m, C, lcuts=None, level=False):
        """The family layout's view table over the sides (side 1 the first
        difference). A view is one family (source, window, channels) with its
        merged dilations; a (view, dilation) owns nblk consecutive 256-column
        blocks starting at dcol / dtab, in the order channel 0 .. k-1 (the
        sorted channels), coarse, change flags, agreement flags (the
        cross-channel words of the plan's ``cross`` code, none when k = 1),
        then, on the first view of plan order keying its (side, channels,
        window, dilation), the level block (own_level[di] = 1). cuts[v] holds
        8 cuts per channel slot; lrow holds each channel slot's cut set.
        wptr, wref and n_wslots number the shared channel words
        (``_word_slots``; config.WORD_TABLE_MB caps their number)."""
        cross = int(self._cross_code())
        lrows = ({j[0]: r for r, j in enumerate(_level_jobs(views, C))}
                 if level else {})
        wins = sorted({w for vp in views for (_, w, _, _) in vp})
        slot = {w: i for i, w in enumerate(wins)}
        lists = _PlanLists(m, level)
        cptr, chs = [0], []
        at_fourier = 0
        for p in range(len(views)):
            lists.begin_side()
            for (s, w, ds, cs) in views[p]:
                chs.extend(cs)
                cptr.append(len(chs))
                if s == "fourier":
                    cut = cuts[at_fourier]
                    at_fourier += 1
                else:
                    cut = np.zeros(_FAMILY_CUTS)
                lists.add_view(s, w, ds, cs[0], 1 if p == 1 else 0, cut,
                               slot[w])
                lists.lrow.extend(lrows.get((p, int(c), w), -1) for c in cs)
                lists.add_slots(w, ds, _family_nblk(len(cs), cross), cs)
            lists.end_side()
        wptr, wref, n_wslots = _word_slots(
            lists.kind, lists.win, cptr, chs, lists.dptr, lists.dils,
            lists.vcut, lists.pv_ptr, m)
        i64 = np.int64
        return dict(lists.arrays(wins, _FAMILY_CUTS, lcuts),
                    cptr=np.asarray(cptr, i64), chs=np.asarray(chs, i64),
                    cross=cross, wptr=wptr, wref=wref, n_wslots=int(n_wslots))
