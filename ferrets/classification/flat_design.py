"""The flat CSR design and its centred operator."""

import numpy as np

from .. import config
from ..helpers.split import _split
from ..kernels.flat import (_fd_colmean, _fd_matvec, _fd_rmatvec,
                            _fd_rmatvec_parts)

_U16_BITS = 16      # bits of a uint16 id: the widest band of local ids


def _sub_widths(widths, bits):
    """(widths of the flat design's blocks, blocks per kept pseudo-channel
    block): with bits <= _U16_BITS a block wider than 1 << bits columns is
    cut into consecutive blocks of that width and a remainder (a zero-width
    block stays one block); above _U16_BITS every block stays whole."""
    widths = np.asarray(widths, dtype=np.int64)
    if bits > _U16_BITS:
        return widths.copy(), np.ones(len(widths), dtype=np.int64)
    band = 1 << int(bits)
    nbs = np.maximum(1, -(-widths // band))
    out = []
    for w, nb in zip(widths.tolist(), nbs.tolist()):
        out.extend([band] * (nb - 1))
        out.append(w - band * (nb - 1))
    return np.asarray(out, dtype=np.int64), nbs


class _FlatDesign:
    """The normalised design as blocks in two flat arrays: block p's row r
    is ids / vals[ptr[p, r]:ptr[p, r+1]] (local ids, ascending), global
    column off[p] + id. A block is a pseudo-channel's kept columns or, when
    that is wider than the id type reaches, one consecutive piece of them
    (group[p] names the pseudo-channel, P0 counts them): the sums over a
    row's entries run block by block, ascending, so the pieces change no
    value."""

    def __init__(self, n, widths, total_nnz, id_dtype, group=None):
        self.n = int(n)
        self.width = np.asarray(widths, dtype=np.int64)
        self.P = len(self.width)
        self.off = np.zeros(self.P + 1, dtype=np.int64)
        np.cumsum(self.width, out=self.off[1:])
        self.F = int(self.off[-1])
        self.ids = np.empty(int(total_nnz), dtype=id_dtype)
        self.vals = np.empty(int(total_nnz), dtype=np.float32)
        self.ptr = np.zeros((self.P, self.n + 1), dtype=np.int64)
        self.group = (np.arange(self.P, dtype=np.int64) if group is None
                      else np.asarray(group, dtype=np.int64))
        self.P0 = int(self.group[-1]) + 1 if self.P else 0

    def ranges(self, n_target):
        """Column ranges (rp, rj0, rj1): block, first and end local column.
        With fewer blocks than n_target each block is cut in proportion to
        its width; a range never crosses a block."""
        rp, rj0, rj1 = [], [], []
        split = self.P < n_target and self.F > 0
        for p in range(self.P):
            w = int(self.width[p])
            if w == 0:
                continue
            k = 1
            if split:
                k = min(w, max(1, int(round(n_target * w / self.F))))
            b = _split(w, k)
            for a in range(k):
                if b[a] < b[a + 1]:
                    rp.append(p)
                    rj0.append(int(b[a]))
                    rj1.append(int(b[a + 1]))
        return (np.asarray(rp, dtype=np.int64),
                np.asarray(rj0, dtype=np.int64),
                np.asarray(rj1, dtype=np.int64))


def _rmv_spans(rp, rj0, rj1, n_groups):
    """The column ranges (rp, rj0, rj1) cut into min(n_groups, ranges)
    consecutive runs, as a static schedule hands them to the threads, and
    each run's touching ranges of one block merged into one span: group k
    holds the spans gs[k]:gs[k + 1] of (sp, sj0, sj1)."""
    k = min(max(1, int(n_groups)), len(rp))
    b = _split(len(rp), k) if k else np.zeros(1, dtype=np.int64)
    gs, sp, sj0, sj1 = [0], [], [], []
    for g in range(k):
        for q in range(b[g], b[g + 1]):
            if (len(sp) > gs[-1] and sp[-1] == rp[q]
                    and sj1[-1] == rj0[q]):
                sj1[-1] = int(rj1[q])
            else:
                sp.append(int(rp[q]))
                sj0.append(int(rj0[q]))
                sj1.append(int(rj1[q]))
        gs.append(len(sp))
    return tuple(np.asarray(a, dtype=np.int64) for a in (gs, sp, sj0, sj1))


class _CenteredOp:
    """(X_S - 1 mu_S^T) on the rows S of a _FlatDesign, multi-column."""

    def __init__(self, D, rows, n_threads):
        self.D = D
        self.rows = np.ascontiguousarray(rows, dtype=np.int64)
        self.rp, self.rj0, self.rj1 = D.ranges(4 * n_threads)
        # groups of the column-range rmatvec, one per thread; spans rather
        # than ranges, so the kernel searches once per row and span
        self.gs, self.sp, self.sj0, self.sj1 = _rmv_spans(
            self.rp, self.rj0, self.rj1, n_threads)
        mu32 = np.zeros(D.F, dtype=np.float32)
        if len(self.rows) and len(self.rp):
            _fd_colmean(D.ptr, D.ids, D.vals, D.off, self.rows, self.rp,
                        self.rj0, self.rj1, np.float32(1.0 / len(self.rows)),
                        mu32)
        self.mu = mu32.astype(np.float64)
        self.n_blocks = int(max(1, min(len(self.rows), 4 * n_threads)))
        self.shape = (len(self.rows), D.F)
        # bands of consecutive blocks (<= RMV_BAND columns unless one block
        # is wider) for the partitioned transposed product
        bands, acc = [0], 0
        for p in range(D.P):
            w = int(D.width[p])
            if acc and acc + w > config.RMV_BAND:
                bands.append(p)
                acc = 0
            acc += w
        bands.append(D.P)
        self.bands = np.asarray(bands, dtype=np.int64)
        self.band_w = int(max(int(D.off[self.bands[k + 1]]
                                  - D.off[self.bands[k]])
                              for k in range(len(bands) - 1))) if D.P else 0
        self._scratch = None

    def matvec(self, V):
        out = self.raw_matvec(V)
        out -= (self.mu @ V)[None, :]
        return out

    def rmatvec(self, U):
        out = self.raw_rmatvec(U)
        out -= self.mu[:, None] * U.sum(axis=0)[None, :]
        return out

    def raw_matvec(self, V):
        """The uncentred product X_S V."""
        G = V.shape[1]
        G4 = G if G == 1 else -(-G // 4) * 4
        Vp = V
        if G4 != G:
            # a column count off a multiple of 4 leaves a scalar remainder in
            # the per-entry column loop, so V is padded with zero columns
            Vp = np.zeros((V.shape[0], G4))
            Vp[:, :G] = V
        out = np.empty((self.shape[0], G4))
        _fd_matvec(self.D.ptr, self.D.ids, self.D.vals, self.D.off, self.rows,
                   Vp, out, np.int64(self.n_blocks))
        return np.ascontiguousarray(out[:, :G]) if G4 != G else out

    def raw_rmatvec(self, U):
        """The uncentred product X_S^T U."""
        D = self.D
        G = U.shape[1]
        G4 = G if G <= 2 else -(-G // 4) * 4
        Up = U
        if G4 != G:
            Up = np.zeros((U.shape[0], G4))
            Up[:, :G] = U
        out = np.empty((D.F, G4))
        if len(self.rp):
            if (min(D.F, config.RMV_BAND) * G4 * 8
                    > config.RMV_SCRATCH_MB * 1e6):
                # a partition's scratch past the cache: column ranges, whose
                # sums (sub x G4, 256 KB) stay in L2. The band is sized by F
                # and RMV_BAND, not by the blocks it packs: the two kernels
                # round differently, so cutting a block into sub-blocks
                # (FLAT_BAND_BITS) must not change which kernel runs
                _fd_rmatvec(D.ptr, D.ids, D.vals, D.off, self.rows, Up,
                            self.gs, self.sp, self.sj0, self.sj1,
                            np.int64(max(8, 32768 // G4)), out)
            else:
                # fixed row partitions, each summing into its own scratch
                if self._scratch is None or self._scratch.shape[2] != G4:
                    self._scratch = np.empty((config.RMV_PARTS, self.band_w,
                                              G4))
                _fd_rmatvec_parts(D.ptr, D.ids, D.vals, D.off, self.rows, Up,
                                  self.bands, np.int64(config.RMV_PARTS),
                                  self._scratch, out)
        return np.ascontiguousarray(out[:, :G]) if G4 != G else out


class _FlatRaw:
    """The uncentred product X_S V of a _FlatDesign's rows (the hold-out's
    validation rows): ``order`` holds the rows, matvec(V) one row per
    entry of ``order``."""

    def __init__(self, D, rows, n_threads):
        self.D = D
        self.order = np.ascontiguousarray(rows, dtype=np.int64)
        self.n_blocks = np.int64(max(1, min(len(self.order), 4 * n_threads)))

    def matvec(self, V):
        D = self.D
        out = np.empty((len(self.order), V.shape[1]))
        _fd_matvec(D.ptr, D.ids, D.vals, D.off, self.order,
                   np.ascontiguousarray(V), out, self.n_blocks)
        return out
