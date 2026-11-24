"""The encoded training set as chunks for the streaming head: the sizing probe
and the counting pass that build the chunk cache, the object that carries
them, and the feed that decodes a chunk for the head's passes."""

import shutil
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import config
from ..constants import head_col_none
from ..kernels.codec import _dec_stream
from .chunk_cache import _build_chunk, _ChunkCache


class _ChunkFeed:
    """The chunks of the epochs, one at a time: a chunk comes from the cache
    (read ahead from disk by one worker thread), or is encoded again when
    there is no cache, and is decoded into one reusable pair of buffers."""

    def __init__(self, chunks, seq):
        self.chunks = chunks
        self.tr, self.X, self.members = (chunks.transformer, chunks.X,
                                         chunks.members)
        self.cache, self.seq = chunks.cache, seq
        self.idf_c, self.trig = chunks.idf_c, chunks.trig
        tr, cache = self.tr, self.cache
        n_ch = len(self.members)
        self.hc = tr._head_col
        self.hc_none = np.int64(head_col_none(self.hc.dtype))
        self.pc_tab = tr._plan["pc_tab"]
        self.inv = np.ones(self.X.shape[0])
        self.seen = np.zeros(n_ch, dtype=bool)
        self.starts = [None] * n_ch
        self.cap = 1
        if cache is not None:
            for k in range(n_ch):
                self.starts[k] = self._row_starts(cache.chunks[k])
                self.cap = max(self.cap, int(self.starts[k][-1]))
        self.buf = None
        self.pool = ThreadPoolExecutor(1) if (
            cache is not None and not chunks.resident
            and cache.mode == "disk"
        ) else None
        self.pending = {}

    @staticmethod
    def _row_starts(ch):
        # unpruned counts per row: an upper bound (the decode's gaps)
        ptr = np.zeros(len(ch.rows) + 1, dtype=np.int64)
        np.cumsum((ch.cptr[:, 1:] - ch.cptr[:, :-1]).sum(axis=0),
                  out=ptr[1:])
        return ptr

    def _buffer(self, nnz):
        bb = self.buf
        if bb is None or bb[0].shape[0] < nnz:
            bb = (np.empty(max(nnz, self.cap), dtype=np.int32),
                  np.empty(max(nnz, self.cap), dtype=np.float32))
            self.buf = bb
        return bb

    def fetch(self, k):
        """(chunk, arrays) of chunk k; a prefetched disk read resolves."""
        if self.cache is None:
            ch, arrs = _build_chunk(self.tr, self.X, self.members[k],
                                    self.trig)
            self.starts[k] = self._row_starts(ch)
            return ch, arrs
        if self.pool is not None:
            return self.cache.chunks[k], self.pending.pop(k).result()
        return self.cache.chunks[k], self.cache.read(k)

    def repeats(self, j):
        """Whether step j visits the chunk that step j - 1 left in the
        buffer: decoded rows, inverse norms and flags are then unchanged."""
        return j > 0 and self.seq[j] == self.seq[j - 1]

    def prefetch(self, j):
        """Start reading the chunk of epoch step j from disk (a repeat of the
        previous step's chunk is not fetched)."""
        if (self.pool is not None and j < len(self.seq)
                and not self.repeats(j)):
            k = int(self.seq[j])
            if k not in self.pending:
                self.pending[k] = self.pool.submit(self.cache.read, k)

    def decode(self, k, ch, arrs):
        """Chunk k into the buffer: (rs, re, ids, vals)."""
        ptr = self.starts[k]
        ids_b, vals_b = self._buffer(int(ptr[-1]))
        rs = ptr[:-1]
        re = np.empty(len(ch.rows), dtype=np.int64)
        hv = bool(self.seen[k])
        self.seen[k] = True
        _dec_stream(ch.cptr, ch.eptr, arrs[0], arrs[1], arrs[2], self.pc_tab,
                    self.hc, self.hc_none, self.tr.feature_offsets_,
                    self.tr._tf_tab, self.idf_c, ch.rows, self.inv, hv, rs,
                    re, ids_b, vals_b)
        return rs, re, ids_b, vals_b

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)


class _Chunks:
    """The encoded training set of a streaming fit, cut into chunks.

    Read by the streaming head: members (each chunk's training row indices),
    n_chunks, n_features, resident (the cache holds one chunk, decoded once)
    and feed(seq). It reads the transformer's fit-time tables, so it lives
    only inside a fit: a context manager, whose exit removes the cache (on
    every exit path) and the tables and, after an error-free block, declares
    the transformer fitted with ``metadata`` when one was given."""

    def __init__(self, transformer, X, members, cache, idf_c, trig):
        self.transformer = transformer
        self.X = X
        self.members = members
        self.cache = cache
        self.idf_c = idf_c
        self.trig = trig
        self.n_chunks = len(members)
        self.n_features = int(transformer.n_features_)
        self.resident = cache is not None and self.n_chunks == 1
        self.metadata = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        finally:
            self.transformer._drop_scratch()
        if exc_type is None and self.metadata is not None:
            self.transformer._mark_fitted(self.metadata)
        return False

    def feed(self, seq):
        """A feed of the chunks in the visiting sequence seq (chunk indices,
        one per step of the epochs). The caller closes it."""
        return _ChunkFeed(self, seq)

    def close(self):
        """Remove the cache (its directory, when on disk)."""
        if self.cache is not None:
            self.cache.close()


def _stream_chunks(tr, X, trig, stats):
    """The chunks of transformer ``tr`` over the training collection X: one
    transform pass encodes every chunk's mode-0 counts (all pseudo-channels)
    into the cache and counts the document frequencies, so the kept columns,
    the idf and the cache come out of the same pass. Sets stats cache_mode,
    chunk_rows, t_probe, t_cache_pass, nnz, cache_bytes and disk_peak_bytes.
    Returns the _Chunks."""
    tt = time.perf_counter
    t0 = tt()
    n = X.shape[0]
    order0 = np.random.RandomState(tr.seed_).permutation(n).astype(np.int64)
    mode, chunk_rows = _cache_plan(tr, X, order0, trig)
    stats["cache_mode"] = mode
    stats["chunk_rows"] = chunk_rows
    stats["t_probe"] = tt() - t0
    members = [np.ascontiguousarray(order0[s:s + chunk_rows])
               for s in range(0, n, chunk_rows)]
    cache = _ChunkCache(mode) if mode != "none" else None
    try:
        idf_c = _cache_pass(tr, X, members, cache, trig, stats)
        return _Chunks(tr, X, members, cache, idf_c, trig)
    except BaseException:
        if cache is not None:
            cache.close()
        raise


def _cache_plan(tr, X, order0, trig):
    """(cache mode, rows per chunk): the unpruned entries per row of the
    first rows of the permutation order0 give the chunk size and the cache
    estimate; the mode is "ram", "disk" or "none" (recompute every epoch)."""
    n = X.shape[0]
    n_pr = int(min(n, config.STREAM_PROBE_ROWS))
    probe = np.ascontiguousarray(order0[:n_pr])
    nnz_pr = sum(tr._encode_pc(X, probe, p, trig).nnz
                 for p in range(tr._plan["P"]))
    per_row = max(1.0, nnz_pr / max(1, n_pr))
    chunk_rows = int(max(1, min(n, config.STREAM_CHUNK_NNZ // per_row)))
    est = config.STREAM_EST_BYTES * per_row * n
    mode = "ram"
    if est > config.STREAM_RAM_MB * 1e6:
        where = tempfile.gettempdir()
        free = float(shutil.disk_usage(where).free)
        mode = "disk" if est <= config.STREAM_DISK_FRAC * free else "none"
        if mode == "none":
            warnings.warn(
                f"the streaming cache would take ~{est / 1e9:.1f} GB, "
                f"past {config.STREAM_DISK_FRAC} of the "
                f"{free / 1e9:.1f} GB free at {where}; recomputing "
                "the transform every epoch",
                RuntimeWarning)
    return mode, chunk_rows


def _cache_pass(tr, X, members, cache, trig, stats):
    """The counting pass: encode every chunk (into the cache, when there is
    one) and count the document frequencies, then fix the kept columns and
    the idf. Returns the idf of every kept column, concatenated."""
    tt = time.perf_counter
    n = X.shape[0]
    P = tr._plan["P"]
    dfs = [np.zeros(int(tr._plan["n_cols"][p]), dtype=np.int64)
           for p in range(P)]
    t1 = tt()
    for rws in members:
        ch, arrs = _build_chunk(tr, X, rws, trig, dfs)
        if cache is not None:
            cache.put(ch, *arrs)
        del arrs
    if cache is not None:
        cache.seal()
        tr._release_buffers()
    keeps, dfk = [], []
    for p in range(P):
        keep = np.flatnonzero(dfs[p] >= int(tr.min_doc_freq))
        keeps.append(keep)
        dfk.append(dfs[p][keep])
        tr._set_keep(p, keep)
    del dfs
    idfs = [tr._idf(d, n, np.float64) for d in dfk]
    tr._finish_features(keeps, idfs)
    stats["nnz"] = int(sum(int(d.sum()) for d in dfk))
    stats["t_cache_pass"] = tt() - t1
    stats["cache_bytes"] = 0 if cache is None else int(cache.bytes)
    stats["disk_peak_bytes"] = (0 if cache is None
                                else cache.disk_bytes())
    return np.concatenate(idfs) if idfs else np.empty(0)
