"""The encoded training design of a batch fit and the decoders the Gram and
LSQR heads read it through."""

import numpy as np

from ..kernels.codec import (_band_counts, _dec_batch, _dec_batch_bands,
                             _dec_slab_count, _dec_slab_fill)


class _PieceBufs:
    """The id and value buffers of the pieces' CSR, grown on demand and
    reused by every piece of every block of a pass, which avoids page-faulting
    a fresh buffer inside every kernel call."""

    def __init__(self, id_dtype):
        self.ids = np.empty(0, dtype=id_dtype)
        self.vals = np.empty(0, dtype=np.float32)

    def get(self, nk):
        """(ids[:nk], vals[:nk]); the contents are the caller's to write."""
        if self.ids.size < nk:
            size = max(nk, self.ids.size + self.ids.size // 4)
            self.ids = np.empty(size, dtype=self.ids.dtype)
            self.vals = np.empty(size, dtype=np.float32)
        return self.ids[:nk], self.vals[:nk]


class _PieceCursor:
    """One block's encoded rows (a _Part) decoded to CSR pieces of
    consecutive kept columns [c0, c1): a per-row cursor into the codes, the
    entries of a piece being those the cursor meets before the first kept id
    >= c1. A piece holds what _block_csr's CSR holds for its columns, entry
    for entry (block-local ids, the same float32 values). The count pass is
    skipped when the piece is the whole block or the block's last piece (the
    row counts are then kptr's less the entries already decoded)."""

    def __init__(self, part, remap, idf, kptr, norm, tf_store, width, bufs):
        n = len(norm)
        self.part = part
        self.remap = remap
        self.idf = idf
        self.norm = norm
        self.tf_store = tf_store
        self.width = int(width)
        self.bufs = bufs
        self.cur_q = part.cptr[:-1].copy()
        self.cur_g = np.full(n, -1, dtype=np.int64)
        self.cur_qe = part.eptr[:-1].copy()
        self.total = np.diff(kptr)
        self.done = np.zeros(n, dtype=np.int64)
        self.ptr = np.zeros(n + 1, dtype=np.int64)
        self.cnt = np.empty(n, dtype=np.int64)
        self.pos = 0

    def piece(self, c0, c1):
        """(ptr, ids, vals): the CSR of kept columns [c0, c1) (c1 >= width:
        the rest of the block). c0 must be where the previous piece ended.
        The arrays are valid until the next call."""
        if c0 != self.pos or c1 <= c0:
            raise ValueError(f"piece [{c0}, {c1}) does not continue the "
                             f"block at column {self.pos}")
        part = self.part
        if c1 >= self.width:
            cnt = self.total - self.done
        else:
            cnt = self.cnt
            _dec_slab_count(part.cptr, part.eptr, part.code_d,
                            part.code_c, part.esc, self.remap, self.cur_q,
                            self.cur_g, self.cur_qe, np.int64(c1), cnt)
        np.cumsum(cnt, out=self.ptr[1:])
        nk = int(self.ptr[-1])
        ids, vals = self.bufs.get(nk)
        if nk:
            # an empty piece skips the fill and leaves the cursor where it
            # is: the count pass does not move it, so the next piece starts
            # from the same codes. Nothing reads the cursor after a block's
            # last piece (its counts are kptr's less self.done), so a walk
            # need not end on a cursor committed to every row's end
            _dec_slab_fill(part.cptr, part.eptr, part.code_d,
                           part.code_c, part.esc, self.remap,
                           self.cur_q, self.cur_g, self.cur_qe,
                           np.int64(c1), self.tf_store, self.idf,
                           self.norm, self.ptr, ids, vals)
        self.done += cnt
        self.pos = c1
        return self.ptr, ids, vals


class _Design:
    """The encoded training design of a batch fit: one encoded block per
    pseudo-channel (None where the RAM budget dropped it), the kept-column
    maps, idf, per-block kept pointers and row norms. The blocks are decoded
    to CSR on demand; a dropped block is encoded again and counted in
    ``recomputed``. It reads the transformer's fit-time tables, so it lives
    only inside a fit: a context manager, whose exit drops the tables (and
    the blocks) and, after an error-free block, declares the transformer
    fitted with ``metadata`` when one was given.

    Read by the heads: n (rows), n_blocks (pseudo-channels), n_features,
    offsets (feature_offsets_), id_dtype (block-local id type), tf_store (the
    stored term-frequency table), nnz, blocks, remaps, idfs, kptrs, norm."""

    def __init__(self, transformer, X, blocks, remaps, idfs, kptrs, norm,
                 trig):
        self.transformer = transformer
        self.X = X
        self.n = int(X.shape[0])
        self.blocks = blocks
        self.remaps = remaps
        self.idfs = idfs
        self.kptrs = kptrs
        self.norm = norm
        self.trig = trig
        self.n_blocks = len(blocks)
        self.n_features = int(transformer.n_features_)
        self.offsets = transformer.feature_offsets_
        self.id_dtype = transformer._id_bt
        self.tf_store = transformer._tf_store
        self.metadata = None
        self.recomputed = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.blocks = []
        self.transformer._drop_scratch()
        if exc_type is None and self.metadata is not None:
            self.transformer._mark_fitted(self.metadata)
        return False

    @property
    def nnz(self):
        """Kept entries of the whole design."""
        return int(sum(int(k[-1]) for k in self.kptrs))

    def release(self, p):
        """Drop block p's encoded rows."""
        self.blocks[p] = None

    def part(self, p):
        """Block p's encoded rows: the held _Part, or, when the design store
        dropped it, a fresh encoding counted in ``recomputed``."""
        part = self.blocks[p]
        if part is None:
            part = self.transformer._encode_pc(
                self.X, np.arange(self.X.shape[0], dtype=np.int64), p,
                self.trig)
            self.recomputed += 1
        return part

    def cursor(self, p, bufs):
        """A _PieceCursor over block p. It takes one ``part`` per call, so a
        dropped block is re-encoded once per cursor (the Gram route builds one
        per block per pass)."""
        part = self.part(p)
        return _PieceCursor(part, self.remaps[p], self.idfs[p], self.kptrs[p],
                            self.norm, self.tf_store, len(self.idfs[p]), bufs)

    def csr(self, p, ids_buf=None, vals_buf=None):
        """Pseudo-channel p's normalised tf-idf CSR (local kept ids)."""
        part = self.part(p)
        kptr = self.kptrs[p]
        nk = int(kptr[-1])
        ids = (np.empty(nk, dtype=self.id_dtype) if ids_buf is None
               else ids_buf[:nk])
        vals = (np.empty(nk, dtype=np.float32) if vals_buf is None
                else vals_buf[:nk])
        if nk:
            _dec_batch(part.cptr, part.eptr, part.code_d, part.code_c,
                       part.esc, self.remaps[p], kptr, self.tf_store,
                       self.idfs[p], self.norm, ids, vals)
        return kptr, ids, vals

    def band_counts(self, p, bits, nb):
        """(n, nb) int64: the kept entries of every row of block p in each of
        its nb bands of 1 << bits columns. The block must be held."""
        part = self.blocks[p]
        if part is None:
            raise RuntimeError("the LSQR route keeps every encoded "
                               "block (keep_all)")
        cnt = np.zeros((self.n, nb), dtype=np.int64)
        _band_counts(part.cptr, part.eptr, part.code_d, part.code_c,
                     part.esc, self.remaps[p], np.int64(bits), cnt)
        return cnt

    def decode_bands(self, p, bits, starts, ids, vals):
        """Block p decoded into ids and vals band by band: starts[r, b] is
        where row r's entries of band b begin (ids local to the band). The
        block must be held."""
        part = self.blocks[p]
        _dec_batch_bands(part.cptr, part.eptr, part.code_d, part.code_c,
                         part.esc, self.remaps[p], np.int64(bits), starts,
                         self.tf_store, self.idfs[p], self.norm, ids, vals)
