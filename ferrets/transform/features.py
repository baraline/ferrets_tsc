"""Kept columns, the batch counting pass and the public transform (the
weighted rows the head scores)."""

import numpy as np
import scipy.sparse as sp

from .. import config
from ..constants import _MODE_GLOBAL, _MODE_SCORE, head_col_none
from ..kernels.codec import _dec_counts, _sq_pass
from ..transform.encoding import _smooth_idf, _tf_tables
from ..transform.run import _buffer_types
from .design import _Design


def _pieces_to_csr(pieces, sq, row_nnz, n, F):
    """Pass 2 of FeaturesMixin._rows: each piece is written in place into the
    CSR (a row's pseudo-channels in column order; ``pieces`` is emptied as it
    goes), then every row is scaled by its norm sqrt(sq) (1 for a zero row)."""
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(row_nnz, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    data = np.empty(int(indptr[-1]), dtype=np.float64)
    cursor = indptr[:-1].copy()
    for i in range(len(pieces)):
        r0, k, cols, vals = pieces[i]
        pieces[i] = None        # placed: freed before the next piece
        r1 = r0 + len(k)
        dest = np.repeat(cursor[r0:r1] - (np.cumsum(k) - k), k)
        dest += np.arange(len(cols))
        indices[dest] = cols
        data[dest] = vals
        cursor[r0:r1] += k
    del cols, vals, dest
    norm = np.sqrt(sq)
    norm[sq <= 0] = 1.0
    rstep = max(1, int(config.TRANSFORM_SLICE_SLOTS)
                // max(1, int(row_nnz.max())))
    for a in range(0, n, rstep):
        b = min(n, a + rstep)
        data[indptr[a]:indptr[b]] /= np.repeat(norm[a:b], row_nnz[a:b])
    Z = sp.csr_matrix((data, indices, indptr), shape=(n, F))
    Z.sort_indices()
    return Z


# The seam between the two estimators: three private entry points of FERRETS
# (transformer.py) are all FERRETSClassifier calls.
#   _fit_design(X, ...)  fits the transform and returns the encoded training
#                        design (transform/design.py) the batch heads learn
#                        from; built by _batch_design below.
#   _fit_chunks(X, ...)  the same fit, the design delivered as cached chunks
#                        (transform/chunks.py) for the streaming head.
#   _score(X, coef)      predict: keys the series and applies the weights.
# Both fit entries return a context manager that ends the fit: it drops the
# fit-time tables and, when the classifier passes its metadata, marks the
# transformer fitted. The heads read the object they are handed, not the
# transformer's private attributes.
class FeaturesMixin:
    """The idf, the kept columns, the batch design and the weighted rows.

    Sets: feature_offsets_, idf_, n_features_.
    Reads: _head_col, _id_bt, _plan, _tf_store, _tf_tab, idf_, n_timepoints_
    (_rows and _score: _head_col, _plan, feature_offsets_, idf_, n_features_;
    they set nothing but the keying schedule's cache, dropped on return, so
    threads may share one fitted model)."""

    def _rows(self, X):
        """The weighted rows phi(X) the head scores: a float64 CSR of shape
        (n, n_features_), sorted indices. Each row holds the tf of the kept
        columns (1 + ln c, or c without sublinear_tf; times idf_ under
        use_idf), counted under the fitted dictionaries and cuts, scaled to
        unit l2 norm; a series with no kept key gets a zero row. Columns
        follow feature_offsets_ (one span per block p, see plan.py). The
        columns were chosen from the training series alone, with no label.

        Pass 1 (_row_pieces): the mode-0 counts of each keying chunk, mapped
        to head columns through _head_col and weighted by tf_tab[c] * idf_,
        the scorer's arithmetic (kernels/emit.py). Pass 2 (_pieces_to_csr):
        the pieces are placed in the CSR and every row is scaled to its
        norm."""
        n = X.shape[0]
        F = int(self.n_features_)
        if n == 0:
            return sp.csr_matrix((0, F), dtype=np.float64)
        pieces, sq, row_nnz = self._row_pieces(X)
        return _pieces_to_csr(pieces, sq, row_nnz, n, F)

    def _design_rows(self, design):
        """_rows of the training series, decoded from a batch design that
        holds every block (``keep_all``) instead of keyed again: the same
        pieces, in the same order, with the same float64 arithmetic."""
        pl = self._plan
        n = design.norm.shape[0]
        sq = np.zeros(n)
        row_nnz = np.zeros(n, dtype=np.int64)
        pieces = []                 # (first row, kept per row, cols, vals)
        for p in range(int(pl["P"])):
            part, kptr = design.blocks[p], design.kptrs[p]
            off = np.int64(self.feature_offsets_[p])
            step = max(1, int(config.TRANSFORM_SLICE_SLOTS)
                       // max(1, int(pl["row_bound"][p])))
            for a in range(0, n, step):
                b = min(n, a + step)
                k = np.diff(kptr[a:b + 1])
                cols = np.empty(int(kptr[b] - kptr[a]), dtype=np.int32)
                cnt = np.empty(len(cols), dtype=np.int32)
                if len(cols):
                    _dec_counts(part.cptr, part.eptr, part.code_d,
                                part.code_c, part.esc, design.remaps[p],
                                kptr, off, np.int64(a), np.int64(b), cols, cnt)
                vals = self._tf_tab[cnt] * self.idf_[cols]
                r = np.repeat(np.arange(b - a), k)
                sq[a:b] += np.bincount(r, weights=vals * vals, minlength=b - a)
                row_nnz[a:b] += k
                pieces.append((a, k, cols, vals))
        return _pieces_to_csr(pieces, sq, row_nnz, n, int(self.n_features_))

    def _row_pieces(self, X):
        """Pass 1 of _rows: the weighted entries of every row, per
        pseudo-channel and keying chunk, as pieces (first row, kept per row,
        columns, values), with the rows' sums of squares and kept counts."""
        pl = self._plan
        n, _, m = X.shape
        trig = self._trig(m)
        tf_tab = _tf_tables(m, bool(pl["sublinear_tf"]))[1]
        id_bt, cnt_bt = _buffer_types(pl, m)
        hc = self._head_col
        none = head_col_none(hc.dtype)
        rows_all = np.arange(n, dtype=np.int64)
        slot = np.dtype(id_bt).itemsize + np.dtype(cnt_bt).itemsize
        sq = np.zeros(n)
        row_nnz = np.zeros(n, dtype=np.int64)
        pieces = []                 # (first row, kept per row, cols, vals)
        ids_buf = cnt_buf = None
        for p in range(int(pl["P"])):
            v0, v1 = int(pl["pv_ptr"][p]), int(pl["pv_ptr"][p + 1])
            rb = max(1, int(pl["row_bound"][p]))
            chunk = max(1, min(n, int(config.TRANSFORM_CHUNK_MB * 1e6
                                      // (rb * slot))))
            step = max(1, int(config.TRANSFORM_SLICE_SLOTS) // rb)
            base = int(pl["pc_tab"][p])
            off = int(self.feature_offsets_[p])
            if ids_buf is None or ids_buf.size < chunk * rb:
                ids_buf = np.empty(chunk * rb, dtype=id_bt)
                cnt_buf = np.empty(chunk * rb, dtype=cnt_bt)
            for s in range(0, n, chunk):
                e = min(n, s + chunk)
                ids = ids_buf[:(e - s) * rb].reshape(e - s, rb)
                cnt = cnt_buf[:(e - s) * rb].reshape(e - s, rb)
                nnz = np.zeros(e - s, dtype=np.int64)
                self._run(X, rows_all[s:e], v0, v1, _MODE_GLOBAL, trig,
                          bufs=(ids, cnt, nnz))
                for a in range(0, e - s, step):
                    b = min(e - s, a + step)
                    filled = np.arange(rb)[None, :] < nnz[a:b, None]
                    h = hc[base + ids[a:b][filled].astype(np.int64)]
                    kept = h != none
                    r = np.repeat(np.arange(b - a), nnz[a:b])[kept]
                    cols = (h[kept].astype(np.int64) + off).astype(np.int32)
                    vals = tf_tab[cnt[a:b][filled][kept]] * self.idf_[cols]
                    k = np.bincount(r, minlength=b - a)
                    sq[s + a:s + b] += np.bincount(r, weights=vals * vals,
                                                   minlength=b - a)
                    row_nnz[s + a:s + b] += k
                    pieces.append((s + a, k, cols, vals))
        return pieces, sq, row_nnz

    def _idf(self, df, n_docs, dtype):
        """The column weights of kept columns with document frequencies df:
        the smooth idf (``use_idf``), else 1, in ``dtype``."""
        if self.use_idf:
            return _smooth_idf(df, n_docs, dtype)
        return np.ones(np.asarray(df).shape[0], dtype=np.dtype(dtype))

    def _set_keep(self, p, keep):
        base = int(self._plan["pc_tab"][p])
        self._head_col[base + keep] = np.arange(len(keep)).astype(
            self._head_col.dtype)

    def _finish_features(self, keeps, idfs):
        pl = self._plan
        widths = np.asarray([len(k) for k in keeps], dtype=np.int64)
        off = np.zeros(len(widths) + 1, dtype=np.int64)
        np.cumsum(widths, out=off[1:])
        self.feature_offsets_ = off
        self.n_features_ = int(off[-1])
        for p in range(len(keeps)):
            v0, v1 = int(pl["pv_ptr"][p]), int(pl["pv_ptr"][p + 1])
            pl["woff"][v0:v1] = off[p]
        if self.n_features_ == 0:
            raise ValueError(
                f"min_doc_freq={self.min_doc_freq} prunes every word (no "
                f"column is shared by that many training series); lower it -- "
                f"on very short series (m={self.n_timepoints_}) even 2 can be "
                f"too strict")
        self.idf_ = (np.concatenate(idfs).astype(np.float32) if idfs
                     else np.empty(0, np.float32))

    def _batch_design(self, X, trig, stats, keep_all=False, ram_mb=0.0):
        """The batch counting pass. Per pseudo-channel: mode-0 counts of
        every training series, encoded; document frequencies; pruning; the
        float32 smooth idf; the rows' tf-idf norms (squares carried across
        pseudo-channels). It then fixes the kept columns and the idf. The
        encoded blocks stay in RAM up to ram_mb (all of them when keep_all);
        _Design recomputes the others. Returns the _Design."""
        n = X.shape[0]
        pl = self._plan
        rows = np.arange(n, dtype=np.int64)
        floor = int(self.min_doc_freq)
        sq = np.zeros(n)
        blocks, keeps, remaps, idfs, kptrs = [], [], [], [], []
        budget = float(ram_mb) * 1e6
        held = 0
        n_kept_blocks = 0
        for p in range(pl["P"]):
            ncols = int(pl["n_cols"][p])
            df = np.zeros(ncols, dtype=np.int64)
            part = self._encode_pc(X, rows, p, trig, df=df)
            keep = np.flatnonzero(df >= floor)
            remap = np.full(ncols, -1, dtype=np.int32)
            remap[keep] = np.arange(len(keep), dtype=np.int32)
            idf = self._idf(df[keep], n, np.float32)
            kcnt = np.zeros(n, dtype=np.int64)
            if len(keep) and part.nnz:
                _sq_pass(part.cptr, part.eptr, part.code_d, part.code_c,
                         part.esc, remap, self._tf_store, idf, sq, kcnt)
            kptr = np.zeros(n + 1, dtype=np.int64)
            np.cumsum(kcnt, out=kptr[1:])
            self._set_keep(p, keep)
            keeps.append(keep)
            remaps.append(remap)
            idfs.append(idf)
            kptrs.append(kptr)
            if held + part.nbytes <= budget or keep_all:
                blocks.append(part)
                held += part.nbytes
                n_kept_blocks += 1
            else:
                blocks.append(None)
            del part
        self._release_buffers()
        stats["blocks_in_ram"] = n_kept_blocks
        stats["bytes_held"] = int(held)
        norm = np.sqrt(sq)
        norm[sq <= 0] = 1.0
        self._finish_features(keeps, idfs)
        return _Design(self, X, blocks, remaps, idfs, kptrs, norm, trig)

    def _score(self, X, coef):
        """The scores transform(X) @ coef (n, coef.shape[1]) float64 without
        assembling the rows: the keying kernel accumulates the head's weights
        while it counts. X is a float64 collection of the fitted shape."""
        n, _, m = X.shape
        acc = np.zeros((n, coef.shape[1]))
        sq = np.zeros(n)
        if n == 0:
            return acc
        trig = self._trig(m)
        # the tf the model was fitted with (fit stores it in the plan)
        tf_tab = _tf_tables(m, bool(self._plan["sublinear_tf"]))[1]
        V = int(self._plan["kind"].shape[0])
        try:
            # every row in one call: scoring holds one int64 per row besides
            # acc and sq, the kernels' scratch is per row block
            self._run(X, np.arange(n, dtype=np.int64), 0, V, _MODE_SCORE,
                      trig, head_col=self._head_col, tf_tab=tf_tab,
                      idf=self.idf_, W=coef, acc=acc, sq=sq)
        finally:
            # the keying schedule does not outlive the call
            self.__dict__.pop("_chain_cache", None)
        norm = np.sqrt(sq)
        norm[sq <= 0] = 1.0
        return acc / norm[:, None]

