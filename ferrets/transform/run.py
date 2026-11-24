"""The keying kernels' single entry and the row buffers."""

import functools

import numba
import numpy as np

from .. import config
from ..constants import (_MODE_GLOBAL, _MODE_SCORE, _NO_ACC, _NO_HC, _NO_IDF,
                         _NO_IDS, _NO_VEC, _NO_W, _U16_NONE, N_COEF,
                         head_col_none)
from ..kernels.chain import _key_univariate_chains
from ..kernels.codec import _df_buffers
from ..kernels.window import _fill_trig
from ..kernels.words_family import _key_families
from ..kernels.words_univariate import _key_univariate
from ..transform.chains import chain_pairs
from ..transform.encoding import _concat_parts, _encode_buffers


def _buffer_types(plan, m):
    """(id type, count type) of the keying buffers: local column ids (< a
    block's column count) and counts (<= m), uint16 when they fit, which
    halves the bytes written per slot against int32. One id type serves every
    block, uint16 only when the widest block's column count fits; it also
    holds the batch design's local kept ids and the head-column table (none =
    _U16_NONE or -1)."""
    return (np.uint16 if int(plan["n_cols"].max()) <= _U16_NONE
            else np.int32,
            np.uint16 if m <= 65535 else np.int32)


@functools.lru_cache(maxsize=4)
def _trig_tables(windows, m):
    """(cos, sin) tables of the window lengths ``windows`` (a tuple) for
    series of length m. Shared by every call with the same key and never
    written after the build: the kernels only read them."""
    nw = len(windows)
    cos_t = np.zeros((max(nw, 1), N_COEF, m))
    sin_t = np.zeros((max(nw, 1), N_COEF, m))
    _fill_trig(np.asarray(windows, dtype=np.int64), cos_t, sin_t)
    return cos_t, sin_t


class TransformMixin:
    """The keying kernels' entry and the row buffers.

    Sets: _chain_cache, _cnt_bt, _id_bt, _tbuf.
    Reads: _plan."""

    def _trig(self, m):
        return _trig_tables(tuple(self._plan["windows"].tolist()), int(m))

    def _word_args(self, pl, m, row_tile):
        """(wptr, wref, n_slots) the family kernel reads: the plan's channel-
        word numbers (transform/plan.py ``_word_slots``) and the number of
        them its table may hold at this row tile, min(n_wslots,
        int(WORD_TABLE_MB x 1e6 // (m x row_tile))), so a block's table of
        row_tile rows (at most) x slots x m one-byte words stays within the
        budget; the plan's less used numbers beyond it count as unshared.
        For a plan without word numbers, empty arrays and 0 slots: the
        kernel reads none."""
        if "n_wslots" not in pl:
            none = np.empty(0, dtype=np.int64)
            return none, none, np.int64(0)
        cap = int(config.WORD_TABLE_MB * 1e6
                  // (max(int(m), 1) * max(int(row_tile), 1)))
        return (pl["wptr"], pl["wref"],
                np.int64(max(0, min(int(pl["n_wslots"]), cap))))

    def _row_tile(self, mode):
        """Rows per tile of the keying kernels' loop interchange: the
        PREDICT_ROW_TILE knob (at least 1) in mode 1, where the head's
        weights are read, and 1 in mode 0, which reads none (and would open
        one stream of writes per row of a tile)."""
        if mode == _MODE_SCORE:
            return max(1, int(config.PREDICT_ROW_TILE))
        return 1

    def _chain_pairs(self, v0, v1, m, mode, row_tile, n_heads, ids, cnt):
        """The chain-major schedule of a univariate keying call
        (transform.chains.chain_pairs; None when it stores no dilation),
        cached per call shape and budgets for the current plan (a refit's
        new plan drops the cache). The cache lives for one call: predict,
        transform, the decoder and a fit (through transformer._drop_scratch)
        drop it when they return, so no schedule reaches the fitted model
        or its pickle. Nothing enters the plan."""
        cache = self.__dict__.get("_chain_cache")
        if cache is None or cache[0] is not self._plan:
            cache = self._chain_cache = (self._plan, {})    # _FIT_SCRATCH
        key = (int(v0), int(v1), int(m), int(mode), int(row_tile),
               int(n_heads), ids.dtype.itemsize, cnt.dtype.itemsize,
               float(config.KEY_SCRATCH_MB), float(config.RESULT_STORE_MB))
        if key not in cache[1]:
            cache[1][key] = chain_pairs(
                self._plan, int(v0), int(v1), int(m), int(mode),
                int(row_tile), int(n_heads), ids.dtype.itemsize,
                cnt.dtype.itemsize, float(config.KEY_SCRATCH_MB),
                float(config.RESULT_STORE_MB))
        return cache[1][key]

    def _run(self, X, rows, v0, v1, mode, trig, head_col=None, tf_tab=None,
             idf=None, W=None, bufs=None, acc=None, sq=None):
        """Key views v0..v1-1 of ``rows`` with the kernel of the plan's
        layout; returns nnz, the per-row entry count of mode 0. Mode
        _MODE_GLOBAL writes (local column, count) pairs into bufs = (ids,
        cnt, nnz) and uses no other argument. Mode _MODE_SCORE adds the rows'
        scores against W to acc and their squared norms to sq, through
        head_col, tf_tab and idf. AblationTransformer._run copies this
        argument preparation: keep them in step."""
        pl = self._plan
        n_rows = len(rows)
        nb = int(max(1, min(n_rows, 4 * numba.get_num_threads())))
        if mode == _MODE_SCORE:
            ids = cnt = _NO_IDS
            nnz = np.zeros(n_rows, dtype=np.int64)
        else:
            ids, cnt, nnz = bufs
            acc, sq, tf_tab, idf, W = _NO_ACC, _NO_VEC, _NO_VEC, _NO_IDF, _NO_W
        if head_col is None:
            head_col = _NO_HC
        hc_none = head_col_none(head_col.dtype)
        nr = np.int64(1 if pl["numerosity_reduction"] else 0)   # as fitted
        tile = np.int64(self._row_tile(mode))
        tail = (head_col, np.int64(hc_none), tf_tab, idf, W, ids, cnt, nnz,
                acc, sq)
        if "cross" in pl:
            _key_families(X, rows, np.int64(nb), pl["kind"], pl["win"],
                          pl["cptr"], pl["chs"], pl["diff"], pl["dptr"],
                          pl["dils"], pl["cuts"], pl["dcol"], pl["dtab"],
                          pl["own_level"], pl["lrow"], pl["lcuts"],
                          pl["woff"], pl["trig"], trig[0], trig[1],
                          *self._word_args(pl, X.shape[2], int(tile)),
                          np.int64(pl["cross"]), np.int64(v0), np.int64(v1),
                          np.int64(mode), nr, tile, *tail)
            return nnz
        uni = (X, rows, np.int64(nb), pl["kind"], pl["win"], pl["ch"],
               pl["diff"], pl["dptr"], pl["dils"], pl["cuts"], pl["flat"],
               pl["dcol"], pl["dtab"], pl["own_level"], pl["lrow"],
               pl["lcuts"], pl["woff"], pl["trig"], trig[0], trig[1])
        cp = None
        if config.CHAIN_KEYING:
            cp = self._chain_pairs(v0, v1, X.shape[2], mode, int(tile),
                                   W.shape[1], ids, cnt)
        if cp is not None:
            _key_univariate_chains(*uni, np.int64(mode), nr, cp.tile, *tail,
                                   *cp[:-1])
        else:
            _key_univariate(*uni, np.int64(v0), np.int64(v1), np.int64(mode),
                            nr, tile, *tail)
        return nnz

    def _set_buffer_types(self, m):
        """The fit's buffer types (_buffer_types) on the estimator."""
        # fit scratch, see _FIT_SCRATCH
        self._id_bt, self._cnt_bt = _buffer_types(self._plan, m)

    def _release_buffers(self):
        """Drop the shared row buffers (_bufs)."""
        self.__dict__.pop("_tbuf", None)

    def _bufs(self, rows, rb):
        """(ids, cnt) row buffers, (rows, rb) views of ONE allocation reused
        across pseudo-channels and chunks, which avoids page-faulting fresh
        multi-hundred-MB buffers inside every kernel call."""
        need = int(rows) * int(rb)
        b = getattr(self, "_tbuf", None)
        if b is None or b[0].size < need:
            b = (np.empty(need, dtype=self._id_bt),
                 np.empty(need, dtype=self._cnt_bt))
            self._tbuf = b          # fit scratch, see _FIT_SCRATCH
        return (b[0][:need].reshape(rows, rb), b[1][:need].reshape(rows, rb))

    def _encode_pc(self, X, rows, p, trig, df=None):
        """Block p's (see plan.py) mode-0 counts over ``rows``
        (local unpruned ids) as one encoded _Part; df accumulated when
        given."""
        pl = self._plan
        v0, v1 = int(pl["pv_ptr"][p]), int(pl["pv_ptr"][p + 1])
        rb = max(1, int(pl["row_bound"][p]))
        n = len(rows)
        slot = np.dtype(self._id_bt).itemsize + np.dtype(self._cnt_bt).itemsize
        chunk = max(1, min(n, int(config.TRANSFORM_CHUNK_MB * 1e6
                                  // (rb * slot))))
        parts = []
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            ids, cnt = self._bufs(e - s, rb)
            nnz = np.zeros(e - s, dtype=np.int64)
            self._run(X, rows[s:e], v0, v1, _MODE_GLOBAL, trig,
                      bufs=(ids, cnt, nnz))
            if df is not None and nnz.any():
                _df_buffers(ids, nnz, df, np.int64(
                    max(1, min(config.ROW_PARTS, (e - s) // 16))))
            parts.append(_encode_buffers(ids, cnt, nnz))
        return _concat_parts(parts)
