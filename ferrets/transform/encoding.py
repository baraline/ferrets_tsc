"""tf tables, the smooth idf and encoded row parts."""

import functools

import numpy as np

from ..kernels.codec import _enc_count, _enc_write
from ..kernels.tf import _fill_count, _fill_tf, _fill_tf_store


@functools.lru_cache(maxsize=8)
def _tf_tables(m, sublinear=True):
    """(store, tab): the batch design's float32 tf table and the float64 one
    of the streaming decode and of scoring, indexed by count c: 1 + ln c if
    sublinear, else c. Shared by every call with the same key and never
    written after the build."""
    store = np.empty(m + 1, dtype=np.float32)
    tab = np.empty(m + 1, dtype=np.float64)
    if sublinear:
        _fill_tf_store(store)
        _fill_tf(tab)
    else:
        _fill_count(store)
        _fill_count(tab)
    return store, tab


def _smooth_idf(df, n_docs, dtype):
    """ln((1 + n) / (1 + df)) + 1 in ``dtype``, computed as sklearn 1.8's
    TfidfTransformer does for a counts matrix of that dtype (float32 for the
    batch design, float64 for the streaming head)."""
    dt = np.dtype(dtype)
    d = np.asarray(df).astype(dt)
    d += dt.type(1)
    idf = np.full_like(d, fill_value=n_docs + 1, dtype=dt)
    idf /= d
    np.log(idf, out=idf)
    idf += dt.type(1)
    return idf


class _Part:
    """Delta-coded (id, count) rows as uint8 codes: row r's codes are
    code_d / code_c[cptr[r]:cptr[r+1]], its escapes esc[eptr[r]:eptr[r+1]]
    (a gap or count that does not fit a byte). Ids are local to one
    pseudo-channel; a row's first gap is from -1."""

    __slots__ = ("cptr", "eptr", "code_d", "code_c", "esc")

    def __init__(self, cptr, eptr, code_d, code_c, esc):
        self.cptr = cptr
        self.eptr = eptr
        self.code_d = code_d
        self.code_c = code_c
        self.esc = esc

    @property
    def nbytes(self):
        return int(self.cptr.nbytes + self.eptr.nbytes + self.code_d.nbytes
                   + self.code_c.nbytes + self.esc.nbytes)

    @property
    def nnz(self):
        return int(self.cptr[-1])


def _encode_buffers(ids, cnt, nnz):
    """A _Part from the keying kernel's padded row buffers (nnz entries per
    row)."""
    n = nnz.shape[0]
    cptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(nnz, out=cptr[1:])
    esc_rows = np.zeros(n, dtype=np.int64)
    if cptr[-1]:
        _enc_count(ids, cnt, nnz, esc_rows)
    eptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(esc_rows, out=eptr[1:])
    code_d = np.empty(int(cptr[-1]), dtype=np.uint8)
    code_c = np.empty(int(cptr[-1]), dtype=np.uint8)
    esc = np.empty(int(eptr[-1]), dtype=np.int64)
    if cptr[-1]:
        _enc_write(ids, cnt, nnz, cptr, eptr, code_d, code_c, esc)
    return _Part(cptr, eptr, code_d, code_c, esc)


def _concat_parts(parts):
    """Row-concatenation of _Parts (disjoint consecutive row ranges)."""
    if len(parts) == 1:
        return parts[0]
    cp, ep = [np.zeros(1, dtype=np.int64)], [np.zeros(1, dtype=np.int64)]
    oc = oe = 0
    for pt in parts:
        cp.append(pt.cptr[1:] + oc)
        ep.append(pt.eptr[1:] + oe)
        oc += int(pt.cptr[-1])
        oe += int(pt.eptr[-1])
    return _Part(np.concatenate(cp), np.concatenate(ep),
                 np.concatenate([p.code_d for p in parts]),
                 np.concatenate([p.code_c for p in parts]),
                 np.concatenate([p.esc for p in parts]))
