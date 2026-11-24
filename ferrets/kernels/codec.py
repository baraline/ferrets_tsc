"""The compact count codec (delta-coded column ids, uint8 counts, escapes),
its document frequencies and its tf-idf decoders."""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, nogil=True)
def _df_buffers(ids, nnz, df, n_parts):
    """df[c] += rows of the padded buffers holding column c (canonical rows:
    each column at most once per row). Fixed row partitions and integer
    sums, so the result does not depend on the thread count."""
    n = nnz.shape[0]
    n_cols = df.shape[0]
    part = np.zeros((n_parts, n_cols), np.int32)
    for g in prange(n_parts):
        lo = g * n // n_parts
        hi = (g + 1) * n // n_parts
        for i in range(lo, hi):
            for e in range(nnz[i]):
                part[g, ids[i, e]] += 1
    for c in prange(n_cols):
        a = np.int64(0)
        for g in range(n_parts):
            a += part[g, c]
        df[c] += a


@njit(cache=True, parallel=True, nogil=True)
def _enc_count(ids, cnt, nnz, esc_rows):
    """Escapes per row of padded (id, count) buffers: a column-id gap of 256
    or more, or a count of 255 or more, does not fit its uint8 code."""
    for i in prange(nnz.shape[0]):
        prev = np.int64(-1)
        q = np.int64(0)
        for e in range(nnz[i]):
            g = np.int64(ids[i, e])
            if g - prev >= 256:
                q += 1
            if cnt[i, e] >= 255:
                q += 1
            prev = g
        esc_rows[i] = q


@njit(cache=True, parallel=True, nogil=True)
def _enc_write(ids, cnt, nnz, cptr, eptr, code_d, code_c, esc):
    """Delta-code each row into code_d / code_c from cptr[i] and esc from
    eptr[i] (offsets from _enc_count's escape counts). Per entry: code_d =
    gap - 1 from the previous column id (-1 before the first), code_c =
    count. A byte of 255 is an escape: the true gap (or count) is the next
    unread esc value, gap before count."""
    for i in prange(nnz.shape[0]):
        prev = np.int64(-1)
        q = cptr[i]
        qe = eptr[i]
        for e in range(nnz[i]):
            g = np.int64(ids[i, e])
            dg = g - prev
            prev = g
            if dg >= 256:
                code_d[q] = np.uint8(255)
                esc[qe] = dg
                qe += 1
            else:
                code_d[q] = np.uint8(dg - 1)
            k = np.int64(cnt[i, e])
            if k >= 255:
                code_c[q] = np.uint8(255)
                esc[qe] = k
                qe += 1
            else:
                code_c[q] = np.uint8(k)
            q += 1


@njit(inline="always")
def _next_entry(q, g, qe, code_d, code_c, esc):
    """Decode the entry at code position q, given the previous column id g and
    the next unread escape position qe. A byte of 255 is an escape: the true
    gap (or count) is the next esc value, gap before count. Returns (column
    id, count, escape position after the entry). A count escape is read
    even by callers that ignore the count."""
    cd = code_d[q]
    if cd == 255:
        g1 = g + esc[qe]
        qe1 = qe + 1
    else:
        g1 = g + np.int64(cd) + 1
        qe1 = qe
    cc = code_c[q]
    if cc == 255:
        cn = esc[qe1]
        qe1 += 1
    else:
        cn = np.int64(cc)
    return g1, cn, qe1


@njit(cache=True, parallel=True, nogil=True)
def _sq_pass(cptr, eptr, code_d, code_c, esc, remap, tf_store, idf, sq, kcnt):
    """Squared row norms of the batch tf-idf, carried across pseudo-channel
    blocks: w = float32 tf(count) * float32 idf, sq += w * w (the square in
    float32, then widened); kcnt[r] = the row's kept entries (remap[g] >= 0).
    """
    for r in prange(cptr.shape[0] - 1):
        s = sq[r]
        g = np.int64(-1)
        qe = eptr[r]
        k = np.int64(0)
        for q in range(cptr[r], cptr[r + 1]):
            g, cn, qe = _next_entry(q, g, qe, code_d, code_c, esc)
            kj = remap[g]
            if kj < 0:
                continue
            wv = tf_store[cn] * idf[kj]
            s += wv * wv
            k += 1
        sq[r] = s
        kcnt[r] = k


@njit(cache=True, parallel=True, nogil=True)
def _dec_batch(cptr, eptr, code_d, code_c, esc, remap, kptr, tf_store, idf,
               norm, out_ids, out_vals):
    """One pseudo-channel block decoded to its normalised tf-idf CSR over
    local kept ids: value float32(w / norm[r]), w as in _sq_pass."""
    for r in prange(cptr.shape[0] - 1):
        g = np.int64(-1)
        qe = eptr[r]
        o = kptr[r]
        nr = norm[r]
        for q in range(cptr[r], cptr[r + 1]):
            g, cn, qe = _next_entry(q, g, qe, code_d, code_c, esc)
            kj = remap[g]
            if kj < 0:
                continue
            wv = tf_store[cn] * idf[kj]
            out_ids[o] = kj
            out_vals[o] = wv / nr
            o += 1


@njit(cache=True, parallel=True, nogil=True)
def _dec_counts(cptr, eptr, code_d, code_c, esc, remap, kptr, off, lo, hi,
                out_cols, out_cnt):
    """Rows lo..hi-1 of one block decoded to their kept entries: the column
    remap[g] + off and the count, from out position kptr[r] - kptr[lo]."""
    for r in prange(lo, hi):
        g = np.int64(-1)
        qe = eptr[r]
        o = kptr[r] - kptr[lo]
        for q in range(cptr[r], cptr[r + 1]):
            g, cn, qe = _next_entry(q, g, qe, code_d, code_c, esc)
            kj = remap[g]
            if kj < 0:
                continue
            out_cols[o] = kj + off
            out_cnt[o] = cn
            o += 1


@njit(cache=True, parallel=True, nogil=True)
def _band_counts(cptr, eptr, code_d, code_c, esc, remap, bshift, out):
    """out[r, k] = the kept entries of row r whose kept id lies in band k,
    ids [k << bshift, (k + 1) << bshift): the decode walk of _sq_pass
    without the float work. One row per task, so rows never share a cell."""
    for r in prange(cptr.shape[0] - 1):
        g = np.int64(-1)
        qe = eptr[r]
        for q in range(cptr[r], cptr[r + 1]):
            g, cn, qe = _next_entry(q, g, qe, code_d, code_c, esc)
            kj = remap[g]
            if kj < 0:
                continue
            out[r, np.int64(kj) >> bshift] += 1


@njit(cache=True, parallel=True, nogil=True)
def _dec_batch_bands(cptr, eptr, code_d, code_c, esc, remap, bshift, cur,
                     tf_store, idf, norm, out_ids, out_vals):
    """_dec_batch with the kept ids cut into bands of 1 << bshift: entry kj
    of row r goes to band k = kj >> bshift, at the next free position
    cur[r, k] of that (row, band) region (the caller sets cur to each
    region's start from _band_counts; it ends at the region's end), with the
    id local to the band, kj - (k << bshift). The value is _dec_batch's.
    Regions are disjoint and one row is one task, so no cell is shared."""
    for r in prange(cptr.shape[0] - 1):
        g = np.int64(-1)
        qe = eptr[r]
        nr = norm[r]
        for q in range(cptr[r], cptr[r + 1]):
            g, cn, qe = _next_entry(q, g, qe, code_d, code_c, esc)
            kj = remap[g]
            if kj < 0:
                continue
            wv = tf_store[cn] * idf[kj]
            k = np.int64(kj) >> bshift
            o = cur[r, k]
            out_ids[o] = np.int64(kj) - (k << bshift)
            out_vals[o] = wv / nr
            cur[r, k] = o + 1


@njit(cache=True, parallel=True, nogil=True)
def _dec_slab_count(cptr, eptr, code_d, code_c, esc, remap, cur_q, cur_g,
                    cur_qe, c1, cnt):
    """cnt[r] = the kept entries of row r, from its cursor (cur_q code
    position, cur_g last column id, cur_qe escape position), whose local kept
    id is below c1: _dec_batch's decode on local copies, stopping at the first
    kept id >= c1 (kept ids ascend along a row: remap is monotone), pruned
    entries consumed on the way. The cursor is not moved."""
    for r in prange(cptr.shape[0] - 1):
        q = cur_q[r]
        g = cur_g[r]
        qe = cur_qe[r]
        end = cptr[r + 1]
        k = np.int64(0)
        while q < end:
            g1, cn, qe1 = _next_entry(q, g, qe, code_d, code_c, esc)
            kj = remap[g1]
            if kj >= c1:
                break
            g = g1
            qe = qe1
            q += 1
            if kj >= 0:
                k += 1
        cnt[r] = k


@njit(cache=True, parallel=True, nogil=True)
def _dec_slab_fill(cptr, eptr, code_d, code_c, esc, remap, cur_q, cur_g,
                   cur_qe, c1, tf_store, idf, norm, ptr_out, out_ids,
                   out_vals):
    """The entries _dec_slab_count counts, written from ptr_out[r] with
    _dec_batch's arithmetic (w = float32 tf x float32 idf, stored
    float32(w / norm[r]), block-local kept ids), then the cursor committed to
    the state just before the first entry with kept id >= c1 (not consumed:
    the next piece decodes it again), or to the row's end."""
    for r in prange(cptr.shape[0] - 1):
        q = cur_q[r]
        g = cur_g[r]
        qe = cur_qe[r]
        end = cptr[r + 1]
        o = ptr_out[r]
        nr = norm[r]
        while q < end:
            g1, cn, qe1 = _next_entry(q, g, qe, code_d, code_c, esc)
            kj = remap[g1]
            if kj >= c1:
                break
            g = g1
            qe = qe1
            q += 1
            if kj < 0:
                continue
            wv = tf_store[cn] * idf[kj]
            out_ids[o] = kj
            out_vals[o] = wv / nr
            o += 1
        cur_q[r] = q
        cur_g[r] = g
        cur_qe[r] = qe


@njit(inline="always")
def _dec_row(r, cptr, eptr, code_d, code_c, esc, tabs, head_col, hc_none,
             woff, tf_tab, idf, rows, inv, have_inv, out_start, out_end,
             out_ids, out_vals):
    """Row r of _dec_stream."""
    P = cptr.shape[0]
    o = out_start[r]
    s = 0.0
    ir = rows[r]
    iv = inv[ir]
    for p in range(P):
        g = np.int64(-1)
        qe = eptr[p, r]
        tb = tabs[p]
        base = woff[p]
        for q in range(cptr[p, r], cptr[p, r + 1]):
            g, cn, qe = _next_entry(q, g, qe, code_d, code_c, esc)
            hc = np.int64(head_col[tb + g])
            if hc == hc_none:
                continue
            col = base + hc
            wv = tf_tab[cn] * idf[col]
            out_ids[o] = np.int32(col)
            if have_inv:
                out_vals[o] = np.float32(wv) * iv
            else:
                out_vals[o] = wv
                s += wv * wv
            o += 1
    out_end[r] = o
    if not have_inv:
        if s > 0.0:
            iv = 1.0 / np.sqrt(s)
            for e in range(out_start[r], o):
                out_vals[e] = out_vals[e] * iv
        else:
            iv = 1.0
        inv[ir] = iv


@njit(cache=True, parallel=True, nogil=True)
def _dec_stream(cptr, eptr, code_d, code_c, esc, tabs, head_col, hc_none,
                woff, tf_tab, idf, rows, inv, have_inv, out_start, out_end,
                out_ids, out_vals):
    """A streaming chunk decoded to its row-major tf-idf rows over the global
    kept columns. The chunk has P pseudo-channel parts of unpruned local ids,
    as counted by the keying kernel in _MODE_GLOBAL: a stored id g of part
    p is kept iff
    head_col[tabs[p] + g] names a column, and becomes woff[p] + that column.

    Arithmetic: w = tf(count) * idf in float64, stored as float32;
    sq += w * w over the unrounded w; then every stored value *= 1 /
    sqrt(sq). Row r's entries land at out_start[r] .. out_end[r]
    (out_start comes from the stored counts, an upper bound: pruned entries
    leave a gap). The row norm depends only on the row, so the first decode
    records inv[rows[r]] and every later one (have_inv) writes float32(w) *
    inv in one pass, with the same values."""
    for r in prange(cptr.shape[1] - 1):
        _dec_row(r, cptr, eptr, code_d, code_c, esc, tabs, head_col, hc_none,
                 woff, tf_tab, idf, rows, inv, have_inv, out_start, out_end,
                 out_ids, out_vals)
