"""The encoded chunks of the streaming route: one chunk's codes, and the
write-once store of all of them, in RAM or on disk."""

import os
import shutil
import tempfile

import numpy as np


class _Chunk:
    """One streaming chunk: its rows (training indices), and its P parts
    concatenated: pointer tables (P, rows+1) into flat code / escape arrays
    (in RAM, or offsets into the cache files)."""

    __slots__ = ("rows", "cptr", "eptr", "n_codes", "n_esc", "arrays", "offs")

    def __init__(self, rows, cptr, eptr, n_codes, n_esc):
        self.rows = rows
        self.cptr = cptr
        self.eptr = eptr
        self.n_codes = int(n_codes)
        self.n_esc = int(n_esc)
        self.arrays = None
        self.offs = None


def _build_chunk(tr, X, rws, trig, dfs=None):
    """One streaming chunk of transformer ``tr``: every pseudo-channel's
    mode-0 counts of its rows encoded (local unpruned ids; df accumulated when
    given), the P parts concatenated. Returns (_Chunk, (code_d, code_c,
    esc))."""
    P = tr._plan["P"]
    nr = len(rws)
    parts = [tr._encode_pc(X, rws, p, trig,
                           df=None if dfs is None else dfs[p])
             for p in range(P)]
    cptr = np.empty((P, nr + 1), dtype=np.int64)
    eptr = np.empty((P, nr + 1), dtype=np.int64)
    oc = oe = 0
    for p, pt in enumerate(parts):
        cptr[p] = pt.cptr + oc
        eptr[p] = pt.eptr + oe
        oc += int(pt.cptr[-1])
        oe += int(pt.eptr[-1])
    arrs = (np.concatenate([pt.code_d for pt in parts]),
            np.concatenate([pt.code_c for pt in parts]),
            np.concatenate([pt.esc for pt in parts]))
    return _Chunk(rws, cptr, eptr, oc, oe), arrs


class _ChunkCache:
    """Write-once store of the encoded chunks: "ram" keeps the arrays,
    "disk" appends them to three flat files read back by offset (no memmap).
    A chunk feed reads the next chunk with one prefetch thread while the
    head's kernels run."""

    _NAMES = ("code_d.bin", "code_c.bin", "esc.bin")

    def __init__(self, mode):
        self.mode = mode
        self.chunks = []
        self.bytes = 0
        self.dir = None
        self.files = None
        self._rf = []
        self.offs = [0, 0, 0]
        if mode == "disk":
            self.dir = tempfile.mkdtemp(prefix="ferrets_stream_")
            self.paths = [os.path.join(self.dir, s) for s in self._NAMES]
            try:
                self.files = [open(p, "wb") for p in self.paths]
            except BaseException:
                self.close()
                raise

    def put(self, chunk, code_d, code_c, esc):
        arrs = (code_d, code_c, esc)
        if self.mode == "ram":
            chunk.arrays = arrs
        else:
            chunk.offs = tuple(self.offs)
            for s, (f, a) in enumerate(zip(self.files, arrs)):
                a.tofile(f)
                self.offs[s] += int(a.nbytes)
        self.bytes += int(sum(a.nbytes for a in arrs)
                          + chunk.cptr.nbytes + chunk.eptr.nbytes)
        self.chunks.append(chunk)

    def seal(self):
        if self.files is not None:
            for f in self.files:
                f.close()
            self.files = None
            self._rf = [open(p, "rb") for p in self.paths]

    def read(self, k):
        ch = self.chunks[k]
        if self.mode == "ram":
            return ch.arrays
        out = (np.empty(ch.n_codes, dtype=np.uint8),
               np.empty(ch.n_codes, dtype=np.uint8),
               np.empty(ch.n_esc, dtype=np.int64))
        for s in range(3):
            f = self._rf[s]
            f.seek(ch.offs[s])
            if out[s].nbytes:
                got = f.readinto(memoryview(out[s]).cast("B"))
                if got != out[s].nbytes:
                    raise IOError(f"short read from the streaming cache "
                                  f"({got} of {out[s].nbytes} bytes)")
        return out

    def disk_bytes(self):
        return int(sum(self.offs)) if self.mode == "disk" else 0

    def close(self):
        try:
            for f in self._rf:
                try:
                    f.close()
                except Exception:
                    pass
            self._rf = []
            if self.files is not None:
                for f in self.files:
                    f.close()
                self.files = None
            self.chunks = []
        finally:
            if self.dir is not None:
                shutil.rmtree(self.dir, ignore_errors=True)
                self.dir = None
