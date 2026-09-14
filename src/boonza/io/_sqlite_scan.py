"""Column-wise scan of SQLite tables straight from the database bytes.

Python's sqlite3 module builds a Python object for every value, which
dominates load time for million-atom DMS files.  Here numba walks a table
b-tree and decodes each column into a numpy array.  Only the part of the
file format that DMS files use is handled; anything else (WAL mode, non-UTF-8
text, records spilling onto overflow pages, values that do not match the
declared column type) makes ``read`` return None so the caller can fall back
to sqlite3.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .._columns import STR

_KIND = {"int": 0, "float": 1, "str": 2, "skip": 3}
_MAX_FIXED_WIDTH = 256


@njit(cache=True)
def _varint(buf, p):
    """SQLite varint at ``p``: (value, next position); next is -1 past the buffer end."""
    n = buf.shape[0]
    v = np.int64(0)
    for i in range(8):
        if p + i >= n:
            return v, np.int64(-1)
        b = np.int64(buf[p + i])
        v = (v << 7) | (b & 0x7F)
        if b < 0x80:
            return v, np.int64(p + i + 1)
    if p + 8 >= n:
        return v, np.int64(-1)
    v = (v << 8) | np.int64(buf[p + 8])
    return v, np.int64(p + 9)


@njit(cache=True)
def _u16(buf, p):
    return (np.int64(buf[p]) << 8) | np.int64(buf[p + 1])


@njit(cache=True)
def _u32(buf, p):
    return (_u16(buf, p) << 16) | _u16(buf, p + 2)


@njit(cache=True)
def _header(page, page_size):
    # page 1 starts with the 100-byte database header
    return (page - 1) * page_size + (100 if page == 1 else 0)


@njit(cache=True)
def _leaf_pages(buf, root, page_size):
    """Leaf pages of the table b-tree at ``root``, in rowid order; error code 0 if fine."""
    npages = buf.shape[0] // page_size
    leaves = np.empty(npages, np.int64)
    stack = np.empty(npages + 1, np.int64)
    nleaves = 0
    visited = 0
    stack[0] = root
    sp = 1
    while sp > 0:
        sp -= 1
        page = stack[sp]
        visited += 1
        if page < 1 or page > npages or visited > npages:
            return leaves[:0], 1
        base = (page - 1) * page_size
        h = _header(page, page_size)
        kind = buf[h]
        if kind == 13:  # table leaf
            leaves[nleaves] = page
            nleaves += 1
        elif kind == 5:  # table interior: children in cell order, then the right pointer
            ncell = _u16(buf, h + 3)
            if sp + ncell + 1 > stack.shape[0]:
                return leaves[:0], 1
            stack[sp] = _u32(buf, h + 8)
            sp += 1
            for k in range(ncell - 1, -1, -1):
                stack[sp] = _u32(buf, base + _u16(buf, h + 12 + 2 * k))
                sp += 1
        else:
            return leaves[:0], 2
    return leaves[:nleaves], 0


@njit(cache=True)
def _count_cells(buf, pages, page_size):
    n = 0
    for page in pages:
        n += _u16(buf, _header(page, page_size) + 3)
    return n


@njit(cache=True)
def _decode(buf, pages, page_size, usable, kinds, slots, rowid_col, out_i, out_f, toff, tlen):
    """Fill the output arrays from every record; returns 0 or an error code."""
    ncols = kinds.shape[0]
    nbuf = buf.shape[0]
    max_local = usable - 35
    scratch = np.zeros(1, np.uint64)
    as_float = scratch.view(np.float64)
    row = 0
    for page in pages:
        base = (page - 1) * page_size
        h = _header(page, page_size)
        for k in range(_u16(buf, h + 3)):
            p = base + _u16(buf, h + 8 + 2 * k)
            if p >= base + page_size:
                return 6  # cell pointer outside its page
            payload, p = _varint(buf, p)
            if p < 0:
                return 6
            rowid, p = _varint(buf, p)
            if p < 0:
                return 6
            if payload > max_local:
                return 3  # record continues on overflow pages
            if p + payload > nbuf:
                return 6
            hsize, q = _varint(buf, p)
            if q < 0 or hsize > payload:
                return 6
            hend = p + hsize
            body = hend
            for c in range(ncols):
                st = np.int64(0)  # missing trailing columns read as NULL
                if q < hend:
                    st, q = _varint(buf, q)
                    if q < 0:
                        return 6
                if st < 5:
                    size = st
                elif st == 5:
                    size = np.int64(6)
                elif st < 8:
                    size = np.int64(8)
                elif st < 10:
                    size = np.int64(0)
                elif st < 12:
                    return 4
                else:
                    size = (st - 12 - (st & 1)) // 2
                if body + size > p + payload:
                    return 6  # value runs past the end of its record
                kind = kinds[c]
                s = slots[c]
                if c == rowid_col:
                    out_i[s, row] = rowid  # INTEGER PRIMARY KEY is stored as the rowid
                elif kind == 3 or st == 0:
                    pass
                elif st < 7 or st == 8 or st == 9:
                    v = np.int64(0)
                    if st == 9:
                        v = np.int64(1)
                    elif st < 7:
                        for b in range(size):
                            v = (v << 8) | np.int64(buf[body + b])
                        if size < 8 and buf[body] >= 0x80:
                            v -= np.int64(1) << (8 * size)
                    if kind == 0:
                        out_i[s, row] = v
                    elif kind == 1:
                        out_f[s, row] = v  # integral REAL values are stored as integers
                    else:
                        return 5
                elif st == 7:
                    if kind != 1:
                        return 5
                    u = np.uint64(0)
                    for b in range(8):
                        u = (u << np.uint64(8)) | np.uint64(buf[body + b])
                    scratch[0] = u
                    out_f[s, row] = as_float[0]
                else:
                    if kind != 2:
                        return 5
                    toff[s, row] = body
                    tlen[s, row] = size
                body += size
            row += 1
    return 0


@njit(cache=True)
def _gather(buf, off, length, width):
    out = np.zeros((off.shape[0], width), np.uint8)
    for i in range(off.shape[0]):
        for j in range(length[i]):
            out[i, j] = buf[off[i] + j]
    return out


def _strings(buf, off, length) -> np.ndarray:
    n = len(off)
    width = int(length.max()) if n else 0
    if width == 0:
        return np.full(n, "", dtype=STR)
    if width > _MAX_FIXED_WIDTH:
        return np.array(
            [
                bytes(buf[o : o + k]).decode()
                for o, k in zip(off.tolist(), length.tolist(), strict=True)
            ],
            dtype=STR,
        )
    # decode each distinct string once; DMS text columns are highly repetitive
    fixed = _gather(buf, off, length, width).view(f"S{width}").reshape(n)
    uniq, inv = np.unique(fixed, return_inverse=True)
    words = np.array([w.decode() for w in uniq.tolist()], dtype=STR)
    return words[inv.reshape(-1)]


class SqliteFile:
    """Read-only column access to the tables of an SQLite database image."""

    def __init__(self, buf):
        self.buf = np.asarray(buf, dtype=np.uint8)
        head = bytes(self.buf[:100])
        size = int.from_bytes(head[16:18], "big")
        self.page_size = 65536 if size == 1 else size
        self.usable = self.page_size - head[20]
        encoding = int.from_bytes(head[56:60], "big")
        self.ok = (
            head[:16] == b"SQLite format 3\x00"
            and head[18] == 1  # rollback journal, not WAL
            and head[19] == 1
            and encoding in (0, 1)  # UTF-8
            and self.page_size >= 512
        )

    def read(self, root: int, kinds: list[str], rowid_col: int = -1):
        """Decode the table whose b-tree starts at page ``root``.

        ``kinds`` holds "int", "float", "str" or "skip" for each column in
        table order.  Returns one array per column (None where skipped), or
        None if the table uses a feature this reader does not handle.
        """
        if not self.ok:
            return None
        if rowid_col >= 0 and kinds[rowid_col] == "skip":
            rowid_col = -1  # the kernel only writes columns that were asked for
        pages, err = _leaf_pages(self.buf, root, self.page_size)
        if err:
            return None
        n = _count_cells(self.buf, pages, self.page_size)
        codes = np.array([_KIND[k] for k in kinds], np.int8)
        slots = np.zeros(len(kinds), np.int64)
        counts = [0, 0, 0, 0]
        for c, code in enumerate(codes):
            slots[c] = counts[code]
            counts[code] += 1
        out_i = np.zeros((counts[0], n), np.int64)
        out_f = np.zeros((counts[1], n), np.float64)
        toff = np.zeros((counts[2], n), np.int64)
        tlen = np.zeros((counts[2], n), np.int64)
        err = _decode(
            self.buf, pages, self.page_size, self.usable, codes, slots, rowid_col,
            out_i, out_f, toff, tlen,
        )  # fmt: skip
        if err:
            return None
        out = []
        for c, kind in enumerate(kinds):
            s = slots[c]
            if kind == "int":
                out.append(out_i[s])
            elif kind == "float":
                out.append(out_f[s])
            elif kind == "str":
                out.append(_strings(self.buf, toff[s], tlen[s]))
            else:
                out.append(None)
        return out
