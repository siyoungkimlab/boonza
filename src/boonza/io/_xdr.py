"""GROMACS XTC and TRR trajectories, read and written natively.

The XTC coordinate compression is a port of ``xdrfile_compress_coord_float``
and ``xdrfile_decompress_coord_float`` from the xdrfile library (Erik Lindahl
and David van der Spoel, BSD license): the same bit packing, the same
float32 rounding, the same water-pair swap and run-length adaptation, so
files are byte-identical to the ones GROMACS writes.  XDR is big-endian:
4-byte integers and floats, opaque data padded to 4 bytes.
"""

from __future__ import annotations

import numpy as np
from numba import njit

_MAGICINTS = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64,
    80, 101, 128, 161, 203, 256, 322, 406, 512, 645, 812, 1024, 1290,
    1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321, 13003,
    16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031,
    131072, 165140, 208063, 262144, 330280, 416127, 524287, 660561,
    832255, 1048576, 1321122, 1664510, 2097152, 2642245, 3329021,
    4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216,
    16777216,  # a guard: the C code can read one past its table
], dtype=np.int64)  # fmt: skip
_FIRSTIDX = 9
_LASTIDX = 73  # entries in the C table (the guard is extra)
_INT_MAX = 2147483647
_U32 = 0xFFFFFFFF
XTC_MAGIC = 1995
TRR_MAGIC = 1993
TRR_VERSION = b"GMX_trn_file"


class XDRError(ValueError):
    """Unreadable XTC/TRR data."""


# ---------------------------------------------------------------------------
# bit packing (state: [byte count, bits in lastbyte, lastbyte])


@njit(cache=True)
def _sizeofint(size):
    num = 1
    bits = 0
    while size >= num and bits < 32:
        bits += 1
        num <<= 1
    return bits


@njit(cache=True)
def _sizeofints(nints, sizes):
    nbytes = 1
    b = np.zeros(32, np.int64)
    b[0] = 1
    bits = 0
    for i in range(nints):
        tmp = 0
        bc = 0
        while bc < nbytes:
            tmp = b[bc] * sizes[i] + tmp
            b[bc] = tmp & 0xFF
            tmp >>= 8
            bc += 1
        while tmp != 0:
            b[bc] = tmp & 0xFF
            bc += 1
            tmp >>= 8
        nbytes = bc
    num = 1
    nbytes -= 1
    while b[nbytes] >= num:
        bits += 1
        num *= 2
    return bits + nbytes * 8


@njit(cache=True)
def _encodebits(data, st, nbits, num):
    cnt, lastbits, lastbyte = st[0], st[1], st[2]
    while nbits >= 8:
        lastbyte = ((lastbyte << 8) | (num >> (nbits - 8))) & _U32
        data[cnt] = (lastbyte >> lastbits) & 0xFF
        cnt += 1
        nbits -= 8
    if nbits > 0:
        lastbyte = ((lastbyte << nbits) | num) & _U32
        lastbits += nbits
        if lastbits >= 8:
            lastbits -= 8
            data[cnt] = (lastbyte >> lastbits) & 0xFF
            cnt += 1
    st[0], st[1], st[2] = cnt, lastbits, lastbyte
    if lastbits > 0:
        data[cnt] = (lastbyte << (8 - lastbits)) & 0xFF


@njit(cache=True)
def _encodeints(data, st, nints, nbits, sizes, nums):
    b = np.zeros(32, np.int64)
    tmp = nums[0]
    nbytes = 0
    while True:
        b[nbytes] = tmp & 0xFF
        nbytes += 1
        tmp >>= 8
        if tmp == 0:
            break
    for i in range(1, nints):
        tmp = nums[i]
        bc = 0
        while bc < nbytes:
            tmp = b[bc] * sizes[i] + tmp
            b[bc] = tmp & 0xFF
            tmp >>= 8
            bc += 1
        while tmp != 0:
            b[bc] = tmp & 0xFF
            bc += 1
            tmp >>= 8
        nbytes = bc
    if nbits >= nbytes * 8:
        for i in range(nbytes):
            _encodebits(data, st, 8, b[i])
        _encodebits(data, st, nbits - nbytes * 8, 0)
    else:
        for i in range(nbytes - 1):
            _encodebits(data, st, 8, b[i])
        _encodebits(data, st, nbits - (nbytes - 1) * 8, b[nbytes - 1])


@njit(cache=True)
def _decodebits(data, st, nbits):
    cnt, lastbits, lastbyte = st[0], st[1], st[2]
    mask = (1 << nbits) - 1
    num = 0
    while nbits >= 8:
        lastbyte = ((lastbyte << 8) | data[cnt]) & _U32
        cnt += 1
        num |= (lastbyte >> lastbits) << (nbits - 8)
        nbits -= 8
    if nbits > 0:
        if lastbits < nbits:
            lastbits += 8
            lastbyte = ((lastbyte << 8) | data[cnt]) & _U32
            cnt += 1
        lastbits -= nbits
        num |= (lastbyte >> lastbits) & ((1 << nbits) - 1)
    num &= mask
    st[0], st[1], st[2] = cnt, lastbits, lastbyte
    return num


@njit(cache=True)
def _decodeints(data, st, nints, nbits, sizes, nums):
    b = np.zeros(32, np.int64)
    nbytes = 0
    while nbits > 8:
        b[nbytes] = _decodebits(data, st, 8)
        nbytes += 1
        nbits -= 8
    if nbits > 0:
        b[nbytes] = _decodebits(data, st, nbits)
        nbytes += 1
    for i in range(nints - 1, 0, -1):
        num = 0
        for j in range(nbytes - 1, -1, -1):
            num = (num << 8) | b[j]
            p = num // sizes[i]
            b[j] = p
            num = num - p * sizes[i]
        nums[i] = num
    nums[0] = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)


# ---------------------------------------------------------------------------
# coordinate compression


@njit(cache=True)
def _decompress(data, natoms, minint, maxint, smallidx, precision):
    """float32 (natoms, 3) from the packed bytes of one frame."""
    out = np.empty(natoms * 3, np.float32)
    sizeint = np.empty(3, np.int64)
    bitsizeint = np.zeros(3, np.int64)
    for k in range(3):
        sizeint[k] = maxint[k] - minint[k] + 1
    if (sizeint[0] | sizeint[1] | sizeint[2]) > 0xFFFFFF:
        for k in range(3):
            bitsizeint[k] = _sizeofint(sizeint[k])
        bitsize = 0
    else:
        bitsize = _sizeofints(3, sizeint)
    tmp = smallidx - 1
    tmp = _FIRSTIDX if _FIRSTIDX > tmp else tmp
    smaller = _MAGICINTS[tmp] // 2
    smallnum = _MAGICINTS[smallidx] // 2
    sizesmall = np.full(3, _MAGICINTS[smallidx], np.int64)
    st = np.zeros(3, np.int64)
    inv = np.float32(1.0 / np.float64(precision))
    prev = np.zeros(3, np.int64)
    this = np.zeros(3, np.int64)
    run = 0
    i = 0
    out_i = 0
    while i < natoms:
        if bitsize == 0:
            for k in range(3):
                this[k] = _decodebits(data, st, bitsizeint[k])
        else:
            _decodeints(data, st, 3, bitsize, sizeint, this)
        i += 1
        for k in range(3):
            this[k] += minint[k]
            prev[k] = this[k]
        flag = _decodebits(data, st, 1)
        is_smaller = 0
        if flag == 1:
            run = _decodebits(data, st, 5)
            is_smaller = run % 3
            run -= is_smaller
            is_smaller -= 1
        if run > 0:
            for k in range(0, run, 3):
                _decodeints(data, st, 3, smallidx, sizesmall, this)
                i += 1
                for m in range(3):
                    this[m] += prev[m] - smallnum
                if k == 0:
                    for m in range(3):  # the first two atoms were swapped (water)
                        t = this[m]
                        this[m] = prev[m]
                        prev[m] = t
                    for m in range(3):
                        out[out_i] = np.float32(np.float32(prev[m]) * inv)
                        out_i += 1
                else:
                    for m in range(3):
                        prev[m] = this[m]
                for m in range(3):
                    out[out_i] = np.float32(np.float32(this[m]) * inv)
                    out_i += 1
        else:
            for m in range(3):
                out[out_i] = np.float32(np.float32(this[m]) * inv)
                out_i += 1
        smallidx += is_smaller
        if is_smaller < 0:
            smallnum = smaller
            if smallidx > _FIRSTIDX:
                smaller = _MAGICINTS[smallidx - 1] // 2
            else:
                smaller = 0
        elif is_smaller > 0:
            smaller = smallnum
            smallnum = _MAGICINTS[smallidx] // 2
        sizesmall[0] = sizesmall[1] = sizesmall[2] = _MAGICINTS[smallidx]
    return out.reshape(natoms, 3)


@njit(cache=True)
def _to_ints(xyz, precision):
    """Round coordinates as xdrfile does: float32 product, then +-0.5 in double."""
    n = xyz.shape[0]
    ints = np.empty(n * 3, np.int64)
    flat = xyz.reshape(n * 3)
    ok = True
    for k in range(n * 3):
        p = np.float32(flat[k] * precision)
        if flat[k] >= 0.0:
            lf = np.float32(np.float64(p) + 0.5)
        else:
            lf = np.float32(np.float64(p) - 0.5)
        if abs(lf) > _INT_MAX - 2:
            ok = False
        ints[k] = np.int64(lf)
    return ints, ok


@njit(cache=True)
def _compress(xyz, precision):
    """(minint, maxint, smallidx, packed bytes) of one frame of float32 coordinates."""
    size = xyz.shape[0]
    ints, ok = _to_ints(xyz, precision)
    minint = np.full(3, _INT_MAX, np.int64)
    maxint = np.full(3, -_INT_MAX - 1, np.int64)
    mindiff = _INT_MAX
    old = np.zeros(3, np.int64)
    for a in range(size):
        diff = 0
        for k in range(3):
            v = ints[3 * a + k]
            if v < minint[k]:
                minint[k] = v
            if v > maxint[k]:
                maxint[k] = v
            diff += abs(old[k] - v)
        if diff < mindiff and a >= 1:
            mindiff = diff
        for k in range(3):
            old[k] = ints[3 * a + k]
    sizeint = np.empty(3, np.int64)
    bitsizeint = np.zeros(3, np.int64)
    for k in range(3):
        if np.float32(maxint[k]) - np.float32(minint[k]) >= _INT_MAX - 2:
            ok = False
        sizeint[k] = maxint[k] - minint[k] + 1
    if (sizeint[0] | sizeint[1] | sizeint[2]) > 0xFFFFFF:
        for k in range(3):
            bitsizeint[k] = _sizeofint(sizeint[k])
        bitsize = 0
    else:
        bitsize = _sizeofints(3, sizeint)
    smallidx = _FIRSTIDX
    while smallidx < _LASTIDX and _MAGICINTS[smallidx] < mindiff:
        smallidx += 1
    start_smallidx = smallidx
    tmp = smallidx + 8
    maxidx = _LASTIDX if _LASTIDX < tmp else tmp
    minidx = maxidx - 8
    tmp = smallidx - 1
    tmp = _FIRSTIDX if _FIRSTIDX > tmp else tmp
    smaller = _MAGICINTS[tmp] // 2
    smallnum = _MAGICINTS[smallidx] // 2
    sizesmall = np.full(3, _MAGICINTS[smallidx], np.int64)
    larger = _MAGICINTS[maxidx] // 2
    data = np.zeros(size * 3 * 4 + 64, np.uint8)
    st = np.zeros(3, np.int64)
    prev = np.zeros(3, np.int64)
    tmpc = np.zeros(30, np.int64)
    prevrun = -1
    i = 0
    while i < size:
        is_small = 0
        t = 3 * i
        if (
            smallidx < maxidx
            and i >= 1
            and abs(ints[t] - prev[0]) < larger
            and abs(ints[t + 1] - prev[1]) < larger
            and abs(ints[t + 2] - prev[2]) < larger
        ):
            is_smaller = 1
        elif smallidx > minidx:
            is_smaller = -1
        else:
            is_smaller = 0
        if i + 1 < size:
            if (
                abs(ints[t] - ints[t + 3]) < smallnum
                and abs(ints[t + 1] - ints[t + 4]) < smallnum
                and abs(ints[t + 2] - ints[t + 5]) < smallnum
            ):
                for m in range(3):  # swap the first two atoms (water)
                    s = ints[t + m]
                    ints[t + m] = ints[t + 3 + m]
                    ints[t + 3 + m] = s
                is_small = 1
        for m in range(3):
            tmpc[m] = ints[t + m] - minint[m]
        if bitsize == 0:
            for m in range(3):
                _encodebits(data, st, bitsizeint[m], tmpc[m])
        else:
            _encodeints(data, st, 3, bitsize, sizeint, tmpc)
        for m in range(3):
            prev[m] = ints[t + m]
        t += 3
        i += 1
        run = 0
        if is_small == 0 and is_smaller == -1:
            is_smaller = 0
        while is_small and run < 8 * 3:
            tmpsum = 0
            for m in range(3):
                d = ints[t + m] - prev[m]
                tmpsum += d * d
            if is_smaller == -1 and tmpsum >= smaller * smaller:
                is_smaller = 0
            for m in range(3):
                tmpc[run] = ints[t + m] - prev[m] + smallnum
                run += 1
            for m in range(3):
                prev[m] = ints[t + m]
            i += 1
            t += 3
            is_small = 0
            if (
                i < size
                and abs(ints[t] - prev[0]) < smallnum
                and abs(ints[t + 1] - prev[1]) < smallnum
                and abs(ints[t + 2] - prev[2]) < smallnum
            ):
                is_small = 1
        if run != prevrun or is_smaller != 0:
            prevrun = run
            _encodebits(data, st, 1, 1)
            _encodebits(data, st, 5, run + is_smaller + 1)
        else:
            _encodebits(data, st, 1, 0)
        for k in range(0, run, 3):
            _encodeints(data, st, 3, smallidx, sizesmall, tmpc[k : k + 3])
        if is_smaller != 0:
            smallidx += is_smaller
            if is_smaller < 0:
                smallnum = smaller
                smaller = _MAGICINTS[smallidx - 1] // 2
            else:
                smaller = smallnum
                smallnum = _MAGICINTS[smallidx] // 2
            sizesmall[0] = sizesmall[1] = sizesmall[2] = _MAGICINTS[smallidx]
    nbytes = st[0] + (1 if st[1] != 0 else 0)
    return minint, maxint, start_smallidx, data[:nbytes].copy(), ok


# ---------------------------------------------------------------------------
# frames


def _pad4(n: int) -> int:
    return (n + 3) & ~3


def xtc_frame_size(buf, offset: int) -> int:
    """Bytes of the XTC frame starting at ``offset``."""
    head = np.frombuffer(buf, ">i4", 3, offset)
    if head[0] != XTC_MAGIC:
        raise XDRError(f"bad XTC magic number {int(head[0])} at byte {offset}")
    natoms = int(head[1])
    pos = offset + 16 + 36
    if natoms <= 9:
        return pos + 4 + 12 * natoms - offset
    nbytes = int(np.frombuffer(buf, ">i4", 1, pos + 4 + 4 + 12 + 12 + 4)[0])
    return pos + 4 + 4 + 12 + 12 + 4 + 4 + _pad4(nbytes) - offset


def read_xtc_frame(buf, offset: int):
    """(natoms, step, time, box (3, 3) nm, coordinates (natoms, 3) nm float32)."""
    magic, natoms, step = np.frombuffer(buf, ">i4", 3, offset).tolist()
    if magic != XTC_MAGIC:
        raise XDRError(f"bad XTC magic number {magic} at byte {offset}")
    time = float(np.frombuffer(buf, ">f4", 1, offset + 12)[0])
    box = np.frombuffer(buf, ">f4", 9, offset + 16).astype(np.float64).reshape(3, 3)
    pos = offset + 52
    n = int(np.frombuffer(buf, ">i4", 1, pos)[0])
    if n != natoms:
        raise XDRError(f"XTC frame at byte {offset}: {n} coordinates for {natoms} atoms")
    pos += 4
    if natoms <= 9:
        xyz = np.frombuffer(buf, ">f4", 3 * natoms, pos).astype(np.float32).reshape(natoms, 3)
        return natoms, step, time, box, xyz
    precision = np.frombuffer(buf, ">f4", 1, pos)[0]
    ints = np.frombuffer(buf, ">i4", 8, pos + 4).astype(np.int64)
    minint, maxint, smallidx = ints[0:3], ints[3:6], int(ints[6])
    nbytes = int(ints[7])
    data = np.frombuffer(buf, np.uint8, nbytes, pos + 4 + 32)
    xyz = _decompress(np.ascontiguousarray(data), natoms, minint, maxint, smallidx,
                      np.float32(precision))  # fmt: skip
    return natoms, step, time, box, xyz


def xtc_frame_bytes(xyz_nm, box_nm, step: int, time: float, precision: float) -> bytes:
    """One XTC frame, byte-identical to xdrfile's write_xtc."""
    xyz = np.ascontiguousarray(xyz_nm, dtype=np.float32).reshape(-1, 3)
    natoms = len(xyz)
    head = np.array([XTC_MAGIC, natoms, int(step)], ">i4").tobytes()
    head += np.array([time], ">f4").tobytes()
    head += np.asarray(box_nm, dtype=">f4").reshape(9).tobytes()
    head += np.array([natoms], ">i4").tobytes()
    if natoms <= 9:
        return head + xyz.astype(">f4").tobytes()
    prec = np.float32(precision if precision > 0 else 1000.0)
    minint, maxint, smallidx, data, ok = _compress(xyz, prec)
    if not ok:
        raise XDRError("coordinates too large to compress at this precision")
    body = np.array([prec], ">f4").tobytes()
    body += np.concatenate([minint, maxint, [smallidx, len(data)]]).astype(">i4").tobytes()
    body += data.tobytes() + b"\0" * (_pad4(len(data)) - len(data))
    return head + body


def _trr_header(buf, offset: int) -> dict:
    ints = np.frombuffer(buf, ">i4", 2, offset)
    if ints[0] != TRR_MAGIC:
        raise XDRError(f"bad TRR magic number {int(ints[0])} at byte {offset}")
    slen = int(np.frombuffer(buf, ">i4", 1, offset + 8)[0])  # XDR string length
    pos = offset + 12 + _pad4(slen)
    names = ("ir_size", "e_size", "box_size", "vir_size", "pres_size", "top_size", "sym_size",
             "x_size", "v_size", "f_size", "natoms", "step", "nre")  # fmt: skip
    vals = np.frombuffer(buf, ">i4", 13, pos).tolist()
    h = dict(zip(names, vals, strict=True))
    pos += 52
    natoms = h["natoms"]
    if h["box_size"]:
        width = h["box_size"] // 9
    elif h["x_size"]:
        width = h["x_size"] // (natoms * 3)
    elif h["v_size"]:
        width = h["v_size"] // (natoms * 3)
    elif h["f_size"]:
        width = h["f_size"] // (natoms * 3)
    else:
        raise XDRError(f"TRR frame at byte {offset} holds no box, positions, velocities or forces")
    real = ">f8" if width == 8 else ">f4"
    h["real"] = real
    t, lam = np.frombuffer(buf, real, 2, pos).tolist()
    h["time"], h["lambda"] = t, lam
    h["data"] = pos + 2 * width
    h["end"] = h["data"] + sum(h[k] for k in ("ir_size", "e_size", "box_size", "vir_size",
                                             "pres_size", "top_size", "sym_size", "x_size",
                                             "v_size", "f_size"))  # fmt: skip
    return h


def trr_frame_size(buf, offset: int) -> int:
    return _trr_header(buf, offset)["end"] - offset


def read_trr_frame(buf, offset: int):
    """(natoms, step, time, box or None, x or None, v or None, f or None), nm units."""
    h = _trr_header(buf, offset)
    n, real = h["natoms"], h["real"]
    pos = h["data"] + h["ir_size"] + h["e_size"]
    box = None
    if h["box_size"]:
        box = np.frombuffer(buf, real, 9, pos).astype(np.float64).reshape(3, 3)
    pos += h["box_size"] + h["vir_size"] + h["pres_size"] + h["top_size"] + h["sym_size"]
    arrays = []
    for key in ("x_size", "v_size", "f_size"):
        if h[key]:
            arrays.append(np.frombuffer(buf, real, 3 * n, pos).reshape(n, 3))
        else:
            arrays.append(None)
        pos += h[key]
    return (n, h["step"], h["time"], box, *arrays)


def trr_frame_bytes(natoms: int, step: int, time: float, box_nm=None, x_nm=None, v=None,
                    f=None, lam: float = 0.0) -> bytes:  # fmt: skip
    """One single-precision TRR frame, as xdrfile's write_trr writes it."""
    sizes = [0, 0, 36 if box_nm is not None else 0, 0, 0, 0, 0,
             natoms * 12 if x_nm is not None else 0, natoms * 12 if v is not None else 0,
             natoms * 12 if f is not None else 0, natoms, int(step), 0]  # fmt: skip
    out = np.array([TRR_MAGIC, len(TRR_VERSION) + 1, len(TRR_VERSION)], ">i4").tobytes()
    out += TRR_VERSION + b"\0" * (_pad4(len(TRR_VERSION)) - len(TRR_VERSION))
    out += np.array(sizes, ">i4").tobytes() + np.array([time, lam], ">f4").tobytes()
    for arr, n in ((box_nm, 9), (x_nm, 3 * natoms), (v, 3 * natoms), (f, 3 * natoms)):
        if arr is not None:
            out += np.asarray(arr, dtype=">f4").reshape(n).tobytes()
    return out


def frame_offsets(buf, kind: str) -> np.ndarray:
    """Byte offset of every frame in an XTC or TRR file."""
    size_of = xtc_frame_size if kind == "xtc" else trr_frame_size
    offsets = []
    pos, end = 0, len(buf)
    while pos < end:
        offsets.append(pos)
        pos += size_of(buf, pos)
    if pos != end:
        raise XDRError(f"the last {kind.upper()} frame is truncated")
    return np.array(offsets, np.int64)
