"""Desmond DTR trajectories (and STK lists of them), read natively.

A DTR is a directory.  ``timekeys`` indexes the frames (time, byte offset,
size) and ``frameNNNNNNNNN`` files hold ``frames_per_file`` frames each.
Every frame is a self-describing block of named, typed fields (POSITION,
VELOCITY, UNITCELL, CHEMICAL_TIME, ...) in the byte order of the machine
that wrote it.  Positions are in Å and times in ps, as in Desmond.

An STK file lists DTRs, one per line (relative paths are taken from the
STK's directory).  As in msys, the frames of each DTR from the first time of
the next non-empty DTR on are dropped, so restarted runs join seamlessly.
"""

from __future__ import annotations

import os

import numpy as np

from ..trajectory import Frames, Trajectory

MAGIC_FRAME = 0x4445534D  # "DESM"
MAGIC_TIMEKEY = 0x4445534B  # "DESK"
_TYPES = {"int32_t": "i4", "uint32_t": "u4", "int64_t": "i8", "uint64_t": "u8", "float": "f4",
          "double": "f8", "char": "S1", "unsigned char": "u1"}  # fmt: skip
_POSITIONS = ("POSITION", "POSN", "POS")
_BOXES = ("UNITCELL", "HOME_BOX")


class DTRError(ValueError):
    """Malformed DTR data."""


def _align8(n: int) -> int:
    return (n + 7) // 8 * 8


def _names(block: bytes, count: int | None = None) -> list[str]:
    """NUL-separated strings, up to the first empty one (or ``count`` of them)."""
    out = []
    for part in block.split(b"\0"):
        if (count is None and not part) or (count is not None and len(out) == count):
            break
        out.append(part.decode("latin-1"))
    return out


def parse_frame(buf) -> dict[str, np.ndarray]:
    """The named fields of one frame, as numpy arrays (strings for char fields)."""
    buf = memoryview(buf).cast("B")
    head = np.frombuffer(buf, ">u4", count=24)
    if head[0] != MAGIC_FRAME:
        raise DTRError(f"frame magic number {int(head[0]):#x}, want {MAGIC_FRAME:#x}")
    headersize, endian, nlabels = int(head[4]), int(head[12]), int(head[13])
    metasize, typesize, labelsize, scalarsize = (int(x) for x in head[14:18])
    if endian not in (1234, 4321):
        raise DTRError(f"unsupported frame endianism {endian}")
    order = "<" if endian == 1234 else ">"
    if nlabels == 0:
        return {}
    meta = np.frombuffer(buf, ">u4", count=4 * nlabels, offset=headersize).reshape(nlabels, 4)
    type_start = headersize + metasize
    label_start = type_start + typesize
    scalar_start = label_start + labelsize
    field_start = scalar_start + scalarsize
    types = _names(bytes(buf[type_start:label_start]))
    labels = _names(bytes(buf[label_start:scalar_start]), nlabels)
    out: dict[str, np.ndarray] = {}
    scalars, fields = scalar_start, field_start
    for label, (code, elemsize, count_lo, count_hi) in zip(labels, meta.tolist(), strict=True):
        count = int(count_lo) | (int(count_hi) << 32)
        nbytes = int(elemsize) * count
        if count <= 1:
            addr, scalars = scalars, scalars + _align8(nbytes)
        else:
            addr, fields = fields, fields + _align8(nbytes)
        kind = _TYPES.get(types[code]) if code < len(types) else None
        if kind is None:
            raise DTRError(f"field {label!r}: unknown type")
        if kind == "S1":
            out[label] = np.array(bytes(buf[addr : addr + nbytes]).rstrip(b"\0").decode("latin-1"))
        else:
            out[label] = np.frombuffer(buf, np.dtype(kind).newbyteorder(order), count, addr)
    return out


def _timekeys(dtr: str):
    """(frames per file, times, offsets, sizes) from a DTR's ``timekeys`` file."""
    raw = np.fromfile(os.path.join(dtr, "timekeys"), dtype=">u4")
    if len(raw) < 3 or raw[0] != MAGIC_TIMEKEY:
        raise DTRError(f"{dtr}: bad timekeys file")
    fpf, record = int(raw[1]), int(raw[2])
    if record != 24 or (len(raw) - 3) % 6:
        raise DTRError(f"{dtr}: timekeys records of {record} bytes, want 24")
    rec = raw[3:].reshape(-1, 6).astype(np.uint64)
    times = ((rec[:, 1] << np.uint64(32)) | rec[:, 0]).view(np.float64)
    offsets = ((rec[:, 3] << np.uint64(32)) | rec[:, 2]).astype(np.int64)
    sizes = ((rec[:, 5] << np.uint64(32)) | rec[:, 4]).astype(np.int64)
    return max(fpf, 1), times, offsets, sizes


def _stk_members(path: str) -> list[str]:
    base = os.path.dirname(os.path.abspath(path))
    with open(path) as fh:
        lines = [line.strip() for line in fh]
    return [os.path.join(base, line) for line in lines if line and not line.startswith("#")]


class DTRTrajectory(Trajectory):
    """Frames of a DTR directory or an STK list of DTRs."""

    format = "dtr"

    def __init__(self, path, system=None):
        super().__init__(path, system)
        dtrs = [self.path] if os.path.isdir(self.path) else _stk_members(self.path)
        keys = [(d, *_timekeys(d)) for d in dtrs]
        keys = [k for k in keys if len(k[2])]
        if not keys:
            raise DTRError(f"{self.path}: no frames")
        where, times = [], []
        for n, key in enumerate(keys):
            t = key[2]
            local = np.arange(len(t))
            if n + 1 < len(keys):  # later runs supersede from their first frame on
                local = local[t < keys[n + 1][2][0]]
            where.extend((n, int(i)) for i in local)
            times.append(t[local])
        self._keys = keys
        self._where = where
        self._times = np.concatenate(times)
        self._fh = None
        self._fh_path = None
        first = self._fields(0)
        pos = next((first[k] for k in _POSITIONS if k in first), None)
        if pos is None:
            raise DTRError(f"{self.path}: frames have no POSITION field")
        self._double = pos.dtype.itemsize == 8
        self._setup(len(pos) // 3, len(where))

    def _fields(self, frame: int) -> dict[str, np.ndarray]:
        n, i = self._where[frame]
        dtr, fpf, _, offsets, sizes = self._keys[n]
        name = os.path.join(dtr, f"frame{i // fpf:09d}")
        if name != self._fh_path:
            if self._fh is not None:
                self._fh.close()
            self._fh, self._fh_path = open(name, "rb"), name
        self._fh.seek(int(offsets[i]))
        buf = self._fh.read(int(sizes[i]))
        if len(buf) != int(sizes[i]):
            raise DTRError(f"{name}: frame {i} is truncated")
        return parse_frame(buf)

    def _read(self, idx: np.ndarray) -> Frames:
        k = len(idx)
        pos = np.empty((k, self._natoms, 3), np.float64 if self._double else np.float32)
        boxes = np.zeros((k, 3, 3))
        times = self._times[idx].astype(np.float64)
        for j, i in enumerate(idx.tolist()):
            f = self._fields(i)
            xyz = next((f[key] for key in _POSITIONS if key in f), None)
            if xyz is None or len(xyz) != 3 * self._natoms:
                raise DTRError(f"{self.path}: frame {i} has no positions for {self._natoms} atoms")
            pos[j] = xyz.reshape(-1, 3)
            box = next((f[key] for key in _BOXES if key in f), None)
            if box is not None and len(box) == 9:
                boxes[j] = box.reshape(3, 3)
            if "CHEMICAL_TIME" in f:
                times[j] = float(f["CHEMICAL_TIME"][0])
        return Frames(idx.copy(), pos, boxes, times, idx.astype(np.int64))

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = self._fh_path = None
