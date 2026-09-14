"""Amber NetCDF trajectories (.nc, .ncdf), read and written natively.

Amber's trajectory format is the AMBER convention on NetCDF-3 (classic or
64-bit offset): ``coordinates`` (frame, atom, spatial) in Å, ``time`` in ps,
and ``cell_lengths``/``cell_angles`` per frame for periodic runs.  The file
layout is simple enough to read with numpy alone: a header lists dimensions,
attributes and variables, and every frame is one fixed-size record.
Restart files (no frame dimension) read as one frame.  NetCDF-4 files (HDF5
containers) are not supported.
"""

from __future__ import annotations

import os
import struct

import numpy as np

from ..trajectory import Frames, Trajectory

_NC_TYPES = {1: ">i1", 2: "S1", 3: ">i2", 4: ">i4", 5: ">f4", 6: ">f8"}
_DIMENSION, _VARIABLE, _ATTRIBUTE = 10, 11, 12


class NCDFError(ValueError):
    """Unreadable NetCDF file."""


def _pad4(n: int) -> int:
    return (n + 3) // 4 * 4


class _Header:
    """Dimensions, global attributes and variables of a NetCDF-3 file."""

    def __init__(self, buf):
        head = bytes(buf[:4])
        if head[:4] == b"\x89HDF":
            raise NCDFError("NetCDF-4 (HDF5) file; save it as NetCDF-3 (e.g. cpptraj "
                            "'trajout x.nc netcdf') to read it")  # fmt: skip
        if head[:3] != b"CDF" or head[3] not in (1, 2):
            raise NCDFError("not a NetCDF-3 file")
        self.version = head[3]
        self._buf, self._pos = buf, 4
        numrecs = self._int()
        self.numrecs = None if numrecs == -1 else numrecs  # -1: streaming
        self.dims = []
        tag, count = self._int(), self._int()
        if tag not in (0, _DIMENSION):
            raise NCDFError("bad dimension list")
        for _ in range(count):
            self.dims.append((self._name(), self._int()))
        self.attrs = self._attrs()
        self.vars = {}
        tag, count = self._int(), self._int()
        if tag not in (0, _VARIABLE):
            raise NCDFError("bad variable list")
        for _ in range(count):
            name = self._name()
            dimids = [self._int() for _ in range(self._int())]
            attrs = self._attrs()
            nctype, vsize = self._int(), self._int()
            begin = self._int64() if self.version == 2 else self._int()
            self.vars[name] = {"dims": [self.dims[d][0] for d in dimids],
                               "shape": [self.dims[d][1] for d in dimids],
                               "record": bool(dimids) and self.dims[dimids[0]][1] == 0,
                               "type": _NC_TYPES[nctype], "vsize": vsize, "begin": begin,
                               "attrs": attrs}  # fmt: skip
        self.size = self._pos

    def _bytes(self, n: int) -> bytes:
        out = bytes(self._buf[self._pos : self._pos + n])
        self._pos += _pad4(n)
        return out

    def _int(self) -> int:
        v = struct.unpack(">i", bytes(self._buf[self._pos : self._pos + 4]))[0]
        self._pos += 4
        return v

    def _int64(self) -> int:
        v = struct.unpack(">q", bytes(self._buf[self._pos : self._pos + 8]))[0]
        self._pos += 8
        return v

    def _name(self) -> str:
        return self._bytes(self._int()).decode("utf-8")

    def _attrs(self) -> dict:
        tag, count = self._int(), self._int()
        if tag not in (0, _ATTRIBUTE):
            raise NCDFError("bad attribute list")
        out = {}
        for _ in range(count):
            name = self._name()
            kind = _NC_TYPES[self._int()]
            n = self._int()
            raw = self._bytes(n * np.dtype(kind).itemsize)
            out[name] = raw.rstrip(b"\0").decode("latin-1") if kind == "S1" else \
                np.frombuffer(raw, kind, n).astype(np.float64)  # fmt: skip
        return out


class NCDFTrajectory(Trajectory):
    """Frames of an Amber NetCDF trajectory or restart file."""

    format = "ncdf"

    def __init__(self, path, system=None):
        super().__init__(path, system)
        self._mm = np.memmap(self.path, np.uint8, "r")
        h = self._h = _Header(self._mm)
        if "coordinates" not in h.vars:
            raise NCDFError(f"{self.path}: no coordinates variable")
        records = [v for v in h.vars.values() if v["record"]]
        self._recsize = 0
        if len(records) == 1:
            self._recsize = int(np.prod(records[0]["shape"][1:]) *
                                np.dtype(records[0]["type"]).itemsize)  # fmt: skip
        elif records:
            self._recsize = sum(v["vsize"] for v in records)
        coords = h.vars["coordinates"]
        natoms = coords["shape"][-2]
        if coords["record"]:
            n = h.numrecs
            if n is None:
                n = (len(self._mm) - coords["begin"]) // max(self._recsize, 1)
        else:
            n = 1
        self._setup(natoms, n)

    def _var(self, name: str, frame: int):
        v = self._h.vars.get(name)
        if v is None:
            return None
        shape = v["shape"][1:] if v["record"] else v["shape"]
        count = int(np.prod(shape)) if shape else 1
        offset = v["begin"] + (frame * self._recsize if v["record"] else 0)
        out = np.frombuffer(self._mm, v["type"], count, offset).astype(np.float64)
        if "scale_factor" in v["attrs"]:
            out = out * float(v["attrs"]["scale_factor"][0])
        return out

    def _read(self, idx: np.ndarray) -> Frames:
        from .pdb import cell_from_lengths_angles

        k = len(idx)
        pos = np.empty((k, self._natoms, 3), np.float32)
        boxes = np.zeros((k, 3, 3))
        times = np.zeros(k)
        for j, i in enumerate(idx.tolist()):
            pos[j] = self._var("coordinates", i).reshape(-1, 3)
            t = self._var("time", i)
            if t is not None:
                times[j] = t[0]
            lengths, angles = self._var("cell_lengths", i), self._var("cell_angles", i)
            if lengths is not None and lengths.any():
                ang = angles if angles is not None else np.full(3, 90.0)
                boxes[j] = cell_from_lengths_angles(*lengths, *ang)
        return Frames(idx.copy(), pos, boxes, times, idx.astype(np.int64))

    def close(self) -> None:
        self._mm = None


# ---------------------------------------------------------------------------
# writing


def _enc_int(v: int) -> bytes:
    return struct.pack(">i", v)


def _enc_name(s: str) -> bytes:
    raw = s.encode("utf-8")
    return _enc_int(len(raw)) + raw + b"\0" * (_pad4(len(raw)) - len(raw))


def _enc_attrs(attrs: dict) -> bytes:
    if not attrs:
        return _enc_int(0) + _enc_int(0)
    out = _enc_int(_ATTRIBUTE) + _enc_int(len(attrs))
    for name, value in attrs.items():
        raw = str(value).encode("latin-1")
        out += _enc_name(name) + _enc_int(2) + _enc_int(len(raw))
        out += raw + b"\0" * (_pad4(len(raw)) - len(raw))
    return out


class NCDFWriter:
    """Write an Amber NetCDF trajectory (NetCDF-3, 64-bit offsets): float32
    coordinates in Å, time in ps, and the cell as lengths and angles."""

    def __init__(self, path, natoms: int, dt: float = 1.0):
        from .. import __version__

        self.path = os.fspath(path)
        self.natoms = int(natoms)
        self.dt = float(dt)
        self.nframes = 0
        dims = [("frame", 0), ("spatial", 3), ("atom", self.natoms), ("cell_spatial", 3),
                ("cell_angular", 3), ("label", 5)]  # fmt: skip
        gattrs = {"Conventions": "AMBER", "ConventionVersion": "1.0", "program": "boonza",
                  "programVersion": __version__}  # fmt: skip
        # (name, dim ids, nc type, bytes per record or total, attrs, fixed data)
        fixed = [("spatial", [1], 2, b"xyz", {}), ("cell_spatial", [3], 2, b"abc", {}),
                 ("cell_angular", [4, 5], 2, b"alphabeta gamma", {})]  # fmt: skip
        record = [("time", [0], 5, 4, {"units": "picosecond"}),
                  ("coordinates", [0, 2, 1], 5, 12 * self.natoms, {"units": "angstrom"}),
                  ("cell_lengths", [0, 3], 6, 24, {"units": "angstrom"}),
                  ("cell_angles", [0, 4], 6, 24, {"units": "degree"})]  # fmt: skip

        def header(begins):
            out = b"CDF\x02" + _enc_int(0)
            out += _enc_int(_DIMENSION) + _enc_int(len(dims))
            out += b"".join(_enc_name(n) + _enc_int(size) for n, size in dims)
            out += _enc_attrs(gattrs)
            allvars = [(n, d, t, _pad4(len(data)), a) for n, d, t, data, a in fixed]
            allvars += [(n, d, t, size, a) for n, d, t, size, a in record]
            out += _enc_int(_VARIABLE) + _enc_int(len(allvars))
            for (name, dimids, kind, vsize, attrs), begin in zip(allvars, begins, strict=True):
                out += _enc_name(name) + _enc_int(len(dimids)) + b"".join(map(_enc_int, dimids))
                out += _enc_attrs(attrs) + _enc_int(kind) + _enc_int(vsize)
                out += struct.pack(">q", begin)
            return out

        nvars = len(fixed) + len(record)
        size = len(header([0] * nvars))
        begins, at = [], size
        for _, _, _, data, _ in fixed:
            begins.append(at)
            at += _pad4(len(data))
        self._recsize = sum(r[3] for r in record)
        for _, _, _, rsize, _ in record:
            begins.append(at)
            at += rsize
        self._fh = open(self.path, "wb")
        self._fh.write(header(begins))
        for _, _, _, data, _ in fixed:
            self._fh.write(data + b"\0" * (_pad4(len(data)) - len(data)))

    def write(self, positions, box=None, time=None, step=None) -> None:
        from .pdb import lengths_angles_from_cell

        pos = np.asarray(positions, dtype=">f4").reshape(self.natoms, 3)
        time = self.nframes * self.dt if time is None else float(time)
        box = None if box is None else np.asarray(box, dtype=np.float64).reshape(3, 3)
        cell = np.zeros(6) if box is None or not box.any() else \
            np.array(lengths_angles_from_cell(box))  # fmt: skip
        record = [np.array([time], ">f4"), pos, cell[:3].astype(">f8"), cell[3:].astype(">f8")]
        self._fh.write(b"".join(part.tobytes() for part in record))
        self.nframes += 1

    def write_frames(self, frames) -> None:
        for frame in frames:
            self.write(frame.positions, frame.box, frame.time, frame.step)

    def close(self) -> None:
        if self._fh is None:
            return
        self._fh.seek(4)
        self._fh.write(_enc_int(self.nframes))
        self._fh.close()
        self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
