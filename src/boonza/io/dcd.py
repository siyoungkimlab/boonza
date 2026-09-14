"""CHARMM / NAMD / X-PLOR DCD trajectories, memory-mapped.

Frames are fixed-size Fortran records, so the file is mapped as an array of
frame records and any frame is read without scanning.  Both byte orders,
the CHARMM unit-cell block, the 4th-dimension block and fixed atoms are
handled.  Unit cells follow MDAnalysis: ``[A, gamma, B, beta, alpha, C]`` with
angle cosines (CHARMM, NAMD > 2.5), degrees (NAMD 2.5), or box vectors
(recent CHARMM).  The frame count comes from the file size, since NAMD may
leave the header count unset.
"""

from __future__ import annotations

import math
import os

import numpy as np

from ..trajectory import Frames, Trajectory
from .pdb import cell_from_lengths_angles, lengths_angles_from_cell

AKMA_PS = 0.04888821  # CHARMM time unit in ps
_IS_CHARMM, _HAS_4D, _HAS_CELL = 0x01, 0x02, 0x04


class DCDError(ValueError):
    """Malformed or unsupported DCD content."""


def _frame_dtype(endian: str, natoms: int, cell: bool, ndims: int) -> np.dtype:
    fields = []
    if cell:
        fields += [("c0", endian + "i4"), ("cell", endian + "f8", (6,)), ("c1", endian + "i4")]
    for axis in "xyzw"[:ndims]:
        fields += [(axis + "0", endian + "i4"), (axis, endian + "f4", (natoms,)),
                   (axis + "1", endian + "i4")]  # fmt: skip
    return np.dtype(fields)


class DCDTrajectory(Trajectory):
    format = "dcd"

    def __init__(self, path, system=None):
        super().__init__(path, system)
        self._read_header()
        cell = bool(self.charmm & _IS_CHARMM and self.charmm & _HAS_CELL)
        ndims = 4 if self.charmm & _IS_CHARMM and self.charmm & _HAS_4D else 3
        n, nfree = self.header["natoms"], self.header["natoms"] - self.header["nfixed"]
        self._first_dtype = _frame_dtype(self._endian, n, cell, ndims)
        self._dtype = _frame_dtype(self._endian, nfree, cell, ndims)
        size = os.path.getsize(self.path) - self._header_size
        if size <= 0:
            nframes = 0
        elif self.header["nfixed"]:
            nframes = 1 + max(size - self._first_dtype.itemsize, 0) // self._dtype.itemsize
        else:
            nframes = size // self._dtype.itemsize
        self._cell = cell
        self._mm_first = self._mm = None
        if nframes:
            first = np.memmap(self.path, self._first_dtype, "r", self._header_size, (1,))
            if self.header["nfixed"]:
                self._mm_first = first
                if nframes > 1:
                    self._mm = np.memmap(self.path, self._dtype, "r",
                                         self._header_size + self._first_dtype.itemsize,
                                         (nframes - 1,))  # fmt: skip
            else:
                self._mm = np.memmap(self.path, self._dtype, "r", self._header_size, (nframes,))
            self._check(first, n)
        self._setup(n, nframes)

    def _read_header(self) -> None:
        with open(self.path, "rb") as f:
            magic = f.read(4)
            if len(magic) < 4:
                raise DCDError("premature end of file in DCD header")
            if int.from_bytes(magic, "little") == 84:
                e = "<"
            elif int.from_bytes(magic, "big") == 84:
                e = ">"
            else:
                raise DCDError("not a DCD file")

            def ints(count: int) -> np.ndarray:
                data = f.read(4 * count)
                if len(data) < 4 * count:
                    raise DCDError("premature end of file in DCD header")
                return np.frombuffer(data, e + "i4")

            hdr = f.read(84)
            if len(hdr) < 84 or hdr[:4] != b"CORD":
                raise DCDError("DCD header lacks the CORD signature")
            ic = np.frombuffer(hdr[4:84], e + "i4")
            charmm = 0
            if ic[19] != 0:
                charmm = _IS_CHARMM
                if ic[10] != 0:
                    charmm |= _HAS_CELL
                if ic[11] == 1:
                    charmm |= _HAS_4D
            if charmm:
                delta = float(np.frombuffer(hdr[40:44], e + "f4")[0])
            else:
                delta = float(np.frombuffer(hdr[40:48], e + "f8")[0])
            if ints(1)[0] != 84:
                raise DCDError("bad DCD header record")
            size = int(ints(1)[0])
            if (size - 4) % 80:
                raise DCDError("bad DCD title record")
            ntitle = int(ints(1)[0])
            title = f.read(80 * ntitle)
            ints(1)
            if ints(1)[0] != 4:
                raise DCDError("bad DCD atom-count record")
            natoms = int(ints(1)[0])
            if ints(1)[0] != 4:
                raise DCDError("bad DCD atom-count record")
            nfixed = int(ic[8])
            self._free = None
            if nfixed:
                nfree = natoms - nfixed
                if ints(1)[0] != 4 * nfree:
                    raise DCDError("bad DCD free-atom record")
                self._free = ints(nfree).astype(np.int64) - 1
                ints(1)
            self._header_size = f.tell()
        self._endian = e
        self.charmm = charmm
        self.header = {
            "natoms": natoms, "nframes_header": int(ic[0]), "istart": int(ic[1]),
            "nsavc": int(ic[2]), "nfixed": nfixed, "delta": delta,
            "remarks": title.decode("latin-1", "replace").rstrip("\0 "),
        }  # fmt: skip

    @staticmethod
    def _check(records, natoms) -> None:
        for axis in "xyz":
            if (records[axis + "0"] != 4 * natoms).any() or (
                records[axis + "1"] != 4 * natoms
            ).any():
                raise DCDError("corrupt DCD frame record")

    @property
    def dt(self) -> float:
        """Time between frames in ps."""
        return self.header["delta"] * AKMA_PS * max(self.header["nsavc"], 1)

    def _read(self, idx: np.ndarray) -> Frames:
        n = self._natoms
        k = len(idx)
        pos = np.empty((k, n, 3), np.float32)
        cells = np.tile(np.array([0.0, 90.0, 0.0, 90.0, 90.0, 0.0]), (k, 1))
        if self.header["nfixed"]:
            first = self._mm_first[0]
            template = np.column_stack([first["x"], first["y"], first["z"]]).astype(np.float32)
            nfree = n - self.header["nfixed"]
            for j, i in enumerate(idx.tolist()):
                if i == 0:
                    pos[j] = template
                    if self._cell:
                        cells[j] = first["cell"]
                    continue
                rec = self._mm[i - 1]
                if rec["x0"] != 4 * nfree:
                    raise DCDError("corrupt DCD frame record")
                pos[j] = template
                for d, axis in enumerate("xyz"):
                    pos[j, self._free, d] = rec[axis]
                if self._cell:
                    cells[j] = rec["cell"]
        elif k:
            rec = self._mm[idx]
            self._check(rec, n)
            for d, axis in enumerate("xyz"):
                pos[:, :, d] = rec[axis]
            if self._cell:
                cells[:] = rec["cell"]
        nsavc = max(self.header["nsavc"], 1)
        times = (idx + self.header["istart"] / nsavc) * self.dt
        steps = self.header["istart"] + idx * self.header["nsavc"]
        return Frames(idx.copy(), pos, _boxes(cells), times, steps)


def _boxes(cells: np.ndarray) -> np.ndarray:
    """DCD unit cell records -> (k, 3, 3) box vectors, following MDAnalysis."""
    out = np.zeros((len(cells), 3, 3))
    for j, raw in enumerate(cells):
        uc = raw[[0, 2, 5, 4, 3, 1]]  # A, B, C, alpha, beta, gamma
        if np.all((uc[3:] >= -1.0) & (uc[3:] <= 1.0)):  # angle cosines
            uc[3:] = 90.0 - np.arcsin(uc[3:]) * 90.0 / (np.pi / 2)
        elif np.any(uc < 0.0) or np.any(uc[3:] > 180.0):  # box vectors
            out[j] = np.array([raw[[0, 1, 3]], raw[[1, 2, 4]], raw[[3, 4, 5]]])
            continue
        if uc[0] > 0 and uc[1] > 0 and uc[2] > 0:
            out[j] = cell_from_lengths_angles(*uc)
    return out


class DCDWriter:
    """Write a CHARMM-style DCD with a unit cell block (NAMD/VMD cosine convention).

    The cell is stored as lengths and angle cosines, so a box that is not
    oriented with a along x and b in the xy plane reads back reoriented
    (same lattice, different vectors).  Use XTC to keep box vectors as-is.
    """

    def __init__(self, path, natoms: int, dt: float = 1.0, nsavc: int = 1, istart=None,
                 remarks: str = "Created by boonza"):  # fmt: skip
        self.path = os.fspath(path)
        self.natoms = int(natoms)
        self.nsavc = int(nsavc)
        self.istart = self.nsavc if istart is None else int(istart)
        self.nframes = 0
        delta = float(dt) / AKMA_PS / self.nsavc
        self._f = open(self.path, "wb")
        i4 = lambda *v: np.array(v, "<i4").tobytes()  # noqa: E731
        head = b"CORD" + i4(0, self.istart, self.nsavc, 0, 0, 0, 0, 0, 0)
        head += np.array([delta], "<f4").tobytes() + i4(1) + i4(*[0] * 8) + i4(24)
        title = remarks.encode("latin-1", "replace")[:239].ljust(240, b"\0")
        self._f.write(i4(84) + head + i4(84) + i4(244, 3) + title + i4(244))
        self._f.write(i4(4, self.natoms, 4))

    def write(self, positions, box=None, time=None, step=None) -> None:
        """Append one frame; ``box`` rows are the cell vectors (None or zeros: no cell)."""
        pos = np.asarray(positions, dtype=np.float32).reshape(self.natoms, 3)
        cell = np.zeros(6)
        if box is not None and np.asarray(box).any():
            a, b, c, al, be, ga = lengths_angles_from_cell(box)
            cos = [math.sin(math.radians(90.0 - x)) for x in (ga, be, al)]
            cell = np.array([a, cos[0], b, cos[1], cos[2], c])
        size = np.array([4 * self.natoms], "<i4").tobytes()
        out = [np.array([48], "<i4").tobytes(), cell.astype("<f8").tobytes(),
               np.array([48], "<i4").tobytes()]  # fmt: skip
        for d in range(3):
            out += [size, np.ascontiguousarray(pos[:, d]).astype("<f4").tobytes(), size]
        self._f.write(b"".join(out))
        self.nframes += 1

    def write_frames(self, frames) -> None:
        for frame in frames:
            self.write(frame.positions, frame.box)

    def close(self) -> None:
        if self._f.closed:
            return
        self._f.seek(8)
        self._f.write(np.array([self.nframes], "<i4").tobytes())
        self._f.seek(20)
        self._f.write(np.array([self.istart + self.nframes * self.nsavc], "<i4").tobytes())
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
