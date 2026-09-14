"""GROMACS XTC trajectories through MDAnalysis's compiled XDR library.

XTC coordinates are compressed, so decoding uses the C implementation from
MDAnalysis (optional dependency).  Positions and boxes are converted from nm
to Å; frame offsets let any frame be read directly.
"""

from __future__ import annotations

import os

import numpy as np

from ..trajectory import Frames, Trajectory

NM = 10.0


def _xdr():
    try:
        from MDAnalysis.lib.formats import libmdaxdr
    except ImportError as e:
        raise ImportError(
            "XTC support uses MDAnalysis's compiled XDR reader; install MDAnalysis"
        ) from e
    return libmdaxdr


class XTCTrajectory(Trajectory):
    format = "xtc"

    def __init__(self, path, system=None):
        super().__init__(path, system)
        self._f = _xdr().XTCFile(self.path, "r")
        self._next = 0
        self._setup(self._f.n_atoms, len(self._f.offsets))

    def _read(self, idx: np.ndarray) -> Frames:
        k = len(idx)
        pos = np.empty((k, self._natoms, 3), np.float32)
        boxes = np.empty((k, 3, 3))
        times = np.empty(k)
        steps = np.empty(k, np.int64)
        for j, i in enumerate(idx.tolist()):
            if i != self._next:
                self._f.seek(i)
            frame = self._f.read_direct_x(pos[j])
            self._next = i + 1
            boxes[j] = np.asarray(frame.box, dtype=np.float64) * NM
            times[j] = frame.time
            steps[j] = frame.step
        pos *= NM
        return Frames(idx.copy(), pos, boxes, times, steps)

    def close(self) -> None:
        self._f.close()


class XTCWriter:
    """Write an XTC file; ``precision`` is in 1/nm (1000 keeps 0.001 nm)."""

    def __init__(self, path, natoms: int, precision: float = 1000.0, dt: float = 1.0):
        self.path = os.fspath(path)
        self.natoms = int(natoms)
        self.precision = float(precision)
        self.dt = float(dt)
        self.nframes = 0
        self._f = _xdr().XTCFile(self.path, "w")

    def write(self, positions, box=None, time=None, step=None) -> None:
        pos = np.asarray(positions, dtype=np.float32).reshape(self.natoms, 3) / NM
        box = np.zeros((3, 3)) if box is None else np.asarray(box, dtype=np.float64) / NM
        time = self.nframes * self.dt if time is None else float(time)
        step = self.nframes if step is None else int(step)
        self._f.write(pos, box.astype(np.float32), step, time, self.precision)
        self.nframes += 1

    def write_frames(self, frames) -> None:
        for frame in frames:
            self.write(frame.positions, frame.box, frame.time, frame.step)

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
