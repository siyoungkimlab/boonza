"""GROMACS TRR trajectories through MDAnalysis's compiled XDR library.

As for XTC, decoding uses the C implementation from MDAnalysis (optional
dependency), positions and boxes are converted from nm to Å, and frame
offsets let any frame be read directly.  TRR frames may carry only
velocities or forces; their positions are NaN.
"""

from __future__ import annotations

import os

import numpy as np

from ..trajectory import Frames, Trajectory
from .xtc import NM, _xdr


class TRRTrajectory(Trajectory):
    format = "trr"

    def __init__(self, path, system=None):
        super().__init__(path, system)
        self._f = _xdr().TRRFile(self.path, "r")
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
            frame = self._f.read()
            self._next = i + 1
            pos[j] = frame.x if frame.hasx else np.nan
            boxes[j] = np.asarray(frame.box, dtype=np.float64) * NM
            times[j] = frame.time
            steps[j] = frame.step
        pos *= NM
        return Frames(idx.copy(), pos, boxes, times, steps)

    def close(self) -> None:
        self._f.close()


class TRRWriter:
    """Write a TRR file (positions and box; full precision, unlike XTC)."""

    def __init__(self, path, natoms: int, dt: float = 1.0):
        self.path = os.fspath(path)
        self.natoms = int(natoms)
        self.dt = float(dt)
        self.nframes = 0
        self._f = _xdr().TRRFile(self.path, "w")

    def write(self, positions, box=None, time=None, step=None) -> None:
        pos = np.asarray(positions, dtype=np.float32).reshape(self.natoms, 3) / NM
        box = np.zeros((3, 3)) if box is None else np.asarray(box, dtype=np.float64) / NM
        time = self.nframes * self.dt if time is None else float(time)
        step = self.nframes if step is None else int(step)
        self._f.write(pos, None, None, box.astype(np.float32), step, time, 0.0, self.natoms)
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
