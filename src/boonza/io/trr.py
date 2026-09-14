"""GROMACS TRR trajectories, read and written natively.

TRR frames are uncompressed XDR: a header, then the box and the positions,
velocities and forces that the frame carries, in single or double
precision.  Positions and boxes are converted from nm to Å; frames that
carry only velocities or forces have NaN positions.  Files are written
byte-identical to GROMACS's xdrfile library.
"""

from __future__ import annotations

import os

import numpy as np

from ..trajectory import Frames, Trajectory
from ._xdr import XDRError, frame_offsets, read_trr_frame, trr_frame_bytes

NM = 10.0


class TRRTrajectory(Trajectory):
    format = "trr"

    def __init__(self, path, system=None):
        super().__init__(path, system)
        if os.path.getsize(self.path) == 0:
            raise XDRError(f"{self.path} is empty")
        self._buf = np.memmap(self.path, np.uint8, "r")
        self._offsets = frame_offsets(self._buf, "trr")
        natoms = read_trr_frame(self._buf, 0)[0]
        self._setup(natoms, len(self._offsets))

    def _read(self, idx: np.ndarray) -> Frames:
        k = len(idx)
        pos = np.empty((k, self._natoms, 3), np.float32)
        boxes = np.zeros((k, 3, 3))
        times = np.empty(k)
        steps = np.empty(k, np.int64)
        for j, i in enumerate(idx.tolist()):
            _, step, time, box, x, _, _ = read_trr_frame(self._buf, int(self._offsets[i]))
            pos[j] = np.nan if x is None else x
            if box is not None:
                boxes[j] = box * NM
            times[j] = time
            steps[j] = step
        pos *= NM
        return Frames(idx.copy(), pos, boxes, times, steps)

    def close(self) -> None:
        self._buf = None


class TRRWriter:
    """Write a single-precision TRR file (positions and box)."""

    def __init__(self, path, natoms: int, dt: float = 1.0):
        self.path = os.fspath(path)
        self.natoms = int(natoms)
        self.dt = float(dt)
        self.nframes = 0
        self._fh = open(self.path, "wb")

    def write(self, positions, box=None, time=None, step=None) -> None:
        pos = np.asarray(positions, dtype=np.float32).reshape(self.natoms, 3) / NM
        box = np.zeros((3, 3)) if box is None else np.asarray(box, dtype=np.float64) / NM
        time = self.nframes * self.dt if time is None else float(time)
        step = self.nframes if step is None else int(step)
        self._fh.write(trr_frame_bytes(self.natoms, step, time, box.astype(np.float32), pos))
        self.nframes += 1

    def write_frames(self, frames) -> None:
        for frame in frames:
            self.write(frame.positions, frame.box, frame.time, frame.step)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
