"""Trajectories: lazy, random-access frames of positions and unit cell.

    traj = boonza.open_trajectory("run.xtc", system)
    len(traj), traj[10].positions, traj[::10]         # frames, views
    for block in traj.chunks(500, atoms="protein"):   # (nframes, natoms, 3) blocks
        ...

Positions are float32 in Å, boxes are (3, 3) float64 with the cell vectors
as rows (zeros when the file has no cell), times in ps.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Frame:
    """One frame: positions (natoms, 3), box (3, 3), time (ps) and MD step."""

    index: int
    positions: np.ndarray
    box: np.ndarray
    time: float
    step: int


@dataclass
class Frames:
    """A block of frames: positions (nframes, natoms, 3), boxes (nframes, 3, 3), ..."""

    indices: np.ndarray
    positions: np.ndarray
    boxes: np.ndarray
    times: np.ndarray
    steps: np.ndarray

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int) -> Frame:
        return Frame(
            int(self.indices[k]), self.positions[k], self.boxes[k],
            float(self.times[k]), int(self.steps[k]),
        )  # fmt: skip

    def __iter__(self):
        return (self[k] for k in range(len(self)))


class Trajectory:
    """Base class; formats implement ``_read(indices) -> Frames`` for all atoms."""

    format = ""

    def __init__(self, path, system=None):
        self.path = os.fspath(path)
        self.system = system
        self._natoms = 0
        self._index = np.empty(0, np.int64)

    def _setup(self, natoms: int, n_frames: int) -> None:
        self._natoms = int(natoms)
        self._index = np.arange(n_frames, dtype=np.int64)
        if self.system is not None and self.system.natoms != self._natoms:
            raise ValueError(
                f"{self.path} has {self._natoms} atoms but the system has {self.system.natoms}"
            )

    # ---- sizes and access ------------------------------------------------
    @property
    def natoms(self) -> int:
        return self._natoms

    @property
    def n_frames(self) -> int:
        return len(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, key):
        """``traj[i]`` is a Frame; slices, index arrays and masks give a lazy view."""
        if isinstance(key, int | np.integer):
            i = int(key) + (len(self) if key < 0 else 0)
            if not 0 <= i < len(self):
                raise IndexError(f"frame {key} out of range ({len(self)} frames)")
            return self.read([i])[0]
        view = copy.copy(self)
        view._index = self._index[key]
        return view

    def _atom_ids(self, atoms):
        if atoms is None:
            return None
        if isinstance(atoms, str):
            if self.system is None:
                raise ValueError("selection strings need a trajectory opened with a system")
            return self.system.select(atoms).ids
        if hasattr(atoms, "ids"):
            return atoms.ids
        return np.asarray(atoms, dtype=np.int64)

    def read(self, frames=None, atoms=None) -> Frames:
        """Read frames (indices, slice or mask into this view; all by default).

        ``atoms`` restricts positions to a subset: indices, an AtomSel, or a
        selection string when the trajectory has a system.
        """
        if frames is None:
            idx = self._index
        else:
            idx = np.atleast_1d(self._index[frames])
        block = self._read(np.asarray(idx, dtype=np.int64))
        ids = self._atom_ids(atoms)
        if ids is not None:
            block.positions = block.positions[:, ids]
        return block

    def chunks(self, size: int = 256, atoms=None):
        """Iterate over blocks of up to ``size`` frames."""
        ids = self._atom_ids(atoms)
        for start in range(0, len(self), size):
            yield self.read(slice(start, start + size), ids)

    def __iter__(self):
        for block in self.chunks():
            yield from block

    def _read(self, indices: np.ndarray) -> Frames:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.path!r}: {len(self)} frames, {self.natoms} atoms>"


def _format(path, format: str | None) -> str:
    return (format or Path(path).suffix).lower().lstrip(".")


def open_trajectory(path, system=None, format: str | None = None) -> Trajectory:
    """Open a DCD or XTC trajectory; with ``system`` the atom counts must agree."""
    fmt = _format(path, format)
    if fmt == "dcd":
        from .io.dcd import DCDTrajectory

        return DCDTrajectory(path, system)
    if fmt == "xtc":
        from .io.xtc import XTCTrajectory

        return XTCTrajectory(path, system)
    raise ValueError(f"cannot read {fmt!r} trajectories; supported: dcd, xtc")


def open_writer(path, natoms: int, format: str | None = None, **kwargs):
    """A DCD or XTC writer: ``with open_writer(p, n) as w: w.write(positions, box)``."""
    fmt = _format(path, format)
    if fmt == "dcd":
        from .io.dcd import DCDWriter

        return DCDWriter(path, natoms, **kwargs)
    if fmt == "xtc":
        from .io.xtc import XTCWriter

        return XTCWriter(path, natoms, **kwargs)
    raise ValueError(f"cannot write {fmt!r} trajectories; supported: dcd, xtc")
