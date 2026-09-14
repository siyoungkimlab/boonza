"""Periodic "glue": whole molecules, glued groups, wrapping and fitting per frame.

A port of msys pfx and viswizard's glue, extended to triclinic cells:

1. make molecules whole along a spanning tree of the bonds;
2. glue: the molecules touching each glue selection move to the periodic
   images that bring them jointly closest (optimal shifts, R. Lippert's
   algorithm, as in msys), and are treated as one unit from then on;
3. wrap every other unit into the box around the ``center`` atoms (or into
   the primary cell when there is no center);
4. fit: superpose the ``fit`` atoms onto a reference (the first frame by
   default); the box vectors rotate with the atoms.

    fix = Glue(system, glue="protein or resname LIG", center="protein",
               fit="protein and name CA")
    pos, box = fix(frame.positions, frame.box)
    for block in fix.frames(traj):   # a Trajectory, fixed chunk by chunk
        ...
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .align import kabsch


@njit(cache=True)
def _spanning_tree(n, off, nbr, include):
    """Breadth-first (parent, child) edges covering each connected component."""
    seen = np.zeros(n, np.bool_)
    queue = np.empty(n, np.int64)
    parent = np.empty(n, np.int64)
    child = np.empty(n, np.int64)
    k = 0
    for s in range(n):
        if seen[s] or not include[s]:
            continue
        seen[s] = True
        head, tail = 0, 1
        queue[0] = s
        while head < tail:
            a = queue[head]
            head += 1
            for t in range(off[a], off[a + 1]):
                b = nbr[t]
                if include[b] and not seen[b]:
                    seen[b] = True
                    queue[tail] = b
                    tail += 1
                    parent[k] = a
                    child[k] = b
                    k += 1
    return parent[:k], child[:k]


@njit(cache=True)
def _make_whole(pos, parent, child, box, inv):
    for k in range(parent.shape[0]):
        p, c = parent[k], child[k]
        dx, dy, dz = pos[c, 0] - pos[p, 0], pos[c, 1] - pos[p, 1], pos[c, 2] - pos[p, 2]
        na = np.rint(dx * inv[0, 0] + dy * inv[1, 0] + dz * inv[2, 0])
        nb = np.rint(dx * inv[0, 1] + dy * inv[1, 1] + dz * inv[2, 1])
        nc = np.rint(dx * inv[0, 2] + dy * inv[1, 2] + dz * inv[2, 2])
        for d in range(3):
            pos[c, d] -= na * box[0, d] + nb * box[1, d] + nc * box[2, d]


def optimal_shifts(x: np.ndarray) -> np.ndarray:
    """Integer shifts that bring 1-d periodic points (unit period) jointly closest."""
    n = len(x)
    if n < 2:
        return np.zeros(n)
    s = -np.floor(x + 0.5)  # bring each point into [-1/2, 1/2)
    y = x + s
    tot, tot2 = y.sum(), (y * y).sum()
    xx = np.sort(y)
    v = np.empty(n)
    v[0] = n * tot2 - tot * tot
    v[1:] = v[0] + np.cumsum((n - 1) - 2 * (tot + np.arange(n - 1) - n * xx[:-1]))
    return s + (y < xx[np.argmin(v)])


def make_whole(system, positions=None, box=None) -> np.ndarray:
    """Positions with every bonded molecule unbroken across the box (a copy)."""
    fix = Glue(system, wrap=False)
    pos, _ = fix(system.positions if positions is None else positions,
                 system.cell if box is None else box)  # fmt: skip
    return pos


class Glue:
    """Per-frame periodic fixing; see the module docstring.

    ``glue``: selection or list of selections (strings or atom indices); the
    molecules touching each one are kept together.  ``center``: atoms whose
    center the box is wrapped around.  ``fit``: atoms superposed onto
    ``reference`` (positions of all atoms or of the fit atoms; the first
    processed frame when None).  ``whole``: atoms to make whole ("all", a
    selection, or None to skip).  ``weights``: "mass" or None for fitting.
    """

    def __init__(self, system, glue=None, center=None, fit=None, reference=None,
                 whole="all", wrap: bool = True, weights=None):  # fmt: skip
        self.system = system
        n = system.natoms
        self.wrap = wrap
        frag = system.fragids.copy()

        include = np.zeros(n, bool)
        if whole is not None:
            include[self._ids(whole)] = True
        if include.any():
            off, nbr, _ = system._adjacency()
            self._tree = _spanning_tree(n, off, nbr, include)
        else:
            self._tree = None

        # glue groups: fragments touching each selection; centers use only the
        # selected atoms of each fragment, as msys does
        self._groups = []
        unit = frag.copy()
        for sel in [glue] if isinstance(glue, str) else (glue or []):
            ids = self._ids(sel)
            if len(ids) == 0:
                continue
            frags = np.unique(frag[ids])
            members = [np.flatnonzero(frag == f) for f in frags]
            centers = [ids[frag[ids] == f] for f in frags]
            self._groups.append((members, centers))
            unit[np.isin(frag, frags)] = frags[0]
        _, self._unit = np.unique(unit, return_inverse=True)
        self._nunits = int(self._unit.max()) + 1 if n else 0
        self._unit_size = np.bincount(self._unit, minlength=self._nunits)

        self._center = None if center is None else self._ids(center)
        self._fixed_units = np.zeros(self._nunits, bool)
        if self._center is not None:
            self._fixed_units[self._unit[self._center]] = True
        self._fit = None if fit is None else self._ids(fit)
        self._weights = None
        if self._fit is not None and weights is not None:
            if weights != "mass":
                raise ValueError("weights must be None or 'mass'")
            self._weights = system.atoms["mass"][self._fit].astype(np.float64)
        self._ref = None
        if reference is not None and self._fit is not None:
            ref = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
            self._ref = ref[self._fit] if len(ref) == n else ref
            if len(self._ref) != len(self._fit):
                raise ValueError("reference has the wrong number of positions")
        self.rmsd = None

    def _ids(self, sel) -> np.ndarray:
        if isinstance(sel, str):
            if sel == "all":
                return np.arange(self.system.natoms)
            return self.system.select(sel).ids
        if hasattr(sel, "ids"):
            return sel.ids
        return np.asarray(sel, dtype=np.int64)

    def __call__(self, positions, box=None):
        """Fix one frame; returns new (positions, box) as float64 arrays."""
        pos = np.array(positions, dtype=np.float64).reshape(-1, 3)
        box = np.zeros((3, 3)) if box is None else np.array(box, dtype=np.float64).reshape(3, 3)
        if box.any():
            inv = np.linalg.inv(box)
            if self._tree is not None:
                _make_whole(pos, self._tree[0], self._tree[1], box, inv)
            for members, centers in self._groups:
                frac = np.array([pos[c].mean(0) for c in centers]) @ inv
                shifts = np.column_stack([optimal_shifts(frac[:, d]) for d in range(3)]) @ box
                for m, shift in zip(members, shifts, strict=True):
                    pos[m] += shift
            if self.wrap and self._nunits:
                centers = np.column_stack([
                    np.bincount(self._unit, pos[:, d], self._nunits) for d in range(3)
                ]) / self._unit_size[:, None]  # fmt: skip
                frac = centers @ inv
                if self._center is not None:
                    shift = -np.rint(frac - pos[self._center].mean(0) @ inv)
                    shift[self._fixed_units] = 0
                else:
                    shift = -np.floor(frac)
                pos += (shift @ box)[self._unit]
        if self._fit is not None:
            current = pos[self._fit]
            if self._ref is None:
                self._ref = current.copy()
            rot, trans = kabsch(current, self._ref, self._weights)
            pos = pos @ rot.T + trans
            box = box @ rot.T
            diff = pos[self._fit] - self._ref
            w = np.ones(len(diff)) if self._weights is None else self._weights
            self.rmsd = float(np.sqrt((w * (diff * diff).sum(1)).sum() / w.sum()))
        return pos, box

    def frames(self, traj, chunk: int = 256):
        """Fix every frame of a Trajectory (or a Frames block), yielding Frames blocks."""
        blocks = traj.chunks(chunk) if hasattr(traj, "chunks") else [traj]
        for block in blocks:
            for k in range(len(block)):
                pos, box = self(block.positions[k], block.boxes[k])
                block.positions[k] = pos
                block.boxes[k] = box
            yield block
