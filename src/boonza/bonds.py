"""Bond guessing from geometry, following msys ``GuessBondConnectivity``.

Two atoms are bonded when closer than 0.6 x (sum of their radii).  H-H and
pseudo-pseudo pairs are never bonded.  Afterwards a hydrogen (or pseudo
particle) left with several bonds keeps only the shortest one to a heavier
atom.  Without periodicity, only atoms in the same ct are bonded.
"""

from __future__ import annotations

import numpy as np

from .elements import radii
from .spatial import pairs_within

_SEARCH = 4.0  # msys searches up to 4 A before applying the radius rule
_MIN_D2 = 0.00001  # some virtual sites sit on their host atom


def guessed_pairs(pos, anum, ct=None, cell=None, periodic: bool = False) -> np.ndarray:
    """Atom pairs (n, 2) that ``guess_bonds`` bonds, for positions and atomic numbers.

    ``ct``: one group id per atom; without periodicity only atoms of the same
    group are bonded.  Light-atom pruning is included.
    """
    pos = np.asarray(pos, dtype=np.float64)
    anum = np.asarray(anum)
    if len(pos) < 2:
        return np.empty((0, 2), np.int64)
    i, j, keep = _candidates(pos, anum, ct, cell, periodic)
    i, j = i[keep], j[keep]
    keep = _prune_pairs(i, j, pos, anum)
    return np.column_stack([i[keep], j[keep]]).astype(np.int64)


def _candidates(pos, anum, ct, cell, periodic):
    rad = radii(anum)
    if periodic:
        i, j, d2 = pairs_within(pos, 1.2 * float(rad.max()), cell)
        keep = d2 < (0.6 * (rad[i] + rad[j])) ** 2
    else:
        # no pair can bond beyond 0.6 x twice the largest radius present, so
        # searching that far finds the same bonds as msys' 4 A search, faster
        i, j, d2 = pairs_within(pos, min(_SEARCH, 1.2 * float(rad.max())))
        keep = (d2 <= _SEARCH * _SEARCH) & (d2 > _MIN_D2) & (d2 < (0.6 * (rad[i] + rad[j])) ** 2)
        if ct is not None:
            keep &= ct[i] == ct[j]
    ai, aj = anum[i], anum[j]
    keep &= ~(((ai == 1) & (aj == 1)) | ((ai == 0) & (aj == 0)))
    return i, j, keep


def guess_bonds(system, periodic: bool = False, replace: bool = True) -> None:
    if replace and system.nbonds:
        system.delete_bonds(np.arange(system.nbonds))
    if system.natoms < 2:
        return
    pos = system.positions
    anum = system._atoms.column("anum")
    cell = system.cell if (periodic and system.cell.any()) else None
    ct = system._level_key("ct") if (not periodic and system.ncts > 1) else None
    if replace or not system.nbonds:
        system.add_bonds(guessed_pairs(pos, anum, ct, cell, periodic))
        return
    i, j, keep = _candidates(pos, anum, ct, cell, periodic)
    system.add_bonds(np.column_stack([i[keep], j[keep]]))
    _prune_light_atoms(system, pos, anum)


def _prune_pairs(bi, bj, pos, anum) -> np.ndarray:
    """Keep-mask over pairs: a light atom with several bonds keeps its shortest one to a
    heavier atom (the same rule as ``_prune_light_atoms``, on arrays)."""
    keep = np.ones(len(bi), bool)
    if not len(bi):
        return keep
    degree = np.bincount(np.concatenate([bi, bj]), minlength=len(anum))
    light = np.flatnonzero((anum <= 1) & (degree > 1))
    if len(light) == 0:
        return keep
    touching = np.flatnonzero(np.isin(bi, light) | np.isin(bj, light))
    by_atom: dict[int, list[int]] = {}
    for b in touching.tolist():
        by_atom.setdefault(int(bi[b]), []).append(b)
        by_atom.setdefault(int(bj[b]), []).append(b)
    for a in light.tolist():
        bonds = [b for b in by_atom.get(a, []) if keep[b]]
        if len(bonds) <= 1:
            continue
        other = [int(bj[b]) if bi[b] == a else int(bi[b]) for b in bonds]
        cands = [(b, o) for b, o in zip(bonds, other, strict=True) if anum[o] > anum[a]]
        if len(cands) <= 1:
            continue
        d2 = [float(((pos[o] - pos[a]) ** 2).sum()) for _, o in cands]
        best = cands[int(np.argmin(d2))][0]
        for b, _ in cands:
            if b != best:
                keep[b] = False
    return keep


def _prune_light_atoms(system, pos, anum) -> None:
    """A hydrogen or pseudo with several bonds keeps its shortest bond to a heavier atom."""
    bi, bj = system._bonds.column("i"), system._bonds.column("j")
    keep = _prune_pairs(bi, bj, pos, anum)
    if not keep.all():
        system.delete_bonds(np.flatnonzero(~keep))
