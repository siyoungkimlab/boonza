"""What a ligand touches, as a number per residue: comparable between molecules.

    a = boonza.interaction_fingerprints(s, frames, ligand="fragid 3 and noh")
    b = boonza.interaction_fingerprints(s, frames, ligand="fragid 7 and noh")
    boonza.similarity(a.mean(), b.mean())   # do these two bind the same way?

:func:`boonza.poses` compares a molecule with itself, frame by frame, so it
cannot say whether *two different* molecules sit the same way: there is no
atom of one that answers to an atom of the other.  A fingerprint describes a
frame by what the ligand is near rather than where its atoms are, so it has
the same length and the same meaning for every ligand, whatever its size or
chemistry.

Each residue contributes how close the ligand comes to it, softened:

    s = 1 / (1 + exp((d - center) / width))

so a residue at the ``center`` distance counts a half, nearer counts more,
and further counts less.  A hard cutoff would make a residue flicker in and
out as the ligand breathes; this does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pbc import capped_distances
from .symmetry import DEFAULT_LIGAND, _boxed_blocks, _ids


@dataclass
class Interactions:
    """A fingerprint per frame and copy, and which residues its columns are."""

    values: np.ndarray  # (rows, nresidues), 0 to 1
    residues: np.ndarray  # residue index in the system of every column
    where: np.ndarray  # (rows, 2): frame, copy
    center: float
    width: float

    def __len__(self) -> int:
        return len(self.values)

    def mean(self) -> np.ndarray:
        """The average fingerprint: what this ligand touches, over the frames given."""
        return self.values.mean(0) if len(self.values) else np.zeros(len(self.residues))

    def touched(self, system, share: float = 0.5) -> list[str]:
        """The residues held above ``share`` on average, as ``TYR34`` and so on."""
        keep = np.flatnonzero(self.mean() >= share)
        return [f"{system.residues['name'][r]}{system.residues['resid'][r]}"
                for r in self.residues[keep]]  # fmt: skip


def similarity(a, b) -> float:
    """How alike two fingerprints are: 1 the same residues to the same degree, 0 none.

    The cosine of the two, which asks about the pattern rather than how
    deeply either sits -- a small fragment in a pocket and a large one
    reaching further into it are alike here if they touch the same residues.
    """
    a, b = np.asarray(a, float).reshape(-1), np.asarray(b, float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"fingerprints of {a.shape} and {b.shape}: they must share residues")
    size = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / size) if size else 0.0


def interaction_fingerprints(system, positions=None, ligand: str = DEFAULT_LIGAND,
                             protein: str = "protein and noh", center: float = 4.0,
                             width: float = 1.0, residues=None, periodic: bool = True,
                             ) -> Interactions:  # fmt: skip
    """How near the ligand comes to each residue, per frame, softened to 0-1.

    Every copy of the ligand -- one per molecule of the selection -- gives one
    fingerprint per frame.  ``residues`` fixes the columns, which is what lets
    fingerprints from different ligands, or different systems with the same
    protein, be compared: pass the ``residues`` of an earlier result.
    """
    lig = _ids(system, ligand)
    lig = lig[system.atoms["anum"][lig] > 1]
    if not len(lig):
        raise ValueError(f"ligand {ligand!r} selects no heavy atoms")
    prot = np.setdiff1d(_ids(system, protein), lig)
    if not len(prot):
        raise ValueError(f"protein {protein!r} selects no atoms outside the ligand")
    of_residue = system.atoms["residue"]
    columns = np.unique(of_residue[prot]) if residues is None else np.asarray(residues)
    rank = {int(r): i for i, r in enumerate(columns.tolist())}
    where_column = np.array([rank.get(int(r), -1) for r in of_residue[prot]])
    prot = prot[where_column >= 0]
    where_column = where_column[where_column >= 0]

    frag = np.asarray(system.fragids)[lig]
    copies = [lig[frag == f] for f in np.unique(frag)]
    need = np.union1d(prot, lig)
    at = np.searchsorted(need, prot)
    copy_at = [np.searchsorted(need, c) for c in copies]
    far = center + 6.0 * width  # beyond this the sigmoid is under 0.3%

    rows, place, frame = [], [], 0
    for xyz, boxes in _boxed_blocks(system, positions, need)[0]:
        for X, box in zip(xyz, boxes, strict=True):
            box = box if periodic and np.asarray(box).any() else None
            for c, atoms in enumerate(copy_at):
                near = np.full(len(columns), np.inf)
                i, _j, d = capped_distances(X[at], X[atoms], far, box)
                if len(i):
                    np.minimum.at(near, where_column[i], d)
                rows.append(1.0 / (1.0 + np.exp((near - center) / width)))
                place.append((frame, c))
            frame += 1
    if not rows:
        raise ValueError("no frames")
    return Interactions(values=np.array(rows), residues=columns, where=np.array(place, np.int64),
                        center=float(center), width=float(width))  # fmt: skip


def similarity_matrix(values) -> np.ndarray:
    """Every fingerprint against every other, as cosines."""
    v = np.asarray(values, float)
    norm = np.linalg.norm(v, axis=1)
    norm[norm == 0] = 1.0
    out = (v / norm[:, None]) @ (v / norm[:, None]).T
    return np.clip(out, -1.0, 1.0)


def site_interactions(system, runs, found, site: int, ligand: str = DEFAULT_LIGAND,
                      protein: str = "protein and noh", center: float = 4.0,
                      width: float = 1.0, periodic: bool = True) -> Interactions:  # fmt: skip
    """One fingerprint per ligand that visits ``site``: what each of them touches there.

    Rows are ``(run, copy)`` rather than frames, each the average over that
    copy's frames in the site.  The columns are the same residues for every
    row, so two rows can be compared however unlike the two molecules are --
    which is the question a pose cannot answer.
    """
    from .sites import _pairs

    pairs = _pairs(system, runs)
    rows, place, columns = [], [], None
    for r, (own, run) in enumerate(pairs):
        mine = found.frames(site)
        mine = mine[mine[:, 0] == r]
        if not len(mine):
            continue
        wanted = np.unique(mine[:, 2])
        seen = interaction_fingerprints(own, run[wanted], ligand=ligand, protein=protein,
                                        center=center, width=width, residues=columns,
                                        periodic=periodic)  # fmt: skip
        columns = seen.residues if columns is None else columns
        at = {int(f): i for i, f in enumerate(wanted.tolist())}
        for copy in np.unique(mine[:, 1]).tolist():
            frames = {at[int(f)] for f in mine[mine[:, 1] == copy][:, 2]}
            keep = np.array([i for i, (f, c) in enumerate(seen.where.tolist())
                             if c == copy and f in frames])  # fmt: skip
            if len(keep):
                rows.append(seen.values[keep].mean(0))
                place.append((r, copy))
    if not rows:
        raise ValueError(f"site {site} has no frames")
    return Interactions(values=np.array(rows), residues=columns, where=np.array(place, np.int64),
                        center=float(center), width=float(width))  # fmt: skip
