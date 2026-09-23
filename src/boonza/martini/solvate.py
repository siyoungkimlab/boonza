"""Martini water and ions around martinized proteins and membranes."""

from __future__ import annotations

import copy

import numpy as np

from .build import Martinized
from .ff import DATA

WATER_BOX = DATA / "water.gro"
# waters closer than this to a protein bead are removed: twice the 0.21 nm
# radius Martini users give gmx solvate
CLASH = 4.2  # Å
SEAM = 3.5  # Å, closer than any two beads in equilibrated water (4.4)


def _water_box() -> tuple[np.ndarray, np.ndarray]:
    """The equilibrated water beads (Å) and the box's edges."""
    lines = WATER_BOX.read_text().splitlines()
    n = int(lines[1])
    xyz = np.array([[float(line[20:28]), float(line[28:36]), float(line[36:44])]
                    for line in lines[2 : 2 + n]]) * 10  # fmt: skip
    return xyz, np.array([float(v) for v in lines[2 + n].split()[:3]]) * 10


def _tile(dims) -> np.ndarray:
    """Water beads filling a box from the origin with edges ``dims``."""
    xyz, edge = _water_box()
    xyz = xyz % edge
    reps = np.ceil(dims / edge).astype(int)
    shifts = np.indices(reps).reshape(3, -1).T * edge
    tiled = (xyz[None] + shifts[:, None]).reshape(-1, 3)
    return tiled[(tiled < dims).all(1)]


def solvate(m: Martinized, padding: float = 10.0, box=None, salt: float = 0.15,
            neutralize: bool = True, clash: float = CLASH, ion_distance: float = 5.0,
            seed: int = 0) -> Martinized:  # fmt: skip
    """``m`` in a box of Martini water (W beads, 4 waters each), with Na+/Cl- ions.

    The water and ions are written for whichever Martini ``m`` is: Martini 3
    gets the ``solvent.itp`` boonza writes, Martini 2 its own upstream
    definitions (``P4`` water, ``Qd``/``Qa`` ions).  The box of water tiled in
    was equilibrated under Martini 3; Martini 2's water is close enough in
    density to start from it, and equilibration settles the difference.

    ``box``: edge lengths (one value or three, Å); by default a cube of the
    proteins' largest extent plus ``padding`` on each side.  The proteins are
    centered, and a box of water equilibrated at 300 K and 1 bar is tiled
    around them; water beads within ``clash`` Å of a protein bead are
    removed, and one of each pair closer than 3.5 Å where the tiles meet the
    box's faces.  Counterions
    neutralize the proteins (``neutralize``), then ion pairs bring the salt
    to ``salt`` mol/L, counted against the waters as boonza's all-atom
    ``neutralize`` counts them.  Ions replace water beads at least
    ``ion_distance`` Å from the proteins, picked in an order fixed by
    ``seed``.
    """
    from ..spatial import min_dist2, pairs_within

    if m.solvent:
        raise ValueError("already solvated")
    protein = np.asarray(m.positions, float)
    if box is None:
        dims = np.full(3, float((protein.max(0) - protein.min(0)).max()) + 2 * padding)
    else:
        dims = np.broadcast_to(np.asarray(box, float), (3,)).copy()
    cell = np.diag(dims)
    lo = protein.mean(0) - 0.5 * dims  # the proteins at the center of a box from the origin
    protein = protein - lo
    water = _tile(dims)
    # waters on a protein bead
    water = water[min_dist2(water, protein, clash, cell=cell) > clash**2]
    # where tiles meet the box faces, a water can sit on another's periodic image:
    # drop one of each such pair (bulk water is never this close)
    i, j, _ = pairs_within(water, SEAM, cell=cell)
    gone = np.zeros(len(water), bool)
    for a, b in zip(i.tolist(), j.tolist(), strict=True):
        if not gone[a] and not gone[b]:
            gone[b] = True
    water = water[~gone]

    charge = sum(count * sum(float(n["charge"]) for n in mol.nodes)
                 for mol, count in zip(m.molecules, m.molecule_copies, strict=True))  # fmt: skip
    net = round(charge)
    if abs(charge - net) > 1e-6:
        raise ValueError(f"the proteins carry a charge of {charge:g}, not a whole number")
    counter = abs(net) if neutralize else 0
    pairs = int(salt / 55.345 * (4 * len(water) - counter)) if salt > 0 else 0
    n_na = pairs + (counter if net < 0 else 0)
    n_cl = pairs + (counter if net > 0 else 0)
    if n_na + n_cl:
        from ..spatial import min_dist2

        far = min_dist2(water, protein, ion_distance, cell=cell) > ion_distance**2
        choices = np.flatnonzero(far)
        if len(choices) < n_na + n_cl:
            raise ValueError(f"only {len(choices)} waters are {ion_distance} Å from the "
                             f"proteins; {n_na + n_cl} ions are needed")  # fmt: skip
        picked = np.random.default_rng(seed).permutation(choices)[: n_na + n_cl]
        na, cl = water[picked[:n_na]], water[picked[n_na:]]
        water = np.delete(water, picked, axis=0)
    else:
        na = cl = np.zeros((0, 3))

    out = copy.deepcopy(m)
    shift = -lo / 10  # nm
    for mol in out.molecules:
        mol.positions = [np.asarray(p) + shift for p in mol.positions]
    out.positions = np.vstack([protein, water, na, cl])
    out.cell = cell
    out.solvent = [("W", len(water)), ("NA", len(na)), ("CL", len(cl))]
    return out
