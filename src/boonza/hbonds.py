"""Hydrogen bonds, per frame, with periodic boundaries (orthorhombic or triclinic).

    hb = boonza.hbonds(system)                         # MDAnalysis-style criteria
    hb = boonza.hbonds(system, frames, between=["protein", "water"])
    hb.donor, hb.hydrogen, hb.acceptor, hb.distance, hb.angle, hb.frame
    hb.counts()                                        # {(donor, hydrogen, acceptor): frames}
    boonza.baker_hubbard(system, frames)               # mdtraj criteria
    boonza.wernet_nilsson(system, frames)

``hbonds`` follows MDAnalysis HydrogenBondAnalysis: donor-hydrogen pairs
come from bonds, and a bond needs a donor-acceptor distance in
(1, ``d_a_cutoff``] Å and a D-H...A angle above ``angle_cutoff`` degrees.
``baker_hubbard`` and ``wernet_nilsson`` follow mdtraj (N-H and O-H donors,
N and O acceptors).  Distances and angles use the minimum image.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import pbc


@dataclass
class HBonds:
    """Hydrogen bonds found over frames; one entry per bond per frame."""

    frame: np.ndarray
    donor: np.ndarray
    hydrogen: np.ndarray
    acceptor: np.ndarray
    distance: np.ndarray  # donor-acceptor, Å
    angle: np.ndarray  # donor-hydrogen-acceptor, degrees
    nframes: int

    def __len__(self) -> int:
        return len(self.frame)

    def per_frame(self) -> np.ndarray:
        """Number of hydrogen bonds in each frame."""
        return np.bincount(self.frame, minlength=self.nframes)

    def counts(self) -> dict:
        """{(donor, hydrogen, acceptor): number of frames with that bond}."""
        keys = zip(self.donor.tolist(), self.hydrogen.tolist(), self.acceptor.tolist(), strict=True)
        return dict(Counter(keys).most_common())

    def frequency(self) -> dict:
        """{(donor, hydrogen, acceptor): fraction of frames with that bond}."""
        return {k: v / self.nframes for k, v in self.counts().items()}


def _ids(system, sel, default):
    if sel is None:
        return default
    if isinstance(sel, str):
        return system.select(sel).ids
    return np.asarray(system._ids("atom", sel), dtype=np.int64)


def _frames(system, positions, box):
    """(nframes, natoms, 3) positions and a per-frame list of 3x3 boxes (or None)."""
    boxes = None
    if positions is None:
        xyz = system.positions[None]
        box = system.cell if box is None else box
    else:
        boxes = getattr(positions, "boxes", None)
        if boxes is None:
            boxes = getattr(positions, "box", None)
        xyz = np.asarray(getattr(positions, "positions", positions), dtype=np.float64)
        if xyz.ndim == 2:
            xyz = xyz[None]
    if xyz.shape[1:] != (system.natoms, 3):
        raise ValueError(f"positions must be (nframes, {system.natoms}, 3)")
    if boxes is None:
        boxes = box
    if boxes is None:
        per = [None] * len(xyz)
    else:
        b = np.asarray(boxes, dtype=np.float64)
        b = np.broadcast_to(b.reshape(-1, 3, 3), (len(xyz), 3, 3)) if b.size in (9,) or \
            b.ndim == 3 else b  # fmt: skip
        per = [None if not np.any(x) else x for x in np.asarray(b).reshape(len(xyz), 3, 3)]
    return xyz, per


def donor_hydrogen_pairs(system, hydrogens=None, donors=None):
    """(donor, hydrogen) atom pairs from bonds: hydrogens bonded to a donor heavy atom.

    Defaults: hydrogens are atoms with atomic number 1, donors are N and O.
    """
    anum = system.atoms["anum"]
    h = _ids(system, hydrogens, np.flatnonzero(anum == 1))
    d = _ids(system, donors, np.flatnonzero((anum == 7) | (anum == 8)))
    is_h = np.zeros(system.natoms, bool)
    is_h[h] = True
    is_d = np.zeros(system.natoms, bool)
    is_d[d] = True
    bi, bj = system.bonds["i"], system.bonds["j"]
    pairs = np.concatenate([np.column_stack([bi, bj]), np.column_stack([bj, bi])])
    keep = is_d[pairs[:, 0]] & is_h[pairs[:, 1]] & (anum[pairs[:, 0]] != 1)
    pairs = pairs[keep]
    return pairs[np.lexsort((pairs[:, 0], pairs[:, 1]))]


def _angles(a, b, c, box):
    """Angle a-b-c at b in radians, minimum image."""
    u = pbc.minimum_image(a - b, box)
    v = pbc.minimum_image(c - b, box)
    cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _pair_filter(system, between, donors, acceptors):
    """Mask of donor/acceptor pairs lying across the ``between`` groups (either direction)."""
    if between is None:
        return None
    groups = [between] if isinstance(between[0], str) or np.ndim(between[0]) == 1 and \
        not isinstance(between[0], list | tuple) else between  # fmt: skip
    mask = np.zeros(len(donors), bool)
    for g1, g2 in groups:
        a = np.zeros(system.natoms, bool)
        b = np.zeros(system.natoms, bool)
        a[_ids(system, g1, None)] = True
        b[_ids(system, g2, None)] = True
        mask |= (a[donors] & b[acceptors]) | (a[acceptors] & b[donors])
    return mask


def hbonds(system, positions=None, box=None, donors=None, hydrogens=None, acceptors=None,
           d_a_cutoff: float = 3.0, angle_cutoff: float = 150.0,
           between=None) -> HBonds:  # fmt: skip
    """Hydrogen bonds by MDAnalysis HydrogenBondAnalysis criteria.

    ``positions``: one frame (natoms, 3), several (nframes, natoms, 3), or a
    Frames/Trajectory block; the system's own positions and cell by default.
    ``box``: 3x3 cell (or one per frame) when ``positions`` has none.
    ``donors``/``hydrogens``/``acceptors``: selections or indices; defaults are
    N and O donors bonded to hydrogens and N and O acceptors.  ``between``:
    a pair of selections, or a list of pairs, to keep only bonds across them.
    """
    xyz, boxes = _frames(system, positions, box)
    dh = donor_hydrogen_pairs(system, hydrogens, donors)
    anum = system.atoms["anum"]
    acc = _ids(system, acceptors, np.flatnonzero((anum == 7) | (anum == 8)))
    out = {k: [] for k in ("frame", "donor", "hydrogen", "acceptor", "distance", "angle")}
    for f, (frame, cell) in enumerate(zip(xyz, boxes, strict=True)):
        if not len(dh) or not len(acc):
            break
        i, j, d = pbc.capped_distances(frame[dh[:, 0]], frame[acc], d_a_cutoff, cell)
        keep = d > 1.0
        i, j, d = i[keep], j[keep], d[keep]
        D, H, A = dh[i, 0], dh[i, 1], acc[j]
        mask = _pair_filter(system, between, D, A)
        if mask is not None:
            D, H, A, d = D[mask], H[mask], A[mask], d[mask]
        ang = np.degrees(_angles(frame[D], frame[H], frame[A], cell))
        ok = ang > angle_cutoff
        for key, val in (("donor", D), ("hydrogen", H), ("acceptor", A), ("distance", d),
                         ("angle", ang)):  # fmt: skip
            out[key].append(val[ok])
        out["frame"].append(np.full(int(ok.sum()), f, np.int64))
    cat = {}
    for k, v in out.items():
        kind = np.float64 if k in ("distance", "angle") else np.int64
        cat[k] = np.concatenate(v) if v else np.empty(0, kind)
    return HBonds(nframes=len(xyz), **cat)


def _triplets(system, exclude_water):
    """mdtraj _get_bond_triplets: (N-H or O-H donor pairs) x (N or O acceptors), no self pairs."""
    anum = system.atoms["anum"]
    ok = np.ones(system.natoms, bool)
    if exclude_water:
        ok[system.select("water").ids] = False
    bi, bj = system.bonds["i"], system.bonds["j"]
    pairs = []
    for heavy in (7, 8):  # N-H donors first, then O-H, each in bond order
        for a, b in zip(bi.tolist(), bj.tolist(), strict=True):
            if {anum[a], anum[b]} == {heavy, 1} and ok[a] and ok[b]:
                pairs.append((a, b) if anum[a] == heavy else (b, a))
    acc = np.flatnonzero(((anum == 7) | (anum == 8)) & ok)
    if not pairs or not len(acc):
        return np.zeros((0, 3), np.int64)
    dh = np.array(pairs, np.int64)
    trip = np.column_stack([np.repeat(dh, len(acc), axis=0), np.tile(acc, len(dh))])
    return trip[trip[:, 0] != trip[:, 2]]


def _bounded(system, positions, box, trip, dist_pair, cutoff, freq):
    """Candidate triplets whose ``dist_pair`` distance is under ``cutoff`` often enough."""
    xyz, boxes = _frames(system, positions, box)
    a, b = trip[:, dist_pair[0]], trip[:, dist_pair[1]]
    # screen with a cell-list search on the atoms involved, then exact per-frame distances
    ua, ub = np.unique(a), np.unique(b)
    present = np.zeros(len(trip), np.int64)
    dists = []
    for frame, cell in zip(xyz, boxes, strict=True):
        i, j, d = pbc.capped_distances(frame[ua], frame[ub], cutoff, cell)
        near = dict(zip(zip(ua[i].tolist(), ub[j].tolist(), strict=True), d.tolist(), strict=True))
        d_all = np.array([near.get((x, y), np.inf) for x, y in zip(a.tolist(), b.tolist(),
                                                                     strict=True)])  # fmt: skip
        present += d_all < cutoff
        dists.append(d_all)
    mask = present / len(xyz) > freq
    return mask, xyz, boxes, np.array(dists)[:, mask]


def baker_hubbard(system, positions=None, box=None, freq: float = 0.1, exclude_water=True,
                  distance_cutoff: float = 2.5,
                  angle_cutoff: float = 120.0) -> np.ndarray:  # fmt: skip
    """(donor, hydrogen, acceptor) triplets bonded in more than ``freq`` of frames (mdtraj).

    Criterion: hydrogen-acceptor distance under ``distance_cutoff`` Å and
    donor-hydrogen-acceptor angle above ``angle_cutoff`` degrees.
    """
    trip = _triplets(system, exclude_water)
    mask, xyz, boxes, dist = _bounded(system, positions, box, trip, (1, 2), distance_cutoff, freq)
    trip = trip[mask]
    if not len(trip):
        return trip
    present = np.zeros(len(trip), np.int64)
    for f, (frame, cell) in enumerate(zip(xyz, boxes, strict=True)):
        ang = np.degrees(_angles(frame[trip[:, 0]], frame[trip[:, 1]], frame[trip[:, 2]], cell))
        present += (dist[f] < distance_cutoff) & (ang > angle_cutoff)
    return trip[present / len(xyz) > freq]


def wernet_nilsson(system, positions=None, box=None, exclude_water=True) -> list[np.ndarray]:
    """Per frame, (donor, hydrogen, acceptor) triplets by the Wernet-Nilsson cone (mdtraj).

    Criterion: donor-acceptor distance under 3.3 Å - 0.00044 Å/deg^2 * theta^2,
    with theta the hydrogen-donor-acceptor angle under 45 degrees.
    """
    trip = _triplets(system, exclude_water)
    mask, xyz, boxes, dist = _bounded(system, positions, box, trip, (0, 2), 3.3, 0.0)
    trip = trip[mask]
    out = []
    for f, (frame, cell) in enumerate(zip(xyz, boxes, strict=True)):
        if not len(trip):
            out.append(trip)
            continue
        ang = np.degrees(_angles(frame[trip[:, 1]], frame[trip[:, 0]], frame[trip[:, 2]], cell))
        ok = (dist[f] < 3.3 - 0.00044 * ang**2) & (ang < 45.0)
        out.append(trip[ok])
    return out
