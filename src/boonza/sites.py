"""Binding sites: where a ligand spends its time, pooled over runs and copies.

    s = boonza.load("swim/sim_000/solvated.dms")
    found = boonza.sites(s, [boonza.open_trajectory(p, s) for p in runs])
    found[0].occupancy, found[0].runs, found[0].arrivals

Every frame of every copy of every run contributes one point: the ligand's
heavy-atom centroid, with the protein superposed on a common reference so
that runs can be compared at all.  Those points are counted onto a grid, and
a site is a connected region the ligand visits far more often than bulk
solvent would explain -- which gives bulk its own place to go, instead of
forcing every frame into some site.

This answers *where*.  Feed a site's frames to :func:`boonza.poses` for
*how*: that measure needs no superposition, so the alignment used here, and
its error, stay at the coarse level where pockets are many angstroms apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .align import kabsch
from .graph import connected_components
from .pbc import distances, minimum_image
from .symmetry import DEFAULT_LIGAND, _boxed_blocks, _ids


@dataclass
class Site:
    """One place the ligand is found, and the evidence for it."""

    center: np.ndarray  # (3,) in the reference's frame
    points: np.ndarray  # rows of the pooled table
    occupancy: float  # share of all pooled frames
    runs: int  # independent runs that visited it
    copies: int  # ligand copies that visited it
    arrivals: int  # separate visits: a copy leaving and coming back counts twice
    spread: float  # rms distance of its points from the centre (A)

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class SiteSet:
    """The sites of a set of runs, most occupied first."""

    sites: list[Site]
    labels: np.ndarray  # (npoints,) site of each pooled frame, -1 for bulk
    where: np.ndarray  # (npoints, 3): run, copy, frame
    centroids: np.ndarray  # (npoints, 3)
    spacing: float
    enrichment: float

    def __len__(self) -> int:
        return len(self.sites)

    def __getitem__(self, k) -> Site:
        return self.sites[k]

    def __iter__(self):
        return iter(self.sites)

    def frames(self, k: int, run: int | None = None) -> np.ndarray:
        """``(run, copy, frame)`` of site ``k``, optionally of one run only."""
        rows = self.where[self.sites[k].points]
        return rows if run is None else rows[rows[:, 0] == run]

    def summary(self) -> str:
        bulk = int((self.labels < 0).sum())
        out = [f"{len(self.sites)} sites of {len(self.centroids)} frames "
               f"({100 * bulk / max(len(self.centroids), 1):.1f}% in bulk)"]  # fmt: skip
        for k, s in enumerate(self.sites):
            out.append(f"  site {k}: {100 * s.occupancy:.1f}% occupied, {s.runs} runs, "
                       f"{s.arrivals} arrivals, spread {s.spread:.1f} A")  # fmt: skip
        return "\n".join(out)


def ligand_centroids(system, positions=None, reference=None, ligand: str = DEFAULT_LIGAND,
                     align: str = "protein and name CA", periodic: bool = True,
                     ) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """``(centroids (nframes ncopies, 3), (frame, copy) of each)`` in the reference's frame.

    Each copy of the ligand -- one per molecule of the selection -- gives one
    centroid per frame, taken in the copy's own periodic image and then moved
    to the image nearest the protein.  The frame's ``align`` atoms are
    superposed on the reference's, and the same transform is applied to the
    centroid, so points from different runs live in one frame of reference.
    """
    lig = _ids(system, ligand)
    lig = lig[system.atoms["anum"][lig] > 1]
    if not len(lig):
        raise ValueError(f"ligand {ligand!r} selects no heavy atoms")
    fit = _ids(system, align)
    if len(fit) < 3:
        raise ValueError(f"align {align!r} selects {len(fit)} atoms; at least 3 are needed")
    frag = np.asarray(system.fragids)[lig]
    copies = [lig[frag == f] for f in np.unique(frag)]

    ref = system if reference is None else reference
    rfit = _ids(ref, align)
    if len(rfit) != len(fit):
        raise ValueError(f"align {align!r} selects {len(fit)} atoms here and {len(rfit)} in the "
                         "reference; they are paired in order")  # fmt: skip
    target = ref.positions[rfit]

    need = np.union1d(fit, lig)
    at = {a: i for i, a in enumerate(need.tolist())}
    fit_at = np.array([at[a] for a in fit.tolist()])
    copy_at = [np.array([at[a] for a in c.tolist()]) for c in copies]
    blocks, _ = _boxed_blocks(system, positions, need)

    out, where, frame = [], [], 0
    for xyz, boxes in blocks:
        for X, box in zip(xyz, boxes, strict=True):
            box = box if periodic and np.asarray(box).any() else None
            rot, shift = kabsch(X[fit_at], target)
            anchor = X[fit_at].mean(0)
            for c, atoms in enumerate(copy_at):
                p = X[atoms]
                whole = p[0] + minimum_image(p - p[0], box).mean(0)
                whole = anchor + minimum_image((whole - anchor)[None], box)[0]
                out.append(whole @ rot.T + shift)
                where.append((frame, c))
            frame += 1
    if not out:
        raise ValueError("no frames")
    return np.array(out), np.array(where, np.int64)


def _dense_cells(points, spacing: float, threshold: float, volume: float):
    """Grid cells the ligand visits more than ``threshold`` times bulk would explain.

    Returns (cell index of every point, the dense cells, their counts).  The
    expected count of a cell if the ligand wandered uniformly through
    ``volume`` is (points x cell volume / volume), so the threshold is an
    enrichment over bulk rather than a number of frames, and it means the
    same thing whatever the box size, the run length or the copy count.
    """
    lo = points.min(0) - spacing
    dims = np.maximum(np.ceil((points.max(0) + spacing - lo) / spacing), 1).astype(np.int64)
    ijk = np.minimum(((points - lo) / spacing).astype(np.int64), dims - 1)
    flat = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    counts = np.bincount(flat, minlength=int(dims.prod()))
    expected = len(points) * spacing**3 / max(volume, 1e-9)
    dense = np.flatnonzero(counts >= max(threshold * expected, 2.0))
    return flat, dense, counts, dims


def _join_neighbours(dense, dims):
    """Group dense cells that touch: 26-connectivity, by connected components."""
    if not len(dense):
        return np.zeros(0, np.int64), 0
    rank = np.full(int(dims.prod()) + 1, -1, np.int64)
    rank[dense] = np.arange(len(dense))
    k = dense[:, None]
    z = k % dims[2]
    y = (k // dims[2]) % dims[1]
    x = k // (dims[2] * dims[1])
    bi, bj = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue  # each pair once
                nx, ny, nz = x + dx, y + dy, z + dz
                ok = ((nx >= 0) & (nx < dims[0]) & (ny >= 0) & (ny < dims[1])
                      & (nz >= 0) & (nz < dims[2]))  # fmt: skip
                other = rank[np.where(ok, (nx * dims[1] + ny) * dims[2] + nz, -1).ravel()]
                here = np.arange(len(dense))
                keep = other >= 0
                bi.append(here[keep])
                bj.append(other[keep])
    bi = np.concatenate(bi) if bi else np.zeros(0, np.int64)
    bj = np.concatenate(bj) if bj else np.zeros(0, np.int64)
    return connected_components(len(dense), bi, bj)


def _visits(rows) -> int:
    """Separate arrivals: a copy leaving a site and coming back counts twice."""
    if not len(rows):
        return 0
    order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))
    r = rows[order]
    same = (r[1:, 0] == r[:-1, 0]) & (r[1:, 1] == r[:-1, 1]) & (r[1:, 2] == r[:-1, 2] + 1)
    return int(1 + (~same).sum())


def sites(system, runs=None, reference=None, ligand: str = DEFAULT_LIGAND,
          align: str = "protein and name CA", spacing: float = 1.0,
          enrichment: float = 20.0, min_occupancy: float = 0.005,
          periodic: bool = True) -> SiteSet:  # fmt: skip
    """Where the ligand is found across ``runs``, most occupied first.

    ``runs`` is one trajectory (or array of frames) or a list of them; each is
    treated as independent evidence, and a site visited by several runs is a
    claim several simulations agree on.  A site is a connected group of grid
    cells the ligand visits at least ``enrichment`` times more often than
    bulk solvent would explain; everything else is bulk, and is labelled -1
    rather than forced into a site.  Sites below ``min_occupancy`` of the
    pooled frames are left out.
    """
    if runs is None or not isinstance(runs, (list, tuple)):
        runs = [runs]
    points, where, volume = [], [], 0.0
    for r, run in enumerate(runs):
        xyz, rows = ligand_centroids(system, run, reference, ligand, align, periodic)
        points.append(xyz)
        where.append(np.column_stack([np.full(len(rows), r), rows[:, 1], rows[:, 0]]))
        cell = np.asarray(system.cell, float)
        volume = max(volume, abs(float(np.linalg.det(cell))))
    points = np.concatenate(points)
    where = np.concatenate(where)
    if volume <= 0:
        hull = points.max(0) - points.min(0)
        volume = float(np.prod(np.maximum(hull, spacing)))

    flat, dense, counts, dims = _dense_cells(points, spacing, enrichment, volume)
    group, ngroups = _join_neighbours(dense, dims)
    of_cell = np.full(int(dims.prod()) + 1, -1, np.int64)
    of_cell[dense] = group
    labels = of_cell[flat]

    found = []
    smallest = max(2, round(min_occupancy * len(points)))
    for g in range(ngroups):
        members = np.flatnonzero(labels == g)
        if len(members) < smallest:
            continue
        rows, xyz = where[members], points[members]
        centre = xyz.mean(0)
        found.append(Site(center=centre, points=members,
                          occupancy=len(members) / len(points),
                          runs=len(np.unique(rows[:, 0])), copies=len(np.unique(rows[:, 1])),
                          arrivals=_visits(rows),
                          spread=float(np.sqrt(((xyz - centre) ** 2).sum(1).mean()))))  # fmt: skip
    found.sort(key=lambda s: (-len(s.points), tuple(s.center)))
    out = np.full(len(points), -1)
    for k, s in enumerate(found):
        out[s.points] = k
    return SiteSet(sites=found, labels=out, where=where, centroids=points,
                   spacing=float(spacing), enrichment=float(enrichment))  # fmt: skip


def site_pocket(system, runs, found: SiteSet, k: int, protein: str = "protein and name CA",
                ligand: str = DEFAULT_LIGAND, cutoff: float = 5.0, share: float = 0.5,
                periodic: bool = True) -> np.ndarray:  # fmt: skip
    """The atoms a site is made of: those the ligand touches in ``share`` of its frames.

    A pocket from one frame is one frame's opinion.  This counts over every
    frame assigned to the site, across runs and copies, so an atom earns its
    place by being there for the ligand rather than by happening to be close
    when the reference was taken.  Hand the result to :func:`boonza.poses` as
    ``pocket=`` and the pose level stops depending on a reference at all.
    """
    if not isinstance(runs, (list, tuple)):
        runs = [runs]
    lig = _ids(system, ligand)
    lig = lig[system.atoms["anum"][lig] > 1]
    frag = np.asarray(system.fragids)[lig]
    copies = [lig[frag == f] for f in np.unique(frag)]
    prot = _ids(system, protein)
    own = np.isin(prot, lig)

    rows = found.frames(k)
    hits, seen = np.zeros(len(prot)), 0
    for r, run in enumerate(runs):
        mine = rows[rows[:, 0] == r]
        if not len(mine):
            continue
        wanted = np.unique(mine[:, 2])
        need = np.union1d(prot, lig)
        pl = np.searchsorted(need, prot)
        blocks, _ = _boxed_blocks(system, run[wanted], need)
        at = {int(f): i for i, f in enumerate(wanted.tolist())}
        by_frame: dict[int, list[int]] = {}
        for frame, copy in mine[:, [2, 1]].tolist():
            by_frame.setdefault(at[frame], []).append(copy)
        i = 0
        for xyz, boxes in blocks:
            for X, box in zip(xyz, boxes, strict=True):
                for c in by_frame.get(i, ()):
                    atoms = np.searchsorted(need, copies[c])
                    near = distances(X[pl], X[atoms], box if periodic else None).min(1) <= cutoff
                    hits += near & ~own
                    seen += 1
                i += 1
    if not seen:
        raise ValueError(f"site {k} has no frames")
    return prot[hits / seen >= share]
