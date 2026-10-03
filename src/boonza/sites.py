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

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .align import kabsch
from .graph import connected_components
from .pbc import distances, minimum_image
from .symmetry import DEFAULT_LIGAND, _boxed_blocks, _ids, molecules_of


@dataclass
class Site:
    """One place the ligand is found, and the evidence for it."""

    center: np.ndarray  # (3,) in the reference's frame
    points: np.ndarray  # rows of the pooled table
    occupancy: float  # share of the frames in which anything is here
    copy_frames: float  # share of the pooled copy-frames, which the copy count dilutes
    runs: int  # independent runs that visited it
    copies: int  # ligand copies that visited it
    arrivals: int  # separate visits: a copy leaving and coming back counts twice
    spread: float  # rms distance of its points from the centre (A)
    volume: float = 0.0  # A^3 of the pocket it sits in, at the same enrichment
    burial: float = 0.0  # how enclosed that pocket is: 1 is shut in, 0 is open water
    cells: np.ndarray | None = None  # the pocket's cells of the occupancy grid
    grid_dims: np.ndarray | None = None  # that grid's shape, origin and spacing, so the
    grid_origin: np.ndarray | None = None  # cells can be turned back into coordinates
    grid_spacing: float = 1.0

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class Density:
    """How often the ligand's centroid was in each cell of a grid, and what bulk would give.

    ``enrichment`` is the map worth looking at: one means as often as
    wandering through the box uniformly would explain, and a site is where it
    is large.  It is a check on the sites that does not come from clustering
    at all.
    """

    origin: np.ndarray  # (3,) the low corner, A
    spacing: float  # A
    counts: np.ndarray  # (nx, ny, nz) frames whose centroid fell in each cell
    expected: float  # what bulk would put in one cell

    @property
    def enrichment(self) -> np.ndarray:
        return self.counts / max(self.expected, 1e-12)

    def write_dx(self, path, values=None) -> None:
        """Write an OpenDX map, as ChimeraX, VMD and PyMOL read."""
        v = self.enrichment if values is None else np.asarray(values)
        nx, ny, nz = v.shape
        # a count belongs to a cell, a DX value belongs to a point: sample at cell centres
        centre = np.asarray(self.origin, float) + 0.5 * self.spacing
        with open(path, "w", encoding="utf-8") as out:
            out.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
            out.write("origin {:g} {:g} {:g}\n".format(*centre))
            for axis in range(3):
                d = [0.0, 0.0, 0.0]
                d[axis] = self.spacing
                out.write("delta {:g} {:g} {:g}\n".format(*d))
            out.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")
            out.write(f"object 3 class array type double rank 0 items {v.size} data follows\n")
            flat = v.reshape(-1)  # z fastest, as DX wants
            for i in range(0, flat.size, 3):
                out.write(" ".join(f"{x:.4g}" for x in flat[i:i + 3]) + "\n")  # fmt: skip
            out.write('object "density" class field\n')


@dataclass
class SiteSet:
    """The sites of a set of runs, most occupied first."""

    sites: list[Site]
    labels: np.ndarray  # (npoints,) site of each pooled frame, -1 for bulk
    where: np.ndarray  # (npoints, 3): run, copy, frame
    centroids: np.ndarray  # (npoints, 3)
    spacing: float
    enrichment: float
    systems: list = field(default_factory=list)  # the system each run was read with
    volume: float = 0.0  # mean box volume of the frames, A^3, not the stored cell
    density: Density | None = None
    occupancy: Occupancy | None = None  # where the atoms go, not only the centres
    drift: np.ndarray | None = None  # per frame: how far the fit atoms are from the reference
    drift_runs: np.ndarray | None = None  # which run each of those frames came from

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


#: How to order the sites found.  Volume is not among them: it moves with the
#: enrichment threshold, and a wide shallow groove would outrank a tight deep one.
_RANKS = {
    # the pocket itself: how much room there is and how enclosed it is, which is
    # what a ligand needs.  It is the default because dwell is not it: on the one
    # protein where the answer is known (a crystal ligand in a 4qoc/3lnz pair),
    # the true pocket came second of two by dwell under Martini and twelfth of 25
    # under SIRAH, and first and third by this
    "pocket": lambda s: (-s.volume * s.burial, -s.occupancy, tuple(s.center)),
    # how much of the run something was there -- dwell, which one sticky copy can carry
    "occupied": lambda s: (-s.occupancy, -len(s.points), tuple(s.center)),
    # how many copies chose it, which for a library of probes is the firmer claim:
    # unlike molecules agreeing beats one molecule staying
    "agreement": lambda s: (-s.copies, -s.occupancy, tuple(s.center)),
    # how enclosed the pocket is, for picking somewhere to put a ligand
    "burial": lambda s: (-s.burial, -s.occupancy, tuple(s.center)),
}


def _remeasure(site, where, points, all_frames: int) -> None:
    """Measure a site again from the rows it now holds, after a merge."""
    rows, xyz = where[site.points], points[site.points]
    site.center = xyz.mean(0)
    site.occupancy = len(np.unique(rows[:, [0, 2]], axis=0)) / max(all_frames, 1)
    site.copy_frames = len(site.points) / max(len(points), 1)
    site.runs = len(np.unique(rows[:, 0]))
    site.copies = len(np.unique(rows[:, 1]))
    site.arrivals = _visits(rows)
    site.spread = float(np.sqrt(((xyz - site.center) ** 2).sum(1).mean()))


def _no_pockets():
    """Empty (cells, labels, centres, volumes, burials)."""
    return (np.empty(0, np.int64), np.empty(0, np.int64), np.empty((0, 3)),
            np.empty(0), np.empty(0))  # fmt: skip


#: What half of sigma is multiplied by for the radius where the pair potential
#: is deepest rather than where it crosses zero: r_min = 2**(1/6) sigma.
RMIN = 2.0 ** (1.0 / 6.0)
#: How enclosed a pocket has to be: the share of 26 directions out of its cells
#: that meet protein, 0 being open water and 1 shut in.
BURIED = 0.4
#: The smallest pocket worth reporting, in cubic angstroms.  Twenty is twenty
#: cells of a 1 A grid, where a pocket a ligand sits in runs to hundreds.
MIN_VOLUME = 20.0


def particle_radii(system, ids, rule: str = "sigma") -> np.ndarray:
    """How wide each of ``ids`` is, from the force field it carries.

    A particle's own size is the sigma of its nonbonded term with itself: half of
    it with ``rule="sigma"`` (where the pair potential crosses zero), half of
    ``2**(1/6) sigma`` with ``rule="rmin"`` (where it is deepest, which is what a
    contact distance means).  Martini writes nonbonded terms per pair of types
    rather than per type -- NBFIX, in Amber's language -- so a bead's own size is
    the pair it makes with itself, and no single number describes what it does
    against every other bead.  For a density map that is enough: the question is
    how much room the thing takes, not what it would feel.

    Without a force field, or for a particle whose terms carry no size (a polar
    hydrogen has no Lennard-Jones at all), the element's radius stands in.
    """
    from .elements import radii as element_radii

    if rule not in ("sigma", "rmin"):
        raise ValueError(f"radius rule {rule!r}: 'sigma' (sigma/2) or 'rmin' (2^(1/6) sigma/2)")
    ids = np.asarray(ids, np.int64)
    anum = np.asarray(system.atoms["anum"])[ids]
    scale = 0.5 * (RMIN if rule == "rmin" else 1.0)
    if "nonbonded" not in system.table_names:
        return np.maximum(element_radii(anum), 0.3)
    nb = system.table("nonbonded")
    alone = {int(a): float(p["sigma"]) for (a, b), p in nb.overrides.items() if a == b}
    pid = np.asarray(nb.param_ids)
    own = np.asarray([float(x) for x in nb.values("sigma")], float)
    out = np.array([alone.get(int(pid[a]), own[a]) for a in ids]) * scale
    poor = ~np.isfinite(out) | (out <= 0.1)  # no Lennard-Jones of its own
    if poor.any():
        out[poor] = np.maximum(element_radii(anum[poor]), 0.3)
    return out


class Occupancy:
    """Where the ligand's atoms go, counted onto a grid as the frames come.

    The centroid map says where a molecule sits; this says what space it
    reaches, which is the shape of a pocket.  A deep one keeps its volume as
    the threshold rises, because the atoms come back to the same cells; a
    shallow one is wide and low and thins out, because molecules brush it from
    every direction without settling.

    The grid is fixed from the reference, so the counts can be accumulated in
    one pass rather than every position kept.
    """

    def __init__(self, reference, spacing: float = 1.0, margin: float = 12.0):
        lo = np.asarray(reference, float).min(0) - margin
        hi = np.asarray(reference, float).max(0) + margin
        self.origin = lo
        self.spacing = float(spacing)
        self.dims = np.maximum(np.ceil((hi - lo) / spacing), 1).astype(np.int64)
        self.counts = np.zeros(int(self.dims.prod()), np.int64)
        self.total = 0
        self._stencils: dict[float, np.ndarray] = {}

    def add(self, xyz, radii=None) -> None:
        """Count atom positions, already in the reference's frame.

        With ``radii`` each atom counts for every cell within its own radius
        rather than for the one its centre fell in: a bead is several atoms
        across, and a map of centres at 1 A is a map of noise at the model's own
        resolution -- the same region comes out as dust that no pocket survives.
        Bulk is counted the same way, so the enrichment is still a ratio.
        """
        xyz = np.asarray(xyz, float)
        ijk = np.floor((xyz - self.origin) / self.spacing).astype(np.int64)
        if radii is None:
            inside = np.all((ijk >= 0) & (ijk < self.dims), axis=1)
            self._deposit(ijk[inside])
            return
        radii = np.asarray(radii, float)
        for r in np.unique(np.round(radii, 2)):
            take = np.round(radii, 2) == r
            spread = ijk[take][:, None, :] + self._stencil(float(r))[None, :, :]
            spread = spread.reshape(-1, 3)
            inside = np.all((spread >= 0) & (spread < self.dims), axis=1)
            self._deposit(spread[inside])

    def _deposit(self, ijk) -> None:
        if not len(ijk):
            return
        flat = (ijk[:, 0] * self.dims[1] + ijk[:, 1]) * self.dims[2] + ijk[:, 2]
        self.counts += np.bincount(flat, minlength=self.counts.size)
        self.total += len(ijk)

    def _stencil(self, radius: float) -> np.ndarray:
        """The cell offsets within ``radius`` of a cell, as a sphere of cells."""
        if radius in self._stencils:
            return self._stencils[radius]
        steps = int(radius / self.spacing)
        grid = np.arange(-steps, steps + 1)
        off = np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T
        self._stencils[radius] = off[np.linalg.norm(off * self.spacing, axis=1) <= radius]
        return self._stencils[radius]

    def density(self, volume: float) -> Density:
        """The counts as a :class:`Density`, with bulk taken over ``volume`` A^3."""
        return Density(origin=self.origin, spacing=self.spacing,
                       counts=self.counts.reshape(self.dims),
                       expected=self.total * self.spacing**3 / max(volume, 1e-9))  # fmt: skip

    def cell_centres(self) -> np.ndarray:  # noqa: D401
        """The centre of every cell, in the reference's frame."""
        ijk = np.array(np.unravel_index(np.arange(self.counts.size), tuple(self.dims))).T
        return self.origin + (ijk + 0.5) * self.spacing

    def burial(self, cells, protein, reach: float = 10.0, touch: float = 2.6) -> np.ndarray:
        """Of 26 directions out of each cell, the share that meet ``protein``.

        A pocket is enclosed; a dent on a convex surface is not, and bulk is
        not at all.  Counting directions rather than neighbours within a radius
        keeps the number comparable between an all-atom protein and a
        coarse-grained one, whose beads are fewer and larger.
        """
        from .spatial import min_dist2

        xyz = self.cell_centres()[cells]
        dirs = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
                         if (x, y, z) != (0, 0, 0)], float)  # fmt: skip
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        steps = np.arange(2.0, reach + 0.1, 1.5)
        blocked = np.zeros(len(xyz))
        for u in dirs:
            pts = (xyz[:, None, :] + u * steps[:, None]).reshape(-1, 3)
            hit = min_dist2(pts, protein, touch * 2, cell=None).reshape(len(xyz), len(steps))
            blocked += (hit <= touch**2).any(1)
        return blocked / len(dirs)

    def pockets(self, threshold: float, volume: float, protein, shell=(2.0, 6.0),
                buried: float = 0.4, min_volume: float = 20.0):  # fmt: skip
        """The pockets: enriched, continuous, against ``protein`` and enclosed by it.

        Returns ``(cells, labels, centres, volumes, burials)`` -- the cells of
        every region, which region each belongs to, and a row per region.  An
        enriched blob in bulk is not a pocket, which is why the shell and the
        enclosure come before the clustering rather than after it.

        A region is held together by shared faces, so it is one solid: cells
        that meet only at a corner are no way through for a molecule, and a
        surface drawn through them comes out as the scatter they are.
        """
        from .spatial import min_dist2

        expected = self.total * self.spacing**3 / max(volume, 1e-9)
        enriched = self.counts >= max(threshold * expected, 2.0)
        near = np.sqrt(min_dist2(self.cell_centres(), protein, shell[1] + 1.0, cell=None))
        candidates = np.flatnonzero(enriched & (near >= shell[0]) & (near <= shell[1]))
        if not len(candidates):
            return _no_pockets()
        held = candidates[self.burial(candidates, protein) >= buried]
        if not len(held):
            return _no_pockets()
        held = _fill_enclosed(held, self.dims, near >= shell[0])
        # by faces: a pocket is a volume you can move through, and a region
        # held together at the corners is drawn as a scatter of pieces
        group, ngroups = _join_neighbours(held, self.dims, faces=True)
        deep = self.burial(held, protein)
        xyz = self.cell_centres()[held]
        keep, labels, centres, volumes, burials = [], [], [], [], []
        for g in range(ngroups):
            members = group == g
            if float(members.sum()) * self.spacing**3 < min_volume:
                continue
            keep.append(held[members])
            labels.append(np.full(int(members.sum()), len(centres)))
            centres.append(xyz[members].mean(0))
            volumes.append(float(members.sum()) * self.spacing**3)
            burials.append(float(deep[members].mean()))
        if not centres:
            return _no_pockets()
        return (np.concatenate(keep), np.concatenate(labels), np.array(centres),
                np.array(volumes), np.array(burials))  # fmt: skip


def ligand_centroids(system, positions=None, reference=None, ligand: str = DEFAULT_LIGAND,
                     align: str = "protein and name CA", periodic: bool = True,
                     occupancy: Occupancy | None = None, radius: str | None = "sigma",
                     drift: list | None = None,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """``(centroids (nframes ncopies, 3), (frame, copy) of each, the box volume of each)``.

    The volumes come from the frames themselves, not from the system's stored
    cell: under a barostat the box is not what the structure file says, and
    the concentration a rate is measured against depends on it.

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
    frag = molecules_of(system, lig)
    copies = [lig[frag == f] for f in np.unique(frag)]
    # what the occupancy map is made of: the molecule as one point, its beads as
    # points, or its beads as the spheres they stand for.  A bead is several atoms
    # across, so a map of bead centres at 1 A asks the model a question finer than
    # it answers, and a map of molecule centres asks a coarser one -- the same
    # question the centroid map already answers.
    if radius in (None, "point", "beads"):
        widths = None
    else:
        widths = [particle_radii(system, ids, radius) for ids in copies]
    centre_only = radius == "point"

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

    out, where, sizes, frame = [], [], [], 0
    for xyz, boxes in blocks:
        for X, box in zip(xyz, boxes, strict=True):
            box = box if periodic and np.asarray(box).any() else None
            size = abs(float(np.linalg.det(box))) if box is not None else 0.0
            rot, shift = kabsch(X[fit_at], target)
            if drift is not None:  # how far this frame's protein is from the reference's
                drift.append(float(np.sqrt((((X[fit_at] @ rot.T + shift) - target) ** 2)
                                           .sum(1).mean())))  # fmt: skip
            anchor = X[fit_at].mean(0)
            for c, atoms in enumerate(copy_at):
                p = X[atoms]
                spread = minimum_image(p - p[0], box)  # the copy made whole, bead by bead
                whole = p[0] + spread.mean(0)
                moved = minimum_image((whole - anchor)[None], box)[0] - (whole - anchor)
                whole = whole + moved
                out.append(whole @ rot.T + shift)
                if occupancy is not None:
                    occupancy.add(
                        out[-1][None] if centre_only else (p[0] + spread + moved) @ rot.T + shift,
                        None if centre_only else (widths[c] if widths is not None else None),
                    )
                where.append((frame, c))
                sizes.append(size)  # one per row, so the three returns line up
            frame += 1
    if not out:
        raise ValueError("no frames")
    return np.array(out), np.array(where, np.int64), np.array(sizes)


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
    return flat, dense, counts, dims, lo, expected


def _fill_enclosed(cells, dims, free):
    """``cells``, plus every cell they seal off from the rest of the box.

    The cells are where a probe's atoms went, which is a sample: one in the
    middle of a pocket that no atom happened to visit is still inside the
    pocket, and a region with a hole in it is not what a molecule sits in.  So a
    hole is filled -- but only a hole: a gap that still opens to the box is a
    way out, so two regions with a channel between them stay two regions, and
    scattered cells a wall away from the main one are left where they are, to
    stand or fall as pockets of their own.

    The protein counts as a wall, since most of a pocket's lid is protein, and
    ``free`` (the cells not inside it) is what the fill may take.
    """
    body = np.zeros(int(dims.prod()), bool)
    body[cells] = True
    empty = np.flatnonzero(free.reshape(-1) & ~body)
    if not len(empty):
        return cells
    group, _ = _join_neighbours(empty, dims)
    ijk = np.array(np.unravel_index(empty, tuple(dims))).T
    edge = ((ijk == 0) | (ijk == np.asarray(dims) - 1)).any(1)
    outside = np.unique(group[edge])
    holes = empty[~np.isin(group, outside)]
    if not len(holes):
        return cells
    return np.sort(np.concatenate([cells, holes]))


def _join_neighbours(dense, dims, faces: bool = False):
    """Group cells that touch, by connected components.

    ``faces`` joins only cells that share a face, which is what makes a region
    one solid: cells meeting at a corner or along an edge share no volume, a
    molecule cannot pass between them, and a surface drawn through them comes
    out as separate pieces pinched at a point.  Off, cells touching any of the
    26 ways are joined, which is what a cluster of points wants.
    """
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
                if faces and abs(dx) + abs(dy) + abs(dz) != 1:
                    continue  # a corner or an edge is not a way through
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
          enrichment: float = 20.0,
          periodic: bool = True, pocket_protein: str | None = None,
          rank: str = "pocket", radius: str | None = "sigma",
          buried: float = BURIED, min_volume: float = MIN_VOLUME) -> SiteSet:  # fmt: skip
    """Where the ligand is found across ``runs``, most occupied first.

    ``runs`` is one trajectory (or array of frames) or a list of them; each is
    treated as independent evidence, and a site visited by several runs is a
    claim several simulations agree on.  A site is a connected group of grid
    cells the ligand visits at least ``enrichment`` times more often than
    bulk solvent would explain; everything else is bulk, and is labelled -1
    rather than forced into a site.

    ``rank`` orders what is found: ``"pocket"`` (the default) by how much room the
    pocket has and how enclosed it is, ``"occupied"`` by dwell, ``"agreement"`` by
    how many copies chose it, ``"burial"`` by enclosure alone.  Dwell is the
    tempting one and the wrong one: a sticky patch of surface holds something for
    most of a run without being anywhere a ligand could sit.

    The result carries ``drift``: how far each frame's ``align`` atoms end up from
    the reference's once superposed.  Every site and every pocket is measured in
    the reference's frame, so a protein that changes shape measures its pockets
    against a shape the run no longer has.

    ``radius`` is what the occupancy map a pocket is cut from is made of:

    * ``"point"`` -- the molecule as one point, its own centre;
    * ``"beads"`` -- every bead as a point, the cell its centre fell in;
    * ``"sigma"`` -- every bead as a sphere of half the sigma of its own
      nonbonded term, so the map is the room the molecule took up;
    * ``"rmin"`` -- the same with half of 2**(1/6) sigma, where that potential is
      deepest.

    The three say what a molecule is: a position, a set of positions, or a volume.
    A dipeptide has twice the beads of a single residue, so the choice matters
    more for the one than the other.

    With ``pocket_protein`` a site has to have a pocket to be reported at all: a
    place the ligand gathered but that the protein does not enclose is bulk
    gathering by chance, and there are many of those.  Without one, nothing can
    be measured against, and every site is returned.

    ``buried`` is how enclosed a pocket has to be, 0 being open water and 1 shut
    in, and ``min_volume`` the smallest one worth reporting in cubic angstroms.
    Between them they decide what is a pocket rather than a dent, and so how long
    a list of them a run comes back with -- which is what a hit has to be found
    in.

    """
    if runs is None or not isinstance(runs, (list, tuple)):
        runs = [runs]
    pairs = _pairs(system, runs)
    reference = reference if reference is not None else pairs[0][0]
    points, where, sizes, drift, drift_runs = [], [], [], [], []
    shape = Occupancy(reference.positions[_ids(reference, align)], spacing)
    for r, (own, run) in enumerate(pairs):
        before = len(drift)
        xyz, rows, boxes = ligand_centroids(own, run, reference, ligand, align, periodic,
                                            shape, radius, drift)  # fmt: skip
        drift_runs += [r] * (len(drift) - before)
        points.append(xyz)
        where.append(np.column_stack([np.full(len(rows), r), rows[:, 1], rows[:, 0]]))
        sizes.append(boxes)
    points = np.concatenate(points)
    where = np.concatenate(where)
    sizes = np.concatenate(sizes)
    volume = float(sizes[sizes > 0].mean()) if (sizes > 0).any() else 0.0
    if volume <= 0:  # no cell anywhere: fall back on the space the ligand covered
        cell = np.asarray(system.cell, float)
        volume = abs(float(np.linalg.det(cell)))
    if volume <= 0:
        hull = points.max(0) - points.min(0)
        volume = float(np.prod(np.maximum(hull, spacing)))

    flat, dense, counts, dims, lo, expected = _dense_cells(points, spacing, enrichment, volume)
    group, ngroups = _join_neighbours(dense, dims)
    of_cell = np.full(int(dims.prod()) + 1, -1, np.int64)
    of_cell[dense] = group
    labels = of_cell[flat]

    found = []
    # the frames there are to be occupied: one row per (run, frame), counted once
    # however many copies the box holds
    all_frames = len(np.unique(where[:, [0, 2]], axis=0))
    for g in range(ngroups):
        members = np.flatnonzero(labels == g)
        if len(members) < 2:  # a cell a copy passed through once is not a site
            continue
        rows, xyz = where[members], points[members]
        held = len(np.unique(rows[:, [0, 2]], axis=0)) / max(all_frames, 1)
        centre = xyz.mean(0)
        found.append(Site(center=centre, points=members, occupancy=held,
                          copy_frames=len(members) / len(points),
                          runs=len(np.unique(rows[:, 0])), copies=len(np.unique(rows[:, 1])),
                          arrivals=_visits(rows),
                          spread=float(np.sqrt(((xyz - centre) ** 2).sum(1).mean()))))  # fmt: skip
    # the pocket each site sits in: enriched, continuous, against the protein and
    # enclosed by it, which an enriched blob in bulk is not
    pocket_atoms = _ids(reference, pocket_protein) if pocket_protein else np.empty(0, np.int64)
    if len(pocket_atoms) >= 4:
        cells, labels, centres, volumes, burials = shape.pockets(
            enrichment,
            volume,
            np.asarray(reference.positions)[pocket_atoms],
            buried=buried,
            min_volume=min_volume,
        )
        claimed: dict[int, list[int]] = {}
        for i, site in enumerate(found):
            if not len(centres):
                continue
            near = np.linalg.norm(centres - site.center, axis=1)
            k = int(np.argmin(near))
            if near[k] <= 6.0:  # its own pocket, not a neighbour's
                claimed.setdefault(k, []).append(i)
        # a big pocket can hold two clusters of centroids -- two spots a molecule
        # sits in, a few angstroms apart -- and they are one site, not two: the
        # pocket is what a ligand would occupy, so the clusters in it are merged
        merged = set()
        for k, mine in claimed.items():
            if len(mine) > 1:
                keep = found[mine[0]]
                keep.points = np.concatenate([found[i].points for i in mine])
                _remeasure(keep, where, points, all_frames)
                merged.update(mine[1:])
            site = found[mine[0]]
            site.volume, site.burial = float(volumes[k]), float(burials[k])
            site.cells = cells[labels == k]
            site.grid_dims, site.grid_origin = shape.dims, shape.origin
            site.grid_spacing = shape.spacing
        if merged:
            found = [s for i, s in enumerate(found) if i not in merged]
        # a site with no pocket is not somewhere a ligand could sit: bulk solvent
        # gathers by chance here and there, and with a protein to measure against
        # those places fail the shell and the enclosure rather than being gated on
        # how long they were held.  Where there is no protein to measure against,
        # every site is kept, there being nothing to tell them apart by.
        found = [s for s in found if s.volume > 0.0]
    found.sort(key=_RANKS[rank])
    out = np.full(len(points), -1)
    for k, s in enumerate(found):
        out[s.points] = k
    return SiteSet(sites=found, labels=out, where=where, centroids=points, occupancy=shape,
                   drift=np.array(drift, float), drift_runs=np.array(drift_runs, np.int64),
                   systems=[own for own, _ in pairs],
                   spacing=float(spacing), enrichment=float(enrichment), volume=volume,
                   density=Density(origin=lo, spacing=float(spacing),
                                   counts=counts.reshape(dims).astype(np.int32),
                                   expected=float(expected)))  # fmt: skip


def _pairs(system, runs) -> list[tuple]:
    """``runs`` as (system, frames) pairs: a run may bring its own system.

    Only the protein has to be shared, because a site is made of ligand
    centres: the ligands themselves may be different molecules, different in
    number and different in size from one run to the next.
    """
    if runs is None or not isinstance(runs, (list, tuple)):
        runs = [runs]
    out = []
    for run in runs:
        if isinstance(run, tuple) and len(run) == 2 and hasattr(run[0], "natoms"):
            out.append((run[0], run[1]))
        else:
            out.append((system, run))
    return out


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
    pairs = _pairs(system, runs)
    prot = _ids(system, protein)
    rows = found.frames(k)
    hits, seen = np.zeros(len(prot)), 0
    for r, (here_system, run) in enumerate(pairs):
        lig = _ids(here_system, ligand)
        lig = lig[here_system.atoms["anum"][lig] > 1]
        frag = molecules_of(here_system, lig)
        copies = [lig[frag == f] for f in np.unique(frag)]
        here = _ids(here_system, protein)
        if len(here) != len(prot):
            raise ValueError(f"protein {protein!r} selects {len(here)} atoms in run {r} and "
                             f"{len(prot)} in the first; they are paired in order")  # fmt: skip
        own = np.isin(here, lig)
        mine = rows[rows[:, 0] == r]
        if not len(mine):
            continue
        wanted = np.unique(mine[:, 2])
        need = np.union1d(here, lig)
        pl = np.searchsorted(need, here)
        blocks, _ = _boxed_blocks(here_system, run[wanted], need)
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


#: Feature families a map may be written for, and a colour to tell them apart.
VIEWER_COLORS = {"donor": "skyblue", "acceptor": "salmon", "aromatic": "violet",
                 "hydrophobe": "yellow", "posionizable": "blue",
                 "negionizable": "red"}  # fmt: skip


#: What a mask is drawn at: half of the one value in it, so the surface is the
#: whole pocket rather than the cells the probes happened to visit most.
POCKET_LEVEL = 0.5


def write_viewer_scripts(directory, sites, level: float = 50.0) -> list[Path]:
    """Write ``sites.pml`` and ``sites.tcl`` beside the maps; return what was written.

    They load each pocket as a solid surface, the enrichment maps as isosurfaces
    at ``level`` times bulk, each feature map beside them (switched off, to turn
    on one at a time), the hotspots as spheres, and a marker at every site's
    centre.  ``source <dir>/sites.tcl`` in VMD, ``@<dir>/sites.pml`` in PyMOL -- a
    session set up by vizard or pizard in either case, since neither viewer reads
    a bead file on its own.

    The files a script names are written into it in full, so it runs from
    whatever directory the viewer happens to be in; move the directory and the
    scripts want writing again.
    """
    d = Path(directory)
    at = d.resolve()

    def file(name: str) -> str:
        """A path a viewer will take whole, even with a space in it."""
        path = str(at / name)
        return f'"{path}"' if " " in path else path

    pml = [
        f"# boonza sites.  @{at / 'sites.pml'} in a pizard session, from any directory.",
        f"# Isosurfaces are {level:g}x bulk.  pocketK.dx is a mask of site K's pocket:",
        f"# 1 inside it, drawn solid at {POCKET_LEVEL:g}, so what you see is the volume that",
        "# was reported.  occupancy.dx is every cell the atoms reach, bulk included, as",
        "# enrichment; density.dx is where a molecule's centre sits, off the surface for",
        "# a dipeptide.",
    ]
    tcl = [
        f"# boonza sites.  source {at / 'sites.tcl'} in a vizard session, from any directory.",
        f"# Isosurfaces are {level:g}x bulk; a pocket is a mask, drawn solid at "
        f"{POCKET_LEVEL:g}.",  # fmt: skip
    ]
    for k in range(len(sites)):  # each site's own pocket: enclosed, against the protein
        if not (d / f"pocket{k}.dx").is_file():
            continue
        pml += [f"load {file(f'pocket{k}.dx')}, pocket{k}_map",
                f"isosurface pocket{k}, pocket{k}_map, {POCKET_LEVEL:g}",
                f"color {'orange' if k % 2 == 0 else 'marine'}, pocket{k}",
                f"set transparency, 0.4, pocket{k}"]  # fmt: skip
        tcl += [f"mol new {file(f'pocket{k}.dx')} type dx waitfor all",
                f"mol modstyle 0 top Isosurface {POCKET_LEVEL:g} 0 0 1 1 1",
                f"mol rename top pocket{k}"]  # fmt: skip
    if (d / "occupancy.dx").is_file():  # every pocket at once, and the bulk with it
        pml += [f"load {file('occupancy.dx')}, occupancy",
                f"isomesh occupancy_mesh, occupancy, {level:g}",
                "color grey50, occupancy_mesh", "disable occupancy_mesh"]  # fmt: skip
        tcl += [f"mol new {file('occupancy.dx')} type dx waitfor all",
                f"mol modstyle 0 top Isosurface {level:g} 0 0 0 1 1",
                "mol rename top occupancy", "mol off top"]  # fmt: skip
    pml += [f"load {file('density.dx')}, density",
            f"isomesh density_mesh, density, {level:g}",
            "color grey70, density_mesh", "disable density_mesh"]  # fmt: skip
    tcl += [f"mol new {file('density.dx')} type dx waitfor all",
            f"mol modstyle 0 top Isosurface {level:g} 0 0 0 1 1",
            "mol rename top density", "mol off top"]  # fmt: skip
    for family, colour in VIEWER_COLORS.items():
        if not (d / f"{family}.dx").is_file():
            continue
        pml += [f"load {file(f'{family}.dx')}, {family}",
                f"isomesh {family}_mesh, {family}, {level:g}",
                f"color {colour}, {family}_mesh", f"disable {family}_mesh"]  # fmt: skip
        tcl += [f"mol new {file(f'{family}.dx')} type dx waitfor all",
                f"mol modstyle 0 top Isosurface {level:g} 0 0 0 1 1",
                f"mol rename top {family}", "mol off top"]  # fmt: skip
    if (d / "hotspots.pdb").is_file():
        pml += [f"load {file('hotspots.pdb')}, hotspots", "show spheres, hotspots",
                "set sphere_scale, 0.3, hotspots"]  # fmt: skip
        tcl += [f"mol new {file('hotspots.pdb')} waitfor all",
                "mol modstyle 0 top VDW 0.3 12", "mol rename top hotspots"]  # fmt: skip
    tcl += ["draw materials on", "draw color red"]
    for k, site in enumerate(sites):
        x, y, z = (round(float(v), 2) for v in site.center)
        held = 100 * site.occupancy
        pml += [f"# site {k}: {held:.1f}% of frames, {site.copies} copies",
                f"pseudoatom site{k}, pos=[{x}, {y}, {z}], label=site{k}",
                f"show spheres, site{k}", f"color red, site{k}",
                f"select around{k}, byres (polymer within 6 of site{k})"]  # fmt: skip
        tcl += [f"# site {k}: {held:.1f}% of frames, {site.copies} copies",
                f"draw sphere {{{x} {y} {z}}} radius 1.5 resolution 20"]  # fmt: skip
    if len(sites):
        pml += ["deselect", "orient around0", "zoom around0, 4"]
    (d / "sites.pml").write_text("\n".join(pml) + "\n")
    (d / "sites.tcl").write_text("\n".join(tcl) + "\n")
    return [d / "sites.pml", d / "sites.tcl"]
