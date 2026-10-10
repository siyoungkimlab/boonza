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
from .pbc import distances, minimum_image_in, prepare_box
from .symmetry import DEFAULT_LIGAND, _boxed_blocks, _ids, molecules_of


@dataclass(eq=False)  # its fields are arrays: == between sites would compare them element-wise
class Site:
    """One place the ligand is found, and the evidence for it.  Sites compare by identity, so
    ``sites.index(site)`` and ``site in sites`` find this very site in any order."""

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
    peak: np.ndarray | None = None  # the densest cell of the site's own points

    def __len__(self) -> int:
        return len(self.points)

    def pocket_points(self) -> np.ndarray:
        """The centres of the pocket's cells, in the reference's frame.

        Empty where the site has no pocket, which is what makes the shape of one
        something to measure against a known ligand.
        """
        if self.cells is None or not len(self.cells) or self.grid_origin is None:
            return np.empty((0, 3))
        ijk = np.array(np.unravel_index(np.asarray(self.cells), tuple(self.grid_dims))).T
        return self.grid_origin + (ijk + 0.5) * self.grid_spacing

    def peak_point(self) -> np.ndarray:
        """Where the ligand sat most, rather than the middle of where it sat.

        The mean is pulled off the pocket when a site is elongated or has a
        shoulder on it; the mode is not.  Empty until the site is built with
        one, in which case the centre is what there is.
        """
        return (np.asarray(self.peak, float) if self.peak is not None
                else np.asarray(self.center, float))  # fmt: skip

    def pocket_center(self) -> np.ndarray:
        """The pocket's own centre, which is what a benchmark measures from; the
        site's centre -- where the ligand's centroids cluster -- when it has none."""
        pts = self.pocket_points()
        return pts.mean(0) if len(pts) else np.asarray(self.center, float)


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
#: The fewest counts a cell may be called enriched on, whatever the ratio says.
#:
#: An enrichment is a ratio and says nothing about how much was counted to get
#: it: a cell holding two counts where bulk expects a tenth of one clears twenty
#: times bulk, and two is a thing that happens.  So a floor is needed, and two is
#: what it is.
#:
#: Measured, two is about right.  Asking instead for a count improbable from bulk
#: -- a Poisson tail at the map's own rate, spread over its cells -- comes to the
#: same two when set loosely, and when set strictly enough to suppress what a
#: thin sample passes on luck it takes real pockets with it: over 181
#: coarse-grained runs of six proteins, five lost theirs, among them pockets
#: covering half of the crystal ligand.  There is no setting that keeps a sparse
#: run's pockets and drops its noise, the difference between them being
#: information the sparse run does not have.
LEAST_COUNT = 2.0
#: How enclosed a pocket has to be: the share of 26 directions out of its cells
#: that meet protein, 0 being open water and 1 shut in.
BURIED = 0.4
#: The smallest pocket worth reporting, in cubic angstroms.  Twenty is twenty
#: cells of a 1 A grid, where a pocket a ligand sits in runs to hundreds.
MIN_VOLUME = 20.0
#: How wide a particle is when no force field says: a heavy atom's van der Waals
#: radius, in angstroms.
HEAVY_ATOM = 1.7
#: How close to the protein a pocket lies and how far from it, in angstroms,
#: measured from the particles' centers -- which is the default.
SHELL = (2.0, 6.0)
#: The same band measured from every particle's van der Waals surface, which is
#: what ``shell="surface"`` asks for: clear of a surface by ``SHELL_SURFACE[0]``
#: and within ``SHELL_SURFACE[1]`` of a center.  Measured from the centers, 2 A
#: is outside a heavy atom, which reaches 1.7, and 0.2 A inside a Martini bead,
#: which reaches 2.15 to 2.35, so at coarse-grained resolution the band begins
#: inside the protein and a pocket comes out drawn over the beads.
#:
#: It is off by default because the two measures of a hit disagree about it.
#: Over 181 coarse-grained runs it took top-1 from 51 to 48 and the oracle from
#: 77 to 73 by ligand coverage, and top-1 from 50 to 54 by DCA: it does not move
#: a pocket's center -- it places it better -- but it shaves the wall-facing
#: cells a ligand's atoms sit against, which is what coverage counts.  It does
#: take the volume inside the protein from 4.0% to none at all, in every force
#: field.  The sizes come from the force field, so a system loaded without
#: its nonbonded tables has none to read and falls back on its elements;
#: ``boonza sites --shell surface`` reads them.
SHELL_SURFACE = (0.3, 6.0)
#: The same band with no near edge at all, which is what ``shell="none"`` asks
#: for: every enriched cell within ``SHELL_ANY[1]`` of a particle's centre is a
#: candidate, however close the protein is.
#:
#: A probe's density is evidence that a probe was there, and the protein it is
#: measured against is one structure out of a run that moved: a cell inside a
#: bead here was open when the probe sat in it, so a near edge of any kind
#: discards sampling for the reference's convenience.
#:
#: Measured over 181 coarse-grained runs it is the best of the three by ligand
#: coverage -- top-1 54 against 51, oracle 80 against 77 -- and the worst by DCA,
#: top-1 45 against 50.  Both come from the same thing: pockets merge through the
#: protein, cells inside a bead touching the open cells on either side of it, so
#: a thin wall no longer separates two pockets.  The merged pocket covers more of
#: a ligand and its centre is further from one: over those runs the median pocket
#: grows from 48 to 58 cubic angstroms, and for SIRAH's single amino acids from
#: 65 to 126 while the count falls from 19 to 12.
#:
#: Of the six runs it finds a hit in where ``"center"`` finds none, five have the
#: pocket's centre within 1.5 A of a ligand atom and three are no larger than the
#: median pocket of their own run, which is not something a blob does; two of them
#: are a cryptic site Martini otherwise never finds.  It loses three, all of them
#: good pockets destroyed by merging, one holding 95% of its ligand.  So it is
#: offered: what it recovers is real, and what it costs is real too.
#:
#: Filling still will not reach inside the protein, there being no evidence there
#: to fill with.
SHELL_ANY = (-np.inf, 6.0)


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


def burial(points, protein, reach: float = 10.0, touch: float = 2.6) -> np.ndarray:
    """Of 26 directions out of each of ``points``, the share that meet ``protein``.

    A pocket is enclosed; a dent on a convex surface is not, and bulk is not at
    all.  Counting directions rather than neighbours within a radius keeps the
    number comparable between an all-atom protein and a coarse-grained one,
    whose beads are fewer and larger.  A direction meets the protein where a
    step along it, 2 to ``reach`` A out, comes within ``touch`` of a particle.
    """
    from .spatial import min_dist2

    xyz = np.asarray(points, float).reshape(-1, 3)
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
        """Of 26 directions out of each of ``cells``, the share that meet ``protein``
        (:func:`burial` at the cells' centres)."""
        return burial(self.cell_centres()[cells], protein, reach, touch)

    def clearance(self, protein, radii=None, reach: float = 6.0) -> np.ndarray:
        """Per cell, how far clear of the nearest of ``protein``'s surfaces it is.

        The distance to a particle's centre less that particle's radius: zero on
        its surface, negative inside it.  The radius is what makes this a surface
        rather than a set of points, and the two resolutions differ by more than
        the question tolerates -- a heavy atom reaches 1.7 A and a Martini bead
        2.15 to 2.35 -- so a cell 2 A from a centre is outside an atom and inside
        a bead.  Without ``radii``, every particle is a heavy atom.

        Only cells within ``reach`` of some surface are measured; the rest come
        back as that bound, which is as much as a shell needs to know.
        """
        from .spatial import min_dist2

        xyz = self.cell_centres()
        radii = np.full(len(protein), HEAVY_ATOM) if radii is None else np.asarray(radii, float)
        # one search per distinct radius, each taking its group's largest: the
        # plain nearest-point search again rather than a weighted one
        bucket = np.ceil(np.asarray(radii) / 0.05) * 0.05
        out = np.full(len(xyz), np.inf)
        for r in np.unique(bucket):
            d = min_dist2(xyz, protein[bucket == r], reach + r + 1.0, cell=None)
            out = np.minimum(out, np.sqrt(d.astype(float)) - r)
        return out

    def pockets(self, threshold: float, volume: float, protein, shell=SHELL,
                buried: float = 0.4, min_volume: float = 20.0, radii=None):  # fmt: skip
        """The pockets: enriched, continuous, against ``protein`` and enclosed by it.

        Returns ``(cells, labels, centres, volumes, burials)`` -- the cells of
        every region, which region each belongs to, and a row per region.  An
        enriched blob in bulk is not a pocket, which is why the shell and the
        enclosure come before the clustering rather than after it.

        ``shell`` is the band a pocket lies in: ``shell[0]`` clear of the protein
        and within ``shell[1]`` of a particle's center.  Without ``radii`` the
        near edge is measured from the centers, which is what it has always been;
        with them (:func:`particle_radii`, the force field's own sizes) it is
        measured from each particle's van der Waals surface, and so is the wall
        that closes a pocket's holes.  The difference is the whole of
        ``shell="surface"``: measured from the centers a band starting 2 A out
        starts 0.2 A inside a Martini bead, so a pocket takes in room no probe
        can reach.  An all-atom protein measures much the same either way.

        A region is held together by shared faces, so it is one solid: cells
        that meet only at a corner are no way through for a molecule, and a
        surface drawn through them comes out as the scatter they are.
        """
        from .spatial import min_dist2

        expected = self.total * self.spacing**3 / max(volume, 1e-9)
        enriched = self.counts >= max(threshold * expected, LEAST_COUNT)
        near = np.sqrt(min_dist2(self.cell_centres(), protein, shell[1] + 1.0, cell=None))
        # from the surfaces when the force field's sizes are given, from the
        # centers when they are not: the far edge only asks whether a cell is
        # near the protein, which needs no sizes either way
        clear = near if radii is None else self.clearance(protein, radii, shell[1])
        candidates = np.flatnonzero(enriched & (clear >= shell[0]) & (near <= shell[1]))
        if not len(candidates):
            return _no_pockets()
        held = candidates[self.burial(candidates, protein) >= buried]
        if not len(held):
            return _no_pockets()
        # the wall a hole is sealed by is the protein itself: its surfaces when
        # their sizes are known, and otherwise the near edge of the band, which
        # is the only thing there is to seal against
        held = _fill_enclosed(held, self.dims, clear >= (0.0 if radii is not None else shell[0]))
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


#: How close a pocket's centre has to be to an atom of the known ligand to count
#: as that ligand's pocket, in angstroms: the DCA of the pocket-prediction
#: benchmarks (distance from the centre to the closest ligand atom).
DCA_HIT = 4.0
#: How close a ligand atom has to be to one of the pocket's cells to count as
#: inside it, in angstroms -- half a cell more than the grid's own spacing.
INSIDE = 1.5


def _pocket_points(pocket) -> np.ndarray:
    """``pocket`` as coordinates: a :class:`Site`, or points already."""
    if isinstance(pocket, Site):
        return pocket.pocket_points()
    return np.asarray(pocket, float).reshape(-1, 3)


def _densest(xyz, spacing: float) -> np.ndarray:
    """The mean of the points in the cell that holds most of them.

    A site's centre is the mean of everywhere the ligand sat, which a long
    pocket or one with a shoulder pulls away from where it actually sat.  This
    is the mode at the map's own resolution: the busiest cell, averaged inside
    it so the answer is not quantised to the grid.
    """
    pts = np.asarray(xyz, float).reshape(-1, 3)
    if len(pts) < 2:
        return pts.mean(0) if len(pts) else np.full(3, np.nan)
    lo = pts.min(0)
    ijk = ((pts - lo) / spacing).astype(np.int64)
    dims = ijk.max(0) + 1
    flat = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    counts = np.bincount(flat)
    return pts[flat == int(counts.argmax())].mean(0)


def _pocket_center(pocket) -> np.ndarray:
    if isinstance(pocket, Site):
        return pocket.pocket_center()
    pts = _pocket_points(pocket)
    return pts.mean(0) if len(pts) else np.full(3, np.nan)


def dca(pocket, ligand) -> float:
    """Distance from ``pocket``'s centre to the closest atom of ``ligand``, in A.

    The DCA of the pocket-prediction benchmarks, which count a prediction right
    within :data:`DCA_HIT`.  ``pocket`` is a :class:`Site` or the points of one,
    ``ligand`` the atoms of the known ligand in the same frame of reference --
    which for a holo structure means superposed on the run's own, since every
    site is measured in the reference's frame.

    It is the lenient of the two measures: a small pocket beside the ligand
    passes it while enclosing little of it, which is what :func:`coverage` asks.
    """
    lig = np.asarray(ligand, float).reshape(-1, 3)
    centre = _pocket_center(pocket)
    if not len(lig) or not np.isfinite(centre).all():
        return float("nan")
    return float(np.sqrt(((lig - centre) ** 2).sum(1).min()))


def dpa(pocket, ligand) -> float:
    """Distance from ``pocket``'s busiest point to the closest atom of ``ligand``.

    :func:`dca` measured from the centre, which is the mean of a pocket's cells
    and so sits between the lobes of a pocket that has more than one.  This
    measures from the peak of the ligand's own density instead -- where it sat
    most rather than the middle of where it sat -- which is the same number for
    a round pocket and a smaller one for a long or a merged one.  Reported
    beside DCA rather than in place of it: the benchmarks are quoted on DCA.
    """
    lig = np.asarray(ligand, float).reshape(-1, 3)
    peak = (pocket.peak_point() if isinstance(pocket, Site)
            else _pocket_center(pocket))  # fmt: skip
    if not len(lig) or not np.isfinite(peak).all():
        return float("nan")
    return float(np.sqrt(((lig - peak) ** 2).sum(1).min()))


def dcc(pocket, ligand) -> float:
    """Distance from ``pocket``'s centre to ``ligand``'s centroid, in A -- the DCC
    the same benchmarks report beside :func:`dca`, and the harsher of the two on a
    pocket that runs past one end of the ligand."""
    lig = np.asarray(ligand, float).reshape(-1, 3)
    centre = _pocket_center(pocket)
    if not len(lig) or not np.isfinite(centre).all():
        return float("nan")
    return float(np.linalg.norm(lig.mean(0) - centre))


def coverage(pocket, ligand, within: float = INSIDE) -> float:
    """The share of ``ligand``'s atoms lying inside ``pocket``, 0 to 1.

    An atom is inside where a cell of the pocket is ``within`` angstroms of it.
    This is the measure that says whether the pocket holds the binding mode
    rather than merely sitting beside it: a pocket can pass :func:`dca` on one
    corner and cover almost none of the ligand.

    Give it the ligand's heavy atoms; hydrogens the model never had would count
    against a coarse-grained pocket for being absent from it.
    """
    from .spatial import min_dist2

    lig = np.asarray(ligand, float).reshape(-1, 3)
    pts = _pocket_points(pocket)
    if not len(lig) or not len(pts):
        return 0.0
    return float((min_dist2(lig, pts, within + 1.0, cell=None) <= within**2).mean())


#: Residues a holo structure carries that are not what was bound: the solvent and
#: the salts and buffer a crystal is grown in.
NOT_A_LIGAND = ("HOH", "WAT", "DOD", "SOL", "TIP3", "SO4", "PO4", "GOL", "EDO", "PEG",
                "MPD", "DMS", "ACT", "ACY", "FMT", "CIT", "TRS", "MES", "EPE", "IMD",
                "CL", "NA", "K", "MG", "CA", "ZN", "MN", "FE", "NI", "CD", "CU", "BR",
                "IOD", "NH4", "NO3", "SCN", "AZI", "FLC", "TLA", "MLI")  # fmt: skip


def known_ligand(holo, reference, ligand: str | None = None, align: str | None = None,
                 ) -> tuple[np.ndarray, dict]:  # fmt: skip
    """A holo structure's ligand, moved into ``reference``'s frame.

    ``(heavy atoms (n, 3), what was used)``.  Every site and pocket is measured in
    the reference's frame, so a crystal structure of the same protein with
    something bound only means anything there: its backbone is superposed on the
    reference's by sequence, and the ligand carried along by the same transform.

    ``align`` is the backbone to fit; by default alpha carbons against whatever
    the reference calls its backbone (``CA``, Martini's ``BB``, SIRAH's ``GC``).
    ``ligand`` selects what was bound; by default the largest residue that is
    neither protein nor nucleic nor one of :data:`NOT_A_LIGAND`.  A structure with
    several copies of the protein is fitted by each in turn and the closest kept,
    since a crystal's chains are the same protein in different places.
    """
    from .align import superpose

    names = set(np.asarray(reference.atoms["name"]).tolist())
    back = next((n for n in ("CA", "BB", "GC") if n in names), None)
    if back is None:
        raise ValueError("the reference has no CA, BB or GC atoms to fit a holo structure on")
    if ligand is None:
        rid = np.asarray(holo.atoms["residue"])
        res = np.asarray(holo.residues["name"])
        taken = np.zeros(holo.natoms, bool)
        for sel in ("protein", "nucleic"):
            taken[_ids(holo, sel)] = True
        sizes: dict[str, int] = {}
        for r in np.unique(rid[~taken]).tolist():
            name = str(res[r]).strip()
            if name.upper() in NOT_A_LIGAND:
                continue
            sizes[name] = max(sizes.get(name, 0), int((rid == r).sum()))
        if not sizes:
            raise ValueError("no ligand in the holo structure; name one with ligand=")
        ligand = f"resname {max(sizes, key=lambda k: sizes[k])}"
    # a chain read from a file without a chain column has no name at all, and
    # 'chain ' does not parse: quote every one of them rather than guess which
    chains = {str(c): f'chain "{c}"'
              for c in dict.fromkeys(np.asarray(holo.chains["name"]).tolist())}  # fmt: skip

    def fitted(selection, chain, carried):
        """The ligand in the reference's frame, or None if that chain cannot fit."""
        sub = holo.select(selection).clone()
        lig = sub.select(f"({ligand}) and not element H").ids
        if not len(lig) or len(sub.select("protein and name CA").ids) < 20:
            return None
        try:
            fit = superpose(sub, reference, sel=align or "protein and name CA",
                            ref_sel=align or f"name {back}", match="sequence")  # fmt: skip
        except ValueError:
            return None
        return (np.asarray(sub.positions)[lig],
                {"fit_rmsd": float(fit.rmsd), "chain": str(chain), "ligand": ligand,
                 "atoms": int(len(lig)), "paired": int(fit.n_used),
                 **({"ligand_chain": carried} if carried is not None else {})})  # fmt: skip

    best = None
    for chain, named in chains.items():  # the ligand sits in the protein's own chain
        got = fitted(named, chain, None)
        if got and (best is None or got[1]["fit_rmsd"] < best[1]["fit_rmsd"]):
            best = got
    if best is None:
        # a ligand written as a chain of its own, as Maestro writes one: fit the
        # protein chain it actually touches and carry the ligand along with it.
        # Nearest first, so a second copy of the protein cannot claim it.
        lig = holo.select(f"({ligand}) and not element H")
        if not len(lig.ids):
            raise ValueError(f"the holo structure has no {ligand}")
        held = np.asarray(holo.positions)[np.asarray(lig.ids, int)]
        where = {str(c) for c in np.asarray(holo.chains["name"])[
            np.asarray(holo.residues["chain"])[np.asarray(holo.atoms["residue"])[
                np.asarray(lig.ids, int)]]]}  # fmt: skip
        pairs = []
        for chain, named in chains.items():
            ca = holo.select(f"{named} and protein and name CA").ids
            if len(ca) < 20 or chain in where:
                continue
            near = np.asarray(holo.positions)[np.asarray(ca, int)]
            apart = float(np.sqrt(((near[:, None] - held[None]) ** 2).sum(-1)).min())
            pairs.append((apart, str(chain)))
        for apart, chain in sorted(pairs):
            for lc in sorted(where):
                got = fitted(f'{chains[chain]} or (({ligand}) and chain "{lc}")', chain, lc)
                if got:
                    got[1]["ligand_apart_A"] = round(apart, 2)
                    best = got
                    break
            if best is not None:
                break
    if best is None:
        raise ValueError(f"no chain of the holo structure has 20 alpha carbons to fit "
                         f"with, next to its {ligand}")  # fmt: skip
    return best


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

    # the copies in blocks of equal size, so a frame's hundreds of them are one set
    # of array operations rather than hundreds of calls on three floats apiece.
    # Which copies are in a block is kept, so the rows still come back in copy order
    blocked = []
    for beads in sorted({len(a) for a in copy_at}):
        who = np.array([c for c, a in enumerate(copy_at) if len(a) == beads])
        blocked.append((who, np.stack([copy_at[c] for c in who]),
                        None if widths is None else [widths[c] for c in who]))  # fmt: skip

    ncopies = len(copy_at)
    out, where, sizes, frame = [], [], [], 0
    for xyz, boxes in blocks:
        for X, box in zip(xyz, boxes, strict=True):
            box = box if periodic and np.asarray(box).any() else None
            size = abs(float(np.linalg.det(box))) if box is not None else 0.0
            cell = prepare_box(box)  # once a frame: every copy in it shares the box
            rot, shift = kabsch(X[fit_at], target)
            if drift is not None:  # how far this frame's protein is from the reference's
                drift.append(float(np.sqrt((((X[fit_at] @ rot.T + shift) - target) ** 2)
                                           .sum(1).mean())))  # fmt: skip
            anchor = X[fit_at].mean(0)
            centres = np.empty((ncopies, 3))
            for who, atoms, wide in blocked:
                p = X[atoms]  # (copies, beads, 3)
                # the copies made whole, bead by bead, each about its own first bead
                spread = minimum_image_in((p - p[:, :1]).reshape(-1, 3), cell).reshape(p.shape)
                whole = p[:, 0] + spread.mean(1)
                away = whole - anchor
                moved = minimum_image_in(away, cell) - away
                centres[who] = (whole + moved) @ rot.T + shift
                if occupancy is None:
                    continue
                if centre_only:
                    occupancy.add(centres[who], None)
                    continue
                beads = (p[:, :1] + spread + moved[:, None]) @ rot.T + shift
                # a copy at a time, as the radii are grouped per call and one
                # block's worth of them would be scanned once for every radius
                for k in range(len(who)):
                    occupancy.add(beads[k], None if wide is None else wide[k])
            out.append(centres)
            where.append(np.column_stack([np.full(ncopies, frame), np.arange(ncopies)]))
            sizes.append(np.full(ncopies, size))
            frame += 1
    if not out:
        raise ValueError("no frames")
    return (np.concatenate(out), np.concatenate(where).astype(np.int64),
            np.concatenate(sizes))  # fmt: skip


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
    dense = np.flatnonzero(counts >= max(threshold * expected, LEAST_COUNT))
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
          buried: float = BURIED, min_volume: float = MIN_VOLUME,
          shell: str = "center") -> SiteSet:  # fmt: skip
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

    ``shell`` is what a pocket's near edge is, and ``"none"`` is no near edge: it
    keeps every enriched cell within 6 A of the protein, however close, because a
    cell's counts are evidence that a probe was there and the protein is measured
    in one structure out of a run that moved.  It keeps sampling the other two
    discard -- over 181 coarse-grained runs the best of the three by ligand
    coverage, top-1 54 against 51 -- and merges pockets that a thin wall of
    protein separates, which makes it the worst by DCA, top-1 45 against 50, and
    takes the median pocket from 48 to 58 cubic angstroms.

    Otherwise ``shell`` is what the near edge is measured from.  ``"center"``, the
    default, keeps it 2 A from the nearest particle's center; ``"surface"`` keeps
    it clear of that particle's van der Waals surface instead, by the force
    field's own sizes, which is the only way to keep a pocket out of a bead:
    2 A from a center is 0.2 A inside a Martini bead.  It is off by default
    because the two measures of a hit disagree about it: over 181 coarse-grained
    runs it costs ligand coverage (top-1 51 to 48, oracle 77 to 73) and gains DCA
    (top-1 50 to 54).  It does not move a pocket's center but shaves the
    wall-facing cells a ligand's atoms lie against, which is what coverage counts
    and DCA does not.  It takes the pocket volume inside the protein, 4.0% of it
    over those runs, to none.
    It also needs the sizes to exist, and they are in the force field: a system
    loaded without its nonbonded tables falls back on its elements, which is a
    heavy atom's 1.7 A where a Martini bead reaches 2.35.  The command line
    reads them for you; :func:`boonza.load` without ``without_tables=True``
    does the same for a system of your own.

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
    if shell not in ("center", "surface", "none"):
        raise ValueError(f"shell {shell!r}: 'center' (2 A from a center), 'surface' or 'none'")
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
        found.append(Site(center=centre, peak=_densest(xyz, spacing), points=members,
                          occupancy=held,
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
            # how wide the protein is, whatever the probes were binned as: a
            # probe binned as a point does not make the beads it meets any
            # smaller, so the protein is measured by a rule of its own
            shell={"surface": SHELL_SURFACE, "none": SHELL_ANY}.get(shell, SHELL),
            radii=(
                particle_radii(reference, pocket_atoms, radius if radius == "rmin" else "sigma")
                if shell != "center"
                else None
            ),
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

    A script names its files relative to itself and finds itself when it runs --
    PyMOL sets ``__script__``, and Tcl's ``info script`` is the path being
    sourced -- so the directory can be moved, or copied off the machine the
    analysis ran on, and the scripts still work.  Written in full they would name
    a cluster's filesystem and break on the way home, which is the usual way to
    read them.

    Where a viewer does not say which script it is running (VMD's ``-e``, as
    against ``source``), the files are looked for in the working directory, so
    running from the directory itself always works.
    """
    d = Path(directory)
    at = d.resolve()

    def file(name: str) -> str:
        """A path a viewer will take whole, even with a space in it."""
        return f'"{name}"' if " " in name else name

    pml = [
        # a semicolon in a .pml comment ends the comment -- PyMOL splits commands on
        # it and hands the rest to Python, which an apostrophe then breaks -- so the
        # prose here has none
        "# boonza sites.  @sites.pml in a pizard session, from any directory and any",
        "# machine: the maps are named beside this file rather than where they were",
        "# written, so this directory can be copied anywhere.",
        f"# Written in {at}.",
        f"# Isosurfaces are {level:g}x bulk.  pocketK.dx is a mask of site K's pocket:",
        f"# 1 inside it, drawn solid at {POCKET_LEVEL:g}, so what you see is the volume that",
        "# was reported.  occupancy.dx is every cell the atoms reach, bulk included, as",
        "# enrichment.  density.dx is where a molecule's centre sits, off the surface",
        "# for a dipeptide.",
        # the maps are named beside this file, so the viewer is pointed at wherever
        # this file turned out to be.  PyMOL sets __script__ to the script it is
        # running, for @ and for -d alike; without it, the working directory
        "python",
        "import os",
        "from pymol import cmd",
        'cmd.cd(os.path.dirname(os.path.abspath(__script__)) if "__script__" in globals()'
        ' else ".")',
        "python end",
    ]
    tcl = [
        "# boonza sites.  source sites.tcl in a vizard session, from any directory and",
        "# any machine: the maps are named beside this file rather than where they were",
        "# written, so this directory can be copied anywhere.",
        f"# Written in {at}.",
        f"# Isosurfaces are {level:g}x bulk; a pocket is a mask, drawn solid at "
        f"{POCKET_LEVEL:g}.",  # fmt: skip
        # as above: Tcl's info script is the file being sourced, and empty when VMD
        # was given -e instead, where the working directory is all there is to go on
        "set boonza_here [file dirname [file normalize [info script]]]",
        'if {[info script] ne ""} { cd $boonza_here }',
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
