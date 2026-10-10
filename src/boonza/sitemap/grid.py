"""What every grid point around a protein needs, computed once.

SiteMap's site finding is a chain of thresholds on per-point numbers -- how far
outside the protein a point is, how enclosed it is, how well a probe there
contacts the protein -- followed by grouping.  The numbers do not depend on the
thresholds, so they are computed here once per structure and grid spacing, and
a parameter trial (sites.py) only compares and groups:

  ratio       min over beads of d^2 / R_i^2 (SiteMap's "outside": >= 2.5)
  dmin        distance to the nearest bead centre, A
  hit         per ray (RAYS directions), the distance at which it first enters a
              bead (within R_i of its centre), in RAY_STEP steps; NO_HIT if not
              within RAY_MAX
  e_polar,    Lennard-Jones energy (kcal/mol) of a probe bead at the point with
  e_apolar    the protein's polar and apolar beads, per probe type, from the
              force field's pair table (NBFIX for Martini), cut at LJ_CUTOFF
  c_polar,    the same with each pair's energy capped at 0: a bead closer than
  c_apolar    the pair's minimum counts as touching rather than clashing, which
              matters where a ligand atom fits but a bead would not

Only points outside every bead (ratio >= 1) and within KEEP of a bead centre
are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .beads import Beads
from .forcefield import pair_table

RAYS = 110  # SiteMap: 110 evenly spaced directions
RAY_STEP = 1.0  # A
RAY_MAX = 10.0  # A: SiteMap's enclosure property reaches 10 A, its filter 8 A
NO_HIT = 255
LJ_CUTOFF = 12.0  # A: Martini's 1.1 nm, rounded up
KEEP = 10.0  # A from a bead centre: site points and SiteMap's extension points
#: probe bead types tried for the contact energy, per model.  New ones go at the end, so a
#: probe's index stays the same in grids and parameter sets made before.  The last ones are
#: each model's smallest (pair sigma ~3.65 A in Martini 3, 3.2-3.3 A in SIRAH against
#: protein beads); Martini 2 has none smaller than C1 (4.7 A against every protein bead).
PROBES = {"martini2": ("C1", "N0", "P1"), "martini3": ("TC3", "SC3", "C3", "N1", "TC5", "TN1"),
          "sirah": ("Y2Ca", "Y1C", "GC", "GO", "GN")}  # fmt: skip


def directions(n: int = RAYS) -> np.ndarray:
    """n near-uniform unit vectors (Fibonacci sphere)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


@dataclass
class Grid:
    spacing: float
    origin: np.ndarray  #: (3,)
    ijk: np.ndarray  #: (n, 3) int32 grid indices of the kept points
    ratio: np.ndarray  #: (n,)
    dmin: np.ndarray  #: (n,)
    hit: np.ndarray  #: (n, RAYS) uint8, in RAY_STEP units
    probes: tuple  #: probe type names
    e_polar: np.ndarray  #: (nprobe, n)
    e_apolar: np.ndarray  #: (nprobe, n)
    c_polar: np.ndarray  #: (nprobe, n), each pair capped at 0
    c_apolar: np.ndarray  #: (nprobe, n)

    def energy(self, probe: int, form: str = "full", part: str = "both") -> np.ndarray:
        """The contact energy of probe ``probe``: ``form`` full or capped, ``part`` both,
        polar or apolar beads."""
        full = form == "full"
        pol, apo = (self.e_polar, self.e_apolar) if full else (self.c_polar, self.c_apolar)
        if part == "polar":
            return pol[probe]
        if part == "apolar":
            return apo[probe]
        return pol[probe] + apo[probe]

    @property
    def xyz(self) -> np.ndarray:
        return self.origin + self.ijk * self.spacing

    def save(self, path) -> None:
        """Write to ``path`` (.npz) whole or not at all: a crash mid-write leaves the old file."""
        path = Path(path)
        tmp = path.with_name(path.stem + ".partial.npz")
        np.savez_compressed(tmp, spacing=self.spacing, origin=self.origin, ijk=self.ijk,
                            ratio=self.ratio, dmin=self.dmin, hit=self.hit,
                            probes=np.array(self.probes), e_polar=self.e_polar,
                            e_apolar=self.e_apolar, c_polar=self.c_polar,
                            c_apolar=self.c_apolar)  # fmt: skip
        tmp.replace(path)

    @classmethod
    def load(cls, path) -> Grid:
        z = np.load(path)
        return cls(float(z["spacing"]), z["origin"], z["ijk"], z["ratio"], z["dmin"], z["hit"],
                   tuple(str(p) for p in z["probes"]), z["e_polar"], z["e_apolar"],
                   z["c_polar"], z["c_apolar"])  # fmt: skip


def add_probes(g: Grid, beads: Beads, probes) -> Grid:
    """``g`` with the contact energies of the ``probes`` it lacks appended."""
    new = tuple(p for p in probes if p not in g.probes)
    if not new:
        return g
    energies = _contact(g.xyz, beads, new)
    old = (g.e_polar, g.e_apolar, g.c_polar, g.c_apolar)
    return Grid(
        g.spacing,
        g.origin,
        g.ijk,
        g.ratio,
        g.dmin,
        g.hit,
        g.probes + new,
        *(np.concatenate([a, b]) for a, b in zip(old, energies, strict=True)),
    )


def _by_radius(beads: Beads):
    r = np.round(beads.radius, 3)
    return [(float(v), beads.xyz[r == v]) for v in np.unique(r)]


def compute(beads: Beads, spacing: float = 2.0, probes=None, outside: float = 1.0) -> Grid:
    """The per-point quantities around ``beads`` on a ``spacing`` grid.

    ``outside``: keep only points with ratio >= it.  Site finding uses no point below
    its own ``outside`` parameter, so a run with fixed parameters (a trajectory) passes
    that and skips the rays and energies of points it would never look at."""
    from ..spatial import min_dist2

    probes = tuple(probes or PROBES[beads.model])
    lo = beads.xyz.min(0) - KEEP
    hi = beads.xyz.max(0) + KEEP
    origin = np.floor(lo / spacing) * spacing
    dims = np.ceil((hi - origin) / spacing).astype(int) + 1
    ijk = np.stack(np.meshgrid(*[np.arange(d) for d in dims], indexing="ij"), -1).reshape(-1, 3)
    xyz = origin + ijk * spacing
    groups = _by_radius(beads)
    ratio = np.full(len(xyz), np.inf)
    dmin = np.full(len(xyz), np.inf)
    for r, pts in groups:
        d2 = min_dist2(xyz, pts, KEEP + r).astype(float)
        ratio = np.minimum(ratio, d2 / r**2)
        dmin = np.minimum(dmin, np.sqrt(d2))
    keep = (ratio >= max(outside, 1.0)) & (dmin <= KEEP)
    ijk, xyz, ratio, dmin = ijk[keep].astype(np.int32), xyz[keep], ratio[keep], dmin[keep]
    hit = _rays(xyz, groups)
    energies = _contact(xyz, beads, probes)
    return Grid(spacing, origin, ijk, ratio.astype(np.float32), dmin.astype(np.float32), hit,
                probes, *energies)  # fmt: skip


CELL = 0.5  # A: the occupancy lattice _rays looks ray samples up in
#: A of slack against float32 rounding in the inside test
SLACK = 0.01


class _Occupancy:
    """Which bead-size groups can contain a point, from a CELL lattice built once.

    The inside test is boonza's ``min_dist2(q, pts, r + 0.01, first_hit=True) <= r^2``
    per group of beads of radius r.  It can stop at a bead between r and r + 0.01
    of q and so call q outside where another bead holds it, so a lattice cell is
    settled for a group only when no bead of the group comes near that band
    anywhere in the cell:

      deep      some bead holds the whole cell and none has its band in it: inside
      boundary  a bead's surface band may cross the cell: asked of min_dist2 itself
      neither   no bead within r + 0.01 of any point of the cell: not inside

    So each answer is the one min_dist2 gives, and only the few samples in a
    boundary cell (of a group no deep cell already settles) are searched.
    """

    def __init__(self, groups, chunk: int = 400):
        half = np.sqrt(3) / 2 * CELL  # from a cell's centre to its corners
        rmax = max(r for r, _ in groups)
        allpts = np.vstack([p for _, p in groups])
        margin = rmax + 2.0  # holds each bead's cube of cells below
        self.lo = allpts.min(0) - margin
        self.dims = np.ceil((allpts.max(0) + margin - self.lo) / CELL).astype(int) + 1
        if len(groups) > 32:
            raise ValueError("more than 32 bead sizes")
        size = int(np.prod(self.dims))
        self.deep = np.zeros(size, bool)
        self.boundary = np.zeros(size, np.uint32)  # bit g: group g is unsettled here
        self.groups = groups
        for g, (r, pts) in enumerate(groups):
            inner, outer = r - half - SLACK, r + 0.01 + half + SLACK
            n = int(np.ceil(outer / CELL)) + 1
            ax = np.arange(-n, n + 1)
            off = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
            deep = np.zeros(size, bool)
            for a in range(0, len(pts), chunk):
                p = pts[a : a + chunk]
                cells = np.floor((p - self.lo) / CELL).astype(np.int64)[:, None] + off[None]
                d = np.linalg.norm(self.lo + (cells + 0.5) * CELL - p[:, None], axis=2)
                flat = np.ravel_multi_index(cells.reshape(-1, 3).T, self.dims)
                d = d.ravel()
                deep[flat[d <= inner]] = True
                self.boundary[flat[(d > inner) & (d <= outer)]] |= np.uint32(1 << g)
            unsettled = ((self.boundary >> np.uint32(g)) & 1).astype(bool)
            self.deep |= deep & ~unsettled

    def inside(self, q: np.ndarray) -> np.ndarray:
        from ..spatial import min_dist2

        cells = np.floor((q - self.lo) / CELL).astype(np.int64)
        ok = ((cells >= 0) & (cells < self.dims)).all(1)
        flat = np.zeros(len(q), np.int64)
        flat[ok] = np.ravel_multi_index(cells[ok].T, self.dims)
        out = np.zeros(len(q), bool)
        out[ok] = self.deep[flat[ok]]
        mask = np.zeros(len(q), np.uint32)
        mask[ok] = self.boundary[flat[ok]]
        for g, (r, pts) in enumerate(self.groups):
            ask = np.flatnonzero(~out & (((mask >> np.uint32(g)) & 1) == 1))
            if len(ask):
                out[ask] = min_dist2(q[ask], pts, r + 0.01, first_hit=True) <= r * r
        return out


def _rays(xyz: np.ndarray, groups, chunk: int = 2000) -> np.ndarray:
    """First ray step (RAY_STEP units) at which each ray enters a bead; NO_HIT if none.
    Identical to _rays_direct: the samples are looked up in an _Occupancy lattice, and
    each ray is marched one step at a time only until it first enters a bead."""
    occ = _Occupancy(groups)
    dirs = directions()
    steps = np.arange(1, int(RAY_MAX / RAY_STEP) + 1) * RAY_STEP
    out = np.full((len(xyz), RAYS), NO_HIT, np.uint8)
    for a in range(0, len(xyz), chunk):
        p = xyz[a : a + chunk]
        first = np.full(len(p) * RAYS, NO_HIT, np.uint8)
        start = np.repeat(p, RAYS, axis=0)
        way = np.tile(dirs, (len(p), 1))
        live = np.arange(len(first))
        for k, step in enumerate(steps, 1):
            ins = occ.inside(start[live] + way[live] * step)  # as _rays_direct's samples
            first[live[ins]] = k
            live = live[~ins]
            if not len(live):
                break
        out[a : a + chunk] = first.reshape(len(p), RAYS)
    return out


def _rays_direct(xyz: np.ndarray, groups, chunk: int = 2000) -> np.ndarray:
    """_rays by a nearest-bead search for every sample (the reference it must equal)."""
    from ..spatial import min_dist2

    dirs = directions()
    steps = np.arange(1, int(RAY_MAX / RAY_STEP) + 1) * RAY_STEP
    out = np.full((len(xyz), RAYS), NO_HIT, np.uint8)
    for a in range(0, len(xyz), chunk):
        p = xyz[a : a + chunk]
        probe = p[:, None, None, :] + dirs[None, :, None, :] * steps[None, None, :, None]
        probe = probe.reshape(-1, 3)
        inside = np.zeros(len(probe), bool)
        for r, pts in groups:
            inside |= min_dist2(probe, pts, r + 0.01, first_hit=True) <= r * r
        inside = inside.reshape(len(p), RAYS, len(steps))
        first = np.where(inside.any(2), inside.argmax(2) + 1, NO_HIT)
        out[a : a + chunk] = first
    return out


def _contact(xyz: np.ndarray, beads: Beads, probes, chunk: int = 1500):
    """Probe-protein Lennard-Jones energies split by polar and apolar beads: full, then
    with each pair capped at 0."""
    table = pair_table(beads.model)
    pk = table.index(probes)
    sig = table.sigma[pk][:, beads.type_index]  # (nprobe, nbeads)
    eps = table.epsilon[pk][:, beads.type_index]
    out = [np.zeros((len(probes), len(xyz)), np.float32) for _ in range(4)]
    polar = beads.polar
    for a in range(0, len(xyz), chunk):
        d2 = ((xyz[a : a + chunk, None, :] - beads.xyz[None]) ** 2).sum(-1)  # (m, nbeads)
        far = d2 > LJ_CUTOFF**2
        for k in range(len(probes)):
            s6 = (sig[k] ** 2 / d2) ** 3
            lj = 4 * eps[k] * (s6 * s6 - s6)
            lj[far] = 0.0
            capped = np.minimum(lj, 0.0)
            for q, (e, m) in enumerate(((lj, polar), (lj, ~polar), (capped, polar),
                                        (capped, ~polar))):  # fmt: skip
                out[q][k, a : a + chunk] = e[:, m].sum(1)
    return out
