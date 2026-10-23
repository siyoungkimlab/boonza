"""What every grid point around a protein needs, computed once.

SiteMap's site finding is a chain of thresholds on per-point numbers -- how far
outside the protein a point is and how enclosed it is -- followed by grouping.
(SiteMap's third number, a probe's contact energy, is not computed: no preset
uses it.)  The numbers do not depend on the
thresholds, so they are computed here once per structure and grid spacing, and
a parameter trial (sites.py) only compares and groups:

  ratio       min over beads of d^2 / R_i^2 (SiteMap's "outside": >= 2.5)
  dmin        distance to the nearest bead centre, A
  hit         per ray (RAYS directions), the distance at which it first enters a
              bead (within R_i of its centre), in ray_step steps; NO_HIT if not
              within RAY_MAX

Only points outside every bead (ratio >= 1) and within KEEP of a bead centre
are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .beads import Beads

RAYS = 110  # SiteMap: 110 evenly spaced directions
RAY_STEP = 1.0  # A
RAY_MAX = 10.0  # A: SiteMap's enclosure property reaches 10 A, its filter 8 A
NO_HIT = 255
KEEP = 10.0  # A from a bead centre: the grid points kept


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
    hit: np.ndarray  #: (n, RAYS) uint8, in ray_step units
    ray_step: float = RAY_STEP  #: A between the samples along a ray

    @property
    def xyz(self) -> np.ndarray:
        return self.origin + self.ijk * self.spacing

    def save(self, path) -> None:
        """Write to ``path`` (.npz) whole or not at all: a crash mid-write leaves the old file."""
        path = Path(path)
        tmp = path.with_name(path.stem + ".partial.npz")
        np.savez_compressed(tmp, spacing=self.spacing, origin=self.origin, ijk=self.ijk,
                            ratio=self.ratio, dmin=self.dmin, hit=self.hit,
                            ray_step=self.ray_step)  # fmt: skip
        tmp.replace(path)

    @classmethod
    def load(cls, path) -> Grid:
        """A saved grid; one saved with contact energies (before they were dropped) loads
        without them."""
        z = np.load(path)
        return cls(float(z["spacing"]), z["origin"], z["ijk"], z["ratio"], z["dmin"], z["hit"],
                   float(z["ray_step"]) if "ray_step" in z else RAY_STEP)  # fmt: skip


def _by_radius(beads: Beads):
    r = np.round(beads.radius, 3)
    return [(float(v), beads.xyz[r == v]) for v in np.unique(r)]


def compute(
    beads: Beads,
    spacing: float = 2.0,
    outside: float = 1.0,
    ray_step: float = RAY_STEP,
) -> Grid:
    """The per-point quantities around ``beads`` on a ``spacing`` grid.

    ``outside``: keep only points with ratio >= it.  Site finding uses no point below
    its own ``outside`` parameter, so a run with fixed parameters (a trajectory) passes
    that and skips the rays of points it would never look at.  ``ray_step``:
    A between the samples along each ray, out to RAY_MAX."""
    from ..spatial import min_dist2

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
    hit = _rays(xyz, groups, step=ray_step)
    return Grid(spacing, origin, ijk, ratio.astype(np.float32), dmin.astype(np.float32), hit,
                ray_step=ray_step)  # fmt: skip


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


def _rays(xyz: np.ndarray, groups, chunk: int = 2000, step: float = RAY_STEP) -> np.ndarray:
    """First ray step (``step`` units) at which each ray enters a bead; NO_HIT if none.
    Identical to _rays_direct: the samples are looked up in an _Occupancy lattice, and
    each ray is marched one step at a time only until it first enters a bead."""
    occ = _Occupancy(groups)
    dirs = directions()
    steps = np.arange(1, int(round(RAY_MAX / step)) + 1) * step
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


def _rays_direct(xyz: np.ndarray, groups, chunk: int = 2000, step: float = RAY_STEP) -> np.ndarray:
    """_rays by a nearest-bead search for every sample (the reference it must equal)."""
    from ..spatial import min_dist2

    dirs = directions()
    steps = np.arange(1, int(round(RAY_MAX / step)) + 1) * step
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
