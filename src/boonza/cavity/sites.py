"""Sites from a precomputed grid: SiteMap's site finding and site properties (Halgren 2009).

Site finding, each step a threshold on grid.py's numbers or a grouping:

  outside     d^2 / R_i^2 >= outside for every bead                   (SiteMap 2.5)
  enclosed    >= enclosure of the rays enter a bead within ray_length  (0.5 within 8 A)
  neighbours  >= neighbours other candidates within neighbour_radius
              grid steps                                              (3 within 1.76 A
                                                                       on a 1 A grid)
  groups      connected by SiteMap's step rule: sum |di| <= 3 and
              sum di^2 <= 5 (grid steps); groups below min_points dropped
  merging     two groups whose closest points lie within merge_gap A
              across outside space, while one of them is no larger than
              merge_cap (A^3)                                        (6.5 A; 100 points)
  sites       the largest max_sites groups; beyond the first, at least
              min_volume A^3                                         (10; 15 points)

Volumes are in A^3 so that a site's size does not depend on the grid: on
SiteMap's 1 A grid a site point is 1 A^3.

SiteMap's third test, a probe's contact energy at the point, is left out: no preset
used it (it was switched off in volume200), and computing it took most of a frame's
time on large proteins.

Site properties, only from numbers site finding computed anyway:

  n               site volume, A^3 (SiteMap's number of site points)
  enclosure_site  share of the site points' rays entering a bead within 10 A

(SiteMap's exposure and its enclosure over the site and its extension points, outside
points around it, are not computed: nothing uses them.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import NO_HIT, Grid

DEFAULTS = {
    "outside": 2.5,
    "ray_length": 8.0,
    "enclosure": 0.5,
    "neighbour_radius": 1.76,
    "neighbours": 3,
    "min_points": 3,
    "merge_gap": 6.5,
    "merge_cap": 100.0,
    "max_sites": 10,
    "min_volume": 15.0,
}
PROPERTY_RAY = 10.0  # A


@dataclass
class Site:
    points: np.ndarray  #: grid-point indices into the Grid
    xyz: np.ndarray  #: (n, 3)
    props: dict = field(default_factory=dict)

    @property
    def centre(self) -> np.ndarray:
        return self.xyz.mean(0)


def _offsets(rule) -> np.ndarray:
    r = np.arange(-2, 3)
    off = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    off = off[np.any(off != 0, axis=1)]
    return off[[rule(o) for o in off]]


LINK = _offsets(lambda o: np.abs(o).sum() <= 3 and (o * o).sum() <= 5)


def _keys(ijk: np.ndarray, dims: np.ndarray) -> np.ndarray:
    return (ijk[:, 0].astype(np.int64) * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]


def _pairs(ijk: np.ndarray, offsets: np.ndarray):
    """(i, j) of points of ``ijk`` that differ by one of ``offsets``."""
    dims = ijk.max(0) + 5
    base = ijk + 2
    keys = _keys(base, dims)
    order = np.argsort(keys)
    sk = keys[order]
    bi, bj = [], []
    for o in offsets:
        if tuple(o) <= (0, 0, 0):  # each pair once
            continue
        q = _keys(base + o, dims)
        pos = np.searchsorted(sk, q)
        pos[pos >= len(sk)] = 0
        ok = sk[pos] == q
        bi.append(np.flatnonzero(ok))
        bj.append(order[pos[ok]])
    if not bi:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(bi), np.concatenate(bj)


def _shut(hit: np.ndarray, length: float, step: float) -> np.ndarray:
    """Share of each point's rays that enter a bead within ``length``, sampled every ``step``."""
    return ((hit != NO_HIT) & (hit * step <= length + 1e-6)).mean(1)


def candidates(g: Grid, p: dict) -> np.ndarray:
    """Indices of the grid points that pass the outside, enclosure and neighbour tests."""
    ok = (g.ratio >= p["outside"]) & (_shut(g.hit, p["ray_length"], g.ray_step) >= p["enclosure"])
    idx = np.flatnonzero(ok)
    if not len(idx):
        return idx
    near = _offsets(lambda o: (o * o).sum() <= p["neighbour_radius"] ** 2 + 1e-9)
    bi, bj = _pairs(g.ijk[idx], near)
    count = np.bincount(np.r_[bi, bj], minlength=len(idx))
    return idx[count >= p["neighbours"]]


def _outside_lookup(g: Grid, outside: float):
    dims = g.ijk.max(0) + 3
    keys = _keys(g.ijk + 1, dims)
    good = np.sort(keys[g.ratio >= outside])

    def is_out(xyz):
        ijk = np.rint((xyz - g.origin) / g.spacing).astype(np.int64) + 1
        if (ijk < 0).any() or (ijk >= dims).any():
            return False
        k = _keys(ijk, dims)
        pos = np.searchsorted(good, k)
        pos[pos >= len(good)] = 0
        return bool((good[pos] == k).all())

    return is_out


def find_sites(g: Grid, params: dict | None = None) -> list[Site]:
    """The sites of one grid, largest first, with their properties."""
    from ..graph import connected_components

    p = {**DEFAULTS, **(params or {})}
    idx = candidates(g, p)
    if not len(idx):
        return []
    bi, bj = _pairs(g.ijk[idx], LINK)
    labels, n = connected_components(len(idx), bi, bj)
    groups = [idx[labels == k] for k in range(n)]
    groups = [x for x in groups if len(x) >= p["min_points"]]
    groups = _merge(g, groups, p)
    v = g.spacing**3
    groups.sort(key=len, reverse=True)
    kept = [x for k, x in enumerate(groups) if k == 0 or len(x) * v >= p["min_volume"]]
    sites = [Site(x, g.xyz[x]) for x in kept[: int(p["max_sites"])]]
    for s in sites:
        s.props = properties(g, s, p)
    return sites


def _merge(g: Grid, groups: list, p: dict) -> list:
    """Merge groups across exposed gaps of at most merge_gap, SiteMap's last step."""
    if len(groups) < 2:
        return groups
    xyz = g.xyz
    is_out = _outside_lookup(g, p["outside"])
    v = g.spacing**3
    groups = [np.asarray(x) for x in groups]
    while True:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if min(len(groups[a]), len(groups[b])) * v > p["merge_cap"]:
                    continue
                pa, pb = xyz[groups[a]], xyz[groups[b]]
                if (np.abs(pa.mean(0) - pb.mean(0)) > p["merge_gap"] + 60).any():
                    continue
                d2 = ((pa[:, None] - pb[None]) ** 2).sum(-1)
                i, j = np.unravel_index(d2.argmin(), d2.shape)
                gap = float(np.sqrt(d2[i, j]))
                if gap > p["merge_gap"] or (best is not None and gap >= best[0]):
                    continue
                steps = max(int(np.ceil(gap / (g.spacing / 2))), 1)
                line = pa[i] + (pb[j] - pa[i]) * np.linspace(0, 1, steps + 1)[:, None]
                if is_out(line):
                    best = (gap, a, b)
        if best is None:
            return groups
        _, a, b = best
        groups[a] = np.concatenate([groups[a], groups[b]])
        del groups[b]


def properties(g: Grid, site: Site, p: dict) -> dict:
    """A site's properties (see the module docstring)."""
    pts = site.points
    shut = _shut(g.hit[pts], PROPERTY_RAY, g.ray_step)
    return {"n": len(pts) * g.spacing**3, "enclosure_site": float(shut.mean())}
