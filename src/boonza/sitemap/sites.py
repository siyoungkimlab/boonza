"""Sites from a precomputed grid: SiteMap's site finding and site properties (Halgren 2009).

Site finding, each step a threshold on grid.py's numbers or a grouping:

  outside     d^2 / R_i^2 >= outside for every bead                   (SiteMap 2.5)
  enclosed    >= enclosure of the rays enter a bead within ray_length  (0.5 within 8 A)
  contact     probe Lennard-Jones energy <= contact (kcal/mol)        (-1.1 for atoms)
              (contact_form full, or capped: each pair at most 0)
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

Site properties, over the site points and SiteMap's extension points (outside
points within 3 A in x, y and z of a site point that contact the protein or lie
>= 4 A from it):

  n           site volume, A^3 (SiteMap's number of site points)
  exposure    extension / (site + extension)
  enclosure   share of rays entering a bead within 10 A
  contact     mean -E(probe)
  philic      mean -E(probe, polar beads)
  phobic      mean -E(probe, apolar beads) x share of rays entering a bead within 6 A
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import NO_HIT, RAY_STEP, Grid

DEFAULTS = {
    "outside": 2.5,
    "ray_length": 8.0,
    "enclosure": 0.5,
    "probe": 0,
    "contact": -2.0,
    "contact_form": "full",
    "neighbour_radius": 1.76,
    "neighbours": 3,
    "min_points": 3,
    "merge_gap": 6.5,
    "merge_cap": 100.0,
    "max_sites": 10,
    "min_volume": 15.0,
}
EXTENSION_BOX = 3.0  # A
EXTENSION_FAR = 4.0  # A from the nearest bead centre
PROPERTY_RAY = 10.0  # A
PHOBIC_RAY = 6.0  # A


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


def _shut(hit: np.ndarray, length: float) -> np.ndarray:
    """Share of each point's rays that enter a bead within ``length``."""
    return ((hit != NO_HIT) & (hit * RAY_STEP <= length + 1e-6)).mean(1)


def candidates(g: Grid, p: dict) -> np.ndarray:
    """Indices of the grid points that pass the outside, enclosure, contact and neighbour tests."""
    e = g.energy(p["probe"], p["contact_form"])
    ok = (g.ratio >= p["outside"]) & (_shut(g.hit, p["ray_length"]) >= p["enclosure"])
    ok &= e <= p["contact"]
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
    """SiteMap's site properties (see the module docstring)."""
    pts = site.points
    v = g.spacing**3
    box = int(np.floor(EXTENSION_BOX / g.spacing))
    e = g.energy(p["probe"], p["contact_form"])
    ext = np.zeros(0, int)
    if box >= 1:
        r = np.arange(-box, box + 1)
        off = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
        dims = g.ijk.max(0) + 2 * box + 3
        keys = _keys(g.ijk + box + 1, dims)
        order = np.argsort(keys)
        sk = keys[order]
        near = set()
        base = g.ijk[pts] + box + 1
        for o in off:
            q = _keys(base + o, dims)
            pos = np.searchsorted(sk, q)
            pos[pos >= len(sk)] = 0
            near.update(order[pos[sk[pos] == q]].tolist())
        cand = np.array(sorted(near - set(pts.tolist())), int)
        if len(cand):
            ok = (g.ratio[cand] >= p["outside"]) & ((e[cand] < 0) | (g.dmin[cand] >= EXTENSION_FAR))
            ext = cand[ok]
    both = np.r_[pts, ext]
    shut6 = _shut(g.hit[both], PHOBIC_RAY)
    return {
        "n": len(pts) * v,
        "exposure": len(ext) / len(both),
        "enclosure": float(_shut(g.hit[both], PROPERTY_RAY).mean()),
        "contact": float(-e[both].mean()),
        "philic": float(-g.energy(p["probe"], p["contact_form"], "polar")[both].mean()),
        "phobic": float(
            (-g.energy(p["probe"], p["contact_form"], "apolar")[both] * shut6).mean()
        ),  # fmt: skip
    }


def sitescore(props: dict, philic_mean: float = 1.0, n_cap: float = 100.0,
              weights=(0.0733, 0.6688, -0.20)) -> float:  # fmt: skip
    """SiteScore (Halgren 2009, eq 3): a sqrt(min(n, cap)) + b e + c min(p, 1), with the
    hydrophilic score p calibrated by ``philic_mean`` (SiteMap: the mean over known sites)."""
    a, b, c = weights
    philic = min(props["philic"] / philic_mean, 1.0)
    return a * np.sqrt(min(props["n"], n_cap)) + b * props["enclosure"] + c * philic
