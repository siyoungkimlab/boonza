"""The enclosed core of an fpocket pocket, after SiteMap's definition of site points.

fpocket's pocket is all the empty space its alpha spheres describe, including
the solvent-exposed fringe; SiteMap keeps only grid points that are outside the
protein, in contact with it and enclosed by it (Halgren 2009). Here a pocket is
trimmed to such points on one frame's beads, each bead with its own radius R_i
(sigma/2 of its type, data/cg_radii/<model>_bead_radii.csv; :func:`bead_radii`,
or :func:`boonza.sites.particle_radii` from a loaded topology):

  inside the pocket   within one of the pocket's alpha spheres
  outside the protein distance to every bead >= OUTSIDE * R_i  (SiteMap: d^2 >= 2.5 r^2)
  in contact          some bead within R_i + CONTACT A        (stands in for SiteMap's
                                                               vdW-contact test)
  enclosed            >= ENCLOSURE of N_RAYS rays meet a bead (within its R_i) inside
                      RAY_LENGTH A                            (SiteMap: 0.5 within 8 A)
  not isolated        >= MIN_NEIGHBOURS kept neighbours among the 18 face/edge
                      neighbours                              (SiteMap: >= 3 within 1.76 A)

on a SPACING grid. Each kept point stands for SPACING^3 of volume.
"""

from __future__ import annotations

import csv
from functools import cache
from pathlib import Path

import numpy as np

__all__ = ["bead_radii", "enclosed_core"]

#: per-bead-type LJ sizes of each force field (sigma, rmin and their halves, in A)
RADII = Path(__file__).resolve().parent.parent / "data" / "cg_radii"
SPACING = 2.0
OUTSIDE = np.sqrt(2.5)
CONTACT = 3.0
ENCLOSURE = 0.5
RAY_LENGTH = 8.0
N_RAYS = 60
MIN_NEIGHBOURS = 2


def bead_radii(types, model: str, rule: str = "sigma") -> np.ndarray:
    """Radius (A) of each bead from its type: sigma/2 (``rule="sigma"``) or rmin/2 of the
    type's self-interaction, read from data/cg_radii/<model>_bead_radii.csv. NaN for a
    type the table does not have.  With a loaded topology,
    :func:`boonza.sites.particle_radii` reads the same numbers off the force field."""
    table = _radii(model, rule)
    return np.array([table.get(str(t).strip(), np.nan) for t in types])


@cache
def _radii(model: str, rule: str) -> dict[str, float]:
    if rule not in ("sigma", "rmin"):
        raise ValueError(f"radius rule {rule!r}: 'sigma' or 'rmin'")
    column = {"sigma": "radius_sigma_A", "rmin": "radius_rmin_A"}[rule]
    path = RADII / f"{model}_bead_radii.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {r["type"]: float(r[column]) for r in csv.DictReader(fh)}


def _directions(n: int) -> np.ndarray:
    """n near-uniform unit vectors (Fibonacci sphere)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


def _per_radius(query, beads, bead_radii, reach: float):
    """(distance to the nearest bead, that group's radius) per query point and per group of
    beads of one radius: a bead's own radius decides each test, so beads are searched in
    groups of equal radius rather than as one set."""
    from ..spatial import min_dist2

    out = []
    for r in np.unique(bead_radii):
        d2 = min_dist2(query, beads[bead_radii == r], reach + r)
        out.append((np.sqrt(d2.astype(float)), float(r)))
    return out


def enclosed_core(centres, radii, beads, bead_radii, spacing: float = SPACING) -> np.ndarray:
    """Grid points (n, 3) of the enclosed core of one pocket (its alpha spheres' ``centres``
    and ``radii``) on one frame (``beads`` with per-bead ``bead_radii``)."""
    centres, radii, beads = (np.asarray(x, float).reshape(-1, k) for x, k in
                             ((centres, 3), (radii, 1), (beads, 3)))  # fmt: skip
    radii = radii.ravel()
    bead_radii = np.round(np.asarray(bead_radii, float), 3)
    if not len(centres) or not len(beads):
        return np.zeros((0, 3))
    lo, hi = (centres - radii[:, None]).min(0), (centres + radii[:, None]).max(0)
    axes = [np.arange(np.floor(a / spacing), np.ceil(b / spacing) + 1) * spacing
            for a, b in zip(lo, hi, strict=True)]  # fmt: skip
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inside = np.zeros(len(grid), bool)
    for k in range(0, len(centres), 256):  # inside some alpha sphere
        c, r = centres[k : k + 256], radii[k : k + 256]
        inside |= (((grid[:, None, :] - c[None]) ** 2).sum(-1) <= r**2).any(1)
    pts = grid[inside]
    if not len(pts):
        return pts
    outside = np.ones(len(pts), bool)
    contact = np.zeros(len(pts), bool)
    for d, r in _per_radius(pts, beads, bead_radii, CONTACT):
        outside &= d >= OUTSIDE * r
        contact |= d <= r + CONTACT
    pts = pts[outside & contact]
    if not len(pts):
        return pts
    dirs = _directions(N_RAYS)
    steps = np.arange(1.0, RAY_LENGTH + 0.5, 1.0)
    probes = pts[:, None, None, :] + dirs[None, :, None, :] * steps[None, None, :, None]
    probes = probes.reshape(-1, 3)
    hit = np.zeros(len(probes), bool)
    for d, r in _per_radius(probes, beads, bead_radii, 0.0):
        hit |= d <= r
    shut = hit.reshape(len(pts), N_RAYS, len(steps)).any(2).mean(1)
    pts = pts[shut >= ENCLOSURE]
    if len(pts) < 2:
        return pts
    from ..spatial import count_within

    neighbours = count_within(pts, pts, spacing * np.sqrt(2) + 1e-3) - 1
    return pts[neighbours >= MIN_NEIGHBOURS]
