"""Pharmacophore hotspots of a cosolvent run, and which pocket each lies in.

The probes' beads are typed into feature families by boonza
(pharmacophore.ligand_features: donor, acceptor, aromatic, hydrophobe,
cation, anion), each feature placed at the centre of its beads, in every
analysed frame -- the frames as traj.load_run returns them, whole and fitted.
boonza's own statistics then say where a family gathers: a map per family of
how much more often than bulk (the family's count spread over the box) each
1 A cell holds one, and its hotspots (pharmacophore.hotspots), regions at least
``enrichment`` times bulk of at least ``min_volume`` A^3, each with the number
of distinct probe molecules that put a feature there; only hotspots at least
MIN_PROBES distinct molecules make are kept.

A hotspot belongs to the one pocket that has a site point within NEAR of its
centre in the most frames (assign).  This is shown only: site finding and pocket
ranking never use the probes.
"""

from __future__ import annotations

import numpy as np

NEAR = 2.0  # A from a pocket's site points
ENRICHMENT = 20.0  # boonza.pharmacophore.hotspots' default
MIN_VOLUME = 3.0  # A^3, its default
SPACING = 1.0  # A, its maps' default
#: distinct probe molecules a hotspot needs: one molecule lingering is not statistics
MIN_PROBES = 3


def hotspots(system, pids, pcoords, enrichment: float = ENRICHMENT,
             min_volume: float = MIN_VOLUME, min_probes: int = MIN_PROBES) -> list:  # fmt: skip
    """boonza Hotspots of the probes (``pids``) over the frames ``pcoords`` that at least
    ``min_probes`` distinct probe molecules put a feature in."""
    from ..pharmacophore import FAMILIES, ligand_features
    from ..pharmacophore import hotspots as find
    from ..sites import Density

    at = {int(a): k for k, a in enumerate(pids)}
    copies = ligand_features(system, "chain LIG")
    places = {f: [] for f in FAMILIES}
    owners = {f: [] for f in FAMILIES}
    for copy, features in enumerate(copies):
        for fam, atoms in features:
            idx = [at[int(a)] for a in atoms]
            places[fam].append(pcoords[:, idx].mean(1))  # (frames, 3)
            owners[fam].append(np.full(len(pcoords), copy))
    volume = abs(float(np.linalg.det(np.asarray(system.cell, float))))
    points = {f: np.concatenate(v) if v else np.empty((0, 3)) for f, v in places.items()}
    every = np.concatenate([p for p in points.values() if len(p)])
    lo = every.min(0) - SPACING
    dims = np.maximum(np.ceil((every.max(0) + SPACING - lo) / SPACING), 1).astype(np.int64)
    maps = {}
    for fam, p in points.items():
        counts = np.zeros(int(dims.prod()), np.int64)
        if len(p):
            ijk = np.minimum(((p - lo) / SPACING).astype(np.int64), dims - 1)
            counts = np.bincount((ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2],
                                 minlength=int(dims.prod()))  # fmt: skip
        maps[fam] = Density(origin=lo, spacing=SPACING, counts=counts.reshape(dims),
                            expected=len(p) * SPACING**3 / max(volume, 1e-9))  # fmt: skip
        maps[fam].places = p
        maps[fam].owners = np.concatenate(owners[fam]) if owners[fam] else np.empty(0, int)
    return [h for h in find(maps, enrichment, min_volume) if h.ligands >= min_probes]


def assign(spots: list, pockets: list) -> dict:
    """{pocket index: [hotspots]}: each hotspot goes to the one pocket that is at it most
    often -- in the most frames with a site point within NEAR of its centre -- so two
    neighbouring pockets whose sites both wander near it do not both claim it."""
    from ..spatial import min_dist2

    out = {k: [] for k in range(len(pockets))}
    if not spots:
        return out
    centres = np.array([h.center for h in spots])
    frames = np.zeros((len(spots), len(pockets)), int)
    for k, p in enumerate(pockets):
        for s in p.per_frame_best().values():
            frames[:, k] += min_dist2(centres, s.xyz, NEAR) <= NEAR**2
    for j, h in enumerate(spots):
        if frames[j].max() > 0:
            out[int(np.argmax(frames[j]))].append(h)
    return out
