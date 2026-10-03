"""Cofactors a coarse-grained model has no residue for, held as inert beads.

A structural zinc, an ADP, a heme: neither Martini nor SIRAH has a residue for
any of them, and a structure carrying one cannot be mapped at all.  Left out, it
leaves a hole where probes then gather, and the pocket it was part of is measured
against a protein missing a piece of itself.

What the models are asked for here is only that the cofactor take up its room:
one uncharged, plainly apolar bead per heavy atom, held in shape and in place.  No
charge and no chemistry are claimed for it -- the phosphates of an ADP and the
+2 of a zinc are simply not represented -- which is honest while the cofactor is
buried, nothing being able to reach it and read as apolar what is really a charge.
:func:`warn_exposed` says when that does not hold.
"""

from __future__ import annotations

import warnings

import numpy as np

#: How close a cofactor's atom has to be to another residue's to be what holds
#: it, in angstroms.  A metal coordinates at 1.8 to 2.3 A, where nothing merely
#: touching comes nearer than 3, so the two are separated without any chemistry.
COORDINATION = 2.6
#: The mass a bead takes when it stands for one atom: that atom's own.
ATOM_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "NA": 22.990,
             "MG": 24.305, "P": 30.974, "S": 32.06, "CL": 35.45, "K": 39.098, "CA": 40.078,
             "MN": 54.938, "FE": 55.845, "CO": 58.933, "NI": 58.693, "CU": 63.546,
             "ZN": 65.38, "SE": 78.97, "BR": 79.904, "CD": 112.41, "I": 126.90}  # fmt: skip
#: A cofactor counts as buried when this many protein heavy atoms lie within
#: :data:`BURIAL_RADIUS` angstroms of it.  A coordinated metal has twenty or
#: more; one sitting in solvent has a handful.
BURIAL_RADIUS, BURIED_ENOUGH = 5.0, 8
#: How far a band may reach from a cofactor to the residues around it, in
#: angstroms.  Coordination alone does not hold a body -- a heme bonded through
#: its iron to one histidine can still swing about that bond -- so the residues
#: around it hold it too, and how far that reaches is the lever that matters.
#:
#: Measured on a heme over 5 ps: at 9 A it strays 2.3 A and its shape drifts 2.1;
#: at 12 A, 1.2 and 1.0; at 15 A, 0.9 and 0.6; at 20 A, 0.6 and 0.5, for 4751
#: bands against 1718.  Twelve is where the returns fall off, and it holds the
#: thing inside a bead's own radius while stiffening the protein around it least
#: -- the backbone there fluctuating 0.24 A against 0.30 at 9 A and 0.20 at 20.
COFACTOR_REACH = 12.0
#: How many beads of a cofactor are banded to the protein, and to how many
#: backbone beads each; ``None`` for every bead to every backbone bead in reach,
#: which is the default and what measurement says to do.
#:
#: Three points fix a rigid body, so a handful of bands ought to be enough -- and
#: it is not, because these bands are soft.  A heme held by nine of them wanders
#: 7.0 A over 5 ps, 4.4 A if they are stiffened to the limit 20 fs allows, and
#: 0.4 A when every bead is banded to the backbone within reach.  Many soft bands
#: beat a few stiff ones tenfold.  Many *stiff* bands are not an option: dense
#: and stiff together blew up on the first step.
COFACTOR_ANCHORS, COFACTOR_BANDS = None, None
#: How many of the nearest backbone beads each cofactor bead is tethered to;
#: ``None`` for every one within :data:`COFACTOR_REACH`, which is what holds.
#:
#: Measured on a heme over 5 ps, tethers per bead against how far it strayed and
#: how far its own shape drifted: one, 12.5 and 12.4 A; two, 11.8 and 11.8;
#: three, 6.7 and 6.7; all, 0.39 and 0.33.  The two columns move together because
#: a sparsely banded body does not stray so much as come apart, and tethers
#: cannot hold a shape that does not hold itself.
COFACTOR_TETHERS = None
#: How many of its nearest neighbours a cofactor bead is banded to inside the
#: cofactor; ``None``, the default, bands every pair within
#: :data:`COFACTOR_SHAPE_REACH`.
#:
#: Dense is not extravagance.  A body of n beads needs about 3n-6 independent
#: bands to be rigid -- 123 of them for a heme's 43 -- where two per bead gives
#: 56, and many of those are redundant along a ring.  Banded that sparsely the
#: heme does not stay a heme: its shape drifts as far as it strays, 12 A of each.
COFACTOR_NEIGHBOURS = None
#: How far a shape band reaches when counting neighbours is not used, in angstroms.
COFACTOR_SHAPE_REACH = 4.5


def shape_bands(xyz, neighbours: int | None = COFACTOR_NEIGHBOURS,
                reach: float = COFACTOR_SHAPE_REACH):  # fmt: skip
    """``(i, j, distance)`` for the bands that hold a cofactor's own shape.

    Each bead is banded to its ``neighbours`` nearest, which is how a molecule's
    own bonds run; with ``neighbours=None`` every pair within ``reach`` instead.
    Neither triangulates the body, so what really holds its shape is that every
    bead is also tethered to the protein.
    """
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    if len(xyz) < 2:
        return []
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
    np.fill_diagonal(d, np.inf)
    out = {}
    for i in range(len(xyz)):
        near = (np.argsort(d[i])[:neighbours] if neighbours
                else np.flatnonzero(d[i] <= reach))  # fmt: skip
        for j in near.tolist():
            out[(min(i, j), max(i, j))] = float(d[i, j])
    return [(i, j, v) for (i, j), v in sorted(out.items())]


def anchor_bands(cofactor, protein, backbone=None, reach: float = COFACTOR_REACH,
                 anchors: int | None = COFACTOR_ANCHORS,
                 bands: int | None = COFACTOR_BANDS):  # fmt: skip
    """``(cofactor bead, protein bead, distance)`` for the bands that hold a
    cofactor where the structure put it.

    Every bead is banded to the backbone beads within ``reach``, a backbone
    because one band per residue is enough and a cage on every side chain is a
    cage on the protein.  ``backbone`` is a boolean mask over ``protein``; without
    it any bead will do.  With ``anchors`` and ``bands``, only that many beads are
    banded and only to that many each, spread across the body rather than crowded
    on its nearest face -- which measurement says not to do, the bands being soft
    enough that a few of them hold nothing.
    """
    cofactor = np.asarray(cofactor, float).reshape(-1, 3)
    protein = np.asarray(protein, float).reshape(-1, 3)
    if not len(cofactor) or not len(protein):
        return []
    pick = np.flatnonzero(backbone) if backbone is not None and np.any(backbone) \
        else np.arange(len(protein))  # fmt: skip
    d = np.linalg.norm(cofactor[:, None] - protein[None, pick], axis=2)
    # the anchors have to span the body, not crowd its nearest face: three beads
    # along one edge of a porphyrin leave it free to rock about that edge, so the
    # first anchor is the closest and each next is the one furthest from those
    # already taken
    reachable = np.flatnonzero(d.min(1) <= reach)
    if not len(reachable):
        return []
    if anchors is None:  # every bead is tethered, which is what holds the body
        out = []
        for i in reachable.tolist():
            near = np.flatnonzero(d[i] <= reach)
            if bands:  # its nearest few, rather than everything in reach
                near = near[np.argsort(d[i][near])[:bands]]
            out += [(int(i), int(pick[j]), float(d[i, j])) for j in near.tolist()]
        return out
    order = [int(reachable[np.argmin(d.min(1)[reachable])])]
    while len(order) < min(anchors, len(reachable)):
        away = np.linalg.norm(cofactor[reachable][:, None] - cofactor[order][None], axis=2)
        order.append(int(reachable[int(np.argmax(away.min(1)))]))
        if len(set(order)) < len(order):  # nothing further to take
            order = list(dict.fromkeys(order))
            break
    out = []
    for i in order:
        near = np.argsort(d[i])[:bands] if bands else np.argsort(d[i])
        out += [(int(i), int(pick[j]), float(d[i, j])) for j in near.tolist()
                if d[i, j] <= reach]  # fmt: skip
    return out


#: The force constant of those bands, kJ/mol/nm^2, as the elastic network's.
#: Stiffer is not better: at 20 fs a band of 70000 on a carbon bead has a period
#: of 82 fs, which is four steps and unstable.
COFACTOR_FC = 700.0


def coordination_pairs(a, b, cutoff: float = COORDINATION):
    """``(index in a, index in b)`` for every pair closer than ``cutoff`` angstroms."""
    a, b = np.asarray(a, float).reshape(-1, 3), np.asarray(b, float).reshape(-1, 3)
    if not len(a) or not len(b):
        return []

    out = []
    for i, x in enumerate(a):
        d = np.linalg.norm(b - x, axis=1)
        out += [(i, int(j)) for j in np.flatnonzero(d <= cutoff).tolist()]
    return out


def warn_exposed(system, named_positions, radius: float = BURIAL_RADIUS,
                 enough: int = BURIED_ENOUGH, stacklevel: int = 3) -> None:  # fmt: skip
    """Warn for each cofactor that is not buried: ``(label, positions in A)``.

    The inert bead is honest where the cofactor cannot be reached.  One on the
    surface can be, and a probe will settle on an apolar bead where a phosphate
    or a charge belongs -- a wrong answer rather than a missing one, which is why
    this is said and not assumed.
    """
    from .spatial import min_dist2

    heavy = system.select("protein and not element H").ids
    if not len(heavy):
        return
    xyz = np.asarray(system.positions)[heavy]
    for label, here in named_positions:
        here = np.asarray(here, float).reshape(-1, 3)
        near = int((min_dist2(xyz, here, radius, cell=None) <= radius**2).sum())
        if near < enough:
            warnings.warn(f"{label} has {near} protein heavy atoms within {radius:g} A, so it is "
                          "not buried: an inert bead there is something a probe can reach and "
                          "read as apolar.  Give it parameters of its own, or leave it out",
                          stacklevel=stacklevel)  # fmt: skip
