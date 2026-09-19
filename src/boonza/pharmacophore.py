"""What the pocket wants where: donor, acceptor, aromatic or greasy, as maps.

    maps = boonza.feature_maps(s, runs, ligand="chain LIG")
    spots = boonza.hotspots(maps)
    boonza.write_hotspots("hotspots.pdb", spots)
    maps["Acceptor"].write_dx("acceptor.dx")

:func:`boonza.sites` says where a ligand goes and
:func:`boonza.interaction_fingerprints` says what it touches.  Neither says
what to *put* there, which is the question a designer asks.  These maps
answer it by counting not atoms but the kind of thing each atom is: where
donors gather, a donor is wanted.

The peaks of those maps are a pharmacophore taken from the simulation rather
than from a handful of crystal structures, and they come with the evidence
that made them -- how many times more often than bulk, over how many runs and
how many distinct molecules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .align import kabsch
from .pbc import minimum_image
from .sites import Density, _join_neighbours, _pairs
from .symmetry import DEFAULT_LIGAND, _boxed_blocks, _ids

FAMILIES = ("Donor", "Acceptor", "Aromatic", "Hydrophobe", "PosIonizable", "NegIonizable")
CODES = {"Donor": "DON", "Acceptor": "ACC", "Aromatic": "ARO", "Hydrophobe": "HYD",
         "PosIonizable": "CAT", "NegIonizable": "ANI"}  # fmt: skip


@dataclass
class Hotspot:
    """One place a kind of atom gathers, and what says so."""

    family: str
    center: np.ndarray  # (3,) in the reference's frame
    enrichment: float  # times more often than bulk would explain, at the peak
    volume: float  # A^3 of the region above the threshold
    points: int  # feature placements inside it
    ligands: int  # distinct molecules that put one there

    @property
    def radius(self) -> float:
        """The radius of a ball of the same volume: a tolerance to design against."""
        return float((3.0 * self.volume / (4.0 * np.pi)) ** (1.0 / 3.0))


def ligand_features(system, ligand: str = DEFAULT_LIGAND, families=FAMILIES) -> list[list[tuple]]:
    """Per ligand copy, ``[(family, atom indices), ...]`` as RDKit types them.

    A feature is placed at the centre of its atoms, so an aromatic ring counts
    once at the middle of the ring rather than six times around it.
    ``ZnBinder`` and ``LumpedHydrophobe`` are left out: the first is a special
    case and the second repeats what ``Hydrophobe`` already says.
    """
    import os

    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures

    from .chem import to_rdkit

    factory = ChemicalFeatures.BuildFeatureFactory(
        os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    )
    lig = _ids(system, ligand)
    frag = np.asarray(system.fragids)[lig]
    out = []
    for f in np.unique(frag):
        atoms = lig[frag == f]
        mol = to_rdkit(system.clone(atoms))
        order = np.asarray(atoms)
        found = []
        for feature in factory.GetFeaturesForMol(mol):
            if feature.GetFamily() in families:
                found.append((feature.GetFamily(), order[list(feature.GetAtomIds())]))
        out.append(found)
    return out


def feature_points(system, positions=None, reference=None, ligand: str = DEFAULT_LIGAND,
                   align: str = "protein and name CA", families=FAMILIES,
                   periodic: bool = True) -> tuple[dict, np.ndarray]:  # fmt: skip
    """``({family: (n, 3) positions}, {family: (n,) which copy})`` in the reference's frame."""
    per_copy = ligand_features(system, ligand, families)
    lig = _ids(system, ligand)
    fit = _ids(system, align)
    ref = system if reference is None else reference
    rfit = _ids(ref, align)
    if len(rfit) != len(fit):
        raise ValueError(f"align {align!r} selects {len(fit)} atoms here and {len(rfit)} in the "
                         "reference; they are paired in order")  # fmt: skip
    target = ref.positions[rfit]

    need = np.union1d(fit, lig)
    fit_at = np.searchsorted(need, fit)
    groups = [[(fam, np.searchsorted(need, atoms)) for fam, atoms in copy] for copy in per_copy]
    places = {fam: [] for fam in families}
    owners = {fam: [] for fam in families}
    for xyz, boxes in _boxed_blocks(system, positions, need)[0]:
        for X, box in zip(xyz, boxes, strict=True):
            box = box if periodic and np.asarray(box).any() else None
            rot, shift = kabsch(X[fit_at], target)
            anchor = X[fit_at].mean(0)
            for copy, features in enumerate(groups):
                for fam, at in features:
                    p = X[at]
                    here = p[0] + minimum_image(p - p[0], box).mean(0)
                    here = anchor + minimum_image((here - anchor)[None], box)[0]
                    places[fam].append(here @ rot.T + shift)
                    owners[fam].append(copy)
    return ({f: np.array(v) if v else np.empty((0, 3)) for f, v in places.items()},
            {f: np.array(v, np.int64) for f, v in owners.items()})  # fmt: skip


def feature_maps(system, runs=None, reference=None, ligand: str = DEFAULT_LIGAND,
                 align: str = "protein and name CA", families=FAMILIES, spacing: float = 1.0,
                 periodic: bool = True) -> dict[str, Density]:  # fmt: skip
    """A map per feature family: how much more often than bulk each kind is found where.

    Every map shares one grid, so they can be read against each other -- a
    place that wants an acceptor and not a donor is the interesting kind.
    Bulk is worked out per family, from that family's own count, so a ligand
    set rich in one kind does not make its map look hot everywhere.
    """
    pairs = _pairs(system, runs)
    reference = reference if reference is not None else pairs[0][0]
    places = {fam: [] for fam in families}
    owners = {fam: [] for fam in families}
    volume, seen = 0.0, 0
    for own, run in pairs:
        here, whose = feature_points(own, run, reference, ligand, align, families, periodic)
        for fam in families:
            places[fam].append(here[fam])
            owners[fam].append(whose[fam] + seen)
        seen += 1 + max((int(w.max()) for w in whose.values() if len(w)), default=0)
        cell = np.asarray(own.cell, float)
        volume = max(volume, abs(float(np.linalg.det(cell))))
    points = {fam: np.concatenate(v) if any(len(x) for x in v) else np.empty((0, 3))
              for fam, v in places.items()}  # fmt: skip
    whose = {fam: np.concatenate(v) if any(len(x) for x in v) else np.empty(0, np.int64)
             for fam, v in owners.items()}  # fmt: skip
    every = np.concatenate([p for p in points.values() if len(p)])
    if not len(every):
        raise ValueError(f"ligand {ligand!r} has none of {families}")
    if volume <= 0:
        hull = every.max(0) - every.min(0)
        volume = float(np.prod(np.maximum(hull, spacing)))

    lo = every.min(0) - spacing
    dims = np.maximum(np.ceil((every.max(0) + spacing - lo) / spacing), 1).astype(np.int64)
    maps = {}
    for fam, p in points.items():
        counts = np.zeros(int(dims.prod()), np.int64)
        if len(p):
            ijk = np.minimum(((p - lo) / spacing).astype(np.int64), dims - 1)
            flat = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
            counts = np.bincount(flat, minlength=int(dims.prod()))
        maps[fam] = Density(origin=lo, spacing=float(spacing),
                            counts=counts.reshape(dims).astype(np.int32),
                            expected=len(p) * spacing**3 / max(volume, 1e-9))  # fmt: skip
        maps[fam].owners = whose[fam]  # which molecule put each one there
        maps[fam].places = p
    return maps


def hotspots(maps, enrichment: float = 20.0, min_volume: float = 3.0) -> list[Hotspot]:
    """The peaks of the maps: what to put where, most enriched first.

    A hotspot is a connected region a family visits at least ``enrichment``
    times more often than bulk would explain, of at least ``min_volume`` A^3
    so that a single lucky frame is not one.  ``ligands`` counts the distinct
    molecules that put a feature there, which is the part worth trusting: a
    place five unlike molecules choose is a better bet than one a single
    molecule sat in for a long time.
    """
    out = []
    for fam, grid in maps.items():
        p = getattr(grid, "places", np.empty((0, 3)))
        if not len(p):
            continue
        dense = np.flatnonzero(grid.counts.reshape(-1) >= max(enrichment * grid.expected, 2.0))
        group, ngroups = _join_neighbours(dense, np.array(grid.counts.shape))
        if not ngroups:
            continue
        dims = np.array(grid.counts.shape)
        ijk = np.minimum(((p - grid.origin) / grid.spacing).astype(np.int64), dims - 1)
        flat = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
        of_cell = np.full(int(dims.prod()) + 1, -1, np.int64)
        of_cell[dense] = group
        label = of_cell[flat]
        owners = getattr(grid, "owners", np.zeros(len(p), np.int64))
        for g in range(ngroups):
            members = np.flatnonzero(label == g)
            cells = int((group == g).sum())
            volume = cells * grid.spacing**3
            if volume < min_volume or not len(members):
                continue
            peak = float(grid.counts.reshape(-1)[dense[group == g]].max() / grid.expected)
            out.append(Hotspot(family=fam, center=p[members].mean(0), enrichment=peak,
                               volume=volume, points=len(members),
                               ligands=len(np.unique(owners[members]))))  # fmt: skip
    out.sort(key=lambda h: (-h.ligands, -h.enrichment))
    return out


def write_hotspots(path, spots) -> None:
    """Write the hotspots as pseudo-atoms: one per peak, named for what it wants.

    ``DON``, ``ACC``, ``ARO``, ``HYD``, ``CAT``, ``ANI``, with the enrichment
    in the B-factor column and the tolerance radius as the occupancy, so a
    viewer can size and colour them without being told anything else.
    """
    with open(path, "w", encoding="utf-8") as out:
        out.write("REMARK   the pocket's preferences, from simulation\n")
        out.write("REMARK   occupancy: tolerance radius (A); B-factor: times bulk\n")
        for i, h in enumerate(spots, 1):
            x, y, z = h.center
            out.write(f"HETATM{i:5d}  X   {CODES.get(h.family, 'UNK'):>3} A{i:4d}    "
                      f"{x:8.3f}{y:8.3f}{z:8.3f}{min(h.radius, 9.99):6.2f}"
                      f"{min(h.enrichment, 999.99):6.2f}          X\n")  # fmt: skip
        out.write("END\n")


def wanted(spots, center, within: float = 6.0) -> list[Hotspot]:
    """The hotspots near a place, most agreed-upon first: what that pocket asks for."""
    center = np.asarray(center, float).reshape(3)
    near = [h for h in spots if float(np.linalg.norm(h.center - center)) <= within]
    return sorted(near, key=lambda h: (-h.ligands, -h.enrichment))
