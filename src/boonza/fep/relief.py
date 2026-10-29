"""Ligand poses that clash with the receptor, relieved before a network is built
(boonza fep prep --rem).

A series is often posed by placing every ligand onto one receptor, and the
larger ones then reach into side chains that would have had to move for them:
heavy atoms a fraction of an ångström apart, which a leg's minimization takes
huge, damaging steps from.  With ``--rem`` each ligand is put into the receptor
and minimized with soft nonbonded interactions (:mod:`boonza.rem`) while the
receptor's heavy atoms are held where they are, and only the ligand's new
coordinates are kept, so every edge still meets one and the same receptor.  The
ligand's own heavy atoms are tethered to where it was posed, so the atoms in a
clash move out of it and the rest stays put: an edge's two ligands still sit on
top of each other, which the mapping and the hybrid both read.  Coulomb is
left out (``C=0``): unscreened, in vacuum, a charged ligand would be drawn to
the nearest charged side chain, and a clash is a matter of overlap, which the
soft repulsion settles; the real electrostatics, in water, are the legs'.  Soft
potentials let atoms pass through each other, a stereocentre's or a double
bond's among them, so the ligand's stereocentres and non-aromatic double bonds
are held as they were posed.  Each leg's own minimization, in water and with the
real force field, finishes what this leaves.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

#: ligand and receptor heavy atoms nearer than this (Å) are a clash
CLASH_A = 2.2
#: what holds the receptor's heavy atoms in place while the ligand moves (kJ/mol/nm^2)
RECEPTOR_HOLD_KJ = 1.0e5
#: what tethers the ligand's heavy atoms to their pose (kJ/mol/nm^2): 0.2 nm costs 10
#: kJ/mol an atom, less than the soft repulsion's 100, so a clashing atom gets out of
#: it, while the whole ligand moving costs that over every atom and does not happen
TETHER_KJ = 500.0


def closest(lig_xyz, rec_xyz) -> float:
    """The shortest distance (Å) between two sets of points."""
    from scipy.spatial import cKDTree

    if not len(lig_xyz) or not len(rec_xyz):
        return float("inf")
    return float(cKDTree(np.asarray(rec_xyz, float)).query(np.asarray(lig_xyz, float))[0].min())


def _heavy(s) -> np.ndarray:
    return np.flatnonzero(np.asarray(s.atoms["anum"]) > 1)


def clashes(receptor, ligs, names) -> list[tuple[str, float]]:
    """(name, closest heavy-atom distance) of each ligand nearer the receptor than
    :data:`CLASH_A`, closest first."""
    rec = np.asarray(receptor.positions, float)[_heavy(receptor)]
    out = []
    for lig, name in zip(ligs, names, strict=True):
        d = closest(np.asarray(lig.positions, float)[_heavy(lig)], rec)
        if d < CLASH_A:
            out.append((name, d))
    return sorted(out, key=lambda x: x[1])


def held_torsions(mol) -> list[tuple[int, int, int, int]]:
    """Torsions (atom indices of ``mol``) that keep its stereochemistry: one about
    each stereocentre, whose sign is its handedness, and one across each double
    bond outside a ring and an aromatic system, whose angle is cis or trans."""
    from rdkit import Chem

    out = []
    centres = Chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                        useLegacyImplementation=False)  # fmt: skip
    for c, _ in centres:
        nbrs = [a.GetIdx() for a in mol.GetAtomWithIdx(c).GetNeighbors()]
        if len(nbrs) >= 3:
            out.append((nbrs[0], c, nbrs[1], nbrs[2]))
    for bond in mol.GetBonds():
        if (bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetIsAromatic()
                or bond.IsInRing()):  # fmt: skip
            continue
        b, c = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        a = [n.GetIdx() for n in mol.GetAtomWithIdx(b).GetNeighbors() if n.GetIdx() != c]
        d = [n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors() if n.GetIdx() != b]
        if a and d:
            out.append((a[0], b, c, d[0]))
    return out


def relieve(receptor, ligs, mols, names, ff, args, log=print) -> list[dict]:
    """Each ligand's pose relieved in ``receptor`` (the module docstring), in
    place: ``ligs`` (boonza systems) and ``mols`` (their RDKit molecules, the
    same atoms in the same order) get the new coordinates.  One row per ligand:
    its closest heavy atom to the receptor before and after (Å), and how far its
    heavy atoms moved (RMSD and most, Å)."""
    import openmm

    from ..omm import to_openmm
    from ..rem import relax
    from .prep import _by_tag, _parameterize

    platform = openmm.Platform.getPlatformByName(args.platform) if args.platform else None
    rec_heavy = _heavy(receptor)
    rec_xyz = np.asarray(receptor.positions, float)[rec_heavy]
    n_rec = receptor.natoms
    rows = []
    for lig, mol, name in zip(ligs, mols, names, strict=True):
        both = receptor.copy()
        both.append(lig)
        both.atoms["fep_index"] = np.arange(1, both.natoms + 1)
        param = _parameterize(both, ff, False)
        where = _by_tag(param, both.natoms)
        _, system, _ = to_openmm(param, nonbonded_method="NoCutoff", constraints=False,
                                 dispersion_correction=False)  # fmt: skip
        hold = openmm.CustomExternalForce("0.5*k_rec*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        hold.addGlobalParameter("k_rec", RECEPTOR_HOLD_KJ)
        for p in ("x0", "y0", "z0"):
            hold.addPerParticleParameter(p)
        xyz = np.asarray(param.positions, float) / 10.0  # nm
        for i in where[rec_heavy]:
            hold.addParticle(int(i), xyz[i].tolist())
        system.addForce(hold)
        lig_idx = where[n_rec + np.arange(lig.natoms)]
        tether = openmm.CustomExternalForce("0.5*k_lig*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        tether.addGlobalParameter("k_lig", TETHER_KJ)
        for p in ("x0", "y0", "z0"):
            tether.addPerParticleParameter(p)
        for i in lig_idx[_heavy(lig)]:
            tether.addParticle(int(i), xyz[i].tolist())
        system.addForce(tether)
        torsions = [tuple(int(lig_idx[k]) for k in t) for t in held_torsions(mol)]
        new, report = relax(system, xyz, torsions=torsions, C=0.0, platform=platform)
        old = np.asarray(lig.positions, float)
        moved = new[lig_idx] * 10.0
        heavy = _heavy(lig)
        shift = np.linalg.norm(moved[heavy] - old[heavy], axis=1)
        row = {"ligand": name,
               "closest_before": round(closest(old[heavy], rec_xyz), 3),
               "closest_after": round(closest(moved[heavy], rec_xyz), 3),
               "moved_rmsd": round(float(np.sqrt(np.mean(shift**2))), 3),
               "moved_most": round(float(shift.max()), 3), **report}  # fmt: skip
        lig.positions = moved
        conf = mol.GetConformer()
        for k, p in enumerate(moved):
            conf.SetAtomPosition(k, p.tolist())
        note = f"  -- still within {CLASH_A} Å" if row["closest_after"] < CLASH_A else ""
        log(f"  {name}: closest heavy atom {row['closest_before']:.2f} -> "
            f"{row['closest_after']:.2f} Å, moved {row['moved_rmsd']:.2f} Å "
            f"(most {row['moved_most']:.2f}){note}")  # fmt: skip
        rows.append(row)
    return rows


def write(work: Path, rows, mols) -> None:
    """``rem.csv`` (a row per ligand) and ``ligands_rem.sdf`` (the relieved poses,
    with every field the ligands came with) in ``work``."""
    from rdkit import Chem

    with (Path(work) / "rem.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with Chem.SDWriter(str(Path(work) / "ligands_rem.sdf")) as w:
        for m in mols:
            w.write(m)
