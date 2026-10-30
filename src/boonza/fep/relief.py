"""A receptor relaxed around every ligand's pose before a network is built
(boonza fep prep --rem).

A series is often posed by placing every ligand onto one receptor, and the
larger ones then reach into side chains that would have had to move for them:
heavy atoms a fraction of an ångström apart, which a leg's minimization takes
huge, damaging steps from.  The poses are what is known, so with ``--rem`` they
stay exactly where they are and the receptor makes room: every ligand is put
into it at once, blind to the others (no ligand-ligand terms, they overlap one
another by design), and held fixed, while the receptor is minimized first with
soft nonbonded interactions (REM, :mod:`boonza.rem`) and then with the real force
field (EM).  One receptor comes out, room made in it for every ligand, so every
edge still meets one and the same receptor.

In REM the residues within :data:`FREE_A` of a ligand are free, backbone and
side chain alike -- a pocket accommodates a larger ligand with its backbone too --
and the rest of the receptor and any cofactor are held to the posed receptor
(:data:`HOLD_KJ`).  REM's own soft, bounded Coulomb stays: without
it a hydrogen bond's or a salt bridge's atoms sit inside each other's
Lennard-Jones radius with nothing holding them there.  Soft potentials let atoms
pass through each other, so the free residues' stereocentres (CA, and Ile's and
Thr's CB) are held as they were.  EM then finishes with the force field's own
Lennard-Jones and Coulomb on what REM was free to move: the free residues,
held firmly to where REM left them (:data:`EM_HOLD_KJ`), and every hydrogen;
the rest of the receptor stays fixed, since in vacuum, unscreened, its charges
would pull at each other.  Each leg's own minimization and equilibration,
in water, follow.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

#: ligand and receptor heavy atoms nearer than this (Å) are a clash
CLASH_A = 2.2
#: residues with a heavy atom this near a ligand's (Å): free in REM, backbone too
FREE_A = 5.0
#: what holds the residues farther out and cofactors to the posed receptor in REM
#: (kJ/mol/nm^2)
HOLD_KJ = 1.0e4
#: what holds the free residues' heavy atoms to where REM left them in EM
#: (kJ/mol/nm^2), the rest of the receptor fixed there: in vacuum, unscreened,
#: charges pull (1000 let a terminus move 2.8 Å toward another, 10000 an Asp 2.4 Å);
#: an overlap REM left pushes far harder
EM_HOLD_KJ = 1.0e5
#: residues whose heavy atoms moved at least this (Å) are listed in rem_residues.csv
MOVED_A = 0.1
#: stereocentres other than CA: (centre, its neighbours as the improper reads them)
_CENTRES = {"ILE": ("CB", ("CA", "CG1", "CG2")), "THR": ("CB", ("CA", "OG1", "CG2"))}


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


def pocket(receptor, ligs, within: float = FREE_A) -> np.ndarray:
    """The receptor's protein residues (indices) with a heavy atom within ``within``
    Å of any ligand's."""
    from scipy.spatial import cKDTree

    heavy = _heavy(receptor)
    protein = np.zeros(receptor.natoms, bool)
    protein[receptor.select("protein").ids] = True
    heavy = heavy[protein[heavy]]
    lig = np.concatenate([np.asarray(g.positions, float)[_heavy(g)] for g in ligs])
    near = cKDTree(lig).query(np.asarray(receptor.positions, float)[heavy])[0] <= within
    return np.unique(np.asarray(receptor.atoms["residue"])[heavy[near]])


def free_atoms(receptor, residues) -> np.ndarray:
    """Every atom of ``residues``, backbone and hydrogens too: what REM leaves free."""
    return np.flatnonzero(np.isin(np.asarray(receptor.atoms["residue"]), residues))


def held_torsions(receptor, residues) -> list[tuple[int, int, int, int]]:
    """Impropers (receptor atom indices) whose sign is the handedness of each of
    ``residues``' stereocentres: CA (N, CA, C, CB) and Ile's and Thr's CB."""
    res = np.asarray(receptor.atoms["residue"])
    name = np.asarray(receptor.atoms["name"])
    resname = np.asarray(receptor.residues["name"])
    out = []
    for r in residues:
        atoms = {str(name[i]): int(i) for i in np.flatnonzero(res == r)}
        if all(a in atoms for a in ("N", "CA", "C", "CB")):
            out.append((atoms["N"], atoms["CA"], atoms["C"], atoms["CB"]))
        centre = _CENTRES.get(str(resname[r]))
        if centre and all(a in atoms for a in (centre[0], *centre[1])):
            a, c, d = centre[1]
            out.append((atoms[a], atoms[centre[0]], atoms[c], atoms[d]))
    return out


def _blind(system, groups) -> None:
    """No nonbonded terms between atoms of different ``groups`` (lists of particle
    indices) of an OpenMM ``system``."""
    import openmm as mm

    pairs = [(int(i), int(j)) for a in range(len(groups)) for b in range(a + 1, len(groups))
             for i in groups[a] for j in groups[b]]  # fmt: skip
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            for i, j in pairs:
                f.addException(i, j, 0.0, 0.1, 0.0)
        elif isinstance(f, mm.CustomNonbondedForce):
            for i, j in pairs:
                f.addExclusion(i, j)


def _hold(system, atoms, xyz_nm, k: float, name: str) -> None:
    import openmm as mm

    f = mm.CustomExternalForce(f"0.5*{name}*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    f.addGlobalParameter(name, float(k))
    for p in ("x0", "y0", "z0"):
        f.addPerParticleParameter(p)
    for i in atoms:
        f.addParticle(int(i), xyz_nm[i].tolist())
    system.addForce(f)


def _minimize(system, xyz_nm, platform) -> tuple[np.ndarray, float, float]:
    import openmm as mm
    import openmm.unit as u

    integrator = mm.VerletIntegrator(0.001)
    context = (mm.Context(system, integrator, platform) if platform is not None
               else mm.Context(system, integrator))  # fmt: skip
    context.setPositions(xyz_nm * u.nanometer)

    def energy():
        e = context.getState(getEnergy=True).getPotentialEnergy()
        return round(e.value_in_unit(u.kilojoule_per_mole), 1)

    before = energy()
    mm.LocalEnergyMinimizer.minimize(context, 10.0, 0)
    after = energy()
    state = context.getState(getPositions=True)
    return np.asarray(state.getPositions(asNumpy=True).value_in_unit(u.nanometer)), before, after


def relieve(receptor, ligs, names, ff, args, log=print) -> tuple[list[dict], list[dict]]:
    """``receptor`` relaxed around every one of ``ligs`` (the module docstring), in
    place; the ligands are not moved.  Rows per ligand (its closest heavy atom to
    the receptor before and after, Å) and per residue that moved (its heavy atoms'
    most, after REM and after EM, Å)."""
    import openmm

    from ..omm import to_openmm
    from ..rem import relax
    from .prep import _by_tag, _parameterize

    platform = openmm.Platform.getPlatformByName(args.platform) if args.platform else None
    n_rec = receptor.natoms
    posed = np.asarray(receptor.positions, float).copy()
    both = receptor.copy()
    for lig in ligs:
        both.append(lig)
    both.atoms["fep_index"] = np.arange(1, both.natoms + 1)
    param = _parameterize(both, ff, False)
    where = _by_tag(param, both.natoms)
    _, system, _ = to_openmm(param, nonbonded_method="NoCutoff", constraints=False,
                             dispersion_correction=False)  # fmt: skip
    xyz = np.asarray(param.positions, float) / 10.0  # nm
    rec = where[:n_rec]
    starts = np.cumsum([n_rec] + [g.natoms for g in ligs])
    groups = [where[a:b] for a, b in zip(starts[:-1], starts[1:], strict=True)]
    for g in groups:  # held fixed: neither minimizer moves a massless particle
        for i in g:
            system.setParticleMass(int(i), 0.0)
    _blind(system, groups)

    residues = pocket(receptor, ligs)
    free = np.zeros(n_rec, bool)
    free[free_atoms(receptor, residues)] = True
    heavy = _heavy(receptor)
    log(f"  {len(residues)} residues within {FREE_A:g} Å of a ligand: free, backbone too")
    soft = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    _hold(soft, rec[heavy[~free[heavy]]], xyz, HOLD_KJ, "k_receptor")
    torsions = [tuple(int(rec[k]) for k in t) for t in held_torsions(receptor, residues)]
    after_rem, report = relax(soft, xyz, torsions=torsions, platform=platform)
    lig_atoms = np.concatenate(groups)
    after_rem[lig_atoms] = xyz[lig_atoms]  # as posed, to the last digit
    hydrogen = np.ones(n_rec, bool)
    hydrogen[heavy] = False
    for i in rec[~(free | hydrogen)]:  # outside the pocket: fixed where REM left it
        system.setParticleMass(int(i), 0.0)
    _hold(system, rec[heavy[free[heavy]]], after_rem, EM_HOLD_KJ, "k_em")
    final, em_before, em_after = _minimize(system, after_rem, platform)
    final[lig_atoms] = xyz[lig_atoms]
    report.update(em_energy_before=em_before, em_energy_after=em_after)
    log(f"  REM energy {report['rem_energy_before']:.1f} -> {report['rem_energy_after']:.1f}, "
        f"EM {em_before:.1f} -> {em_after:.1f} kJ/mol")  # fmt: skip
    receptor.positions = final[rec] * 10.0

    new = np.asarray(receptor.positions, float)
    rem_xyz = after_rem[rec] * 10.0
    lig_rows = []
    for lig, name in zip(ligs, names, strict=True):
        at = np.asarray(lig.positions, float)[_heavy(lig)]
        row = {"ligand": name, "closest_before": round(closest(at, posed[heavy]), 3),
               "closest_after": round(closest(at, new[heavy]), 3)}  # fmt: skip
        note = f"  -- still within {CLASH_A} Å" if row["closest_after"] < CLASH_A else ""
        log(f"  {name}: closest receptor heavy atom {row['closest_before']:.2f} -> "
            f"{row['closest_after']:.2f} Å{note}")  # fmt: skip
        lig_rows.append(row)
    res = np.asarray(receptor.atoms["residue"])[heavy]
    moved_rem = np.linalg.norm(rem_xyz[heavy] - posed[heavy], axis=1)
    moved = np.linalg.norm(new[heavy] - posed[heavy], axis=1)
    chains = np.asarray(receptor.chains["name"])
    res_rows = []
    for r in np.unique(res):
        most = float(moved[res == r].max())
        if most < MOVED_A:
            continue
        res_rows.append({"chain": str(chains[receptor.residues["chain"][r]]),
                         "resid": int(receptor.residues["resid"][r]),
                         "resname": str(receptor.residues["name"][r]),
                         "moved_rem": round(float(moved_rem[res == r].max()), 3),
                         "moved": round(most, 3)})  # fmt: skip
    res_rows.sort(key=lambda x: -x["moved"])
    top = ", ".join(f"{x['resname']}{x['resid']} {x['moved']:.2f}" for x in res_rows[:5])
    log(f"  {len(res_rows)} residues moved at least {MOVED_A:g} Å" + (f" (most: {top} Å)"
        if top else ""))  # fmt: skip
    return lig_rows, res_rows


def write(work: Path, receptor, lig_rows, res_rows, suffix: str = ".mae") -> None:
    """``receptor_rem<suffix>`` (the relaxed receptor), ``rem.csv`` (a row per ligand)
    and ``rem_residues.csv`` (a row per residue that moved) in ``work``."""
    import boonza

    work = Path(work)
    boonza.save(receptor, work / f"receptor_rem{suffix}")
    for path, rows, head in ((work / "rem.csv", lig_rows, ["ligand", "closest_before",
                              "closest_after"]),
                             (work / "rem_residues.csv", res_rows, ["chain", "resid", "resname",
                              "moved_rem", "moved"])):  # fmt: skip
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, head)
            w.writeheader()
            w.writerows(rows)
