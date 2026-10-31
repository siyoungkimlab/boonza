"""A co-complex for every ligand, grown in the pocket from a reference one.

    boonza fep grow receptor.pdb --reference crystal.sdf --ligands series.sdf --workdir W

writes W/complexes/NAME.mae (the protein and the ligand, for boonza fep prep
--complexes) and W/grow/NAME.json (every step, every candidate and its score).

The reference ligand sits in the receptor as it was crystallized.  For each
target, the two graphs' maximum common substructure (rings whole, elements
matched) is the core: its atoms take the reference's coordinates and are held
there (restrained) from then on.  What the target has beyond the core is grown
onto it a step at a time, each step a unit that makes chemical sense:

* a ring, or a fused ring system, whole;
* a functional group, never split (carboxylate, guanidinium, amidine, amide,
  ester, carbamate, urea, sulfonamide, sulfone, nitro, nitrile, phosphate,
  CF3, ...), nor any atoms joined by a double or triple bond;
* otherwise a chain of up to ``max_heavy`` (3) heavy atoms, cut only at single
  bonds.

Every intermediate is a real molecule: the target's atoms placed so far, an
open valence capped with a hydrogen.  A step places its unit by embedding many
conformers with every atom placed before held where it is (the torsions about
the new bond, a ring's flips, its puckers), drops those that clash with the
protein, and takes the ``keep`` most distinct of the rest through REM and EM (the
pocket making room, boonza.fep.relief) and a short MD with the core restrained,
the residues within ``POCKET_A`` of the ligand as it now is free and the rest of
the protein held; without solvent, in a reaction field of water's dielectric
(``water``: in TIP3P).  Each is scored on the new unit:

    score = sum over its polar atoms of the fraction of frames they hydrogen-bond
            - 2 x (1 - that fraction) for each one that is buried (protein all around)
            - 0.5 x its drift (Å) + 0.01 x its protein contacts

so a pyridine N that points at a backbone NH wins over the flip that buries it
against a leucine.  A candidate whose unit drifts more than ``MAX_DRIFT`` is
dropped.  The ``beam`` best go on to the next step, so one wrong turn early does
not decide the rest.  Placed atoms carry on into the next step where the MD left
them, the protein too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: functional groups grown whole (their atoms outside the core, at least two)
GROUPS = {
    "carboxylate": "[CX3](=O)[OX1H0-,OX2H1]",
    "amide": "[CX3](=O)[NX3]",
    "ester": "[CX3](=O)[OX2][#6]",
    "carbamate": "[NX3][CX3](=O)[OX2]",
    "urea": "[NX3][CX3](=O)[NX3]",
    "guanidine": "[NX3,NX2+0,NX3+][CX3](=[NX2,NX3+])[NX3]",
    "amidine": "[NX3][CX3]=[NX2,NX3+]",
    "sulfonamide": "[SX4](=O)(=O)[NX3]",
    "sulfone": "[SX4](=O)(=O)",
    "nitro": "[NX3+](=O)[O-]",
    "nitrile": "C#N",
    "phosphate": "P(=O)(~O)~O",
    "trihalomethyl": "C([F,Cl,Br])([F,Cl,Br])[F,Cl,Br]",
    "carbonyl": "[CX3]=[OX1]",
}
#: Å: a unit whose mean position moves farther than this in its MD is dropped
MAX_DRIFT = 2.0
#: Å: heavy atoms nearer than this to the protein's count as a clash when placing
CLASH_A = 2.5
#: Å: a hydrogen bond's donor-acceptor distance at most, and its angle at least (°)
HBOND_A, HBOND_DEG = 3.3, 120.0
#: a polar atom with fewer waters than this within 3.5 Å, on average, is buried
BURIED_WATERS = 0.5
#: without water: protein heavy atoms within 4.5 Å of a polar atom that bury it
CROWDED = 8
#: kJ/mol/nm^2: what holds the core's heavy atoms in the MD (2000 let a held
#: alpha carbon slip 0.5 Å through the plane of its substituents and invert)
CORE_KJ = 1.0e5
#: Å: residues with a heavy atom this near the ligand move, the rest are held; found
#: again for every placement, from the ligand as it now is, so the pocket grows with it
POCKET_A = 8.0


# --- the plan: core, units, steps --------------------------------------------------------


def core_map(reference, target) -> dict[int, int]:
    """{target heavy atom: reference heavy atom} over the two graphs' maximum
    common substructure, rings whole, elements and stereocentres matched."""
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    def heavy(m):
        m = Chem.Mol(m)
        for a in m.GetAtoms():
            a.SetIntProp("orig", a.GetIdx())
        return Chem.RemoveHs(m)

    r, t = heavy(reference), heavy(target)
    res = rdFMCS.FindMCS([r, t], ringMatchesRingOnly=True, completeRingsOnly=True,
                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareOrder, matchChiralTag=False,
                         timeout=30)  # fmt: skip
    if not res.numAtoms:
        raise ValueError("the target shares nothing with the reference")
    query = Chem.MolFromSmarts(res.smartsString)
    best = None
    for tm in t.GetSubstructMatches(query, useChirality=False, uniquify=False, maxMatches=50):
        for rm in r.GetSubstructMatches(query, useChirality=False, uniquify=False,
                                        maxMatches=50):  # fmt: skip
            pairs = {t.GetAtomWithIdx(i).GetIntProp("orig"): r.GetAtomWithIdx(j).GetIntProp("orig")
                     for i, j in zip(tm, rm, strict=True)}  # fmt: skip
            best = best or pairs
            if _stereo_agrees(reference, target, pairs):
                return pairs
    return best


def _stereo_agrees(reference, target, pairs) -> bool:
    """Whether the core's stereocentres, as the reference has them in 3D, are the
    target's."""
    from rdkit import Chem

    ref = Chem.Mol(reference)
    Chem.AssignStereochemistryFrom3D(ref)
    want = {a.GetIdx(): a.GetProp("_CIPCode") for a in target.GetAtoms() if a.HasProp("_CIPCode")}
    if not want:
        Chem.AssignStereochemistry(target, cleanIt=True, force=True)
        want = {a.GetIdx(): a.GetProp("_CIPCode") for a in target.GetAtoms()
                if a.HasProp("_CIPCode")}  # fmt: skip
    for t, r in pairs.items():
        if t in want and ref.GetAtomWithIdx(r).HasProp("_CIPCode"):
            # CIP labels depend on substituents beyond the core; compare only when
            # those are the same, i.e. trust only a disagreement in the core itself
            if ref.GetAtomWithIdx(r).GetProp("_CIPCode") != want[t]:
                return False
    return True


def _same_stereo(mol, xyz) -> bool:
    """Whether ``mol`` at positions ``xyz`` (Å) has its stereocentres as written."""
    from rdkit import Chem

    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    conf = Chem.Conformer(m.GetNumAtoms())
    for k, p in enumerate(xyz):
        conf.SetAtomPosition(k, [float(v) for v in p])
    m.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(m)
    return Chem.MolToSmiles(Chem.RemoveHs(m)) == Chem.MolToSmiles(Chem.RemoveHs(mol))


def _ring_systems(mol) -> list[set[int]]:
    systems = []
    for ring in mol.GetRingInfo().AtomRings():
        ring = set(ring)
        merged = [s for s in systems if s & ring]
        for s in merged:
            systems.remove(s)
            ring |= s
        systems.append(ring)
    return systems


def units(mol, placed) -> list[frozenset[int]]:
    """The target's heavy atoms outside ``placed``, in the pieces grown whole:
    ring systems, functional groups (GROUPS), atoms joined by a multiple bond,
    and every other atom alone."""
    from rdkit import Chem

    free = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1} - set(placed)
    parent = {i: i for i in free}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(atoms):
        atoms = [i for i in atoms if i in free]
        for i in atoms[1:]:
            parent[root(i)] = root(atoms[0])

    for system in _ring_systems(mol):
        join(sorted(system))
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE:
            join([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])
    for smarts in GROUPS.values():
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)):
            join(list(match))
    out = {}
    for i in free:
        out.setdefault(root(i), set()).add(i)
    return [frozenset(u) for u in out.values()]


def _heavy_neighbours(mol, i) -> set[int]:
    return {n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() > 1}


def steps(mol, core, max_heavy: int = 3) -> list[list[int]]:
    """The order to grow the target's heavy atoms outside ``core`` in: each step
    one unit (units), or a chain of acyclic ones up to ``max_heavy`` heavy atoms,
    always bonded to what is placed already."""
    ring_atoms = set().union(*_ring_systems(mol)) if mol.GetRingInfo().NumRings() else set()
    placed, left, out = set(core), list(units(mol, core)), []

    def touches(u):
        return any(_heavy_neighbours(mol, i) & placed for i in u)

    while left:
        frontier = sorted((u for u in left if touches(u)), key=lambda u: (min(u), len(u)))
        if not frontier:
            raise ValueError("a part of the target is not bonded to the core")
        unit = frontier[0]
        left.remove(unit)
        step = set(unit)
        placed |= unit
        while not step & ring_atoms and len(step) < max_heavy:
            nxt = [u for u in left if not u & ring_atoms and touches(u)
                   and len(step) + len(u) <= max_heavy
                   and any(_heavy_neighbours(mol, i) & step for i in u)]  # fmt: skip
            if not nxt:
                break
            u = min(nxt, key=lambda u: (min(u), len(u)))
            left.remove(u)
            step |= u
            placed |= u
        out.append(sorted(step))
    return out


def intermediate(mol, placed):
    """The target cut down to the heavy atoms ``placed`` and their hydrogens, an
    open valence capped with a hydrogen; and {target atom: its atom}."""
    from rdkit import Chem

    placed = set(placed)
    keep, caps = set(), set()
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if a.GetAtomicNum() > 1:
            if i in placed:
                keep.add(i)
            elif _heavy_neighbours(mol, i) & placed:
                caps.add(i)
        elif {n.GetIdx() for n in a.GetNeighbors()} & placed:
            keep.add(i)
    rw = Chem.RWMol(mol)
    for i in caps:
        a = rw.GetAtomWithIdx(i)
        a.SetAtomicNum(1)
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
        a.SetNumExplicitHs(0)
        a.SetNoImplicit(True)
        for b in a.GetBonds():
            if b.GetOtherAtomIdx(i) in placed:
                b.SetBondType(Chem.BondType.SINGLE)
                b.SetIsAromatic(False)
    drop = sorted(set(range(mol.GetNumAtoms())) - keep - caps, reverse=True)
    for i in drop:
        rw.RemoveAtom(i)
    kept = sorted(keep | caps)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out, {old: new for new, old in enumerate(kept)}


# --- placing a step ------------------------------------------------------------------------


def embed(mol, fixed: dict[int, np.ndarray], n: int = 60, seed: int = 7) -> list[np.ndarray]:
    """Up to ``n`` conformers of ``mol`` (Å) with its atoms ``fixed`` where they are,
    alike ones dropped.  Embedded with the fixed atoms as a coordinate map where
    that works; a crystal's exact geometry often cannot be met, and then each
    conformer is embedded freely and superposed on the fixed atoms within two bonds
    of the new ones (the attachment, so the new atoms leave it as they should).
    Either way the fixed atoms are put back exactly and the rest relaxed around
    them (MMFF, the fixed atoms held)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D

    from ..align import kabsch

    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    keys = sorted(fixed)
    target = np.array([fixed[i] for i in keys]) if keys else np.zeros((0, 3))
    new = [i for i in range(m.GetNumAtoms())
           if i not in fixed and m.GetAtomWithIdx(i).GetAtomicNum() > 1]  # fmt: skip
    cmap = {i: Point3D(*map(float, xyz)) for i, xyz in fixed.items()}
    ids = list(AllChem.EmbedMultipleConfs(m, numConfs=n, coordMap=cmap, randomSeed=seed,
                                          useRandomCoords=True, enforceChirality=True,
                                          maxAttempts=5))  # fmt: skip
    anchor = keys
    if not ids:  # free, superposed on the attachment
        ids = list(AllChem.EmbedMultipleConfs(m, numConfs=n, randomSeed=seed,
                                              enforceChirality=True))  # fmt: skip
        if not ids:
            raise RuntimeError(f"RDKit cannot embed {Chem.MolToSmiles(Chem.RemoveHs(mol))}")
        near = set()
        for a in new:
            for b in m.GetAtomWithIdx(a).GetNeighbors():
                near.add(b.GetIdx())
                near |= {c.GetIdx() for c in b.GetNeighbors()}
        near = sorted(i for i in near if i in fixed)
        anchor = near if len(near) >= 3 else keys
    where = [keys.index(i) for i in anchor]
    props = AllChem.MMFFGetMoleculeProperties(m)
    want = Chem.MolToSmiles(Chem.RemoveHs(mol))  # its stereocentres, as the target has them
    out = []
    for cid in ids:
        xyz = m.GetConformer(cid).GetPositions()
        if len(anchor) >= 3:
            R, t = kabsch(xyz[anchor], target[where])
            xyz = xyz @ R.T + t
        xyz[keys] = target
        conf = m.GetConformer(cid)
        for k, p in enumerate(xyz):
            conf.SetAtomPosition(k, p.tolist())
        if props is not None and keys:  # the joins and hydrogens made good, fixed atoms held
            field = AllChem.MMFFGetMoleculeForceField(m, props, confId=cid)
            for i in keys:
                field.AddFixedPoint(i)
            field.Minimize(maxIts=500)
            xyz = m.GetConformer(cid).GetPositions()
            xyz[keys] = target
        # a hydrogen left on the wrong face of a held centre inverts it: such a
        # conformer is not the target
        probe = Chem.Mol(m, confId=cid)
        Chem.AssignStereochemistryFrom3D(probe, confId=cid)
        if Chem.MolToSmiles(Chem.RemoveHs(probe)) != want:
            continue
        if new and any(np.sqrt(((xyz[new] - o[new]) ** 2).sum(1).mean()) < 0.5 for o in out):
            continue
        out.append(xyz)
    if not out:
        raise RuntimeError(f"no conformer of {want} keeps the atoms placed before and its "
                           "stereocentres")  # fmt: skip
    return out


def clashes(xyz, heavy, protein_heavy) -> tuple[int, float]:
    """Heavy atoms nearer than CLASH_A to the protein's, and the nearest distance."""
    from scipy.spatial import cKDTree

    if not len(heavy):
        return 0, float("inf")
    d = cKDTree(protein_heavy).query(xyz[heavy])[0]
    return int((d < CLASH_A).sum()), float(d.min())


@dataclass
class State:
    """Where a growth stands: the protein's positions and the placed atoms'."""

    receptor: np.ndarray  # Å
    placed: dict  # target atom -> Å
    score: float = 0.0
    history: list = field(default_factory=list)


def _ligand(mol, xyz, name="LIG"):
    from ..chem import from_rdkit

    m = mol.__class__(mol)
    conf = m.GetConformer() if m.GetNumConformers() else None
    if conf is None:
        from rdkit import Chem

        conf = Chem.Conformer(m.GetNumAtoms())
        m.AddConformer(conf, assignId=True)
        conf = m.GetConformer()
    for k, p in enumerate(xyz):
        conf.SetAtomPosition(k, p.tolist())
    s = from_rdkit(m)
    s.residues["name"] = [name] * s.nresidues
    return s


def _forcefield(receptor, lig, ff, args):
    from .. import gaff, viparr

    both = receptor.copy()
    both.append(lig)
    patch = gaff.gaff2_patch(both, ff, charges=getattr(args, "ligand_charges", None),
                             parents=getattr(args, "parents", None),
                             protein_extent=getattr(args, "protein_extent", "matched"),
                             charge_geometry="reference")  # fmt: skip
    out = list(ff)
    host = gaff.host_index(out)
    out[host] = viparr.merge_forcefields(out[host], patch)
    return out


def md_score(receptor, lig, core, new, ff, args, ps: float = 100.0, seed: int = 1,
             water: bool = False, pocket_a: float = POCKET_A, log=print) -> dict:  # fmt: skip
    """A short MD of ``receptor`` and ``lig``, ``core`` (ligand atoms) restrained, the
    residues within ``pocket_a`` of the ligand free and the rest of the protein held
    where it is; without ``water`` (the default) in a reaction field of water's
    dielectric, no solvent, a polar atom buried where protein crowds it, else in
    TIP3P water, buried where no water is near.  The new unit's (``new``, ligand
    heavy atoms) hydrogen bonds, burial, drift and contacts, the score, and where
    the MD left the protein and the ligand (Å)."""
    import openmm as mm
    import openmm.unit as u
    from scipy.spatial import cKDTree

    from ..build import neutralize, solvate
    from ..omm import to_openmm
    from .prep import _by_tag, _parameterize

    both = receptor.copy()
    both.append(lig)
    n_rec, n_lig = receptor.natoms, lig.natoms
    charge = float(_parameterize(both, ff, False).atoms["charge"].sum())
    if water:
        box = solvate(both, thickness=10.0)
        box = neutralize(box, cation="Na", anion="Cl", charge=charge, concentration=args.saltM)
    else:
        box = both.copy()
    box.atoms["fep_index"] = np.arange(1, box.natoms + 1)
    param = _parameterize(box, ff, args.hmr)
    where = _by_tag(param, box.natoms)
    if water:
        _, system, _ = to_openmm(param, nonbonded_method="PME", cutoff=10.0 * args.cutoff_nm,
                                 constraints=True, dispersion_correction=True)  # fmt: skip
    else:  # charges screened as by water beyond the cutoff: no salt bridge in vacuum
        _, system, _ = to_openmm(param, nonbonded_method="CutoffNonPeriodic", cutoff=12.0,
                                 constraints=True, dispersion_correction=False,
                                 epsilon_rf=78.5)  # fmt: skip
    xyz0 = np.asarray(param.positions, float) / 10.0
    rec, lig_ids = where[:n_rec], where[n_rec:][:n_lig]
    anum = np.asarray(param.atoms["anum"])
    hold = mm.CustomExternalForce("0.5*k*periodicdistance(x,y,z,x0,y0,z0)^2" if water
                                  else "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")  # fmt: skip
    hold.addPerParticleParameter("k")
    for p in ("x0", "y0", "z0"):
        hold.addPerParticleParameter(p)
    for i in lig_ids[list(core)]:
        if anum[i] > 1:
            hold.addParticle(int(i), [CORE_KJ, *xyz0[i]])
    lig_heavy = lig_ids[anum[lig_ids] > 1]
    rec_heavy = rec[anum[rec] > 1]
    # the pocket's residues free, every other protein heavy atom held where it is
    res = np.asarray(param.atoms["residue"])
    near = cKDTree(xyz0[lig_heavy]).query(xyz0[rec_heavy])[0] <= pocket_a / 10.0
    pocket = set(res[rec_heavy[near]].tolist())
    for i in rec_heavy:
        if res[i] not in pocket:
            hold.addParticle(int(i), [5000.0, *xyz0[i]])
    system.addForce(hold)
    platform = mm.Platform.getPlatformByName(args.platform) if args.platform else None
    integ = mm.LangevinMiddleIntegrator(args.temperature * u.kelvin, 1.0 / u.picosecond,
                                        args.integration_fs * u.femtosecond)  # fmt: skip
    ctx = (mm.Context(system, integ, platform) if platform is not None
           else mm.Context(system, integ))  # fmt: skip
    if water:
        ctx.setPeriodicBoxVectors(*(np.asarray(param.cell, float) / 10.0))
    ctx.setPositions(xyz0)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 500)
    ctx.setVelocitiesToTemperature(args.temperature * u.kelvin, seed)
    every = max(1, int(round(2.0 / (args.integration_fs / 1000.0))))  # a frame every 2 ps
    integ.step(5 * every)  # 10 ps to settle
    # polar atoms, and which hydrogens sit on them
    hydrogens = {}
    tab = param.bonds
    for i, j in zip(np.asarray(tab["i"]), np.asarray(tab["j"]), strict=True):
        if anum[i] == 1 and anum[j] in (7, 8):
            hydrogens.setdefault(int(j), []).append(int(i))
        elif anum[j] == 1 and anum[i] in (7, 8):
            hydrogens.setdefault(int(i), []).append(int(j))
    polar_all = np.flatnonzero(np.isin(anum, (7, 8)))
    new_ids = lig_ids[list(new)]
    polar_new = [int(i) for i in new_ids if anum[i] in (7, 8)]
    water_o = np.intersect1d(np.asarray(param.select("water").ids), np.flatnonzero(anum == 8))
    start = None
    bonded = {int(i): set() for i in polar_new}
    for i, j in zip(np.asarray(tab["i"]), np.asarray(tab["j"]), strict=True):
        if int(i) in bonded:
            bonded[int(i)].add(int(j))
        if int(j) in bonded:
            bonded[int(j)].add(int(i))
    hb = {i: 0 for i in polar_new}
    wet = {i: 0.0 for i in polar_new}
    drift, contacts, frames = [], [], 0
    n_frames = max(1, int(round(ps / 2.0)))
    for _ in range(n_frames):
        integ.step(every)
        x = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.angstrom)
        if start is None:
            start = x[new_ids].copy()
        frames += 1
        tree = cKDTree(x[polar_all])
        for i in polar_new:
            near = [int(polar_all[k]) for k in tree.query_ball_point(x[i], HBOND_A)]
            ok = False
            for j in near:
                if j == i or j in bonded[i]:
                    continue
                for donor, acceptor in ((j, i), (i, j)):
                    for h in hydrogens.get(donor, []):
                        a = x[donor] - x[h]
                        b = x[acceptor] - x[h]
                        cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                        if np.degrees(np.arccos(np.clip(cos, -1, 1))) >= HBOND_DEG:
                            ok = True
                            break
                    if ok:
                        break
                if ok:
                    break
            hb[i] += ok
            if water:
                wet[i] += float((np.linalg.norm(x[water_o] - x[i], axis=1) < 3.5).sum())
            else:  # open to solvent where little protein crowds it
                crowd = (np.linalg.norm(x[rec_heavy] - x[i], axis=1) < 4.5).sum()
                wet[i] += float(crowd < CROWDED)
        new_heavy = new_ids[anum[new_ids] > 1]
        if len(new_heavy):
            drift.append(np.linalg.norm(x[new_ids].mean(0) - start.mean(0)))
            contacts.append(int((cKDTree(x[rec_heavy]).query(x[new_heavy])[0] < 4.0).sum()))
    persistence = {i: hb[i] / frames for i in polar_new}
    buried = [i for i in polar_new if wet[i] / frames < BURIED_WATERS]
    mean_drift = float(np.mean(drift)) if drift else 0.0
    mean_contacts = float(np.mean(contacts)) if contacts else 0.0
    score = (sum(persistence.values()) - 2.0 * sum(1.0 - persistence[i] for i in buried)
             - 0.5 * mean_drift + 0.01 * mean_contacts)  # fmt: skip
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 200)
    x = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.angstrom)
    ref = np.asarray(receptor.positions, float)
    # back into the receptor's frame: the box moved everything by one vector
    shift = (xyz0[rec] * 10.0 - ref).mean(0)
    return {"score": float(score), "drift": mean_drift, "contacts": mean_contacts,
            "hbonds": [round(persistence[i], 2) for i in polar_new],
            "buried_unsatisfied": sum(1 for i in buried if persistence[i] < 0.5),
            "receptor": x[rec] - shift, "ligand": x[lig_ids] - shift}  # fmt: skip


# --- growing one target -----------------------------------------------------------------


def grow(receptor, reference, target, name: str, ff, args, workdir, beam: int = 2,
         keep: int = 3, n_conformers: int = 60, md_ps: float = 100.0, max_heavy: int = 3,
         water: bool = False, log=print) -> Path:  # fmt: skip
    """``target`` (RDKit, with hydrogens) grown from ``reference`` (posed in
    ``receptor``) step by step (the module docstring); its complex's path."""
    from ..io import save
    from . import relief

    work = Path(workdir)
    (work / "complexes").mkdir(parents=True, exist_ok=True)
    (work / "grow").mkdir(parents=True, exist_ok=True)
    core = core_map(reference, target)
    ref_xyz = reference.GetConformer().GetPositions()
    plan = steps(target, core, max_heavy)
    if not plan:  # nothing to grow: the core is the whole target; settle its hydrogens
        plan = [[]]
    log(f"{name}: core {len(core)} heavy atoms from the reference; "
        f"{sum(len(s) for s in plan)} to grow in {len(plan)} step(s): "
        + " | ".join(" ".join(target.GetAtomWithIdx(i).GetSymbol() for i in s) or "-"
                     for s in plan))  # fmt: skip
    states = [State(np.asarray(receptor.positions, float),
                    {t: ref_xyz[r] for t, r in core.items()})]  # fmt: skip
    record = {"name": name, "core": len(core), "steps": []}
    placed = set(core)
    for k, step in enumerate(plan, 1):
        placed |= set(step)
        mol, index = intermediate(target, placed)
        new = [index[i] for i in step]
        found = []
        lig_ff = None
        for s, state in enumerate(states):
            fixed = {index[i]: xyz for i, xyz in state.placed.items() if i in index}
            confs = embed(mol, fixed, n=n_conformers, seed=7 + 31 * k + s)
            rec = receptor.copy()
            rec.positions = state.receptor
            prot_heavy = state.receptor[np.asarray(rec.atoms["anum"]) > 1]
            ranked = sorted(confs, key=lambda c: (clashes(c, new, prot_heavy)[0],
                                                  -clashes(c, new, prot_heavy)[1]))  # fmt: skip
            for c, xyz in enumerate(ranked[:keep]):
                lig = _ligand(mol, xyz)
                if lig_ff is None:
                    lig_ff = _forcefield(rec, lig, ff, args)
                here = rec.copy()
                relief.relieve(here, [lig], [name], lig_ff, args, log=lambda *_: None)
                core_ids = [index[i] for i in state.placed if i in index]
                out = md_score(here, lig, core_ids, new, lig_ff, args, ps=md_ps,
                               seed=1000 * k + 10 * s + c, water=water, log=log)  # fmt: skip
                row = {k_: v for k_, v in out.items() if k_ not in ("receptor", "ligand")}
                row.update(step=k, parent=s, candidate=c)
                record["steps"].append(row)
                if out["drift"] > MAX_DRIFT:
                    continue
                if not _same_stereo(mol, out["ligand"]):  # a centre inverted in the MD
                    row["inverted"] = True
                    continue
                back = {i: out["ligand"][j] for i, j in index.items()
                        if target.GetAtomWithIdx(i).GetAtomicNum() > 1 and i in placed}  # fmt: skip
                found.append((out["score"], State(out["receptor"], back, out["score"],
                                                  state.history + [row]), out, mol))  # fmt: skip
        if not found:
            raise RuntimeError(f"{name}: no placement of step {k} stays where it was put")
        found.sort(key=lambda f: -f[0])
        states = [f[1] for f in found[:beam]]
        best = found[0]
        log(f"  step {k}: {len(found)} placements kept; best score {best[0]:.2f} "
            f"(hydrogen bonds {best[2]['hbonds']}, buried unsatisfied "
            f"{best[2]['buried_unsatisfied']}, drift {best[2]['drift']:.2f} Å)")  # fmt: skip
    _, final, out, mol = best
    complex_ = receptor.copy()
    complex_.positions = final.receptor
    complex_.append(_ligand(mol, out["ligand"]))
    path = work / "complexes" / f"{name}.mae"
    save(complex_, path)
    record["score"] = final.score
    (work / "grow" / f"{name}.json").write_text(json.dumps(record, indent=1, default=float))
    return path
