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
* an unbranched chain whole, however long (a butoxy, an ethylene glycol): a bare
  chain reaches nothing, so where it should point cannot be judged until its end
  is there;
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

    score = - its interaction energy with the protein (kJ/mol), averaged over the
              MD's frames, over ``INTERACTION_KJ``
            - 0.5 x its drift (Å) + 0.01 x its protein contacts

so a pyridine N that points at a backbone NH wins over the flip that buries it
against a leucine, and a coordinated cation beats a loosely touching one by the
margin its electrostatics deserve rather than by a hand-set weight.  Hydrogen
bonds (per polar atom, with the donor and acceptor the right way round), burial
and cation-pi contacts are measured and reported for every placement, but do not
enter the score; the energy already prices what they describe, except for
desolvation, which no term covers while there is no explicit water.

A candidate whose unit drifts more than ``MAX_DRIFT`` is
dropped.  The ``beam`` best go on to the next step, so one wrong turn early does
not decide the rest; a step that adds no polar atom keeps twice as many
placements and carries every one on, since its score (drift and contacts alone)
says little about which way a chain should lie.  Placed atoms carry on into the
next step where the MD left them, the protein too.
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
#: kJ/mol of interaction energy with the protein per unit of score
INTERACTION_KJ = 50.0
#: water's dielectric, for the reaction field the interaction energy is read in
EPSILON_RF = 78.5
#: kJ Å / mol / e^2: Coulomb's constant, as OpenMM's ONE_4PI_EPS0 in these units
ONE_4PI_EPS0 = 1389.35456
#: Å: a cation this far from an aromatic ring's plane, and this far off the axis
#: through its centre, counts as a cation-pi contact (reported, not scored)
CATION_PI_A, CATION_PI_OFFSET_A = 4.5, 2.0
#: the aromatic rings of the protein's side chains, by residue and atom names
#: (His under any of its protonation states' names)
AROMATIC = {
    "PHE": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "TYR": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "TRP": (("CG", "CD1", "NE1", "CE2", "CD2"), ("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3")),
    "HIS": (("CG", "ND1", "CD2", "CE1", "NE2"),),
}
#: a polar atom with fewer waters than this within 3.5 Å, on average, is buried
BURIED_WATERS = 0.5
#: without water: protein heavy atoms within 4.5 Å of a polar atom that bury it
CROWDED = 8
#: kJ/mol/nm^2: what holds the core's heavy atoms in the MD (2000 let a held
#: alpha carbon slip 0.5 Å through the plane of its substituents and invert)
CORE_KJ = 1.0e5
#: kJ/mol: what holds the phi and psi of residues in helices and sheets (boonza md's
#: dihedral_restraint "ss"), so the free pocket keeps its secondary structure
SS_KJ = 20.0
#: Å: how far a loosely held atom may go before it is held (hold="rings")
LOOSE_A = 2.0
#: Å: two runs of --repeats agree on a pose when their heavy atoms are this close
AGREE_A = 2.0
#: the seed every run's final pose is measured with, so the measurement is the same one
RESCORE_SEED = 4242
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


def _centres(mol) -> dict[int, str]:
    """The stereocentres ``mol`` specifies (as written, or from its 3D
    coordinates when it has them and was so assigned), atom -> R/S."""
    from rdkit import Chem

    return dict(Chem.FindMolChiralCenters(mol, includeUnassigned=False,
                                          useLegacyImplementation=False))  # fmt: skip


def _stereo_agrees(reference, target, pairs) -> bool:
    """Whether the target's core, put on the reference's coordinates through
    ``pairs``, has the stereocentres the target specifies there: of each centre
    whose heavy neighbours are all in the core (its hydrogen, if any, opposite
    them), the handedness those coordinates fix.  A ring the reference has
    without substituents (a cyclopropyl) maps either way round onto the target's
    substituted one, and only one way gives the target's centre."""
    from rdkit import Chem

    want = _centres(target)
    if not want:
        return True
    rx = reference.GetConformer().GetPositions()
    m = Chem.Mol(target)
    m.RemoveAllConformers()
    xyz = np.zeros((m.GetNumAtoms(), 3))
    for t, r in pairs.items():
        xyz[t] = rx[r]
    judged = []
    for a, _ in want.items():
        atom = m.GetAtomWithIdx(a)
        heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        hs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
        if a not in pairs or not all(h in pairs for h in heavy) or len(heavy) < 3:
            continue
        if hs:  # opposite the heavy neighbours
            v = sum((xyz[h] - xyz[a]) / (np.linalg.norm(xyz[h] - xyz[a]) + 1e-9) for h in heavy)
            xyz[hs[0]] = xyz[a] - 1.09 * v / (np.linalg.norm(v) + 1e-9)
        judged.append(a)
    if not judged:
        return True
    conf = Chem.Conformer(m.GetNumAtoms())
    for k, p in enumerate(xyz):
        conf.SetAtomPosition(k, p.tolist())
    m.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(m)
    got = _centres(m)
    return all(got.get(a) == want[a] for a in judged)


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
    got = _centres(m)  # only the centres the target specifies: one left open may go either way
    return all(got.get(a) == r for a, r in _centres(mol).items())


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


def _chain_like(mol, unit) -> bool:
    """Whether ``unit`` is one atom of a chain (a CH2, an ether O): in a ring none,
    unbranched, and not in a group that is grown whole."""
    if len(unit) != 1:
        return False
    i = next(iter(unit))
    a = mol.GetAtomWithIdx(i)
    return not a.IsInRing() and len(_heavy_neighbours(mol, i)) <= 2


def steps(mol, core, max_heavy: int = 3) -> list[list[int]]:
    """The order to grow the target's heavy atoms outside ``core`` in: each step
    one unit (units), a chain of acyclic ones up to ``max_heavy`` heavy atoms, or
    a whole flexible chain where its atoms are unbranched links (a butoxy, an
    ethylene glycol), since a step is placed by what it reaches and a bare chain
    reaches nothing: its direction cannot be judged until its end is there."""
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
        chain = _chain_like(mol, unit)
        while not step & ring_atoms and (chain or len(step) < max_heavy):
            nxt = [u for u in left if not u & ring_atoms and touches(u)
                   and (chain and _chain_like(mol, u) or len(step) + len(u) <= max_heavy)
                   and any(_heavy_neighbours(mol, i) & step for i in u)]  # fmt: skip
            if chain:  # only the links themselves; its terminal group is its own step
                nxt = [u for u in nxt if _chain_like(mol, u)]
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
        if not _same_stereo(mol, xyz):
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


def agree(poses, cut: float = AGREE_A) -> list[list[int]]:
    """Group ``poses`` (heavy-atom coordinates in one frame) by whether they are within
    ``cut`` RMSD of each other, nearest first, largest group first.

    Whether several runs found the same thing -- reported, not used to choose.  Across
    24 growths of 9B54's ligand from 6N4B, 19 settled on one pose 4.8 A from the
    crystal's and 3 found it to 0.9 A: the search is biased, not noisy, so the group
    most runs fall into is the wrong one and a majority vote is worse than the score
    (0% within 2.5 A at 8 runs, against 76%).  What it does say is whether the runs
    agree at all, which no single score can.

    Compare the atoms that were grown, not the whole ligand: a held core of 29 heavy
    atoms puts every pose of a 38-atom ligand inside 2 A of every other, whatever its
    guanidinium did."""
    n = len(poses)
    seen, groups = set(), []
    for i in range(n):
        if i in seen:
            continue
        group = [i]
        seen.add(i)
        todo = [i]
        while todo:  # single linkage: a chain of near poses is one group
            k = todo.pop()
            for j in range(n):
                if j in seen:
                    continue
                d = float(np.sqrt(((poses[k] - poses[j]) ** 2).sum(1).mean()))
                if d <= cut:
                    seen.add(j)
                    group.append(j)
                    todo.append(j)
        groups.append(sorted(group))
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups


def spread(confs, new, protein_heavy, n: int) -> list:
    """``n`` of ``confs`` that put the new atoms in as many different places as the
    pocket allows: the least-clashing first, then each time the one whose new atoms sit
    furthest (RMSD) from every one already taken.

    Taking the least-clashing ``n`` instead left them near-duplicates in one corner,
    and embedding more conformers made that worse rather than better -- 120 conformers
    did worse than 60, because the six least-clashing of a larger pool are more alike,
    not less.  Growing 9B54's chain from 6N4B, 7 of 34 placements put its end within
    2 A of the crystal's while only 6 were kept, which is the rate at which the whole
    growth found the right answer."""
    if len(confs) <= n:
        return list(confs)
    scored = [clashes(c, new, protein_heavy) for c in confs]
    order = sorted(range(len(confs)), key=lambda k: (scored[k][0], -scored[k][1]))
    fewest = scored[order[0]][0]
    pool = [k for k in order if scored[k][0] == fewest]
    if len(pool) < n:  # not enough that clash least: take the next best too
        pool = order[: max(n, len(pool))]
    if not new:  # nothing grew this step: the ranking is all there is
        return [confs[k] for k in pool[:n]]
    picked = [pool[0]]
    rest = [k for k in pool[1:]]
    while len(picked) < n and rest:
        far = max(rest, key=lambda k: min(
            float(np.sqrt(((confs[k][new] - confs[q][new]) ** 2).sum(1).mean()))
            for q in picked))  # fmt: skip
        picked.append(far)
        rest.remove(far)
    return [confs[k] for k in picked]


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


def _write_candidates(path, poses) -> None:
    """Every placement tried, its ligand where its MD left it, with its step, parent,
    candidate and score as fields: to see what the search found and rejected.  Beside
    it, ``<path>.npz`` holds each one's receptor as its MD left it too, under the same
    name, so any candidate's complex can be rebuilt and its contacts measured -- the
    ligand alone, against the receptor the search started from, is not a geometry the
    MD ever saw."""
    from rdkit import Chem

    receptors = {}
    with Chem.SDWriter(str(path)) as w:
        for mol, xyz, rec, row in poses:
            name = f"step{row['step']}_from{row['parent']}_c{row['candidate']}"
            if row.get("run"):  # run 0 keeps the plain name it always had
                name = f"run{row['run'] + 1}_{name}"
            receptors[name] = np.asarray(rec, np.float32)
            m = Chem.Mol(mol)
            m.RemoveAllConformers()
            conf = Chem.Conformer(m.GetNumAtoms())
            for k, p in enumerate(xyz):
                conf.SetAtomPosition(k, [float(v) for v in p])
            m.AddConformer(conf, assignId=True)
            m.SetProp("_Name", name)
            for key in ("step", "parent", "candidate", "score", "interaction",
                        "interaction_ligand", "drift", "contacts", "cation_pi",
                        "buried_unsatisfied"):  # fmt: skip
                m.SetProp(key, str(round(row[key], 3) if isinstance(row[key], float) else row[key]))
            m.SetProp("hbonds", " ".join(str(h) for h in row["hbonds"]))
            w.write(m)
    np.savez_compressed(str(path) + ".npz", **receptors)


def _acceptors(anum, hydrogens, ids) -> dict[int, bool]:
    """Which of ``ids`` can accept a hydrogen bond, by whether a lone pair is free:
    an oxygen always can, a nitrogen only with no hydrogen on it.  So a guanidinium
    or ammonium nitrogen, an amide's, Trp's NE1 and a backbone N donate and do not
    accept, while His's unprotonated N, a pyridine's and a tertiary amine's accept.
    A neutral primary or secondary amine is taken to donate alone, which at pH 7 is
    mostly what it does."""
    return {int(i): bool(anum[i] == 8 or not hydrogens.get(int(i))) for i in ids}


def _rings(param, rec_ids, near_xyz, within: float = 12.0):
    """The protein's aromatic rings within ``within`` Å of ``near_xyz`` (nm), as
    arrays of atom indices."""
    res = np.asarray(param.atoms["residue"])
    name = np.asarray(param.atoms["name"])
    rname = np.asarray(param.residues["name"])
    xyz = np.asarray(param.positions, float) / 10.0
    by_residue: dict[int, dict[str, int]] = {}
    for i in rec_ids:
        by_residue.setdefault(int(res[i]), {})[str(name[i])] = int(i)
    out = []
    for r, atoms in by_residue.items():
        key = str(rname[r])[:3].upper()
        key = "HIS" if key.startswith("HI") else key
        for names in AROMATIC.get(key, ()):
            if all(n in atoms for n in names):
                ring = np.array([atoms[n] for n in names])
                if np.linalg.norm(xyz[ring].mean(0) - near_xyz, axis=-1).min() <= within / 10.0:
                    out.append(ring)
    return out


def _normal(ring_xyz):
    """The unit normal of a ring's plane."""
    return np.linalg.svd(ring_xyz - ring_xyz.mean(0))[2][2]


def _cation_centres(lig, lig_ids, new, bonds, anum):
    """The centre of every positively charged group of the new unit: its charged atom
    and the heavy atoms bonded to it, since a guanidinium's charge is spread over all
    three nitrogens and sits on its carbon."""
    formal = np.asarray(lig.atoms["formal_charge"])
    out = []
    for k in new:
        if formal[k] <= 0:
            continue
        i = int(lig_ids[k])
        group = [i] + [j for j in bonds.get(i, ()) if anum[j] > 1]
        out.append(np.array(group))
    return out


def _nonbonded(system):
    """Every particle's charge (e), sigma (Å) and epsilon (kJ/mol), and the nonbonded
    cutoff (Å), read out of an OpenMM system's NonbondedForce."""
    import openmm as mm
    import openmm.unit as u

    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    n = system.getNumParticles()
    q, sigma, epsilon = (np.empty(n) for _ in range(3))
    for i in range(n):
        a, b, c = nb.getParticleParameters(i)
        q[i] = a.value_in_unit(u.elementary_charge)
        sigma[i] = b.value_in_unit(u.angstrom)
        epsilon[i] = c.value_in_unit(u.kilojoule_per_mole)
    return q, sigma, epsilon, nb.getCutoffDistance().value_in_unit(u.angstrom)


def _interaction(x, a, b, q, sigma, epsilon, cutoff: float) -> float:
    """The nonbonded interaction energy of atoms ``a`` with atoms ``b`` at ``x`` (Å),
    in kJ/mol: Coulomb in a reaction field of water's dielectric out to ``cutoff``,
    plus Lennard-Jones, combined as the force field does.  The two are different
    molecules, so no pair of them is bonded or excluded.  Read directly rather than
    from a second OpenMM context, which cannot share a GPU held exclusively; with
    explicit water the MD itself uses PME, which this does not reproduce exactly,
    and periodic images are left out -- it ranks like against like."""
    if not len(a) or not len(b):
        return 0.0
    a, b = np.asarray(a, int), np.asarray(b, int)
    r = np.linalg.norm(x[a][:, None, :] - x[b][None, :, :], axis=2)
    near = (r <= cutoff) & (r > 0.0)
    if not near.any():
        return 0.0
    i, j = np.nonzero(near)
    r = r[near]
    krf = (1.0 / cutoff**3) * (EPSILON_RF - 1.0) / (2.0 * EPSILON_RF + 1.0)
    crf = (1.0 / cutoff) * (3.0 * EPSILON_RF) / (2.0 * EPSILON_RF + 1.0)
    coulomb = ONE_4PI_EPS0 * q[a[i]] * q[b[j]] * (1.0 / r + krf * r**2 - crf)
    s = 0.5 * (sigma[a[i]] + sigma[b[j]])
    e = np.sqrt(epsilon[a[i]] * epsilon[b[j]])
    sr6 = (s / r) ** 6
    return float((coulomb + 4.0 * e * (sr6**2 - sr6)).sum())


def md_score(receptor, lig, core, new, ff, args, ps: float = 100.0, seed: int = 1,
             water: bool = False, pocket_a: float = POCKET_A, loose=(),
             secondary: bool = True, log=print) -> dict:  # fmt: skip
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
    loose = set(loose)
    for k_, i in zip(core, lig_ids[list(core)], strict=True):
        if anum[i] > 1 and k_ not in loose:
            hold.addParticle(int(i), [CORE_KJ, *xyz0[i]])
    if loose:  # free within LOOSE_A of where they are, held beyond
        dist = "periodicdistance(x,y,z,x0,y0,z0)" if water else "sqrt((x-x0)^2+(y-y0)^2+(z-z0)^2)"
        wall = mm.CustomExternalForce(f"0.5*{CORE_KJ}*max(0, {dist}-{LOOSE_A / 10.0})^2")
        for p in ("x0", "y0", "z0"):
            wall.addPerParticleParameter(p)
        for k_ in sorted(loose):
            i = lig_ids[k_]
            if anum[i] > 1:
                wall.addParticle(int(i), list(xyz0[i]))
        system.addForce(wall)
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
    if secondary:  # helices and sheets kept as they are; the pocket's loops and side chains free
        from ..md.restraints import add_dihedral_restraints

        try:
            add_dihedral_restraints(system, param, "ss", SS_KJ)
        except ValueError:  # no helix or sheet to hold
            pass
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
    neighbours: dict[int, list[int]] = {}
    for i, j in zip(np.asarray(tab["i"]), np.asarray(tab["j"]), strict=True):
        neighbours.setdefault(int(i), []).append(int(j))
        neighbours.setdefault(int(j), []).append(int(i))
    accepts = _acceptors(anum, hydrogens, polar_all)
    cations = _cation_centres(lig, lig_ids, new, neighbours, anum)
    rings = _rings(param, rec, xyz0[lig_heavy]) if cations else []
    q, sigma, epsilon, cutoff_a = _nonbonded(system)
    # the unit the energy is of: its heavy atoms and the hydrogens on them.  Without
    # them a guanidinium is N,C,N,N at about -1.6 e instead of the +1 e it really
    # carries, and its electrostatics with the protein come out the wrong way round
    unit = np.array(sorted({int(i) for i in new_ids}
                           | {j for i in new_ids for j in neighbours.get(int(i), ())
                              if anum[j] == 1}), int)  # fmt: skip
    energies: list[float] = []
    whole: list[float] = []
    hb = {i: 0 for i in polar_new}
    wet = {i: 0.0 for i in polar_new}
    cation_pi = 0
    drift, contacts, frames = [], [], 0
    n_frames = max(1, int(round(ps / 2.0)))
    for _ in range(n_frames):
        integ.step(every)
        x = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.angstrom)
        if start is None:
            start = x[new_ids].copy()
        frames += 1
        # what the score is: the new unit against the protein, this frame
        energies.append(_interaction(x, unit, rec, q, sigma, epsilon, cutoff_a))
        whole.append(_interaction(x, lig_ids, rec, q, sigma, epsilon, cutoff_a))
        touching = False  # cation-pi: a charged group of the unit on an aromatic ring
        for centre in cations:
            p = x[centre].mean(0)
            for ring in rings:
                rx = x[ring]
                v = p - rx.mean(0)
                perp = abs(float(v @ _normal(rx)))
                off = float(np.sqrt(max(float(v @ v) - perp**2, 0.0)))
                if perp <= CATION_PI_A and off <= CATION_PI_OFFSET_A:
                    touching = True
                    break
            if touching:
                break
        cation_pi += touching
        tree = cKDTree(x[polar_all])
        for i in polar_new:
            near = [int(polar_all[k]) for k in tree.query_ball_point(x[i], HBOND_A)]
            ok = False
            for j in near:
                if j == i or j in bonded[i]:
                    continue
                for donor, acceptor in ((j, i), (i, j)):
                    if not accepts[acceptor]:  # two donors cannot bond to each other
                        continue
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
    interaction = float(np.mean(energies)) if energies else 0.0
    # the whole ligand's, reported beside it: cutting the unit off its chain leaves it
    # a few tenths of an electron out, while the ligand is whole and balanced
    interaction_all = float(np.mean(whole)) if whole else 0.0
    score = -interaction / INTERACTION_KJ - 0.5 * mean_drift + 0.01 * mean_contacts
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 200)
    x = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.angstrom)
    ref = np.asarray(receptor.positions, float)
    # back into the receptor's frame: the box moved everything by one vector
    shift = (xyz0[rec] * 10.0 - ref).mean(0)
    return {"score": float(score), "interaction": interaction,
            "interaction_ligand": interaction_all, "drift": mean_drift,
            "contacts": mean_contacts, "cation_pi": cation_pi / frames,
            "hbonds": [round(persistence[i], 2) for i in polar_new],
            "buried_unsatisfied": sum(1 for i in buried if persistence[i] < 0.5),
            "receptor": x[rec] - shift, "ligand": x[lig_ids] - shift}  # fmt: skip


# --- growing one target -----------------------------------------------------------------


def grow(receptor, reference, target, name: str, ff, args, workdir, beam: int = 2,
         keep: int = 3, n_conformers: int = 60, md_ps: float = 100.0, max_heavy: int = 3,
         water: bool = False, hold: str = "rings", repeats: int = 1, seed: int = 0,
         log=print) -> Path:  # fmt: skip
    """``target`` (RDKit, with hydrogens) grown from ``reference`` (posed in
    ``receptor``) step by step (the module docstring); its complex's path.

    ``repeats``: grow it this many times and keep the best-scoring run.  The MD is
    not reproducible on a GPU, so two runs of the same search take different
    branches: growing 9B54's ligand from 6N4B five times reached the crystal's
    salt bridge once, and the other four settled a few Angstrom away.  Repeats
    buy that back, and with a reaction field and no explicit water a placement
    costs seconds."""
    from ..io import save
    from . import relief

    work = Path(workdir)
    (work / "complexes").mkdir(parents=True, exist_ok=True)
    (work / "grow").mkdir(parents=True, exist_ok=True)
    if repeats > 1:
        (work / "runs").mkdir(parents=True, exist_ok=True)
    core = core_map(reference, target)
    ref_xyz = reference.GetConformer().GetPositions()
    plan = steps(target, core, max_heavy)
    if not plan:  # nothing to grow: the core is the whole target; settle its hydrogens
        plan = [[]]
    log(f"{name}: core {len(core)} heavy atoms from the reference; "
        f"{sum(len(s) for s in plan)} to grow in {len(plan)} step(s): "
        + " | ".join(" ".join(target.GetAtomWithIdx(i).GetSymbol() for i in s) or "-"
                     for s in plan))  # fmt: skip
    # held firmly: the core's ring atoms; its other atoms and everything grown are
    # held only loosely (hold="rings"), since a reference's ester or tBu need not sit
    # where the target's do (6N4B's and 9B54's shared core differ by 2.3 Å)
    rings = set().union(*_ring_systems(target)) if target.GetRingInfo().NumRings() else set()
    firm = {i for i in core if i in rings} if hold == "rings" else set(core)
    record = {"name": name, "core": len(core), "steps": [], "hold": hold,
              "repeats": repeats}  # fmt: skip
    poses = []  # every placement's ligand after its MD, for grow/NAME_candidates.sdf

    def search(rep: int):
        """One growth, its seeds shifted by ``rep``: (best, index, lig_ff)."""
        offset = seed + 100_000 * rep
        tag = f"run {rep + 1}/{repeats}: " if repeats > 1 else ""
        states = [State(np.asarray(receptor.positions, float),
                        {t: ref_xyz[r] for t, r in core.items()})]  # fmt: skip
        placed = set(core)
        best = index = lig_ff = None
        for k, step in enumerate(plan, 1):
            placed |= set(step)
            mol, index = intermediate(target, placed)
            new = [index[i] for i in step]
            # a step that adds no polar atom cannot be judged: a bare chain makes no
            # hydrogen bond wherever it points, and its score is drift and contacts
            # alone.  More of its placements are kept and every one goes on, to be
            # judged by what the next step reaches from it
            blind = not any(target.GetAtomWithIdx(i).GetAtomicNum() in (7, 8, 16) for i in step)
            here_keep = 2 * keep if blind and step else keep
            # a long chain has a torsion at every bond: more conformers to draw from
            here_confs = n_conformers * max(1, (len(step) + 2) // 3)
            found = []
            lig_ff = None
            for s, state in enumerate(states):
                fixed = {index[i]: xyz for i, xyz in state.placed.items() if i in index}
                confs = embed(mol, fixed, n=here_confs, seed=offset + 7 + 31 * k + s)
                rec = receptor.copy()
                rec.positions = state.receptor
                prot_heavy = state.receptor[np.asarray(rec.atoms["anum"]) > 1]
                for c, xyz in enumerate(spread(confs, new, prot_heavy, here_keep)):
                    lig = _ligand(mol, xyz)
                    if lig_ff is None:
                        lig_ff = _forcefield(rec, lig, ff, args)
                    here = rec.copy()
                    relief.relieve(here, [lig], [name], lig_ff, args, log=lambda *_: None)
                    core_ids = [index[i] for i in state.placed if i in index]
                    loose = [index[i] for i in state.placed if i in index and i not in firm]
                    out = md_score(here, lig, core_ids, new, lig_ff, args, ps=md_ps,
                                   seed=offset + 1000 * k + 10 * s + c, water=water,
                                   loose=loose if hold == "rings" else (), log=log)  # fmt: skip
                    row = {k_: v for k_, v in out.items() if k_ not in ("receptor", "ligand")}
                    row.update(step=k, parent=s, candidate=c, run=rep)
                    record["steps"].append(row)
                    poses.append((mol, out["ligand"], out["receptor"], row))
                    if out["drift"] > MAX_DRIFT:
                        continue
                    if not _same_stereo(mol, out["ligand"]):  # a centre inverted in the MD
                        row["inverted"] = True
                        continue
                    heavy = target.GetAtomWithIdx
                    back = {i: out["ligand"][j] for i, j in index.items()
                            if heavy(i).GetAtomicNum() > 1 and i in placed}  # fmt: skip
                    state_ = State(out["receptor"], back, out["score"], state.history + [row])
                    found.append((out["score"], state_, out, mol))
            if not found:
                raise RuntimeError(f"{name}: no placement of step {k} stays where it was put")
            found.sort(key=lambda f: -f[0])
            states = [f[1] for f in found] if blind and step else [f[1] for f in found[:beam]]
            best = found[0]
            log(f"  {tag}step {k}: {len(found)} placements kept"
                + (", all carried on (no polar atom to judge them by)"
                   if blind and step else "")
                + f"; best score {best[0]:.2f} "
                f"({best[2]['interaction']:.0f} kJ/mol with the protein, the whole ligand "
                f"{best[2]['interaction_ligand']:.0f}, hydrogen bonds "
                f"{best[2]['hbonds']}, buried unsatisfied {best[2]['buried_unsatisfied']}, "
                f"cation-pi {best[2]['cation_pi']:.2f}, "
                f"drift {best[2]['drift']:.2f} Å)")  # fmt: skip
        return best, index, lig_ff

    runs = []
    for rep in range(max(1, repeats)):
        got = search(rep)
        runs.append(got)
        if repeats > 1:
            log(f"  run {rep + 1}/{repeats}: best score {got[0][0]:.2f}")

    def finish(got):
        """A run's last relaxation: the whole ligand loosely held, the core's rings
        firm.  Let the rings loose here too and they drift the whole LOOSE_A away,
        taking the pose with them, while what this relaxation is for is the rest."""
        best_, index_, lig_ff_ = got
        _, final_, out_, mol_ = best_
        if hold != "rings":
            return final_, out_, mol_, index_, lig_ff_
        rec = receptor.copy()
        rec.positions = final_.receptor
        lig = _ligand(mol_, out_["ligand"])
        every = list(range(mol_.GetNumAtoms()))
        firm_rings = {index_[i] for i in firm if i in index_}
        last = md_score(rec, lig, every, [], lig_ff_, args, ps=2 * md_ps, seed=99,
                        water=water, loose=[i for i in every if i not in firm_rings],
                        log=log)  # fmt: skip
        if _same_stereo(mol_, last["ligand"]):
            out_ = {**out_, "ligand": last["ligand"]}
            final_ = State(last["receptor"], final_.placed, final_.score, final_.history)
            log(f"  relaxed whole: the core's rings held, the rest within {LOOSE_A:g} Å "
                "of where it grew")  # fmt: skip
        return final_, out_, mol_, index_, lig_ff_

    def rescore(final_, out_, mol_, lig_ff_):
        """The whole ligand's interaction energy with the protein, under one protocol
        for every run: the same length, the same seed, the same restraints.

        A run's own score comes from its own trajectory at its own step, which ranks
        runs only weakly.  Measured the same way for all of them it ranks better where
        it matters: over 48 poses the two agreed most of the time, and of the times
        they did not this picked the closer pose in 92% (within 2.5 A at 8 runs: 82%
        against 75%, and within 1.5 A 20% against 13%)."""
        rec = receptor.copy()
        rec.positions = final_.receptor
        lig = _ligand(mol_, out_["ligand"])
        every = list(range(mol_.GetNumAtoms()))
        heavy = [i for i in every if mol_.GetAtomWithIdx(i).GetAtomicNum() > 1]
        got = md_score(rec, lig, every, heavy, lig_ff_, args, ps=md_ps, seed=RESCORE_SEED,
                       water=water, loose=every, log=lambda *_: None)  # fmt: skip
        return float(got["interaction"])

    if repeats > 1:
        done = [finish(g) for g in runs]
        energies = [rescore(f, o, m, ff_) for f, o, m, _, ff_ in done]
        index = done[0][3]
        grown = sorted(set(index.values()) - {index[i] for i in core if i in index})
        groups = (agree([o["ligand"][grown] for _, o, _, _, _ in done]) if grown
                  else [list(range(len(done)))])  # fmt: skip
        # the most negative energy, and nothing else.  Measured on 18 growths of 9B54's
        # ligand from 6N4B, this alone put the kept pose within 2.5 Å of the crystal in
        # 17 of 18; it beat each run's own score in 92% of the cases where the two
        # disagreed.  Grouping the poses is reported below but deliberately not used to
        # choose: the right pose is a minority across replicates, so anything that
        # rewards agreement selects the common wrong one (0% within 2.5 Å against 76%).
        # Vetoing only the lone poses in a near-tie was tried too and dropped -- over 10
        # held-out trials it changed the pick twice and improved it neither time
        chosen = min(range(len(energies)), key=lambda k: energies[k])
        final, out, mol, index, lig_ff = done[chosen]
        # every run's pose kept, not only the one chosen: the others are what a second
        # opinion would be built from
        for k, (f_, o_, m_, _, _) in enumerate(done):
            each = receptor.copy()
            each.positions = f_.receptor
            each.append(_ligand(m_, o_["ligand"]))
            save(each, work / "runs" / f"{name}_run{k + 1}.mae")
        log(
            f"{name}: {repeats} runs, {' '.join(f'{e:.0f}' for e in energies)} kJ/mol "
            f"with the protein measured alike; their grown parts fall in "
            f"{len(groups)} group(s) of {', '.join(str(len(g)) for g in groups)}.  "
            f"Keeping run {chosen + 1}, the lowest of them; "
            f"{sum(len(g) for g in groups if chosen in g) - 1} other run(s) put the "
            f"grown part where it did.  All {repeats} are in runs/"
        )
        record["run_scores"] = [float(g[0][0]) for g in runs]
        record["rescored"] = [round(e, 1) for e in energies]
        record["agreement"] = [[int(k) for k in g] for g in groups]
        record["kept_run"] = chosen
    else:
        final, out, mol, index, lig_ff = finish(runs[0])
    complex_ = receptor.copy()
    complex_.positions = final.receptor
    complex_.append(_ligand(mol, out["ligand"]))
    path = work / "complexes" / f"{name}.mae"
    save(complex_, path)
    record["score"] = final.score
    (work / "grow" / f"{name}.json").write_text(json.dumps(record, indent=1, default=float))
    _write_candidates(work / "grow" / f"{name}_candidates.sdf", poses)
    return path
