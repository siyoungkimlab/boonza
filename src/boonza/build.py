"""Building simulation systems: solvating, adding ions, repartitioning hydrogen masses.

    s = boonza.solvate(protein, thickness=10.0)                 # TIP3P box around it
    s = boonza.neutralize(s, concentration=0.15)                # Na+/Cl-, 150 mM
    s = boonza.repartition_hydrogen_masses(s, "not water", 3.024)

These follow msys's ``dms-solvate``, ``dms-neutralize`` and ``dms-hmr``
tools step by step, so the same inputs give the same systems (the tests
compare with msys).  The default solvent is the TIP3P box distributed with
msys.  Added waters and ions carry no force-field terms, as in msys:
parameterize the finished system (or add the terms) before simulating.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

from .system import System

WATER_BOX = Path(__file__).parent / "data" / "tip3p.dms"
WATER_RADIUS = 2.4  # Å between solvent (oxygens) and solute
WATER_CONTACT = 1.0  # Å between solvent atoms across the periodic boundary

# msys GroupForElement for the ions it can make from an element name
_ION_CHARGE = {3: 1, 11: 1, 19: 1, 37: 1, 55: 1, 4: 2, 12: 2, 20: 2, 38: 2, 56: 2,
               9: -1, 17: -1, 35: -1, 53: -1}  # fmt: skip


def _load(what) -> System:
    if isinstance(what, System):
        return what.copy()
    from .io import load

    return load(what)


def solvate(solute: System, solvent=None, box=None, thickness: float = 5.0,
            min_solute_dist: float = WATER_RADIUS,
            min_solvent_dist: float = WATER_CONTACT, solvent_selection: str = "oxygen",
            center_selection: str = "all") -> System:  # fmt: skip
    """Tile a solvent box around ``solute`` and remove overlaps (msys ``dms-solvate``).

    ``box``: the box edge lengths (one value or three, Å); by default a cube
    of the solute's largest extent plus ``thickness`` on each side.  The
    solute is first centered on ``center_selection`` ("none" to skip).
    ``solvent``: a System or file with a periodic cell (default: the
    bundled TIP3P box).  Solvent molecules are removed when a
    ``solvent_selection`` atom is within ``min_solute_dist`` of the solute
    (periodically), when their center lies outside the box, or when any of
    their atoms is within ``min_solvent_dist`` of a periodic image of the
    solvent.  Water chains are named W1, W2, ... with residues numbered from 1.
    """
    mol = solute.copy()
    wat = _load(WATER_BOX if solvent is None else solvent)
    npro = mol.natoms
    if box is None:
        pos = mol.positions
        extent = float((pos.max(0) - pos.min(0)).max()) if npro else 0.0
        dims = np.full(3, extent + 2 * thickness)
    else:
        dims = np.broadcast_to(np.asarray(box, dtype=np.float64), (3,)).copy()
    if center_selection != "none" and npro:
        ids = mol.select(center_selection).ids
        if not len(ids):
            raise ValueError(f"center selection {center_selection!r} selected no atoms")
        pos = mol.positions
        mol.positions = pos - pos[ids].mean(0)
    mol.cell = np.diag(dims)
    watsize = np.diag(wat.cell).copy()
    if (watsize <= 0).any():
        raise ValueError(f"the solvent has no usable periodic cell: {wat.cell.tolist()}")

    center = mol.positions.mean(0) if npro else np.zeros(3)
    dmin, dmax = center - 0.5 * dims, center + 0.5 * dims
    nrep = (dims / watsize).astype(int) + 1
    shift = -0.5 * (nrep - 1) * watsize
    mol.append(_tiled(wat, nrep, shift, watsize))

    # solvent molecules too close to the solute
    mol = mol.clone(f"not same fragid as (index >= {npro} and ({solvent_selection}) and "
                    f"(pbwithin {min_solute_dist} of index < {npro}))")  # fmt: skip
    # solvent molecules whose center (of solvent_selection atoms) is outside the box
    pos = mol.positions
    frag = np.asarray(mol.fragids)
    solvent = np.arange(mol.natoms) >= npro
    out = solvent & ((pos < dmin).any(1) | (pos > dmax).any(1))
    if out.any():
        chosen = np.zeros(mol.natoms, bool)
        chosen[mol.select(solvent_selection).ids] = True
        drop = []
        for f in np.unique(frag[out]).tolist():
            ids = np.flatnonzero(chosen & (frag == f))
            if len(ids):
                c = pos[ids].mean(0)
                if (c < dmin).any() or (c > dmax).any():
                    drop.append(f)
        if drop:
            mol = mol.clone(np.flatnonzero(~np.isin(frag, drop)))
    mol = _remove_periodic_contacts(mol, npro, min_solvent_dist)

    # name the water chains W1, W2, ... and number their residues from 1
    solvent_ct = mol.ncts - 1
    chains = np.flatnonzero(mol.chains["ct"] == solvent_ct)
    names = mol.chains["name"].copy()
    resid = mol.residues["resid"].copy()
    res_chain = mol.residues["chain"]
    for n, c in enumerate(chains.tolist()):
        names[c] = f"W{n + 1}"
        rows = np.flatnonzero(res_chain == c)
        resid[rows] = np.arange(1, len(rows) + 1)
    mol.chains["name"] = names
    mol.residues["resid"] = resid
    return mol


def _tiled(wat: System, nrep, shift, watsize) -> System:
    """Copies of the solvent box on an nrep grid, all in one ct named "solvate"."""
    nx, ny, nz = (int(x) for x in nrep)
    copies = [(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)]
    n = wat.natoms
    res = wat.atoms["residue"]
    chain = wat.residues["chain"][res]
    pos = np.concatenate([wat.positions + shift + watsize * np.array(c) for c in copies])
    # a separate chain per copy, as msys's append makes: tag copies apart, then restore
    segid = wat.chains["segid"][chain].tolist() * len(copies)
    tags = [f"{sg}|{k // n}" for k, sg in enumerate(segid)]
    columns = {name: np.tile(wat._atoms.column(name), (len(copies),) + (1,) *
                             (wat._atoms.column(name).ndim - 1))
               for name in wat._atoms.names
               if name not in ("residue", "name", "anum", "pos")}  # fmt: skip
    bonds = np.column_stack([wat.bonds["i"], wat.bonds["j"]])
    block = System.from_arrays(
        pos, names=np.tile(wat.atoms["name"], len(copies)),
        anum=np.tile(wat.atoms["anum"], len(copies)),
        resnames=np.tile(wat.residues["name"][res], len(copies)),
        resids=np.tile(wat.residues["resid"][res], len(copies)),
        chains=np.tile(wat.chains["name"][chain], len(copies)),
        segids=tags,
        insertions=np.tile(wat.residues["insertion"][res], len(copies)),
        bonds=(bonds[None, :, :] + n * np.arange(len(copies))[:, None, None]).reshape(-1, 2),
        **columns,
    )  # fmt: skip
    block.chains["segid"] = np.array([s.rsplit("|", 1)[0] for s in block.chains["segid"].tolist()])
    block._ct_names[0] = "solvate"
    return block


def _remove_periodic_contacts(mol: System, npro: int, dist: float) -> System:
    """Drop solvent molecules with an atom within ``dist`` of a periodic image of the solvent."""
    from .pbc import capped_distances

    pos = mol.positions
    solvent = pos[npro:]
    a, b, c = mol.cell
    bad = set()
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                if i == j == k == 0:
                    continue
                ii, _, d = capped_distances(solvent, solvent + i * a + j * b + k * c, dist)
                bad.update((ii[d <= dist] + npro).tolist())
    if not bad:
        return mol
    frag = np.asarray(mol.fragids)
    drop = np.unique(frag[sorted(bad)])
    return mol.clone(np.flatnonzero(~np.isin(frag, drop)))


# ---------------------------------------------------------------------------
# ions


def _ion(name) -> dict:
    from .elements import element_for_abbreviation, msys_symbol

    anum = element_for_abbreviation(name.capitalize()) if isinstance(name, str) else int(name)
    if anum not in _ION_CHARGE:
        raise ValueError(f"cannot make an ion of {name!r}: not an alkali, alkaline earth or "
                         "halogen element")  # fmt: skip
    return {"anum": anum, "name": msys_symbol(anum), "charge": float(_ION_CHARGE[anum])}


def _residue_center(mol: System, residue: int) -> np.ndarray:
    """Mass-weighted center of a residue (zero if it has no mass, as in msys)."""
    ids = np.flatnonzero(mol.atoms["residue"] == residue)
    m = mol.atoms["mass"][ids]
    total = m.sum()
    if not total:
        return np.zeros(3)
    return (m[:, None] * mol.positions[ids]).sum(0) / total


def neutralize(system: System, cation="Na", anion="Cl", charge="formal_charge",
               chain: str = "ION", chain2: str = "ION2", solute_pad: float = 5.0,
               ion_pad: float = 3.0, water_pad: float = 0.0, concentration: float = 0.0,
               keep: str = "none", random_seed: int = 0) -> System:  # fmt: skip
    """Replace water molecules with ions (msys ``dms-neutralize``).

    Enough counterions are added to cancel the solute's charge (the sum of
    ``charge``: "formal_charge", "charge", or a number), then ion pairs up to
    ``concentration`` (mol/L, counted against the waters as msys does).
    Waters within ``solute_pad`` of anything that is not water, or matching
    ``keep``, are never replaced; they are picked in a random order fixed by
    ``random_seed``.  As in msys, candidate waters closer than
    ``ion_pad`` squared (Å) to an earlier pick are skipped.  Ions sit at the
    replaced water's mass-weighted center, in a new ct: counterions in chain
    ``chain`` and the others in ``chain2``, numbered from 1.  Unlike msys,
    ions get their element mass.
    """
    mol = system.copy()
    cat, an = _ion(cation), _ion(anion)
    if cat["anum"] == an["anum"]:
        raise ValueError("cation and anion are the same element")
    frag = np.asarray(mol.fragids)
    size = np.bincount(frag)
    single = size[frag] == 1
    anum = mol.atoms["anum"]
    existing = {}
    for ion in (cat, an):
        atoms = np.flatnonzero(single & (anum == ion["anum"]))
        existing[ion["anum"]] = sorted(set(frag[atoms].tolist()))
    ion_frags = existing[cat["anum"]] + existing[an["anum"]]
    solute = ~np.isin(frag, ion_frags)
    if charge == "formal_charge":
        cg = float(mol.atoms["formal_charge"][solute].sum())
    elif charge == "charge":
        cg = float(mol.atoms["charge"][solute].sum())
    else:
        cg = float(charge)
    ion, other = (an, cat) if cg >= 0 else (cat, an)
    nions_prev, nother_prev = len(existing[ion["anum"]]), len(existing[other["anum"]])
    nions = int(math.fabs(cg / ion["charge"]) + 0.5)

    water = mol.select(f"water and noh and (not pbwithin {solute_pad} of (not water)) "
                       f"and (not ({keep}))").ids  # fmt: skip
    residues = sorted(set(mol.atoms["residue"][water].tolist()))
    nwat = len(residues)
    ntotalwat = len(set(mol.atoms["residue"][mol.select("water").ids].tolist()))
    cgratio = math.fabs(other["charge"] / ion["charge"])
    nother = int((concentration / 55.345) * (ntotalwat - nions + nions_prev))
    nions += int(nother * cgratio)
    nions -= nions_prev
    nother -= nother_prev
    removed_ions = []
    for count, key in ((nions, ion["anum"]), (nother, other["anum"])):
        if count < 0:
            if len(existing[key]) < -count:
                raise ValueError("cannot lower the concentration: not enough ions to delete")
            sel = mol.select(f"fragid {' '.join(map(str, existing[key]))} and not ({keep})").ids
            removed_ions.extend(sel[:-count].tolist())
    nions, nother = max(nions, 0), max(nother, 0)
    if nwat < nions + nother:
        raise ValueError(f"only {nwat} waters can be replaced; not enough for the ions")

    random.seed(random_seed)
    random.shuffle(residues)
    centers: dict[int, np.ndarray] = {}
    pad2 = ion_pad * ion_pad
    for i in range(nions + nother):
        if i >= len(residues):
            raise ValueError("not enough waters, or ion_pad too large")
        pi = centers.setdefault(residues[i], _residue_center(mol, residues[i]))
        j = i + 1
        while j < nions + nother:
            if len(residues) < nions + nother:
                raise ValueError("not enough waters, or ion_pad too large")
            pj = centers.setdefault(residues[j], _residue_center(mol, residues[j]))
            if math.sqrt(float(((pi - pj) ** 2).sum())) < pad2:
                del residues[j]
            else:
                j += 1

    keep_res = set(mol.atoms["residue"][mol.select(f"({keep})").ids].tolist())
    real = mol.atoms["anum"] > 0
    placed = []  # (center, ion, chain name, resid)
    for i in range(nions + nother):
        r = residues[i]
        if int((real & (mol.atoms["residue"] == r)).sum()) > 3:
            raise ValueError(f"residue {r} holds several waters; split it into residues first")
        kind, cname, rid = (ion, chain, i + 1) if i < nions else (other, chain2, i + 1 - nions)
        placed.append((_residue_center(mol, r), kind, cname, rid))
    replaced = residues[: nions + nother]
    assert not set(replaced) & keep_res

    doomed = np.isin(mol.atoms["residue"], replaced)
    doomed[removed_ions] = True
    mol = mol.clone(np.flatnonzero(~doomed))
    if placed:
        from .elements import MSYS_MASSES

        ions = System.from_arrays(
            np.array([p[0] for p in placed]),
            names=[p[1]["name"] for p in placed], anum=[p[1]["anum"] for p in placed],
            resnames=[""] * len(placed), resids=[p[3] for p in placed],
            chains=[p[2] for p in placed],
            charge=[p[1]["charge"] for p in placed],
            formal_charge=[int(p[1]["charge"]) for p in placed],
            mass=[float(MSYS_MASSES[p[1]["anum"]]) for p in placed],
        )  # fmt: skip
        ions._ct_names[0] = "ion"
        mol.append(ions)
        return mol.clone(f"not (same residue as (water and pbwithin {water_pad} of "
                         f"ct {mol.ncts - 1}))")  # fmt: skip
    return mol


# ---------------------------------------------------------------------------
# hydrogen masses


def repartition_hydrogen_masses(system: System, selection: str = "not water",
                                mass: float = 3.024,
                                repartition: bool = True) -> System:  # fmt: skip
    """Set the mass of the hydrogens in ``selection`` (msys ``dms-hmr``).

    With ``repartition`` the added mass is taken from the heavy atom each
    hydrogen is bonded to, so the total mass is unchanged (hydrogen mass
    repartitioning, for 4 fs time steps); without it only the hydrogens
    change (for example deuterium, 2.014).
    """
    s = system.copy()
    masses = s.atoms["mass"].astype(np.float64).copy()
    before = masses.sum()
    for h in s.select(f"hydrogen and ({selection})").ids.tolist():
        old = masses[h]
        masses[h] = mass
        if repartition:
            partners = s.bonded_atoms(h)
            if not len(partners):
                raise ValueError(f"hydrogen {h} has no bond to take mass from")
            b = int(partners[0])
            masses[b] -= mass - old
            if masses[b] <= 0:
                raise ValueError(f"atom {b} ({s.atom(b).name}) would get mass {masses[b]}")
    if repartition and abs(masses.sum() - before) > 1e-3:
        raise ValueError("the total mass changed while repartitioning")
    s.atoms["mass"] = masses
    return s


# ---------------------------------------------------------------------------
# from text: SMILES and sequences

HELIX = (-57.0, -47.0)
SHEET = (-120.0, 130.0)
EXTENDED = (180.0, 180.0)
POLYPROLINE = (-75.0, 145.0)
CONFORMATIONS = {"helix": HELIX, "alpha": HELIX, "sheet": SHEET, "beta": SHEET,
                 "extended": EXTENDED, "polyproline": POLYPROLINE}  # fmt: skip


def _unique_names(s: System) -> None:
    """Atom names element + count (C1, C2, ..., H1, ...), as ligand files name them."""
    from .elements import msys_symbol

    counts: dict[str, int] = {}
    names = []
    for z in s.atoms["anum"].tolist():
        sym = msys_symbol(int(z)) or "X"
        counts[sym] = counts.get(sym, 0) + 1
        names.append(f"{sym}{counts[sym]}")
    s.atoms["name"] = np.array(names)


def from_smiles(smiles: str, name: str = "LIG", seed: int = 42, optimize: bool = True,
                conformers: int = 1) -> System:  # fmt: skip
    """A 3D molecule from a SMILES string (through RDKit).

    Hydrogens are added, ``conformers`` conformers are embedded (RDKit ETKDG,
    reproducible with ``seed``) and, with ``optimize``, minimized with MMFF94
    (UFF when MMFF lacks parameters); the lowest-energy one is kept.  Bond
    orders and formal charges come from the SMILES.  The molecule is one
    residue named ``name`` with atoms named C1, C2, ..., H1, ...
    """
    from rdkit.Chem import AllChem

    from .chem import _chem, from_rdkit

    Chem = _chem()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse the SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=max(1, int(conformers)), params=params))
    if not ids:
        raise ValueError(f"RDKit could not embed {smiles!r} in 3D")
    best = ids[0]
    if optimize:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            result = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
        else:
            result = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)
        best = ids[int(np.argmin([energy for _, energy in result]))]
    s = from_rdkit(mol, conf_id=best, name=name)
    s.residues["name"] = np.full(s.nresidues, name)
    _unique_names(s)
    return s


def peptide(sequence: str, conformation="helix", seed: int = 0, optimize: bool = True) -> System:
    """A peptide built from a one-letter sequence (through RDKit).

    ``conformation``: "helix", "sheet", "extended", "polyproline", one
    (phi, psi) pair, or one pair per residue (degrees).  The chain is built
    by RDKit with PDB atom and residue names, every peptide bond is set
    trans, and phi/psi are set residue by residue (proline's phi is fixed by
    its ring).  With ``optimize`` the structure is minimized with MMFF94
    while phi/psi are held, so side chains relax without losing the
    backbone (omega included).  Termini are free amine and acid, as RDKit
    builds them.
    """
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdMolTransforms as T

    from .chem import _chem, from_rdkit

    Chem = _chem()
    seq = sequence.strip().upper()
    mol = Chem.MolFromSequence(seq) if seq else None
    if mol is None:
        raise ValueError(f"cannot build a peptide from {sequence!r}")
    n = len(seq)
    if isinstance(conformation, str):
        if conformation not in CONFORMATIONS:
            raise ValueError(f"conformation must be one of {sorted(CONFORMATIONS)} or angles")
        angles = [CONFORMATIONS[conformation]] * n
    else:
        arr = np.asarray(conformation, dtype=np.float64)
        angles = [tuple(arr)] * n if arr.shape == (2,) else [tuple(r) for r in arr]
        if len(angles) != n:
            raise ValueError(f"got {len(angles)} (phi, psi) pairs for {n} residues")
    mol = Chem.AddHs(mol, addResidueInfo=True)
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True  # RDKit's advice for large, flexible molecules
    for attempt in range(5):
        params.randomSeed = int(seed) + attempt
        if AllChem.EmbedMolecule(mol, params) >= 0:
            break
    else:
        raise ValueError(f"RDKit could not embed the peptide {seq}")
    backbone: dict[int, dict[str, int]] = {}
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is not None and info.GetName().strip() in ("N", "CA", "C"):
            backbone.setdefault(info.GetResidueNumber(), {})[info.GetName().strip()] = atom.GetIdx()
    res = [backbone[k] for k in sorted(backbone)]
    conf = mol.GetConformer()
    held = []
    for k in range(n):
        phi, psi = angles[k]
        r = res[k]
        if k > 0:
            p = res[k - 1]
            T.SetDihedralDeg(conf, p["CA"], p["C"], r["N"], r["CA"], 180.0)  # omega
            held.append((p["CA"], p["C"], r["N"], r["CA"], 180.0))
            if seq[k] != "P":
                T.SetDihedralDeg(conf, p["C"], r["N"], r["CA"], r["C"], phi)
                held.append((p["C"], r["N"], r["CA"], r["C"], phi))
        if k < n - 1:
            T.SetDihedralDeg(conf, r["N"], r["CA"], r["C"], res[k + 1]["N"], psi)
            held.append((r["N"], r["CA"], r["C"], res[k + 1]["N"], psi))
    if optimize and AllChem.MMFFHasAllMoleculeParams(mol):
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
        for a, b, c, d, value in held:  # RDKit handles windows that cross +-180
            ff.MMFFAddTorsionConstraint(a, b, c, d, False, value - 1.0, value + 1.0, 1e4)
        ff.Minimize(maxIts=5000)
    return from_rdkit(mol, name=f"peptide {seq}")
