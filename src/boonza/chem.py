"""RDKit bridge: molecules of a System to RDKit and back, and bond-order perception.

    mol = boonza.to_rdkit(system, atom.fragment)   # one molecule
    mol = system.select("resname LIG").to_rdkit()
    sys = boonza.from_rdkit(mol)
    boonza.assign_bond_orders(system, "resname LIG")  # PDB/GRO/CIF ligands

Every RDKit atom carries its system index as the int property
``boonza_index`` and its atom name as ``_Name``; residue data is stored as
PDB residue info.  Hydrogens are explicit on both sides.
"""

from __future__ import annotations

import numpy as np

from ._columns import STR
from .elements import msys_symbol
from .system import System


def _chem():
    try:
        from rdkit import Chem
    except ImportError as e:
        raise ImportError(
            "the RDKit bridge needs rdkit (conda install -c conda-forge rdkit)"
        ) from e
    return Chem


def _atom_ids(system, atoms) -> np.ndarray:
    if atoms is None:
        return np.arange(system.natoms)
    if isinstance(atoms, str):
        return system.select(atoms).ids
    return system._ids("atom", atoms)


def _pdb_name(name: str) -> str:
    return name if len(name) >= 4 else (" " + name).ljust(4)


def to_rdkit(system, atoms=None, sanitize: bool = True, implicit_hydrogens: bool = True,
             conformer: bool = True, stereo: bool = True, residue_info: bool = True):  # fmt: skip
    """RDKit molecule of ``atoms`` (indices, AtomSel, selection string; default all).

    Bonds use the system's orders; bonds to pseudo particles are zero-order,
    and bonds with a true ``aromatic`` property stay aromatic.  Like RDKit's
    MOL reader, open valences are filled with implicit hydrogens; pass
    ``implicit_hydrogens=False`` to keep them as radicals.  With a conformer,
    stereochemistry is assigned from 3D.  ct properties become molecule
    properties.
    """
    Chem = _chem()
    ids = _atom_ids(system, atoms)
    local = np.full(system.natoms, -1, np.int64)
    local[ids] = np.arange(len(ids))
    A, R, C, B = system._atoms, system._residues, system._chains, system._bonds
    anum = A.column("anum")[ids].tolist()
    charge = A.column("formal_charge")[ids].tolist()
    names = A.column("name")[ids].tolist()
    isotope = A.column("isotope")[ids].tolist() if "isotope" in A else [0] * len(ids)
    res = A.column("residue")[ids]
    chn = R.column("chain")[res]
    resnames = R.column("name")[res].tolist()
    resids = R.column("resid")[res].tolist()
    inserts = R.column("insertion")[res].tolist()
    chains = C.column("name")[chn].tolist()
    segids = C.column("segid")[chn].tolist()

    rw = Chem.RWMol()
    for k, idx in enumerate(ids.tolist()):
        atom = Chem.Atom(int(anum[k]))
        atom.SetFormalCharge(int(charge[k]))
        atom.SetNoImplicit(not implicit_hydrogens)
        if isotope[k]:
            atom.SetIsotope(int(isotope[k]))
        atom.SetIntProp("boonza_index", idx)
        atom.SetProp("_Name", names[k])
        if residue_info:
            info = Chem.AtomPDBResidueInfo()
            info.SetName(_pdb_name(names[k]))
            info.SetResidueName(resnames[k])
            info.SetResidueNumber(int(resids[k]))
            info.SetChainId(chains[k])
            info.SetInsertionCode(inserts[k] or " ")
            info.SetSegmentNumber(0)
            if segids[k]:
                atom.SetProp("segid", segids[k])
            atom.SetMonomerInfo(info)
        rw.AddAtom(atom)

    bi, bj = local[B.column("i")], local[B.column("j")]
    keep = np.flatnonzero((bi >= 0) & (bj >= 0))
    orders = B.column("order")[keep].tolist()
    aromatic = B.column("aromatic")[keep].tolist() if "aromatic" in B else [0] * len(keep)
    types = {0: Chem.BondType.ZERO, 1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
             3: Chem.BondType.TRIPLE, 4: Chem.BondType.QUADRUPLE}  # fmt: skip
    for k, (i, j) in enumerate(zip(bi[keep].tolist(), bj[keep].tolist(), strict=True)):
        if anum[i] == 0 or anum[j] == 0:
            kind = Chem.BondType.ZERO
        elif aromatic[k]:
            kind = Chem.BondType.AROMATIC
        else:
            kind = types.get(int(orders[k]), Chem.BondType.SINGLE)
        rw.AddBond(i, j, kind)
        if kind == Chem.BondType.AROMATIC:
            b = rw.GetBondBetweenAtoms(i, j)
            b.SetIsAromatic(True)
            rw.GetAtomWithIdx(i).SetIsAromatic(True)
            rw.GetAtomWithIdx(j).SetIsAromatic(True)
    mol = rw.GetMol()
    if system.ncts:
        ct = int(C.column("ct")[chn[0]]) if len(chn) else 0
        mol.SetProp("_Name", system._ct_names[ct])
        for key, value in system._ct_props[ct].items():
            if isinstance(value, bool | int | np.integer):
                mol.SetIntProp(key, int(value))
            elif isinstance(value, float | np.floating):
                mol.SetDoubleProp(key, float(value))
            else:
                mol.SetProp(key, str(value))
    if conformer and len(ids):
        conf = Chem.Conformer(len(ids))
        conf.Set3D(True)
        for k, xyz in enumerate(A.column("pos")[ids].tolist()):
            conf.SetAtomPosition(k, xyz)
        mol.AddConformer(conf, assignId=True)
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            raise ValueError(
                f"RDKit could not sanitize the molecule ({e}); bond orders or formal charges "
                "are probably missing: run boonza.assign_bond_orders() or pass sanitize=False"
            ) from e
    if stereo and conformer and len(ids):
        Chem.AssignStereochemistryFrom3D(mol)
    return mol


def fragments_to_rdkit(system, atoms=None, **kwargs) -> list:
    """One RDKit molecule per molecule (fragment) touching ``atoms``."""
    ids = _atom_ids(system, atoms)
    return [to_rdkit(system, system.fragment_atoms(f), **kwargs)
            for f in np.unique(system.fragids[ids]).tolist()]  # fmt: skip


def from_rdkit(
    mol, conf_id: int = -1, add_hydrogens: bool = False, name: str | None = None
) -> System:
    """System from an RDKit molecule; bonds are kekulized.

    Implicit hydrogens are an error unless ``add_hydrogens`` (then they are
    added, with coordinates when the molecule has a conformer).
    """
    Chem = _chem()
    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=mol.GetNumConformers() > 0)
    else:
        mol.UpdatePropertyCache(strict=False)
        if any(a.GetNumImplicitHs() for a in mol.GetAtoms()):
            raise ValueError("molecule has implicit hydrogens; pass add_hydrogens=True "
                             "or use Chem.AddHs()")  # fmt: skip
    mol = Chem.Mol(mol)
    aromatic_left = False
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:  # unsanitized aromatic input: keep the flags
        aromatic_left = True
    atoms = list(mol.GetAtoms())
    n = len(atoms)
    anum = np.array([a.GetAtomicNum() for a in atoms], np.int64)
    names = []
    for a in atoms:
        info = a.GetPDBResidueInfo()
        if a.HasProp("_Name"):
            names.append(a.GetProp("_Name"))
        elif info is not None:
            names.append(info.GetName().strip())
        else:
            names.append(msys_symbol(a.GetAtomicNum()))
    title = name if name is not None else (mol.GetProp("_Name") if mol.HasProp("_Name") else "")
    s = System(title)
    raw = mol.GetPropsAsDict(includePrivate=False, includeComputed=False)
    props = {k: v for k, v in raw.items()
             if isinstance(v, int | float | str) and not isinstance(v, bool)}  # fmt: skip
    ct = s.add_ct(title, **props)
    infos = [a.GetPDBResidueInfo() for a in atoms]
    if any(info is not None for info in infos):
        # hydrogens added by Chem.AddHs carry no residue info: take a bonded atom's
        for k, a in enumerate(atoms):
            if infos[k] is None:
                infos[k] = next((infos[b.GetIdx()] for b in a.GetNeighbors()
                                 if infos[b.GetIdx()] is not None), None)  # fmt: skip
    if n and all(info is not None for info in infos):
        from .io.dms import _factorize

        chain = np.array([i.GetChainId() for i in infos], dtype=STR)
        resname = np.array([i.GetResidueName().strip() for i in infos], dtype=STR)
        resid = np.array([i.GetResidueNumber() for i in infos], np.int64)
        ins = np.array([i.GetInsertionCode().strip() for i in infos], dtype=STR)
        key, chain_first = _factorize(chain)
        res_code, res_first = _factorize(key, resid, resname, ins)
        s._chains.append(len(chain_first), {"ct": ct.id, "name": chain[chain_first]})
        s._residues.append(len(res_first), {"chain": key[res_first], "resid": resid[res_first],
                                            "name": resname[res_first],
                                            "insertion": ins[res_first]})  # fmt: skip
        residue = res_code
    else:
        residue = s.add_residue(s.add_chain(ct)).id
    pos = np.zeros((n, 3))
    if mol.GetNumConformers():
        pos = np.asarray(mol.GetConformer(conf_id).GetPositions(), dtype=np.float64)
    if isinstance(residue, np.ndarray):
        s._atoms.append(n, {"residue": residue, "name": np.array(names, dtype=STR), "anum": anum,
                            "pos": pos})  # fmt: skip
    else:
        s.add_atoms(n, residue, name=names, anum=anum, pos=pos)
    s.atoms["formal_charge"] = np.array([a.GetFormalCharge() for a in atoms], np.int64)
    iso = [a.GetIsotope() for a in atoms]
    if any(iso):
        s.atoms["isotope"] = np.array(iso, np.int64)
    bonds = list(mol.GetBonds())
    if bonds:
        pairs = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in bonds], np.int64)
        orders, arom = [], []
        for b in bonds:
            kind = b.GetBondType()
            is_arom = kind == Chem.BondType.AROMATIC
            arom.append(int(is_arom))
            if is_arom or kind in (Chem.BondType.DATIVE, Chem.BondType.SINGLE):
                orders.append(1)
            elif kind == Chem.BondType.ZERO:
                orders.append(0)
            else:
                orders.append(int(round(b.GetBondTypeAsDouble())))
        ids = s.add_bonds(pairs)
        s._bonds.set("order", np.array(orders), ids)
        if aromatic_left and any(arom):
            s.bonds.add_prop("aromatic", int)
            s._bonds.set("aromatic", np.array(arom), ids)
    s._cache.clear()
    return s


def _fewest_charges(mol):
    """The resonance form of ``mol`` with the fewest charged atoms (ties keep ``mol``)."""
    from rdkit import Chem

    def ncharged(m):
        return sum(a.GetFormalCharge() != 0 for a in m.GetAtoms())

    best, score = mol, ncharged(mol)
    if score == 0:
        return mol
    try:
        work = Chem.Mol(mol)
        Chem.SanitizeMol(work)  # resonance enumeration needs ring and conjugation info
        forms = list(Chem.ResonanceMolSupplier(work, Chem.KEKULE_ALL))
    except Exception:
        return mol
    for form in forms:
        if form is not None and ncharged(form) < score:
            best, score = Chem.Mol(form), ncharged(form)
    if best is not mol:
        Chem.SanitizeMol(best)
    return best


def assign_bond_orders(system, atoms=None, charge: int | None = None) -> None:
    """Perceive bond orders and formal charges from connectivity and geometry, in place.

    Uses RDKit's DetermineBondOrders (xyz2mol) on each molecule touching
    ``atoms``; hydrogens must be present.  The total charge of each molecule
    is its current formal charge sum unless ``charge`` is given (only for a
    single molecule).  Best suited to ligands and other small molecules.
    """
    _chem()
    from rdkit.Chem import rdDetermineBonds

    ids = _atom_ids(system, atoms)
    frags = np.unique(system.fragids[ids])
    if charge is not None and len(frags) != 1:
        raise ValueError("an explicit charge needs a selection within one molecule")
    for f in frags.tolist():
        fatoms = system.fragment_atoms(f)
        mol = to_rdkit(system, fatoms, sanitize=False, implicit_hydrogens=False, stereo=False,
                       residue_info=False)  # fmt: skip
        total = int(system.atoms["formal_charge"][fatoms].sum()) if charge is None else charge
        rdDetermineBonds.DetermineBondOrders(mol, charge=total, embedChiral=False)
        mol = _fewest_charges(mol)
        from rdkit import Chem

        Chem.Kekulize(mol, clearAromaticFlags=True)
        fc = np.array([a.GetFormalCharge() for a in mol.GetAtoms()])
        system._atoms.set("formal_charge", fc, fatoms)
        for b in mol.GetBonds():
            i, j = fatoms[b.GetBeginAtomIdx()], fatoms[b.GetEndAtomIdx()]
            bond = system.find_bond(i, j)
            if bond is not None:
                bond.order = int(round(b.GetBondTypeAsDouble()))
