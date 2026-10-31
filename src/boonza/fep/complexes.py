"""A receptor for every ligand: boonza fep prep --complexes DIR.

DIR holds one co-complex per ligand -- an MD snapshot, a crystal structure --
named after it (``Cmpd1.mae``, ``Cmpd1.dms``, ...): the protein and the ligand,
with hydrogens; water and ions are left out (each leg is solvated afresh) and
anything else is refused.  Every protein must be the same molecule atom for
atom (the same residues, protonation states, caps and atom names), since a
hybrid's two end states share their environment.

Each complex is made whole across its box, superposed on the first by the
protein's alpha carbons, and its ligand matched to the SDF's record of that name
by element and bonds (with its hydrogens, so its protonation state too, and
its stereocentres checked); the ligand's pose then comes from its complex.  An
edge A -> B is built in A's receptor, and its windows from lambda 0.5 on start
from B's receptor, B's pose of the atoms the two share, and B's own atoms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: what a complex file may be, in the order looked for
SUFFIXES = (".mae", ".maegz", ".cms", ".dms", ".cif", ".pdb")


def find(directory, name: str) -> Path:
    """The complex of ligand ``name`` in ``directory``."""
    d = Path(directory)
    for suffix in SUFFIXES:
        p = d / f"{name}{suffix}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"{d}: no complex for {name} ({name}" + "|".join(SUFFIXES) + ")")


def _graph(system, ids):
    import networkx as nx

    local = {int(i): k for k, i in enumerate(ids)}
    g = nx.Graph()
    anum = np.asarray(system.atoms["anum"])
    for i in ids:
        g.add_node(local[int(i)], z=int(anum[i]))
    b = system.bonds
    for p, q in zip(np.asarray(b["i"]), np.asarray(b["j"]), strict=True):
        if int(p) in local and int(q) in local:
            g.add_edge(local[int(p)], local[int(q)])
    return g


def _mol_graph(mol):
    import networkx as nx

    g = nx.Graph()
    for a in mol.GetAtoms():
        g.add_node(a.GetIdx(), z=a.GetAtomicNum())
    for b in mol.GetBonds():
        g.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    return g


def _match(mol, system, ids, path) -> np.ndarray:
    """For each atom of ``mol``, the atom of ``system`` (among ``ids``) it is."""
    from networkx.algorithms.isomorphism import GraphMatcher

    gm = GraphMatcher(_mol_graph(mol), _graph(system, ids),
                      node_match=lambda a, b: a["z"] == b["z"])  # fmt: skip
    found = next(gm.isomorphisms_iter(), None)
    if found is None:
        raise ValueError(f"{path}: its ligand is not {mol.GetProp('_Name')} of the SDF as a "
                         "graph (another protonation state or tautomer?)")  # fmt: skip
    return np.asarray([ids[found[k]] for k in range(mol.GetNumAtoms())])


def _same_stereo(mol, xyz) -> bool:
    from rdkit import Chem

    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for k, p in enumerate(xyz):
        conf.SetAtomPosition(k, p.tolist())
    Chem.AssignStereochemistryFrom3D(m)
    ref = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(ref)
    return Chem.MolToSmiles(Chem.RemoveHs(m)) == Chem.MolToSmiles(Chem.RemoveHs(ref))


def load(directory, mols, ligs, names, log=print):
    """The receptor (the first complex's protein, its topology for every edge) and
    every ligand's receptor positions (Å, superposed on it), by name.  ``mols``
    (RDKit) and ``ligs`` (boonza) take their poses from the complexes, in place."""
    from ..align import kabsch
    from ..glue import make_whole
    from ..md.prepare import load_input

    receptor, positions, ca = None, {}, None
    for mol, lig, name in zip(mols, ligs, names, strict=True):
        path = find(directory, name)
        s = load_input(str(path))
        if s.cell is not None and np.any(np.asarray(s.cell)):
            s.positions = make_whole(s)
        keep = s.select("not (water or ions)").ids
        s = s.clone(keep)
        protein = np.asarray(s.select("protein").ids)
        rest = np.setdiff1d(np.arange(s.natoms), protein)
        ligand = _match(mol, s, rest, path)
        other = np.setdiff1d(rest, ligand)
        if len(other):
            residues = sorted({str(s.residues["name"][r]) for r in s.atoms["residue"][other]})
            raise ValueError(f"{path}: besides the protein and {name}, {', '.join(residues)}: "
                             "--complexes takes the protein and the ligand alone")  # fmt: skip
        prot = s.clone(protein)
        xyz = np.asarray(prot.positions, float)
        if receptor is None:
            receptor = prot
            ca = np.flatnonzero(np.asarray(prot.atoms["name"]) == "CA")
            target = xyz[ca]
        elif (prot.natoms != receptor.natoms
              or list(prot.atoms["name"]) != list(receptor.atoms["name"])
              or list(prot.residues["name"]) != list(receptor.residues["name"])):  # fmt: skip
            raise ValueError(f"{path}: its protein is not {names[0]}'s atom for atom (residues, "
                             "protonation states, caps and atom names must all agree)")  # fmt: skip
        R, t = kabsch(xyz[ca], target)
        positions[name] = xyz @ R.T + t
        pose = np.asarray(s.positions, float)[ligand] @ R.T + t
        if not _same_stereo(mol, pose):
            raise ValueError(f"{path}: its {name} has other stereocentres than the SDF's")
        conf = mol.GetConformer()
        for k, p in enumerate(pose):
            conf.SetAtomPosition(k, p.tolist())
        lig.positions = pose
        rmsd = float(np.sqrt(((xyz[ca] @ R.T + t - target) ** 2).sum(1).mean()))
        log(f"  {name}: {path.name}, {prot.natoms} protein atoms, C-alpha RMSD to "
            f"{names[0]}'s {rmsd:.2f} Å")  # fmt: skip
    receptor.positions = positions[names[0]]
    return receptor, positions
