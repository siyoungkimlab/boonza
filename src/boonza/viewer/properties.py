"""What a molecular surface is coloured by: each atom's charge and hydrophobicity.

Charges, from the first of these a structure has:

1. its own partial charges (a parameterized system: DMS, a CG frame);
2. its formal charges (Maestro sets them), each shared with the atoms of the
   same element sitting on the same atom and nothing else (a carboxylate's two
   oxygens, an arginine's NH1 and NH2), as resonance shares them;
3. charged residues at pH 7 by name (Asp, Glu -1; Lys, Arg +1; HIP +1; chain
   ends +1 and -1), ions by element; beads (a Martini structure without its
   topology) on the charged side-chain bead (Asp, Glu SC1; Lys, Arg SC2).

Lipophilicity (all-atom structures) is each heavy atom's share of log P, by
Wildman and Crippen's atom types (J. Chem. Inf. Comput. Sci. 1999, 39, 868;
RDKit's), its hydrogens' shares added to it.  The page sums them over the
atoms near each point of a surface into a molecular lipophilicity potential,
as ChimeraX's mlp does with its own table for amino acids; typed from the
bonds, ligands, nucleic acids and modified residues have theirs too.

Hydrophobicity (all a bead has, with no atom types) is a sign per atom, no
scale fitted: +1 for carbon and sulfur with no N or O on them and for Cl, Br
and I; -1 for N and O and for anything charged; 0 for carbon on N or O.
Hydrogens take their atom's.  Beads (coarse
grained) by their residue: side chains of Ala, Val, Leu, Ile, Met, Phe, Trp,
Pro and Cys +1, Tyr and Gly 0, other side chains -1; the backbone's amide -0.5.
"""

from __future__ import annotations

import numpy as np

#: residue charges at pH 7, on these atoms, shared equally
TEMPLATE = {"ASP": (("OD1", "OD2"), -1), "GLU": (("OE1", "OE2"), -1), "LYS": (("NZ",), 1),
            "ARG": (("NH1", "NH2", "NE"), 1), "HIP": (("ND1", "NE2"), 1),
            "HSP": (("ND1", "NE2"), 1)}  # fmt: skip
#: the same on Martini's beads (2 and 3 alike)
CG_TEMPLATE = {"ASP": (("SC1",), -1), "GLU": (("SC1",), -1), "LYS": (("SC2",), 1),
               "ARG": (("SC2",), 1)}  # fmt: skip
#: ions by atomic number
IONS = {3: 1, 11: 1, 19: 1, 37: 1, 55: 1, 12: 2, 20: 2, 30: 2, 25: 2, 26: 2, 27: 2, 29: 2,
        9: -1, 17: -1, 35: -1, 53: -1}  # fmt: skip
#: one-letter codes to residue names (SIRAH writes sA, sHe, ...)
THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
         "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
         "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
         "X": "CYS"}  # fmt: skip
BEAD_PHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "CYS", "CYX"}
BEAD_NEUTRAL = {"TYR", "GLY"}
CG_BACKBONE = {"BB", "GN", "GC", "GO", "PX"}


def _neighbours(system) -> list[list[int]]:
    out: list[list[int]] = [[] for _ in range(system.natoms)]
    for i, j in zip(np.asarray(system.bonds["i"]).tolist(), np.asarray(system.bonds["j"]).tolist(),
                    strict=True):  # fmt: skip
        out[i].append(j)
        out[j].append(i)
    return out


def residue_name(name: str) -> str:
    """A residue's standard name; SIRAH's sX names by their one-letter code."""
    if len(name) in (2, 3) and name[0] == "s" and name[1].isupper():
        return THREE.get(name[1], name)
    return name


def charges(system, cg: bool = False) -> np.ndarray:
    """Each atom's charge (e), as the module says."""
    q = np.asarray(system.atoms["charge"], float).copy()
    if np.any(q != 0):
        return q
    return group_charges(system, cg)


def group_charges(system, cg: bool = False) -> np.ndarray:
    """The charged groups alone (2 and 3 of the module's list): where the charges
    sit, whatever partial charges a force field has spread over the rest."""
    anum = np.asarray(system.atoms["anum"])
    formal = np.asarray(system.atoms["formal_charge"], float)
    nbrs = _neighbours(system)
    q = np.zeros(system.natoms)
    if np.any(formal != 0):
        for a in np.flatnonzero(formal).tolist():
            heavy = [x for x in nbrs[a] if anum[x] > 1]
            mates = [a]
            if len(heavy) == 1:  # terminal: share with its twins on the same atom
                mates += [x for x in nbrs[heavy[0]] if x != a and anum[x] == anum[a]
                          and sum(anum[y] > 1 for y in nbrs[x]) == 1]  # fmt: skip
            q[mates] += formal[a] / len(mates)
        return q
    return template_charges(system, cg)


def template_charges(system, cg: bool = False) -> np.ndarray:
    """Charged residues at pH 7 by name, chain ends and ions (3 of the module's list)."""
    anum = np.asarray(system.atoms["anum"])
    nbrs = _neighbours(system)
    q = np.zeros(system.natoms)
    res = np.asarray(system.atoms["residue"])
    names = [str(x) for x in system.atoms["name"]]
    rnames = [residue_name(str(x)) for x in system.residues["name"]]
    for a in range(system.natoms):
        if anum[a] in IONS and not nbrs[a]:
            q[a] = IONS[int(anum[a])]
    template = CG_TEMPLATE if cg else TEMPLATE
    for r in range(system.nresidues):
        if rnames[r] not in template:
            continue
        atoms, total = template[rnames[r]]
        here = [a for a in system.residue_atoms(r) if names[a] in atoms]
        if here:
            q[here] += total / len(here)
    # chain ends: N +1, the terminal carboxylate's oxygens -1
    for a in range(system.natoms):
        if names[a] == "OXT":
            mates = [x for x in system.residue_atoms(res[a]) if names[x] in ("O", "OXT")]
            q[mates] -= 1 / len(mates)
    if not cg:
        protein = system.select("protein and name N").ids
        firsts = {}
        for a in protein.tolist():
            chain = int(np.asarray(system.residues["chain"])[res[a]])
            if chain not in firsts or res[a] < res[firsts[chain]]:
                firsts[chain] = a
        for a in firsts.values():
            if not any(names[x] == "C" for x in nbrs[a]):  # not capped (ACE)
                q[a] += 1
    return q


def hydrophobicity(system, q=None, cg: bool = False) -> np.ndarray:
    """Each atom's sign: +1 hydrophobic, -1 hydrophilic, 0 between."""
    n = system.natoms
    q = charges(system, cg) if q is None else q
    out = np.zeros(n)
    names = [str(x) for x in system.atoms["name"]]
    if cg:
        rnames = [residue_name(str(x)) for x in system.residues["name"]]
        res = np.asarray(system.atoms["residue"])
        for a in range(n):
            r = rnames[res[a]]
            if names[a] in CG_BACKBONE:
                out[a] = -0.5
            elif r in BEAD_PHOBIC:
                out[a] = 1
            elif r in BEAD_NEUTRAL:
                out[a] = 0
            else:
                out[a] = -1
        out[np.abs(q) >= 0.5] = -1
        return out
    anum = np.asarray(system.atoms["anum"])
    nbrs = _neighbours(system)
    for a in range(n):
        z = int(anum[a])
        if z in (6, 16):
            out[a] = 0 if any(anum[x] in (7, 8) for x in nbrs[a]) else 1
        elif z in (7, 8):
            out[a] = -1
        elif z in (17, 35, 53):
            out[a] = 1
    for a in range(n):  # hydrogens: their atom's
        if anum[a] == 1 and nbrs[a]:
            out[a] = out[nbrs[a][0]]
    heavy = anum > 1
    out[heavy & (np.abs(q) >= 0.3)] = -1
    return out


#: an atom RDKit could not type (a residue it cannot make sense of), by element
LIPO_ELEMENT = {6: 0.3, 7: -0.6, 8: -0.3, 15: -0.1, 16: 0.5, 9: 0.4, 17: 0.7, 35: 0.8, 53: 0.9}


def lipophilicity(system) -> np.ndarray:
    """Each heavy atom's share of log P (Wildman-Crippen), its hydrogens' with it.

    A structure rarely says which groups are charged, and typed as written an
    aspartate is the acid, with an oxygen as lipophilic as a carbon.  An atom
    of a charged group (group_charges) is therefore no more lipophilic than
    -0.3 - 0.5 |q|: -0.55 for each oxygen of a carboxylate, -0.8 for a lysine's N.

    A residue is typed with the atoms up to two bonds outside it, so that a
    peptide's N and C are the amide's they are, and once for all residues
    alike (same names, same bonds): a protein is twenty-odd small molecules
    to type, and one residue RDKit refuses costs only itself.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors

    n = system.natoms
    out = np.zeros(n)
    anum = np.asarray(system.atoms["anum"]).tolist()
    formal = np.asarray(system.atoms["formal_charge"]).tolist()
    res = np.asarray(system.atoms["residue"]).tolist()
    names = [str(x) for x in system.atoms["name"]]
    rnames = [str(x) for x in system.residues["name"]]
    bi = np.asarray(system.bonds["i"]).tolist()
    bj = np.asarray(system.bonds["j"]).tolist()
    order = np.asarray(system.bonds["order"]).tolist()
    nbrs: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i, j, o in zip(bi, bj, order, strict=True):
        nbrs[i].append((j, o))
        nbrs[j].append((i, o))
    kinds = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE,
             1.5: Chem.BondType.AROMATIC}  # fmt: skip
    members: list[list[int]] = [[] for _ in rnames]
    for a in range(n):
        members[res[a]].append(a)
    cache: dict = {}
    RDLogger.DisableLog("rdApp.*")
    try:
        for r, own in enumerate(members):
            if not own:
                continue
            if len(own) == 1 and not nbrs[own[0]]:  # an ion: hydrophilic
                out[own[0]] = -0.5 if anum[own[0]] in IONS else LIPO_ELEMENT.get(anum[own[0]], 0)
                continue
            inside = set(own)
            first = [x for a in own for x, _ in nbrs[a] if x not in inside]
            second = [y for x in first for y, _ in nbrs[x] if y not in inside]
            atoms = own + sorted(set(first) | set(second))
            local = {a: k for k, a in enumerate(atoms)}
            bonds = sorted({(min(local[a], local[x]), max(local[a], local[x]), o)
                            for a in atoms for x, o in nbrs[a] if x in local})  # fmt: skip
            key = (rnames[r], tuple(names[a] for a in own),
                   tuple((anum[a], formal[a]) for a in atoms), tuple(bonds))  # fmt: skip
            if key not in cache:
                cache[key] = _crippen([anum[a] for a in atoms], [formal[a] for a in atoms],
                                      bonds, len(own), kinds, Chem, rdMolDescriptors)  # fmt: skip
            got = cache[key]
            if got is None:  # by element, hydrogens nothing
                got = [LIPO_ELEMENT.get(anum[a], 0.0) if anum[a] > 1 else 0.0 for a in own]
                for k, a in enumerate(own):  # carbon on N or O: neither
                    if anum[a] == 6 and any(anum[x] in (7, 8) for x, _ in nbrs[a]):
                        got[k] = 0.0
            out[own] = got
    finally:
        RDLogger.EnableLog("rdApp.*")
    # by name too: a file with a formal charge on its metal alone says nothing of its aspartates
    q = np.maximum(np.abs(group_charges(system)), np.abs(template_charges(system)))
    charged = (q >= 0.3) & (np.asarray(anum) > 1)
    out[charged] = np.minimum(out[charged], -0.3 - 0.5 * np.minimum(q[charged], 1))
    return out


def _crippen(anum, formal, bonds, nown, kinds, Chem, rdMolDescriptors):
    """The first ``nown`` atoms' shares of log P, hydrogens' on their atoms; None
    if RDKit cannot make a molecule of them (bond orders missing, a metal's bonds)."""
    rw = Chem.RWMol()
    for z, q in zip(anum, formal, strict=True):
        if z < 1:
            return None
        atom = Chem.Atom(int(z))
        atom.SetFormalCharge(int(q))
        rw.AddAtom(atom)
    for i, j, o in bonds:
        if o not in kinds:
            return None
        rw.AddBond(i, j, kinds[o])
    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)
        shares = rdMolDescriptors._CalcCrippenContribs(mol)
    except Exception:  # noqa: BLE001  (RDKit raises its own kinds)
        return None
    out = [0.0] * nown
    for atom in mol.GetAtoms():
        k = atom.GetIdx()
        if atom.GetAtomicNum() == 1 and atom.GetDegree():
            k = atom.GetNeighbors()[0].GetIdx()
        if k < nown:
            out[k] += shares[atom.GetIdx()][0]
    return out
