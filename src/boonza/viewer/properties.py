"""What a molecular surface is coloured by: each atom's charge and hydrophobicity.

Charges, from the first of these a structure has:

1. its own partial charges (a parameterized system: DMS, a CG frame);
2. its formal charges (Maestro sets them), each shared with the atoms of the
   same element sitting on the same atom and nothing else (a carboxylate's two
   oxygens, an arginine's NH1 and NH2), as resonance shares them;
3. charged residues at pH 7 by name (Asp, Glu -1; Lys, Arg +1; HIP +1; chain
   ends +1 and -1), ions by element.

Hydrophobicity is a sign per atom, no scale fitted: +1 for carbon and sulfur
with no N or O on them and for Cl, Br and I; -1 for N and O and for anything
charged; 0 for carbon on N or O.  Hydrogens take their atom's.  Beads (coarse
grained) by their residue: side chains of Ala, Val, Leu, Ile, Met, Phe, Trp,
Pro and Cys +1, Tyr and Gly 0, other side chains -1; the backbone's amide -0.5.
"""

from __future__ import annotations

import numpy as np

#: residue charges at pH 7, on these atoms, shared equally
TEMPLATE = {"ASP": (("OD1", "OD2"), -1), "GLU": (("OE1", "OE2"), -1), "LYS": (("NZ",), 1),
            "ARG": (("NH1", "NH2", "NE"), 1), "HIP": (("ND1", "NE2"), 1),
            "HSP": (("ND1", "NE2"), 1)}  # fmt: skip
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
    res = np.asarray(system.atoms["residue"])
    names = [str(x) for x in system.atoms["name"]]
    rnames = [residue_name(str(x)) for x in system.residues["name"]]
    for a in range(system.natoms):
        if anum[a] in IONS and not nbrs[a]:
            q[a] = IONS[int(anum[a])]
    for r in range(system.nresidues):
        if rnames[r] not in TEMPLATE:
            continue
        atoms, total = TEMPLATE[rnames[r]]
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
