"""Residue and atom classification: protein, nucleic, water, lipid; backbone, sidechain.

Follows msys ``Analyze()`` so that selection macros such as ``protein`` and
``backbone`` select the same atoms as msys.
"""

from __future__ import annotations

import numpy as np

RES_OTHER, RES_PROTEIN, RES_NUCLEIC, RES_WATER, RES_LIPID = range(5)
ATOM_OTHER, ATOM_PROBACK, ATOM_PROSIDE, ATOM_NUCBACK = range(4)

WATER_NAMES = [
    "H2O",
    "HH0",
    "OHH",
    "HOH",
    "OH2",
    "SOL",
    "WAT",
    "TIP",
    "TIP2",
    "TIP3",
    "TIP4",
    "SPC",
]
LIPID_NAMES = ["DLPE", "DMPC", "DPPC", "GPC", "LPPC", "PALM", "PC", "PGCL", "POPC", "POPE", "POPS"]

_BACKBONE = {"CA": ATOM_PROBACK, "C": ATOM_PROBACK, "O": ATOM_PROBACK, "N": ATOM_PROBACK}
_BACKBONE.update(
    dict.fromkeys(
        ["P", "O1P", "O2P", "OP1", "OP2", "C3*", "C3'", "O3*", "O3'", "C4*", "C4'",
         "C5*", "C5'", "O5*", "O5'"],
        ATOM_NUCBACK,
    )
)  # fmt: skip
_TERMINAL = {
    "OT1": ATOM_PROBACK, "OT2": ATOM_PROBACK, "OX1": ATOM_PROBACK, "O1": ATOM_PROBACK,
    "O2": ATOM_PROBACK, "H5T": ATOM_NUCBACK, "H3T": ATOM_NUCBACK,
}  # fmt: skip


def classify(system) -> tuple[np.ndarray, np.ndarray]:
    """Return (residue type per residue, atom type per atom) as int8 arrays."""
    natoms, nres = system.natoms, system.nresidues
    restype = np.zeros(nres, np.int8)
    atomtype = np.zeros(natoms, np.int8)
    if natoms == 0:
        return restype, atomtype

    res = system._atoms.column("residue")
    anum = system._atoms.column("anum")
    real = anum >= 1
    bi, bj = system._bonds.column("i"), system._bonds.column("j")
    # bonds to atoms with atomic number 0 (pseudo particles) do not count
    degree = np.bincount(bi[anum[bj] != 0], minlength=natoms) + np.bincount(
        bj[anum[bi] != 0], minlength=natoms
    )

    # water: by residue name, or three real atoms O(-H)2 with no other bonds
    water = np.isin(system._residues.column("name"), WATER_NAMES)
    count = lambda mask: np.bincount(res[mask], minlength=nres)  # noqa: E731
    candidate = (count(real) == 3) & (count(anum == 8) == 1) & (count(anum == 1) == 2)
    same = res[bi] == res[bj]
    oh = same & (np.minimum(anum[bi], anum[bj]) == 1) & (np.maximum(anum[bi], anum[bj]) == 8)
    n_oh = np.bincount(res[bi][oh], minlength=nres)
    o_deg = np.bincount(res[anum == 8], weights=degree[anum == 8], minlength=nres)
    h_deg = np.bincount(res[anum == 1], weights=degree[anum == 1], minlength=nres)
    water |= candidate & (n_oh == 2) & (o_deg == 2) & (h_deg == 2)
    restype[water] = RES_WATER

    off, order = system._csr("rescsr", nres, res)
    sizes = np.diff(off)
    lipid = ~water & (sizes >= 4) & np.isin(system._residues.column("name"), LIPID_NAMES)
    restype[lipid] = RES_LIPID

    adj_off, adj_nbr, _ = system._adjacency()
    names = system._atoms.column("name").tolist()
    anum_l = anum.tolist()
    res_l = res.tolist()
    deg_l = degree.tolist()

    def nbrs(a):
        return adj_nbr[adj_off[a] : adj_off[a + 1]].tolist()

    for r in np.flatnonzero(~water & ~lipid & (sizes >= 4)).tolist():
        atoms = order[off[r] : off[r + 1]].tolist()
        if _is_cap(atoms, anum_l, deg_l, nbrs):
            restype[r] = RES_PROTEIN
            continue
        npro = nnuc = 0
        ca = c = n = -1
        seen = set()
        for a in atoms:
            nm = names[a]
            if nm in seen:
                continue
            seen.add(nm)
            t = _BACKBONE.get(nm)
            if t is None:
                t = ATOM_OTHER
                want = _TERMINAL.get(nm)
                if want is not None and any(atomtype[b] == want for b in nbrs(a)):
                    t = want
            elif nm == "CA":
                ca = a
            elif nm == "C":
                c = a
            elif nm == "N":
                n = a
            atomtype[a] = t
            npro += t == ATOM_PROBACK
            nnuc += t == ATOM_NUCBACK
        if (
            npro >= 4
            and min(ca, c, n) >= 0
            and n in nbrs(ca)
            and c in nbrs(ca)
            and n not in nbrs(c)
        ):
            restype[r] = RES_PROTEIN
            _sidechain(r, ca, atomtype, anum_l, names, res_l, nbrs)
        elif nnuc >= 4:
            restype[r] = RES_NUCLEIC
        else:
            atomtype[atoms] = ATOM_OTHER
    return restype, atomtype


def _sidechain(r, ca, atomtype, anum, names, res, nbrs) -> None:
    # pick a C-beta (the last heavy non-backbone neighbor), else a hydrogen
    cb = -1
    for b in nbrs(ca):
        if atomtype[b] == ATOM_PROBACK:
            continue
        if cb < 0:
            cb = b
        elif not (anum[b] == 1 or names[b].startswith("H")):
            cb = b
    if cb < 0:
        return
    queue = [cb]
    while queue:
        a = queue.pop(0)
        atomtype[a] = ATOM_PROSIDE
        queue += [b for b in nbrs(a) if atomtype[b] == ATOM_OTHER and res[b] == r]


def _is_cap(atoms, anum, degree, nbrs) -> bool:
    """ACE (CH3-C=O) or NME (NH-CH3) capping group, matched like msys' graph match."""
    real = [a for a in atoms if anum[a] >= 1]
    if len(real) != 6:
        return False
    elems = sorted(anum[a] for a in real)
    inside = set(real)
    if elems == [1, 1, 1, 6, 6, 8]:  # ACE: C(4)-3H, C(4)-C(3), C(3)=O(1)
        methyl = [a for a in real if anum[a] == 6 and degree[a] == 4]
        carbonyl = [a for a in real if anum[a] == 6 and degree[a] == 3]
        oxygen = [a for a in real if anum[a] == 8 and degree[a] == 1]
        if len(methyl) != 1 or len(carbonyl) != 1 or len(oxygen) != 1:
            return False
        m, c, o = methyl[0], carbonyl[0], oxygen[0]
        hs = [b for b in nbrs(m) if b in inside and anum[b] == 1 and degree[b] == 1]
        return c in nbrs(m) and o in nbrs(c) and len(hs) == 3
    if elems == [1, 1, 1, 1, 6, 7]:  # NME: C(4)-3H, C(4)-N(3), N-H
        carbon = [a for a in real if anum[a] == 6 and degree[a] == 4]
        nitrogen = [a for a in real if anum[a] == 7 and degree[a] == 3]
        if len(carbon) != 1 or len(nitrogen) != 1:
            return False
        c, n = carbon[0], nitrogen[0]
        hc = [b for b in nbrs(c) if b in inside and anum[b] == 1 and degree[b] == 1]
        hn = [b for b in nbrs(n) if b in inside and anum[b] == 1 and degree[b] == 1]
        return n in nbrs(c) and len(hc) == 3 and len(hn) == 1
    return False
