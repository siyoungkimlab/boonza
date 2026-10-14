"""Bond orders from names and geometry, for structures that come without them.

PDB and mmCIF files (and anything bonds were guessed for) have every bond
single.  :func:`guess_bond_orders` puts back the double and triple bonds:

1. Amino acids by their atom names: the backbone carbonyl, the side-chain
   carbonyls of Asp, Glu, Asn and Gln, Arg's guanidinium, and the rings of Phe,
   Tyr, Trp and His (His by its hydrogens: HD1 alone is HID, else HIE).
2. Everything else from geometry, hydrogens or not:

   * a terminal O or S on carbon is a carbonyl or thione when the bond is short
     (C-O under 1.30 A, C-S under 1.72 A); of a carboxylate's two oxygens, the
     shorter; a phosphate keeps one P=O, a sulfonyl both S=O, a nitro one N=O;
   * a nitrile (C-N under 1.20 A) and an alkyne (C-C under 1.25 A, both ends
     linear) are triple;
   * the remaining unsaturated atoms -- planar (sp2) or linear (sp) by their
     bond angles, or, with hydrogens, short of their valence -- pair up along
     their short bonds (C-C under 1.45 A, C-N under 1.40, N-N under 1.35) by a
     maximum matching, so an aromatic ring comes out as one Kekule structure.

Formal charges are left as they are.  :func:`boonza.assign_bond_orders`
(RDKit's xyz2mol) is the stricter route for a small molecule with all its
hydrogens.
"""

from __future__ import annotations

import numpy as np

#: double bonds of amino acids, by atom names
DOUBLE = {
    "ARG": [("CZ", "NH2")], "ASN": [("CG", "OD1")], "ASP": [("CG", "OD1")],
    "ASH": [("CG", "OD1")], "GLN": [("CD", "OE1")], "GLU": [("CD", "OE1")],
    "GLH": [("CD", "OE1")],
    "PHE": [("CG", "CD1"), ("CE1", "CZ"), ("CE2", "CD2")],
    "TYR": [("CG", "CD1"), ("CE1", "CZ"), ("CE2", "CD2")],
    "TRP": [("CG", "CD1"), ("CD2", "CE2"), ("CE3", "CZ3"), ("CZ2", "CH2")],
    "HIE": [("CG", "CD2"), ("CE1", "ND1")], "HSE": [("CG", "CD2"), ("CE1", "ND1")],
    "HID": [("CG", "CD2"), ("CE1", "NE2")], "HSD": [("CG", "CD2"), ("CE1", "NE2")],
    "HIP": [("CG", "CD2"), ("CE1", "ND1")], "HSP": [("CG", "CD2"), ("CE1", "ND1")],
}  # fmt: skip
AMINO_ACIDS = {"ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYX", "CYM", "GLN", "GLU", "GLH", "GLY",
               "HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "ILE", "LEU", "LYS", "LYN", "MET",
               "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}  # fmt: skip
#: bonds shorter than this (A) may be double (or aromatic), by element pair
SHORT = {(6, 6): 1.45, (6, 7): 1.40, (7, 7): 1.35, (6, 8): 1.32, (7, 8): 1.30, (6, 16): 1.72}
#: what an element's atoms bond to at most (bond orders summed)
VALENCE = {6: 4, 7: 3, 8: 2, 16: 2, 5: 3}


def guess_bond_orders(system, atoms=None, force: bool = False) -> int:
    """Double and triple bonds from names (amino acids) and geometry (the rest), in
    place.  Only when no bond of ``atoms`` (default all) is above single, unless
    ``force``.  Returns the number of bonds made double or triple."""
    ids = np.arange(system.natoms) if atoms is None else system._ids("atom", atoms)
    mine = np.zeros(system.natoms, bool)
    mine[ids] = True
    bi, bj = np.asarray(system.bonds["i"]), np.asarray(system.bonds["j"])
    order = np.asarray(system.bonds["order"]).copy()
    inside = mine[bi] & mine[bj]
    if not force and (order[inside] > 1).any():
        return 0
    order[inside] = np.where(order[inside] > 1, 1, order[inside])
    anum = np.asarray(system.atoms["anum"])
    pos = np.asarray(system.positions, float)
    nbrs: list[list[tuple[int, int]]] = [[] for _ in range(system.natoms)]
    for k, (i, j) in enumerate(zip(bi.tolist(), bj.tolist(), strict=True)):
        if inside[k]:
            nbrs[i].append((j, k))
            nbrs[j].append((i, k))
    used = np.zeros(system.natoms, np.int64)  # bond order beyond single, per atom
    done = np.zeros(system.natoms, bool)  # atoms an amino-acid template decided

    def set_order(k, o):
        i, j = int(bi[k]), int(bj[k])
        used[i] += o - order[k]
        used[j] += o - order[k]
        order[k] = o

    def bond(i, j):
        for x, k in nbrs[i]:
            if x == j:
                return k
        return None

    # 1. amino acids, by name
    res = np.asarray(system.atoms["residue"])
    names = [str(x) for x in system.atoms["name"]]
    rname = [str(x) for x in system.residues["name"]]
    for r in np.unique(res[ids]).tolist():
        rn = rname[r]
        if rn not in AMINO_ACIDS:
            continue
        here = {names[a]: int(a) for a in system.residue_atoms(r) if mine[a]}
        pairs = [("C", "O")] + DOUBLE.get(rn, [])
        if rn == "HIS":  # by its hydrogens: HD1 alone is HID
            pairs += DOUBLE["HID" if "HD1" in here and "HE2" not in here else "HIE"]
        for x, y in pairs:
            if x in here and y in here:
                k = bond(here[x], here[y])
                if k is not None:
                    set_order(k, 2)
        for a in here.values():
            done[a] = True

    hcount = np.array([sum(1 for x, _ in nbrs[a] if anum[x] == 1) for a in range(system.natoms)])
    heavy = [[(x, k) for x, k in nbrs[a] if anum[x] > 1] for a in range(system.natoms)]
    has_h = (anum[ids] == 1).any()

    def length(k):
        return float(np.linalg.norm(pos[bi[k]] - pos[bj[k]]))

    # 2a. terminal O, S and N, by bond length
    for y in ids.tolist():
        if done[y] or anum[y] <= 1:
            continue
        terminal = [(x, k) for x, k in heavy[y] if anum[x] in (7, 8, 16) and len(heavy[x]) == 1
                    and hcount[x] == 0 and not done[x]]  # fmt: skip
        if not terminal:
            continue
        zy = int(anum[y])
        oxy = sorted(((length(k), x, k) for x, k in terminal if anum[x] in (8, 16)))
        if zy == 6 and oxy:
            d, x, k = oxy[0]
            if d < (1.30 if anum[x] == 8 else 1.72) or len(oxy) > 1 and d < 1.36:
                set_order(k, 2)
        elif zy == 15 and oxy and oxy[0][0] < 1.55:
            set_order(oxy[0][2], 2)
        elif zy == 16:
            for d, _, k in oxy[:2]:
                if d < 1.50:
                    set_order(k, 2)
        elif zy == 7 and oxy and oxy[0][0] < 1.30:
            set_order(oxy[0][2], 2)
        for x, k in terminal:  # nitriles
            if anum[x] == 7 and zy == 6 and length(k) < 1.20 and order[k] == 1:
                set_order(k, 3)

    # 2b. alkynes, then the unsaturated atoms, paired by a maximum matching
    def angle(a, b, c):
        u, v = pos[a] - pos[b], pos[c] - pos[b]
        return np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))

    def room(a) -> int:
        """How much more bond order atom a can take (0: saturated)."""
        z = int(anum[a])
        if z not in VALENCE or done[a]:
            return 0
        total = len(nbrs[a]) + int(used[a])
        if has_h:
            charge = int(system.atoms["formal_charge"][a]) if z == 6 else 0
            return max(VALENCE[z] - total - charge, 0)
        nh = len(heavy[a])
        if z == 6:
            if nh == 3:  # planar: sp2
                n = pos[[x for x, _ in heavy[a]]] - pos[a]
                normal = np.cross(n[1] - n[0], n[2] - n[0])
                flat = abs((pos[a] - pos[heavy[a][0][0]]) @ normal) / (np.linalg.norm(normal) or 1)
                return 1 if flat < 0.15 and used[a] == 0 else 0
            if nh == 2:
                return 1 if angle(heavy[a][0][0], a, heavy[a][1][0]) > 115 and used[a] == 0 else 0
            return 1 if nh == 1 and used[a] == 0 else 0
        if z == 7:
            return 1 if nh <= 2 and used[a] == 0 else 0
        return 0

    for k in np.flatnonzero(inside).tolist():
        i, j = int(bi[k]), int(bj[k])
        if anum[i] == 6 and anum[j] == 6 and order[k] == 1 and length(k) < 1.25 and not done[i]:

            def straight(a, b):
                other = [x for x, _ in heavy[a] if x != b]
                return not other or angle(other[0], a, b) > 165

            linear = straight(i, j) and straight(j, i)
            if linear and (not has_h or (room(i) >= 2 and room(j) >= 2)):
                set_order(k, 3)

    import networkx as nx

    g = nx.Graph()
    for k in np.flatnonzero(inside).tolist():
        i, j = int(bi[k]), int(bj[k])
        if order[k] != 1:
            continue
        pair = tuple(sorted((int(anum[i]), int(anum[j]))))
        if pair not in SHORT or length(k) >= SHORT[pair] + (0.03 if has_h else 0):
            continue
        if room(i) and room(j):
            g.add_edge(i, j, weight=SHORT[pair] - length(k) + 1.0, k=k)
    for i, j in nx.max_weight_matching(g, maxcardinality=True):
        set_order(g.edges[i, j]["k"], 2)

    changed = int((order != np.asarray(system.bonds["order"])).sum())
    if changed:
        system.bonds["order"] = order
    return changed
