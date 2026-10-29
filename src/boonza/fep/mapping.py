"""Atom mappings between two posed ligands, in 3D, after kartograf.

The ligands sit where they bind, so atoms that play the same part sit on top
of each other.  Each atom of A is paired with an atom of B by the assignment
that minimizes the summed distance, among pairs within ``max_distance`` (0.95
A, kartograf's default) and hydrogen with hydrogen only.  Then rules a hybrid
topology needs are applied until none is broken:

* a ring maps only as a whole, onto a ring of the same size (rings of a fused
  system may map one by one);
* no bond breaks or forms: two mapped atoms bonded in one ligand are bonded in
  the other, and a single bond never becomes a triple one;
* no chiral centre turns into its mirror image;
* a hydrogen maps only with the atom it is bonded to;
* only the largest connected piece is kept.

``score`` rates a mapping as LOMAP does, exp(-0.1 x heavy atoms that change),
for choosing which ligands to join in a network.
"""

from __future__ import annotations

import numpy as np

MAX_DISTANCE = 0.95  # A


def _graph(mol):
    n = mol.GetNumAtoms()
    nbrs = [[] for _ in range(n)]
    order = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        nbrs[i].append(j)
        nbrs[j].append(i)
        order[i, j] = order[j, i] = b.GetBondTypeAsDouble()
    anum = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    return nbrs, order, anum, rings


def map_atoms(a, b, max_distance: float = MAX_DISTANCE, element_change: bool = True) -> dict:
    """{atom of A: atom of B} for two RDKit molecules with hydrogens and 3D
    coordinates in one frame (posed ligands)."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    xa = a.GetConformer().GetPositions()
    xb = b.GetConformer().GetPositions()
    ga, gb = _graph(a), _graph(b)
    za, zb = ga[2], gb[2]
    d = cdist(xa, xb)
    ok = d <= max_distance
    ok &= (za[:, None] == 1) == (zb[None, :] == 1)  # hydrogen with hydrogen only
    if not element_change:
        ok &= za[:, None] == zb[None, :]
    cost = np.where(ok, d, 1e6)
    rows, cols = linear_sum_assignment(cost)
    mapping = {int(i): int(j) for i, j in zip(rows, cols, strict=True) if ok[i, j]}
    while True:
        before = dict(mapping)
        mapping = _rings(mapping, ga, gb)
        mapping = _bonds(mapping, ga, gb, d)
        mapping = _chirality(mapping, ga, gb, xa, xb)
        mapping = _hydrogens(mapping, ga, gb)
        mapping = _connected(mapping, ga)
        if mapping == before:
            return mapping


def _rings(m, ga, gb):
    """Ring atoms stay mapped only in rings mapped whole onto a ring of B."""
    rings_b = {frozenset(r) for r in gb[3]}
    good = [r for r in ga[3]
            if all(i in m for i in r) and frozenset(m[i] for i in r) in rings_b]  # fmt: skip
    keep_a = set().union(*good)
    keep_b = {m[i] for i in keep_a}
    loose_a = set().union(*ga[3]) - keep_a  # ring atoms outside a whole mapped ring
    loose_b = set().union(*gb[3]) - keep_b
    return {i: j for i, j in m.items() if i not in loose_a and j not in loose_b}


def _bonds(m, ga, gb, d):
    """While some mapped pair is bonded differently in A and B (or single <->
    triple), drop the one of its two atoms that maps over the longer distance."""
    nbrs_a, order_a, nbrs_b, order_b = ga[0], ga[1], gb[0], gb[1]
    m = dict(m)
    while True:
        inv = {v: k for k, v in m.items()}
        bad = []
        for i, j in m.items():
            for k in nbrs_a[i]:
                if k in m:
                    o = order_b.get((j, m[k]))
                    if o is None or {order_a[i, k], o} == {1.0, 3.0}:
                        bad.append((i, k))
            for jl in nbrs_b[j]:
                if jl in inv and (i, inv[jl]) not in order_a:
                    bad.append((i, inv[jl]))
        if not bad:
            return m
        worst = max((d[x, m[x]], x) for pair in bad for x in pair)[1]
        m.pop(worst)


def _chirality(m, ga, gb, xa, xb):
    """Unmap a tetrahedral centre whose mapped neighbours wind the other way in B."""
    out = dict(m)
    for i, j in m.items():
        if len(ga[0][i]) != 4 or len(gb[0][j]) != 4:
            continue
        nb = [k for k in ga[0][i] if k in m and m[k] in gb[0][j]][:3]
        if len(nb) < 3:
            continue
        va = np.linalg.det(np.array([xa[k] - xa[i] for k in nb]))
        vb = np.linalg.det(np.array([xb[m[k]] - xb[j] for k in nb]))
        if va * vb < 0:
            out.pop(i)
    return out


def _hydrogens(m, ga, gb):
    """A hydrogen stays mapped only if its heavy atom maps onto its partner's."""
    out = {}
    for i, j in m.items():
        if ga[2][i] == 1:
            hi, hj = ga[0][i], gb[0][j]
            if not hi or not hj or m.get(hi[0]) != hj[0]:
                continue
        out[i] = j
    return out


def _connected(m, ga):
    """The largest connected piece of the mapped atoms (in A's bonds)."""
    left, best = set(m), set()
    while left:
        start = left.pop()
        piece, stack = {start}, [start]
        while stack:
            for k in ga[0][stack.pop()]:
                if k in left:
                    left.discard(k)
                    piece.add(k)
                    stack.append(k)
        heavy = sum(ga[2][k] > 1 for k in piece)
        if heavy > sum(ga[2][k] > 1 for k in best):
            best = piece
    return {i: m[i] for i in sorted(best)}


def score(a, b, mapping: dict, charge_change: float = 0.0) -> float:
    """LOMAP's measure of how small the change is: exp(-0.1 x (heavy atoms of A
    and of B left out of the mapping, plus mapped heavy atoms that change
    element)); 1 for identical, near 0 for unrelated.  A change of net charge
    is multiplied by ``charge_change``: 0 (never run it), or a penalty so a
    network takes one only where it must."""
    from rdkit import Chem

    za = np.array([x.GetAtomicNum() for x in a.GetAtoms()])
    zb = np.array([x.GetAtomicNum() for x in b.GetAtoms()])
    factor = charge_change if Chem.GetFormalCharge(a) != Chem.GetFormalCharge(b) else 1.0
    if factor == 0:
        return 0.0
    mapped = [(i, j) for i, j in mapping.items() if za[i] > 1]
    changed = (int((za > 1).sum()) - len(mapped)) + (int((zb > 1).sum()) - len(mapped))
    swapped = sum(za[i] != zb[j] for i, j in mapped)
    return float(factor * np.exp(-0.1 * (changed + swapped)))
