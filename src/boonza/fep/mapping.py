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

``map_graph`` pairs atoms by the two graphs' maximum common substructure
instead, for ligands that bind alike but are not posed on top of each other (a
series docked or grown in different frames), where the 3D mapping finds little
or nothing to pair.  The poses still decide between a substructure's symmetric
matches, and ``fit_on_core`` then moves B onto A over the pair, since a hybrid
topology needs the atoms it pairs in one place.
"""

from __future__ import annotations

import numpy as np

MAX_DISTANCE = 0.95  # A
#: how atoms are paired between two ligands by default (boonza fep prep --map): by
#: the ligands' graphs (``map_graph``), B then moved onto the atoms they share, since
#: a series' poses need not sit on top of each other for the same part of two ligands
#: to be the same part.  Where they do, it pairs what the poses pair and leaves B
#: where it is.  "pose" is ``map_atoms``, the poses alone (kartograf's way)
MAPPING_METHOD = "graph"
#: symmetric matches of one substructure tried, among which the poses choose
MAX_SYMMETRY = 200


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
    return _refine(mapping, ga, gb, d, xa, xb)


def _refine(mapping: dict, ga, gb, d, xa, xb) -> dict:
    """The rules a hybrid topology needs, applied until none is broken (the module
    docstring).  ``d``: how far every pair of atoms is apart, which decides which
    atom of a pair breaking a rule is dropped."""
    while True:
        before = dict(mapping)
        mapping = _rings(mapping, ga, gb)
        mapping = _bonds(mapping, ga, gb, d)
        mapping = _chirality(mapping, ga, gb, xa, xb)
        mapping = _hydrogens(mapping, ga, gb)
        mapping = _connected(mapping, ga)
        if mapping == before:
            return mapping


def _kabsch(p, q):
    """The rotation and translation that move ``p`` (n, 3) onto ``q`` by least squares."""
    pc, qc = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
    flip = np.sign(np.linalg.det(vt.T @ u.T))  # a rotation, never a reflection
    r = vt.T @ np.diag([1.0, 1.0, flip]) @ u.T
    return r, qc - r @ pc


def fit_on_core(a, b, core: dict) -> np.ndarray:
    """B's positions (Å) moved rigidly so the atoms ``core`` pairs sit on A's.

    A graph mapping says which atoms correspond, not where they are, and a hybrid
    topology needs them in one place: a core atom is one particle at A's position,
    while a bond from it to one of B's own atoms is kept as it is, so B's pose has
    to meet A's or the bond starts stretched across the gap.  Fewer than three
    pairs fix no rotation, and B is left where it is."""
    xb = np.asarray(b.GetConformer().GetPositions(), float)
    if len(core) < 3:
        return xb
    xa = np.asarray(a.GetConformer().GetPositions(), float)
    pairs = sorted(core.items())
    r, t = _kabsch(xb[[j for _, j in pairs]], xa[[i for i, _ in pairs]])
    return (r @ xb.T).T + t


def map_graph(a, b, element_change: bool = True, timeout: int = 10) -> dict:
    """{atom of A: atom of B} from the two graphs' maximum common substructure
    (rings whole) rather than from the poses, for ligands that are not posed on top
    of each other.  Which of that substructure's symmetric matches (a ring's flip,
    two like substituents) is taken is still the poses' to decide: the one whose
    atoms fit A's best once B is moved rigidly onto them.  The rules
    :func:`map_atoms` applies are applied here too, over that fitted geometry, so
    what a hybrid is built from is a mapping of the same kind -- with
    :func:`fit_on_core` for the positions it implies."""
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    ga, gb = _graph(a), _graph(b)
    heavy_a = [i for i, z in enumerate(ga[2]) if z > 1]
    heavy_b = [i for i, z in enumerate(gb[2]) if z > 1]
    ha, hb = Chem.RemoveHs(a), Chem.RemoveHs(b)  # heavy atoms keep their order
    res = rdFMCS.FindMCS(
        [ha, hb], ringMatchesRingOnly=True, completeRingsOnly=True,
        atomCompare=(rdFMCS.AtomCompare.CompareAny if element_change
                     else rdFMCS.AtomCompare.CompareElements),
        bondCompare=rdFMCS.BondCompare.CompareAny,  # _bonds drops single <-> triple
        timeout=timeout,
    )  # fmt: skip
    if not res.numAtoms:
        return {}
    pattern = Chem.MolFromSmarts(res.smartsString)
    in_a = ha.GetSubstructMatches(pattern, uniquify=False, maxMatches=MAX_SYMMETRY)
    in_b = hb.GetSubstructMatches(pattern, uniquify=False, maxMatches=MAX_SYMMETRY)
    if not in_a or not in_b:
        return {}
    xa = np.asarray(a.GetConformer().GetPositions(), float)
    xb = np.asarray(b.GetConformer().GetPositions(), float)
    best = None
    for match_a in in_a:
        p = xa[[heavy_a[k] for k in match_a]]
        for match_b in in_b:
            q = xb[[heavy_b[k] for k in match_b]]
            r, t = _kabsch(q, p)
            rmsd = float(np.sqrt((((r @ q.T).T + t - p) ** 2).sum(1).mean()))
            if best is None or rmsd < best[0]:
                best = (rmsd, match_a, match_b)
    core = {heavy_a[k]: heavy_b[x] for k, x in zip(best[1], best[2], strict=True)}
    xb = fit_on_core(a, b, core)
    d = cdist(xa, xb)
    for i, j in list(core.items()):  # a pair's hydrogens, by where the fit leaves them
        hi = [k for k in ga[0][i] if ga[2][k] == 1]
        hj = [k for k in gb[0][j] if gb[2][k] == 1]
        if not hi or not hj:
            continue
        rows, cols = linear_sum_assignment(d[np.ix_(hi, hj)])
        core.update({hi[r]: hj[c] for r, c in zip(rows, cols, strict=True)})
    return _refine(core, ga, gb, d, xa, xb)


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
