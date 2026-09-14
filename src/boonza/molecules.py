"""Groups of identical molecules, as msys FindDistinctFragments.

    groups = boonza.distinct_fragments(system)   # {representative fragid: [fragids]}

Two molecules (fragments) are identical when their bond graphs are
isomorphic with atoms matched by element and number of bonded non-pseudo
atoms, as msys compares them: bond orders and stereochemistry are ignored
and pseudo particles are not part of the graph.  Each group is keyed by its
lowest fragment id and lists its fragment ids in increasing order.

The common case is vectorized and exact: atoms get isomorphism-invariant
colors (element, degree, then a few rounds of neighborhood refinement over
the whole system), each molecule's atoms are ordered by (color, index), and
molecules with the same color sequence and the same bond list in that order
are identical by construction (the ordering is an explicit isomorphism;
every member is checked element by element, not just by hash).  Molecules
that share a color composition but not a bond list (e.g. the same molecule
written in a different atom order) are merged by an exact backtracking
isomorphism test between one representative of each class.
"""

from __future__ import annotations

import numpy as np
from numba import njit

_ROUNDS = 3  # refinement rounds; colors only speed things up, correctness never depends on them


@njit(cache=True)
def _neighbor_sums(colors, off, nbr, weights):
    n = colors.shape[0]
    out = np.empty(n, np.uint64)
    for i in range(n):
        acc = np.uint64(0)
        for k in range(off[i], off[i + 1]):
            acc += weights[colors[nbr[k]]]
        out[i] = acc
    return out


@njit(cache=True)
def _fragment_hashes(seq, fstart, elo, ehi, estart):
    nf = fstart.shape[0] - 1
    h1 = np.empty(nf, np.uint64)
    h2 = np.empty(nf, np.uint64)
    prime = np.uint64(1099511628211)
    for f in range(nf):
        h = np.uint64(1469598103934665603)
        for k in range(fstart[f], fstart[f + 1]):
            h = (h ^ np.uint64(seq[k])) * prime
        h1[f] = h
        h = np.uint64(1469598103934665603)
        for k in range(estart[f], estart[f + 1]):
            h = (h ^ np.uint64(elo[k])) * prime
            h = (h ^ np.uint64(ehi[k])) * prime
        h2[f] = h
    return h1, h2


@njit(cache=True)
def _verify(fo, gstart, seq, fstart, elo, ehi, estart):
    """For each position in ``fo``, whether it differs from its group's first fragment."""
    bad = np.zeros(fo.shape[0], np.bool_)
    for g in range(gstart.shape[0] - 1):
        r = fo[gstart[g]]
        for p in range(gstart[g] + 1, gstart[g + 1]):
            f = fo[p]
            same = True
            a0, b0 = fstart[r], fstart[f]
            for k in range(fstart[r + 1] - a0):
                if seq[a0 + k] != seq[b0 + k]:
                    same = False
                    break
            if same:
                a0, b0 = estart[r], estart[f]
                for k in range(estart[r + 1] - a0):
                    if elo[a0 + k] != elo[b0 + k] or ehi[a0 + k] != ehi[b0 + k]:
                        same = False
                        break
            bad[p] = not same
    return bad


def _relabel(*keys) -> np.ndarray:
    """Compact integer labels for the distinct tuples of ``keys`` (first key most significant)."""
    order = np.lexsort(keys[::-1])
    change = np.zeros(len(order), bool)
    for k in keys:
        ks = k[order]
        change[1:] |= ks[1:] != ks[:-1]
    labels = np.empty(len(order), np.int64)
    labels[order] = np.cumsum(change)
    return labels


class _Adjacency:
    """Neighbor sets from a CSR adjacency, built on demand."""

    def __init__(self, off, nbr):
        self.off, self.nbr, self.cache = off, nbr, {}

    def __getitem__(self, a):
        s = self.cache.get(a)
        if s is None:
            s = self.cache[a] = set(self.nbr[self.off[a] : self.off[a + 1]].tolist())
        return s


def _isomorphic(atoms_a, atoms_b, colors, adj) -> bool:
    """Exact check that fragments a and b are isomorphic, respecting colors."""
    if len(atoms_a) != len(atoms_b):
        return False
    if not len(atoms_a):
        return True
    if (np.sort(colors[atoms_a]) != np.sort(colors[atoms_b])).any():
        return False
    # match a's atoms in BFS order so each new atom has mapped neighbors to check
    start = int(atoms_a[0])
    order, seen = [start], {start}
    for a in order:
        for nb in adj[a]:
            if nb not in seen:
                seen.add(nb)
                order.append(nb)
    order += [int(a) for a in atoms_a if int(a) not in seen]
    by_color: dict[int, list[int]] = {}
    for b in atoms_b.tolist():
        by_color.setdefault(int(colors[b]), []).append(b)
    a2b: dict[int, int] = {}
    used: set[int] = set()

    def candidates(a):
        mapped = [a2b[x] for x in adj[a] if x in a2b]
        pool = adj[mapped[0]] if mapped else by_color[int(colors[a])]
        c = colors[a]
        deg = len(adj[a])
        for b in pool:
            if b in used or colors[b] != c or len(adj[b]) != deg:
                continue
            if all(a2b[x] in adj[b] for x in adj[a] if x in a2b):
                yield b

    stack = [candidates(order[0])]
    while stack:
        depth = len(stack) - 1
        a = order[depth]
        if a in a2b:  # undo the previous choice at this depth
            used.discard(a2b.pop(a))
        b = next(stack[-1], None)
        if b is None:
            stack.pop()
            continue
        a2b[a] = b
        used.add(b)
        if depth + 1 == len(order):
            return True
        stack.append(candidates(order[depth + 1]))
    return False


def distinct_fragments(system) -> dict[int, list[int]]:
    """{representative fragment id: fragment ids of all identical molecules}."""
    s = system
    n, nfrag = s.natoms, s.nfragments
    if nfrag == 0:
        return {}
    frag = np.asarray(s.fragids, np.int64)
    anum = s.atoms["anum"]
    real = anum > 0
    bi, bj = s.bonds["i"], s.bonds["j"]
    keep = real[bi] & real[bj]
    bi, bj = bi[keep].astype(np.int64), bj[keep].astype(np.int64)
    src = np.concatenate([bi, bj])
    dst = np.concatenate([bj, bi])
    off = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=off[1:])
    nbr = dst[np.argsort(src, kind="stable")]

    # isomorphism-invariant atom colors
    colors = _relabel(anum.astype(np.int64), np.diff(off))
    rng = np.random.default_rng(12345)
    for _ in range(_ROUNDS):
        ncolors = int(colors.max()) + 1
        weights = rng.integers(1, 2**63, size=ncolors, dtype=np.uint64)
        new = _relabel(colors, _neighbor_sums(colors, off, nbr, weights))
        if int(new.max()) + 1 == ncolors:
            break
        colors = new

    # each molecule's real atoms ordered by (color, index); bonds as sorted rank pairs
    ra = np.flatnonzero(real)
    o = ra[np.lexsort((ra, colors[ra], frag[ra]))]
    fstart = np.searchsorted(frag[o], np.arange(nfrag + 1))
    rank = np.zeros(n, np.int64)
    rank[o] = np.arange(len(o)) - fstart[frag[o]]
    ef = frag[bi]
    lo, hi = np.minimum(rank[bi], rank[bj]), np.maximum(rank[bi], rank[bj])
    eo = np.lexsort((hi, lo, ef))
    elo, ehi = lo[eo], hi[eo]
    estart = np.searchsorted(ef[eo], np.arange(nfrag + 1))
    seq = colors[o]
    h1, h2 = _fragment_hashes(seq, fstart, elo, ehi, estart)
    nat, ned = np.diff(fstart), np.diff(estart)

    # candidate classes: same size, bond count and hashes; verified exactly
    fo = np.lexsort((np.arange(nfrag), h2, h1, ned, nat))
    change = np.zeros(nfrag, bool)
    for key in (nat, ned, h1, h2):
        kf = key[fo]
        change[1:] |= kf[1:] != kf[:-1]
    gstart = np.concatenate([[0], np.flatnonzero(change), [nfrag]])
    bad = _verify(fo, gstart, seq, fstart, elo, ehi, estart)
    classes = []
    for g in range(len(gstart) - 1):
        seg = fo[gstart[g] : gstart[g + 1]]
        mask = bad[gstart[g] : gstart[g + 1]]
        classes.append(seg[~mask])
        classes.extend(np.array([f]) for f in seg[mask].tolist())

    # merge classes that are isomorphic under a different atom order
    buckets: dict[bytes, list[int]] = {}
    for c, members in enumerate(classes):
        r = int(members[0])
        key = nat[r].tobytes() + seq[fstart[r] : fstart[r + 1]].tobytes()
        buckets.setdefault(key, []).append(c)
    parent = list(range(len(classes)))

    def root(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    adj = _Adjacency(off, nbr)
    for cs in buckets.values():
        if len(cs) < 2:
            continue
        reps: list[int] = []
        for c in sorted(cs, key=lambda c: int(classes[c][0])):
            atoms_c = o[fstart[classes[c][0]] : fstart[classes[c][0] + 1]]
            for r in reps:
                atoms_r = o[fstart[classes[r][0]] : fstart[classes[r][0] + 1]]
                if _isomorphic(atoms_r, atoms_c, colors, adj):
                    parent[root(c)] = root(r)
                    break
            else:
                reps.append(c)
    merged: dict[int, list[np.ndarray]] = {}
    for c, members in enumerate(classes):
        merged.setdefault(root(c), []).append(members)
    groups = {}
    for parts in merged.values():
        members = np.sort(np.concatenate(parts))
        groups[int(members[0])] = members.tolist()
    return {r: groups[r] for r in sorted(groups)}
