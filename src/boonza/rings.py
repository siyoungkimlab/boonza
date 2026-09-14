"""Rings: msys's smallest set of smallest rings (SSSR) and ring systems.

A port of msys src/analyze/sssr.cxx, Copyright D. E. Shaw Research (see
NOTICE): biconnected components (Hopcroft-Tarjan), a cycle basis minimized
after Berger, Gritzmann and de Vries, and optionally all relevant cycles
(Bauer).  Pseudo particles and metals are left out, as in msys.  Rings, and
the atom order within each ring, match msys.

    rings = boonza.sssr(system)                   # list of atom-index lists
    rings = boonza.sssr(system, all_relevant=True)
    boonza.ring_systems(system, rings)            # fused rings, as ring indices
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numba import njit


def _metal_or_virtual(anum: np.ndarray) -> np.ndarray:
    return ((anum < 1) | ((anum >= 21) & (anum <= 32)) | ((anum >= 39) & (anum <= 51))
            | ((anum >= 57) & (anum <= 84)) | ((anum >= 89) & (anum <= 117)))  # fmt: skip


@njit(cache=True)
def _biconnected(n, off, adj, ea, eb):
    """Edges of each biconnected component in msys's pop order.

    Returns the tail vertex and edge of each popped edge and the component
    boundaries in that sequence.
    """
    ne = len(ea)
    depth = np.full(n, -1, np.int64)
    s_idx = np.empty(n, np.int64)
    s_low = np.empty(n, np.int64)
    s_par = np.empty(n, np.int64)
    s_next = np.empty(n, np.int64)
    e_tail = np.empty(ne, np.int64)
    e_edge = np.empty(ne, np.int64)
    pop_tail = np.empty(ne, np.int64)
    pop_edge = np.empty(ne, np.int64)
    bounds = np.zeros(ne + 1, np.int64)
    npop = 0
    ncomp = 0
    for root in range(n):
        if off[root + 1] == off[root] or depth[root] >= 0:
            continue
        top = 0
        s_idx[0], s_low[0], s_par[0], s_next[0] = root, 0, -1, 0
        depth[root] = 0
        etop = 0
        while top >= 0:
            v = s_idx[top]
            found = False
            while s_next[top] < off[v + 1] - off[v]:
                e = adj[off[v] + s_next[top]]
                s_next[top] += 1
                w = eb[e] if ea[e] == v else ea[e]
                if depth[w] == -1:
                    e_tail[etop], e_edge[etop] = v, e
                    etop += 1
                    top += 1
                    s_idx[top], s_low[top], s_par[top], s_next[top] = w, depth[v] + 1, v, 0
                    depth[w] = depth[v] + 1
                    found = True
                    break
                elif depth[w] < depth[v] and w != s_par[top]:
                    e_tail[etop], e_edge[etop] = v, e
                    etop += 1
                    if depth[w] < s_low[top]:
                        s_low[top] = depth[w]
            if found:
                continue
            low = s_low[top]
            top -= 1
            if top >= 0:
                if low < s_low[top]:
                    s_low[top] = low
                if low >= depth[s_idx[top]]:
                    pd = depth[s_idx[top]]
                    while etop > 0 and depth[e_tail[etop - 1]] > pd:
                        etop -= 1
                        pop_tail[npop], pop_edge[npop] = e_tail[etop], e_edge[etop]
                        npop += 1
                    etop -= 1
                    pop_tail[npop], pop_edge[npop] = e_tail[etop], e_edge[etop]
                    npop += 1
                    ncomp += 1
                    bounds[ncomp] = npop
    return pop_tail[:npop], pop_edge[:npop], bounds[: ncomp + 1]


class _Sub:
    """Edge-indicator subgraph with an ordered vertex list (msys SSSR::Subgraph)."""

    __slots__ = ("edges", "vertices")

    def __init__(self, edges, vertices):
        self.edges = edges
        self.vertices = vertices

    def copy(self) -> _Sub:
        return _Sub(self.edges.copy(), list(self.vertices))


class _Graph:
    def __init__(self, edges, nv):
        self.edges = edges
        self.v_to_e = [[] for _ in range(nv)]
        for k, (a, b) in enumerate(edges):
            self.v_to_e[a].append(k)
            self.v_to_e[b].append(k)

    def other(self, e, v):
        a, b = self.edges[e]
        return b if a == v else a


def _subgraph_path(g, sub, start, end, covered) -> _Sub:
    """Shortest path from start to end using edges of ``sub``; marks them in ``covered``."""
    visited = [False] * len(g.v_to_e)
    visited[start] = True
    queue = deque([[(-1, start)]])
    while queue:
        path = queue[0]
        v = path[-1][1]
        found = False
        for e in g.v_to_e[v]:
            if not sub[e]:
                continue
            w = g.other(e, v)
            if w == end:
                path.append((e, w))
                found = True
                break
            if visited[w]:
                continue
            visited[w] = True
            queue.append([*path, (e, w)])
        if found:
            break
        queue.popleft()
    if not queue:
        raise RuntimeError("no path found between vertices in subgraph")
    path = queue[0]
    out = _Sub([False] * len(g.edges), [start] + [w for _, w in path[1:]])
    for e, _ in path[1:]:
        out.edges[e] = True
        covered[e] = True
    return out


def _odd_path(g, odd, start, end, multiple, max_length) -> list[_Sub]:
    """Shortest path(s) from start to end crossing an odd number of ``odd`` edges."""
    nv, ne = len(g.v_to_e), len(g.edges)
    d_odd, d_even = [-1] * nv, [-1] * nv
    d_even[start] = 0
    queue = deque([[(-1, start, False)]])
    shortest = -1
    paths = []
    while queue:
        path = queue[0]
        v, parity = path[-1][1], path[-1][2]
        depth = len(path)
        for e in g.v_to_e[v]:
            o = parity ^ odd[e]
            w = g.other(e, v)
            if w == end and o:
                sub = _Sub([False] * ne, [start] + [p[1] for p in path[1:]] + [end])
                for p in path[1:]:
                    sub.edges[p[0]] = True
                sub.edges[e] = True
                paths.append(sub)
                if not multiple:
                    return paths
                shortest = len(sub.vertices) - 1
            else:
                seen = d_odd if o else d_even
                if seen[w] != -1 and seen[w] < depth:
                    continue
                seen[w] = depth
                if (shortest == -1 or depth < shortest) and (
                    max_length == -1 or depth < max_length
                ):
                    queue.append([*path, (e, w, o)])
        queue.popleft()
    return paths


def _cycle_basis(g):
    ne, nv = len(g.edges), len(g.v_to_e)
    basis, pivots, non_pivots = deque(), deque(), []
    allowed, covered, removed = [True] * ne, [False] * ne, [True] * ne
    for i in range(ne):
        if covered[i]:
            continue
        pivots.appendleft(i)
        removed[i] = False
        allowed[i] = False
        cycle = _subgraph_path(g, allowed, *g.edges[i], covered)
        cycle.edges[i] = True
        basis.appendleft(cycle)
        allowed[i] = True
    nfixed = len(pivots)
    basis, pivots = list(basis), list(pivots)
    tree, visited = [False] * ne, [False] * nv
    visited[0] = True
    queue = deque([0])
    while queue:
        v = queue.popleft()
        for e in g.v_to_e[v]:
            if not removed[e]:
                continue
            w = g.other(e, v)
            if not visited[w]:
                tree[e] = visited[w] = True
                non_pivots.append(e)
                queue.append(w)
            elif not tree[e] and w < v:
                pivots.append(e)
    for p in pivots[nfixed:]:
        cycle = _subgraph_path(g, tree, *g.edges[p], covered)
        cycle.edges[p] = True
        basis.append(cycle)
    return basis, pivots, non_pivots, nfixed


def _minimize(g, basis, pivots, nfixed, non_pivots) -> list[_Sub]:
    ne, nv = len(g.edges), len(g.v_to_e)
    out = [basis[i].copy() for i in range(nfixed)]
    for i in range(nfixed, len(pivots)):
        U, touched = [False] * ne, [False] * nv
        U[pivots[i]] = True
        for v in g.edges[pivots[i]]:
            touched[v] = True
        for j in range(i - 1, -1, -1):
            u = basis[j].edges[pivots[i]]
            for k in range(j + 1, i):
                u ^= U[pivots[k]] & basis[j].edges[pivots[k]]
            U[pivots[j]] = u
            if u:
                for v in g.edges[pivots[j]]:
                    touched[v] = True
        replaced, shortest = False, basis[i].copy()
        for v in range(nv):
            if not touched[v]:
                continue
            found = _odd_path(g, U, v, v, False, -1)
            if not found:
                raise RuntimeError("no odd path between vertices")
            found[0].vertices.pop()
            if len(found[0].vertices) < len(shortest.vertices):
                replaced, shortest = True, found[0]
        if not replaced:
            out.append(basis[i].copy())
            continue
        out.append(shortest.copy())
        edges = shortest.edges
        for j in range(len(pivots)):
            if j < i and edges[pivots[j]]:
                for k in range(j, len(pivots)):
                    edges[pivots[k]] ^= basis[j].edges[pivots[k]]
                for k in non_pivots:
                    edges[k] ^= basis[j].edges[k]
            basis[i].edges[pivots[j]] = edges[pivots[j]]
        for k in non_pivots:
            basis[i].edges[k] = edges[k]
    return out


def _relevant(g, min_basis, pivots) -> list[list[int]]:
    ne, npiv = len(g.edges), len(pivots)
    kernels = [[False] * ne for _ in range(npiv)]
    for i in range(npiv):
        kernels[i][pivots[i]] = True

    def eliminate(i, j):
        for k in range(npiv):
            if k >= i:
                min_basis[j].edges[pivots[k]] ^= min_basis[i].edges[pivots[k]]
            kernels[k][pivots[j]] ^= kernels[k][pivots[i]]

    for i in range(npiv):
        if not min_basis[i].edges[pivots[i]]:
            j = next((j for j in range(i + 1, npiv) if min_basis[j].edges[pivots[i]]), None)
            if j is None:
                raise RuntimeError("min basis indicator matrix is singular")
            min_basis[i].edges, min_basis[j].edges = min_basis[j].edges, min_basis[i].edges
            for kern in kernels:
                kern[pivots[i]], kern[pivots[j]] = kern[pivots[j]], kern[pivots[i]]
        for j in range(i + 1, npiv):
            if min_basis[j].edges[pivots[i]]:
                eliminate(i, j)
    for i in range(npiv - 1, 0, -1):
        for j in range(i):
            if min_basis[j].edges[pivots[i]]:
                eliminate(i, j)

    paths = []
    for i in range(npiv):
        checked = set()
        for j in range(ne):
            if not kernels[i][j]:
                continue
            for v in g.edges[j]:
                if v not in checked:
                    checked.add(v)
                    paths += _odd_path(g, kernels[i], v, v, True, len(min_basis[i].vertices))

    unique = {}
    for p in paths:
        path = p.vertices[:-1]
        n = len(path)
        lo = min(range(n), key=path.__getitem__)
        right, left = (lo + 1) % n, (lo - 1) % n
        verts = [path[lo]]
        step = 1 if path[right] < path[left] else -1
        j = (lo + step) % n
        while j != lo:
            verts.append(path[j])
            j = (j + step) % n
        unique[(tuple(verts), tuple(p.edges))] = verts
    return [unique[k] for k in sorted(unique)]


def _atom_ids(system, atoms) -> np.ndarray:
    if atoms is None:
        return np.arange(system.natoms)
    if isinstance(atoms, str):
        return system.select(atoms).ids
    return np.asarray(system._ids("atom", atoms), dtype=np.int64)


def sssr(system, atoms=None, all_relevant: bool = False) -> list[list[int]]:
    """Smallest set of smallest rings among ``atoms`` (default all), as msys GetSSSR.

    The SSSR is not unique; ``all_relevant`` returns the union of all such
    sets (every relevant cycle).  Each ring lists its atoms in ring order.
    """
    s = system
    ids = _atom_ids(s, atoms)
    n = len(ids)
    anum = s.atoms["anum"]
    keep = ~_metal_or_virtual(anum)
    where = np.full(s.natoms, -1, np.int64)
    where[ids] = np.arange(n)
    real = np.full(s.natoms, -1, np.int64)
    sel_real = anum[ids] >= 1
    real[ids[sel_real]] = np.flatnonzero(sel_real)
    bi, bj = s.bonds["i"], s.bonds["j"]
    lo, hi = np.minimum(bi, bj), np.maximum(bi, bj)
    ok = keep[lo] & keep[hi] & (where[lo] >= 0) & (real[hi] >= 0) & (lo != hi)
    bond_ids = np.flatnonzero(ok)
    # msys adds each edge while visiting its lower atom, in bond order
    order = np.lexsort((bond_ids, where[lo[bond_ids]]))
    bond_ids = bond_ids[order]
    ea, eb = where[lo[bond_ids]], real[hi[bond_ids]]
    ends = np.empty(2 * len(ea), np.int64)
    ends[0::2], ends[1::2] = ea, eb
    by_vertex = np.argsort(ends, kind="stable")
    adj = (by_vertex // 2).astype(np.int64)
    off = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(ends, minlength=n), out=off[1:])
    pop_tail, pop_edge, bounds = _biconnected(n, off, adj, ea, eb)

    rings = []
    ea_l, eb_l = ea.tolist(), eb.tolist()
    tails, edges = pop_tail.tolist(), pop_edge.tolist()
    bounds = bounds.tolist()
    for c in range(len(bounds) - 1):
        if bounds[c + 1] - bounds[c] < 2:  # a bridge: two vertices
            continue
        g2c, c2g, cedges = {}, [], []
        for v1, e in zip(tails[bounds[c] : bounds[c + 1]], edges[bounds[c] : bounds[c + 1]],
                         strict=True):  # fmt: skip
            v2 = eb_l[e] if ea_l[e] == v1 else ea_l[e]
            for v in (v1, v2):
                if v not in g2c:
                    g2c[v] = len(c2g)
                    c2g.append(v)
            cedges.append((g2c[v1], g2c[v2]))
        g = _Graph(cedges, len(c2g))
        basis, pivots, non_pivots, nfixed = _cycle_basis(g)
        min_basis = _minimize(g, basis, pivots, nfixed, non_pivots)
        cycles = (_relevant(g, min_basis, pivots) if all_relevant
                  else [b.vertices for b in min_basis])  # fmt: skip
        rings += [[int(ids[c2g[v]]) for v in cyc] for cyc in cycles]
    return rings


def ring_systems(system, rings) -> list[list[int]]:
    """Group ``rings`` into fused systems sharing bonds (msys RingSystems); ring indices."""
    bond_rings: dict[int, list[int]] = {}
    ring_bonds = []
    for k, ring in enumerate(rings):
        ids = []
        for a, b in zip(ring, [*ring[1:], ring[0]], strict=True):
            bond = system.find_bond(a, b)
            if bond is None:
                raise ValueError(f"ring bond {a}-{b} not found in system")
            ids.append(bond.id)
            bond_rings.setdefault(bond.id, []).append(k)
        ring_bonds.append(ids)
    processed = set()
    systems = []
    for bond in sorted(bond_rings):
        if bond in processed:
            continue
        processed.add(bond)
        members, seen_bonds, queue = set(), {bond}, deque([bond])
        while queue:
            for ring in bond_rings.get(queue.popleft(), []):
                members.add(ring)
                for rb in ring_bonds[ring]:
                    if rb not in seen_bonds:
                        seen_bonds.add(rb)
                        processed.add(rb)
                        queue.append(rb)
        systems.append(sorted(members))
    return systems
