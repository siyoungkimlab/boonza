"""Which ligand pairs to run, and from their ΔΔG, a value for every ligand.

Edges: every pair is mapped and scored (boonza.fep.mapping); a minimal
spanning tree on 1 - score joins all ligands through the easiest changes, and
each ligand left on one edge gets its next best too, so every ligand sits in a
cycle and a bad edge shows as a cycle that does not close.

Values: the ΔG of each ligand (relative to their mean, or to a reference) by
weighted least squares over the edges' ΔΔG -- the maximum likelihood estimate
cinnabar makes -- with its covariance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mapping import map_atoms, score


@dataclass
class Edge:
    a: int  # ligand indices
    b: int
    mapping: dict  # atom of a -> atom of b
    score: float


def edges(ligands, redundant: bool = True, max_distance: float = 0.95,
          charge_change: float = 0.0) -> list[Edge]:  # fmt: skip
    """The edges to run between posed ``ligands`` (RDKit molecules with hydrogens).
    A change of net charge scores ``charge_change`` times LOMAP's score
    (boonza.fep.mapping.score): 0 never takes one.  Pairs that cannot be joined
    (score 0) are never chosen; ligands they leave alone raise an error."""
    import networkx as nx
    from rdkit import Chem

    n = len(ligands)
    if n < 2:
        raise ValueError("a network needs at least two ligands")
    scored = {}
    for i in range(n):
        for j in range(i + 1, n):
            m = map_atoms(ligands[i], ligands[j], max_distance=max_distance)
            s = score(ligands[i], ligands[j], m, charge_change) if m else 0.0
            if s > 0:
                scored[i, j] = Edge(i, j, m, s)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for (i, j), e in scored.items():
        g.add_edge(i, j, weight=1.0 - e.score)
    if not nx.is_connected(g):
        alone = [sorted(c) for c in nx.connected_components(g)]
        charges = {Chem.GetFormalCharge(m) for m in ligands}
        why = (" (their net charges differ, and a change of charge needs a receptor: "
               "relative hydration free energies keep the charge)"
               if len(charges) > 1 and not charge_change else "")  # fmt: skip
        raise ValueError(f"the ligands fall apart into groups no mapping joins: {alone}{why}")
    tree = nx.minimum_spanning_tree(g)
    chosen = {tuple(sorted(e)) for e in tree.edges}
    if redundant:
        for node in range(n):
            if tree.degree(node) == 1:
                pairs = [tuple(sorted((node, k))) for k in g[node]]
                spare = sorted((g[node][k]["weight"], pair)
                               for k, pair in zip(g[node], pairs, strict=True)
                               if pair not in chosen)  # fmt: skip
                if spare:
                    chosen.add(spare[0][1])
    return [scored[e] for e in sorted(chosen)]


def extend(ligands, existing, new, redundant: bool = True, max_distance: float = 0.95,
           charge_change: float = 0.0) -> list[Edge]:  # fmt: skip
    """The edges that join ligands ``new`` (indices) to a network already run:
    ``existing`` its (a, b) pairs, kept as they are.  Only pairs with a new ligand
    are mapped; a minimal spanning tree over them, the existing edges free, joins
    every new ligand through the easiest changes, and with ``redundant`` each new
    ligand left on one edge gets its next best too, as :func:`edges` does."""
    import networkx as nx

    n, new = len(ligands), sorted(set(new))
    if not new:
        raise ValueError("no new ligands to add")
    scored = {}
    for j in new:
        for i in range(n):
            if i == j or (i in new and i > j):
                continue
            a, b = min(i, j), max(i, j)
            m = map_atoms(ligands[a], ligands[b], max_distance=max_distance)
            s = score(ligands[a], ligands[b], m, charge_change) if m else 0.0
            if s > 0:
                scored[a, b] = Edge(a, b, m, s)
    have = {tuple(sorted(p)) for p in existing}
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(have, weight=0.0)
    for (i, j), e in scored.items():
        g.add_edge(i, j, weight=1.0 - e.score)
    if not nx.is_connected(g):
        alone = [sorted(c) for c in nx.connected_components(g) if set(c) <= set(new)]
        raise ValueError(f"no mapping joins these new ligands to the network: {alone}")
    tree = nx.minimum_spanning_tree(g)
    chosen = {tuple(sorted(e)) for e in tree.edges} - have
    if redundant:
        degree = {k: 0 for k in new}
        for a, b in have | chosen:
            for k in (a, b):
                if k in degree:
                    degree[k] += 1
        for node in new:
            if degree[node] == 1:
                spare = sorted((1.0 - scored[p].score, p) for p in scored
                               if node in p and p not in chosen)  # fmt: skip
                if spare:
                    chosen.add(spare[0][1])
    return [scored[e] for e in sorted(chosen)]


def estimate(n: int, results, reference: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """ΔG of each of ``n`` ligands from ``results`` [(a, b, ΔΔG_ab, error)], by
    weighted least squares; relative to ligand ``reference``, else to their mean.
    Returns the values and their standard errors."""
    rows, rhs, w = [], [], []
    for a, b, ddg, err in results:
        r = np.zeros(n)
        r[a], r[b] = -1.0, 1.0
        rows.append(r)
        rhs.append(ddg)
        w.append(1.0 / max(float(err), 1e-3) ** 2)
    A, y, W = np.array(rows), np.array(rhs), np.diag(w)
    fisher = A.T @ W @ A
    # the gauge: values sum to zero (or the reference is zero)
    c = np.zeros(n)
    if reference is None:
        c[:] = 1.0
    else:
        c[reference] = 1.0
    kkt = np.block([[fisher, c[:, None]], [c[None, :], np.zeros((1, 1))]])
    if np.linalg.matrix_rank(kkt) < n + 1:
        raise ValueError("the results do not connect every ligand")
    rhs_full = np.concatenate([A.T @ W @ y, [0.0]])
    g = np.linalg.solve(kkt, rhs_full)[:n]
    cov = np.linalg.inv(kkt)[:n, :n]
    return g, np.sqrt(np.clip(np.diag(cov), 0, None))


def cycle_closures(n: int, results) -> list[tuple[list[int], float]]:
    """Each independent cycle of the results and the sum of its ΔΔG (should be ~0)."""
    import networkx as nx

    g = nx.Graph()
    value = {}
    for a, b, ddg, _ in results:
        g.add_edge(a, b)
        value[a, b], value[b, a] = ddg, -ddg
    out = []
    for cycle in nx.cycle_basis(g):
        total = sum(value[cycle[k], cycle[(k + 1) % len(cycle)]] for k in range(len(cycle)))
        out.append((cycle, float(total)))
    return out
