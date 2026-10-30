"""Which ligand pairs to run, and from their ΔΔG, a value for every ligand.

Edges: every pair is scored as LOMAP does, exp(-0.1 x heavy atoms that change),
on its 3D atom mapping (boonza.fep.mapping: atoms that sit on top of each other
in the poses), or with ``method="2d"`` on the two graphs' maximum common
substructure (MCS), which leaves the poses out of the choice.  A spanning tree
over 1 - score then joins every ligand through the easiest changes, with three
rules:

* an edge where more than ``max_change`` heavy atoms change is taken only where
  no route through smaller ones joins its ligands (a whole substituent grown in
  one edge samples poorly);
* no ligand gets more than ``max_degree`` edges, the redundant ones included,
  unless it can be joined no other way without a large edge (said in the log):
  no hubs, whose one ligand's error every spoke inherits -- but a hub of small
  changes before a large change;
* a ``reference`` ligand (the one with a crystal pose) is preferred, and free of
  the cap, so the network grows out from the pose that is known.

Each ligand left on one edge then gets its next best within the rules, so every
ligand sits in a cycle and a bad edge shows as a cycle that does not close.
Whatever scored the pairs, an edge is built from its 3D mapping.

Values: the ΔG of each ligand (relative to their mean, or to a reference) by
weighted least squares over the edges' ΔΔG -- the maximum likelihood estimate
cinnabar makes -- with its covariance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mapping import map_atoms, score

#: heavy atoms an edge may change before it is taken only where no smaller route is
MAX_CHANGE = 8
#: edges a ligand may have (its redundant one included)
MAX_DEGREE = 4
#: what a reference ligand's edges are preferred by (1 - score)
REFERENCE_BONUS = 0.15
#: a 3D mapping this many heavy atoms smaller than the MCS is warned about
MAPPING_SHORTFALL = 2


@dataclass
class Edge:
    a: int  # ligand indices
    b: int
    mapping: dict  # atom of a -> atom of b
    score: float
    changed: int = 0  # heavy atoms that change (unmapped on either side, or swapped)
    role: str = "tree"  # or "redundant": added so a ligand on one edge sits in a cycle


def changed_heavy(a, b, mapping: dict) -> int:
    """Heavy atoms of A and of B left out of ``mapping``, plus mapped ones that
    change element: what LOMAP's score counts."""
    za = np.array([x.GetAtomicNum() for x in a.GetAtoms()])
    zb = np.array([x.GetAtomicNum() for x in b.GetAtoms()])
    mapped = [(i, j) for i, j in mapping.items() if za[i] > 1]
    swapped = sum(int(za[i] != zb[j]) for i, j in mapped)
    return int((za > 1).sum() + (zb > 1).sum() - 2 * len(mapped) + swapped)


def mcs_heavy(a, b, timeout: int = 10) -> int:
    """Heavy atoms of the two graphs' maximum common substructure (rings whole,
    elements matched)."""
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    ha, hb = Chem.RemoveHs(a), Chem.RemoveHs(b)
    res = rdFMCS.FindMCS([ha, hb], ringMatchesRingOnly=True, completeRingsOnly=True,
                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareOrder, timeout=timeout)  # fmt: skip
    return int(res.numAtoms)


def _pair(a, b, i, j, method, max_distance, charge_change, log) -> Edge | None:
    """The scored edge between ligands ``a`` (index i) and ``b`` (j), or None."""
    from rdkit import Chem

    m = map_atoms(a, b, max_distance=max_distance)
    if not m:
        return None
    s = score(a, b, m, charge_change)
    changed = changed_heavy(a, b, m)
    if method == "2d":
        common = mcs_heavy(a, b)
        heavy = sum(x.GetAtomicNum() > 1 for x in a.GetAtoms())
        heavy += sum(x.GetAtomicNum() > 1 for x in b.GetAtoms())
        mapped3d = (heavy - changed) // 2
        factor = charge_change if Chem.GetFormalCharge(a) != Chem.GetFormalCharge(b) else 1.0
        changed = heavy - 2 * common
        s = float(factor * np.exp(-0.1 * changed))
        if s > 0 and mapped3d < common - MAPPING_SHORTFALL and log:
            log(f"  warning: ligands {i} and {j} share {common} heavy atoms as graphs, "
                f"but only {mapped3d} sit on top of each other in their poses")  # fmt: skip
    return Edge(i, j, m, s, changed) if s > 0 else None


def _weight(e: Edge, max_change: int, reference) -> float:
    w = 1.0 - e.score
    if max_change and e.changed > max_change:
        w += 1.0  # after every small edge: taken only where nothing smaller joins
    if reference is not None and reference in (e.a, e.b):
        w -= REFERENCE_BONUS
    return w


def _choose(n, scored: dict, fixed=(), redundant=True, max_change=MAX_CHANGE,
            max_degree=MAX_DEGREE, reference=None, free=None, log=None) -> dict:  # fmt: skip
    """The pairs to add to ``fixed`` (pairs kept as they are), each with its role: a
    spanning tree under the module's rules ("tree"), then the redundant edges of
    the ligands ``free`` (default: all) left on one ("redundant")."""
    parent = list(range(n))

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    degree = [0] * n
    for a, b in fixed:
        parent[root(a)] = root(b)
        degree[a] += 1
        degree[b] += 1

    def capped(k):
        return bool(max_degree) and k != reference and degree[k] >= max_degree

    order = sorted(scored, key=lambda p: (_weight(scored[p], max_change, reference), p))

    def large(p):
        return bool(max_change) and scored[p].changed > max_change

    chosen = {}
    # small edges within the cap; small ones past it; only then large ones: a hub of
    # small changes is better than a large change
    for allow_large, strict in ((False, True), (False, False), (True, True), (True, False)):
        for a, b in order:
            if root(a) == root(b) or (large((a, b)) and not allow_large):
                continue
            if strict and (capped(a) or capped(b)):
                continue
            if not strict and log:
                over = [k for k in (a, b) if capped(k)]
                if over:
                    log(f"  ligand(s) {over} past {max_degree} edges: no other edge joins them")
            parent[root(a)] = root(b)
            chosen[a, b] = "tree"
            degree[a] += 1
            degree[b] += 1
    if redundant:
        for node in sorted(free if free is not None else range(n)):
            if degree[node] != 1:
                continue
            for p in order:
                if node not in p or p in chosen or p in fixed:
                    continue
                other = p[0] if p[1] == node else p[1]
                if capped(other) or (max_change and scored[p].changed > max_change):
                    continue
                chosen[p] = "redundant"
                degree[node] += 1
                degree[other] += 1
                break
    return chosen


def _unjoined(n, pairs) -> list[list[int]]:
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(pairs)
    return [sorted(c) for c in nx.connected_components(g)]


def edges(ligands, redundant: bool = True, max_distance: float = 0.95,
          charge_change: float = 0.0, method: str = "3d", max_change: int = MAX_CHANGE,
          max_degree: int = MAX_DEGREE, reference: int | None = None,
          log=None) -> list[Edge]:  # fmt: skip
    """The edges to run between posed ``ligands`` (RDKit molecules with hydrogens),
    under the module's rules.  A change of net charge scores ``charge_change``
    times LOMAP's score (boonza.fep.mapping.score): 0 never takes one.  Pairs that
    cannot be joined (score 0) are never chosen; ligands they leave alone raise an
    error.  ``max_change`` or ``max_degree`` 0: no such rule."""
    from rdkit import Chem

    n = len(ligands)
    if n < 2:
        raise ValueError("a network needs at least two ligands")
    if method not in ("3d", "2d"):
        raise ValueError(f"method is 3d or 2d, not {method!r}")
    scored = {}
    for i in range(n):
        for j in range(i + 1, n):
            e = _pair(ligands[i], ligands[j], i, j, method, max_distance, charge_change, log)
            if e is not None:
                scored[i, j] = e
    parts = _unjoined(n, scored)
    if len(parts) > 1:
        charges = {Chem.GetFormalCharge(m) for m in ligands}
        why = (" (their net charges differ, and a change of charge needs a receptor: "
               "relative hydration free energies keep the charge)"
               if len(charges) > 1 and not charge_change else "")  # fmt: skip
        raise ValueError(f"the ligands fall apart into groups no mapping joins: {parts}{why}")
    chosen = _choose(n, scored, (), redundant, max_change, max_degree, reference, log=log)
    return [_as(scored[p], role) for p, role in sorted(chosen.items())]


def extend(ligands, existing, new, redundant: bool = True, max_distance: float = 0.95,
           charge_change: float = 0.0, method: str = "3d", max_change: int = MAX_CHANGE,
           max_degree: int = MAX_DEGREE, reference: int | None = None,
           log=None) -> list[Edge]:  # fmt: skip
    """The edges that join ligands ``new`` (indices) to a network already run:
    ``existing`` its (a, b) pairs, kept as they are and counted against the cap.
    Only pairs with a new ligand are scored; the rest as :func:`edges` does."""
    n, new = len(ligands), sorted(set(new))
    if not new:
        raise ValueError("no new ligands to add")
    scored = {}
    for j in new:
        for i in range(n):
            if i == j or (i in new and i > j):
                continue
            a, b = min(i, j), max(i, j)
            e = _pair(ligands[a], ligands[b], a, b, method, max_distance, charge_change, log)
            if e is not None:
                scored[a, b] = e
    have = {tuple(sorted(p)) for p in existing}
    parts = _unjoined(n, list(have) + list(scored))
    if len(parts) > 1:
        alone = [c for c in parts if set(c) <= set(new)]
        raise ValueError(f"no mapping joins these new ligands to the network: {alone}")
    chosen = _choose(n, scored, have, redundant, max_change, max_degree, reference,
                     free=new, log=log)  # fmt: skip
    return [_as(scored[p], role) for p, role in sorted(chosen.items())]


def _as(e: Edge, role: str) -> Edge:
    e.role = role
    return e


def summary(names, pairs, changed, settings: dict) -> str:
    """One line on how a network was built and what it came out as: ``pairs``
    (a, b) every edge, ``changed`` their heavy atoms that change (None unknown)."""
    degree = np.zeros(len(names), int)
    for a, b in pairs:
        degree[a] += 1
        degree[b] += 1
    top = int(degree.argmax())
    rules = [settings.get("method", "3d"), f"max change {settings.get('max_change')}",
             f"max degree {settings.get('max_degree')}"]  # fmt: skip
    if settings.get("reference") is not None:
        rules.append(f"reference {settings['reference']}")
    known = [c for c in changed if c is not None]
    sizes = (f"; heavy atoms changed: median {np.median(known):g}, max {max(known)}"
             if known else "")  # fmt: skip
    return (f"Network ({', '.join(rules)}): {len(pairs)} edges over {len(names)} ligands; "
            f"edges per ligand {degree.min()}-{degree.max()} ({names[top]}: {degree[top]})"
            + sizes)  # fmt: skip


def draw(path_stem, names, pairs, changed=None, roles=None, reference=None,
         max_change: int = MAX_CHANGE, log=print) -> list:  # fmt: skip
    """The network as a picture, ``<path_stem>_white.pdf`` and ``_black.pdf``: each
    ligand a labelled node (the reference ringed), each edge a line coloured by the
    heavy atoms it changes (past ``max_change`` in red), dashed where redundant;
    the files, or [] without matplotlib."""
    import warnings
    from pathlib import Path

    try:
        import matplotlib
    except ImportError:
        return []
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import networkx as nx

    from .plot_correlation import THEMES, font_ready, style

    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    g = nx.Graph()
    g.add_nodes_from(range(len(names)))
    g.add_edges_from(pairs)
    xy = nx.kamada_kawai_layout(g) if g.number_of_edges() else nx.circular_layout(g)
    changed = list(changed) if changed is not None else [None] * len(pairs)
    roles = list(roles) if roles is not None else ["tree"] * len(pairs)
    known = [c for c in changed if c is not None]
    top = max(known + [max_change or 1, 1])
    cmap = mpl.colormaps["viridis"]
    font = font_ready(log)
    written = []
    size = max(4.5, 0.45 * len(names) ** 0.75 + 3)
    for background, ink in THEMES.items():
        rc = style(ink)
        if not font:
            rc.pop("font.family")
        with mpl.rc_context(rc), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig, ax = plt.subplots(figsize=(size, size))
            for (a, b), c, role in zip(pairs, changed, roles, strict=True):
                color = (ink if c is None else "tab:red" if max_change and c > max_change
                         else cmap(0.85 * c / top))  # fmt: skip
                (x0, y0), (x1, y1) = xy[a], xy[b]
                ax.plot([x0, x1], [y0, y1], color=color, lw=2.0, zorder=1,
                        ls="--" if role == "redundant" else "-")  # fmt: skip
                if c is not None:
                    ax.text((x0 + x1) / 2, (y0 + y1) / 2, str(c), fontsize=12, color=ink,
                            ha="center", va="center", zorder=3)  # fmt: skip
            for k, name in enumerate(names):
                ref = reference is not None and k == reference
                ax.scatter(*xy[k], s=260 if ref else 160, color="tab:blue", zorder=2,
                           edgecolors="tab:orange" if ref else "none", linewidths=2.5)  # fmt: skip
                ax.annotate(name, xy[k], xytext=(0, 11), textcoords="offset points",
                            ha="center", va="bottom", fontsize=12, color=ink, zorder=4)  # fmt: skip
            ax.set_axis_off()
            ax.margins(0.12)
            note = "line: heavy atoms changed; dashed: redundant"
            if max_change:
                note += f"; red: more than {max_change}"
            ax.set_title(note, fontsize=12)
            path = stem.with_name(f"{stem.name}_{background}.pdf")
            fig.savefig(path, format="pdf", transparent=True)
            plt.close(fig)
            written.append(path)
    return written


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
