"""Applying vermouth links: find every place a link's beads match, then add,
replace or remove interactions there.

A link matches as vermouth's does: an induced subgraph isomorphism (bonded
beads must be bonded in the molecule, unbonded ones unbonded), bead
attributes equal to the link's, residue orders as the link's prefixes say,
no forbidden neighbours (``non-edges``), and at least one of its
``patterns`` when it has any.  Matches are searched residue by residue
rather than over the whole graph.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from .ff import Effector, Interaction, Link, attributes_match


class CGMolecule:
    """Beads, the bonds between them, and their interactions, one molecule."""

    def __init__(self):
        self.nodes: list[dict] = []
        self.adj: list[set] = []
        self.interactions: dict[str, list[Interaction]] = defaultdict(list)
        self.meta: dict = {}
        self.positions: list = []  # nm, per bead
        self._by_name: dict = {}
        self._names: dict = defaultdict(list)
        self._slots: dict = defaultdict(list)  # (kind, atoms) -> list positions

    def add_node(self, attrs: dict, position) -> int:
        k = len(self.nodes)
        self.nodes.append(attrs)
        self.adj.append(set())
        self.positions.append(position)
        self._by_name[(attrs["resid"], attrs["atomname"])] = k
        self._names[attrs["atomname"]].append(k)
        return k

    def add_edge(self, a: int, b: int) -> None:
        if a != b:
            self.adj[a].add(b)
            self.adj[b].add(a)

    def find(self, resid: int, name: str) -> int | None:
        return self._by_name.get((resid, name))

    def by_name(self, name: str) -> list[int]:
        return self._names.get(name, [])

    def add(self, kind: str, interaction: Interaction) -> None:
        lst = self.interactions[kind]
        self._slots[(kind, interaction.atoms)].append(len(lst))
        lst.append(interaction)

    def add_or_replace(self, kind: str, atoms: tuple, params: list, meta: dict) -> None:
        """Same atoms, same order and same ``version``: replace; otherwise add."""
        version = meta.get("version", 0)
        lst = self.interactions[kind]
        for k in self._slots.get((kind, atoms), ()):
            if lst[k] is not None and lst[k].meta.get("version", 0) == version:
                lst[k] = Interaction(atoms, params, meta)
                return
        self.add(kind, Interaction(atoms, params, meta))

    def remove_matching(self, kind: str, template: Interaction) -> None:
        """Remove the first interaction the template matches, if any."""
        lst = self.interactions.get(kind, [])
        for k in self._slots.get((kind, template.atoms), ()):
            old = lst[k]
            if old is None:
                continue
            if template.params and list(template.params) != list(old.params):
                continue
            attrs = template.atom_attrs or [{}] * len(old.atoms)
            pairs = zip(old.atoms, attrs, strict=True)
            if not all(attributes_match(self.nodes[a], t) for a, t in pairs):
                continue
            if not attributes_match(old.meta, template.meta):
                continue
            lst[k] = None  # compacted once the links are applied
            return

    def compact(self) -> None:
        for kind, lst in self.interactions.items():
            self.interactions[kind] = [t for t in lst if t is not None]
        self._slots.clear()


def _atoms_match(node: dict, template: dict) -> bool:
    mods = node.get("modifications", [])
    if "modifications" in template:
        want = template["modifications"]
        if not want:
            ok = not mods
        elif not mods:
            ok = False
        elif isinstance(want, list):
            ok = sorted(mods) == sorted(want)
        else:
            ok = all(attributes_match({"_": m}, {"_": want}) for m in mods)
        if not ok:
            return False
    return attributes_match(node, template, ignore=("order", "replace", "modifications"))


def _order_kind(order):
    if isinstance(order, bool):
        raise ValueError(f"invalid order {order!r}")
    if isinstance(order, int):
        return "number", order
    if order[0] in "><":
        return "><", (1 if order[0] == ">" else -1) * len(order)
    return "*", len(order)


def _sign(x) -> int:
    return (x > 0) - (x < 0)


def match_order(order1, resid1, order2, resid2) -> bool:
    """vermouth's match_order: whether two residues satisfy their link orders."""
    (t1, o1), (t2, o2) = _order_kind(order1), _order_kind(order2)
    if t1 == "number":
        if t2 == "number":
            return o2 - o1 == resid2 - resid1
        if o1 == 0:
            if t2 == "><" and _sign(resid2 - resid1) != _sign(o2):
                return False
            if t2 == "*" and resid1 == resid2:
                return False
    elif t1 == "><":
        if t2 == "number" and o2 == 0 and _sign(resid1 - resid2) != _sign(o1):
            return False
        if t2 == "><" and _sign(resid2 - resid1) != _sign(o2 - o1):
            return False
    else:
        if t2 == "number" and o2 == 0 and resid1 == resid2:
            return False
        if t2 == "*" and (o1 == o2) != (resid1 == resid2):
            return False
    return True


def _search_order(link: Link) -> list[str]:
    """Numbered beads first, starting from one at order 0; each next bead
    bonded to one already placed when possible."""
    keys = list(link.nodes)
    numbered = [k for k in keys if isinstance(link.nodes[k]["order"], int)]
    start = min(numbered, key=lambda k: abs(link.nodes[k]["order"])) if numbered else keys[0]
    out, left = [start], [k for k in keys if k != start]
    while left:
        placed = set(out)
        nxt = next((k for k in left if isinstance(link.nodes[k]["order"], int)), None)
        if nxt is None:
            nxt = next((k for k in left if any(frozenset((k, p)) in link.edges
                                              for p in placed)), left[0])  # fmt: skip
        out.append(nxt)
        left.remove(nxt)
    return out


def match_link(mol: CGMolecule, link: Link):
    """Yield {link key: bead} for every match of ``link`` in ``mol``."""
    if not attributes_match(mol.meta, link.molmeta):
        return
    order = _search_order(link)
    nodes = link.nodes
    first = order[0]
    first_order = nodes[first]["order"]

    def candidates(key, assigned):
        attrs = nodes[key]
        o = attrs["order"]
        if isinstance(o, int) and isinstance(first_order, int) and first in assigned:
            anchor = mol.nodes[assigned[first]]["resid"] - first_order
            k = mol.find(anchor + o, attrs["atomname"])
            return [] if k is None else [k]
        for p, bead in assigned.items():
            if frozenset((key, p)) in link.edges:
                return sorted(mol.adj[bead])
        return mol.by_name(attrs["atomname"])

    def extend(i, assigned):
        if i == len(order):
            yield dict(assigned)
            return
        key = order[i]
        used = set(assigned.values())
        for bead in candidates(key, assigned):
            if bead in used or not _atoms_match(mol.nodes[bead], nodes[key]):
                continue
            ok = all((frozenset((key, p)) in link.edges) == (b in mol.adj[bead])
                     for p, b in assigned.items())  # fmt: skip
            if ok:
                assigned[key] = bead
                yield from extend(i + 1, assigned)
                del assigned[key]

    for match in extend(0, {}):
        if not _non_edges_ok(mol, link, match):
            continue
        if link.patterns and not any(_pattern_ok(mol, p, match) for p in link.patterns):
            continue
        if _orders_ok(mol, link, match):
            yield match


def _non_edges_ok(mol, link, match) -> bool:
    for key, to_attrs in link.non_edges:
        if key not in match:
            continue
        bead = match[key]
        resid = mol.nodes[bead]["resid"]
        for nb in mol.adj[bead]:
            if mol.nodes[nb]["resid"] == resid + to_attrs.get("order", 0) and _atoms_match(
                mol.nodes[nb], to_attrs
            ):
                return False
    return True


def _pattern_ok(mol, pattern, match) -> bool:
    for key, attrs in pattern:
        if key not in match or not _atoms_match(mol.nodes[match[key]], attrs):
            return False
    return True


def _orders_ok(mol, link, match) -> bool:
    resid_of: dict = {}
    for key, bead in match.items():
        o, r = link.nodes[key]["order"], mol.nodes[bead]["resid"]
        if resid_of.setdefault(o, r) != r:
            return False
    return all(match_order(o1, r1, o2, r2)
               for (o1, r1), (o2, r2) in combinations(resid_of.items(), 2))  # fmt: skip


def _params(mol, params, match):
    return [p([mol.positions[match[k]] for k in p.keys]) if isinstance(p, Effector) else p
            for p in params]  # fmt: skip


def apply_links(mol: CGMolecule, links: list[Link]) -> None:
    """Apply every link, in file order, wherever it matches."""
    for link in links:
        for match in list(match_link(mol, link)):
            for key, attrs in link.nodes.items():
                if "replace" in attrs:
                    mol.nodes[match[key]].update(attrs["replace"])
            for kind, templates in link.removed.items():
                for t in templates:
                    atoms = tuple(match[k] for k in t.atoms)
                    mol.remove_matching(kind, Interaction(atoms, _params(mol, t.params, match),
                                                          t.meta, t.atom_attrs))  # fmt: skip
            for kind, templates in link.interactions.items():
                for t in templates:
                    atoms = tuple(match[k] for k in t.atoms)
                    mol.add_or_replace(kind, atoms, _params(mol, t.params, match), dict(t.meta))
    mol.compact()
