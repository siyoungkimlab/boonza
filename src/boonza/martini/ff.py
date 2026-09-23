"""Reader for vermouth's force-field (``.ff``) and mapping (``.map``) files.

The format is vermouth's: residue blocks as GROMACS molecule types, links
that add or change interactions wherever their pattern of beads matches,
and modifications (termini, protonation states) that retype beads.  Only
what the Martini 3 protein files use is read.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "martini"

NATOMS = {"bonds": 2, "constraints": 2, "angles": 3, "dihedrals": 4, "impropers": 4, "pairs": 2}
EDGE_SECTIONS = ("bonds", "angles", "dihedrals", "cmap", "constraints")


class Choice(tuple):
    """An attribute that matches any of several values (``"A|B"`` in the files)."""


class NotDefinedOrNot:
    """``not(value)``: the attribute is absent or differs from ``value``."""

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, NotDefinedOrNot) and other.value == self.value

    def __hash__(self):
        return hash(("not", self.value))


def attributes_match(node: dict, template: dict, ignore=()) -> bool:
    """vermouth's rule: every template attribute equals the node's, or is a
    predicate the node's value satisfies (an absent attribute reads as None)."""
    for key, want in template.items():
        if key in ignore:
            continue
        have = node.get(key)
        if have == want:
            continue
        if isinstance(want, Choice) and have in want:
            continue
        if isinstance(want, NotDefinedOrNot) and (key not in node or have != want.value):
            continue
        return False
    return True


@dataclass
class Effector:
    """A parameter computed from the mapped positions, such as ``dihphase(...)``."""

    kind: str
    keys: tuple
    fmt: str | None

    def __call__(self, positions: list) -> str | float:
        x = np.asarray(positions, float)
        if self.kind == "dist":
            value = float(np.linalg.norm(x[1] - x[0]))
        elif self.kind == "angle":
            u, v = x[0] - x[1], x[2] - x[1]
            value = math.degrees(math.acos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
        else:
            value = _dihedral(x)
            if self.kind == "dihphase":
                value -= math.pi
                if value > math.pi:
                    value -= 2 * math.pi
                if value < -math.pi:
                    value += 2 * math.pi
            value = math.degrees(value)
        return format(value, self.fmt) if self.fmt else value


def _dihedral(x) -> float:
    """vermouth.geometry.dihedral, in radians."""
    ab, bc, cd = x[1] - x[0], x[2] - x[1], x[3] - x[2]
    n1, n2 = np.cross(ab, bc), np.cross(bc, cd)
    return math.atan2(np.dot(n1, cd) * np.linalg.norm(bc), np.dot(n1, n2))


@dataclass
class Interaction:
    atoms: tuple
    params: list
    meta: dict
    atom_attrs: list | None = None  # removal templates only


@dataclass
class Block:
    name: str
    nrexcl: int
    atoms: dict = field(default_factory=dict)  # name -> attributes, in file order
    interactions: dict = field(default_factory=lambda: defaultdict(list))
    edges: set = field(default_factory=set)
    meta: dict = field(default_factory=dict)
    apply_meta: dict = field(default_factory=lambda: defaultdict(dict))


@dataclass
class Link:
    nodes: dict = field(default_factory=dict)  # prefixed key -> attributes
    edges: set = field(default_factory=set)
    interactions: dict = field(default_factory=lambda: defaultdict(list))
    removed: dict = field(default_factory=lambda: defaultdict(list))
    non_edges: list = field(default_factory=list)
    patterns: list = field(default_factory=list)
    molmeta: dict = field(default_factory=dict)
    apply_nodes: dict = field(default_factory=dict)
    apply_meta: dict = field(default_factory=lambda: defaultdict(dict))


@dataclass
class Modification:
    name: str = ""
    nodes: dict = field(default_factory=dict)
    edges: set = field(default_factory=set)
    apply_nodes: dict = field(default_factory=dict)
    apply_meta: dict = field(default_factory=lambda: defaultdict(dict))


@dataclass
class ForceField:
    name: str
    blocks: dict = field(default_factory=dict)
    links: list = field(default_factory=list)
    modifications: dict = field(default_factory=dict)
    variables: dict = field(default_factory=dict)


def tokenize(line: str) -> list[str]:
    """Split on whitespace, keeping ``{...}`` attribute groups whole."""
    tokens, depth, cur = [], 0, []
    for ch in line:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch.isspace() and depth == 0:
            if cur:
                tokens.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        tokens.append("".join(cur))
    return tokens


def _substitute(line: str, macros: dict) -> str:
    """Replace ``$name`` by its macro; a name ends at one of `` ${}\\t"``."""
    out, k = [], 0
    while (start := line.find("$", k)) >= 0:
        end = start + 1
        while end < len(line) and line[end] not in ' ${}\t"':
            end += 1
        out += [line[k:start], macros[line[start + 1 : end]]]
        k = end
    return "".join(out) + line[k:]


def _attributes(token: str) -> dict:
    attrs = json.loads(token)
    for key, value in attrs.items():
        if isinstance(value, str) and "|" in value:
            attrs[key] = Choice(value.split("|"))
    return attrs


def _split_prefix(reference: str):
    k = 0
    while k < len(reference) and reference[k] in "+-<>*":
        k += 1
    return reference[:k], reference[k:]


def _node(reference: str, attrs: dict) -> tuple[str, dict]:
    """vermouth's _treat_atom_prefix: the prefix gives the order, and the key
    carries the prefix that the order implies."""
    prefix, base = _split_prefix(reference)
    attrs = dict(attrs)
    if prefix:
        if prefix[0] in "+-":
            order = len(prefix) * (1 if prefix[0] == "+" else -1)
        else:
            order = prefix
        attrs.setdefault("order", order)
        key = reference
    else:
        order = attrs.setdefault("order", 0)
        if isinstance(order, int):
            key = ("+" * order if order > 0 else "-" * -order) + base
        else:
            key = order + base
    attrs.setdefault("atomname", base)
    return key, attrs


def _atoms(tokens: list, natoms: int | None) -> tuple[list, list]:
    """Atom references (with their attributes) and what follows them."""
    atoms, k = [], 0
    while k < len(tokens):
        if tokens[k] == "--":
            k += 1
            break
        if natoms is not None and len(atoms) >= natoms:
            break
        ref, attrs = tokens[k], {}
        k += 1
        if k < len(tokens) and tokens[k].startswith("{"):
            attrs = _attributes(tokens[k])
            k += 1
        atoms.append((ref, attrs))
    return atoms, tokens[k:]


def _params(tokens: list) -> tuple[list, dict]:
    meta = {}
    if tokens and tokens[-1].startswith("{"):
        meta = json.loads(tokens[-1])
        tokens = tokens[:-1]
    params = []
    for tok in tokens:
        open_ = tok.find("(")
        if open_ > 0 and tok.endswith(")") and tok[:open_] in ("dist", "angle", "dihedral",
                                                               "dihphase"):  # fmt: skip
            inner = tok[open_ + 1 : -1]
            fmt = None
            if "|" in inner:
                inner, fmt = inner.split("|")
            params.append(Effector(tok[:open_], tuple(inner.split(",")), fmt))
        else:
            params.append(tok)
    return params, meta


def _link_value(value: str):
    if "|" in value:
        return Choice(json.loads(value).split("|"))
    if "(" in value and value.endswith(")") and not value.startswith("("):
        func, arg = value[: value.find("(")], value[value.find("(") + 1 : -1]
        if func != "not":
            raise ValueError(f"unknown predicate {func!r}")
        return NotDefinedOrNot(json.loads(arg))
    return json.loads(value)


def read_ff(paths, name: str) -> ForceField:
    """Read vermouth ``.ff`` files, in order, into one force field."""
    ff = ForceField(name)
    for path in paths:
        _read_ff_file(Path(path), ff)
    return ff


def _read_ff_file(path: Path, ff: ForceField) -> None:
    macros: dict = {}
    top, section, ctx = None, None, None
    for raw in path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            header = line[1:-1].strip().lower()
            if header in ("moleculetype", "link", "modification", "macros", "variables",
                          "citations"):  # fmt: skip
                top, section = header, header
                if header == "link":
                    ctx = Link()
                    ff.links.append(ctx)
                elif header == "modification":
                    ctx = Modification()
                elif header == "moleculetype":
                    ctx = None
            else:
                section = header
            continue
        if top == "macros":
            key, value = line.split(maxsplit=1)
            macros[key] = value.strip()
            continue
        line = _substitute(line, macros)
        if top == "variables":
            key, value = line.split(maxsplit=1)
            try:
                ff.variables[key] = json.loads(value)
            except json.JSONDecodeError:
                ff.variables[key] = value
            continue
        if top == "citations" or section in ("citation", "citations", "features", "info",
                                             "chiral"):  # fmt: skip
            continue
        if line.startswith("#meta"):
            ctx.apply_meta[section].update(json.loads(line[5:].strip()))
            continue
        if line.startswith("#"):
            raise ValueError(f"{path.name}: unsupported directive {line!r}")
        tokens = tokenize(line)
        if top == "moleculetype":
            if section == "moleculetype":
                ctx = Block(tokens[0], int(tokens[1]))
                ff.blocks[ctx.name] = ctx
            else:
                _block_line(ctx, section, tokens)
        elif top == "link":
            _link_line(ctx, section, tokens)
        elif top == "modification":
            if section == "modification":
                ctx.name = line
                ff.modifications[line] = ctx
            elif section == "atoms":
                ref, attrs = tokens[0], _attributes(tokens[1]) if len(tokens) > 1 else {}
                ctx.nodes[ref] = {**attrs, "atomname": attrs.get("atomname", ref)}
            elif section == "edges":
                ctx.edges.add(frozenset(tokens[:2]))
            else:
                raise ValueError(f"{path.name}: unsupported modification section {section!r}")


def _block_line(block: Block, section: str, tokens: list) -> None:
    if section == "atoms":
        attrs = _attributes(tokens.pop()) if tokens[-1].startswith("{") else {}
        _, atype, resid, resname, name, cgnr, *rest = tokens
        block.atoms[name] = {"atomname": name, "atype": atype, "resname": resname,
                             "resid": int(resid), "charge_group": int(cgnr),
                             "charge": float(rest[0]) if rest else 0.0,
                             **({"mass": float(rest[1])} if len(rest) > 1 else {}),
                             **attrs}  # fmt: skip
        return
    if section == "meta":
        block.meta[tokens[0]] = tokens[1] if len(tokens) > 1 else True
        return
    if section == "edges":
        block.edges.add(frozenset(tokens[:2]))
        return
    atoms, rest = _atoms(tokens, NATOMS.get(section))
    names = list(block.atoms)
    refs = tuple(names[int(r) - 1] if r.isdigit() else r for r, _ in atoms)
    params, meta = _params(rest)
    meta = {**block.apply_meta[section], **meta}
    block.interactions[section].append(Interaction(refs, params, meta))
    if section in EDGE_SECTIONS and meta.get("edge", True):
        block.edges.update(frozenset(p) for p in zip(refs[:-1], refs[1:], strict=True))


def _link_line(link: Link, section: str, tokens: list) -> None:
    if section == "link" or section == "molmeta":
        key, value = tokens
        (link.apply_nodes if section == "link" else link.molmeta)[key] = _link_value(value)
        return
    if section == "atoms":
        ref, attrs = tokens[0], _attributes(tokens[1]) if len(tokens) > 1 else {}
        key, attrs = _node(ref, {**link.apply_nodes, **attrs})
        link.nodes[key] = {**link.nodes.get(key, {}), **attrs}
        return
    if section in ("edges", "non-edges"):
        (a, aa), (b, ba) = _atoms(tokens, 2)[0]
        ka, attrs_a = _node(a, aa)
        kb, attrs_b = _node(b, ba)
        if section == "non-edges":
            link.non_edges.append((ka, {**link.apply_nodes, **attrs_b}))
        else:
            for k, at in ((ka, attrs_a), (kb, attrs_b)):
                link.nodes[k] = {**link.apply_nodes, **at, **link.nodes.get(k, {})}
            link.edges.add(frozenset((ka, kb)))
        return
    if section == "patterns":
        link.patterns.append(_atoms(tokens, None)[0])
        return
    delete = section.startswith("!")
    kind = section.lstrip("!")
    atoms, rest = _atoms(tokens, NATOMS.get(kind))
    keys = []
    for ref, attrs in atoms:
        key, full = _node(ref, {**link.apply_nodes, **attrs})
        link.nodes[key] = {**link.nodes.get(key, {}), **full}
        keys.append(key)
    params, meta = _params(rest)
    meta = {**link.apply_meta[section], **meta}
    if delete:
        link.removed[kind].append(Interaction(tuple(keys), params, meta, [a for _, a in atoms]))
        return
    link.interactions[kind].append(Interaction(tuple(keys), params, meta))
    if kind in EDGE_SECTIONS and meta.get("edge", True):
        link.edges.update(frozenset(p) for p in zip(keys[:-1], keys[1:], strict=True))


def read_map(path) -> dict[str, dict[str, float]]:
    """Atom -> {bead: weight} from a vermouth ``.map`` file; ``!bead`` is weight 0
    and an atom listed for several beads is shared among them."""
    beads_of: dict[str, list[str]] = {}
    section = None
    for raw in Path(path).read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            section = line.strip("[] ").lower()
            continue
        if section == "atoms":
            _, atom, *beads = line.split()
            beads_of.setdefault(atom, []).extend(beads)
    weights = {}
    for atom, beads in beads_of.items():
        real = [b for b in beads if not b.startswith("!")]
        w = {b: real.count(b) / len(real) for b in dict.fromkeys(real)}
        for b in beads:
            if b.startswith("!"):
                w.setdefault(b[1:], 0.0)
        weights[atom] = w
    return weights


def read_rtp_parents(path) -> dict[str, dict[str, str]]:
    """Residue -> {hydrogen: heavy atom it is bonded to}, from a GROMACS ``.rtp``."""
    out: dict[str, dict[str, str]] = {}
    res, section = None, None
    for raw in Path(path).read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            name = line.strip("[] ")
            if name in ("atoms", "bonds", "impropers", "cmap", "bondedtypes"):
                section = name
            else:
                res, section = name, None
                out[res] = {}
            continue
        if section == "bonds" and res:
            a, b = line.split()[:2]
            if a[0] == "H" and not b.startswith(("+", "-")):
                out[res][a] = b
            elif b[0] == "H" and not a.startswith(("+", "-")):
                out[res][b] = a
    return out


def read_modification_mappings(path) -> dict[str, dict[str, dict[str, float]]]:
    """Modification -> atom -> {bead: weight} from a vermouth ``.mapping`` file.

    These weights override the residue's for the atoms they list."""
    out: dict = {}
    name, section = None, None
    for raw in Path(path).read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            section = line.strip("[] ").lower()
            if section == "modification":
                name = None
            continue
        if section == "from blocks":
            name = line.split()[0]
            out.setdefault(name, {})
        elif section == "mapping" and name:
            atom, bead, *weight = line.split()
            out[name].setdefault(atom, {})[bead] = float(weight[0]) if weight else 1.0
    return out
