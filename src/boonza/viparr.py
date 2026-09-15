"""viparr force fields: read the directories and parameterize systems with them.

A viparr force field (the format of D. E. Shaw Research's viparr 3 and of the
``viparr-ffpublic`` collection) is a directory of JSON files:

- ``rules``: the Lennard-Jones form and combining rule, the exclusion rule
  and 1-4 scale factors, and the plugins (functional forms) to apply;
- ``templates*``: residue templates with atom types, charges, bonds,
  impropers, CMAP tuples and virtual sites. ``$1``, ``$2``, ... name atoms of
  neighbouring residues;
- one file per parameter table (``stretch_harm``, ``angle_harm``,
  ``dihedral_trig``, ``improper_harm``, ``improper_trig``,
  ``ureybradley_harm``, ``torsiontorsion_cmap``, ``vdw1``, ``vdw1_14``,
  ``vdw2``, ``mass``, ``virtuals_*``): rows of atom-type patterns and
  parameters;
- ``cmap``: the CMAP grids that ``torsiontorsion_cmap`` rows refer to.

:func:`parameterize` follows viparr step for step, and the test suite checks
its output term by term against viparr itself:

- each residue matches a template by its bond graph: the element and number
  of bonds of every atom, bonds to neighbouring residues included. Names do
  not matter;
- a molecule (bonded fragment) is parameterized by the *first* force field in
  the list whose templates match all its residues. A later force field that
  also matches it is ignored with a warning; two templates of one force field
  matching the same residue is an error;
- bonds, angles and proper dihedrals come from the bond graph; impropers,
  CMAP tuples, extra exclusions and virtual sites come from the templates;
- a parameter row matches a tuple of atom types exactly or through ``*``
  wildcards, forwards or backwards. An exact row wins; otherwise the first
  wildcard row in file order. A proper dihedral takes every consecutive row
  with the same types (multi-term CHARMM dihedrals);
- 1-2 and 1-3 pairs, and 1-4 pairs as the rules say, are excluded; scaled 1-4
  interactions go to ``pair_12_6_es`` (with ``vdw1_14`` types when present),
  and ``vdw2`` rows become NBFIX overrides of the nonbonded table.

:func:`merge_forcefields` reproduces viparr's ``-m``/``-a`` patches: the
patch's templates replace templates of the same name, and its parameter rows
replace rows with the same types.

Where boonza differs from viparr on purpose:

- CMAP grids describe L amino acids. viparr cannot tell mirror images apart
  (templates match by graph), so D residues get the L map. boonza reads the
  chirality of each CMAP residue from its coordinates and gives D residues the
  mirrored map, E_D(phi, psi) = E_L(-phi, -psi), which is how CHARMM builds
  its D-amino-acid parameters. ``cmap_chirality=False`` reproduces viparr.
- With two force fields that both have CMAP tables, viparr re-points the
  first one's CMAP terms to the second one's grids; boonza keeps each force
  field's own grids.
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np

from ._columns import STR
from .elements import symbol
from .schemas import TERM_SCHEMAS
from .system import NonbondedInfo, System
from .terms import ParamTable

__all__ = [
    "ParamRow",
    "Rules",
    "Template",
    "ViparrError",
    "ViparrForcefield",
    "ViparrWarning",
    "build_constraints",
    "bundled_version",
    "find_forcefield",
    "list_forcefields",
    "load_forcefield",
    "merge_forcefields",
    "parameterize",
]


class ViparrError(ValueError):
    """A force field that cannot be read, or a system it cannot parameterize."""


class ViparrWarning(UserWarning):
    """What viparr reports as a warning (unmatched optional terms, ...)."""


_BOND_STRINGS = frozenset({"-", "=", "#", ":", "~"})
_BOND_STRING = {1: "-", 2: "=", 3: "#"}
# viparr's registry of Lennard-Jones forms: name -> (nonbonded table, rules)
_VDW_FUNCS = {"lj12_6_sig_epsilon": ("vdw_12_6", ("geometric", "arithmetic/geometric"))}
# (N-CA).((C-CA)x(CB-CA)) is positive at the CA of an L amino acid
_L_SIGN = 1.0


# ---- force field directories -------------------------------------------------


def _search_path(path=None) -> list[Path]:
    if path is None:
        path = os.environ.get("VIPARR_FFPATH", "")
    if isinstance(path, str | os.PathLike):
        path = str(path).split(os.pathsep)
    return [Path(p).expanduser() for p in path if str(p)]


_BUNDLED = Path(__file__).resolve().parent / "data" / "viparr-ffpublic.zip"


@cache
def _bundled():
    """The ``ff`` directory of the viparr-ffpublic copy shipped with boonza."""
    return zipfile.Path(zipfile.ZipFile(_BUNDLED), at="ff/")


def bundled_version() -> str:
    """Which viparr-ffpublic commit is bundled with boonza."""
    return zipfile.Path(_bundled().root, at="VERSION").read_text().strip()


def list_forcefields(path=None, bundled: bool = True) -> list[str]:
    """Force field names in ``path`` (default ``$VIPARR_FFPATH``, colon-separated)
    and, with ``bundled``, in the viparr-ffpublic copy shipped with boonza."""
    dirs = _search_path(path) + ([_bundled()] if bundled else [])
    names = set()
    for d in dirs:
        if d.is_dir():
            names.update(p.name for p in d.iterdir() if (p / "rules").is_file())
    return sorted(names)


def find_forcefield(name, path=None):
    """The directory of force field ``name``: a directory path, a name in
    ``path`` (default ``$VIPARR_FFPATH``, searched in order), or a name in the
    viparr-ffpublic copy shipped with boonza."""
    p = Path(name).expanduser()
    if p.is_dir():
        return p
    dirs = _search_path(path)
    for d in dirs:
        if (d / str(name)).is_dir():
            return d / str(name)
    bundled = _bundled() / str(name)
    if bundled.is_dir():
        return bundled
    where = ", ".join([*map(str, dirs), "the bundled viparr-ffpublic"])
    raise ViparrError(f"force field {name!r} not found; looked in {where}")


def _read_json(path):
    try:
        with path.open() as fh:
            return json.load(fh)
    except json.JSONDecodeError as e:
        raise ViparrError(f"Misformatted '{path}' file: {e}") from e


# ---- rules -------------------------------------------------------------------


@dataclass
class Rules:
    """A force field's ``rules`` file.

    ``es_scale[k]`` and ``lj_scale[k]`` scale the interactions of atoms
    ``k + 2`` atoms apart counting both ends (1-2, 1-3, 1-4, ...);
    :attr:`exclusions` is the farthest excluded separation.
    """

    info: list[str] = field(default_factory=list)
    vdw_func: str = ""
    vdw_comb_rule: str = ""
    es_scale: list[float] = field(default_factory=list)
    lj_scale: list[float] = field(default_factory=list)
    plugins: list[str] = field(default_factory=list)
    fatal: bool = True
    nbfix_identifier: str = ""

    @property
    def exclusions(self) -> int:
        return len(self.es_scale) + 1

    def _is_empty(self) -> bool:
        return (not self.info and not self.vdw_func and not self.vdw_comb_rule
                and not self.plugins and self.exclusions == 1 and self.fatal)  # fmt: skip


def _read_rules(path: Path) -> Rules:
    js = _read_json(path)
    if not isinstance(js, dict):
        raise ViparrError(f"'{path}' must be of object type")
    excl, es, lj = js.get("exclusions"), js.get("es_scale"), js.get("lj_scale")
    if excl is not None and not isinstance(excl, int):
        raise ViparrError("'exclusions' field must be of int type")
    rule = excl if excl is not None else len(es) + 1 if es is not None else (
        len(lj) + 1 if lj is not None else 4)  # fmt: skip
    es = [float(x) for x in es or []]
    lj = [float(x) for x in lj or []]
    if es or lj:
        if rule != len(es) + 1 or rule != len(lj) + 1:
            raise ViparrError("Incorrect number of es or lj scaling factors for exclusion rule")
    else:
        es, lj = [0.0] * (rule - 1), [0.0] * (rule - 1)
    plugins = []
    for i, entry in enumerate(js.get("plugins", [])):
        if isinstance(entry, list):
            if len(entry) != 2 or not isinstance(entry[0], str) or not isinstance(entry[1], int):
                raise ViparrError(f"Plugin {i} must be of string type")
            name = "mass2" if entry[0] == "mass" and entry[1] == 1 else entry[0]
        elif isinstance(entry, str):
            name = entry
        else:
            raise ViparrError(f"Plugin {i} must be of string type")
        plugins.append(name)
    return Rules(info=[str(x) for x in js.get("info", [])],
                 vdw_func=str(js.get("vdw_func", "")).lower(),
                 vdw_comb_rule=str(js.get("vdw_comb_rule", "")).lower(),
                 es_scale=es, lj_scale=lj, plugins=plugins, fatal=bool(js.get("fatal", True)),
                 nbfix_identifier=str(js.get("nbfix_identifier", "")))  # fmt: skip


def _merge_vdw(old: str, new: str) -> str:
    if not old:
        return new
    if new and new != old:
        raise ViparrError(f"Incompatible VDW functions or combine rules: {old}, {new}")
    return old


# ---- templates -----------------------------------------------------------------


class _Graph:
    """msys's Graph: nodes are atoms with atomic number >= 1, colored by
    element and number of non-pseudo bonds (bonds leaving the atom set count);
    ``match`` is msys's depth-first isomorphism search, so equivalent atoms
    are mapped exactly as viparr maps them."""

    __slots__ = ("ids", "attr", "nbr")

    def __init__(self, atoms, anum, nbrs):
        parts: dict[tuple[int, int], int] = {}
        members: list[list[int]] = []
        for a in atoms:
            if anum[a] < 1:
                continue
            key = (anum[a], len(nbrs[a]))
            k = parts.get(key)
            if k is None:
                k = parts[key] = len(members)
                members.append([])
            members[k].append(a)
        order = sorted(range(len(members)), key=lambda k: (len(members[k]), k))
        self.ids = [a for k in order for a in members[k]]
        index = {a: i for i, a in enumerate(self.ids)}
        self.attr, self.nbr = [], []
        for a in self.ids:
            degree, nb = 0, []
            for o in nbrs[a]:
                c = anum[o]
                if c == 0:  # pseudo particle
                    continue
                degree += 1
                if c == -1:  # external atom of a template
                    continue
                i = index.get(o)
                if i is not None:
                    nb.append(i)
            self.attr.append((anum[a], degree))
            self.nbr.append(nb)

    def match(self, other: _Graph):
        """Pairs (own atom, other's atom) of the first isomorphism, or None."""
        n = len(self.ids)
        if n != len(other.ids):
            return None
        if n == 0:
            return []
        for i in range(n):
            perm = self._match_common(other, 0, i)
            if perm is not None:
                return perm
        return None

    def _match_node(self, g, h, other, GtoH, HtoG) -> bool:
        if self.attr[g] != other.attr[h]:
            return False
        gn, hn = self.nbr[g], other.nbr[h]
        for x in gn:
            y = GtoH[x]
            if y != -1 and y not in hn:
                return False
        for y in hn:
            x = HtoG[y]
            if x != -1 and x not in gn:
                return False
        return True

    def _match_common(self, other: _Graph, root: int, oroot: int):
        n = len(self.ids)
        if len(self.nbr[root]) != len(other.nbr[oroot]) or self.attr[root] != other.attr[oroot]:
            return None
        GtoH, HtoG = [-1] * n, [-1] * n
        GtoH[root], HtoG[oroot] = oroot, root
        if n == 1:
            return [(self.ids[0], other.ids[0])]
        if not self.nbr[root]:
            return None
        matched_g, matched_h, progress = [root], [oroot], []
        g, gi, h, hi = root, 0, oroot, -1
        while True:
            gnbr, hnbr = self.nbr[g][gi], -1
            hlist = other.nbr[h]
            for j in range(hi + 1, len(hlist)):
                cand = hlist[j]
                if HtoG[cand] != -1 or not self._match_node(gnbr, cand, other, GtoH, HtoG):
                    continue
                hi, hnbr = j, cand
                break
            if hnbr != -1:
                progress.append((g, gi, h, hi))
                GtoH[gnbr], HtoG[hnbr] = hnbr, gnbr
                matched_g.append(gnbr)
                matched_h.append(hnbr)
                if len(matched_g) == n:
                    return [(self.ids[i], other.ids[GtoH[i]]) for i in range(n)]
                g, gi, h, hi = gnbr, -1, hnbr, -1
                while True:  # the next unmatched neighbor of g, backing up if needed
                    nxt = next((i for i in range(gi + 1, len(self.nbr[g]))
                                if GtoH[self.nbr[g][i]] == -1), None)  # fmt: skip
                    if nxt is not None:
                        gi = nxt
                        break
                    if not progress:
                        raise ViparrError(
                            "graph isomorphism error: check that the graph is connected"
                        )
                    g, gi, h, hi = progress.pop()
                hi = -1
            else:
                if not progress:
                    return None
                lg, lgi, lh, lhi = progress[-1]
                gn, hn = self.nbr[lg][lgi], other.nbr[lh][lhi]
                while matched_g[-1] != gn:
                    GtoH[matched_g.pop()] = -1
                    HtoG[matched_h.pop()] = -1
                GtoH[gn] = HtoG[hn] = -1
                matched_g.pop()
                matched_h.pop()
                g, gi, h, hi = progress.pop()


def _formula_hash(atoms, anum, nbrs) -> tuple:
    """msys Graph::hash: element counts and 2 x internal + external bonds."""
    counts: Counter = Counter()
    nb = 0
    for a in atoms:
        if anum[a] < 1:
            continue
        counts[anum[a]] += 1
        nb += sum(1 for o in nbrs[a] if anum[o] != 0)
    return (nb, tuple(sorted(counts.items())))


def _formula(anums) -> str:
    counts = Counter(a for a in anums if a > 0)
    return "".join(symbol(a) + (str(n) if n > 1 else "") for a, n in sorted(counts.items()))


@dataclass(eq=False)
class Template:
    """A residue template.

    Its atoms are the template's own atoms, then the external atoms ``$1``,
    ``$2``, ... of neighbouring residues (atomic number -1), then pseudo
    particles (atomic number 0). Tuples hold indices into that list.
    ``external_anum`` pins the element of external atoms: the atom of the
    neighbouring residue must have that atomic number (``external_elements``
    in the template file, a boonza extension that viparr ignores), and
    ``external_formula`` the chemical formula of that atom's residue
    (``external_residues``), so that, say, a Cys bound to a ligand's sulfur
    is told apart from a Cys in a disulfide.
    """

    name: str
    names: list[str]
    anum: list[int]
    charge: list[float]
    btype: list[str]
    nbtype: list[str]
    pset: list[str]
    bonds: list[tuple[int, int]]
    impropers: list[tuple[int, ...]] = field(default_factory=list)
    cmaps: list[tuple[int, ...]] = field(default_factory=list)
    exclusions: list[tuple[int, ...]] = field(default_factory=list)
    pseudos: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)
    external_anum: dict[int, int] = field(default_factory=dict)
    external_formula: dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        self._nbrs = [[] for _ in self.names]
        for i, j in self.bonds:
            self._nbrs[i].append(j)
            self._nbrs[j].append(i)
        every = range(len(self.names))
        self.hash = _formula_hash(every, self.anum, self._nbrs)
        self.graph = _Graph(every, self.anum, self._nbrs)

    @property
    def natoms(self) -> int:
        """Number of real atoms (no external atoms or pseudo particles)."""
        return sum(1 for a in self.anum if a > 0)

    @property
    def total_charge(self) -> float:
        return float(sum(c for c, a in zip(self.charge, self.anum, strict=True) if a != -1))

    def __repr__(self) -> str:
        ext = sum(1 for a in self.anum if a == -1)
        return (f"<Template {self.name}: {self.natoms} atoms, {ext} external bonds, "
                f"{len(self.pseudos)} pseudos, charge {self.total_charge:+.4g}>")  # fmt: skip


def _types(value, where: str) -> tuple[str, str]:
    if (not isinstance(value, list) or not 1 <= len(value) <= 2
            or not all(isinstance(v, str) for v in value)):  # fmt: skip
        raise ViparrError(f"{where}: type must be of the form [type] or [btype, nbtype]")
    btype, nbtype = value[0], value[-1]
    for t in (btype, nbtype):
        if not t:
            raise ViparrError(f"{where}: type has 0 length")
        if any(c.isspace() for c in t):
            raise ViparrError(f"{where}: type '{t}' contains embedded white space")
    return btype, nbtype


def _template(name: str, js: dict) -> Template:
    names, anum, charge, btype, nbtype, pset = [], [], [], [], [], []
    bonds: list[tuple[int, int]] = []
    index: dict[str, int] = {}

    def add(nm, z, q, bt="", nbt="", ps=""):
        names.append(nm)
        anum.append(z)
        charge.append(q)
        btype.append(bt)
        nbtype.append(nbt)
        pset.append(ps)
        return len(names) - 1

    atoms = js.get("atoms")
    if not atoms:
        raise ViparrError("No atoms in template")
    for i, a in enumerate(atoms):
        if (not isinstance(a, list) or len(a) not in (4, 5) or not isinstance(a[0], str)
                or not isinstance(a[1], int) or not isinstance(a[2], int | float)
                or not isinstance(a[3], list)):  # fmt: skip
            raise ViparrError(f"Atom {i} must be of the form "
                              "[name, atomic_number, charge, [type(s)] (, memo)]")  # fmt: skip
        if a[0] in index:
            raise ViparrError(f"Duplicate atom name '{a[0]}'")
        index[a[0]] = add(a[0], a[1], float(a[2]), *_types(a[3], f"Atom {i}"))
    for bi, pair in enumerate(js.get("bonds") or []):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(x, str) for x in pair)
        ):
            raise ViparrError(f"Bond {bi} must be of the form [atom_name_1, atom_name_2]")
        i, j = index.get(pair[0]), index.get(pair[1])
        if i is not None and j is not None:
            bonds.append((i, j))
        elif i is not None or j is not None:  # the other atom is external
            inner, ext = (i, pair[1]) if i is not None else (j, pair[0])
            index[ext] = add(ext, -1, 0.0)
            bonds.append((inner, index[ext]))
    _check_connected(name, names, bonds)
    tuples = {}
    for kind, size in (("exclusions", 2), ("impropers", 4), ("cmap", 8)):
        tuples[kind] = []
        for i, arr in enumerate(js.get(kind) or []):
            if not isinstance(arr, list) or not all(isinstance(x, str) for x in arr):
                raise ViparrError(f"Item {i} of {kind} list must be of the form "
                                  "[atom_name_1,...,atom_name_n]")  # fmt: skip
            if len(arr) != size:
                raise ViparrError(f"Item {i} of {kind} list must have exactly {size} atoms")
            missing = [x for x in arr if x not in index]
            if missing:
                raise ViparrError(f"Item {i} of {kind} list includes atom '{missing[0]}' "
                                  "not in template")  # fmt: skip
            tuples[kind].append(tuple(index[x] for x in arr))
    pseudos = []
    for i, arr in enumerate(js.get("pseudos") or []):
        if (not isinstance(arr, list) or len(arr) < 6 or not isinstance(arr[0], str)
                or not isinstance(arr[1], int | float) or not isinstance(arr[2], list)
                or not isinstance(arr[3], str) or not isinstance(arr[-1], str)):  # fmt: skip
            raise ViparrError(f"Item {i} of pseudos list must be of the form "
                              "[name,charge,[types],field,site_1,...,site_n,pset]")  # fmt: skip
        pid = add(arr[0], 0, float(arr[1]), *_types(arr[2], f"Pseudo {i}"), arr[-1])
        sites = []
        for s in arr[4:-1]:
            if s not in index:
                raise ViparrError(f"Virtual site {s} of pseudo {i} is not in template")
            sites.append(index[s])
        pseudos.append((arr[3], (pid, *sites)))
        bonds.append((pid, sites[0]))
        if arr[3] == "virtual_midpoint":
            bonds.append((pid, sites[1]))
    from .elements import atomic_number

    pinned = {}
    for ext, element in (js.get("external_elements") or {}).items():
        k = index.get(ext)
        if k is None or anum[k] != -1:
            raise ViparrError(f"external_elements: '{ext}' is not an external atom")
        pinned[k] = element if isinstance(element, int) else atomic_number(str(element))
    formulas = {}
    for ext, formula in (js.get("external_residues") or {}).items():
        k = index.get(ext)
        if k is None or anum[k] != -1:
            raise ViparrError(f"external_residues: '{ext}' is not an external atom")
        formulas[k] = str(formula)
    return Template(name, names, anum, charge, btype, nbtype, pset, bonds, tuples["impropers"],
                    tuples["cmap"], tuples["exclusions"], pseudos, pinned, formulas)  # fmt: skip


def _check_connected(name, names, bonds) -> None:
    parent = list(range(len(names)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in bonds:
        parent[root(i)] = root(j)
    groups: dict[int, list[str]] = {}
    for a, nm in enumerate(names):
        groups.setdefault(root(a), []).append(nm)
    if len(groups) > 1:
        parts = "; ".join(" ".join(g) for g in groups.values())
        raise ViparrError(f"Template {name}: atoms are disconnected: {parts}")


def _read_templates(path: Path) -> list[Template]:
    js = _read_json(path)
    if not isinstance(js, dict):
        raise ViparrError(f"{path}: template file must be of object type")
    out = []
    for name, tpl in js.items():
        try:
            out.append(_template(name, tpl))
        except ViparrError as e:
            raise ViparrError(f"{path}: template {name}: {e}") from None
    return out


# ---- parameter tables ------------------------------------------------------------


@dataclass
class ParamRow:
    """One row of a parameter file: an atom-type pattern and its parameters."""

    type: str
    params: dict
    memo: str = ""


def _read_params(name: str, path: Path) -> list[ParamRow]:
    js = _read_json(path)
    if not isinstance(js, list):
        raise ViparrError("Param file must be of array type")
    if not js:
        return []
    first = js[0]
    if not isinstance(first, dict) or "type" not in first or "params" not in first:
        raise ViparrError("Row 0 must be an object with fields 'type' and 'params'")
    t0 = first["type"]
    npat = len(t0) if isinstance(t0, list) else len(str(t0).split())
    if name == "vdw1" and npat != 1:
        raise ViparrError("vdw1 table must have exactly one type")
    if name == "vdw2" and npat != 2:
        raise ViparrError("vdw2 table must have exactly two types")
    keys = list(first["params"])
    for k in keys:
        if k in ("type", "memo", "nbfix_identifier") or k.startswith("mode_"):
            raise ViparrError(f"Parameter name '{k}' is reserved, please rename")
    rows = []
    for i, row in enumerate(js):
        if not isinstance(row, dict) or "type" not in row or "params" not in row:
            raise ViparrError(f"Row {i} must be an object with fields 'type' and 'params'")
        t = row["type"]
        prefix = f"__mode__{row['mode']} " if row.get("mode") else ""
        if isinstance(t, list):
            if len(t) != npat or not all(isinstance(x, str) for x in t):
                raise ViparrError(f"'type' field of row {i} must be an array of {npat} strings")
            if any(any(c.isspace() for c in x) for x in t):
                raise ViparrError(f"'type' strings in row {i} cannot contain whitespaces")
            tstr = prefix + " ".join(t)
        else:
            if len(str(t).split()) != npat:
                raise ViparrError(f"'type' field of row {i} must have {npat} patterns")
            tstr = prefix + str(t)
        params = {}
        for k, v in row["params"].items():
            if k not in keys:
                raise ViparrError(f"row {i} has parameter {k!r} that row 0 does not")
            if k == "cmapid":
                params[k] = f"cmap{int(v)}"
            elif isinstance(v, bool) or not isinstance(v, int | float | str):
                raise ViparrError(f"Parameter {k} must be of float, int, or string type")
            else:
                params[k] = float(v) if isinstance(v, int | float) else v
        rows.append(ParamRow(tstr, {k: params.get(k, 0.0) for k in keys}, str(row.get("memo", ""))))
    return rows


def _read_cmaps(path: Path) -> list[np.ndarray | None]:
    js = _read_json(path)
    if not isinstance(js, list):
        raise ViparrError(f"'{path}' must be of array type")
    out = []
    for i, tab in enumerate(js):
        if tab is None:
            out.append(None)
            continue
        arr = np.asarray(tab, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ViparrError(f"rows of cmap table {i} must be [phi, psi, energy] triples")
        out.append(arr)
    return out


# ---- force fields ------------------------------------------------------------------


class ViparrForcefield:
    """A viparr force field: rules, templates, parameter tables and CMAP grids.

    ``params[table]`` lists the :class:`ParamRow` of each parameter file in
    file order; ``cmaps[k - 1]`` is the grid that ``cmapid`` ``cmapk`` names,
    an array of (phi, psi, energy) rows.
    """

    def __init__(self, name: str, rules: Rules, templates, params: dict, cmaps=()):
        self.name = name
        self.rules = rules
        self.templates: list[Template] = list(templates)
        self.params: dict[str, list[ParamRow]] = dict(params)
        self.cmaps: list[np.ndarray | None] = list(cmaps)
        self._index: dict | None = None

    def candidates(self, hash_) -> list[Template]:
        if self._index is None:
            self._index = {}
            for t in self.templates:
                self._index.setdefault(t.hash, []).append(t)
        return self._index.get(hash_, [])

    def template(self, name: str) -> Template:
        """The template called ``name`` (the first, if several share it)."""
        for t in self.templates:
            if t.name == name:
                return t
        raise KeyError(f"{self.name} has no template {name!r}")

    def copy(self) -> ViparrForcefield:
        rules = Rules(**{**self.rules.__dict__, "info": list(self.rules.info),
                         "plugins": list(self.rules.plugins)})  # fmt: skip
        params = {k: list(v) for k, v in self.params.items()}
        return ViparrForcefield(self.name, rules, self.templates, params, self.cmaps)

    def __repr__(self) -> str:
        tables = ", ".join(f"{k} {len(v)}" for k, v in self.params.items() if v)
        return (f"<ViparrForcefield {Path(self.name).name}: {len(self.templates)} templates; "
                f"{tables}; plugins {', '.join(self.rules.plugins)}>")  # fmt: skip


def load_forcefield(name, path=None, require_rules: bool = True) -> ViparrForcefield:
    """Read a viparr force field: a directory, a name in ``path`` (default
    ``$VIPARR_FFPATH``), or a name in the viparr-ffpublic copy bundled with
    boonza (see :func:`bundled_version`).

    ``require_rules=False`` reads a patch that has no ``rules`` file (for
    :func:`merge_forcefields`).
    """
    d = find_forcefield(name, path)
    if (d / "rules").is_file():
        rules = _read_rules(d / "rules")
    elif require_rules:
        raise ViparrError(f"{d}: force field directory is missing rules file")
    else:
        rules = Rules(es_scale=[], lj_scale=[])
    templates, params, cmaps = [], {}, []
    for p in sorted(d.iterdir(), key=lambda q: q.name):
        n = p.name
        if n.endswith("~") or n.startswith(".") or p.is_dir() or n in ("README", "rules"):
            continue
        if n.endswith(".def"):
            continue
        if n.startswith("templates"):
            templates += _read_templates(p)
        elif n == "cmap":
            cmaps = _read_cmaps(p)
        else:
            if n == "vdw2" and not rules.nbfix_identifier:
                raise ViparrError(f"{p}: cannot have vdw2 table without nbfix_identifier in rules")
            try:
                params[n] = _read_params(n, p)
            except ViparrError as e:
                warnings.warn(f"failed to load {p}: {e}; continuing without it", ViparrWarning,
                              stacklevel=2)  # fmt: skip
    return ViparrForcefield(str(d).rstrip("/"), rules, templates, params, cmaps)


def merge_forcefields(base, patch, append_only: bool = False, path=None) -> ViparrForcefield:
    """``base`` with ``patch`` merged in, as viparr's ``-m`` (or ``-a`` with
    ``append_only``) options do.

    Templates of the patch replace templates of the same name; parameter rows
    of the patch replace the rows of ``base`` with the same types, and rows
    with new types go first; CMAP grids are replaced. Rules merge (plugins
    and info are joined; functional forms and scale factors must agree).
    ``append_only`` refuses to replace anything. ``base`` and ``patch`` may
    be force fields or names in ``path``; neither is modified.
    """
    src = base.copy() if isinstance(base, ViparrForcefield) else load_forcefield(base, path)
    if not isinstance(patch, ViparrForcefield):
        patch = load_forcefield(patch, path, require_rules=False)
    _merge_rules(src.rules, patch.rules)
    by_name: dict[str, list[Template]] = {}
    for t in patch.templates:
        by_name.setdefault(t.name, []).append(t)
    for name in sorted(by_name):
        if any(t.name == name for t in src.templates):
            if append_only:
                raise ViparrError(
                    f"Merge error: Cannot overwrite template {name} in append-only mode"
                )
            src.templates = [t for t in src.templates if t.name != name]
        src.templates += by_name[name]
    for table in sorted(k for k, v in patch.params.items() if v):
        src.params[table] = _merge_params(
            src.params.get(table, []), patch.params[table], append_only
        )
    if patch.cmaps:
        same = len(patch.cmaps) == len(src.cmaps) and all(
            (a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b))
            for a, b in zip(patch.cmaps, src.cmaps, strict=True))  # fmt: skip
        if not same:
            if append_only and src.cmaps:
                raise ViparrError("Merge error: cannot replace cmap tables in append-only mode")
            src.cmaps = list(patch.cmaps)
    src.name += ("__-a_" if append_only else "__-m_") + patch.name
    src._index = None
    return src


def _merge_rules(src: Rules, patch: Rules) -> None:
    if patch._is_empty():
        return
    src.vdw_func = _merge_vdw(src.vdw_func, patch.vdw_func)
    src.vdw_comb_rule = _merge_vdw(src.vdw_comb_rule, patch.vdw_comb_rule)
    if patch.exclusions != 1:
        if src.exclusions != patch.exclusions:
            raise ViparrError("Cannot merge rules: Exclusion rules do not match")
        if src.es_scale != patch.es_scale:
            raise ViparrError("Cannot merge rules: ES scales do not match")
        if src.lj_scale != patch.lj_scale:
            raise ViparrError("Cannot merge rules: LJ scales do not match")
    if not patch.fatal:
        src.fatal = False
    if not src.nbfix_identifier:
        src.nbfix_identifier = patch.nbfix_identifier
    elif patch.nbfix_identifier and patch.nbfix_identifier != src.nbfix_identifier:
        raise ViparrError("Cannot merge rules: NBFix identifiers do not match")
    src.info += [x for x in patch.info if x not in src.info]
    src.plugins += [x for x in patch.plugins if x not in src.plugins]


def _merge_params(src: list[ParamRow], patch: list[ParamRow], append_only: bool) -> list[ParamRow]:
    src_types: dict[str, list[ParamRow]] = {}
    for r in src:
        src_types.setdefault(r.type, []).append(r)
    patch_types: dict[str, list[ParamRow]] = {}
    out = []
    for r in patch:
        patch_types.setdefault(r.type, []).append(r)
        if r.type not in src_types:
            out.append(r)
    done = set()
    for r in src:
        if r.type not in patch_types:
            out.append(r)
            continue
        if r.type in done:
            continue
        mine, theirs = src_types[r.type], patch_types[r.type]
        same = len(mine) == len(theirs) and all(
            any(p.params == q.params for q in mine) for p in theirs
        )
        if not same and append_only:
            raise ViparrError(f"Merge error: Cannot overwrite type {r.type} in append_only mode")
        out += theirs
        done.add(r.type)
    return out


def write_forcefield(ff: ViparrForcefield, directory) -> Path:
    """Write ``ff`` as a viparr force-field directory: ``rules`` (left out
    for a patch without rules), ``templates`` and one file per parameter
    table, which :func:`load_forcefield` and viparr read back."""
    if ff.cmaps:
        raise ViparrError("writing CMAP grids is not supported")
    names = [t.name for t in ff.templates]
    if len(set(names)) != len(names):
        raise ViparrError("cannot write several templates with the same name")
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    r = ff.rules
    if not r._is_empty():
        rules = {"info": r.info, "vdw_func": r.vdw_func, "vdw_comb_rule": r.vdw_comb_rule,
                 "exclusions": r.exclusions, "es_scale": r.es_scale, "lj_scale": r.lj_scale,
                 "plugins": [["mass", 1] if p == "mass2" else p for p in r.plugins],
                 "fatal": r.fatal}  # fmt: skip
        if r.nbfix_identifier:
            rules["nbfix_identifier"] = r.nbfix_identifier
        _write_json(d / "rules", rules)
    if ff.templates:
        _write_json(d / "templates", {t.name: _template_json(t) for t in ff.templates})
    for table, rows in ff.params.items():
        if rows:
            _write_json(d / table, [_row_json(row) for row in rows])
    return d


def patch_from_system(system: System, name: str, tag: str, rules: Rules | None = None):
    """A viparr patch that gives one molecule the force field it already has.

    ``system`` is one parameterized molecule. The patch has one template
    (every atom, as one residue ``name``), a type per atom
    (``<element><n>~<tag>``) and a parameter row per term, from the
    ``stretch_harm``, ``angle_harm``, ``dihedral_trig`` (propers and
    impropers, told apart by the bonds) and ``improper_harm`` tables and a
    12-6 Lennard-Jones ``nonbonded`` table. Exclusions and 1-4 pairs are
    made by the rules of the force field the patch joins: given ``rules``,
    the molecule's own 1-4 pairs must be the ones they make, or this raises.
    """
    s = system
    anum = s.atoms["anum"].tolist()
    if any(z < 1 for z in anum):
        raise ViparrError("virtual sites are not supported")
    if s.nfragments != 1:
        raise ViparrError(f"{name}: {s.nfragments} molecules; give one")
    if "nonbonded" not in s.table_names:
        raise ViparrError(f"{name}: no nonbonded table, so no force field to take")
    funct, rule = s.nonbonded_info.vdw_funct or "vdw_12_6", s.nonbonded_info.vdw_rule
    if funct != "vdw_12_6" or (rule or "arithmetic/geometric") != "arithmetic/geometric":
        raise ViparrError(f"{name}: {funct} {rule} Lennard-Jones; only 12-6 with "
                          "arithmetic/geometric combining joins Amber force fields")  # fmt: skip
    known = {"stretch_harm", "angle_harm", "dihedral_trig", "improper_harm", "nonbonded",
             "pair_12_6_es", "exclusion"}  # fmt: skip
    other = [t for t in s.table_names if t not in known and not t.startswith("constraint_")]
    if other:
        raise ViparrError(f"{name}: tables a template cannot carry: {', '.join(sorted(other))}")
    count: dict[str, int] = {}
    types, names = [], []
    for z in anum:
        sym = symbol(z)
        count[sym] = count.get(sym, 0) + 1
        types.append(f"{sym}{count[sym]}~{tag}")
        names.append(f"{sym}{count[sym]}")
    bi, bj = s.bonds["i"].tolist(), s.bonds["j"].tolist()
    bonds = sorted({(min(i, j), max(i, j)) for i, j in zip(bi, bj, strict=True)})
    bonded = set(bonds) | {(j, i) for i, j in bonds}
    params: dict[str, list[ParamRow]] = {}
    memo = f"from {name}"

    def rows_of(table):
        if table not in s.table_names:
            return
        t = s.table(table)
        cols = [c for c in t.params.props if t.params.prop_type(c) != "str"]
        for atoms, pid in zip(t.atoms.tolist(), t.param_ids.tolist(), strict=True):
            yield tuple(atoms), {c: float(t.params[c][pid]) for c in cols}

    def add(table, atoms, row):
        params.setdefault(table, []).append(ParamRow(" ".join(types[a] for a in atoms), row, memo))

    for atoms, row in rows_of("stretch_harm"):
        if atoms not in bonded:
            raise ViparrError(f"{name}: a stretch term between atoms that are not bonded "
                              "(Urey-Bradley) cannot be carried")  # fmt: skip
        add("stretch_harm", atoms, row)
    for atoms, row in rows_of("angle_harm"):
        add("angle_harm", atoms, row)
    propers: dict[tuple, list[dict]] = {}
    impropers: list[tuple[int, ...]] = []
    seen_improper: set[frozenset] = set()
    for atoms, row in rows_of("dihedral_trig"):
        a, b, c, d = atoms
        if (a, b) in bonded and (b, c) in bonded and (c, d) in bonded:
            key = atoms if atoms <= atoms[::-1] else atoms[::-1]
            propers.setdefault(key, []).append(row)
        else:
            if frozenset(atoms) in seen_improper:
                raise ViparrError(f"{name}: an improper with several terms cannot be carried")
            seen_improper.add(frozenset(atoms))
            add("improper_trig", atoms, row)
            impropers.append(atoms)
    for atoms, row in rows_of("improper_harm"):
        add("improper_harm", atoms, row)
        impropers.append(atoms)
    zero = None
    if propers:
        zero = dict.fromkeys(next(iter(propers.values()))[0], 0.0)
    nbrs: list[list[int]] = [[] for _ in anum]
    for i, j in bonds:
        nbrs[i].append(j)
        nbrs[j].append(i)
    for b, c in bonds:  # every proper dihedral needs a row, zero where the molecule has none
        for a in nbrs[b]:
            for d in nbrs[c]:
                if a not in (b, c) and d not in (b, c) and a != d:
                    key = min((a, b, c, d), (d, c, b, a))
                    if key not in propers:
                        propers[key] = [dict(zero or {"phi0": 0.0, "fc0": 0.0})]
    for key in sorted(propers):  # the terms of one dihedral stay together
        for row in propers[key]:
            add("dihedral_trig", key, row)
    nb = s.table("nonbonded")
    sigma, epsilon = nb.params["sigma"], nb.params["epsilon"]
    per_atom = dict(zip((t[0] for t in nb.atoms.tolist()), nb.param_ids.tolist(), strict=True))
    mass, charge = s.atoms["mass"].tolist(), s.atoms["charge"].tolist()
    for a in range(len(anum)):
        p = per_atom[a]
        add("vdw1", (a,), {"sigma": float(sigma[p]), "epsilon": float(epsilon[p])})
        add("mass", (a,), {"amu": float(mass[a])})
    if rules is not None and len(rules.es_scale) >= 3:
        _check_pairs(s, name, nbrs, per_atom, sigma, epsilon, charge, rules)
    tpl = Template(name, names, anum, [float(q) for q in charge], list(types), list(types),
                   [""] * len(anum), bonds, impropers)  # fmt: skip
    return ViparrForcefield(tag, Rules(es_scale=[], lj_scale=[]), [tpl], params)


def _check_pairs(s, name, nbrs, per_atom, sigma, epsilon, charge, rules: Rules) -> None:
    """The molecule's 1-4 pairs must be the ones ``rules`` make from its atoms."""
    es, lj = rules.es_scale[2], rules.lj_scale[2]
    n = len(nbrs)
    want = {}
    for a in range(n):  # atoms three bonds away and no closer
        dist = {a: 0}
        frontier = [a]
        for step in (1, 2, 3):
            nxt = []
            for x in frontier:
                for y in nbrs[x]:
                    if y not in dist:
                        dist[y] = step
                        nxt.append(y)
            frontier = nxt
        for b, d in dist.items():
            if d == 3 and a < b:
                sa, sb = sigma[per_atom[a]], sigma[per_atom[b]]
                ea, eb = epsilon[per_atom[a]], epsilon[per_atom[b]]
                sij, eij = 0.5 * (sa + sb), math.sqrt(ea * eb)
                want[(a, b)] = (lj * 4 * eij * sij**12, lj * 4 * eij * sij**6,
                                es * charge[a] * charge[b])  # fmt: skip
    have: dict[tuple[int, int], list[float]] = {}
    if "pair_12_6_es" in s.table_names:
        t = s.table("pair_12_6_es")
        cols = [t.params[c] for c in ("aij", "bij", "qij")]
        for (a, b), p in zip(t.atoms.tolist(), t.param_ids.tolist(), strict=True):
            got = have.setdefault((min(a, b), max(a, b)), [0.0, 0.0, 0.0])
            for k in range(3):
                got[k] += float(cols[k][p])
    for key in set(want) | set(have):
        w, h = want.get(key, (0.0, 0.0, 0.0)), have.get(key, [0.0, 0.0, 0.0])
        if any(abs(x - y) > 2e-3 * max(abs(x), abs(y)) + 1e-6 for x, y in zip(w, h, strict=True)):
            raise ViparrError(
                f"{name}: its 1-4 pairs are not those the force field it joins makes "
                f"(1-4 scales {es:g} electrostatic, {lj:g} Lennard-Jones)"
            )


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1) + "\n")


def _template_json(t: Template) -> dict:
    def types(i):
        return [t.btype[i]] if t.btype[i] == t.nbtype[i] else [t.btype[i], t.nbtype[i]]

    js: dict = {
        "atoms": [[t.names[i], t.anum[i], t.charge[i], types(i)]
                  for i in range(len(t.names)) if t.anum[i] > 0],
        "bonds": [[t.names[i], t.names[j]] for i, j in t.bonds
                  if t.anum[i] != 0 and t.anum[j] != 0],
    }  # fmt: skip
    for key, tuples in (("impropers", t.impropers), ("cmap", t.cmaps),
                        ("exclusions", t.exclusions)):  # fmt: skip
        if tuples:
            js[key] = [[t.names[i] for i in tup] for tup in tuples]
    if t.pseudos:
        js["pseudos"] = [[t.names[pid], t.charge[pid], types(pid), kind,
                          *(t.names[s] for s in sites), t.pset[pid]]
                         for kind, (pid, *sites) in t.pseudos]  # fmt: skip
    if t.external_anum:
        js["external_elements"] = {t.names[k]: symbol(z) for k, z in t.external_anum.items()}
    if t.external_formula:
        js["external_residues"] = {t.names[k]: f for k, f in t.external_formula.items()}
    return js


def _row_json(row: ParamRow) -> dict:
    t, mode = row.type, ""
    if t.startswith("__mode__"):
        mode, t = t[len("__mode__") :].split(" ", 1)
    params = {k: int(v[4:]) if k == "cmapid" else v for k, v in row.params.items()}
    out: dict = {"type": t.split(), "params": params}
    if mode:
        out["mode"] = mode
    if row.memo:
        out["memo"] = row.memo
    return out


# ---- parameter matching --------------------------------------------------------------


def _glob(pattern: str, wild: str):
    if wild not in pattern:
        return pattern.__eq__
    rx = re.compile("".join(".*" if c == wild else re.escape(c) for c in pattern) + r"\Z", re.S)
    return lambda s: rx.match(s) is not None


def _identity(atoms, bonds):
    return atoms, bonds


def _reverse(atoms, bonds):
    return atoms[::-1], bonds[::-1]


_BOTH = (_identity, _reverse)
_NOT_FOUND = object()


def _all(tests, values) -> bool:
    return len(tests) == len(values) and all(f(v) for f, v in zip(tests, values, strict=True))


class _Matcher:
    """viparr's ParameterMatcher for one table of one force field."""

    def __init__(self, table: str, rows: list[ParamRow], perms=_BOTH, pseudo: bool = False):
        self.table, self.rows, self.perms = table, rows, perms
        self.cache: dict = {}
        compiled, self.exact, self.wild = [], [], []
        match_bonds = None
        for i, r in enumerate(rows):
            tokens = r.type.split()
            flags = ()
            if pseudo:
                flags, tokens = (tokens[-1],), tokens[:-1]
            atoms = [t for t in tokens if t not in _BOND_STRINGS]
            bonds = [t for t in tokens if t in _BOND_STRINGS]
            if match_bonds is None:
                match_bonds = bool(bonds)
            elif match_bonds != bool(bonds):
                raise ViparrError(f"{table}: either all rows or no rows can match bond types")
            compiled.append(([_glob(a, "*") for a in atoms], [_glob(b, "~") for b in bonds], flags))
            (self.wild if any("*" in a for a in atoms) else self.exact).append(i)
        self.compiled = compiled
        self.match_bonds = bool(match_bonds)
        # exact rows can only match their own atom types, forwards or backwards
        self.by_types: dict[tuple, list[int]] = {}
        for i in self.exact:
            key = tuple(r for r in rows[i].type.split()[: len(compiled[i][0]) + len(compiled[i][1])]
                        if r not in _BOND_STRINGS)  # fmt: skip
            self.by_types.setdefault(key, []).append(i)

    def _fits(self, pattern, i: int) -> bool:
        atom_f, bond_f, flags = self.compiled[i]
        if pattern[2] != flags:
            return False
        for perm in self.perms:
            atoms, bonds = perm(pattern[0], pattern[1])
            if _all(atom_f, atoms) and (not bond_f or _all(bond_f, bonds)):
                return True
        return False

    def match(self, pattern, allow_repeat: bool = False):
        """Index of the matching row for a pattern (atom types, bond strings,
        flags), or None."""
        hit = self.cache.get(pattern, _NOT_FOUND)
        if hit is not _NOT_FOUND:
            return hit
        found = None
        candidates = set()
        for perm in self.perms:
            candidates.update(self.by_types.get(tuple(perm(pattern[0], pattern[1])[0]), ()))
        for i in sorted(candidates):
            if self._fits(pattern, i):
                if found is None:
                    found = i
                elif not allow_repeat:
                    raise ViparrError(f"For {self.table}: Found two matches for {_show(pattern)} "
                                      f"(at the same priority): {self.rows[i].type} and "
                                      f"{self.rows[found].type}")  # fmt: skip
        if found is None:
            found = next((i for i in self.wild if self._fits(pattern, i)), None)
        self.cache[pattern] = found
        return found

    def same_type_run(self, i: int) -> list[int]:
        """Row ``i`` and the rows right after it with the same types."""
        out = [i]
        while out[-1] + 1 < len(self.rows) and self.rows[out[-1] + 1].type == self.rows[i].type:
            out.append(out[-1] + 1)
        return out


def _show(pattern, with_bonds: bool = True) -> str:
    return "(" + ", ".join([*pattern[0], *(pattern[1] if with_bonds else ()), *pattern[2]]) + ")"


# ---- parameterization -------------------------------------------------------------------


@dataclass
class _Typed:
    """What viparr's TemplatedSystem holds for one force field."""

    atoms: list = field(default_factory=list)
    bonds: list = field(default_factory=list)
    angles: list = field(default_factory=list)
    dihedrals: list = field(default_factory=list)
    exclusions: list = field(default_factory=list)
    impropers: list = field(default_factory=list)
    cmaps: list = field(default_factory=list)
    pseudos: dict = field(default_factory=dict)  # kind -> [(pseudo, site_1, ...)]


class _Terms:
    """Terms of one output table, with parameter rows in order of first use."""

    def __init__(self, natoms: int, category: str = "bond"):
        self.natoms, self.category = natoms, category
        self.atoms: list[tuple] = []
        self.keys: list = []
        self.params: dict = {}

    def add(self, atoms, key, row: dict) -> None:
        if key not in self.params:
            self.params[key] = row
        self.atoms.append(tuple(atoms))
        self.keys.append(key)


def _row(r: ParamRow, **extra) -> dict:
    return {"type": r.type, **r.params, **extra, "memo": r.memo}


def _mapped(t, tmap, ambiguous, tpl, what) -> tuple:
    """A template tuple in system atoms."""
    if any(x in ambiguous for x in t):
        raise ViparrError(f"{what} in template {tpl.name} references an ambiguous "
                          "externally bonded atom")  # fmt: skip
    return tuple(tmap[x] for x in t)


def _externals_fit(t: Template, perm, atoms, anum, nbrs, formula_of) -> bool:
    """Whether the atoms bonded to a match's pinned external atoms have their
    elements and belong to residues with their formulas."""
    if not t.external_anum and not t.external_formula:
        return True
    to_system = dict(perm)
    inside = set(atoms)
    need: dict[int, list[tuple]] = {}
    for ext in sorted(set(t.external_anum) | set(t.external_formula)):
        want = (t.external_anum.get(ext), t.external_formula.get(ext))
        for ti in t._nbrs[ext]:
            need.setdefault(ti, []).append(want)
    for ti, wants in need.items():
        have = [o for o in nbrs[to_system[ti]] if o not in inside and anum[o] != 0]
        if not _take_externals(wants, have, anum, formula_of):
            return False
    return True


def _npins(t: Template) -> int:
    return len(t.external_anum) + len(t.external_formula)


def _take_externals(wants, have, anum, formula_of) -> bool:
    """Whether each wanted (element, formula) gets a different atom of ``have``."""
    if not wants:
        return True
    (z, f), rest = wants[0], wants[1:]
    for k, o in enumerate(have):
        if (z is None or anum[o] == z) and (f is None or formula_of(o) == f):
            if _take_externals(rest, have[:k] + have[k + 1 :], anum, formula_of):
                return True
    return False


class _Parameterizer:
    def __init__(self, system: System, ffs: list[ViparrForcefield], rename_atoms: bool,
                 rename_residues: bool, fatal: bool):  # fmt: skip
        s = system.clone(structure_only=True)
        s.nonbonded_info = NonbondedInfo()  # the force fields decide
        order = s.bonds["order"]
        if (order == 0).any():
            s.bonds["order"] = np.where(order == 0, 1, order)
        self.s, self.ffs = s, ffs
        self.rename_atoms, self.rename_residues, self.fatal = rename_atoms, rename_residues, fatal
        n = s.natoms
        self.n0 = n
        self.anum = s.atoms["anum"].tolist()
        self.residue = s.atoms["residue"].tolist()
        self.names = s.atoms["name"].tolist()
        self.resnames = s.residues["name"].tolist()
        self.charge = s.atoms["charge"].tolist()
        self.pos = s.positions.copy()
        self.nbrs: list[list[int]] = [[] for _ in range(n)]
        self.order: dict[tuple[int, int], int] = {}
        bi, bj, bo = s.bonds["i"].tolist(), s.bonds["j"].tolist(), s.bonds["order"].tolist()
        for i, j, o in zip(bi, bj, bo, strict=True):
            self.nbrs[i].append(j)
            self.nbrs[j].append(i)
            self.order[(min(i, j), max(i, j))] = o
        self.btype, self.nbtype, self.pset = [""] * n, [""] * n, [""] * n
        self.pseudos: list[tuple] = []  # (name, residue, parent, charge)
        self.new_bonds: list[tuple[int, int]] = []
        self.tables: dict[str, _Terms] = {}
        self.cmap_terms: list[tuple] = []  # (atoms, ff index, row)
        self.ub_terms: list[tuple] = []
        self.improper_trig: list[tuple] = []
        self.nonbonded: dict[int, tuple[int, int]] = {}
        self.vdw14: dict[int, tuple[int, int]] | None = None
        self.vdw2: list[tuple[int, ParamRow]] = []
        self.mass: dict[int, float] = {}
        self.exclusion: dict[tuple[int, int], tuple] = {}
        self.exclusion_params: dict[tuple, tuple[int, float, float]] = {}
        self.plugins: set[str] = set()
        self.match_cache: dict = {}
        self._formulas: dict[int, str] = {}
        self._res_atoms: dict[int, list[int]] | None = None
        self._pins_formula: dict[int, bool] = {}

    # -- matching residues to templates ----------------------------------------
    def _formula_of(self, atom: int) -> str:
        """The chemical formula of the residue of ``atom``."""
        r = self.residue[atom]
        f = self._formulas.get(r)
        if f is None:
            if self._res_atoms is None:
                self._res_atoms = {}
                for a, rr in enumerate(self.residue):
                    self._res_atoms.setdefault(rr, []).append(a)
            f = self._formulas[r] = _formula(self.anum[a] for a in self._res_atoms[r])
        return f

    def _residue_key(self, atoms, formulas: bool = False):
        local = {a: k for k, a in enumerate(atoms)}
        # atoms of other residues are keyed by what templates may pin: the
        # element, and the residue's formula where a template pins that
        if formulas:
            return tuple((self.anum[a], tuple(local[o] if o in local
                                              else (self.anum[o], self._formula_of(o))
                                              for o in self.nbrs[a]))
                         for a in atoms)  # fmt: skip
        return tuple((self.anum[a], tuple(local.get(o, -1 - self.anum[o]) for o in self.nbrs[a]))
                     for a in atoms)  # fmt: skip

    def _find(self, ff: ViparrForcefield, ffi: int, atoms: list[int]):
        """(template, [(template atom, system atom)]) or (None, reason)."""
        formulas = self._pins_formula.get(ffi)
        if formulas is None:
            formulas = self._pins_formula[ffi] = any(t.external_formula for t in ff.templates)
        key = (ffi, self._residue_key(atoms, formulas))
        hit = self.match_cache.get(key)
        if hit is None:
            hit = self.match_cache[key] = self._find_uncached(ff, atoms)
        tpl, local = hit
        if tpl is None:
            return None, local
        return tpl, [(ti, atoms[k]) for ti, k in local]

    def _find_uncached(self, ff, atoms):
        res = self.residue[atoms[0]]
        candidates = ff.candidates(_formula_hash(atoms, self.anum, self.nbrs))
        if not candidates:
            return None, ("formula", _formula(self.anum[a] for a in atoms))
        target = _Graph(atoms, self.anum, self.nbrs)
        found = []
        for t in candidates:
            perm = t.graph.match(target)
            if perm is not None and _externals_fit(t, perm, atoms, self.anum, self.nbrs,
                                                   self._formula_of):  # fmt: skip
                found.append((t, perm))
        if len(found) > 1:  # templates that pin their external atoms win (viparr: an error)
            most = max(_npins(t) for t, _ in found)
            found = [f for f in found if _npins(f[0]) == most]
        if len(found) > 1:
            raise ViparrError(
                f"Multiple templates {found[1][0].name} and {found[0][0].name} from a single "
                f"forcefield ({ff.name}) match residue {res} ({self.resnames[res]})"
            )
        if not found:
            return None, ("topology", [t.name for t in candidates])
        local = {a: k for k, a in enumerate(atoms)}
        return found[0][0], [(ti, local[a]) for ti, a in found[0][1]]

    def _why_not(self, ff, res, reason) -> str:
        kind, info = reason
        head = f"Forcefield {ff.name} "
        if kind == "formula":
            msg = (f"{head}has no template with matching formula for residue {res} "
                   f"({self.resnames[res]}, {info})")  # fmt: skip
            if any(t.name == self.resnames[res] for t in ff.templates):
                msg += (f"\n\ta template with name {self.resnames[res]} was found but has a "
                        "different chemical formula and/or terminal locations")  # fmt: skip
            return msg + "."
        return (f"{head}has no template with matching topology for residue {res} "
                f"({self.resnames[res]}), but templates found with matching formula and different "
                f"bond topology: {' '.join(info)}.")  # fmt: skip

    def _match_fragment(self, ff, ffi, frag):
        residues: dict[int, list[int]] = {}
        for a in frag:
            residues.setdefault(self.residue[a], []).append(a)
        matches = []
        for res in sorted(residues):
            tpl, pairs = self._find(ff, ffi, residues[res])
            if tpl is None:
                return None, self._why_not(ff, res, pairs)
            matches.append((tpl, pairs))
        return matches, None

    # -- assigning a matched fragment ------------------------------------------
    def _assign(self, typed: _Typed, matches) -> None:
        assigned = []
        for tpl, pairs in matches:
            tmap = [-1] * len(tpl.names)
            for ti, a in pairs:
                tmap[ti] = a
            for ti, a in enumerate(tmap):
                if a < 0:
                    continue
                self.btype[a], self.nbtype[a] = tpl.btype[ti], tpl.nbtype[ti]
                self.charge[a] = tpl.charge[ti]
                if self.rename_atoms:
                    self.names[a] = tpl.names[ti]
                if self.rename_residues:
                    self.resnames[self.residue[a]] = tpl.name
                typed.atoms.append(a)
                assigned.append(a)
            ambiguous = set()
            for i, j in tpl.bonds:
                if tpl.anum[i] == 0 or tpl.anum[j] == 0:
                    continue
                if tpl.anum[i] > 0 and tpl.anum[j] > 0:
                    if (min(tmap[i], tmap[j]), max(tmap[i], tmap[j])) not in self.order:
                        raise ViparrError("Incorrect match; system is missing bond")
                    continue
                t_in, t_ex = (i, j) if tpl.anum[j] == -1 else (j, i)
                s_in, s_ex = tmap[t_in], -1
                mapped = set(tmap)
                for o in self.nbrs[s_in]:
                    if o in mapped:
                        continue
                    if s_ex != -1:
                        ambiguous.add(t_ex)
                    s_ex = o
                if s_ex == -1:
                    raise ViparrError("Incorrect match; system is missing bond")
                tmap[t_ex] = s_ex

            typed.exclusions += [
                _mapped(t, tmap, ambiguous, tpl, "Exclusion") for t in tpl.exclusions
            ]
            typed.impropers += [_mapped(t, tmap, ambiguous, tpl, "Improper") for t in tpl.impropers]
            typed.cmaps += [_mapped(t, tmap, ambiguous, tpl, "Cmap") for t in tpl.cmaps]
            kinds = list(dict.fromkeys(k for k, _ in tpl.pseudos))
            kinds = [k for k in kinds if not k.startswith("drude")] + [
                k for k in kinds if k.startswith("drude")
            ]
            for kind in kinds:
                for k2, sites in tpl.pseudos:
                    if k2 != kind:
                        continue
                    tid = sites[0]
                    site_atoms = _mapped(sites[1:], tmap, ambiguous, tpl, "Virtual site definition")
                    if any(x < 0 for x in site_atoms):
                        raise ViparrError("pseudo has a site atom not yet processed")
                    pid = self._add_pseudo(tpl, tid, site_atoms[0])
                    tmap[tid] = pid
                    typed.pseudos.setdefault(kind, []).append((pid, *site_atoms))
                    typed.atoms.append(pid)
                    assigned.append(pid)
                    for o in tpl._nbrs[tid]:
                        if tmap[o] < 0:
                            if tpl.anum[o] != 0:
                                raise ViparrError("pseudo bonded to an atom not yet processed")
                            continue
                        self._add_bond(pid, tmap[o])
        self._tuples(typed, assigned)

    def _add_pseudo(self, tpl: Template, tid: int, parent: int) -> int:
        pid = len(self.anum)
        self.anum.append(0)
        self.residue.append(self.residue[parent])
        self.names.append(tpl.names[tid])
        self.charge.append(tpl.charge[tid])
        self.btype.append(tpl.btype[tid])
        self.nbtype.append(tpl.nbtype[tid])
        self.pset.append(tpl.pset[tid])
        self.nbrs.append([])
        self.pseudos.append(parent)
        return pid

    def _add_bond(self, i: int, j: int) -> None:
        self.nbrs[i].append(j)
        self.nbrs[j].append(i)
        self.order[(min(i, j), max(i, j))] = 1
        self.new_bonds.append((i, j))

    def _tuples(self, typed: _Typed, atoms) -> None:
        """viparr's GetBondsAnglesDihedrals for the atoms of one fragment."""
        inside = set(atoms)
        anum, nbrs = self.anum, self.nbrs
        for ai in atoms:
            if anum[ai] == 0:
                continue
            ib = nbrs[ai]
            if any(o not in inside for o in ib):
                raise ViparrError("Cannot get tuples: incomplete fragment")
            for aj in ib:
                if anum[aj] == 0:
                    continue
                typed.angles += [(aj, ai, ak) for ak in ib if anum[ak] != 0 and aj < ak]
                if ai > aj:
                    continue
                typed.bonds.append((ai, aj))
                for ah in ib:
                    if anum[ah] == 0 or ah == aj:
                        continue
                    for ak in nbrs[aj]:
                        if anum[ak] == 0 or ak == ai or ak == ah:
                            continue
                        typed.dihedrals.append((ah, ai, aj, ak) if ah < ak else (ak, aj, ai, ah))

    # -- patterns --------------------------------------------------------------
    def _bond_string(self, i: int, j: int) -> str:
        o = self.order.get((min(i, j), max(i, j)))
        if o is None:
            raise ViparrError(f"bond between atom {i} and atom {j} does not exist")
        if o not in _BOND_STRING:
            raise ViparrError(f"cannot handle bond order {o} between atoms {i} and {j}")
        return _BOND_STRING[o]

    def _pattern(self, kind: str, t) -> tuple:
        if kind == "btype":
            return (tuple(self.btype[a] for a in t), (), ())
        if kind == "nbtype":
            return (tuple(self.nbtype[a] for a in t), (), ())
        if kind == "bonded":
            bonds = tuple(self._bond_string(t[k - 1], t[k]) for k in range(1, len(t)))
            return (tuple(self.btype[a] for a in t), bonds, ())
        if kind == "pseudo":
            return (tuple(self.btype[a] for a in t[1:]), (), (self.pset[t[0]],))
        if kind == "pseudo_bond_to_second":
            return (tuple(self.btype[a] for a in t[1:]),
                    (self._bond_string(t[1], t[2]), *(self._bond_string(t[2], a) for a in t[3:])),
                    (self.pset[t[0]],))  # fmt: skip
        raise AssertionError(kind)

    def _missing(self, ff, table, kind, t, matcher, warn_only: bool) -> None:
        patt = self._pattern(kind, t)
        msg = (f"No match found for table '{table}', pattern "
               f"{_show(patt, matcher.match_bonds)}, atoms ({','.join(map(str, t))})")  # fmt: skip
        if warn_only or not (ff.rules.fatal and self.fatal):
            warnings.warn(msg, ViparrWarning, stacklevel=4)
        else:
            raise ViparrError(f"{Path(ff.name).name}: {msg}")

    def _nbody(
        self, ff, ffi, table, tuples, kind, perms=_BOTH, required=True, pseudo=False, rows=None
    ):
        """AddNbodyTable: [(tuple, row index)] for the tuples that match."""
        rows = ff.params.get(table, []) if rows is None else rows
        if not rows:
            raise ViparrError(f"{Path(ff.name).name}: must have '{table}' table")
        matcher = _Matcher(table, rows, perms, pseudo)
        out = []
        for t in tuples:
            i = matcher.match(self._pattern(kind, t))
            if i is None:
                if required:
                    self._missing(ff, table, kind, t, matcher, False)
                continue
            out.append((t, i))
        return out, rows

    def _table(self, name: str, natoms: int, category: str = "bond") -> _Terms:
        t = self.tables.get(name)
        if t is None:
            t = self.tables[name] = _Terms(natoms, category)
        return t

    # -- plugins ---------------------------------------------------------------
    def _plugin(self, name, ff, ffi, typed: _Typed) -> None:
        P = ff.params
        if name == "bonds":
            hits, rows = self._nbody(ff, ffi, "stretch_harm", typed.bonds, "bonded")
            tab = self._table("stretch_harm", 2)
            for t, i in hits:
                tab.add(t, (ffi, "stretch_harm", i), _row(rows[i]))
        elif name == "angles":
            hits, rows = self._nbody(ff, ffi, "angle_harm", typed.angles, "bonded")
            tab = self._table("angle_harm", 3)
            for t, i in hits:
                tab.add(t, (ffi, "angle_harm", i), _row(rows[i]))
        elif name in ("propers", "propers_allowmissing"):
            rows = P.get("dihedral_trig", [])
            if not rows:
                raise ViparrError("Must have 'dihedral_trig' table for 'propers' plugin")
            matcher = _Matcher("dihedral_trig", rows)
            tab = self._table("dihedral_trig", 4)
            for t in typed.dihedrals:
                i = matcher.match(self._pattern("bonded", t), allow_repeat=True)
                if i is None:
                    self._missing(ff, "dihedral_trig", "bonded", t, matcher, name != "propers")
                    continue
                for k in matcher.same_type_run(i):
                    tab.add(t, (ffi, "dihedral_trig", k), _row(rows[k]))
        elif name == "impropers":
            if P.get("improper_harm"):
                hits, rows = self._nbody(ff, ffi, "improper_harm", typed.impropers, "btype")
                tab = self._table("improper_harm", 4)
                for t, i in hits:
                    tab.add(t, (ffi, "improper_harm", i), _row(rows[i]))
            if P.get("improper_anharm"):
                raise ViparrError("improper_anharm parameters are not supported")
            if P.get("improper_trig"):
                rows = P["improper_trig"]
                matcher = _Matcher("improper_trig", rows)
                for t in typed.impropers:
                    i = matcher.match(self._pattern("btype", t))
                    if i is None:
                        self._missing(ff, "improper_trig", "btype", t, matcher, True)
                        continue
                    self.improper_trig.append((t, ffi, rows[i]))
        elif name == "cmap":
            hits, rows = self._nbody(ff, ffi, "torsiontorsion_cmap", typed.cmaps, "btype")
            for t, i in hits:
                self.cmap_terms.append((t, ffi, i))
        elif name == "ureybradley":
            rows = P.get("ureybradley_harm", [])
            if not rows:
                raise ViparrError("Must have 'ureybradley_harm' table for 'ureybradley' plugin")
            matcher = _Matcher("ureybradley_harm", rows)
            for t in typed.angles:
                i = matcher.match(self._pattern("bonded", t))
                if i is not None:
                    self.ub_terms.append(((t[0], t[2]), rows[i]))
        elif name == "vdw1":
            hits, rows = self._nbody(
                ff, ffi, "vdw1", [(a,) for a in typed.atoms], "nbtype", (_identity,)
            )
            for (a,), i in hits:
                self.nonbonded[a] = (ffi, i)
        elif name == "vdw2":
            if not P.get("vdw2"):
                raise ViparrError("Must have 'vdw2' table for 'vdw2' plugin")
            self.vdw2 += [(ffi, r) for r in P["vdw2"]]
        elif name in ("mass", "mass2"):
            kind = "btype" if name == "mass" else "nbtype"
            hits, rows = self._nbody(
                ff, ffi, "mass", [(a,) for a in typed.atoms], kind, (_identity,)
            )
            for (a,), i in hits:
                self.mass[a] = float(rows[i].params["amu"])
        elif name in ("virtuals", "virtuals_regular"):
            for kind, sites in typed.pseudos.items():
                if not kind.startswith("virtual_") or kind == "virtual_shift" or not sites:
                    continue
                table = "virtuals_" + kind[8:]
                pk = "pseudo_bond_to_second" if kind == "virtual_fdat3" else "pseudo"
                hits, rows = self._nbody(ff, ffi, table, sites, pk, (_identity,), pseudo=True)
                tab = self._table(kind, len(sites[0]), "virtual")
                for t, i in hits:
                    tab.add(t, (ffi, table, i), _row(rows[i]))
        elif name == "exclusions":
            self._exclusions(ff, ffi, typed)
            if P.get("vdw1_14"):
                hits, rows = self._nbody(ff, ffi, "vdw1_14", [(a,) for a in typed.atoms], "btype",
                                         (_identity,), required=False)  # fmt: skip
                if self.vdw14 is None:
                    self.vdw14 = {}
                for (a,), i in hits:
                    self.vdw14[a] = (ffi, i)
        else:
            raise ViparrError(f"plugin {name!r} (in {Path(ff.name).name}) is not supported")

    def _exclusions(self, ff, ffi, typed: _Typed) -> None:
        rule = ff.rules.exclusions
        if rule > 4:
            raise ViparrError("Maximum supported exclusion rule is 4")
        key = (ffi, -1)
        self.exclusion_params[key] = (-1, 0.0, 0.0)
        for a1, a2 in typed.exclusions:
            self._exclude(a1, a2, key)
        tuples = [[(a,) for a in typed.atoms if self.anum[a] > 0], typed.bonds, typed.angles,
                  typed.dihedrals]  # fmt: skip
        for sep in range(1, rule + 1):
            key = (ffi, sep)
            es = 0.0 if sep == 1 else ff.rules.es_scale[sep - 2]
            lj = 0.0 if sep == 1 else ff.rules.lj_scale[sep - 2]
            self.exclusion_params[key] = (sep, es, lj)
            for t in tuples[sep - 1]:
                self._exclude(t[0], t[sep - 1], key)

    def _exclude(self, a1: int, a2: int, key) -> None:
        group1 = [a1] + [o for o in self.nbrs[a1] if self.anum[o] == 0]
        group2 = [a2] + [o for o in self.nbrs[a2] if self.anum[o] == 0]
        for x in group1:
            for y in group2:
                if x != y:
                    self.exclusion.setdefault((min(x, y), max(x, y)), key)

    # -- the whole run ---------------------------------------------------------
    def run(self, cmap_chirality: bool) -> System:
        s = self.s
        frags: dict[int, list[int]] = {}
        for a, f in enumerate(s.fragids.tolist()):
            frags.setdefault(f, []).append(a)
        fragments = [frags[k] for k in sorted(frags)]
        assigned = [False] * len(fragments)
        whynot: list[list[str]] = [[] for _ in fragments]
        warned: set[str] = set()
        for ffi, ff in enumerate(self.ffs):
            typed = _Typed()
            for k, frag in enumerate(fragments):
                matches, why = self._match_fragment(ff, ffi, frag)
                if matches is None:
                    whynot[k].append(why)
                elif assigned[k]:
                    formula = _formula(self.anum[a] for a in frag)
                    if formula not in warned:
                        warnings.warn(f"fragment {k} ({formula}) was matched by multiple force "
                                      "fields; the first match takes precedence",
                                      ViparrWarning, stacklevel=3)  # fmt: skip
                        warned.add(formula)
                else:
                    self._assign(typed, matches)
                    assigned[k] = True
            for name in ff.rules.plugins:
                self.plugins.add(name)
                self._plugin(name, ff, ffi, typed)
        for k, ok in enumerate(assigned):
            if not ok:
                formula = _formula(self.anum[a] for a in fragments[k])
                raise ViparrError(f"No force field could parameterize fragment {k} ({formula}):\n"
                                  + "\n".join(whynot[k]))  # fmt: skip
        return self._compile(cmap_chirality)

    # -- compiling tables ------------------------------------------------------
    def _vdw_rule(self) -> tuple[str, str]:
        func = rule = ""
        for ff in self.ffs:
            func = _merge_vdw(func, ff.rules.vdw_func)
            rule = _merge_vdw(rule, ff.rules.vdw_comb_rule)
        if not func:
            warnings.warn("No vdw_func provided; assuming default of lj12_6_sig_epsilon",
                          ViparrWarning, stacklevel=4)  # fmt: skip
            func = "lj12_6_sig_epsilon"
        if func not in _VDW_FUNCS:
            raise ViparrError(f"Unsupported VDW function: {func}")
        table, rules = _VDW_FUNCS[func]
        if not rule:
            warnings.warn(f"No vdw_comb_rule provided; assuming default of {rules[0]}",
                          ViparrWarning, stacklevel=4)  # fmt: skip
            rule = rules[0]
        if rule not in rules:
            raise ViparrError(f"Unsupported combine rule {rule} for VDW function: {func}")
        return table, rule

    def _compile(self, cmap_chirality: bool) -> System:
        s = self.s
        funct, rule = self._vdw_rule()
        # pseudo particles and their bonds
        k = len(self.pseudos)
        if k:
            parents = np.array(self.pseudos, np.int64)
            s.add_atoms(k, residue=np.array(self.residue[self.n0:], np.int64),
                        name=np.array(self.names[self.n0:], dtype=STR), anum=np.zeros(k, np.int64),
                        pos=self.pos[parents])  # fmt: skip
        if self.new_bonds:
            s.add_bonds(np.array(self.new_bonds, np.int64), order=1)
        s.atoms["charge"] = np.array(self.charge, np.float64)
        if self.rename_atoms:
            s.atoms["name"] = np.array(self.names, dtype=STR)
        if self.rename_residues:
            s.residues["name"] = np.array(self.resnames, dtype=STR)
        # tables made while compiling: Urey-Bradley and improper_trig
        if self.ub_terms:
            tab = self._table("stretch_harm", 2)
            for n, (pair, r) in enumerate(self.ub_terms):
                tab.add(
                    pair, ("ub", n), {"type": "ureybradley " + r.type, **r.params, "memo": r.memo}
                )
        if self.improper_trig:
            tab = self._table("dihedral_trig", 4)
            seen: dict[frozenset, int] = {}
            for n, (t, _ffi, r) in enumerate(self.improper_trig):
                row = _row(r)
                old = seen.get(frozenset(t))
                if old is not None:  # viparr overwrites an earlier improper on the same atoms
                    tab.params[tab.keys[old]] = row
                    continue
                seen[frozenset(t)] = len(tab.atoms)
                tab.add(t, ("improper", n), row)
        if self.cmap_terms:
            self._cmap(cmap_chirality)
        for name, tab in self.tables.items():
            self._emit(name, tab)
        self._emit_nonbonded(funct, rule)
        if "exclusions" in self.plugins:
            self._emit_exclusions_and_pairs(rule)
        if self.mass:
            mass = s.atoms["mass"].copy()
            for a, m in self.mass.items():
                mass[a] = m
            s.atoms["mass"] = mass
        s.nonbonded_info.vdw_funct, s.nonbonded_info.vdw_rule = funct, rule
        meta = ParamTable()
        meta.add_prop("path", str)
        meta.add_prop("info", str)
        for ff in self.ffs:
            meta.add_param(path=ff.name, info="".join(x + "\n" for x in ff.rules.info))
        s.aux_tables["forcefield"] = meta
        return s

    def _cmap(self, chirality: bool) -> None:
        taken = set(self.s.aux_tables)
        grids: dict[str, np.ndarray] = {}
        mirrored = set()
        if chirality:
            for n, (t, _ffi, _i) in enumerate(self.cmap_terms):
                if self._is_d(t[1], t[2], t[3]):
                    mirrored.add(n)

        def free_name():
            k = 1
            while f"cmap{k}" in taken:
                k += 1
            return f"cmap{k}"

        names: dict[tuple, str] = {}
        for mirror in (False, True):  # L names exactly as viparr gives them; mirrored maps after
            for n, (_t, ffi, i) in enumerate(self.cmap_terms):
                if (n in mirrored) != mirror:
                    continue
                row = self.ffs[ffi].params["torsiontorsion_cmap"][i]
                cid = row.params["cmapid"]
                key = (ffi, cid, mirror)
                if key in names:
                    continue
                num = int(cid[4:])
                cmaps = self.ffs[ffi].cmaps
                if num < 1 or num > len(cmaps) or cmaps[num - 1] is None:
                    raise ViparrError(f"Missing cmap table {num} in {self.ffs[ffi].name}")
                name = cid if (cid not in taken and not mirror) else free_name()
                taken.add(name)
                names[key] = name
                grids[name] = _mirror_cmap(cmaps[num - 1]) if mirror else cmaps[num - 1]
        tab = self._table("torsiontorsion_cmap", 8)
        for n, (t, ffi, i) in enumerate(self.cmap_terms):
            row = self.ffs[ffi].params["torsiontorsion_cmap"][i]
            mirror = n in mirrored
            name = names[(ffi, row.params["cmapid"], mirror)]
            memo = row.memo
            if mirror:
                memo = (
                    memo + "; " if memo else ""
                ) + "mirrored for a D residue: E(phi, psi) = E_L(-phi, -psi)"
            tab.add(t, (ffi, "cmap", i, mirror), {"type": row.type, "cmapid": name, "memo": memo})
        for name, grid in grids.items():
            aux = ParamTable()
            for prop in ("phi", "psi", "energy"):
                aux.add_prop(prop)
            aux.add_params(len(grid), phi=grid[:, 0], psi=grid[:, 1], energy=grid[:, 2])
            self.s.aux_tables[name] = aux

    def _is_d(self, n: int, ca: int, c: int) -> bool:
        side = [o for o in self.nbrs[ca] if o not in (n, c) and self.anum[o] > 1]
        if len(side) != 1:
            return False  # glycine (or not an amino acid): nothing to mirror
        p = self.pos
        v = np.dot(p[n] - p[ca], np.cross(p[c] - p[ca], p[side[0]] - p[ca]))
        if abs(v) < 1e-6:
            warnings.warn(f"cannot tell the chirality of atom {ca} from its coordinates; "
                          "using the L CMAP", ViparrWarning, stacklevel=5)  # fmt: skip
            return False
        return v * _L_SIGN < 0

    def _emit(self, name: str, tab: _Terms) -> None:
        s = self.s
        if name in TERM_SCHEMAS:
            t = s.add_table_from_schema(name)
        else:
            t = s.add_table(name, tab.natoms, tab.category)
        rows = list(tab.params.values())
        cols: dict[str, list] = {}
        for r in rows:
            for c in r:
                cols.setdefault(c, [])
        for c in cols:
            if c not in t.params.props:
                t.params.add_prop(
                    c, str if isinstance(next(r[c] for r in rows if c in r), str) else float
                )
            kind = t.params.prop_type(c)
            default = "" if kind == "str" else 0.0
            cols[c] = [r.get(c, default) for r in rows]
        arrays = {c: np.array(v, dtype=STR if t.params.prop_type(c) == "str" else np.float64)
                  for c, v in cols.items()}  # fmt: skip
        pids = t.params.add_params(len(rows), **arrays)
        where = {k: p for k, p in zip(tab.params, pids.tolist(), strict=True)}
        t.add_terms(np.array(tab.atoms, np.int64).reshape(-1, tab.natoms),
                    params=np.array([where[k] for k in tab.keys], np.int64))  # fmt: skip

    def _emit_nonbonded(self, funct: str, rule: str) -> None:
        if not self.nonbonded:
            return
        s = self.s
        nb = s.add_nonbonded_from_schema(funct, rule)
        for c in ("type", "nbfix_identifier", "memo"):
            nb.params.add_prop(c, str)
        keys = list(dict.fromkeys(self.nonbonded.values()))
        where = {k: n for n, k in enumerate(keys)}
        rows = [self.ffs[ffi].params["vdw1"][i] for ffi, i in keys]
        idents = [self.ffs[ffi].rules.nbfix_identifier for ffi, _ in keys]
        pids = nb.params.add_params(
            len(rows), sigma=np.array([r.params["sigma"] for r in rows], np.float64),
            epsilon=np.array([r.params["epsilon"] for r in rows], np.float64),
            type=np.array([r.type for r in rows], dtype=STR),
            nbfix_identifier=np.array(idents, dtype=STR),
            memo=np.array([r.memo for r in rows], dtype=STR))  # fmt: skip
        atoms = list(self.nonbonded)
        nb.add_terms(np.array(atoms, np.int64)[:, None],
                     params=pids[[where[self.nonbonded[a]] for a in atoms]])  # fmt: skip
        # NBFIX: vdw2 rows override pairs of vdw1 types that share an nbfix_identifier
        done: dict[tuple, dict] = {}
        for ffi, r in self.vdw2:
            ident = self.ffs[ffi].rules.nbfix_identifier
            t1, t2 = r.type.split()
            values = {**r.params, "nbfix_identifier": ident}
            prev = done.get((ident, t1, t2)) or done.get((ident, t2, t1))
            if prev is not None:
                if prev != values:
                    raise ViparrError(f"vdw2 rows override the same types ({t1}, {t2}) with "
                                      "different parameters")  # fmt: skip
                continue
            done[(ident, t1, t2)] = values
            p1 = [where[k] for k, row in zip(keys, rows, strict=True)
                  if row.type == t1 and self.ffs[k[0]].rules.nbfix_identifier == ident]  # fmt: skip
            p2 = [where[k] for k, row in zip(keys, rows, strict=True)
                  if row.type == t2 and self.ffs[k[0]].rules.nbfix_identifier == ident]  # fmt: skip
            for a in p1:
                for b in p2:
                    nb.overrides.set(
                        a, b, type=r.type, **r.params, nbfix_identifier=ident, memo=r.memo
                    )

    def _emit_exclusions_and_pairs(self, rule: str) -> None:
        s = self.s
        pairs = list(self.exclusion)
        if not pairs:
            return
        s.add_table_from_schema("exclusion").add_terms(np.array(pairs, np.int64))
        q = self.charge
        pair_params: dict[tuple[int, int], dict] = {}
        for pair, key in self.exclusion.items():  # scaled electrostatics
            es = self.exclusion_params[key][1]
            if es != 0:
                pair_params.setdefault(pair, {"aij": 0.0, "bij": 0.0, "qij": 0.0})["qij"] = (
                    es * q[pair[0]] * q[pair[1]]
                )
        vdw1 = {a: self.ffs[f].params["vdw1"][i].params for a, (f, i) in self.nonbonded.items()}
        vdw14 = None
        if self.vdw14 is not None:
            vdw14 = {a: self.ffs[f].params["vdw1_14"][i].params for a, (f, i) in self.vdw14.items()}
        for pair, key in self.exclusion.items():  # scaled Lennard-Jones
            sep, _, lj = self.exclusion_params[key]
            if lj == 0:
                continue
            vi = vj = None
            if vdw14 is not None and sep == 4:
                vi, vj = vdw14.get(pair[0]), vdw14.get(pair[1])
            vi = vi if vi is not None else vdw1.get(pair[0])
            vj = vj if vj is not None else vdw1.get(pair[1])
            if vi is None or vj is None:
                warnings.warn(f"Missing vdw term for atom {pair[0] if vi is None else pair[1]}",
                              ViparrWarning, stacklevel=4)  # fmt: skip
                continue
            sij = (vi["sigma"] + vj["sigma"]) * 0.5 if rule == "arithmetic/geometric" else (
                (vi["sigma"] * vj["sigma"]) ** 0.5)  # fmt: skip
            eij = (vi["epsilon"] * vj["epsilon"]) ** 0.5
            p = pair_params.setdefault(pair, {"aij": 0.0, "bij": 0.0, "qij": 0.0})
            p["aij"] = lj * sij**12 * eij * 4.0
            p["bij"] = lj * sij**6 * eij * 4.0
        if not pair_params:
            return
        pt = s.add_table_from_schema("pair_12_6_es")
        for c in ("type", "memo"):
            pt.params.add_prop(c, str)
        items = list(pair_params.items())
        pids = pt.params.add_params(len(items), **{c: np.array([p[c] for _, p in items], np.float64)
                                                   for c in ("aij", "bij", "qij")})  # fmt: skip
        pt.add_terms(np.array([k for k, _ in items], np.int64), params=pids)


def _mirror_cmap(grid: np.ndarray) -> np.ndarray:
    """The grid of the mirror image: energy at (phi, psi) is the L energy at (-phi, -psi)."""

    def wrap(x):
        return np.round((x + 180.0) % 360.0 - 180.0, 6)

    lookup = {
        (p, s): e for p, s, e in zip(wrap(grid[:, 0]), wrap(grid[:, 1]), grid[:, 2], strict=True)
    }
    out = grid.copy()
    for k, (p, s) in enumerate(zip(wrap(-grid[:, 0]), wrap(-grid[:, 1]), strict=True)):
        e = lookup.get((p, s))
        if e is None:
            raise ViparrError(f"CMAP grid has no point at ({p}, {s}); cannot mirror it")
        out[k, 2] = e
    return out


def _fix_masses(s: System) -> None:
    """viparr's FixMasses: atoms of one element get the median of their masses."""
    anum, mass = s.atoms["anum"], s.atoms["mass"].copy()
    for z in np.unique(anum[anum > 0]).tolist():
        sel = np.flatnonzero(anum == z)
        if len(np.unique(mass[sel])) > 1:
            mass[sel] = np.sort(mass[sel])[len(sel) // 2]
    s.atoms["mass"] = mass


def _reorder_ids(s: System) -> System:
    """Pseudo particles right after their parent atoms (viparr ``--reorder-ids``)."""
    anum = s.atoms["anum"]
    parent = {}
    for i, j in zip(s.bonds["i"].tolist(), s.bonds["j"].tolist(), strict=True):
        if anum[j] == 0 and anum[i] != 0:
            parent.setdefault(j, i)
        elif anum[i] == 0 and anum[j] != 0:
            parent.setdefault(i, j)
    children: dict[int, list[int]] = {}
    for p, a in sorted(parent.items()):
        children.setdefault(a, []).append(p)
    order = []
    for a in range(s.natoms):
        if anum[a] != 0 or a not in parent:
            order.append(a)
            order += children.get(a, [])
    s.reorder_atoms(np.array(order, np.int64))
    return s


def build_constraints(system: System, atoms=None, keep: bool = False, exclude=()) -> None:
    """Add viparr's constraints to a parameterized system, in place.

    A heavy atom and the hydrogens bonded to it become one ``constraint_ahN``
    term (heavy atom first, lengths from ``stretch_harm``); a water oxygen
    with two hydrogens becomes ``constraint_hoh`` (with the H-O-H angle from
    ``angle_harm``). The ``stretch_harm`` and ``angle_harm`` terms they
    replace get ``constrained = 1``, so :func:`boonza.to_openmm` turns them
    into OpenMM constraints (``keep=True`` leaves them unconstrained).
    ``atoms`` limits the atoms considered; ``exclude`` skips kinds such as
    ``"hoh"`` or ``"ah1"``. Existing constraints of those atoms are replaced.
    """
    s = system
    n = s.natoms
    sel = np.arange(n) if atoms is None else np.unique(np.asarray(getattr(atoms, "ids", atoms)))
    insel = np.zeros(n, bool)
    insel[sel] = True
    for name in list(s.table_names):
        t = s.table(name)
        if t.category == "constraint" and len(t):
            hit = np.flatnonzero(insel[t.atoms].any(axis=1))
            if len(hit):
                t.delete_terms(hit)
    taken = np.zeros(n, bool)
    for name in s.table_names:
        t = s.table(name)
        if t.category == "constraint" and len(t):
            a = t.atoms.ravel()
            if len(np.unique(a)) != len(a) or taken[a].any():
                raise ViparrError("Existing constraints in system overlap")
            taken[a] = True
    anum = s.atoms["anum"].tolist()
    nbrs: list[list[int]] = [[] for _ in range(n)]
    for i, j in zip(s.bonds["i"].tolist(), s.bonds["j"].tolist(), strict=True):
        nbrs[i].append(j)
        nbrs[j].append(i)
    st, at = s.tables.get("stretch_harm"), s.tables.get("angle_harm")
    bond_terms: dict[tuple[int, int], list[int]] = {}
    angle_terms: dict[frozenset, list[int]] = {}
    if st is not None:
        for k, (i, j) in enumerate(st.atoms.tolist()):
            bond_terms.setdefault((min(i, j), max(i, j)), []).append(k)
        r0 = st.values("r0")
    if at is not None:
        for k, a in enumerate(at.atoms.tolist()):
            angle_terms.setdefault(frozenset(a), []).append(k)
        theta0 = at.values("theta0")

    def one(found, what):
        if len(found) != 1:
            many = "Multiple" if found else "No"
            raise ViparrError(f"Cannot build constraint: {many} {what}")
        return found[0]

    groups: dict[str, list] = {}
    cbonds: list[int] = []
    cangles: list[int] = []
    done = np.zeros(n, bool)
    for a in sel.tolist():
        if anum[a] == 0:
            continue
        if anum[a] == 1:  # a hydrogen stands for its heavy atom
            a = next((b for b in nbrs[a] if anum[b] > 1), -1)
            if a < 0:
                continue
        if done[a]:
            continue
        done[a] = True
        hs = [b for b in nbrs[a] if anum[b] == 1]
        if not hs:
            continue
        hoh = anum[a] == 8 and len(hs) == 2 and sum(1 for b in nbrs[a] if anum[b] > 0) == 2
        kind = "hoh" if hoh else f"ah{len(hs)}"
        if kind in exclude:
            continue
        group = [a, *(sorted(hs) if hoh else hs)]
        if taken[group].any():
            raise ViparrError(f"Constraint with heavy atom {a} would overlap other constraints")
        taken[group] = True
        if st is None or (hoh and at is None):
            raise ViparrError(
                "Must have stretch_harm and angle_harm terms before adding constraints"
            )
        values = []
        if hoh:
            k = one(angle_terms.get(frozenset(group), []),
                    f"angle_harm terms for water ({group[1]}, {a}, {group[2]})")  # fmt: skip
            cangles.append(k)
            values.append(float(theta0[k]))
        for h in group[1:]:
            k = one(
                bond_terms.get((min(a, h), max(a, h)), []),
                f"stretch_harm terms for bond ({a}, {h})",
            )
            cbonds.append(k)
            values.append(float(r0[k]))
        groups.setdefault(f"constraint_{kind}", []).append((group, tuple(values)))
    for name, items in groups.items():
        if name in TERM_SCHEMAS:
            t = s.add_table_from_schema(name)
        else:
            t = s.add_table(name, len(items[0][0]), "constraint")
            for k in range(1, len(items[0][0])):
                t.params.add_prop(f"r{k}")
        uniq = list(dict.fromkeys(v for _, v in items))
        where = {v: k for k, v in enumerate(uniq)}
        arr = np.array(uniq, np.float64)
        pids = t.params.add_params(
            len(uniq), **{c: arr[:, k] for k, c in enumerate(t.params.props)}
        )
        t.add_terms(np.array([g for g, _ in items], np.int64),
                    params=pids[[where[v] for _, v in items]])  # fmt: skip
    for table, rows in ((st, cbonds), (at, cangles)):
        if not rows:
            continue
        if "constrained" not in table.term_props:
            table.add_term_prop("constrained", "int")
        flag = table.values("constrained")
        flag[insel[table.atoms].any(axis=1)] = 0
        if not keep:
            flag[rows] = 1
        table._t.set("constrained", flag)


def parameterize(system: System, forcefields, *, rename_atoms: bool = False,
                 rename_residues: bool = False, fix_masses: bool = True, fatal: bool = True,
                 cmap_chirality: bool = True, reorder_ids: bool = True,
                 constraints: bool = True, path=None) -> System:  # fmt: skip
    """A copy of ``system`` with a force field from viparr force fields.

    ``forcefields`` is a list of :class:`ViparrForcefield` objects, names in
    ``path`` (default ``$VIPARR_FFPATH``) or directories. Each molecule takes
    its parameters from the first force field whose templates match all its
    residues, so list the force field you trust most first; combine
    force fields into one with :func:`merge_forcefields`.

    The input's force field and pseudo particles are dropped; virtual sites
    come from the templates. ``rename_atoms``/``rename_residues`` copy names
    from the matched templates. ``fix_masses`` gives all atoms of an element
    the median of their masses, as viparr does by default. ``fatal=False``
    turns missing parameters into warnings. ``cmap_chirality=False`` applies
    L CMAP grids to D residues as viparr does. Pseudo particles go
    right after their parent atoms, so every residue's atoms stay together as
    OpenMM requires (viparr's ``--reorder-ids``); ``reorder_ids=False``
    appends them after all real atoms, as viparr does by default.
    ``constraints`` adds viparr's constraints (see :func:`build_constraints`),
    as viparr does by default.
    """
    ffs = [ff if isinstance(ff, ViparrForcefield) else load_forcefield(ff, path)
           for ff in ([forcefields] if isinstance(forcefields, str | os.PathLike | ViparrForcefield)
                      else forcefields)]  # fmt: skip
    if not ffs:
        raise ViparrError("no force fields given")
    run = _Parameterizer(system, ffs, rename_atoms, rename_residues, fatal)
    s = run.run(cmap_chirality)
    if constraints:
        build_constraints(s)
    if fix_masses:
        _fix_masses(s)
    if reorder_ids:
        s = _reorder_ids(s)
    return s
