"""OpenMM force fields (ffxml): read the XML files and parameterize systems natively.

This reads OpenMM's XML force fields (``amber19-all.xml``, ``charmm36_2024.xml``,
water models, or your own files) and builds the force field that OpenMM's
``ForceField.createSystem`` would build, as boonza tables, without calling
OpenMM.  It follows OpenMM's own rules, and the tests compare it term by term
with ``createSystem``:

- residues match templates by element and bond graph (bonds to neighbouring
  residues included), with OpenMM's search order, so equivalent atoms get the
  same template atoms; residues that match no template are tried with every
  combination of one-residue patches, then with multi-residue patches (such as
  disulfides); two templates matching one residue is an error unless they
  assign identical parameters;
- bonds and angles take the first matching row; torsions prefer rows without
  wildcards; impropers follow the ``default``, ``amber`` or ``charmm``
  ordering rules of their section;
- 1-4 pairs are scaled by the force field's ``coulomb14scale``/``lj14scale``;
  CHARMM's ``LennardJonesForce`` gives per-type Lennard-Jones with NBFIX
  overrides and its own 1-4 terms; Urey-Bradley terms join ``stretch_harm``;
- CHARMM36 (2024)'s impropers come from a ``<Script>`` in the XML that works
  on residue and atom names.  boonza reads the script's tables (as literals,
  without running it) and applies them the same way, so for those impropers
  the structure needs CHARMM atom names.  Other scripts are not supported.

The output uses the same tables, units and conventions as
:func:`boonza.from_openmm` (one ``dihedral_trig`` term per periodicity,
1-4 electrostatics and Lennard-Jones as separate ``pair_12_6_es`` terms for
CHARMM), and constrained bonds and angles are kept and marked ``constrained``
as :func:`boonza.parameterize` does.  Extra particles a template has but the
structure lacks (TIP4P/OPC sites) are added after their residue's atoms and
placed as OpenMM's ``Modeller.addExtraParticles`` would.

Not supported: custom forces other than the ones above, implicit solvent,
AMOEBA and Drude force fields, residue template generators (GAFF/SMIRNOFF)
and templates spanning several residues.
"""

from __future__ import annotations

import ast
import heapq
import itertools
import math
import os
import xml.etree.ElementTree as etree
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from ._columns import STR
from .elements import atomic_number
from .system import NonbondedInfo, System
from .terms import ParamTable

__all__ = ["FFXMLError", "OpenMMForcefield", "load_openmm_forcefield", "parameterize_openmm"]

KCAL = 4.184  # kJ per kcal
NM = 0.1  # nm per Å
_FDAT3_WEIGHTS = [[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]


class FFXMLError(ValueError):
    """An XML force field that cannot be read, or a system it cannot parameterize."""


def _number(text) -> float:
    return float(text)


# ---- force field data ------------------------------------------------------------


class _AtomType:
    __slots__ = ("name", "cls", "mass", "anum")

    def __init__(self, name, cls, mass, anum):
        self.name, self.cls, self.mass, self.anum = name, cls, mass, anum


class _TAtom:
    """An atom of a residue template (anum 0: an extra particle with no element)."""

    __slots__ = ("name", "type", "anum", "params", "bonded", "external")

    def __init__(self, name, type_, anum, params):
        self.name, self.type, self.anum, self.params = name, type_, anum, params
        self.bonded: list[int] = []
        self.external = 0


class _Site:
    """A virtual site of a template: ``index`` is placed from ``atoms``."""

    def __init__(self, attrib, index_of):
        self.kind = attrib["type"]
        if self.kind == "average2":
            n, self.weights = 2, [float(attrib["weight1"]), float(attrib["weight2"])]
        elif self.kind == "average3":
            n = 3
            self.weights = [float(attrib[f"weight{k}"]) for k in (1, 2, 3)]
        elif self.kind == "outOfPlane":
            n = 3
            self.weights = [float(attrib[k]) for k in ("weight12", "weight13", "weightCross")]
        elif self.kind == "localCoords":
            n, self.origin, self.xw, self.yw = 0, [], [], []
            while f"wo{n + 1}" in attrib:
                n += 1
                self.origin.append(float(attrib[f"wo{n}"]))
                self.xw.append(float(attrib[f"wx{n}"]))
                self.yw.append(float(attrib[f"wy{n}"]))
            self.local = [float(attrib[k]) for k in ("p1", "p2", "p3")]
        else:
            raise FFXMLError(f"Unknown virtual site type: {self.kind}")
        if "siteName" in attrib:
            self.index = index_of[attrib["siteName"]]
            self.atoms = [index_of[attrib[f"atomName{i + 1}"]] for i in range(n)]
        else:
            self.index = int(attrib["index"])
            self.atoms = [int(attrib[f"atom{i + 1}"]) for i in range(n)]
        if "excludeWith" in attrib:
            self.exclude_with = int(attrib["excludeWith"])
        else:
            self.exclude_with = self.atoms[0]

    def same(self, other) -> bool:
        return (self.kind == other.kind and self.index == other.index and self.atoms == other.atoms
                and getattr(self, "weights", None) == getattr(other, "weights", None)
                and getattr(self, "local", None) == getattr(other, "local", None))  # fmt: skip


class _Template:
    def __init__(self, name):
        self.name = name
        self.atoms: list[_TAtom] = []
        self.index: dict[str, int] = {}
        self.sites: list[_Site] = []
        self.bonds: list[tuple[int, int]] = []
        self.external: list[int] = []
        self.constraints: list[tuple[int, int, float]] = []
        self.override = 0
        self.rigid_water = True

    def add_atom(self, atom: _TAtom) -> None:
        self.atoms.append(atom)
        self.index[atom.name] = len(self.atoms) - 1

    def atom_index(self, name) -> int:
        k = self.index.get(name)
        if k is None:
            raise FFXMLError(f"Atom name '{name}' not found in residue template '{self.name}'")
        return k

    def add_bond(self, i, j) -> None:
        self.bonds.append((i, j))
        self.atoms[i].bonded.append(j)
        self.atoms[j].bonded.append(i)

    def add_external(self, i) -> None:
        self.external.append(i)
        self.atoms[i].external += 1

    def identical_parameters(self, other, m1, m2) -> bool:
        a1 = [self.atoms[m] for m in m1]
        a2 = [other.atoms[m] for m in m2]
        if any(x.type != y.type or x.params != y.params for x, y in zip(a1, a2, strict=True)):
            return False
        if set(self.constraints) != set(other.constraints):
            return False
        return len(self.sites) == len(other.sites) and all(
            x.same(y) for x, y in zip(self.sites, other.sites, strict=True))  # fmt: skip


class _PatchAtom:
    __slots__ = ("residue", "name")

    def __init__(self, description):
        if ":" in description:
            k = description.index(":")
            self.residue, self.name = int(description[:k]) - 1, description[k + 1 :]
        else:
            self.residue, self.name = 0, description


class _Patch:
    def __init__(self, name, nres):
        self.name, self.nres = name, nres
        self.added = [[] for _ in range(nres)]
        self.changed = [[] for _ in range(nres)]
        self.deleted: list[_PatchAtom] = []
        self.added_bonds: list[tuple[_PatchAtom, _PatchAtom]] = []
        self.deleted_bonds: list[tuple[_PatchAtom, _PatchAtom]] = []
        self.added_external: list[_PatchAtom] = []
        self.deleted_external: list[_PatchAtom] = []
        self.names: set[str] = set()
        self.sites = [[] for _ in range(nres)]
        self.index = 0

    def create(self, templates) -> list[_Template]:
        """OpenMM's createPatchedTemplates."""
        if len(templates) != self.nres:
            raise FFXMLError(f"Patch '{self.name}' expected {self.nres} templates")
        out = []
        for index, template in enumerate(templates):
            new = _Template(f"{template.name}-{self.name}")
            out.append(new)
            for atom in template.atoms:
                if not any(d.name == atom.name and d.residue == index for d in self.deleted):
                    new.add_atom(_TAtom(atom.name, atom.type, atom.anum, atom.params))
            for atom in self.added[index]:
                if any(a.name == atom.name for a in new.atoms):
                    raise FFXMLError(f"Patch '{self.name}' adds an atom with an existing name")
                new.add_atom(_TAtom(atom.name, atom.type, atom.anum, atom.params))
            old_index = {a.name: i for i, a in enumerate(template.atoms)}
            new_index = {a.name: i for i, a in enumerate(new.atoms)}
            for atom in self.changed[index]:
                if atom.name not in new_index:
                    raise FFXMLError(f"Patch '{self.name}' modifies nonexistent atom '{atom.name}'")
                new.atoms[new_index[atom.name]] = _TAtom(
                    atom.name, atom.type, atom.anum, atom.params
                )
            index_map = {old_index[nm]: new_index[nm] for nm in new_index if nm in old_index}
            for site in template.sites:
                if site.index in index_map and all(i in index_map for i in site.atoms):
                    ns = deepcopy(site)
                    ns.index = index_map[site.index]
                    ns.atoms = [index_map[i] for i in site.atoms]
                    ns.exclude_with = index_map[site.exclude_with]
                    new.sites.append(ns)
            atom_map = {template.atoms[i]: index_map[i] for i in index_map}
            gone = [
                (a.name, b.name)
                for a, b in self.deleted_bonds
                if a.residue == index and b.residue == index
            ]
            for i, j in template.bonds:
                a1, a2 = template.atoms[i], template.atoms[j]
                if a1 in atom_map and a2 in atom_map and (a1.name, a2.name) not in gone \
                        and (a2.name, a1.name) not in gone:  # fmt: skip
                    new.add_bond(atom_map[a1], atom_map[a2])
            gone_ext = [a.name for a in self.deleted_external if a.residue == index]
            for i in template.external:
                if template.atoms[i].name not in gone_ext:
                    new.add_external(index_map[i])
            for a1, a2 in self.added_bonds:
                if a1.residue == index and a2.residue == index:
                    new.add_bond(new.atom_index(a1.name), new.atom_index(a2.name))
                elif a1.residue == index:
                    new.add_external(new.atom_index(a1.name))
                elif a2.residue == index:
                    new.add_external(new.atom_index(a2.name))
            for a in self.added_external:
                new.add_external(new.atom_index(a.name))
            for i, j, d in template.constraints:
                a1, a2 = template.atoms[i], template.atoms[j]
                if a1 in atom_map and a2 in atom_map:
                    new.constraints.append((atom_map[a1], atom_map[a2], d))
            site_map = {
                i: new_index[a.name] for i, a in enumerate(self.added[index] + self.changed[index])
            }
            for site in self.sites[index]:
                ns = deepcopy(site)
                ns.index = site_map[site.index]
                ns.atoms = [site_map[i] for i in site.atoms]
                ns.exclude_with = site_map[site.exclude_with]
                new.sites = [s for s in new.sites if s.index != ns.index]
                new.sites.append(ns)
        return out


def _signature(anums) -> tuple:
    return tuple(sorted(Counter(a for a in anums if a > 0).items()))


# ---- parameter sections ---------------------------------------------------------------


class _TypeParams:
    """OpenMM's _AtomTypeParameters: per-type values, or values from the residue."""

    def __init__(self, ff, force, names):
        self.ff, self.force, self.names = ff, force, tuple(names)
        self.for_type: dict[str, dict] = {}
        self.extra: dict[str, dict] = {}

    def register(self, attrib, expected=None):
        expected = self.names if expected is None else expected
        types = self.ff._find_types(attrib, 1)
        if None in types:
            return
        values, extra = {}, {}
        for key, v in attrib.items():
            if key in expected:
                values[key] = _number(v)
            elif key not in ("type", "class"):
                extra[key] = v
        missing = [k for k in expected if k not in values]
        if missing:
            raise FFXMLError(f'{self.force}: No value specified for "{missing[0]}"')
        for t in types[0]:
            self.for_type[t] = values
            self.extra[t] = extra

    def parse(self, element, atom_tag="Atom"):
        expected = list(self.names)
        for node in element.findall("UseAttributeFromResidue"):
            if node.attrib["name"] not in expected:
                raise FFXMLError(f"{self.force}: <UseAttributeFromResidue> of an invalid attribute")
            expected.remove(node.attrib["name"])
        for atom in element.findall(atom_tag):
            self.register(atom.attrib, expected)

    def values(self, atype, residue_params) -> list[float]:
        if atype not in self.for_type:
            raise FFXMLError(f"{self.force}: No parameters defined for atom type {atype}")
        own = self.for_type[atype]
        out = []
        for name in self.names:
            if name in own:
                out.append(own[name])
            elif name in residue_params:
                out.append(float(residue_params[name]))
            else:
                raise FFXMLError(f'{self.force}: No value specified for "{name}"')
        return out


class _Bonds:
    def __init__(self, ff):
        self.ff, self.rows, self.by_type = ff, [], defaultdict(set)

    def parse(self, element):
        for node in element.findall("Bond"):
            types = self.ff._find_types(node.attrib, 2)
            if None in types:
                continue
            k = len(self.rows)
            self.rows.append(
                (types[0], types[1], _number(node.attrib["length"]), _number(node.attrib["k"]))
            )
            for t in types[0] | types[1]:
                self.by_type[t].add(k)


class _Angles:
    def __init__(self, ff):
        self.ff, self.rows, self.by_type2 = ff, [], defaultdict(list)

    def parse(self, element):
        for node in element.findall("Angle"):
            types = self.ff._find_types(node.attrib, 3)
            if None in types:
                continue
            k = len(self.rows)
            self.rows.append((*types, _number(node.attrib["angle"]), _number(node.attrib["k"])))
            for t in types[1]:
                self.by_type2[t].append(k)


class _Torsion:
    __slots__ = ("types", "terms", "ordering")

    def __init__(self, types, terms, ordering="default"):
        self.types, self.terms, self.ordering = types, terms, ordering


class _Torsions:
    def __init__(self, ff):
        self.ff, self.proper, self.improper = ff, [], []
        self.by_type = defaultdict(set)

    def _parse_one(self, attrib):
        types = self.ff._find_types(attrib, 4)
        if None in types:
            return None
        terms, k = [], 1
        while f"phase{k}" in attrib:
            terms.append(
                (
                    int(attrib[f"periodicity{k}"]),
                    _number(attrib[f"phase{k}"]),
                    _number(attrib[f"k{k}"]),
                )
            )
            k += 1
        return _Torsion(types, terms)

    def parse(self, element):
        ordering = element.attrib.get("ordering", "default")
        if ordering not in ("default", "charmm", "amber"):
            raise FFXMLError(f"improper ordering {ordering!r} is not supported")
        for node in element.findall("Proper"):
            t = self._parse_one(node.attrib)
            if t is not None:
                k = len(self.proper)
                self.proper.append(t)
                for x in t.types[1] | t.types[2]:
                    self.by_type[x].add(k)
        for node in element.findall("Improper"):
            t = self._parse_one(node.attrib)
            if t is not None:
                t.ordering = ordering
                self.improper.append(t)


class _Cmap:
    def __init__(self, ff):
        self.ff, self.maps, self.torsions = ff, [], []

    def parse(self, element):
        offset = len(self.maps)
        for node in element.findall("Map"):
            values = [float(x) for x in node.text.split()]
            size = int(round(math.sqrt(len(values))))
            if size * size != len(values):
                raise FFXMLError("CMAP must have the same number of elements along each dimension")
            self.maps.append(values)
        for node in element.findall("Torsion"):
            types = self.ff._find_types(node.attrib, 5)
            if None not in types:
                self.torsions.append((types, int(node.attrib["map"]) + offset))


class _Nonbonded:
    def __init__(self, ff, c14, lj14):
        self.ff, self.c14, self.lj14 = ff, c14, lj14
        self.params = _TypeParams(ff, "NonbondedForce", ("charge", "sigma", "epsilon"))


class _LennardJones:
    def __init__(self, ff, lj14):
        self.ff, self.lj14 = ff, lj14
        self.params = _TypeParams(ff, "LennardJonesForce", ("sigma", "epsilon"))
        self.nbfix: list[tuple[float, float]] = []
        self.types1, self.types2 = defaultdict(set), defaultdict(set)

    def parse(self, element):
        for node in element.findall("Atom"):
            self.params.register(node.attrib)
        for node in element.findall("NBFixPair"):
            types = self.ff._find_types(node.attrib, 2)
            if None in types:
                continue
            k = len(self.nbfix)
            self.nbfix.append((_number(node.attrib["sigma"]), _number(node.attrib["epsilon"])))
            for t in types[0]:
                self.types1[t].add(k)
            for t in types[1]:
                self.types2[t].add(k)

    def pair(self, t1, t2):
        found = (self.types1[t1] & self.types2[t2]) | (self.types2[t1] & self.types1[t2])
        if not found:
            return None
        if len(found) > 1:
            raise FFXMLError(f"Multiple NBFixPair entries match atom types {t1}-{t2}.")
        return self.nbfix[next(iter(found))]


class _UreyBradley:
    def __init__(self):
        self.rows = {True: [], False: []}  # by class, by type

    def parse(self, element):
        for node in element.findall("UreyBradley"):
            a = node.attrib
            is_class = any(k.startswith("class") for k in a)
            if is_class and any(k.startswith("type") for k in a):
                raise FFXMLError("Urey-Bradley terms cannot mix classes and types")
            p = "class" if is_class else "type"
            key = (a[f"{p}1"], a[f"{p}2"], a[f"{p}3"])
            self.rows[is_class].append((key, float(a["k"]), float(a["d"])))


_CHARMM_SCRIPT_TABLES = (
    "residue_impropers", "patch_impropers", "delete_impropers", "residue_order",
    "patch_order", "periodic_improper_types", "harmonic_improper_types",
)  # fmt: skip


def _script_tables(text):
    """(kind, tables) for the scripts boonza knows, read as literals (never executed):
    CHARMM36's improper tables, and Amber lipid21's per-torsion 1-4 scale factors."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    found = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    if (
        set(_CHARMM_SCRIPT_TABLES) <= set(found)
        and "harmonic_force = openmm.CustomTorsionForce" in text
    ):
        return "charmm_impropers", {k: found[k] for k in _CHARMM_SCRIPT_TABLES}
    if {"scee_table", "scnb_table"} <= set(found) and "setExceptionParameters" in text:
        return "scaled_14", {"scee": found["scee_table"], "scnb": found["scnb_table"]}
    return None


_UNSUPPORTED = {
    "RBTorsionForce", "CustomBondForce", "CustomAngleForce", "CustomTorsionForce",
    "CustomNonbondedForce", "CustomGBForce", "CustomHbondForce", "CustomManyParticleForce",
    "GBSAOBCForce", "DrudeForce",
}  # fmt: skip


class OpenMMForcefield:
    """OpenMM XML force field files, read into boonza (see :func:`load_openmm_forcefield`)."""

    def __init__(self):
        self.files: list[str] = []
        self.types: dict[str, _AtomType] = {}
        self.classes: dict[str, set] = {"": set()}
        self.templates: dict[str, _Template] = {}
        self.signatures: dict[tuple, list[_Template]] = {}
        self.patches: dict[str, _Patch] = {}
        self.template_patches: dict[str, set] = {}
        self.bonds = self.angles = self.torsions = self.cmap = None
        self.nonbonded = self.lennard_jones = self.urey_bradley = None
        self.charmm_impropers: dict | None = None
        self.scaled_14: list[dict] = []  # Amber scripts: 1-4 scale factors per torsion types

    def __repr__(self) -> str:
        parts = [f"{len(self.templates)} templates", f"{len(self.patches)} patches"]
        for label, g in (("bonds", self.bonds), ("angles", self.angles)):
            if g is not None:
                parts.append(f"{len(g.rows)} {label}")
        if self.torsions is not None:
            parts.append(
                f"{len(self.torsions.proper)} propers, {len(self.torsions.improper)} impropers"
            )
        if self.cmap is not None:
            parts.append(f"{len(self.cmap.maps)} CMAP maps")
        if self.lennard_jones is not None:
            parts.append(f"{len(self.lennard_jones.nbfix)} NBFIX pairs")
        names = ", ".join(Path(f).name for f in self.files[:3]) + (
            " ..." if len(self.files) > 3 else ""
        )
        return f"<OpenMMForcefield {names}: {'; '.join(parts)}>"

    # -- lookups, as ForceField._findAtomTypes
    def _find_types(self, attrib, num):
        out = []
        for i in range(num):
            suffix = "" if num == 1 else str(i + 1)
            c, t = f"class{suffix}", f"type{suffix}"
            if c in attrib:
                if t in attrib:
                    raise FFXMLError(
                        f"Specified both a type and a class for the same atom: {attrib}"
                    )
                out.append(self.classes.get(attrib[c]))
            elif t in attrib:
                if attrib[t] == "":
                    out.append(self.classes[""])
                elif attrib[t] not in self.types:
                    out.append(None)
                else:
                    out.append({attrib[t]})
            else:
                out.append(None)
        return out

    def _register_template(self, template):
        old = self.templates.get(template.name)
        if old is not None:
            if template.override < old.override:
                return
            if template.override == old.override:
                raise FFXMLError(f"Residue template {template.name} with the same override level "
                                 f"{template.override} already exists.")  # fmt: skip
            self.signatures[_signature(a.anum for a in old.atoms)].remove(old)
        self.templates[template.name] = template
        self.signatures.setdefault(_signature(a.anum for a in template.atoms), []).append(template)

    def _register_template_patch(self, residue, patch, index):
        self.template_patches.setdefault(residue, set()).add((patch, index))


def _data_dirs() -> list[str]:
    try:
        from openmm.app.forcefield import _getDataDirectories

        return list(_getDataDirectories())
    except Exception:  # pragma: no cover - OpenMM is a dependency, but be safe
        return []


def load_openmm_forcefield(*files) -> OpenMMForcefield:
    """Read OpenMM force field XML files, as ``openmm.app.ForceField(*files)`` does.

    Names that are not paths are looked up in OpenMM's data directories, so
    ``load_openmm_forcefield("amber19-all.xml", "amber19/opc.xml")`` works;
    ``<Include>`` files are followed.
    """
    ff = OpenMMForcefield()
    queue = [os.fspath(f) for f in files]
    trees, k = [], 0
    while k < len(queue):
        f = queue[k]
        path = f
        if not os.path.isfile(path):
            for d in _data_dirs():
                if os.path.isfile(os.path.join(d, f)):
                    path = os.path.join(d, f)
                    break
        if not os.path.isfile(path):
            raise FFXMLError(f'Could not locate file "{f}"')
        try:
            tree = etree.parse(path)
        except etree.ParseError as e:
            raise FFXMLError(f'error reading "{path}": {e}') from e
        trees.append(tree)
        ff.files.append(path)
        k += 1
        parent = os.path.dirname(path)
        for inc in tree.getroot().findall("Include"):
            name = inc.attrib["file"]
            joined = os.path.join(parent, name)
            if os.path.isfile(joined):
                name = joined
            if name not in queue:
                queue.append(name)
    roots = [t.getroot() for t in trees]
    for root in roots:
        types = root.find("AtomTypes")
        for node in [] if types is None else types.findall("Type"):
            a = node.attrib
            name = a["name"]
            anum = atomic_number(a["element"]) if a.get("element") else 0
            if name in ff.types:
                old = ff.types[name]
                if old.cls == a["class"] and old.mass == float(a["mass"]) and old.anum == anum:
                    continue
                raise FFXMLError(f"Found multiple definitions for atom type: {name}")
            ff.types[name] = _AtomType(name, a["class"], float(a["mass"]), anum)
            ff.classes.setdefault(a["class"], set()).add(name)
            ff.classes[""].add(name)
    for root in roots:
        residues = root.find("Residues")
        for node in [] if residues is None else residues.findall("Residue"):
            t = _Template(node.attrib["name"])
            if "override" in node.attrib:
                t.override = int(node.attrib["override"])
            if "rigidWater" in node.attrib:
                t.rigid_water = node.attrib["rigidWater"].lower() == "true"
            for atom in node.findall("Atom"):
                params = {
                    k: _number(v) for k, v in atom.attrib.items() if k not in ("name", "type")
                }
                if atom.attrib["name"] in t.index:
                    raise FFXMLError(
                        f"Residue {t.name} contains multiple atoms named {atom.attrib['name']}"
                    )
                typ = atom.attrib["type"]
                if typ not in ff.types:
                    raise FFXMLError(f"Residue {t.name}: unknown atom type {typ}")
                t.add_atom(_TAtom(atom.attrib["name"], typ, ff.types[typ].anum, params))
            for site in node.findall("VirtualSite"):
                t.sites.append(_Site(site.attrib, t.index))
            for bond in node.findall("Bond"):
                if "atomName1" in bond.attrib:
                    t.add_bond(
                        t.atom_index(bond.attrib["atomName1"]),
                        t.atom_index(bond.attrib["atomName2"]),
                    )
                else:
                    t.add_bond(int(bond.attrib["from"]), int(bond.attrib["to"]))
            for bond in node.findall("ExternalBond"):
                if "atomName" in bond.attrib:
                    t.add_external(t.atom_index(bond.attrib["atomName"]))
                else:
                    t.add_external(int(bond.attrib["from"]))
            for c in node.findall("Constraint"):
                a1, a2 = t.atom_index(c.attrib["atomName1"]), t.atom_index(c.attrib["atomName2"])
                t.constraints.append((a1, a2, float(c.attrib["distance"])))
            for p in node.findall("AllowPatch"):
                name = p.attrib["name"]
                if ":" in name:
                    ff._register_template_patch(
                        t.name, name[: name.index(":")], int(name[name.index(":") + 1 :]) - 1
                    )
                else:
                    ff._register_template_patch(t.name, name, 0)
            ff._register_template(t)
    for root in roots:
        patches = root.find("Patches")
        for node in [] if patches is None else patches.findall("Patch"):
            p = _Patch(node.attrib["name"], int(node.attrib.get("residues", 1)))
            for tag, target in (("AddAtom", p.added), ("ChangeAtom", p.changed)):
                for atom in node.findall(tag):
                    params = {
                        k: _number(v) for k, v in atom.attrib.items() if k not in ("name", "type")
                    }
                    if atom.attrib["name"] in p.names:
                        raise FFXMLError(
                            f"Patch {p.name} contains multiple atoms named {atom.attrib['name']}"
                        )
                    p.names.add(atom.attrib["name"])
                    desc = _PatchAtom(atom.attrib["name"])
                    typ = atom.attrib["type"]
                    target[desc.residue].append(_TAtom(desc.name, typ, ff.types[typ].anum, params))
            for atom in node.findall("RemoveAtom"):
                p.names.add(atom.attrib["name"])
                p.deleted.append(_PatchAtom(atom.attrib["name"]))
            for tag, target in (("AddBond", p.added_bonds), ("RemoveBond", p.deleted_bonds)):
                for bond in node.findall(tag):
                    target.append(
                        (_PatchAtom(bond.attrib["atomName1"]), _PatchAtom(bond.attrib["atomName2"]))
                    )
            for tag, target in (
                ("AddExternalBond", p.added_external),
                ("RemoveExternalBond", p.deleted_external),
            ):
                for bond in node.findall(tag):
                    target.append(_PatchAtom(bond.attrib["atomName"]))
            index_of = {a.name: i for i, a in enumerate(p.added[0] + p.changed[0])}
            for site in node.findall("VirtualSite"):
                p.sites[0].append(_Site(site.attrib, index_of))
            for res in node.findall("ApplyToResidue"):
                name = res.attrib["name"]
                if ":" in name:
                    ff._register_template_patch(
                        name[name.index(":") + 1 :], p.name, int(name[: name.index(":")]) - 1
                    )
                else:
                    ff._register_template_patch(name, p.name, 0)
            p.index = len(ff.patches)
            ff.patches[p.name] = p
    for root in roots:
        for child in root:
            tag = child.tag
            if tag == "HarmonicBondForce":
                ff.bonds = ff.bonds or _Bonds(ff)
                ff.bonds.parse(child)
            elif tag == "HarmonicAngleForce":
                ff.angles = ff.angles or _Angles(ff)
                ff.angles.parse(child)
            elif tag == "PeriodicTorsionForce":
                ff.torsions = ff.torsions or _Torsions(ff)
                ff.torsions.parse(child)
            elif tag == "CMAPTorsionForce":
                ff.cmap = ff.cmap or _Cmap(ff)
                ff.cmap.parse(child)
            elif tag == "NonbondedForce":
                c14, lj14 = float(child.attrib["coulomb14scale"]), float(child.attrib["lj14scale"])
                if ff.nonbonded is None:
                    ff.nonbonded = _Nonbonded(ff, c14, lj14)
                elif abs(ff.nonbonded.c14 - c14) > 1e-5 or abs(ff.nonbonded.lj14 - lj14) > 1e-5:
                    raise FFXMLError("Found multiple NonbondedForce tags with different 1-4 scales")
                ff.nonbonded.params.parse(child)
            elif tag == "LennardJonesForce":
                lj14 = float(child.attrib["lj14scale"])
                if ff.lennard_jones is None:
                    ff.lennard_jones = _LennardJones(ff, lj14)
                elif abs(ff.lennard_jones.lj14 - lj14) > 1e-5:
                    raise FFXMLError(
                        "Found multiple LennardJonesForce tags with different 1-4 scales"
                    )
                ff.lennard_jones.parse(child)
            elif tag == "AmoebaUreyBradleyForce":
                ff.urey_bradley = ff.urey_bradley or _UreyBradley()
                ff.urey_bradley.parse(child)
            elif tag == "Script":
                found = _script_tables(child.text or "")
                if found is None:
                    raise FFXMLError("this force field has a <Script> that boonza cannot interpret")
                if found[0] == "charmm_impropers":
                    ff.charmm_impropers = found[1]
                else:
                    ff.scaled_14.append(found[1])
            elif tag in _UNSUPPORTED or (
                tag.endswith("Force") and tag.startswith(("Amoeba", "Hippo"))
            ):
                if len(child):
                    raise FFXMLError(f"<{tag}> is not supported")
    return ff


# ---- template matching (OpenMM's compiled.matchResidueToTemplate) -------------------------------


def _match(atoms, anum, names, bonded, template, ignore_external, ignore_extra):
    """Template atom index for each atom of ``atoms``, or None."""
    if ignore_extra:
        atoms = [a for a in atoms if anum[a] > 0]
        keep = [k for k, a in enumerate(template.atoms) if a.anum > 0]
        where = {k: m for m, k in enumerate(keep)}
        t_atoms = [template.atoms[k] for k in keep]
        t_bonded = [
            [where[j] for j in template.atoms[k].bonded if template.atoms[j].anum > 0] for k in keep
        ]
    else:
        keep = list(range(len(template.atoms)))
        t_atoms = template.atoms
        t_bonded = [a.bonded for a in template.atoms]
    n = len(atoms)
    if n != len(t_atoms):
        return None
    local = {a: i for i, a in enumerate(atoms)}
    bonded_to = [[local[x] for x in bonded[a] if x in local] for a in atoms]
    external = [
        0 if ignore_external else sum(1 for x in bonded[a] if x not in local) for a in atoms
    ]
    if Counter((anum[a], len(bonded_to[i]), external[i]) for i, a in enumerate(atoms)) != Counter(
            (ta.anum, len(t_bonded[j]), 0 if ignore_external else ta.external)
            for j, ta in enumerate(t_atoms)):  # fmt: skip
        return None
    candidates = [[] for _ in range(n)]
    for i, a in enumerate(atoms):
        exact = anum[a] == 0 and any(ta.anum == 0 and ta.name == names[a] for ta in t_atoms)
        for j, ta in enumerate(t_atoms):
            if (ta.anum != 0 and ta.anum != anum[a]) or (exact and ta.name != names[a]):
                continue
            if len(t_bonded[j]) != len(bonded_to[i]):
                continue
            if not ignore_external and ta.external != external[i]:
                continue
            candidates[i].append(j)
    order, todo, in_heap, heap = [], set(range(n)), set(), []
    while todo:
        if not in_heap:
            fewest = n + 1
            for i in sorted(todo):
                if len(candidates[i]) < fewest:
                    nxt, fewest = i, len(candidates[i])
        else:
            nxt = heapq.heappop(heap)[1]
            in_heap.remove(nxt)
        order.append(nxt)
        todo.remove(nxt)
        for i in bonded_to[nxt]:
            if i in todo and i not in in_heap:
                in_heap.add(i)
                heapq.heappush(heap, (len(candidates[i]), i))
    inverse = [0] * n
    for k, i in enumerate(order):
        inverse[i] = k
    bonded_to = [[inverse[j] for j in bonded_to[i]] for i in order]
    candidates = [candidates[i] for i in order]
    cand_sets = [set(c) for c in candidates]
    matches, used = [0] * n, [False] * n

    def search(pos) -> bool:
        if pos == n:
            return True
        options = candidates[pos]
        for b in bonded_to[pos]:
            if b < pos:
                options = t_bonded[matches[b]]
                break
        for i in options:
            if used[i] or i not in cand_sets[pos]:
                continue
            if all(b > pos or matches[b] in t_bonded[i] for b in bonded_to[pos]):
                matches[pos], used[i] = i, True
                if search(pos + 1):
                    return True
                used[i] = False
        return False

    if not search(0):
        return None
    return [keep[matches[inverse[i]]] for i in range(n)]


# ---- parameterization ---------------------------------------------------------------------------


class _Rows:
    """Terms of one output table with de-duplicated parameter rows."""

    def __init__(self, natoms, props):
        self.natoms, self.props = natoms, props
        self.atoms, self.pids, self.rows, self.index = [], [], [], {}
        self.flags = []

    def add(self, atoms, values, constrained=0):
        key = tuple(values)
        p = self.index.get(key)
        if p is None:
            p = self.index[key] = len(self.rows)
            self.rows.append(key)
        self.atoms.append(tuple(atoms))
        self.pids.append(p)
        self.flags.append(constrained)


class _Run:
    def __init__(self, system: System, ff: OpenMMForcefield, residue_templates, ignore_external):
        self.ff = ff
        self.ignore_external = ignore_external
        s = system.clone(structure_only=True)
        s.nonbonded_info = NonbondedInfo()
        self.s = s
        self.residue_templates = residue_templates or {}
        self._load_arrays()

    def _load_arrays(self):
        s = self.s
        self.n = s.natoms
        self.anum = s.atoms["anum"].tolist()
        self.names = s.atoms["name"].tolist()
        self.residue = s.atoms["residue"].tolist()
        self.bond_list = list(zip(s.bonds["i"].tolist(), s.bonds["j"].tolist(), strict=True))
        nb = [set() for _ in range(self.n)]
        for i, j in self.bond_list:
            nb[i].add(j)
            nb[j].add(i)
        self.bonded = [sorted(x) for x in nb]
        self.res_atoms = [s.residue_atoms(r).tolist() for r in range(s.nresidues)]

    # -- matching ---------------------------------------------------------------------------------
    def _template_matches(self, atoms, signatures, cache):
        anum, names, bonded = self.anum, self.names, self.bonded
        local = {a: k for k, a in enumerate(atoms)}
        key = (
            id(signatures),
            tuple((anum[a], tuple(local.get(o, -1) for o in bonded[a])) for a in atoms),
        )
        if key in cache:
            t, m = cache[key]
            return t, (None if m is None else list(m))
        sig = _signature(anum[a] for a in atoms)
        found = []
        for t in signatures.get(sig, []):
            m = _match(atoms, anum, names, bonded, t, self.ignore_external, True)
            if m is not None:
                found.append((t, m))
        result = (None, None)
        if len(found) == 1:
            result = found[0]
        elif len(found) > 1:
            t1, m1 = found[0]
            for t2, m2 in found[1:]:
                if not t1.identical_parameters(t2, m1, m2):
                    r = self.residue[atoms[0]]
                    raise FFXMLError(f"Multiple non-identical matching templates found "
                                     f"for residue {r} "
                                     f"({self.s.residues['name'][r]}): "
                                     f"{', '.join(x.name for x, _ in found)}.")  # fmt: skip
            result = found[0]
        cache[key] = result
        return result[0], (None if result[1] is None else list(result[1]))

    def match_all(self):
        ff = self.ff
        self.template_of: dict[
            int, _Template
        ] = {}  # residues matched as a whole (for the CHARMM script)
        self.matched: dict[int, tuple[_Template, list[int]]] = {}
        cache: dict = {}
        unmatched = []
        for r, atoms in enumerate(self.res_atoms):
            if r in self.residue_templates:
                t = ff.templates[self.residue_templates[r]]
                m = _match(atoms, self.anum, self.names, self.bonded, t, self.ignore_external, True)
                if m is None:
                    raise FFXMLError(f"User-supplied template {t.name} does not match residue {r}")
            else:
                t, m = self._template_matches(atoms, ff.signatures, cache)
            if m is None:
                unmatched.append(r)
            else:
                self.matched[r] = (t, m)
                self.template_of[r] = t
        if unmatched:
            unmatched = self._patches(unmatched)
        if unmatched:
            r = unmatched[0]
            name = self.s.residues["name"][r]
            hint = ""
            same = [t for t in ff.templates.values() if t.name == name]
            if same:
                hint = (f" A template {name} exists but has {len(same[0].atoms)} atoms or "
                        f"different bonds (the residue has "
                        f"{len(self.res_atoms[r])} atoms).")  # fmt: skip
            raise FFXMLError(f"No template found for residue {r} ({name}).{hint}")

    def _patches(self, residues):
        ff = self.ff
        signatures, patched = {}, {}
        wanted = {_signature(self.anum[a] for a in self.res_atoms[r]) for r in residues}
        in_multi = {t for t, entries in ff.template_patches.items() for p, _ in entries
                    if p in ff.patches and ff.patches[p].nres > 1}  # fmt: skip
        for name, template in ff.templates.items():
            if name not in ff.template_patches:
                continue
            ps = [ff.patches[p] for p, _ in ff.template_patches[name]
                  if p in ff.patches and ff.patches[p].nres == 1]  # fmt: skip
            ps.sort(key=lambda p: p.index)
            if not ps:
                continue
            new: list[_Template] = []
            if name in in_multi:  # multi-residue patches take every patched version, as in OpenMM
                _single_patches(template, ps, 0, new, set())
                patched[name] = new
            else:
                new = _lazy_patches(template, ps, wanted)
            for t in new:
                signatures.setdefault(_signature(a.anum for a in t.atoms), []).append(t)
        unmatched, cache = [], {}
        for r in residues:
            t, m = self._template_matches(self.res_atoms[r], signatures, cache)
            if m is None:
                unmatched.append(r)
            else:
                self.matched[r] = (t, m)
                self.template_of[r] = t
        if not unmatched:
            return []
        multi, largest = {}, 0
        for p in ff.patches.values():
            if p.nres > 1:
                multi[p.name] = [[] for _ in range(p.nres)]
                largest = max(largest, p.nres)
        if not largest:
            return unmatched
        for tname, entries in ff.template_patches.items():
            for pname, k in entries:
                if pname in multi and tname in ff.templates:
                    multi[pname][k].append(ff.templates[tname])
                    multi[pname][k] += patched.get(tname, [])
        left = set(unmatched)
        bonds = set()
        for i, j in self.bond_list:
            ri, rj = self.residue[i], self.residue[j]
            if ri != rj and ri in left and rj in left:
                bonds.add(tuple(sorted((ri, rj))))
        size, clusters = 2, set(bonds)
        while size <= largest:
            for pname, candidates in multi.items():
                patch = ff.patches[pname]
                if patch.nres != size:
                    continue
                for cluster in self._multi_patch(clusters, patch, candidates):
                    left -= set(cluster)
                bonds = {b for b in bonds if b[0] in left and b[1] in left}
            larger = set()
            for cluster in clusters:
                for a, b in bonds:
                    if a in cluster and b not in cluster:
                        larger.add(tuple(sorted((*cluster, b))))
                    elif b in cluster and a not in cluster:
                        larger.add(tuple(sorted((*cluster, a))))
            if not larger:
                break
            clusters, size = larger, size + 1
        return [r for r in unmatched if r in left]

    def _multi_patch(self, clusters, patch, candidates):
        matched_clusters = []
        for choice in itertools.product(*candidates):
            try:
                templates = patch.create(list(choice))
            except FFXMLError:
                continue
            newly = []
            for cluster in list(clusters):
                for residues in itertools.permutations(cluster):
                    found = []
                    for r, t in zip(residues, templates, strict=True):
                        m = _match(self.res_atoms[r], self.anum, self.names, self.bonded, t,
                                   self.ignore_external, True)  # fmt: skip
                        if m is None:
                            found = None
                            break
                        found.append(m)
                    if found is None:
                        continue
                    ok = True
                    for a1, a2 in patch.added_bonds:
                        if a1.residue == a2.residue:
                            continue
                        t1, t2 = templates[a1.residue], templates[a2.residue]
                        m1, m2 = found[a1.residue], found[a2.residue]
                        i1 = next(i for i in range(len(m1)) if t1.atoms[m1[i]].name == a1.name)
                        i2 = next(i for i in range(len(m2)) if t2.atoms[m2[i]].name == a2.name)
                        x = self.res_atoms[residues[a1.residue]][i1]
                        y = self.res_atoms[residues[a2.residue]][i2]
                        ok &= y in self.bonded[x]
                    if ok:
                        for r, t, m in zip(residues, templates, found, strict=True):
                            self.matched[r] = (t, m)
                        newly.append(cluster)
                        break
            matched_clusters += newly
            for c in newly:
                clusters.discard(c)
        return matched_clusters

    # -- extra particles --------------------------------------------------------------------------
    def add_extra_particles(self):
        """Add template particles without an element that the structure lacks."""
        s, pos = self.s, self.s.positions
        new_atoms = []  # (residue, template atom, position)
        for r, (t, m) in self.matched.items():
            present = set(m)
            missing = [k for k, a in enumerate(t.atoms) if a.anum == 0 and k not in present]
            if not missing:
                continue
            where = {k: pos[a] for a, k in zip(self.res_atoms[r], m, strict=True)}
            for k in missing:
                site = next((x for x in t.sites if x.index == k), None)
                p = None
                if site is not None:
                    try:
                        p = _site_position(site, where)
                    except KeyError:
                        p = None
                if p is None:
                    raise FFXMLError(
                        f"cannot place extra particle {t.atoms[k].name} of residue {r}"
                    )
                where[k] = p
                new_atoms.append((r, k, p))
        if not new_atoms:
            return
        start = s.natoms
        names = np.array([self.matched[r][0].atoms[k].name for r, k, _ in new_atoms], dtype=STR)
        s.add_atoms(len(new_atoms), residue=np.array([r for r, _, _ in new_atoms], np.int64),
                    name=names, anum=np.zeros(len(new_atoms), np.int64),
                    pos=np.array([p for _, _, p in new_atoms]))  # fmt: skip
        for idx, (r, k, _) in enumerate(new_atoms):
            t, m = self.matched[r]
            self.matched[r] = (t, [*m, k])
            self.res_atoms[r] = [*self.res_atoms[r], start + idx]
        for r, k, _ in new_atoms:
            t, m = self.matched[r]
            atoms = self.res_atoms[r]
            index_of = dict(zip(m, atoms, strict=True))
            for i, j in t.bonds:
                if k in (i, j) and i in index_of and j in index_of:
                    s.add_bond(index_of[i], index_of[j])
        order = [a for r in range(s.nresidues) for a in self.res_atoms[r]]
        amap = s.reorder_atoms(np.array(order, np.int64))
        self.res_atoms = [[int(amap[a]) for a in atoms] for atoms in self.res_atoms]
        self._load_arrays_keep()

    def _load_arrays_keep(self):
        res_atoms = self.res_atoms
        self._load_arrays()
        self.res_atoms = res_atoms

    # -- the rest of createSystem -----------------------------------------------------------------
    def build(self, constraints, rigid_water, hydrogen_mass) -> System:
        ff, s, n = self.ff, self.s, self.n
        self.atom_type = [None] * n
        self.atom_params = [None] * n
        self.tindex = [None] * n
        sites = {}  # particle -> (site, parents, exclude-with atom)
        for r, (t, m) in self.matched.items():
            atoms = self.res_atoms[r]
            index_of = dict(zip(m, atoms, strict=True))
            for a, k in zip(atoms, m, strict=True):
                self.atom_type[a] = t.atoms[k].type
                self.atom_params[a] = t.atoms[k].params
                self.tindex[a] = k
                for site in t.sites:
                    if site.index == k:
                        sites[a] = (
                            site,
                            [index_of[i] for i in site.atoms],
                            index_of[site.exclude_with],
                        )
        resname = s.residues["name"].tolist()
        rigid = [False] * s.nresidues
        for r, t in self.template_of.items():
            if resname[r] == "HOH":
                rigid[r] = t.rigid_water if rigid_water is None else bool(rigid_water)
        mass = [ff.types[t].mass for t in self.atom_type]
        if hydrogen_mass is not None:
            for i, j in self.bond_list:
                if self.anum[i] == 1:
                    i, j = j, i
                if self.anum[j] == 1 and self.anum[i] not in (1, 0) and not rigid[self.residue[j]]:
                    transfer = hydrogen_mass - mass[j]
                    mass[j] = hydrogen_mass
                    mass[i] -= transfer
        # angles, propers and impropers exactly as createSystem lists them
        bonded = self.bonded
        angles = set()
        for i, j in self.bond_list:
            for a in bonded[i]:
                if a != j:
                    angles.add((a, i, j) if a < j else (j, i, a))
            for a in bonded[j]:
                if a != i:
                    angles.add((i, j, a) if a > i else (a, j, i))
        self.angles = sorted(angles)
        propers = set()
        for a0, a1, a2 in self.angles:
            for a in bonded[a0]:
                if a not in (a0, a1, a2):
                    propers.add((a, a0, a1, a2) if a < a2 else (a2, a1, a0, a))
            for a in bonded[a2]:
                if a not in (a0, a1, a2):
                    propers.add((a0, a1, a2, a) if a > a0 else (a, a2, a1, a0))
        self.propers = sorted(propers)
        self.impropers = [(a, *sub) for a in range(n) if len(bonded[a]) > 2
                          for sub in itertools.combinations(bonded[a], 3)]  # fmt: skip
        # constraints
        bond_constrained = []
        for i, j in self.bond_list:
            c = False
            if constraints in ("allbonds", "hangles"):
                c = True
            elif constraints == "hbonds":
                c = self.anum[i] == 1 or self.anum[j] == 1
            if rigid[self.residue[i]] and rigid[self.residue[j]]:
                c = True
            elif self.residue[i] in self.matched:
                t = self.matched[self.residue[i]][0]
                ti, tj = self.tindex[i], self.tindex[j]
                if any(sorted((ti, tj)) == sorted((x, y)) for x, y, _ in t.constraints):
                    c = True
            bond_constrained.append(c)
        angle_constrained = []
        for a0, a1, a2 in self.angles:
            c = False
            if constraints == "hangles":
                nh = (self.anum[a0] == 1) + (self.anum[a2] == 1)
                c = nh == 2 or (nh == 1 and self.anum[a1] == 8)
            if rigid[self.residue[a0]] and rigid[self.residue[a1]] and rigid[self.residue[a2]]:
                c = True
            angle_constrained.append(c)
        self.cons: dict[tuple[int, int], float] = {}
        tables: dict[str, _Rows] = {}
        self._bonds(tables, bond_constrained)
        self._urey_bradley(tables, angle_constrained)
        self._angles(tables, angle_constrained)
        self._torsions(tables)
        self._cmap(tables)
        charges = self._nonbonded(tables, sites)
        if ff.charmm_impropers is not None:
            self._charmm_impropers(tables)
        # template constraints override the others
        for r, t in self.template_of.items():
            if not t.constraints:
                continue
            by_index = dict(zip(self.matched[r][1], self.res_atoms[r], strict=True))
            for i, j, d in t.constraints:
                a1, a2 = by_index[i], by_index[j]
                self.cons[(min(a1, a2), max(a1, a2))] = d / NM
        return self._emit(tables, mass, charges, sites)

    def _key(self, i):
        return self.atom_type[i]

    def _bonds(self, tables, constrained):
        g = self.ff.bonds
        self.bond_length = [None] * len(self.bond_list)
        if g is None:
            return
        rows = tables.setdefault("stretch_harm", _Rows(2, ("r0", "fc")))
        for b, (i, j) in enumerate(self.bond_list):
            t1, t2 = self.atom_type[i], self.atom_type[j]
            for k in sorted(g.by_type[t1]):
                ty1, ty2, length, kk = g.rows[k]
                if (t1 in ty1 and t2 in ty2) or (t1 in ty2 and t2 in ty1):
                    self.bond_length[b] = length
                    if constrained[b]:
                        self.cons.setdefault((min(i, j), max(i, j)), length / NM)
                    if kk != 0:
                        rows.add((i, j), (length / NM, kk / 2 / KCAL * NM**2), int(constrained[b]))
                    break

    def _urey_bradley(self, tables, constrained):
        g = self.ff.urey_bradley
        if g is None:
            return
        rows = tables.setdefault("stretch_harm", _Rows(2, ("r0", "fc")))
        classes = [self.ff.types[t].cls for t in self.atom_type]
        for is_class, labels in ((True, classes), (False, self.atom_type)):
            for (a1, a2, a3), c in zip(self.angles, constrained, strict=True):
                if c:
                    continue
                key = (labels[a1], labels[a2], labels[a3])
                for row, k, d in g.rows[is_class]:
                    if row == key or row == key[::-1]:
                        if k != 0:
                            rows.add((a1, a3), (d / NM, 2 * k / 2 / KCAL * NM**2))
                        break

    def _angles(self, tables, constrained):
        g = self.ff.angles
        if g is None:
            return
        rows = tables.setdefault("angle_harm", _Rows(3, ("theta0", "fc")))
        bond_index = {}
        atom_bonds = defaultdict(list)
        for b, (i, j) in enumerate(self.bond_list):
            atom_bonds[i].append(b)
            atom_bonds[j].append(b)
            bond_index[(i, j)] = b
        for (a0, a1, a2), c in zip(self.angles, constrained, strict=True):
            t1, t2, t3 = self.atom_type[a0], self.atom_type[a1], self.atom_type[a2]
            for k in g.by_type2[t2]:
                ty1, ty2, ty3, theta, kk = g.rows[k]
                if (t1 in ty1 and t2 in ty2 and t3 in ty3) or (
                    t1 in ty3 and t2 in ty2 and t3 in ty1
                ):
                    if c:
                        b1 = b2 = None
                        for b in atom_bonds[a1]:
                            x, y = self.bond_list[b]
                            if x == a0 or y == a0:
                                b1 = b
                            elif x == a2 or y == a2:
                                b2 = b
                        if b1 is not None and b2 is not None:
                            l1, l2 = self.bond_length[b1], self.bond_length[b2]
                            if l1 is not None and l2 is not None:
                                d = math.sqrt(l1 * l1 + l2 * l2 - 2 * l1 * l2 * math.cos(theta))
                                self.cons.setdefault((min(a0, a2), max(a0, a2)), d / NM)
                    if kk != 0:
                        rows.add((a0, a1, a2), (math.degrees(theta), kk / 2 / KCAL), int(c))
                    break

    def _torsion_row(self, rows, atoms, per, phase, k):
        kc = k / KCAL
        rows.add(atoms, (math.degrees(phase), kc, *[kc if m == per else 0.0 for m in range(1, 7)]))

    def _torsions(self, tables):
        g = self.ff.torsions
        if g is None:
            return
        rows = tables.setdefault("dihedral_trig", _Rows(4, ("phi0", *[f"fc{k}" for k in range(7)])))
        wild = self.ff.classes[""]
        cache = {}
        for torsion in self.propers:
            types = tuple(self.atom_type[a] for a in torsion)
            sig = frozenset((types, types[::-1]))
            match = cache.get(sig)
            if match == -1:
                continue
            if match is None:
                t1, t2, t3, t4 = types
                for k in sorted(g.by_type[t2]):
                    td = g.proper[k]
                    y1, y2, y3, y4 = td.types
                    if (t2 in y2 and t3 in y3 and t4 in y4 and t1 in y1) or \
                            (t2 in y3 and t3 in y2 and t4 in y1 and t1 in y4):  # fmt: skip
                        has_wild = any(y is wild for y in td.types)
                        if match is None or not has_wild:
                            match = td
                        if not has_wild:
                            break
                cache[sig] = -1 if match is None else match
            if match is not None and match != -1:
                for per, phase, k in match.terms:
                    if k != 0:
                        self._torsion_row(rows, torsion, per, phase, k)
        icache = {}
        for torsion in self.impropers:
            sig = tuple(self.atom_type[a] for a in torsion)
            hit = icache.get(sig)
            if hit == -1:
                continue
            if hit is not None:
                order, td = hit
                match = (*(torsion[i] for i in order), td)
            else:
                match = self._match_improper(torsion, g.improper)
                if match is None:
                    icache[sig] = -1
                    continue
                icache[sig] = (tuple(torsion.index(a) for a in match[:4]), match[4])
            *atoms, td = match
            for per, phase, k in td.terms:
                if k != 0:
                    self._torsion_row(rows, atoms, per, phase, k)

    def _match_improper(self, torsion, impropers):
        """OpenMM's _matchImproper."""
        types = [self.atom_type[a] for a in torsion]
        wild = self.ff.classes[""]
        anum, res = self.anum, self.residue
        ti = self.tindex

        def later(x, y):
            return res[x] > res[y] or (res[x] == res[y] and ti[x] > ti[y])

        match = None
        for td in impropers:
            y1, y2, y3, y4 = td.types
            has_wild = any(y is wild for y in td.types)
            if match is not None and has_wild:
                continue
            if types[0] not in y1:
                continue
            for t2, t3, t4 in itertools.permutations(((types[1], 1), (types[2], 2), (types[3], 3))):
                if not (t2[0] in y2 and t3[0] in y3 and t4[0] in y4):
                    continue
                if td.ordering == "default" or (td.ordering == "charmm" and has_wild):
                    a1, a2 = torsion[t2[1]], torsion[t3[1]]
                    e1, e2 = anum[a1], anum[a2]
                    if e1 == e2 and a1 > a2:
                        a1, a2 = a2, a1
                    elif e1 != 6 and (e2 == 6 or _mass_of(e1) < _mass_of(e2)):
                        a1, a2 = a2, a1
                    match = (a1, a2, torsion[0], torsion[t4[1]], td)
                elif td.ordering == "charmm":
                    match = (torsion[0], torsion[t2[1]], torsion[t3[1]], torsion[t4[1]], td)
                else:  # amber
                    a2, a3, a4 = torsion[t2[1]], torsion[t3[1]], torsion[t4[1]]
                    if not has_wild:
                        if t2[0] == t4[0] and later(a2, a4):
                            a2, a4 = a4, a2
                        if t3[0] == t4[0] and later(a3, a4):
                            a3, a4 = a4, a3
                        if t2[0] == t3[0] and later(a2, a3):
                            a2, a3 = a3, a2
                    else:
                        if anum[a2] == anum[a4] and later(a2, a4):
                            a2, a4 = a4, a2
                        if anum[a3] == anum[a4] and later(a3, a4):
                            a3, a4 = a4, a3
                        if later(a2, a3):
                            a2, a3 = a3, a2
                    match = (a2, a3, torsion[0], a4, td)
                break
        return match

    def _cmap(self, tables):
        g = self.ff.cmap
        if g is None:
            return
        self.cmap_rows = rows = tables.setdefault("torsiontorsion_cmap", _Rows(8, ("cmapid",)))
        bonded = self.bonded
        unique = set()
        for t in self.propers:
            for a in bonded[t[0]]:
                if a != t[1]:
                    unique.add((a, *t))
            for a in bonded[t[3]]:
                if a != t[2]:
                    unique.add((*t, a))
        wild = self.ff.classes[""]
        for tor in sorted(unique):
            types = [self.atom_type[a] for a in tor]
            match = None
            for td_types, m in g.torsions:
                forward = all(t in y for t, y in zip(types, td_types, strict=True))
                if forward or all(t in y for t, y in zip(types[::-1], td_types, strict=True)):
                    has_wild = any(y is wild for y in td_types)
                    if match is None or not has_wild:
                        match = m
                    if not has_wild:
                        break
            if match is not None:
                rows.add(
                    (tor[0], tor[1], tor[2], tor[3], tor[1], tor[2], tor[3], tor[4]),
                    (f"cmap{match + 1}",),
                )

    def _nonbonded(self, tables, sites):
        ff, n = self.ff, self.n
        charges = np.zeros(n)
        sig = np.full(n, 1.0)
        eps = np.zeros(n)
        if ff.nonbonded is not None:
            for a in range(n):
                q, sg, ep = ff.nonbonded.params.values(self.atom_type[a], self.atom_params[a])
                charges[a], sig[a], eps[a] = q, sg, ep
        # bonds used for exclusions: bonds, virtual sites that exclude with a parent
        exclude_with = defaultdict(list)
        for a, (_site, _parents, ew) in sites.items():
            exclude_with[ew].append(a)
        pairs = list(self.bond_list)
        for a1, a2 in list(pairs):
            for c1 in exclude_with[a1]:
                pairs.append((c1, a2))
                for c2 in exclude_with[a2]:
                    pairs.append((c1, c2))
            for c2 in exclude_with[a2]:
                pairs.append((a1, c2))
        for a in range(n):
            for c in exclude_with[a]:
                pairs.append((c, a))
        graph = [set() for _ in range(n)]
        for i, j in pairs:
            if i != j:
                graph[i].add(j)
                graph[j].add(i)
        within2, only3 = set(), set()
        for a in range(n):
            d1 = graph[a]
            d2 = set().union(*(graph[x] for x in d1)) if d1 else set()
            d3 = set().union(*(graph[x] for x in d2)) if d2 else set()
            for b in d1 | d2:
                if b > a:
                    within2.add((a, b))
            for b in d3 - d1 - d2 - {a}:
                if b > a:
                    only3.add((a, b))
        self.exclusions = sorted(within2 | only3)
        pair_rows = tables.setdefault("pair_12_6_es", _Rows(2, ("aij", "bij", "qij")))
        # Amber scripts (lipid21) set SCEE/SCNB for 1-4 pairs of given torsion types
        scaled = {}
        for tab in ff.scaled_14:
            for a1, a2, a3, a4 in self.propers:
                key = (
                    self.atom_type[a1],
                    self.atom_type[a2],
                    self.atom_type[a3],
                    self.atom_type[a4],
                )
                pair = (min(a1, a4), max(a1, a4))
                if key in tab["scee"]:
                    scaled.setdefault(pair, [1.2, 2.0])[0] = tab["scee"][key]
                if key in tab["scnb"]:
                    scaled.setdefault(pair, [1.2, 2.0])[1] = tab["scnb"][key]
        if ff.nonbonded is not None:
            c14, lj14 = ff.nonbonded.c14, ff.nonbonded.lj14
            for a, b in sorted(only3):
                qq = c14 * charges[a] * charges[b]
                s_ = 0.5 * (sig[a] + sig[b]) / NM
                e_ = lj14 * math.sqrt(eps[a] * eps[b]) / KCAL
                if (qq != 0 or e_ != 0) and (a, b) in scaled:
                    scee, scnb = scaled[(a, b)]
                    qq = charges[a] * charges[b] / scee
                    e_ = math.sqrt(eps[a] * eps[b]) / scnb / KCAL
                if qq != 0 or e_ != 0:
                    pair_rows.add((a, b), (4 * e_ * s_**12, 4 * e_ * s_**6, qq))
        lj = ff.lennard_jones
        if lj is not None:
            if (eps != 0).any():
                raise FFXMLError("Lennard-Jones in both NonbondedForce and LennardJonesForce")
            self.lj_types = [self.atom_type[a] for a in range(n)]
            if lj.lj14 != 0:
                for a, b in sorted(only3):
                    fix = lj.pair(self.atom_type[a], self.atom_type[b])
                    if fix is not None:
                        s_, e_ = fix
                    else:
                        v1 = lj.params.values(self.atom_type[a], self.atom_params[a])
                        v2 = lj.params.values(self.atom_type[b], self.atom_params[b])
                        x1, x2 = (
                            lj.params.extra[self.atom_type[a]],
                            lj.params.extra[self.atom_type[b]],
                        )
                        s1 = float(x1["sigma14"]) if "sigma14" in x1 else v1[0]
                        s2 = float(x2["sigma14"]) if "sigma14" in x2 else v2[0]
                        e1 = float(x1["epsilon14"]) if "epsilon14" in x1 else v1[1]
                        e2 = float(x2["epsilon14"]) if "epsilon14" in x2 else v2[1]
                        s_, e_ = 0.5 * (s1 + s2), math.sqrt(e1 * e2)
                    s_, e_ = s_ / NM, lj.lj14 * e_ / KCAL
                    pair_rows.add((a, b), (4 * e_ * s_**12, 4 * e_ * s_**6, 0.0))
        else:
            self.lj_types = None
        self.vdw = (sig / NM, eps / KCAL)
        return charges

    def _charmm_impropers(self, tables):
        """CHARMM36's improper script, applied from its tables."""
        tab = self.ff.charmm_impropers
        residue_impropers, patch_impropers = tab["residue_impropers"], tab["patch_impropers"]
        delete_impropers = tab["delete_impropers"]
        periodic_types, harmonic_types = (
            tab["periodic_improper_types"],
            tab["harmonic_improper_types"],
        )
        cls = [self.ff.types[t].cls for t in self.atom_type]
        residue_data = []
        for r in range(self.s.nresidues):
            templates = set()
            atoms = self.res_atoms[r]
            if r not in self.template_of:
                names, skip = [], False
                for a in atoms:
                    raw = self.atom_type[a]
                    prefix = ""
                    if raw.startswith("Drude-"):
                        raw, prefix = raw[len("Drude-") :], "D"
                    parts = raw.rsplit("-", maxsplit=1)
                    if len(parts) < 2:
                        skip = True
                        break
                    templates.add(parts[0])
                    names.append(prefix + parts[1])
                if skip:
                    residue_data.append(({}, [], []))
                    continue
            else:
                t = self.template_of[r]
                for k, part in enumerate(t.name.split("-")):
                    templates.add(part.rsplit("_", maxsplit=1)[0] if k else part)
                names = [t.atoms[self.tindex[a]].name for a in atoms]
            res_t = sorted(
                (x for x in templates if x in residue_impropers), key=tab["residue_order"].get
            )
            patch_t = sorted(
                (x for x in templates if x in patch_impropers), key=tab["patch_order"].get
            )
            data = {nm: (a, cls[a]) for a, nm in zip(atoms, names, strict=True)}
            if len(data) != len(names):
                raise FFXMLError(f"CHARMM: atom name collision in residue with index {r}")
            residue_data.append((data, res_t, patch_t))
        periodic, harmonic = [], []
        for r, (_, res_t, patch_t) in enumerate(residue_data):
            impropers = []
            for x in res_t:
                for imp in residue_impropers[x]:
                    if imp not in impropers:
                        impropers.append(imp)
            for x in patch_t:
                for imp in delete_impropers.get(x, []):
                    if imp in impropers:
                        impropers.remove(imp)
                for imp in patch_impropers.get(x, []):
                    if imp not in impropers:
                        impropers.append(imp)
            for imp in impropers:
                idx, classes = [], []
                for offset, name in imp:
                    k = r + offset
                    if not 0 <= k < len(residue_data) or name not in residue_data[k][0]:
                        idx = None
                        break
                    a, c = residue_data[k][0][name]
                    idx.append(a)
                    classes.append(c)
                if idx is None:
                    continue
                b1, b4 = self.bonded[idx[0]], self.bonded[idx[3]]
                if not ((idx[1] in b1 and idx[2] in b1 and idx[3] in b1)
                        or (idx[0] in b4 and idx[1] in b4 and idx[2] in b4)):  # fmt: skip
                    continue
                per = _find_improper(periodic_types, classes)
                har = _find_improper(harmonic_types, classes)
                if per is None and har is None:
                    raise FFXMLError(
                        f"CHARMM: neither periodic nor harmonic improper found for {classes}"
                    )
                if per is not None and har is not None:
                    raise FFXMLError(
                        f"CHARMM: both periodic and harmonic impropers found for {classes}"
                    )
                if per is not None:
                    periodic.append((idx, per))
                else:
                    harmonic.append((idx, har))
        if periodic:
            rows = tables.setdefault(
                "dihedral_trig", _Rows(4, ("phi0", *[f"fc{k}" for k in range(7)]))
            )
            for idx, (p, phase, k) in periodic:
                self._torsion_row(rows, idx, p, phase, k)
        if harmonic:
            rows = tables.setdefault("improper_harm", _Rows(4, ("phi0", "fc")))
            for idx, params in harmonic:
                k, theta0 = params[0]
                rows.add(idx, (math.degrees(theta0), k / KCAL))

    # -- output -----------------------------------------------------------------------------------
    def _emit(self, tables, mass, charges, sites) -> System:
        from .omm import _constraint_tables

        s = self.s
        s.atoms["mass"] = np.array(mass, np.float64)
        s.atoms["charge"] = charges
        for name, rows in tables.items():
            if not rows.atoms:
                continue
            t = s.add_table_from_schema(name)
            if rows.props == ("cmapid",):
                pids = t.params.add_params(
                    len(rows.rows), cmapid=np.array([r[0] for r in rows.rows], dtype=STR)
                )
            else:
                arr = np.array(rows.rows, np.float64).reshape(len(rows.rows), len(rows.props))
                pids = t.params.add_params(
                    len(rows.rows), **{p: arr[:, k] for k, p in enumerate(rows.props)}
                )
            extra = (
                {"constrained": np.array(rows.flags, np.int64)}
                if "constrained" in t.term_props
                else {}
            )
            t.add_terms(
                np.array(rows.atoms, np.int64), params=pids[np.array(rows.pids, np.int64)], **extra
            )
        if self.ff.cmap is not None and "torsiontorsion_cmap" in s.table_names:
            for m, values in enumerate(self.ff.cmap.maps):
                size = int(round(math.sqrt(len(values))))
                e = np.array(values).reshape(size, size).T
                grid = np.roll(np.roll(e, size // 2, axis=0), size // 2, axis=1) / KCAL
                aux = ParamTable()
                for p in ("phi", "psi", "energy"):
                    aux.add_prop(p, float)
                ang = -180.0 + 360.0 / size * np.arange(size)
                phi, psi = np.meshgrid(ang, ang, indexing="ij")
                aux.add_params(size * size, phi=phi.ravel(), psi=psi.ravel(), energy=grid.ravel())
                s.aux_tables[f"cmap{m + 1}"] = aux
        n = self.n
        sig, eps = self.vdw
        lj = self.ff.lennard_jones
        if lj is None:
            nb = s.add_nonbonded_from_schema("vdw_12_6", "arithmetic/geometric")
            keys = list(zip(sig.tolist(), eps.tolist(), strict=True))
            uniq = list(dict.fromkeys(keys))
            where = {k: i for i, k in enumerate(uniq)}
            arr = np.array(uniq, np.float64).reshape(-1, 2)
            pids = nb.params.add_params(len(uniq), sigma=arr[:, 0], epsilon=arr[:, 1])
            nb.add_terms(np.arange(n)[:, None], params=pids[[where[k] for k in keys]])
        else:
            nb = s.add_nonbonded_from_schema("vdw_12_6", "arithmetic/geometric")
            nb.params.add_prop("type", str)
            types = list(dict.fromkeys(self.lj_types))
            where = {t: i for i, t in enumerate(types)}
            vals = [lj.params.values(t, {}) for t in types]
            pids = nb.params.add_params(len(types), sigma=np.array([v[0] / NM for v in vals]),
                                        epsilon=np.array([v[1] / KCAL for v in vals]),
                                        type=np.array(types, dtype=STR))  # fmt: skip
            nb.add_terms(np.arange(n)[:, None], params=pids[[where[t] for t in self.lj_types]])
            for i, t1 in enumerate(types):
                for j, t2 in enumerate(types[i:], i):
                    fix = lj.pair(t1, t2)
                    if fix is not None:
                        nb.overrides.set(pids[i], pids[j], sigma=fix[0] / NM, epsilon=fix[1] / KCAL)
        if self.exclusions:
            s.add_table_from_schema("exclusion").add_terms(np.array(self.exclusions, np.int64))
        s.nonbonded_info.vdw_funct, s.nonbonded_info.vdw_rule = "vdw_12_6", "arithmetic/geometric"
        if self.cons:
            _constraint_tables(s, self.cons)
            at = s.tables.get("angle_harm")
            if at is not None and "constraint_hoh" in s.table_names:
                hoh = {frozenset(a) for a in s.table("constraint_hoh").atoms.tolist()}
                flags = at.values("constrained")
                for k, a in enumerate(at.atoms.tolist()):
                    if frozenset(a) in hoh:
                        flags[k] = 1
                at._t.set("constrained", flags)
        _virtual_site_tables(s, sites)
        return s


def _mass_of(anum: int) -> float:
    from .elements import guessed_masses

    return float(guessed_masses(np.array([anum]))[0])


def _find_improper(types, classes):
    for count in range(len(classes) + 1):
        for wild in itertools.combinations(range(len(classes)), count):
            key = list(classes)
            for w in wild:
                key[w] = None
            key = tuple(key)
            if key in types:
                return types[key]
    return None


def _single_patches(template, patches, index, out, altered):
    """OpenMM's _generatePatchedSingleResidueTemplates."""
    new = None
    if not altered & patches[index].names:
        try:
            new = patches[index].create([template])[0]
            out.append(new)
        except (FFXMLError, KeyError, IndexError):
            new = None
    if index + 1 < len(patches):
        _single_patches(template, patches, index + 1, out, altered)
        if new is not None:
            _single_patches(new, patches, index + 1, out, altered | patches[index].names)


def _light_apply(state, patch):
    """What ``patch.create`` would leave (atom names -> element, external-bond atoms),
    or None where it would fail; the cheap stand-in for building the template."""
    names, external = state
    gone = {d.name for d in patch.deleted if d.residue == 0}
    new = {nm: z for nm, z in names.items() if nm not in gone}
    for a in patch.added[0]:
        if a.name in new:
            return None
        new[a.name] = a.anum
    for a in patch.changed[0]:
        if a.name not in new:
            return None
        new[a.name] = a.anum
    gone_ext = {a.name for a in patch.deleted_external if a.residue == 0}
    ext = []
    for nm in external:
        if nm in gone_ext:
            continue
        if nm not in new:
            return None  # the atom holding an external bond was deleted
        ext.append(nm)
    for a1, a2 in patch.added_bonds:
        if a1.residue == 0 and a2.residue == 0:
            if a1.name not in new or a2.name not in new:
                return None
        elif a1.residue == 0 or a2.residue == 0:
            nm = a1.name if a1.residue == 0 else a2.name
            if nm not in new:
                return None
            ext.append(nm)
    for a in patch.added_external:
        if a.name not in new:
            return None
        ext.append(a.name)
    return new, ext


def _lazy_patches(template, patches, wanted) -> list[_Template]:
    """The templates _single_patches would make, in the same order, but only those
    whose element signature is in ``wanted`` (the others cannot match any residue)."""
    start = (
        {a.name: a.anum for a in template.atoms},
        [template.atoms[i].name for i in template.external],
    )
    paths = []

    def walk(state, path, index, altered):
        p = patches[index]
        new = None
        if not altered & p.names:
            new = _light_apply(state, p)
            if new is not None:
                paths.append(((*path, index), new))
        if index + 1 < len(patches):
            walk(state, path, index + 1, altered)
            if new is not None:
                walk(new, (*path, index), index + 1, altered | p.names)

    walk(start, (), 0, set())
    out = []
    for path, (names, _) in paths:
        if _signature(names.values()) not in wanted:
            continue
        t = template
        try:
            for k in path:
                t = patches[k].create([t])[0]
        except (FFXMLError, KeyError, IndexError):
            continue
        out.append(t)
    return out


def _site_position(site, where):
    p = [np.asarray(where[i], np.float64) for i in site.atoms]
    if site.kind == "average2":
        return site.weights[0] * p[0] + site.weights[1] * p[1]
    if site.kind == "average3":
        return site.weights[0] * p[0] + site.weights[1] * p[1] + site.weights[2] * p[2]
    if site.kind == "outOfPlane":
        v1, v2 = p[1] - p[0], p[2] - p[0]
        # weightCross is per nm; positions are in Å
        return (
            p[0]
            + site.weights[0] * v1
            + site.weights[1] * v2
            + site.weights[2] * NM * np.cross(v1, v2)
        )
    origin = sum(w * x for w, x in zip(site.origin, p, strict=True))
    xdir = sum(w * x for w, x in zip(site.xw, p, strict=True))
    ydir = sum(w * x for w, x in zip(site.yw, p, strict=True))
    zdir = np.cross(xdir, ydir)
    xdir = xdir / np.linalg.norm(xdir)
    zdir = zdir / np.linalg.norm(zdir)
    ydir = np.cross(zdir, xdir)
    return origin + (xdir * site.local[0] + ydir * site.local[1] + zdir * site.local[2]) / NM


def _virtual_site_tables(s, sites):
    groups: dict[str, list] = {}
    for v, (site, parents, _) in sorted(sites.items()):
        if site.kind == "average2":
            name, vals = "virtual_lc2", [site.weights[1]]
        elif site.kind == "average3":
            name, vals = "virtual_lc3", [site.weights[1], site.weights[2]]
        elif site.kind == "outOfPlane":
            name, vals = "virtual_out3", [site.weights[0], site.weights[1], site.weights[2] * NM]
        else:
            if len(parents) != 3 or not np.allclose(
                [site.origin, site.xw, site.yw], _FDAT3_WEIGHTS
            ):
                raise FFXMLError(
                    "localCoords virtual sites other than a fdat3 frame are not supported"
                )
            lx, ly, lz = (x / NM for x in site.local)
            a, b, c = -lx, ly, lz
            d = math.sqrt(a * a + b * b + c * c)
            name, vals = "virtual_fdat3", [d, math.degrees(math.acos(a / d)) if d else 0.0,
                                           math.degrees(math.atan2(c, b))]  # fmt: skip
        groups.setdefault(name, []).append(([v, *parents], vals))
    for name, items in groups.items():
        t = s.add_table_from_schema(name)
        rows = _Rows(len(items[0][0]), tuple(t.params.props))
        for atoms, vals in items:
            rows.add(atoms, vals)
        arr = np.array(rows.rows, np.float64).reshape(len(rows.rows), -1)
        pids = t.params.add_params(
            len(rows.rows), **{p: arr[:, k] for k, p in enumerate(rows.props)}
        )
        t.add_terms(np.array(rows.atoms, np.int64), params=pids[np.array(rows.pids, np.int64)])


def parameterize_openmm(system: System, forcefield, *, constraints=None, rigid_water=None,
                        hydrogen_mass=None, residue_templates=None,
                        ignore_external_bonds: bool = False) -> System:  # fmt: skip
    """A copy of ``system`` with the force field OpenMM's ``createSystem`` would build.

    ``forcefield`` is an :class:`OpenMMForcefield` or XML file names (as for
    ``openmm.app.ForceField``).  ``constraints`` is None, ``"hbonds"``,
    ``"allbonds"`` or ``"hangles"``; ``rigid_water`` (default: the templates'
    choice, rigid) makes residues named HOH rigid; ``hydrogen_mass`` (amu)
    repartitions mass onto hydrogens.  Constrained bonds and angles stay in
    their tables, marked ``constrained``, and the constraints go to
    ``constraint_ahN``/``constraint_hoh`` tables.  ``residue_templates`` maps
    residue indices to template names.
    """
    ff = forcefield if isinstance(forcefield, OpenMMForcefield) else load_openmm_forcefield(
        *([forcefield] if isinstance(forcefield, str | os.PathLike) else forcefield))  # fmt: skip
    if constraints not in (None, "hbonds", "allbonds", "hangles"):
        raise FFXMLError(
            f"constraints must be None, 'hbonds', 'allbonds' or 'hangles', not {constraints!r}"
        )
    run = _Run(system, ff, residue_templates, ignore_external_bonds)
    run.match_all()
    run.add_extra_particles()
    return run.build(constraints, rigid_water, hydrogen_mass)
