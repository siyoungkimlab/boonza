"""The System: structure, force-field tables and derived topology.

Atoms, bonds, residues and chains are columnar tables indexed 0..N-1.  The
hierarchy is atom -> residue -> chain -> ct; a fragment (molecule) is a set
of covalently connected atoms and is derived from the bonds on demand.
Derived data (fragments, adjacency, residue membership) is cached and
dropped whenever the topology changes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from . import graph
from ._columns import ColumnTable, infer_kind
from .handles import Atom, AtomSel, Bond, Chain, Ct, Residue, _Handle
from .schemas import NONBONDED_SCHEMAS, TERM_SCHEMAS
from .terms import ParamTable, TermTable


@dataclass
class NonbondedInfo:
    vdw_funct: str = ""
    vdw_rule: str = ""
    es_funct: str = ""

    def merge(self, other: NonbondedInfo) -> None:
        """Take unset fields from ``other``; raise if a set field disagrees."""
        for field in ("vdw_funct", "vdw_rule", "es_funct"):
            mine, theirs = getattr(self, field), getattr(other, field)
            if theirs and mine and mine.lower() != theirs.lower():
                raise ValueError(f"incompatible nonbonded_info.{field}: {mine!r} vs {theirs!r}")
        for field in ("vdw_funct", "vdw_rule", "es_funct"):
            if not getattr(self, field):
                setattr(self, field, getattr(other, field))


_BUILTINS = {
    "atom": (
        ("residue", "int", (), -1),
        ("name", "str", (), ""),
        ("anum", "int", (), 0),
        ("pos", "float", (3,), 0.0),
        ("vel", "float", (3,), 0.0),
        ("mass", "float", (), 0.0),
        ("charge", "float", (), 0.0),
        ("formal_charge", "int", (), 0),
    ),
    "bond": (("i", "int", (), -1), ("j", "int", (), -1), ("order", "int", (), 1)),
    "residue": (
        ("chain", "int", (), -1),
        ("resid", "int", (), 0),
        ("name", "str", (), ""),
        ("insertion", "str", (), ""),
    ),
    "chain": (("ct", "int", (), -1), ("name", "str", (), ""), ("segid", "str", (), "")),
}
_ATTR = {"atom": "_atoms", "bond": "_bonds", "residue": "_residues", "chain": "_chains"}
_PARENT = {"atom": ("residue", "residue"), "residue": ("chain", "chain"), "chain": ("ct", "ct")}
_PROTECTED = {"atom": {"residue"}, "bond": {"i", "j"}, "residue": {"chain"}, "chain": {"ct"}}
_HANDLE = {"atom": Atom, "bond": Bond, "residue": Residue, "chain": Chain}


def _new_table(kind: str) -> ColumnTable:
    specs = _BUILTINS[kind]
    t = ColumnTable(builtin=[s[0] for s in specs])
    for name, k, shape, default in specs:
        t.add_column(name, k, shape, default)
    return t


def _remap(ids: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Map ids through ``mapping``, leaving -1 (unset) alone."""
    out = np.full(ids.shape, -1, np.int64)
    ok = ids >= 0
    out[ok] = mapping[ids[ok]]
    return out


def _check_compatible(mine: ColumnTable, theirs: ColumnTable, what: str) -> None:
    for name in theirs.names:
        if name in mine and mine.spec(name)[:2] != theirs.spec(name)[:2]:
            raise ValueError(f"{what} property {name!r} has different types in the two systems")


class TableView:
    """Column access to atoms, bonds, residues or chains; items are handles.

    ``view["name"]`` returns a numpy view of the column (valid until the
    table grows); assigning to a new name creates a column.
    """

    __slots__ = ("_sys", "_kind")

    def __init__(self, system: System, kind: str):
        self._sys = system
        self._kind = kind

    @property
    def _t(self) -> ColumnTable:
        return getattr(self._sys, _ATTR[self._kind])

    def __len__(self) -> int:
        return len(self._t)

    def __iter__(self):
        cls = _HANDLE[self._kind]
        return (cls(self._sys, i) for i in range(len(self._t)))

    def __contains__(self, name) -> bool:
        return name in self._t

    def __getitem__(self, key):
        if isinstance(key, str):
            col = self._t.column(key)
            if key in _PROTECTED[self._kind]:
                col = col.view()
                col.flags.writeable = False
            return col
        n = len(self._t)
        if isinstance(key, int | np.integer):
            i = int(key) + (n if key < 0 else 0)
            if not 0 <= i < n:
                raise IndexError(f"{self._kind} {key} out of range ({n} {self._kind}s)")
            return _HANDLE[self._kind](self._sys, i)
        ids = np.arange(n)[key]
        if self._kind == "atom":
            return AtomSel(self._sys, ids)
        return [_HANDLE[self._kind](self._sys, i) for i in ids]

    def __setitem__(self, name: str, values) -> None:
        if name not in self._t:
            arr = np.asarray(values)
            shape = arr.shape[1:] if arr.ndim > 1 else ()
            self._t.add_column(name, infer_kind(arr), shape)
        self._sys._set_values(self._kind, name, slice(None), values)

    @property
    def props(self) -> list[str]:
        """User-defined columns."""
        return self._t.props

    @property
    def names(self) -> list[str]:
        """All columns, built-in first."""
        return self._t.names

    def add_prop(self, name: str, type=float, default=None, shape=()) -> None:
        self._t.add_column(name, type, shape, default)

    def del_prop(self, name: str) -> None:
        self._t.del_column(name)

    def to_pandas(self):
        """The columns as a pandas DataFrame (vector columns as name_x/_y/_z or name_0, ...)."""
        import pandas as pd

        t = self._t
        data = {}
        for name in t.names:
            col = t.column(name)
            if col.ndim == 1:
                data[name] = col.copy()
                continue
            flat = col.reshape(len(col), -1)
            suffix = "xyz" if flat.shape[1] == 3 else [str(k) for k in range(flat.shape[1])]
            for k, sfx in enumerate(suffix):
                data[f"{name}_{sfx}"] = flat[:, k].copy()
        return pd.DataFrame(data)

    def prop_type(self, name: str) -> str:
        return self._t.kind(name)

    def __repr__(self) -> str:
        return f"<{self._kind}s: {len(self)} rows; columns {', '.join(self.names)}>"


class System:
    """A molecular system: atoms, bonds, residues, chains, cts and force-field tables."""

    def __init__(self, name: str = ""):
        self.name = name
        self._atoms = _new_table("atom")
        self._bonds = _new_table("bond")
        self._residues = _new_table("residue")
        self._chains = _new_table("chain")
        self._ct_names: list[str] = []
        self._ct_props: list[dict] = []
        self.tables: dict[str, TermTable] = {}
        self.aux_tables: dict[str, ParamTable] = {}
        self.nonbonded_info = NonbondedInfo()
        self._cell = np.zeros((3, 3))
        self.provenance: list[dict] = []
        self._epochs = {"atom": 0, "bond": 0, "residue": 0, "chain": 0, "ct": 0}
        self._cache: dict = {}
        self._bond_index = None

    # ---- sizes and views ------------------------------------------------
    @property
    def natoms(self) -> int:
        return len(self._atoms)

    @property
    def nbonds(self) -> int:
        return len(self._bonds)

    @property
    def nresidues(self) -> int:
        return len(self._residues)

    @property
    def nchains(self) -> int:
        return len(self._chains)

    @property
    def ncts(self) -> int:
        return len(self._ct_names)

    @property
    def atoms(self) -> TableView:
        return TableView(self, "atom")

    @property
    def bonds(self) -> TableView:
        return TableView(self, "bond")

    @property
    def residues(self) -> TableView:
        return TableView(self, "residue")

    @property
    def chains(self) -> TableView:
        return TableView(self, "chain")

    @property
    def cts(self) -> list[Ct]:
        return [Ct(self, i) for i in range(self.ncts)]

    def atom(self, i) -> Atom:
        return Atom(self, self._index("atom", i))

    def bond(self, i) -> Bond:
        return Bond(self, self._index("bond", i))

    def residue(self, i) -> Residue:
        return Residue(self, self._index("residue", i))

    def chain(self, i) -> Chain:
        return Chain(self, self._index("chain", i))

    def ct(self, i) -> Ct:
        return Ct(self, self._index("ct", i))

    @property
    def positions(self) -> np.ndarray:
        """(natoms, 3) view of positions in Å; valid until atoms are added."""
        return self._atoms.column("pos")

    @positions.setter
    def positions(self, xyz) -> None:
        self._atoms.set("pos", np.asarray(xyz, dtype=np.float64).reshape(self.natoms, 3))

    @property
    def velocities(self) -> np.ndarray:
        return self._atoms.column("vel")

    @velocities.setter
    def velocities(self, v) -> None:
        self._atoms.set("vel", np.asarray(v, dtype=np.float64).reshape(self.natoms, 3))

    @property
    def cell(self) -> np.ndarray:
        """Unit cell; rows are the box vectors a, b, c in Å."""
        return self._cell

    @cell.setter
    def cell(self, value) -> None:
        self._cell = np.array(value, dtype=np.float64).reshape(3, 3)

    def __repr__(self) -> str:
        return (
            f"<System {self.name!r}: {self.natoms} atoms, {self.nbonds} bonds, "
            f"{self.nresidues} residues, {self.nchains} chains, {len(self.tables)} tables>"
        )

    # ---- index helpers --------------------------------------------------
    def _count(self, kind: str) -> int:
        return self.ncts if kind == "ct" else len(getattr(self, _ATTR[kind]))

    def _index(self, kind: str, obj) -> int:
        if isinstance(obj, _Handle) and obj._sys is not self:
            raise ValueError(f"{kind} handle belongs to a different system")
        i = int(obj)
        n = self._count(kind)
        if not 0 <= i < n:
            raise IndexError(f"{kind} {i} out of range ({n} {kind}s)")
        return i

    def _ids(self, kind: str, obj) -> np.ndarray:
        n = self._count(kind)
        if isinstance(obj, str):
            if kind != "atom":
                raise TypeError(f"selection strings choose atoms, not {kind}s")
            return self.select(obj).ids
        if isinstance(obj, AtomSel):
            if obj._sys is not self:
                raise ValueError("selection belongs to a different system")
            ids = obj.ids
        elif isinstance(obj, _Handle | int | np.integer):
            ids = np.array([self._index(kind, obj)], dtype=np.int64)
        else:
            arr = np.asarray(obj)
            if arr.dtype == bool:
                if arr.shape != (n,):
                    raise IndexError(f"boolean mask must have shape ({n},)")
                return np.flatnonzero(arr)
            if arr.dtype == object:
                arr = np.array([self._index(kind, x) for x in obj], dtype=np.int64)
            ids = arr.astype(np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= n):
            raise IndexError(f"{kind} index out of range ({n} {kind}s)")
        return ids

    def _touch(self, *renumbered: str) -> None:
        self._cache.clear()
        for kind in renumbered:
            self._epochs[kind] += 1

    def _set_values(self, kind: str, name: str, rows, values) -> None:
        if name in _PROTECTED[kind]:
            raise ValueError(
                f"{kind} column {name!r} links the hierarchy; change it through a handle "
                f"(e.g. atom.residue = res) or a System method"
            )
        getattr(self, _ATTR[kind]).set(name, values, rows)
        self._cache.pop("analyze", None)  # residue typing depends on names

    def _set_parent(self, kind: str, i: int, parent) -> None:
        column, parent_kind = _PARENT[kind]
        getattr(self, _ATTR[kind]).set(column, self._index(parent_kind, parent), i)
        self._cache.clear()

    def _check_props(self, kind: str, props: dict) -> dict:
        t = getattr(self, _ATTR[kind])
        for name in props:
            if name not in t:
                raise KeyError(
                    f"no {kind} property {name!r}; "
                    f"add it with system.{kind}s.add_prop({name!r}, type)"
                )
            if name in _PROTECTED[kind]:
                raise ValueError(f"{kind} column {name!r} is set by the method arguments")
        return props

    # ---- building ---------------------------------------------------------
    def add_ct(self, name: str = "", **props) -> Ct:
        self._ct_names.append(str(name))
        self._ct_props.append(dict(props))
        self._cache.clear()
        return Ct(self, self.ncts - 1)

    def add_chain(self, ct=None, name: str = "", segid: str = "") -> Chain:
        """Add a chain to ``ct`` (default: the last ct, created if needed)."""
        if ct is None:
            ct = self.ncts - 1 if self.ncts else self.add_ct().id
        c = self._chains.append(1, {"ct": self._index("ct", ct), "name": name, "segid": segid})
        self._cache.clear()
        return Chain(self, c)

    def add_residue(
        self, chain=None, name: str = "", resid: int = 0, insertion: str = ""
    ) -> Residue:
        """Add a residue to ``chain`` (default: the last chain, created if needed)."""
        if chain is None:
            chain = self.nchains - 1 if self.nchains else self.add_chain().id
        values = {"chain": self._index("chain", chain), "name": name, "resid": resid}
        r = self._residues.append(1, {**values, "insertion": insertion})
        self._cache.clear()
        return Residue(self, r)

    def _parent_ids(self, kind: str, value, count: int) -> np.ndarray:
        """One parent index per new row: a single handle/index, or one per row."""
        if isinstance(value, _Handle | int | np.integer):
            return np.full(count, self._index(kind, value), np.int64)
        ids = np.asarray([v.id if isinstance(v, _Handle) else v for v in value], dtype=np.int64)
        n = self.ncts if kind == "ct" else len(getattr(self, _ATTR[kind]))
        if len(ids) != count:
            raise ValueError(f"got {len(ids)} {kind}s for {count} new rows")
        if ids.size and (ids.min() < 0 or ids.max() >= n):
            raise IndexError(f"{kind} index out of range ({n} {kind}s)")
        return ids

    @staticmethod
    def _per_row(value, count: int, dtype) -> np.ndarray:
        from ._columns import STR

        dtype = STR if dtype is str else dtype
        if isinstance(value, str | int | float | np.integer | np.floating):
            return np.full(count, value, dtype=dtype)
        arr = np.asarray(value).astype(dtype)
        if len(arr) != count:
            raise ValueError(f"got {len(arr)} values for {count} new rows")
        return arr

    def add_chains(self, count: int, ct=None, name="", segid="") -> np.ndarray:
        """Add ``count`` chains at once; each argument is one value or one per chain.

        Returns the new chain indices."""
        if ct is None:
            ct = self.ncts - 1 if self.ncts else self.add_ct().id
        values = {"ct": self._parent_ids("ct", ct, count), "name": self._per_row(name, count, str),
                  "segid": self._per_row(segid, count, str)}  # fmt: skip
        start = self._chains.append(count, values)
        self._cache.clear()
        return np.arange(start, start + count)

    def add_residues(self, count: int, chain=None, name="", resid=0,
                     insertion="") -> np.ndarray:  # fmt: skip
        """Add ``count`` residues at once; each argument is one value or one per residue.

        Returns the new residue indices, e.g. to pass as ``add_atoms(residue=...)``."""
        if chain is None:
            chain = self.nchains - 1 if self.nchains else self.add_chain().id
        values = {"chain": self._parent_ids("chain", chain, count),
                  "name": self._per_row(name, count, str),
                  "resid": self._per_row(resid, count, np.int64),
                  "insertion": self._per_row(insertion, count, str)}  # fmt: skip
        start = self._residues.append(count, values)
        self._cache.clear()
        return np.arange(start, start + count)

    @classmethod
    def from_arrays(cls, positions, names=None, elements=None, anum=None, resnames=None,
                    resids=None, chains=None, segids=None, insertions=None, cell=None,
                    bonds=None, **columns) -> System:  # fmt: skip
        """Build a system from per-atom arrays in one step.

        Residues are the distinct (chain, segid, resid, resname, insertion)
        combinations and chains the distinct (chain, segid) pairs, numbered in
        order of first appearance, as when reading a PDB file.  Elements come
        from ``anum``, else ``elements`` symbols, else are guessed from the
        names.  ``bonds``: (n, 2) atom pairs.  Other keyword arguments set atom
        columns (``charge``, ``mass``, or new ones).
        """
        from .elements import element_for_abbreviation
        from .io.dms import _factorize
        from .io.pdb import _element

        pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        n = len(pos)

        def strings(v):
            return cls._per_row("" if v is None else v, n, str)

        names_, resnames_, chains_ = strings(names), strings(resnames), strings(chains)
        segids_, insertions_ = strings(segids), strings(insertions)
        resids_ = cls._per_row(1 if resids is None else resids, n, np.int64)
        if anum is not None:
            z = cls._per_row(anum, n, np.int64)
        else:
            sym = strings(elements)
            combo, first = _factorize(sym, names_, resnames_)
            guess = [element_for_abbreviation(str(sym[k])) if sym[k]
                     else _element("", str(names_[k]), str(resnames_[k]))
                     for k in first.tolist()]  # fmt: skip
            z = np.array(guess, np.int64)[combo] if n else np.zeros(0, np.int64)
        s = cls()
        ct = s.add_ct()
        chain_code, chain_first = _factorize(chains_, segids_)
        res_code, res_first = _factorize(chain_code, resids_, resnames_, insertions_)
        s._chains.append(len(chain_first), {"ct": ct.id, "name": chains_[chain_first],
                                             "segid": segids_[chain_first]})  # fmt: skip
        s._residues.append(len(res_first), {"chain": chain_code[res_first],
                                             "resid": resids_[res_first],
                                             "name": resnames_[res_first],
                                             "insertion": insertions_[res_first]})  # fmt: skip
        s._atoms.append(n, {"residue": res_code, "name": names_, "anum": z, "pos": pos})
        s._cache.clear()
        for key, values in columns.items():
            s.atoms[key] = values
        if cell is not None:
            s.cell = cell
        if bonds is not None:
            s.add_bonds(bonds)
        return s

    def add_atom(self, residue=None, **props) -> Atom:
        """Add an atom to ``residue`` (default: the last residue, created if needed).

        Keyword arguments set columns, e.g. ``name="CA", anum=6, pos=(0, 0, 0)``.
        """
        if residue is None:
            residue = self.nresidues - 1 if self.nresidues else self.add_residue().id
        values = {"residue": self._index("residue", residue), **self._check_props("atom", props)}
        a = self._atoms.append(1, values)
        self._cache.clear()
        return Atom(self, a)

    def add_atoms(self, count: int, residue=None, **columns) -> AtomSel:
        """Add ``count`` atoms at once; ``residue`` may be one residue or one per atom."""
        if residue is None:
            residue = self.nresidues - 1 if self.nresidues else self.add_residue().id
        if isinstance(residue, _Handle | int | np.integer):
            res = np.full(count, self._index("residue", residue), np.int64)
        else:
            res = self._ids("residue", residue)
            if len(res) != count:
                raise ValueError(f"got {len(res)} residues for {count} atoms")
        start = self._atoms.append(count, {"residue": res, **self._check_props("atom", columns)})
        self._cache.clear()
        return AtomSel(self, np.arange(start, start + count))

    # ---- bonds --------------------------------------------------------
    def _bond_keys(self):
        idx = self._bond_index
        if idx is None:
            keys = (self._bonds.column("i") << 32) | self._bonds.column("j")
            order = np.argsort(keys, kind="stable")
            idx = self._bond_index = (keys[order], order, {})
        return idx

    def _find_bond_id(self, lo: int, hi: int) -> int:
        keys, order, extra = self._bond_keys()
        k = (lo << 32) | hi
        p = int(np.searchsorted(keys, k))
        if p < len(keys) and keys[p] == k:
            return int(order[p])
        return extra.get(k, -1)

    def find_bond(self, a, b) -> Bond | None:
        i, j = self._index("atom", a), self._index("atom", b)
        bid = self._find_bond_id(min(i, j), max(i, j))
        return None if bid < 0 else Bond(self, bid)

    def add_bond(self, a, b, order: int = 1, **props) -> Bond:
        """Bond two atoms; an existing bond between them is returned unchanged."""
        i, j = self._index("atom", a), self._index("atom", b)
        if i == j:
            raise ValueError("cannot bond an atom to itself")
        lo, hi = min(i, j), max(i, j)
        bid = self._find_bond_id(lo, hi)
        if bid >= 0:
            return Bond(self, bid)
        values = {"i": lo, "j": hi, "order": order, **self._check_props("bond", props)}
        bid = self._bonds.append(1, values)
        keys, _, extra = self._bond_index
        extra[(lo << 32) | hi] = bid
        if len(extra) > max(1024, len(keys) // 4):
            self._bond_index = None
        self._cache.clear()
        return Bond(self, bid)

    def add_bonds(self, pairs, order=1, **props) -> np.ndarray:
        """Bond many atom pairs at once; returns the bond index for every pair.

        Pairs that are already bonded (or repeated) are not added twice.
        ``order`` and ``props`` may be scalars or one value per pair.
        """
        p = np.asarray(pairs)
        if p.dtype == object:
            p = np.array([[int(a), int(b)] for a, b in pairs])
        p = p.astype(np.int64).reshape(-1, 2)
        n = self.natoms
        if p.size and (p.min() < 0 or p.max() >= n):
            raise IndexError(f"atom index out of range ({n} atoms)")
        if (p[:, 0] == p[:, 1]).any():
            raise ValueError("cannot bond an atom to itself")
        self._check_props("bond", props)
        lo, hi = p.min(axis=1), p.max(axis=1)
        keys = (lo << 32) | hi
        self._bond_index = None
        skeys, sorder, _ = self._bond_keys()
        out = np.full(len(keys), -1, np.int64)
        if len(skeys):
            pos = np.minimum(np.searchsorted(skeys, keys), len(skeys) - 1)
            exists = skeys[pos] == keys
            out[exists] = sorder[pos[exists]]
        new = np.flatnonzero(out < 0)
        if len(new):
            _, first, inv = np.unique(keys[new], return_index=True, return_inverse=True)
            appear = np.argsort(first, kind="stable")
            rank = np.empty(len(appear), np.int64)
            rank[appear] = np.arange(len(appear))
            src = new[first[appear]]
            values = {"i": lo[src], "j": hi[src]}
            for name, v in {"order": order, **props}.items():
                v = np.asarray(v)
                values[name] = v[src] if v.ndim else v
            start = self._bonds.append(len(src), values)
            out[new] = start + rank[inv.reshape(-1)]
        self._bond_index = None
        self._cache.clear()
        return out

    def guess_bonds(self, periodic: bool = False, replace: bool = True) -> None:
        """Bond atoms closer than 0.6 x the sum of their radii (msys rules).

        ``replace`` removes existing bonds first; ``periodic`` uses the unit
        cell and bonds across cts.
        """
        from .bonds import guess_bonds

        guess_bonds(self, periodic=periodic, replace=replace)

    # ---- deletion -------------------------------------------------------
    def delete_atoms(self, atoms) -> np.ndarray:
        """Delete atoms, with their bonds and terms, and any residues/chains left empty.

        Returns the old-to-new atom index map (-1 for deleted atoms).  Handles
        created before the call become stale.
        """
        ids = self._ids("atom", atoms)
        keep = np.ones(self.natoms, bool)
        keep[ids] = False
        return self._retain_atoms(np.flatnonzero(keep))

    def delete_residues(self, residues) -> np.ndarray:
        ids = self._ids("residue", residues)
        mask = np.isin(self._atoms.column("residue"), ids)
        return self._retain_atoms(np.flatnonzero(~mask))

    def delete_chains(self, chains) -> np.ndarray:
        ids = self._ids("chain", chains)
        mask = np.isin(self._level_key("chain"), ids)
        return self._retain_atoms(np.flatnonzero(~mask))

    def delete_bonds(self, bonds) -> None:
        ids = self._ids("bond", bonds)
        keep = np.ones(self.nbonds, bool)
        keep[ids] = False
        self._bonds.keep(keep)
        self._bond_index = None
        self._touch("bond")

    def reorder_atoms(self, order) -> np.ndarray:
        """Renumber atoms so that new atom k is old atom ``order[k]``."""
        order = self._ids("atom", order)
        if len(order) != self.natoms or len(np.unique(order)) != self.natoms:
            raise ValueError("order must be a permutation of all atoms")
        return self._retain_atoms(order)

    def _retain_atoms(self, keep: np.ndarray) -> np.ndarray:
        amap = np.full(self.natoms, -1, np.int64)
        amap[keep] = np.arange(len(keep))
        self._atoms.keep(keep)
        nbonds = self.nbonds
        i = amap[self._bonds.column("i")]
        j = amap[self._bonds.column("j")]
        bkeep = np.flatnonzero((i >= 0) & (j >= 0))
        self._bonds.keep(bkeep)
        m = len(bkeep)
        self._bonds._data["i"][:m] = np.minimum(i[bkeep], j[bkeep])
        self._bonds._data["j"][:m] = np.maximum(i[bkeep], j[bkeep])
        for t in self.tables.values():
            t._remap_atoms(amap)
        renumbered = ["atom", *self._prune_hierarchy()]
        if m != nbonds:
            renumbered.append("bond")
        self._bond_index = None
        self._touch(*renumbered)
        return amap

    def _prune_hierarchy(self) -> list[str]:
        """Drop residues without atoms and chains without residues."""
        changed = []
        used = np.zeros(self.nresidues, bool)
        used[self._atoms.column("residue")] = True
        if not used.all():
            rmap = np.cumsum(used) - 1
            self._residues.keep(used)
            self._atoms._data["residue"][: self.natoms] = rmap[self._atoms.column("residue")]
            changed.append("residue")
        used = np.zeros(self.nchains, bool)
        used[self._residues.column("chain")] = True
        if not used.all():
            cmap = np.cumsum(used) - 1
            self._chains.keep(used)
            self._residues._data["chain"][: self.nresidues] = cmap[self._residues.column("chain")]
            changed.append("chain")
        return changed

    # ---- term tables ------------------------------------------------------
    def add_table(self, name: str, natoms: int, category: str = "bond", params=None) -> TermTable:
        """Add a term table (or return the existing one with the same name)."""
        t = self.tables.get(name)
        if t is not None:
            if t.natoms != natoms or t.category != category:
                raise ValueError(
                    f"table {name!r} exists with natoms={t.natoms}, category={t.category!r}"
                )
            return t
        t = self.tables[name] = TermTable(name, natoms, category, params=params, system=self)
        return t

    def add_table_from_schema(self, schema: str, name: str | None = None) -> TermTable:
        try:
            s = TERM_SCHEMAS[schema]
        except KeyError:
            raise KeyError(f"unknown table schema {schema!r}") from None
        t = self.add_table(name or schema, s.natoms, s.category)
        for prop, kind in s.params:
            t.params.add_prop(prop, kind)
        for prop, kind in s.term_props:
            t.add_term_prop(prop, kind)
        return t

    def add_nonbonded_from_schema(self, funct: str = "vdw_12_6", rule: str = "") -> TermTable:
        try:
            s = NONBONDED_SCHEMAS[funct]
        except KeyError:
            raise KeyError(f"unknown nonbonded schema {funct!r}") from None
        self.nonbonded_info.merge(NonbondedInfo(vdw_funct=funct, vdw_rule=rule.lower()))
        t = self.add_table("nonbonded", 1, "nonbonded")
        for prop, kind in s.params:
            t.params.add_prop(prop, kind)
        return t

    def table(self, name: str) -> TermTable:
        try:
            return self.tables[name]
        except KeyError:
            raise KeyError(f"no table {name!r}; tables: {sorted(self.tables)}") from None

    @property
    def table_names(self) -> list[str]:
        return sorted(self.tables)

    def del_table(self, name: str) -> None:
        del self.tables[name]

    # ---- derived topology ----------------------------------------------------
    def _fragments(self) -> tuple[np.ndarray, int]:
        c = self._cache.get("frag")
        if c is None:
            labels, nfrag = graph.connected_components(
                self.natoms, self._bonds.column("i"), self._bonds.column("j")
            )
            labels.flags.writeable = False
            c = self._cache["frag"] = (labels, nfrag)
        return c

    @property
    def fragids(self) -> np.ndarray:
        """Fragment (molecule) index of every atom, numbered by first atom."""
        return self._fragments()[0]

    @property
    def nfragments(self) -> int:
        return self._fragments()[1]

    def _csr(self, key: str, n: int, groups: np.ndarray):
        c = self._cache.get(key)
        if c is None:
            c = self._cache[key] = graph.csr(n, groups)
        return c

    def fragment_atoms(self, fragid: int) -> np.ndarray:
        off, order = self._csr("fragcsr", self.nfragments, self.fragids)
        return order[off[fragid] : off[fragid + 1]]

    def fragments(self) -> list[AtomSel]:
        off, order = self._csr("fragcsr", self.nfragments, self.fragids)
        return [AtomSel(self, order[off[k] : off[k + 1]]) for k in range(self.nfragments)]

    def _adjacency(self):
        c = self._cache.get("adj")
        if c is None:
            i, j = self._bonds.column("i"), self._bonds.column("j")
            src = np.concatenate([i, j])
            nbr = np.concatenate([j, i])
            bid = np.tile(np.arange(len(i)), 2)
            off, _ = graph.csr(self.natoms, src)
            order = np.lexsort((bid, src))  # neighbors in bond order, as in msys
            c = self._cache["adj"] = (off, nbr[order], bid[order])
        return c

    def bonded_atoms(self, atom) -> np.ndarray:
        i = self._index("atom", atom)
        off, nbr, _ = self._adjacency()
        return nbr[off[i] : off[i + 1]].copy()

    def _atom_bonds(self, i: int) -> np.ndarray:
        off, _, bid = self._adjacency()
        return bid[off[i] : off[i + 1]]

    def residue_atoms(self, residue) -> np.ndarray:
        r = self._index("residue", residue)
        off, order = self._csr("rescsr", self.nresidues, self._atoms.column("residue"))
        return order[off[r] : off[r + 1]]

    def chain_residues(self, chain) -> np.ndarray:
        c = self._index("chain", chain)
        off, order = self._csr("chncsr", self.nchains, self._residues.column("chain"))
        return order[off[c] : off[c + 1]]

    def chain_atoms(self, chain) -> np.ndarray:
        return np.flatnonzero(self._level_key("chain") == self._index("chain", chain))

    def ct_chains(self, ct) -> np.ndarray:
        return np.flatnonzero(self._chains.column("ct") == self._index("ct", ct))

    def ct_atoms(self, ct) -> np.ndarray:
        return np.flatnonzero(self._level_key("ct") == self._index("ct", ct))

    def _level_key(self, level: str) -> np.ndarray:
        """Per-atom residue, chain, ct or fragment index."""
        if level in ("fragment", "molecule"):
            return self.fragids
        key = self._atoms.column("residue")
        if level == "residue":
            return key
        key = self._residues.column("chain")[key]
        if level == "chain":
            return key
        if level == "ct":
            return self._chains.column("ct")[key]
        raise ValueError(f"unknown level {level!r}")

    def select(self, atoms, pos=None, cell=None) -> AtomSel:
        """Atoms from a selection string (msys/VMD syntax), indices, a boolean mask or handles.

        For strings, ``pos`` and ``cell`` optionally replace the positions and
        unit cell (e.g. to select on a trajectory frame).
        """
        if isinstance(atoms, str):
            from .atomsel import select

            return AtomSel(self, select(self, atoms, pos, cell))
        return AtomSel(self, self._ids("atom", atoms))

    @property
    def all(self) -> AtomSel:
        return AtomSel(self, np.arange(self.natoms))

    def to_pandas(self, atoms=None):
        """Atoms as a pandas DataFrame, one row each, with residue, chain and molecule.

        Columns: atom (index), name, element, the atom table's columns (``pos``
        as x, y, z), then resid, resname, insertion, residue (index), chain,
        segid, ct and fragid.  ``s.atoms.to_pandas()`` gives the raw table.
        """
        import pandas as pd

        from .elements import msys_symbol

        ids = np.arange(self.natoms) if atoms is None else self._ids("atom", atoms)
        ids = np.asarray(ids, dtype=np.int64)
        A, R, C = self._atoms, self._residues, self._chains
        res = A.column("residue")[ids]
        ch = R.column("chain")[res]
        anum = A.column("anum")[ids]
        symbol = {int(z): msys_symbol(int(z)) for z in np.unique(anum).tolist()}
        data = {"atom": ids, "name": A.column("name")[ids],
                "element": [symbol[int(z)] for z in anum.tolist()]}  # fmt: skip
        for name in A.names:
            if name in ("residue", "name"):
                continue
            col = A.column(name)[ids]
            if col.ndim == 1:
                data[name] = col
            elif name == "pos":
                data.update(x=col[:, 0], y=col[:, 1], z=col[:, 2])
            else:
                flat = col.reshape(len(col), -1)
                suffix = "xyz" if flat.shape[1] == 3 else [str(k) for k in range(flat.shape[1])]
                data.update({f"{name}_{sfx}": flat[:, k] for k, sfx in enumerate(suffix)})
        data.update(resid=R.column("resid")[res], resname=R.column("name")[res],
                    insertion=R.column("insertion")[res], residue=res,
                    chain=C.column("name")[ch], segid=C.column("segid")[ch],
                    ct=C.column("ct")[ch], fragid=np.asarray(self.fragids)[ids])  # fmt: skip
        return pd.DataFrame(data)

    def view(self, atoms=None, **kwargs):
        """Draw the system in a Jupyter notebook with 3Dmol.js (see ``boonza.view``)."""
        from .view import view

        return view(self, atoms, **kwargs)

    def to_rdkit(self, atoms=None, **kwargs):
        """RDKit molecule of ``atoms`` (all by default); see ``boonza.chem.to_rdkit``."""
        from .chem import to_rdkit

        return to_rdkit(self, atoms, **kwargs)

    def describe(self, atoms=None, **kwargs):
        """Force-field report of ``atoms`` (all by default); see ``boonza.describe``."""
        from .describe import describe

        return describe(self, atoms, **kwargs)

    def distinct_fragments(self) -> dict[int, list[int]]:
        """Identical molecules by representative fragment; see ``boonza.distinct_fragments``."""
        from .molecules import distinct_fragments

        return distinct_fragments(self)

    # ---- copying and merging ----------------------------------------------
    def clone(self, atoms=None, structure_only: bool = False) -> System:
        """Copy the system, or just ``atoms`` in the given order.

        A term is kept when all of its atoms are kept; only referenced params
        are copied, and param tables shared within this system stay shared.
        ``structure_only`` drops pseudo particles and all force-field tables.
        """
        ids = np.arange(self.natoms) if atoms is None else self._ids("atom", atoms)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("duplicate atoms in clone selection")
        if structure_only:
            ids = ids[self._atoms.column("anum")[ids] > 0]
        amap = np.full(self.natoms, -1, np.int64)
        amap[ids] = np.arange(len(ids))

        atom_res = self._atoms.column("residue")[ids]
        res_old = graph.unique_in_order(atom_res)
        rmap = np.full(self.nresidues, -1, np.int64)
        rmap[res_old] = np.arange(len(res_old))
        res_chn = self._residues.column("chain")[res_old]
        chn_old = graph.unique_in_order(res_chn)
        cmap = np.full(self.nchains, -1, np.int64)
        cmap[chn_old] = np.arange(len(chn_old))
        chn_ct = self._chains.column("ct")[chn_old]
        ct_old = graph.unique_in_order(chn_ct)
        ctmap = np.full(self.ncts, -1, np.int64)
        ctmap[ct_old] = np.arange(len(ct_old))

        dst = System(self.name)
        dst._ct_names = [self._ct_names[c] for c in ct_old]
        dst._ct_props = [copy.deepcopy(self._ct_props[c]) for c in ct_old]
        dst._chains = self._chains.take(chn_old)
        dst._chains._data["ct"][:] = ctmap[chn_ct]
        dst._residues = self._residues.take(res_old)
        dst._residues._data["chain"][:] = cmap[res_chn]
        dst._atoms = self._atoms.take(ids)
        dst._atoms._data["residue"][:] = rmap[atom_res]
        i = amap[self._bonds.column("i")]
        j = amap[self._bonds.column("j")]
        bkeep = np.flatnonzero((i >= 0) & (j >= 0))
        dst._bonds = self._bonds.take(bkeep)
        dst._bonds._data["i"][:] = np.minimum(i[bkeep], j[bkeep])
        dst._bonds._data["j"][:] = np.maximum(i[bkeep], j[bkeep])

        if not structure_only:
            self._clone_tables(dst, amap)
            dst.aux_tables = {k: v.copy() for k, v in self.aux_tables.items()}
        dst.nonbonded_info = copy.copy(self.nonbonded_info)
        dst._cell = self._cell.copy()
        dst.provenance = copy.deepcopy(self.provenance)
        return dst

    def _clone_tables(self, dst: System, amap: np.ndarray) -> None:
        groups: dict[int, tuple[ParamTable, list[TermTable]]] = {}
        for t in self.tables.values():
            groups.setdefault(id(t.params), (t.params, []))[1].append(t)
        for params, tabs in groups.values():
            kept, used = [], []
            for t in tabs:
                mapped = amap[t._t.column("atoms")]
                keep = np.flatnonzero((mapped >= 0).all(axis=1))
                pids = t._t.column("param")[keep]
                kept.append((t, keep, mapped[keep], pids))
                used.append(pids[pids >= 0])
            used = np.unique(np.concatenate(used))
            pmap = np.full(len(params), -1, np.int64)
            pmap[used] = np.arange(len(used))
            new_params = params.take(used)
            for t, keep, atoms, pids in kept:
                nt = TermTable(t.name, t.natoms, t.category, params=new_params, system=dst)
                nt._t = t._t.take(keep)
                nt._t._data["atoms"][:] = atoms
                nt._t._data["param"][:] = _remap(pids, pmap)
                nt.overrides = t.overrides.remapped(pmap)
                dst.tables[t.name] = nt

    def copy(self) -> System:
        return self.clone()

    def append(self, other: System) -> np.ndarray:
        """Append a copy of ``other``, force field included; return its new atom indices.

        Cts of ``other`` become new cts.  The cell is taken from ``other`` only
        when this system's cell is all zeros.
        """
        if other is self:
            other = self.clone()
        info = copy.copy(self.nonbonded_info)
        info.merge(other.nonbonded_info)
        for t in other.tables.values():
            mine = self.tables.get(t.name)
            if mine is not None and (mine.natoms != t.natoms or mine.category != t.category):
                raise ValueError(f"table {t.name!r} has different arity or category")
        for kind in _ATTR:
            _check_compatible(getattr(self, _ATTR[kind]), getattr(other, _ATTR[kind]), kind)
        self.nonbonded_info = info
        if not self._cell.any():
            self._cell = other._cell.copy()

        ct_off, chn_off, res_off, atom_off = self.ncts, self.nchains, self.nresidues, self.natoms
        self._ct_names += other._ct_names
        self._ct_props += copy.deepcopy(other._ct_props)
        oc, orr, oa, ob = other._chains, other._residues, other._atoms, other._bonds
        self._chains.extend(oc, overrides={"ct": oc.column("ct") + ct_off})
        self._residues.extend(orr, overrides={"chain": orr.column("chain") + chn_off})
        self._atoms.extend(oa, overrides={"residue": oa.column("residue") + res_off})
        self._bonds.extend(
            ob, overrides={"i": ob.column("i") + atom_off, "j": ob.column("j") + atom_off}
        )
        for t in other.tables.values():
            dt = self.add_table(t.name, t.natoms, t.category)
            pids = t._t.column("param")
            used = np.unique(pids[pids >= 0])
            start = dt.params.extend(t.params, used)
            pmap = np.full(len(t.params), -1, np.int64)
            pmap[used] = np.arange(start, start + len(used))
            dt._t.extend(
                t._t,
                overrides={"atoms": t._t.column("atoms") + atom_off, "param": _remap(pids, pmap)},
            )
            for (p1, p2), values in t.overrides.items():
                if pmap[p1] >= 0 and pmap[p2] >= 0:
                    dt.overrides.set(pmap[p1], pmap[p2], **values)
        for name, tab in other.aux_tables.items():
            if name not in self.aux_tables:
                self.aux_tables[name] = tab.copy()
        self._bond_index = None
        self._cache.clear()
        return np.arange(atom_off, atom_off + other.natoms)
