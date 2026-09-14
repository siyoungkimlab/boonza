"""Force-field tables: parameters, terms and nonbonded overrides.

The model follows msys.  A ``TermTable`` holds interactions of fixed arity
(1 atom for nonbonded, 2 for stretches, 4 for dihedrals, ...).  Each term
names its atoms and points at one row of a ``ParamTable``.  Param tables
may be shared by several term tables, and one param row is usually shared
by many terms, so editing a param through a ``Term`` copies it first.
"""

from __future__ import annotations

import numpy as np

from ._columns import ColumnTable, infer_kind, scalar

CATEGORIES = ("bond", "constraint", "virtual", "polar", "nonbonded", "exclusion")


class ParamTable:
    """Rows of parameters; columns are named properties."""

    def __init__(self):
        self._t = ColumnTable()

    def __len__(self) -> int:
        return len(self._t)

    @property
    def nparams(self) -> int:
        return len(self._t)

    @property
    def props(self) -> list[str]:
        return self._t.names

    def prop_type(self, name: str) -> str:
        return self._t.kind(name)

    def add_prop(self, name: str, type=float, default=None) -> None:
        self._t.add_column(name, type, default=default)

    def del_prop(self, name: str) -> None:
        self._t.del_column(name)

    def add_param(self, **values) -> int:
        """Append one row and return its index."""
        self._check(values)
        return self._t.append(1, values)

    def add_params(self, count: int, **columns) -> np.ndarray:
        self._check(columns)
        start = self._t.append(count, columns)
        return np.arange(start, start + count)

    def _check(self, values):
        for name in values:
            if name not in self._t:
                raise KeyError(f"no param property {name!r}; add it with add_prop()")

    def __getitem__(self, prop: str) -> np.ndarray:
        return self._t.column(prop)

    def __setitem__(self, prop: str, values) -> None:
        if prop not in self._t:
            self._t.add_column(prop, infer_kind(values))
        self._t.set(prop, values)

    def param(self, i: int) -> Param:
        i = int(i)
        if not 0 <= i < len(self):
            raise IndexError(f"param {i} out of range for table of {len(self)}")
        return Param(self, i)

    def __iter__(self):
        return (Param(self, i) for i in range(len(self)))

    def row(self, i: int) -> dict:
        return {k: scalar(self._t.column(k)[i]) for k in self._t.names}

    def find(self, prop: str, value) -> np.ndarray:
        return np.flatnonzero(self._t.column(prop) == value)

    def duplicate(self, i: int) -> int:
        return self._t.extend(self._t, rows=[int(i)])

    def take(self, rows) -> ParamTable:
        out = ParamTable()
        out._t = self._t.take(rows)
        return out

    def extend(self, other: ParamTable, rows=None) -> int:
        return self._t.extend(other._t, rows=rows)

    def copy(self) -> ParamTable:
        return self.take(None)

    def __repr__(self) -> str:
        return f"<ParamTable {len(self)} params: {', '.join(self.props)}>"


class Param:
    """One row of a ParamTable."""

    __slots__ = ("table", "id")

    def __init__(self, table: ParamTable, id: int):
        self.table = table
        self.id = id

    def __getitem__(self, prop: str):
        return scalar(self.table[prop][self.id])

    def __setitem__(self, prop: str, value) -> None:
        self.table._t.set(prop, value, self.id)

    def keys(self) -> list[str]:
        return self.table.props

    def as_dict(self) -> dict:
        return self.table.row(self.id)

    def __eq__(self, other):
        return isinstance(other, Param) and other.table is self.table and other.id == self.id

    def __hash__(self):
        return hash((id(self.table), self.id))

    def __repr__(self) -> str:
        vals = " ".join(f"{k}={v!r}" for k, v in self.as_dict().items())
        return f"<Param {self.id} {vals}>"


class OverrideTable:
    """Pairwise replacement parameters, keyed by a pair of param ids (NBFIX-style)."""

    def __init__(self):
        self.params = ParamTable()
        self._index: dict[tuple[int, int], int] = {}

    def __len__(self) -> int:
        return len(self._index)

    @staticmethod
    def _key(p1, p2) -> tuple[int, int]:
        p1, p2 = int(p1), int(p2)
        return (p1, p2) if p1 <= p2 else (p2, p1)

    def set(self, p1, p2, **values) -> None:
        for name, v in values.items():
            if name not in self.params.props:
                self.params.add_prop(name, infer_kind(v))
        key = self._key(p1, p2)
        row = self._index.get(key)
        if row is None:
            self._index[key] = self.params.add_param(**values)
        else:
            for name, v in values.items():
                self.params._t.set(name, v, row)

    def get(self, p1, p2) -> dict | None:
        row = self._index.get(self._key(p1, p2))
        return None if row is None else self.params.row(row)

    def remove(self, p1, p2) -> None:
        self._index.pop(self._key(p1, p2), None)

    def pairs(self) -> list[tuple[int, int]]:
        return sorted(self._index)

    def items(self):
        return [(k, self.params.row(self._index[k])) for k in self.pairs()]

    def remapped(self, pmap: np.ndarray) -> OverrideTable:
        """Copy with param ids renumbered through ``pmap``; pairs losing a param are dropped."""
        out = OverrideTable()
        for name in self.params.props:
            out.params.add_prop(name, self.params.prop_type(name))
        for (p1, p2), values in self.items():
            q1, q2 = int(pmap[p1]), int(pmap[p2])
            if q1 >= 0 and q2 >= 0:
                out.set(q1, q2, **values)
        return out

    def copy(self) -> OverrideTable:
        return self.remapped(np.arange(max((max(k) for k in self._index), default=-1) + 1))


class TermTable:
    """Interactions of fixed arity, each pointing at one param row."""

    def __init__(self, name: str, natoms: int, category: str = "bond", params=None, system=None):
        if category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}, got {category!r}")
        self.name = name
        self.natoms = int(natoms)
        self.category = category
        self.params = ParamTable() if params is None else params
        self.overrides = OverrideTable()
        self._system = system
        self._epoch = 0
        self._t = ColumnTable(builtin=("atoms", "param"))
        self._t.add_column("atoms", "int", shape=(self.natoms,), default=-1)
        self._t.add_column("param", "int", default=-1)

    # ---- sizes and raw arrays ----------------------------------------
    def __len__(self) -> int:
        return len(self._t)

    @property
    def nterms(self) -> int:
        return len(self._t)

    @property
    def atoms(self) -> np.ndarray:
        """(nterms, natoms) atom indices (read-only view)."""
        v = self._t.column("atoms").view()
        v.flags.writeable = False
        return v

    @property
    def param_ids(self) -> np.ndarray:
        """Param row of each term, -1 when unset (read-only view)."""
        v = self._t.column("param").view()
        v.flags.writeable = False
        return v

    @property
    def term_props(self) -> list[str]:
        return self._t.props

    def add_term_prop(self, name: str, type=float, default=None) -> None:
        self._t.add_column(name, type, default=default)

    def del_term_prop(self, name: str) -> None:
        self._t.del_column(name)

    def term_prop_type(self, name: str) -> str:
        return self._t.kind(name)

    # ---- building -----------------------------------------------------
    def _check_atoms(self, atoms: np.ndarray) -> None:
        if atoms.shape[-1] != self.natoms:
            raise ValueError(f"table {self.name!r} takes {self.natoms} atoms per term")
        n = None if self._system is None else self._system.natoms
        if atoms.size and (atoms.min() < 0 or (n is not None and atoms.max() >= n)):
            raise IndexError(f"atom index out of range for table {self.name!r}")

    def _check_params(self, params: np.ndarray) -> None:
        if params.size and (params.min() < -1 or params.max() >= len(self.params)):
            raise IndexError(f"param index out of range for table {self.name!r}")

    def add_term(self, atoms, param=None, **props) -> Term:
        ids = np.array([int(a) for a in atoms], dtype=np.int64)
        self._check_atoms(ids)
        pid = -1 if param is None else int(param.id if isinstance(param, Param) else param)
        self._check_params(np.array([pid]))
        for name in props:
            if name not in self._t:
                raise KeyError(f"no term property {name!r}; add it with add_term_prop()")
        return Term(self, self._t.append(1, {"atoms": ids, "param": pid, **props}))

    def add_terms(self, atoms, params=None, **props) -> np.ndarray:
        """Vectorized add; ``atoms`` is (m, natoms); returns the new term indices."""
        atoms = np.asarray(atoms, dtype=np.int64).reshape(-1, self.natoms)
        self._check_atoms(atoms)
        pids = np.full(len(atoms), -1, np.int64) if params is None else np.asarray(params)
        pids = np.broadcast_to(pids, (len(atoms),)).astype(np.int64)
        self._check_params(pids)
        for name in props:
            if name not in self._t:
                raise KeyError(f"no term property {name!r}; add it with add_term_prop()")
        start = self._t.append(len(atoms), {"atoms": atoms, "param": pids, **props})
        return np.arange(start, start + len(atoms))

    def to_pandas(self):
        """Terms as a pandas DataFrame: atom0..atomN, param, the param columns, term columns."""
        import pandas as pd

        atoms, pids = self.atoms, self.param_ids
        data = {f"atom{k}": atoms[:, k].copy() for k in range(self.natoms)}
        data["param"] = pids.copy()
        missing = pids < 0
        for prop in self.params.props:
            vals = np.asarray(self.params[prop])
            col = pd.Series(vals[np.maximum(pids, 0)] if len(vals) else np.full(len(pids), None))
            if missing.any():
                col = col.astype(object)
                col[missing] = None
            data[prop] = col.to_numpy()
        for prop in self.term_props:
            key = prop if prop not in data else f"term_{prop}"
            data[key] = self._t.column(prop).copy()
        return pd.DataFrame(data)

    def set_params(self, terms, params) -> None:
        pids = np.asarray(params, dtype=np.int64)
        self._check_params(np.atleast_1d(pids))
        self._t.set("param", pids, np.asarray(terms, dtype=np.int64))

    def delete_terms(self, terms) -> None:
        keep = np.ones(len(self), bool)
        keep[np.asarray(terms, dtype=np.int64)] = False
        self._t.keep(keep)
        self._epoch += 1

    # ---- access -------------------------------------------------------
    def term(self, i: int) -> Term:
        i = int(i)
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(f"term {i} out of range for table of {len(self)}")
        return Term(self, i)

    def __getitem__(self, i) -> Term:
        return self.term(i)

    def __iter__(self):
        return (Term(self, i) for i in range(len(self)))

    @property
    def terms(self) -> list[Term]:
        return list(self)

    def values(self, prop: str) -> np.ndarray:
        """Per-term value of a term property or param property."""
        if prop in self._t.props:
            return self._t.column(prop).copy()
        if prop not in self.params.props:
            raise KeyError(f"table {self.name!r} has no property {prop!r}")
        pids = self.param_ids
        if (pids < 0).any():
            raise ValueError(f"table {self.name!r} has terms without params")
        return self.params[prop][pids]

    def find_with_all(self, atoms) -> np.ndarray:
        """Terms containing every atom in ``atoms``."""
        a = self._t.column("atoms")
        mask = np.ones(len(a), bool)
        for x in np.asarray([int(v) for v in atoms]):
            mask &= (a == x).any(axis=1)
        return np.flatnonzero(mask)

    def find_with_any(self, atoms) -> np.ndarray:
        ids = np.asarray([int(v) for v in atoms], dtype=np.int64)
        return np.flatnonzero(np.isin(self._t.column("atoms"), ids).any(axis=1))

    def find_with_only(self, atoms) -> np.ndarray:
        ids = np.asarray([int(v) for v in atoms], dtype=np.int64)
        return np.flatnonzero(np.isin(self._t.column("atoms"), ids).all(axis=1))

    def find_exact(self, atoms) -> np.ndarray:
        """Terms whose atoms are exactly ``atoms`` in the given order."""
        ids = np.asarray([int(v) for v in atoms], dtype=np.int64)
        if len(ids) != self.natoms:
            return np.empty(0, np.int64)
        return np.flatnonzero((self._t.column("atoms") == ids).all(axis=1))

    # ---- internals ----------------------------------------------------
    def _param_refcount(self, pid: int) -> int:
        tables = [self]
        if self._system is not None:
            tables += [
                t for t in self._system.tables.values() if t.params is self.params and t is not self
            ]
        return sum(int(np.count_nonzero(t._t.column("param") == pid)) for t in tables)

    def _remap_atoms(self, amap: np.ndarray) -> None:
        """Renumber atoms through ``amap``; terms touching a removed atom are dropped."""
        if len(self) == 0:
            return
        mapped = amap[self._t.column("atoms")]
        keep = np.flatnonzero((mapped >= 0).all(axis=1))
        self._t.keep(keep)
        self._t._data["atoms"][: len(keep)] = mapped[keep]
        self._epoch += 1

    def __repr__(self) -> str:
        return (
            f"<TermTable {self.name!r} {self.category} natoms={self.natoms}: "
            f"{len(self)} terms, {len(self.params)} params>"
        )


class Term:
    """One interaction in a TermTable."""

    __slots__ = ("table", "_id", "_epoch")

    def __init__(self, table: TermTable, id: int):
        self.table = table
        self._id = int(id)
        self._epoch = table._epoch

    @property
    def id(self) -> int:
        if self._epoch != self.table._epoch:
            from .handles import StaleHandleError

            raise StaleHandleError(
                f"term {self._id} of table {self.table.name!r} was renumbered; get a fresh handle"
            )
        return self._id

    @property
    def atom_ids(self) -> np.ndarray:
        return self.table._t.column("atoms")[self.id].copy()

    @property
    def atoms(self):
        sys = self.table._system
        if sys is None:
            return self.atom_ids.tolist()
        return [sys.atom(i) for i in self.atom_ids]

    @property
    def param(self) -> Param | None:
        pid = int(self.table._t.column("param")[self.id])
        return None if pid < 0 else Param(self.table.params, pid)

    @param.setter
    def param(self, value) -> None:
        pid = -1 if value is None else int(value.id if isinstance(value, Param) else value)
        self.table.set_params([self.id], [pid])

    def keys(self) -> list[str]:
        return sorted(set(self.table.params.props) | set(self.table.term_props))

    def __getitem__(self, prop: str):
        t = self.table
        if prop in t.term_props:
            return scalar(t._t.column(prop)[self.id])
        if prop not in t.params.props:
            raise KeyError(f"no property {prop!r} in table {t.name!r}")
        pid = int(t._t.column("param")[self.id])
        if pid < 0:
            raise ValueError(f"term {self.id} of {t.name!r} has no param")
        return scalar(t.params[prop][pid])

    def __setitem__(self, prop: str, value) -> None:
        """Set a property.  A param shared with other terms is copied first."""
        t = self.table
        if prop in t.term_props:
            t._t.set(prop, value, self.id)
            return
        if prop not in t.params.props:
            raise KeyError(f"no property {prop!r} in table {t.name!r}")
        pid = int(t._t.column("param")[self.id])
        if pid < 0:
            raise ValueError(f"term {self.id} of {t.name!r} has no param")
        if t._param_refcount(pid) > 1:
            pid = t.params.duplicate(pid)
            t._t.set("param", pid, self.id)
        t.params._t.set(prop, value, pid)

    def as_dict(self) -> dict:
        return {k: self[k] for k in self.keys() if k in self.table.term_props or self.param}

    def __eq__(self, other):
        return isinstance(other, Term) and other.table is self.table and other._id == self._id

    def __hash__(self):
        return hash((id(self.table), self._id))

    def __repr__(self) -> str:
        vals = " ".join(f"{k}={v!r}" for k, v in self.as_dict().items())
        return f"<Term {self.table.name}[{self._id}] atoms={tuple(self.atom_ids.tolist())} {vals}>"
