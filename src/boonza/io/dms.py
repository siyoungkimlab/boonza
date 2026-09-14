"""DESRES DMS files (SQLite), read and written the way msys does.

Layout: ``particle`` and ``bond`` hold the structure.  Metatables
``bond_term``, ``constraint_term``, ``virtual_term``, ``polar_term`` and
``nonbonded_table`` list term tables by name; each is stored as
``<name>_term`` (p0..pN, term props, param) plus ``<name>_param``.
Nonbonded params live in ``nonbonded_param`` referenced by
``particle.nbtype``; pair overrides in ``nonbonded_combined_param``.
Anything unrecognized is kept as an auxiliary table (cmap, forcefield...).
"""

from __future__ import annotations

import bz2
import getpass
import gzip
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from .._columns import STR, kind_of, scalar
from ..system import System
from ..terms import ParamTable

DMS_VERSION = (1, 7)
_CATEGORY_METATABLES = ("bond", "constraint", "virtual", "polar")
_PN = re.compile(r"p\d+$")
_DECL = {"int": "integer", "float": "float", "str": "text"}
_DEFAULT = {"int": 0, "float": 0.0, "str": ""}
_DTYPE = {"int": np.int64, "float": np.float64, "str": STR}
_PARTICLE_COLUMNS = {
    "id", "anum", "name", "x", "y", "z", "vx", "vy", "vz", "resname", "resid", "chain",
    "segid", "mass", "charge", "formal_charge", "insertion", "msys_ct", "nbtype",
}  # fmt: skip
_PROVENANCE = ("version", "timestamp", "user", "workdir", "cmdline", "executable")

# tables with at least this many rows are decoded by the numba scanner
FAST_MIN_ROWS = 20000


class DMSError(ValueError):
    """Malformed or unsupported DMS content."""


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _decl_kind(decl: str | None) -> str | None:
    d = (decl or "").lower()
    if "int" in d:
        return "int"
    if any(s in d for s in ("real", "floa", "doub", "num")):
        return "float"
    return "str" if d else None


def _array(values, kind: str | None):
    """sqlite column values -> (numpy array, kind); NULL becomes the default."""
    if isinstance(values, np.ndarray):  # already decoded by the fast scanner
        kind = kind or kind_of(values.dtype)
        return values.astype(_DTYPE[kind], copy=False), kind
    if kind is None:
        kind = next(
            ("int" if isinstance(v, int) else "float" if isinstance(v, float) else "str"
             for v in values if v is not None),
            "str",
        )  # fmt: skip
    if None in values:
        d = _DEFAULT[kind]
        values = [d if v is None else v for v in values]
    try:
        return np.array(values, dtype=_DTYPE[kind]), kind
    except (TypeError, ValueError):
        conv = {"int": lambda v: int(float(v)), "float": float, "str": str}[kind]
        return np.array([conv(v) for v in values], dtype=_DTYPE[kind]), kind


def _codes(values: np.ndarray) -> np.ndarray:
    """Integer codes such that equal values share a code."""
    try:
        return np.unique(values, return_inverse=True)[1].reshape(-1).astype(np.int64)
    except TypeError:
        seen: dict = {}
        return np.fromiter(
            (seen.setdefault(v, len(seen)) for v in values.tolist()), np.int64, len(values)
        )


def _factorize(*keys):
    """Group rows by several key columns; codes follow order of first appearance.

    Returns (group code per row, first row of each group).  Rows usually come
    residue by residue, so runs of identical keys are collapsed first and only
    one row per run is sorted; equal keys still share a group when they are
    not adjacent.
    """
    n = len(keys[0])
    if n == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    starts = np.zeros(n, bool)
    starts[0] = True
    for k in keys:
        starts[1:] |= k[1:] != k[:-1]
    reps = np.flatnonzero(starts)
    run = np.cumsum(starts) - 1
    code = np.zeros(len(reps), np.int64)
    for k in keys:
        c = _codes(k[reps])
        code = _codes(code * (int(c.max()) + 1) + c)
    _, first, inv = np.unique(code, return_index=True, return_inverse=True)
    appear = np.argsort(first, kind="stable")
    rank = np.empty(len(appear), np.int64)
    rank[appear] = np.arange(len(appear))
    return rank[inv.reshape(-1)][run], reps[first[appear]]


class _IdMap:
    """Maps ids stored in the file (particle ids, param ids) to row indices."""

    def __init__(self, ids, what: str):
        ids = np.asarray(ids, dtype=np.int64)
        self.n = len(ids)
        self.what = what
        self.identity = bool((ids == np.arange(self.n)).all())
        if not self.identity:
            self.order = np.argsort(ids, kind="stable")
            self.sorted = ids[self.order]

    def __call__(self, values, context: str) -> np.ndarray:
        values = np.asarray(values, dtype=np.int64)
        if self.identity:
            bad = (values < 0) | (values >= self.n)
            if bad.any():
                raise DMSError(f"{context}: invalid {self.what} id {values[bad][0]}")
            return values
        pos = np.minimum(np.searchsorted(self.sorted, values), max(self.n - 1, 0))
        bad = (self.sorted[pos] != values) if self.n else np.ones(values.shape, bool)
        if bad.any():
            raise DMSError(f"{context}: invalid {self.what} id {values[bad][0]}")
        return self.order[pos]


class _Reader:
    def __init__(self, con: sqlite3.Connection, buf: np.ndarray | None = None):
        self.con = con
        rows = con.execute(
            "select name, type, rootpage from sqlite_master where type in ('table', 'view')"
        ).fetchall()
        self.order = [n for n, t, _ in rows if t == "table"]
        self.tables = set(self.order)
        self.views = {n for n, t, _ in rows if t == "view"}
        self.roots = {n: root for n, t, root in rows if t == "table"}
        self.known: set[str] = set()
        self._buf = buf
        self._scan = None

    def has(self, name: str) -> bool:
        return name in self.tables

    def has_rows(self, name: str) -> bool:
        # msys treats an empty optional table as absent
        if name not in self.tables:
            return False
        return self.con.execute(f"select 1 from {_q(name)} limit 1").fetchone() is not None

    def columns(self, name: str) -> list[tuple[str, str | None]]:
        info = self.con.execute(f"pragma table_info({_q(name)})")
        return [(r[1], _decl_kind(r[2])) for r in info]

    def fetch(self, name: str, cols: list[str]) -> dict:
        """Column name -> values (a tuple, or a numpy array from the fast scanner)."""
        if not cols:
            return {}
        fast = self._fetch_fast(name, cols)
        if fast is not None:
            return fast
        rows = self.con.execute(f"select {', '.join(map(_q, cols))} from {_q(name)}").fetchall()
        if not rows:
            return {c: () for c in cols}
        return dict(zip(cols, zip(*rows, strict=True), strict=True))

    def names(self, meta: str) -> list[str]:
        return [r[0] for r in self.con.execute(f"select name from {_q(meta)}")]

    def _fetch_fast(self, name: str, cols: list[str]) -> dict | None:
        if self._buf is None or name not in self.roots:
            return None
        try:
            (last,) = self.con.execute(f"select max(rowid) from {_q(name)}").fetchone()
        except sqlite3.OperationalError:  # WITHOUT ROWID table
            return None
        if last is None or last < FAST_MIN_ROWS:
            return None
        info = self.con.execute(f"pragma table_info({_q(name)})").fetchall()
        wanted = set(cols)
        kinds = []
        for _, col, decl, _, _, _ in info:
            kind = _decl_kind(decl) if col in wanted else "skip"
            if kind is None:
                return None
            kinds.append(kind)
        pks = [r for r in info if r[5]]
        rowid_col = pks[0][0] if len(pks) == 1 and (pks[0][2] or "").upper() == "INTEGER" else -1
        if self._scan is None:
            from ._sqlite_scan import SqliteFile

            self._scan = SqliteFile(self._buf)
        arrays = self._scan.read(self.roots[name], kinds, rowid_col)
        if arrays is None:
            return None
        names = [r[1] for r in info]
        return {c: arrays[names.index(c)] for c in cols}


# ---------------------------------------------------------------------------
# reading


def _connect(path: str) -> tuple[sqlite3.Connection, np.ndarray | None]:
    """Open the database; also return its raw bytes for the fast scanner."""
    if path.endswith((".gz", ".bz2")):
        opener = gzip.open if path.endswith(".gz") else bz2.open
        with opener(path, "rb") as f:
            data = f.read()
        con = sqlite3.connect(":memory:")
        con.deserialize(data)
        return con, np.frombuffer(data, np.uint8)
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    con = sqlite3.connect(p.absolute().as_uri() + "?mode=ro", uri=True)
    buf = np.memmap(p, dtype=np.uint8, mode="r") if p.stat().st_size >= 100 else None
    return con, buf


def load_dms(path, structure_only: bool = False, without_tables: bool = False) -> System:
    """Read a DMS file.

    ``without_tables`` skips the force field; ``structure_only`` also drops
    pseudo particles (atomic number 0).
    """
    path = os.fspath(path)
    con, buf = _connect(path)
    try:
        return _import(_Reader(con, buf), path, structure_only, without_tables)
    finally:
        con.close()


def _import(r: _Reader, name: str, structure_only: bool, without_tables: bool) -> System:
    r.known.update(
        {"particle", "bond", "msys_hash", "dms_version", "msys_ct", "global_cell", "provenance"}
    )
    if not r.has("particle"):
        raise DMSError("missing particle table")
    s = System(name)
    ct_names, ct_props = _read_cts(r)

    cols = r.columns("particle")
    data = r.fetch("particle", [c for c, _ in cols])
    n = len(next(iter(data.values()))) if data else 0

    def get(col, kind, default):
        if col in data:
            return _array(data[col], kind)[0]
        return np.full(n, default, dtype=_DTYPE[kind])

    strip = np.strings.strip
    ct = get("msys_ct", "int", 0)
    chain, segid = strip(get("chain", "str", "")), strip(get("segid", "str", ""))
    resname, insertion = strip(get("resname", "str", "")), strip(get("insertion", "str", ""))
    resid = get("resid", "int", 0)
    if n and ct.min() < 0:
        raise DMSError("particle table has a negative msys_ct")

    chain_code, chain_first = _factorize(ct, chain, segid)
    res_code, res_first = _factorize(chain_code, resid, resname, insertion)

    ncts = max(len(ct_names), int(ct.max()) + 1 if n else 0)
    s._ct_names = ct_names + [""] * (ncts - len(ct_names))
    s._ct_props = ct_props + [{} for _ in range(ncts - len(ct_props))]
    s._chains.append(
        len(chain_first),
        {"ct": ct[chain_first], "name": chain[chain_first], "segid": segid[chain_first]},
    )
    s._residues.append(
        len(res_first),
        {
            "chain": chain_code[res_first],
            "resid": resid[res_first],
            "name": resname[res_first],
            "insertion": insertion[res_first],
        },
    )
    s._atoms.append(
        n,
        {
            "residue": res_code,
            "name": strip(get("name", "str", "")),
            "anum": get("anum", "int", 0),
            "pos": np.column_stack([get(c, "float", 0.0) for c in ("x", "y", "z")]),
            "vel": np.column_stack([get(c, "float", 0.0) for c in ("vx", "vy", "vz")]),
            "mass": get("mass", "float", 0.0),
            "charge": get("charge", "float", 0.0),
            "formal_charge": get("formal_charge", "int", 0),
        },
    )
    for col, kind in cols:
        if col in _PARTICLE_COLUMNS:
            continue
        if col in s._atoms:
            raise DMSError(f"particle column {col!r} collides with a built-in atom column")
        arr, kind = _array(data[col], kind)
        s._atoms.add_column(col, kind)
        s._atoms.set(col, arr)
    atom_ids = _IdMap(get("id", "int", 0) if "id" in data else np.arange(n), "particle")

    _read_bonds(r, s, atom_ids)
    s._cell = _read_cell(r)
    s.provenance = _read_provenance(r)
    if not (without_tables or structure_only):
        _read_metatables(r, s, atom_ids)
        nb_ids = _read_nonbonded(r, s, data.get("nbtype"), n)
        _read_combined(r, s, nb_ids)
        _read_exclusions(r, s, atom_ids)
        _read_nbinfo(r, s)
        _read_aux(r, s)
    if structure_only and n and (s._atoms.column("anum") <= 0).any():
        s = s.clone(structure_only=True)
    return s


def _read_cts(r: _Reader):
    r.known.add("msys_ct")
    if not r.has("msys_ct"):
        return [], []
    cols = [c for c, _ in r.columns("msys_ct")]
    names: dict[int, str] = {}
    props: dict[int, dict] = {}
    for row in r.con.execute("select * from msys_ct"):
        rec = dict(zip(cols, row, strict=True))
        cid = int(rec.pop("id"))
        names[cid] = rec.pop("msys_name", "") or ""
        kv: dict = {}
        for key, val in rec.items():
            val = "" if val is None else val
            key = key.split("_msys_converted_")[0]
            if key in kv:
                # same key written twice (old and new msys conventions): prefer
                # the non-empty value, concatenate two non-empty ones
                if kv[key] == "":
                    kv[key] = val
                elif val != "":
                    kv[key] = f"{kv[key]}|{val}"
                continue
            kv[key] = val
        props[cid] = kv
    count = max(names) + 1 if names else 0
    return [names.get(i, "") for i in range(count)], [props.get(i, {}) for i in range(count)]


def _read_bonds(r: _Reader, s: System, atom_ids: _IdMap) -> None:
    if not r.has("bond"):
        return
    cols = r.columns("bond")
    data = r.fetch("bond", [c for c, _ in cols])
    if "p0" not in data or "p1" not in data:
        raise DMSError("bond table needs p0 and p1 columns")
    p0 = atom_ids(_array(data["p0"], "int")[0], "bond table")
    p1 = atom_ids(_array(data["p1"], "int")[0], "bond table")
    if (p0 == p1).any():
        raise DMSError("bond table bonds an atom to itself")
    order = _array(data["order"], "int")[0] if "order" in data else np.ones(len(p0), np.int64)
    lo, hi = np.minimum(p0, p1), np.maximum(p0, p1)
    _, first = np.unique((lo << 32) | hi, return_index=True)
    first.sort()
    s._bonds.append(len(first), {"i": lo[first], "j": hi[first], "order": order[first]})
    for col, kind in cols:
        if col in ("p0", "p1", "order"):
            continue
        if col in s._bonds:
            raise DMSError(f"bond column {col!r} collides with a built-in bond column")
        arr, kind = _array(data[col], kind)
        s._bonds.add_column(col, kind)
        s._bonds.set(col, arr[first])


def _read_cell(r: _Reader) -> np.ndarray:
    cell = np.zeros((3, 3))
    if r.has("global_cell"):
        rows = r.con.execute("select x, y, z from global_cell").fetchall()
        if len(rows) > 3:
            raise DMSError("global_cell table has too many rows")
        for k, row in enumerate(rows):
            cell[k] = [0.0 if v is None else v for v in row]
    return cell


def _read_provenance(r: _Reader) -> list[dict]:
    if not r.has("provenance"):
        return []
    cols = [c for c, _ in r.columns("provenance") if c in _PROVENANCE]
    data = r.fetch("provenance", cols)
    count = len(next(iter(data.values()))) if data else 0
    return [{c: (data[c][k] or "") for c in cols} for k in range(count)]


def _read_metatables(r: _Reader, s: System, atom_ids: _IdMap) -> None:
    for category in _CATEGORY_METATABLES:
        meta = f"{category}_term"
        r.known.add(meta)
        if r.has(meta):
            for name in r.names(meta):
                _read_table(r, s, category, name, atom_ids)
    r.known.add("nonbonded_table")
    if r.has("nonbonded_table"):
        for name in r.names("nonbonded_table"):
            _read_table(r, s, "nonbonded", name, atom_ids)


def _read_params(r: _Reader, table: str, params: ParamTable, ignore_ids: bool = True) -> _IdMap:
    cols = r.columns(table)
    data = r.fetch(table, [c for c, _ in cols])
    n = len(next(iter(data.values()))) if data else 0
    ids = None
    values = {}
    for col, kind in cols:
        if ignore_ids and col == "id":
            ids = _array(data[col], "int")[0]
            continue
        arr, kind = _array(data[col], kind)
        params.add_prop(col, kind)
        values[col] = arr
    params.add_params(n, **values)
    return _IdMap(np.arange(n) if ids is None else ids, f"{table} param")


def _read_table(r: _Reader, s: System, category: str, table: str, atom_ids: _IdMap) -> None:
    term_table, param_table = f"{table}_term", f"{table}_param"
    r.known.update({term_table, param_table})
    if not r.has(term_table):
        term_table = table
        r.known.add(table)
        if not (r.has(table) or table in r.views):
            raise DMSError(f"{category} table {table!r} not found")
    cols = r.columns(term_table)
    names = [c for c, _ in cols]
    if "paramA" in names or "paramB" in names:
        raise NotImplementedError(
            f"alchemical table {table!r}: alchemical DMS is not supported yet"
        )
    if table in s.tables:
        raise DMSError(f"table {table!r} is listed twice")
    pcols = [c for c in names if _PN.match(c)]
    extras = [(c, k) for c, k in cols if not _PN.match(c) and c != "param"]
    data = r.fetch(term_table, names)
    nterms = len(data[names[0]]) if names else 0
    context = f"table {table!r}"
    atoms = np.empty((nterms, len(pcols)), np.int64)
    for k, c in enumerate(pcols):
        atoms[:, k] = atom_ids(_array(data[c], "int")[0], context)

    t = s.add_table(table, len(pcols), category)
    term_props = extras
    if "param" in names:
        if not r.has(param_table):
            raise DMSError(f"missing param table {param_table!r}")
        pmap = _read_params(r, param_table, t.params)
        params = pmap(_array(data["param"], "int")[0], context)
    elif r.has(param_table):
        _read_params(r, param_table, t.params)
        params = np.full(nterms, -1, np.int64)
    else:
        # no param table: the extra columns are per-term parameters
        values = {c: _array(data[c], k) for c, k in extras}
        for c, (_, kind) in values.items():
            t.params.add_prop(c, kind)
        params = t.params.add_params(nterms, **{c: arr for c, (arr, _) in values.items()})
        term_props = []
    t._t.append(nterms, {"atoms": atoms, "param": params})
    for c, kind in term_props:
        arr, kind = _array(data[c], kind)
        t.add_term_prop(c, kind)
        t._t.set(c, arr)


def _read_nonbonded(r: _Reader, s: System, nbtype, n: int) -> _IdMap | None:
    r.known.update({"nonbonded_param", "alchemical_particle"})
    if not r.has_rows("nonbonded_param"):
        return None
    if r.has("alchemical_particle"):
        raise NotImplementedError("alchemical_particle: alchemical DMS is not supported yet")
    t = s.add_table("nonbonded", 1, "nonbonded")
    pmap = _read_params(r, "nonbonded_param", t.params)
    if nbtype is None:
        params = np.full(n, -1, np.int64)
    else:
        params = pmap(_array(nbtype, "int")[0], "particle nbtype")
    t._t.append(n, {"atoms": np.arange(n).reshape(n, 1), "param": params})
    return pmap


def _read_combined(r: _Reader, s: System, pmap: _IdMap | None) -> None:
    table = "nonbonded_combined_param"
    r.known.add(table)
    if not r.has_rows(table):
        return
    t = s.tables.get("nonbonded")
    if t is None or pmap is None:
        raise DMSError(f"{table} without nonbonded_param")
    cols = r.columns(table)
    data = r.fetch(table, [c for c, _ in cols])
    if "param1" not in data or "param2" not in data:
        raise DMSError(f"{table} needs param1 and param2 columns")
    p1 = pmap(_array(data["param1"], "int")[0], table)
    p2 = pmap(_array(data["param2"], "int")[0], table)
    values = {c: _array(data[c], k) for c, k in cols if c not in ("param1", "param2")}
    for c, (_, kind) in values.items():
        t.overrides.params.add_prop(c, kind)
    for k in range(len(p1)):
        row = {c: scalar(arr[k]) for c, (arr, _) in values.items()}
        old = t.overrides.get(p1[k], p2[k])
        if old is not None and old != row:
            raise DMSError(f"conflicting {table} entries for params {p1[k]}, {p2[k]}")
        t.overrides.set(p1[k], p2[k], **row)


def _read_exclusions(r: _Reader, s: System, atom_ids: _IdMap) -> None:
    r.known.update({"exclusion", "exclusion_term", "exclusion_param"})
    if not r.has_rows("exclusion"):
        return
    cols = [c for c, _ in r.columns("exclusion")][:2]
    data = r.fetch("exclusion", cols)
    a = atom_ids(_array(data[cols[0]], "int")[0], "exclusion table")
    b = atom_ids(_array(data[cols[1]], "int")[0], "exclusion table")
    t = s.add_table("exclusion", 2, "exclusion")
    t._t.append(len(a), {"atoms": np.column_stack([a, b]), "param": -1})


def _read_nbinfo(r: _Reader, s: System) -> None:
    r.known.add("nonbonded_info")
    if not r.has("nonbonded_info"):
        return
    cols = [c for c, _ in r.columns("nonbonded_info")]
    row = r.con.execute("select * from nonbonded_info").fetchone()
    if row is None:
        return
    rec = dict(zip(cols, row, strict=True))
    for field in ("vdw_funct", "vdw_rule", "es_funct"):
        setattr(s.nonbonded_info, field, rec.get(field) or "")


def _read_aux(r: _Reader, s: System) -> None:
    for name in r.order:
        if name in r.known or name.startswith("sqlite_") or not r.has_rows(name):
            continue
        params = ParamTable()
        _read_params(r, name, params, ignore_ids=False)
        s.aux_tables[name] = params


# ---------------------------------------------------------------------------
# writing


def save_dms(system: System, path, structure_only: bool = False) -> None:
    """Write a DMS file; ``.gz`` / ``.bz2`` suffixes are compressed."""
    path = os.fspath(path)
    compressed = path.endswith((".gz", ".bz2"))
    if os.path.exists(path):
        os.unlink(path)
    con = sqlite3.connect(":memory:" if compressed else path, isolation_level=None)
    try:
        con.execute("pragma journal_mode=off")
        con.execute("pragma synchronous=off")
        con.execute("begin")
        _export(con, system, structure_only)
        con.execute("commit")
        if compressed:
            opener = gzip.open if path.endswith(".gz") else bz2.open
            with opener(path, "wb") as f:
                f.write(con.serialize())
    except BaseException:
        con.close()
        if os.path.exists(path):
            os.unlink(path)
        raise
    con.close()


def _create(con: sqlite3.Connection, table: str, columns: list[tuple[str, str, object]]) -> None:
    """Create ``table`` from (name, declaration, values) triples and fill it."""
    defs = ", ".join(f"{_q(name)} {decl}" for name, decl, _ in columns)
    con.execute(f"create table {_q(table)} ({defs})")
    lists = [v.tolist() if isinstance(v, np.ndarray) else list(v) for _, _, v in columns]
    if lists and lists[0]:
        names = ", ".join(_q(name) for name, _, _ in columns)
        marks = ", ".join("?" * len(columns))
        con.executemany(
            f"insert into {_q(table)} ({names}) values ({marks})", zip(*lists, strict=True)
        )


def _scalar_column(table, name: str, what: str) -> tuple[str, str, np.ndarray]:
    kind, shape, _ = table.spec(name)
    if shape:
        raise DMSError(f"{what} property {name!r} has shape {shape}; DMS columns must be scalar")
    return (name, _DECL[kind], table.column(name))


def _export(con: sqlite3.Connection, s: System, structure_only: bool) -> None:
    _export_cts(con, s)
    _export_cell(con, s)
    _export_particles(con, s, structure_only)
    _export_bonds(con, s)
    if not structure_only:
        _export_tables(con, s)
        for name, params in s.aux_tables.items():
            _export_params(con, params, name, with_id=False)
        nb = s.nonbonded_info
        _create(
            con,
            "nonbonded_info",
            [("vdw_funct", "text", [nb.vdw_funct]), ("vdw_rule", "text", [nb.vdw_rule]),
             ("es_funct", "text", [nb.es_funct])],
        )  # fmt: skip
    _export_provenance(con, s)
    _create(con, "dms_version", [
        ("major", "integer not null", [DMS_VERSION[0]]),
        ("minor", "integer not null", [DMS_VERSION[1]]),
    ])  # fmt: skip


def _export_cts(con: sqlite3.Connection, s: System) -> None:
    # sqlite column names are case-insensitive; disambiguate colliding keys
    # with the suffix msys strips on load
    keys = sorted({k for props in s._ct_props for k in props})
    seen = {"id", "msys_name"}
    columns = {}
    for salt, key in enumerate(keys):
        low = key.lower()
        columns[key] = f"{key}_msys_converted_{salt}" if low in seen else key
        seen.add(low)
    cols = [
        ("id", "integer primary key", range(s.ncts)),
        ("msys_name", "text", s._ct_names),
    ]
    for key, colname in columns.items():
        values = [scalar(p.get(key, "")) for p in s._ct_props]
        cols.append((colname, _ct_decl(values), values))
    _create(con, "msys_ct", cols)


def _ct_decl(values) -> str:
    # declare the stored type so numeric ct values read back as numbers
    types = {type(v) for v in values}
    if types <= {int, bool}:
        return "integer"
    if types <= {int, float}:
        return "float"
    return "text"


def _export_cell(con: sqlite3.Connection, s: System) -> None:
    c = s.cell
    _create(con, "global_cell", [
        ("id", "integer primary key", [1, 2, 3]),
        ("x", "float", c[:, 0]), ("y", "float", c[:, 1]), ("z", "float", c[:, 2]),
    ])  # fmt: skip


def _nbtypes(s: System) -> np.ndarray | None:
    t = s.tables.get("nonbonded")
    if t is None:
        return None
    if t.natoms != 1:
        raise DMSError("nonbonded table must have one atom per term")
    atoms = t.atoms[:, 0]
    counts = np.bincount(atoms, minlength=s.natoms)
    if (counts > 1).any():
        raise DMSError(f"nonbonded table lists particle {np.argmax(counts > 1)} more than once")
    nb = np.full(s.natoms, -1, np.int64)
    nb[atoms] = t.param_ids
    bad = np.flatnonzero((nb < 0) | (nb >= len(t.params)))
    if len(bad):
        raise DMSError(f"particle {bad[0]} has no valid nonbonded param")
    return nb


def _export_particles(con: sqlite3.Connection, s: System, structure_only: bool) -> None:
    nbtypes = None if structure_only else _nbtypes(s)
    A, R, C = s._atoms, s._residues, s._chains
    res = A.column("residue")
    chn = R.column("chain")[res]
    pos, vel = A.column("pos"), A.column("vel")
    cols = [
        ("id", "integer primary key", np.arange(s.natoms)),
        ("anum", "integer", A.column("anum")),
        ("name", "text not null", A.column("name")),
        ("x", "float", pos[:, 0]),
        ("y", "float", pos[:, 1]),
        ("z", "float", pos[:, 2]),
        ("vx", "float", vel[:, 0]),
        ("vy", "float", vel[:, 1]),
        ("vz", "float", vel[:, 2]),
        ("resname", "text not null", R.column("name")[res]),
        ("resid", "integer", R.column("resid")[res]),
        ("chain", "text not null", C.column("name")[chn]),
        ("segid", "text not null", C.column("segid")[chn]),
        ("mass", "float", A.column("mass")),
        ("charge", "float", A.column("charge")),
        ("formal_charge", "integer", A.column("formal_charge")),
        ("insertion", "text not null", R.column("insertion")[res]),
        ("msys_ct", "integer not null", C.column("ct")[chn]),
    ]
    cols += [_scalar_column(A, p, "atom") for p in A.props]
    if nbtypes is not None:
        cols.append(("nbtype", "integer not null", nbtypes))
    _create(con, "particle", cols)


def _export_bonds(con: sqlite3.Connection, s: System) -> None:
    B = s._bonds
    cols = [
        ("p0", "integer", B.column("i")),
        ("p1", "integer", B.column("j")),
        ("order", "integer", B.column("order")),
    ]
    cols += [_scalar_column(B, p, "bond") for p in B.props]
    _create(con, "bond", cols)


def _export_params(con, params: ParamTable, table: str, with_id: bool = True) -> None:
    props = params.props
    if not props and not with_id:
        return
    cols = [_scalar_column(params._t, p, f"{table} param") for p in props]
    if with_id:
        cols.append(("id", "integer primary key", np.arange(len(params))))
    _create(con, table, cols)


def _export_terms(con, t, table: str) -> None:
    pids = t.param_ids
    bad = np.flatnonzero((pids < 0) | (pids >= len(t.params)))
    if len(bad):
        raise DMSError(f"table {t.name!r} term {bad[0]} has a missing or invalid param")
    atoms = t.atoms
    cols = [(f"p{k}", "integer", atoms[:, k]) for k in range(t.natoms)]
    cols += [_scalar_column(t._t, p, f"{t.name} term") for p in t.term_props]
    cols.append(("param", "integer not null", pids))
    _create(con, table, cols)


def _export_view(con, t, name: str) -> None:
    props = [c for c in t.params.props if c != "id"] + [c for c in t.term_props if c != "param"]
    select = [f"p{k}" for k in range(t.natoms)] + [_q(c) for c in props]
    if select:
        con.execute(
            f"create view {_q(name)} as select {', '.join(select)} "
            f"from {_q(name + '_param')} join {_q(name + '_term')} on param=id"
        )


def _export_tables(con: sqlite3.Connection, s: System) -> None:
    for category in _CATEGORY_METATABLES:
        con.execute(f"create table {category}_term (name text)")
    have_exclusion = have_nbtable = False
    for name in sorted(s.tables):
        t = s.tables[name]
        if t.category == "exclusion":
            if t.natoms != 2 or have_exclusion:
                raise DMSError("DMS holds exactly one exclusion table with two atoms per term")
            atoms = t.atoms
            _create(
                con, "exclusion", [("p0", "integer", atoms[:, 0]), ("p1", "integer", atoms[:, 1])]
            )
            have_exclusion = True
        elif t.category == "nonbonded" and name == "nonbonded":
            _export_params(con, t.params, "nonbonded_param")
            _export_overrides(con, t)
        else:
            _export_terms(con, t, f"{name}_term")
            _export_params(con, t.params, f"{name}_param")
            _export_view(con, t, name)
            if t.category == "nonbonded":
                if not have_nbtable:
                    con.execute("create table nonbonded_table (name text)")
                    have_nbtable = True
                con.execute("insert into nonbonded_table values (?)", (name,))
            else:
                con.execute(f"insert into {t.category}_term values (?)", (name,))


def _export_overrides(con, t) -> None:
    if not len(t.overrides):
        return
    items = t.overrides.items()
    oparams = t.overrides.params
    cols = [
        ("param1", "integer", [k[0] for k, _ in items]),
        ("param2", "integer", [k[1] for k, _ in items]),
    ]
    cols += [(p, _DECL[oparams.prop_type(p)], [v[p] for _, v in items]) for p in oparams.props]
    _create(con, "nonbonded_combined_param", cols)


def _export_provenance(con: sqlite3.Connection, s: System) -> None:
    from .. import __version__

    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    entry = {
        "version": f"boonza {__version__}",
        "timestamp": time.asctime(),
        "user": user,
        "workdir": os.getcwd(),
        "cmdline": " ".join(sys.argv),
        "executable": sys.executable,
    }
    rows = [*s.provenance, entry]
    cols = [("id", "integer primary key", range(1, len(rows) + 1))]
    cols += [(f, "text", [row.get(f, "") for row in rows]) for f in _PROVENANCE]
    _create(con, "provenance", cols)
