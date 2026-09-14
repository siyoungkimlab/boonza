"""Differences between two systems, after msys dms-diff and dms-diff-ff.

    for d in boonza.diff(a, b):
        print(d)
    boonza.diff(a, b, atom_map=amap, positions=False)

Atoms of ``a`` pair with atoms of ``b`` by index, or through ``atom_map``
(``atom_map[i]`` is the atom of ``b`` matching atom ``i`` of ``a``).
Force-field terms are compared as sets keyed by their mapped atoms in a
canonical order, as msys dms-diff-ff does: a term equals its reverse,
constraint partners are unordered and impropers list their center first.
Term order, parameter sharing and annotation columns (type, memo, ...) do
not matter; numbers are compared with ``rtol``/``atol``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

_IGNORED = {"memo", "comment", "annotation", "type", "typekey", "ff", "nbfix_identifier",
            "qij_annotation"}  # fmt: skip
_SHOW = 5  # examples quoted per message
_ATOM_COLUMNS = ("name", "anum", "mass", "charge", "formal_charge")


@dataclass(frozen=True)
class Difference:
    """One difference: what kind of data differs and how."""

    kind: str
    message: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.message}"


def _quote(items) -> str:
    items = list(items)
    return ", ".join(map(str, items[:_SHOW])) + (", ..." if len(items) > _SHOW else "")


def _fmt(v) -> str:
    """A value for messages: numbers to 6 significant digits, strings quoted."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        return f"{v:.6g}"
    return repr(v)


def _numeric(x: np.ndarray) -> bool:
    return x.dtype.kind in "biuf"


def _same(x, y, rtol, atol) -> np.ndarray:
    """Elementwise equality: numbers within tolerance (NaN equals NaN), others exactly."""
    x, y = np.asarray(x), np.asarray(y)
    if _numeric(x) and _numeric(y):
        return np.isclose(x.astype(float), y.astype(float), rtol=rtol, atol=atol, equal_nan=True)
    if _numeric(x) != _numeric(y):
        return np.zeros(x.shape, bool)
    return x == y


def diff(a, b, atom_map=None, rtol: float = 1e-6, atol: float = 1e-9, positions: bool = True,
         tables=None) -> list[Difference]:  # fmt: skip
    """Differences between systems ``a`` and ``b`` (empty when they match).

    ``positions=False`` ignores coordinates, velocities and the cell;
    ``tables`` limits the force-field tables compared.
    """
    out: list[Difference] = []
    if atom_map is None:
        m = np.arange(a.natoms) if a.natoms == b.natoms else None
    else:
        m = np.asarray(atom_map, dtype=np.int64)
        if (len(m) != a.natoms or (len(m) and (m.min() < 0 or m.max() >= b.natoms))
                or len(np.unique(m)) != len(m)):  # fmt: skip
            raise ValueError("atom_map must map every atom of a to a distinct atom of b")
    if a.natoms != b.natoms:
        out.append(Difference("atoms", f"{a.natoms} atoms != {b.natoms} atoms"))
    for kind, na, nb in (("residues", a.nresidues, b.nresidues), ("chains", a.nchains, b.nchains),
                         ("cts", a.ncts, b.ncts)):  # fmt: skip
        if na != nb:
            out.append(Difference(kind, f"{na} {kind} != {nb} {kind}"))
    if positions and not np.allclose(a.cell, b.cell, rtol=rtol, atol=atol):
        out.append(Difference("cell", f"{a.cell.tolist()} != {b.cell.tolist()}"))
    ia, ib = a.nonbonded_info, b.nonbonded_info
    ka = (ia.vdw_funct, ia.vdw_rule.lower(), ia.es_funct)
    kb = (ib.vdw_funct, ib.vdw_rule.lower(), ib.es_funct)
    if ka != kb:
        out.append(Difference("nonbonded_info", f"{ka} != {kb}"))
    if m is not None:
        out += _atoms(a, b, m, rtol, atol, positions)
        out += _bonds(a, b, m)
    out += _tables(a, b, m, rtol, atol, tables)
    return out


def _atoms(a, b, m, rtol, atol, positions) -> list[Difference]:
    out = []
    A, B = a.atoms, b.atoms
    cols = list(_ATOM_COLUMNS) + (["pos", "vel"] if positions else [])
    extra_a, extra_b = set(A.props), set(B.props)
    for c in cols + sorted(extra_a & extra_b):
        x, y = A[c], B[c][m]
        ok = _same(x, y, rtol, atol)
        if ok.ndim > 1:
            ok = ok.reshape(len(ok), -1).all(axis=1)
        bad = np.flatnonzero(~ok)
        if not len(bad):
            continue
        if x.ndim > 1:
            dev = np.abs(x[bad].astype(float) - y[bad].astype(float)).max()
            detail = f"max deviation {dev:g}; atoms {_quote(bad.tolist())}"
        else:
            detail = _quote(f"{i}: {_fmt(x[i])} != {_fmt(y[i])}" for i in bad.tolist())
        out.append(Difference("atoms", f"{c} differs for {len(bad)} atoms: {detail}"))
    for c in sorted(extra_a - extra_b):
        out.append(Difference("atoms", f"column {c!r} only in the first system"))
    for c in sorted(extra_b - extra_a):
        out.append(Difference("atoms", f"column {c!r} only in the second system"))

    ra, rb = A["residue"], B["residue"][m]
    ca, cb = a.residues["chain"][ra], b.residues["chain"][rb]
    fields = [("residue name", a.residues["name"][ra], b.residues["name"][rb]),
              ("resid", a.residues["resid"][ra], b.residues["resid"][rb]),
              ("insertion code", a.residues["insertion"][ra], b.residues["insertion"][rb]),
              ("chain", a.chains["name"][ca], b.chains["name"][cb]),
              ("segid", a.chains["segid"][ca], b.chains["segid"][cb])]  # fmt: skip
    for label, x, y in fields:
        bad = np.flatnonzero(x != y)
        if len(bad):
            detail = _quote(f"{i}: {_fmt(x[i])} != {_fmt(y[i])}" for i in bad.tolist())
            out.append(Difference("residues", f"{label} differs for {len(bad)} atoms: {detail}"))
    return out


def _bond_keys(s, to_ref) -> dict:
    i, j = to_ref[s.bonds["i"]], to_ref[s.bonds["j"]]
    lo, hi = np.minimum(i, j).tolist(), np.maximum(i, j).tolist()
    return dict(zip(zip(lo, hi, strict=True), s.bonds["order"].tolist(), strict=True))


def _bonds(a, b, m) -> list[Difference]:
    inv = np.full(b.natoms, -1, np.int64)
    inv[m] = np.arange(len(m))
    ka, kb = _bond_keys(a, np.arange(a.natoms)), _bond_keys(b, inv)
    kb = {k: v for k, v in kb.items() if k[0] >= 0}
    out = []
    for label, keys in (("only in the first system", ka.keys() - kb.keys()),
                        ("only in the second system", kb.keys() - ka.keys())):  # fmt: skip
        if keys:
            out.append(Difference("bonds", f"{len(keys)} bonds {label}: "
                                  f"{_quote(f'{i}-{j}' for i, j in sorted(keys))}"))  # fmt: skip
    changed = sorted(k for k in ka.keys() & kb.keys() if ka[k] != kb[k])
    if changed:
        detail = _quote(f"{i}-{j}: {ka[(i, j)]} != {kb[(i, j)]}" for i, j in changed)
        out.append(Difference("bonds", f"order differs for {len(changed)} bonds: {detail}"))
    return out


def _canonical(s, t, to_ref) -> np.ndarray:
    """Mapped term atoms in msys dms-diff-ff order."""
    own = t.atoms
    atoms = to_ref[own].copy()
    if not len(atoms) or t.natoms < 2:
        return atoms
    if t.name.startswith("constraint"):
        atoms[:, 1:] = np.sort(atoms[:, 1:], axis=1)
        return atoms
    rev = atoms[:, -1] < atoms[:, 0]
    atoms[rev] = atoms[rev, ::-1]
    if t.name.startswith("improper") and t.natoms == 4:
        for k, row in enumerate(own.tolist()):
            centers = [x for x in row if all(s.find_bond(x, y) is not None for y in row if y != x)]
            if len(centers) == 1:
                c = centers[0]
                atoms[k] = [to_ref[c], *sorted(to_ref[y] for y in row if y != c)]
    return atoms


def _grid_key(s, name) -> bytes:
    tab = s.aux_tables.get(name)
    if tab is None:
        return name.encode()
    cols = sorted(tab.props)
    return np.round(np.column_stack([tab[c].astype(float) for c in cols]), 6).tobytes()


def _values(s, t, props, registry) -> np.ndarray:
    """Per-term values of ``props`` as floats; strings become registry codes.

    CMAP ids are coded by the content of their grids, so renamed maps match.
    """
    pids = t.param_ids
    has = pids >= 0
    cols = []
    for p in props:
        if p in t.term_props:
            col = t._t.column(p)
        else:
            col = np.empty(len(t), dtype=t.params[p].dtype)
            if has.any():
                col[has] = t.params[p][pids[has]]
        if _numeric(col):
            v = col.astype(float)
        else:
            v = np.array([float(registry.setdefault(
                _grid_key(s, x) if p.startswith("cmapid") else ("str", x), len(registry)))
                for x in col.tolist()])  # fmt: skip
        if not _numeric(col) or p not in t.term_props:
            v[~has] = np.nan
        cols.append(v)
    return np.column_stack(cols) if cols else np.zeros((len(t), 0))


def _sort(keys, vals):
    order = np.lexsort(tuple(vals.T[::-1]) + tuple(keys.T[::-1])) if keys.size else np.arange(0)
    return keys[order], vals[order]


def _tables(a, b, m, rtol, atol, only) -> list[Difference]:
    out = []
    na, nb = set(a.tables), set(b.tables)
    if only is not None:
        na, nb = na & set(only), nb & set(only)
    for name in sorted(na - nb):
        out.append(Difference("tables", f"{name} only in the first system"))
    for name in sorted(nb - na):
        out.append(Difference("tables", f"{name} only in the second system"))
    inv = None
    if m is not None:
        inv = np.full(b.natoms, -1, np.int64)
        inv[m] = np.arange(len(m))
    registry: dict = {}
    for name in sorted(na & nb):
        ta, tb = a.tables[name], b.tables[name]
        if ta.natoms != tb.natoms or ta.category != tb.category:
            out.append(Difference("tables", f"{name}: {ta.category}/{ta.natoms} atoms != "
                                  f"{tb.category}/{tb.natoms} atoms"))  # fmt: skip
            continue
        if inv is None:
            if len(ta) != len(tb):
                out.append(Difference("tables", f"{name}: {len(ta)} terms != {len(tb)} terms"))
            continue
        pa = [p for p in ta.params.props if p not in _IGNORED]
        pb = [p for p in tb.params.props if p not in _IGNORED]
        for p in sorted(set(pa) ^ set(pb)):
            where = "first" if p in pa else "second"
            out.append(Difference("tables", f"{name}: parameter {p!r} only in the {where} system"))
        props = sorted(set(pa) & set(pb))
        props += sorted(set(ta.term_props) & set(tb.term_props) - _IGNORED)
        keys_a = _canonical(a, ta, np.arange(a.natoms))
        keys_b = _canonical(b, tb, inv)
        vals_a, vals_b = _values(a, ta, props, registry), _values(b, tb, props, registry)
        out += _compare_terms(name, props, keys_a, vals_a, keys_b, vals_b, rtol, atol)
        if len(ta.overrides) or len(tb.overrides):
            out += _compare_overrides(name, ta, tb, rtol, atol)
    return out


def _compare_terms(name, props, ka, va, kb, vb, rtol, atol) -> list[Difference]:
    out = []
    ka, va = _sort(ka, va)
    kb, vb = _sort(kb, vb)
    if ka.shape != kb.shape or (ka != kb).any():
        ca = Counter(map(tuple, ka.tolist()))
        cb = Counter(map(tuple, kb.tolist()))
        only_a, only_b = ca - cb, cb - ca
        for label, extra in (("only in the first system", only_a),
                             ("only in the second system", only_b)):  # fmt: skip
            if extra:
                n = sum(extra.values())
                out.append(Difference("terms", f"{name}: {n} terms {label}: "
                                      f"{_quote(sorted(extra))}"))  # fmt: skip
        common = {k for k in ca.keys() & cb.keys() if ca[k] == cb[k]}
        keep_a = np.array([tuple(r) in common for r in ka.tolist()], bool)
        keep_b = np.array([tuple(r) in common for r in kb.tolist()], bool)
        ka, va, vb = ka[keep_a], va[keep_a], vb[keep_b]
    if not len(ka) or not props:
        return out
    ok = _same(va, vb, rtol, atol)
    bad = np.flatnonzero(~ok.all(axis=1))
    if len(bad):
        with np.errstate(invalid="ignore"):
            dev = np.nanmax(np.abs(va[bad] - vb[bad]), axis=0)
        devs = ", ".join(f"{p}={d:g}" for p, d, col in zip(props, dev, ok[bad].T, strict=True)
                         if not col.all())  # fmt: skip
        examples = []
        for k in bad[:_SHOW].tolist():
            diffs = " ".join(f"{p} {va[k, c]:g}!={vb[k, c]:g}" for c, p in enumerate(props)
                             if not ok[k, c])  # fmt: skip
            examples.append(f"{tuple(ka[k].tolist())} {diffs}")
        out.append(Difference("terms", f"{name}: {len(bad)} of {len(ka)} terms differ "
                              f"(max deviation {devs}): {_quote(examples)}"))  # fmt: skip
    return out


def _compare_overrides(name, ta, tb, rtol, atol) -> list[Difference]:
    """Pair overrides keyed by the values of the two params they join."""

    def keyed(t):
        props = [p for p in t.params.props if p not in _IGNORED and _numeric(t.params[p])]
        row = {i: tuple(float(t.params[p][i]) for p in props) for i in range(len(t.params))}
        return {tuple(sorted((row[p1], row[p2]))): vals for (p1, p2), vals in t.overrides.items()}

    oa, ob = keyed(ta), keyed(tb)
    out = []
    missing = len(oa.keys() ^ ob.keys())
    if missing:
        out.append(Difference("overrides", f"{name}: {missing} pair overrides are in only one "
                              "system"))  # fmt: skip
    changed = [k for k in oa.keys() & ob.keys()
               if oa[k].keys() != ob[k].keys()
               or not all(_same(oa[k][p], ob[k][p], rtol, atol) for p in oa[k])]  # fmt: skip
    if changed:
        out.append(Difference("overrides", f"{name}: {len(changed)} pair overrides differ"))
    return out
