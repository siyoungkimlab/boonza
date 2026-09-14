"""Force-field parameters of chosen atoms: charges, van der Waals, pairs and bonded terms.

    report = boonza.describe(system, "resname LIG and name C1 C2")
    print(report)                                # text tables
    report.atoms, report.pairs, report.terms     # the same rows as dicts

Values are in DMS units (Å, degrees, kcal/mol, e).  Pairs cover every pair
of chosen atoms: the van der Waals parameters from the combining rule or a
pair override (NBFIX), whether the pair is excluded, and for scaled pairs
(``pair_12_6_es``, usually 1-4) the effective sigma/epsilon and the scale
factors relative to the full interaction.

Pairs also carry their topological distance (``bonds``: bonds along the
shortest path, None when not connected), their distance ``r`` and the bare
pair energy at the current positions (``e_vdw``, ``e_es``, ``energy``):
full Lennard-Jones and Coulomb unless excluded, plus any scaled pair term,
without cutoff or periodic images.  Summed over all pairs of a molecule it
is the nonbonded energy OpenMM computes with NoCutoff.

    d = boonza.topological_distances(s, "resname LIG")     # bonds apart, -1 if unconnected
    frames = report.to_pandas()                            # {"atoms", "pairs", tables...}
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from ._columns import scalar
from .elements import msys_symbol

FORMS = {
    "stretch_harm": "fc (r - r0)^2",
    "angle_harm": "fc (theta - theta0)^2",
    "dihedral_trig": "fc0 + sum_n fcn cos(n phi - phi0)",
    "improper_harm": "fc (phi - phi0)^2",
    "pair_12_6_es": "aij/r^12 - bij/r^6 + qij/r",
    "posre_harm": "0.5 (fcx (x - x0)^2 + fcy (y - y0)^2 + fcz (z - z0)^2)",
    "torsiontorsion_cmap": "cmap(phi, psi)",
    "vdw_12_6": "4 epsilon ((sigma/r)^12 - (sigma/r)^6)",
}
_MAX_AUTO_PAIRS = 30  # atoms; above this pairs are only listed when asked for
COULOMB = 138.935456 * 10.0 / 4.184  # kcal Å / (mol e^2), OpenMM's value


@dataclass
class ForceFieldReport:
    """Rows describing the force field of a set of atoms (see ``describe``)."""

    nonbonded_info: dict
    atoms: list[dict]
    pairs: list[dict] = field(default_factory=list)
    terms: dict[str, list[dict]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"nonbonded_info": self.nonbonded_info, "atoms": self.atoms,
                "pairs": self.pairs, "terms": self.terms}  # fmt: skip

    def __str__(self) -> str:
        info = self.nonbonded_info
        funct = info.get("vdw_funct") or "none"
        head = f"units: Å, degrees, kcal/mol, e.  nonbonded {funct}"
        if funct in FORMS:
            head += f" ({FORMS[funct]})"
        if info.get("vdw_rule"):
            head += f", combining rule {info['vdw_rule']}"
        out = [head]
        out += ["", "atoms", _table(self.atoms)]
        if self.pairs:
            out += ["", "pairs", _table(self.pairs)]
        for name, rows in self.terms.items():
            form = f"  E = {FORMS[name]}" if name in FORMS else ""
            out += ["", f"{name} ({len(rows)} terms){form}", _table(_drop_zero_fc(rows))]
        return "\n".join(out)

    def to_pandas(self) -> dict:
        """pandas DataFrames: "atoms", "pairs" and one per term table (atoms as atom0, atom1...)."""
        import pandas as pd

        out = {"atoms": pd.DataFrame(self.atoms), "pairs": pd.DataFrame(self.pairs)}
        for name, rows in self.terms.items():
            df = pd.DataFrame(rows)
            if len(df):
                atoms = pd.DataFrame(df.pop("atoms").tolist())
                atoms.columns = [f"atom{k}" for k in range(atoms.shape[1])]
                df = pd.concat([atoms, df], axis=1)
            out[name] = df
        return out

    def __repr__(self) -> str:
        nterms = sum(len(r) for r in self.terms.values())
        return (f"<ForceFieldReport {len(self.atoms)} atoms, {len(self.pairs)} pairs, "
                f"{nterms} terms in {len(self.terms)} tables>")  # fmt: skip


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, tuple | list):
        return ",".join(map(str, v))
    return str(v)


def _table(rows: list[dict]) -> str:
    if not rows:
        return "  (none)"
    cols = list(dict.fromkeys(k for r in rows for k in r))
    cells = [[_fmt(r.get(c)) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[k]) for row in cells)) for k, c in enumerate(cols)]
    lines = ["  " + "  ".join(c.rjust(w) for c, w in zip(cols, widths, strict=True))]
    lines += ["  " + "  ".join(v.rjust(w) for v, w in zip(row, widths, strict=True))
              for row in cells]  # fmt: skip
    return "\n".join(lines)


def _drop_zero_fc(rows: list[dict]) -> list[dict]:
    """Hide Fourier-coefficient columns that are zero in every row."""
    zero = {k for k in (rows[0] if rows else {}) if re.fullmatch(r"fc\d+", k)
            and all(r.get(k) == 0 for r in rows)}  # fmt: skip
    return [{k: v for k, v in r.items() if k not in zero} for r in rows]


def _atom_ids(system, atoms) -> np.ndarray:
    if atoms is None:
        return np.arange(system.natoms)
    if isinstance(atoms, str):
        return system.select(atoms).ids
    return np.unique(system._ids("atom", atoms))


@njit(cache=True)
def _bfs(indptr, indices, sources, target_pos, ntargets, max_depth):
    n = indptr.shape[0] - 1
    out = np.full((sources.shape[0], ntargets), -1, np.int64)
    dist = np.full(n, -1, np.int64)
    queue = np.empty(n, np.int64)
    for s in range(sources.shape[0]):
        head, tail = 0, 1
        queue[0] = sources[s]
        dist[sources[s]] = 0
        found = 0
        while head < tail:
            a = queue[head]
            head += 1
            d = dist[a]
            if target_pos[a] >= 0:
                out[s, target_pos[a]] = d
                found += 1
                if found == ntargets:
                    break
            if max_depth >= 0 and d >= max_depth:
                continue
            for k in range(indptr[a], indptr[a + 1]):
                b = indices[k]
                if dist[b] < 0:
                    dist[b] = d + 1
                    queue[tail] = b
                    tail += 1
        for k in range(tail):
            dist[queue[k]] = -1
    return out


def topological_distances(system, atoms=None, targets=None, max_distance=None) -> np.ndarray:
    """Bonds along the shortest bond path, shape (len(atoms), len(targets)).

    ``atoms`` and ``targets`` are indices, AtomSels or selection strings
    (``targets`` defaults to ``atoms``, ``atoms`` to every atom).  -1 marks
    pairs that are not connected, or farther apart than ``max_distance``.
    """

    def ids_of(sel):
        if sel is None:
            return np.arange(system.natoms)
        if isinstance(sel, str):
            return system.select(sel).ids
        return np.asarray(system._ids("atom", sel), dtype=np.int64)

    src = ids_of(atoms)
    tgt = src if targets is None else ids_of(targets)
    n = system.natoms
    bi = np.asarray(system.bonds["i"], np.int64)
    bj = np.asarray(system.bonds["j"], np.int64)
    heads, tails = np.concatenate([bi, bj]), np.concatenate([bj, bi])
    order = np.argsort(heads, kind="stable")
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(heads, minlength=n), out=indptr[1:])
    uniq, inv = np.unique(tgt, return_inverse=True)
    target_pos = np.full(n, -1, np.int64)
    target_pos[uniq] = np.arange(len(uniq))
    depth = -1 if max_distance is None else int(max_distance)
    out = _bfs(indptr, tails[order], src.astype(np.int64), target_pos, len(uniq), depth)
    return out[:, inv.reshape(-1)]


def _labeler(s):
    res = s.atoms["residue"]
    rname, rid, ins, rchain = (s.residues[k] for k in ("name", "resid", "insertion", "chain"))
    cname, aname = s.chains["name"], s.atoms["name"]

    def label(i: int) -> str:
        r = res[i]
        chain = str(cname[rchain[r]])
        return f"{chain + ':' if chain else ''}{rname[r]}{rid[r]}{ins[r]}:{aname[i]}"

    return label


def _pair_lookup(table, ids) -> dict:
    """Terms of a 2-atom table with both atoms in ``ids``, keyed by sorted pair -> term rows."""
    if table is None or len(table) == 0:
        return {}
    a = np.sort(table.atoms, axis=1)
    rows = np.flatnonzero(np.isin(a, ids).all(axis=1))
    out: dict = {}
    for k, (i, j) in zip(rows.tolist(), a[rows].tolist(), strict=True):
        out.setdefault((i, j), []).append(k)
    return out


def describe(system, atoms=None, terms: str = "any", pairs: bool | None = None,
             tables=None) -> ForceFieldReport:  # fmt: skip
    """Force-field report for ``atoms`` (indices, AtomSel or selection string).

    ``terms="any"`` lists every term touching one of the atoms; ``"all"``
    only terms whose atoms are all chosen.  ``pairs`` lists every pair of
    chosen atoms (default: when at most 30 atoms are chosen).  ``tables``
    limits the term tables reported.
    """
    if terms not in ("any", "all"):
        raise ValueError("terms must be 'any' or 'all'")
    s = system
    ids = _atom_ids(s, atoms)
    label = _labeler(s)
    info = s.nonbonded_info
    nbinfo = {"vdw_funct": info.vdw_funct, "vdw_rule": info.vdw_rule, "es_funct": info.es_funct}
    nb = s.tables.get("nonbonded")
    ptype = np.full(s.natoms, -1, np.int64)
    if nb is not None:
        ptype[nb.atoms[:, 0]] = nb.param_ids
    anum, mass, charge = s.atoms["anum"], s.atoms["mass"], s.atoms["charge"]

    atom_rows = []
    for i in ids.tolist():
        row = {"index": i, "label": label(i), "element": msys_symbol(int(anum[i])),
               "mass": float(mass[i]), "charge": float(charge[i])}  # fmt: skip
        if nb is not None and ptype[i] >= 0:
            row["param"] = int(ptype[i])
            row.update(nb.params.row(ptype[i]))
        atom_rows.append(row)

    pair_rows = []
    if pairs or (pairs is None and len(ids) <= _MAX_AUTO_PAIRS):
        pair_rows = _pairs(s, ids, label, nb, ptype, info.vdw_funct.lower() or "vdw_12_6",
                           info.vdw_rule.lower())  # fmt: skip

    term_rows: dict[str, list[dict]] = {}
    for name in sorted(s.tables):
        t = s.tables[name]
        if t.category in ("nonbonded", "exclusion") or (tables is not None and name not in tables):
            continue
        inside = np.isin(t.atoms, ids)
        found = np.flatnonzero(inside.any(axis=1) if terms == "any" else inside.all(axis=1))
        if not len(found):
            continue
        pids = t.param_ids
        rows = []
        for k in found.tolist():
            atoms_k = tuple(t.atoms[k].tolist())
            row = {"atoms": atoms_k, "labels": " ".join(label(a) for a in atoms_k)}
            if pids[k] >= 0:
                row["param"] = int(pids[k])
                row.update(t.params.row(pids[k]))
            row.update({p: scalar(t._t.column(p)[k]) for p in t.term_props})
            rows.append(row)
        term_rows[name] = rows
    return ForceFieldReport(nbinfo, atom_rows, pair_rows, term_rows)


def _pairs(s, ids, label, nb, ptype, funct, rule) -> list[dict]:
    excl = _pair_lookup(s.tables.get("exclusion"), ids)
    pt = s.tables.get("pair_12_6_es")
    scaled = _pair_lookup(pt, ids)
    lj = nb is not None and funct == "vdw_12_6" and {"sigma", "epsilon"} <= set(nb.params.props)
    if lj:
        psig, peps = nb.params["sigma"], nb.params["epsilon"]
    charge = s.atoms["charge"]
    pos = s.positions
    topo = topological_distances(s, ids)
    rows = []
    for ka, a in enumerate(ids.tolist()):
        for kb in range(ka + 1, len(ids)):
            b = int(ids[kb])
            qq = float(charge[a] * charge[b])
            hops = int(topo[ka, kb])
            r = float(np.sqrt(((pos[b] - pos[a]) ** 2).sum()))
            excluded = (a, b) in excl
            row = {"i": a, "j": b, "label_i": label(a), "label_j": label(b),
                   "bonds": hops if hops >= 0 else None, "r": r, "qq": qq,
                   "excluded": excluded}  # fmt: skip
            sig = eps = None
            if lj and ptype[a] >= 0 and ptype[b] >= 0:
                pa, pb = int(ptype[a]), int(ptype[b])
                ov = nb.overrides.get(pa, pb)
                if ov is not None:
                    sig, eps = float(ov["sigma"]), float(ov["epsilon"])
                else:
                    sa, sb, ea, eb = psig[pa], psig[pb], peps[pa], peps[pb]
                    sig = float(math.sqrt(sa * sb) if rule == "geometric" else 0.5 * (sa + sb))
                    eps = float(math.sqrt(ea * eb))
                row.update(sigma=sig, epsilon=eps, nbfix=ov is not None)
            # bare pair energy: full LJ + Coulomb unless excluded, plus the scaled pair term
            e_vdw = 0.0 if (excluded or sig is not None) else None
            e_es = 0.0
            if not excluded and r > 0:
                e_es += COULOMB * qq / r
                if sig is not None:
                    sr6 = (sig / r) ** 6
                    e_vdw += 4.0 * eps * (sr6 * sr6 - sr6)
            if (a, b) in scaled:
                vals = [pt.params.row(pt.param_ids[k]) for k in scaled[(a, b)]]
                aij, bij, qij = (sum(float(v[p]) for v in vals) for p in ("aij", "bij", "qij"))
                row.update(aij=aij, bij=bij, qij=qij)
                s14 = (aij / bij) ** (1 / 6) if aij > 0 and bij > 0 else None
                e14 = bij * bij / (4 * aij) if aij > 0 else None
                row.update(sigma_pair=s14, epsilon_pair=e14,
                           lj_scale=e14 / eps if e14 is not None and eps else None,
                           es_scale=qij / qq if qq else None)  # fmt: skip
                if r > 0:
                    if e_vdw is not None:
                        e_vdw += aij / r**12 - bij / r**6
                    e_es += COULOMB * qij / r
            if r > 0:
                row.update(e_vdw=e_vdw, e_es=e_es,
                           energy=None if e_vdw is None else e_vdw + e_es)  # fmt: skip
            rows.append(row)
    return rows
