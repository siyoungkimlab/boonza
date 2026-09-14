"""Maestro MAE/CMS files, read the way msys does.

Each ``f_m_ct`` block becomes a ct.  Atoms come from ``m_atom`` plus pseudo
particles from ``ffio_ff.ffio_pseudo``; bonds from ``m_bond``.  The
``ffio_ff`` block describes the force field of one copy of each molecule
type ("sites"); its terms are repeated over every copy in the ct.  Each
``ffio_*`` block is translated to DMS-style term tables by a handler that
mirrors the corresponding msys importer.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

from .._columns import STR, scalar
from ..system import System
from ..terms import ParamTable
from ._maeparse import MaeArray, MaeError, parse_mae, read_text
from .dms import _factorize

_INF = float("inf")


def load_mae(
    path, structure_only: bool = False, without_tables: bool = False,
    ignore_unrecognized: bool = False,
) -> System:  # fmt: skip
    """Read an MAE/CMS file (optionally gzip/bzip2 compressed); all cts go into one System."""
    path = os.fspath(path)
    blocks = parse_mae(read_text(path))
    stages = {b.get("fepio_stage") for b in blocks}
    if 1 in stages and 2 in stages:
        raise NotImplementedError("alchemical (FEP) MAE files are not supported yet")
    s = System(path)
    for ct in blocks:
        if ct.get("ffio_ct_type") == "full_system":
            continue
        _append_ct(s, ct, ignore_unrecognized, without_tables or structure_only)
    if structure_only and s.natoms and (s._atoms.column("anum") <= 0).any():
        s = s.clone(structure_only=True)
    s.name = path
    return s


# ---------------------------------------------------------------------------
# column helpers


def _str(v, default: str = "") -> str:
    return v if isinstance(v, str) else default


def _column(arr, key: str, n: int, default, kind: str) -> np.ndarray:
    dtype = {"i": np.int64, "r": np.float64, "s": STR}[kind]
    if not isinstance(arr, MaeArray) or key not in arr:
        return np.full(n, default, dtype=dtype)
    v = arr[key]
    if v.dtype != dtype:
        if kind == "s":
            v = v.astype(STR)
        else:
            v = np.array([scalar(x) for x in v.tolist()], dtype=np.float64).astype(dtype)
    null = arr.nulls.get(key)
    if null is not None:
        v = v.copy()
        v[null] = default
    return v


def _ints(blk: MaeArray, keys) -> np.ndarray:
    return np.column_stack([_column(blk, k, blk.size, 0, "i") for k in keys]).reshape(
        blk.size, len(keys)
    )


def _floats(blk: MaeArray, keys) -> np.ndarray:
    return np.column_stack([_column(blk, k, blk.size, 0.0, "r") for k in keys]).reshape(
        blk.size, len(keys)
    )


def _funct(blk: MaeArray) -> np.ndarray:
    return np.strings.lower(_column(blk, "ffio_funct", blk.size, "", "s"))


def _site_columns(blk: MaeArray) -> np.ndarray:
    """``ffio_index`` (if present) then ``ffio_ai``, ``ffio_aj``, ... as an int matrix."""
    keys = ["ffio_index"] if "ffio_index" in blk else []
    k = 0
    while f"ffio_a{chr(ord('i') + k)}" in blk:
        keys.append(f"ffio_a{chr(ord('i') + k)}")
        k += 1
    return _ints(blk, keys)


def _default_columns(table) -> list[str]:
    return [f"ffio_c{k + 1}" for k in range(len(table.params.props))]


class _ParamMap:
    """Adds rows of parameter values to a ParamTable, reusing identical rows."""

    def __init__(self, params: ParamTable, props: list[str] | None = None):
        self.params = params
        self.props = props or params.props
        self.seen: dict[tuple, int] = {}

    def add(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1, len(self.props))
        if len(values) == 0:
            return np.empty(0, np.int64)
        uniq, first, inv = np.unique(values, axis=0, return_index=True, return_inverse=True)
        ids = np.empty(len(uniq), np.int64)
        for u in np.argsort(first, kind="stable").tolist():  # first-appearance order
            key = tuple(uniq[u].tolist())
            pid = self.seen.get(key)
            if pid is None:
                pid = self.seen[key] = self.params.add_param(
                    **{p: v for p, v in zip(self.props, key, strict=True)}
                )
            ids[u] = pid
        return ids[inv.reshape(-1)]


# ---------------------------------------------------------------------------
# structure


def _append_ct(s: System, ct: dict, ignore_unrecognized: bool, without_tables: bool) -> None:
    cell = np.zeros((3, 3))
    for i, a in enumerate("abc"):
        for j, x in enumerate("xyz"):
            v = ct.get(f"chorus_box_{a}{x}")
            if isinstance(v, int | float) and not isinstance(v, bool):
                cell[i, j] = v
    s._cell = cell  # every ct overwrites the cell, as in msys

    particles = _read_particles(s, ct)
    if not s.provenance and isinstance(ct.get("msys_provenance"), MaeArray):
        prov = ct["msys_provenance"]
        fields = ("version", "timestamp", "user", "workdir", "cmdline", "executable")
        cols = {f: _column(prov, f, prov.size, "", "s").tolist() for f in fields}
        s.provenance = [{f: cols[f][k] for f in fields} for k in range(prov.size)]
    ff = ct.get("ffio_ff")
    if without_tables or particles is None or not isinstance(ff, dict):
        return
    _write_ffinfo(s, ff)
    sites = ff.get("ffio_atoms") if isinstance(ff.get("ffio_atoms"), MaeArray) else None
    if sites is None:
        sites = ff.get("ffio_sites")
    ctx = _Ctx(s, ff, sites, *particles)
    for name, blk in ff.items():
        if not isinstance(blk, MaeArray) or blk.size == 0:
            continue
        if name in _SKIPPABLE or name.startswith("ffio_cmap"):
            continue
        handler = _HANDLERS.get(name)
        if handler is None:
            if name.endswith("_alchemical"):
                raise NotImplementedError(f"{name}: alchemical MAE blocks are not supported yet")
            if ignore_unrecognized:
                warnings.warn(f"skipping unrecognized block '{name}'", stacklevel=2)
                continue
            raise MaeError(f"No handler for block '{name}'")
        handler(ctx, ff if name == "ffio_torsion_torsion" else blk)


def _add_keyval(props: dict, key: str, value) -> None:
    if key in props and type(props[key]) is not type(value):
        warnings.warn(f"skipping ct property '{key}' with a different type", stacklevel=3)
        return
    props[key] = value


def _read_particles(s: System, ct: dict):
    m_atom = ct.get("m_atom")
    if not isinstance(m_atom, MaeArray):
        return None
    ct_id = s.add_ct(_str(ct.get("m_title"))).id
    props = s._ct_props[ct_id]
    for key, val in ct.items():
        if key.startswith("chorus_box_") or key in ("m_title", "__name__"):
            continue
        if isinstance(val, bool) or val is None:
            continue
        if isinstance(val, int | float | str):
            _add_keyval(props, key, val)
        elif key == "m_depend" and isinstance(val, MaeArray):
            if "m_depend_property" in val and "m_depend_dependency" in val:
                for p, d in zip(val["m_depend_property"].tolist(),
                                val["m_depend_dependency"].tolist(), strict=True):  # fmt: skip
                    _add_keyval(props, "m_depend/" + p, d)

    n = m_atom.size
    ff = ct.get("ffio_ff")
    pseudo = ff.get("ffio_pseudo") if isinstance(ff, dict) else None
    npseudo = pseudo.size if isinstance(pseudo, MaeArray) else 0

    def both(atom_key, pseudo_key, default, kind):
        return np.concatenate([
            _column(m_atom, atom_key, n, default, kind),
            _column(pseudo, pseudo_key, npseudo, default, kind),
        ])  # fmt: skip

    strip = np.strings.strip
    chain = strip(both("m_chain_name", "ffio_chain_name", "", "s"))
    segid = strip(both("m_pdb_segment_name", "ffio_pdb_segment_name", "", "s"))
    resname = strip(both("m_pdb_residue_name", "ffio_pdb_residue_name", "UNK", "s"))
    resid = both("m_residue_number", "ffio_residue_number", 0, "i")
    names = strip(both("m_pdb_atom_name", "ffio_atom_name", "", "s"))
    insertion = strip(both("m_insertion_code", "", "", "s"))
    pos = np.column_stack([both(f"m_{c}_coord", f"ffio_{c}_coord", 0.0, "r") for c in "xyz"])
    vel = np.column_stack([both(f"ffio_{c}_vel", f"ffio_{c}_vel", 0.0, "r") for c in "xyz"])
    total = n + npseudo

    chain_code, chain_first = _factorize(chain, segid)
    res_code, res_first = _factorize(chain_code, resid, resname, insertion)
    chain_off, res_off, atom_off = s.nchains, s.nresidues, s.natoms
    s._chains.append(
        len(chain_first), {"ct": ct_id, "name": chain[chain_first], "segid": segid[chain_first]}
    )
    s._residues.append(
        len(res_first),
        {
            "chain": chain_code[res_first] + chain_off,
            "resid": resid[res_first],
            "name": resname[res_first],
            "insertion": insertion[res_first],
        },
    )
    s._atoms.append(
        total,
        {
            "residue": res_code + res_off,
            "name": names,
            "anum": both("m_atomic_number", "", 0, "i"),
            "pos": pos.reshape(total, 3),
            "vel": vel.reshape(total, 3),
            "formal_charge": both("m_formal_charge", "", 0, "i"),
        },
    )
    rows = slice(atom_off, atom_off + total)
    for mae_key, prop in (
        ("ffio_grp_thermostat", "grp_temperature"),
        ("ffio_grp_energy", "grp_energy"),
        ("ffio_grp_ligand", "grp_ligand"),
        ("ffio_grp_cm_moi", "grp_bias"),
        ("ffio_grp_frozen", "grp_frozen"),
        ("fep_mapping", "fep_mapping"),
    ):
        if mae_key in m_atom:
            s._atoms.add_column(prop, "int")
            pseudo_key = "fep_mapping" if prop == "fep_mapping" else ""
            s._atoms.set(prop, both(mae_key, pseudo_key, 0, "i"), rows)
    if "m_grow_name" in m_atom:
        grow = strip(both("m_grow_name", "", "", "s"))
        if (grow != "").any():
            s._atoms.add_column("m_grow_name", "str")
            s._atoms.set("m_grow_name", grow, rows)
    if "m_mmod_type" in m_atom:
        s._atoms.add_column("m_mmod_type", "int")
        s._atoms.set("m_mmod_type", both("m_mmod_type", "", 0, "i"), rows)

    m_bond = ct.get("m_bond")
    if isinstance(m_bond, MaeArray) and m_bond.size:
        fr, to = (
            _column(m_bond, "m_from", m_bond.size, 0, "i"),
            _column(m_bond, "m_to", m_bond.size, 0, "i"),
        )
        order = _column(m_bond, "m_order", m_bond.size, 1, "i")
        order = np.where(order == 0, 1, order)
        bad = (fr < 1) | (to < 1) | (fr > n) | (to > n)
        if bad.any():
            k = int(np.argmax(bad))
            raise MaeError(f"Bond {k + 1} between nonexistent atoms {fr[k]}, {to[k]}")
        if (fr == to).any():
            raise MaeError(f"Bond {int(np.argmax(fr == to)) + 1} to self")
        s.add_bonds(np.column_stack([fr, to]) - 1 + atom_off, order=order)
    s._cache.clear()
    return np.arange(atom_off, atom_off + total), n, npseudo


def _write_ffinfo(s: System, ff: dict) -> None:
    extra = s.aux_tables.get("forcefield")
    if extra is None:
        extra = ParamTable()
        extra.add_prop("id", int)
        extra.add_prop("path", str)
        extra.add_prop("info", str)
        s.aux_tables["forcefield"] = extra
    known = set(extra["path"].tolist()) if "path" in extra.props else set()

    def add(path, info=""):
        extra.add_param(id=len(extra), path=path, info=info)

    cmd = _str(ff.get("viparr_command"))
    workdir = ff.get("viparr_workdir")
    workdir = (workdir if isinstance(workdir, str) else ".") + "/"
    tokens = [t for t in cmd.split(" ") if t]
    if len(tokens) > 2:
        for i in range(1, len(tokens) - 1):
            if tokens[i] in ("-f", "-d", "-m"):
                path = tokens[i + 1]
                if tokens[i] != "-f" and not path.startswith("/"):
                    path = workdir + path
                if path not in known:
                    add(path)
                    known.add(path)
    info_blk = ff.get("msys_forcefield")
    if isinstance(info_blk, MaeArray):
        paths = _column(info_blk, "path", info_blk.size, "", "s").tolist()
        infos = _column(info_blk, "info", info_blk.size, "", "s").tolist()
        for path, info in zip(paths, infos, strict=True):
            if (not path and not info) or path in known:
                continue
            add(path, info.replace("\\n", "\n"))
    if not s.provenance:
        s.provenance = [{
            "version": "",
            "timestamp": _str(ff.get("date")),
            "user": _str(ff.get("user")),
            "workdir": _str(ff.get("viparr_workdir")),
            "cmdline": cmd,
            "executable": "",
        }]  # fmt: skip


class _VdwMap:
    def __init__(self, ff: dict, sites):
        self.rule = _str(ff.get("ffio_comb_rule"))
        self.funct = ""
        self.params: dict[str, list[float]] = {}
        self.types: list[str] = []
        self.combined: dict[tuple[str, str], list[float]] = {}
        vt = ff.get("ffio_vdwtypes")
        if not isinstance(vt, MaeArray) or vt.size == 0:
            return
        functs = set(_column(vt, "ffio_funct", vt.size, "", "s").tolist())
        if len(functs) != 1:
            raise MaeError(f"Expected 1 kind of ffio_funct; got {len(functs)}")
        names_key = "ffio_name1" if "ffio_name1" in vt else "ffio_name"
        names = _column(vt, names_key, vt.size, "", "s").tolist()
        cols = self._ccols(vt)
        values = _floats(vt, cols).tolist()
        self.params = dict(zip(names, values, strict=True))
        self.funct = functs.pop().lower()
        self.rule = self.rule.lower()
        if isinstance(sites, MaeArray) and "ffio_vdwtype" in sites:
            self.types = _column(sites, "ffio_vdwtype", sites.size, "", "s").tolist()
            if "ffio_vdwtypeB" in sites:
                null = sites.nulls.get("ffio_vdwtypeB")
                if null is None or not null.all():
                    raise NotImplementedError("alchemical vdw types are not supported yet")
        comb = ff.get("ffio_vdwtypes_combined")
        if isinstance(comb, MaeArray) and "ffio_name1" in comb:
            n1 = _column(comb, "ffio_name1", comb.size, "", "s").tolist()
            n2 = _column(comb, "ffio_name2", comb.size, "", "s").tolist()
            vals = _floats(comb, self._ccols(comb)).tolist()
            for a, b, v in zip(n1, n2, vals, strict=True):
                self.combined[(a, b)] = v
                self.combined[(b, a)] = v

    @staticmethod
    def _ccols(blk) -> list[str]:
        cols = []
        while f"ffio_c{len(cols) + 1}" in blk:
            cols.append(f"ffio_c{len(cols) + 1}")
        return cols

    def type(self, site: int) -> str:
        if not 1 <= site <= len(self.types):
            raise MaeError(f"illegal site id {site}")
        return self.types[site - 1]

    def param(self, name: str) -> list[float]:
        try:
            return self.params[name]
        except KeyError:
            raise MaeError(f"Missing vdw parameter with name '{name}'") from None


class _Ctx:
    """Site-to-atom mapping for one ct, plus shared helpers for the handlers."""

    def __init__(self, s: System, ff: dict, sites, atoms, natoms: int, npseudos: int):
        self.s = s
        self.ff = ff
        self.atoms = atoms
        self.nsites = sites.size if isinstance(sites, MaeArray) else 0
        self.nblocks = len(atoms) // self.nsites if self.nsites else 0
        self.s2p = np.zeros(len(atoms), np.int64)
        if self.nsites:
            kind = np.strings.lower(_column(sites, "ffio_type", self.nsites, "atom", "s"))
            is_atom = kind == "atom"
            na, npp = int(is_atom.sum()), int((~is_atom).sum())
            rank_a, rank_p = np.cumsum(is_atom) - 1, np.cumsum(~is_atom) - 1
            j = np.arange(self.nblocks)[:, None]
            block = np.where(is_atom[None, :], j * na + rank_a, natoms + j * npp + rank_p)
            m = self.nblocks * self.nsites
            if m and block.max() >= len(atoms):
                raise MaeError("ffio sites do not match the atoms of the ct")
            self.s2p[:m] = block.ravel()
            ids = atoms[self.s2p[:m]]
            for prop, key in (("mass", "ffio_mass"), ("charge", "ffio_charge")):
                vals = np.tile(_column(sites, key, self.nsites, 0.0, "r"), self.nblocks)
                s._atoms.set(prop, vals, ids)
        self.vdw = _VdwMap(ff, sites)

    def table(self, name: str):
        try:
            return self.s.add_table_from_schema(name)
        except KeyError:
            raise MaeError(f"unknown term table '{name}'") from None

    def site(self, site: int) -> int:
        return int(self.atoms[self.s2p[site - 1]])

    def unroll(self, sites) -> np.ndarray:
        """Atom ids for 1-based site ids (m, k), repeated over every molecule copy."""
        sites = np.asarray(sites, dtype=np.int64)
        m, k = sites.shape
        if sites.size and (sites.min() < 1 or sites.max() > self.nsites):
            bad = sites[(sites < 1) | (sites > self.nsites)][0]
            raise MaeError(f"id {bad} is out of bounds (nsites={self.nsites})")
        offsets = (np.arange(self.nblocks) * self.nsites)[None, :, None]
        return self.atoms[self.s2p[sites[:, None, :] - 1 + offsets]].reshape(m * self.nblocks, k)

    def add(self, table, sites, pids, constrained=None) -> None:
        atoms = self.unroll(sites)
        pids = np.repeat(np.asarray(pids, dtype=np.int64), self.nblocks)
        new = table.add_terms(atoms, pids)
        if constrained is not None and np.any(constrained):
            flags = np.repeat(np.asarray(constrained, dtype=np.int64), self.nblocks)
            table._t.set("constrained", flags, new)


# ---------------------------------------------------------------------------
# ffio handlers


def _harm_constrained(f: np.ndarray) -> np.ndarray:
    return np.strings.startswith(f, "harm_") & np.strings.endswith(f, "_constrained")


def _check(ok: np.ndarray, f: np.ndarray, block: str) -> None:
    if not ok.all():
        raise MaeError(f"Unsupported ffio_funct '{f[~ok][0]}' in {block}")


def _bonds(ctx: _Ctx, blk: MaeArray) -> None:
    t = ctx.table("stretch_harm")
    f = _funct(blk)
    constrained = _harm_constrained(f)
    _check((f == "harm") | constrained, f, "ffio_bonds")
    pids = _ParamMap(t.params).add(_floats(blk, _default_columns(t)))
    ctx.add(t, _ints(blk, ["ffio_ai", "ffio_aj"]), pids, constrained)


def _morse(ctx: _Ctx, blk: MaeArray) -> None:
    t = ctx.table("stretch_morse")
    ctx.add(
        t,
        _ints(blk, ["ffio_ai", "ffio_aj"]),
        _ParamMap(t.params).add(_floats(blk, _default_columns(t))),
    )


def _angles(ctx: _Ctx, blk: MaeArray) -> None:
    bonds, angles = ctx.table("stretch_harm"), ctx.table("angle_harm")
    f = _funct(blk)
    ub = f == "ub"
    constrained = _harm_constrained(f)
    harm = (f == "harm") | constrained
    _check(ub | harm, f, "ffio_angles")
    vals = _floats(blk, ["ffio_c1", "ffio_c2"])
    ijk = _ints(blk, ["ffio_ai", "ffio_aj", "ffio_ak"])
    if ub.any():
        ctx.add(bonds, ijk[ub][:, [0, 2]], _ParamMap(bonds.params).add(vals[ub]))
    if harm.any():
        pids = _ParamMap(angles.params).add(vals[harm])
        ctx.add(angles, ijk[harm], pids, constrained[harm])


def _dihedrals(ctx: _Ctx, blk: MaeArray) -> None:
    f = _funct(blk)
    trig = np.isin(f, ["proper_trig", "improper_trig"])
    opls = np.isin(f, ["opls_proper", "opls_improper"])
    harm = np.isin(f, ["proper_harm", "improper_harm"])
    anharm = f == "improper_anharm"
    _check(trig | opls | harm | anharm, f, "ffio_dihedrals")
    c = _floats(blk, [f"ffio_c{k}" for k in range(8)])
    atoms = _ints(blk, ["ffio_ai", "ffio_aj", "ffio_ak", "ffio_al"])
    rows = trig | opls
    if rows.any():
        vals = c.copy()
        phi, c1, c2, c3, c4 = (c[opls, k] for k in range(5))
        vals[opls] = np.column_stack([
            phi, 0.5 * (c1 + c2 + c3 + c4), 0.5 * c1, -0.5 * c2, 0.5 * c3, -0.5 * c4,
            np.zeros_like(phi), np.zeros_like(phi),
        ])  # fmt: skip
        t = ctx.table("dihedral_trig")
        ctx.add(t, atoms[rows], _ParamMap(t.params).add(vals[rows]))
    for mask, name in ((harm, "improper_harm"), (anharm, "improper_anharm")):
        if mask.any():
            t = ctx.table(name)
            ctx.add(t, atoms[mask], _ParamMap(t.params).add(c[mask][:, :2]))


def _dihedral6(ctx: _Ctx, blk: MaeArray) -> None:
    f = _funct(blk)
    _check(f == "trig", f, "ffio_dihedrals6atom")
    t = ctx.table("dihedral6_trig")
    ctx.add(t, _site_columns(blk), _ParamMap(t.params).add(_floats(blk, _default_columns(t))))


def _by_funct(ctx: _Ctx, blk: MaeArray, prefix: str, cut: int | None = None, pseudo=False):
    f = _funct(blk)
    sites = _site_columns(blk)
    for funct in dict.fromkeys(f.tolist()):
        rows = f == funct
        t = ctx.table(prefix + (funct[:cut] if cut else funct))
        if sites.shape[1] < t.natoms:
            raise MaeError(f"{blk.name}: not enough site columns for {t.name}")
        ids = sites[rows][:, : t.natoms]
        ctx.add(t, ids, _ParamMap(t.params).add(_floats(blk, _default_columns(t))[rows]))
        if pseudo:  # virtual sites: bond each pseudo (first id) to its parent (second)
            parent, child = ctx.unroll(ids[:, [1]])[:, 0], ctx.unroll(ids[:, [0]])[:, 0]
            ctx.s.add_bonds(np.column_stack([parent, child]))
            res = ctx.s._atoms._data["residue"]
            res[child] = res[parent]
            ctx.s._cache.clear()


def _constraints(ctx: _Ctx, blk: MaeArray) -> None:
    _by_funct(ctx, blk, "constraint_", cut=3)


def _virtuals(ctx: _Ctx, blk: MaeArray) -> None:
    _by_funct(ctx, blk, "virtual_", pseudo=True)


def _exclusions(ctx: _Ctx, blk: MaeArray) -> None:
    t = ctx.table("exclusion")
    ctx.add(t, _ints(blk, ["ffio_ai", "ffio_aj"]), np.full(blk.size, -1))


def _sig_eps(sij, eij, sf, p) -> None:
    sf *= eij * 4
    s3 = sij * sij * sij
    s6 = s3 * s3
    s12 = s6 * s6
    p[0] = sf * s12
    p[1] = sf * s6


def _geometric(vi, vj, sf, p) -> None:
    _sig_eps(np.sqrt(vi[0] * vj[0]), np.sqrt(vi[1] * vj[1]), sf, p)


def _arith_geom(vi, vj, sf, p) -> None:
    _sig_eps(0.5 * (vi[0] + vj[0]), np.sqrt(vi[1] * vj[1]), sf, p)


def _lb_geom(vi, vj, sf, p) -> None:
    def abc(v):
        if v[0] == 0:
            return v[0], v[1], 0.0
        return (
            (6.0 * v[1] * np.exp(v[0])) / (v[0] - 6.0),
            v[2] / v[0],
            (v[0] ** 7 * v[1]) / (v[0] - 6.0),
        )

    ai, bi, ci = abc(vi)
    aj, bj, cj = abc(vj)
    p[0] = sf * np.sqrt(ai * aj)
    p[1] = 0.5 * (bi + bj)
    p[2] = sf * np.sqrt(ci * cj)


def _pairs(ctx: _Ctx, blk: MaeArray) -> None:
    vm = ctx.vdw
    combine = None
    if vm.funct in ("lj12_6_sig_epsilon", "polynomial_cij"):
        name = "pair_12_6_es"
        combine = {"geometric": _geometric, "arithmetic/geometric": _arith_geom}.get(vm.rule)
    elif vm.funct == "exp_6x":
        name = "pair_exp_6_es"
        combine = _lb_geom if vm.rule == "lb/geometric" else None
    else:
        raise MaeError(f"Unsupported ffio_funct '{vm.funct}' for ffio_pairs")
    if combine is None:
        raise MaeError(f"Unsupported ffio_comb_rule '{vm.rule}' for ffio_funct '{vm.funct}'")
    t = ctx.table(name)
    pmap = _ParamMap(t.params)
    nprops = len(t.params.props)
    ai = _column(blk, "ffio_ai", blk.size, 0, "i").tolist()
    aj = _column(blk, "ffio_aj", blk.size, 0, "i").tolist()
    c1 = _column(blk, "ffio_c1", blk.size, 0.0, "r").tolist()
    c2 = _column(blk, "ffio_c2", blk.size, 0.0, "r").tolist()
    funct = _funct(blk).tolist()
    charge = ctx.s._atoms.column("charge")
    skipped = set(range(blk.size))
    # a pair may be listed several times; each pass takes at most one ES and
    # one LJ row per pair and leaves the rest for the next pass (as msys does)
    while skipped:
        pairs: dict[tuple[int, int], list[float]] = {}
        for i in sorted(skipped):
            p = pairs.setdefault((ai[i], aj[i]), [_INF] * nprops)
            f = funct[i]
            if f in ("coulomb", "coulomb_scale"):
                if p[-1] != _INF:
                    continue
                p[-1] = c1[i] * charge[ctx.site(ai[i])] * charge[ctx.site(aj[i])]
            elif f == "coulomb_qij":
                if p[-1] != _INF:
                    continue
                p[-1] = c1[i]
            elif f in ("lj", "lj_scale"):
                if p[0] != _INF:
                    continue
                ti, tj = vm.type(ai[i]), vm.type(aj[i])
                if (ti, tj) in vm.combined:
                    v = vm.combined[(ti, tj)]
                    combine(v, v, c1[i], p)
                else:
                    combine(vm.param(ti), vm.param(tj), c1[i], p)
            elif f == "lj12_6_sig_epsilon":
                if p[0] != _INF:
                    continue
                _sig_eps(c1[i], c2[i], 1.0, p)
            else:
                raise MaeError(f"Unsupported ffio_funct '{f}' in ffio_pairs")
            skipped.discard(i)
        keys = sorted(pairs)
        vals = np.array([[0.0 if v == _INF else float(v) for v in pairs[k]] for k in keys])
        ctx.add(t, np.array(keys, dtype=np.int64).reshape(-1, 2), pmap.add(vals))


def _vdwtypes(ctx: _Ctx, blk: MaeArray) -> None:
    vm = ctx.vdw
    funct = {"lj12_6_sig_epsilon": "vdw_12_6", "exp_6x": "vdw_exp_6",
             "polynomial_cij": "polynomial_cij"}.get(vm.funct)  # fmt: skip
    if funct is None:
        raise MaeError(f"Unrecognized mae vdw_funct '{vm.funct}'")
    t = ctx.s.add_nonbonded_from_schema(funct, vm.rule)
    props = [p for p in t.params.props if p != "type"]
    t.params.add_prop("type", str)
    types: dict[str, int] = {}
    pids = []
    for site in range(1, ctx.nsites + 1):
        name = vm.type(site)
        if name not in types:
            vals = vm.param(name)
            types[name] = t.params.add_param(
                type=name, **{p: v for p, v in zip(props, vals, strict=False)}
            )
        pids.append(types[name])
    sites = np.arange(1, ctx.nsites + 1).reshape(-1, 1)
    ctx.add(t, sites, pids)
    if vm.combined:
        for p in t.params.props:
            if p not in t.overrides.params.props:
                t.overrides.params.add_prop(p, t.params.prop_type(p))
        oprops = t.overrides.params.props
        for ti in sorted(types):
            for tj in sorted(types):
                if (ti, tj) in vm.combined:
                    vals = vm.combined[(ti, tj)]
                    t.overrides.set(types[ti], types[tj], **dict(zip(oprops, vals, strict=False)))


def _cmap(ctx: _Ctx, ff: dict) -> None:
    blk = ff["ffio_torsion_torsion"]
    if "ffio_c1" not in blk:
        raise MaeError("ffio_torsion_torsion missing ffio_c1")
    t = ctx.table("torsiontorsion_cmap")
    ids: dict[int, int] = {}
    pids = []
    for cid in _column(blk, "ffio_c1", blk.size, 0, "i").tolist():
        if cid not in ids:
            ids[cid] = t.params.add_param(cmapid=f"cmap{cid}")
            grid = ff.get(f"ffio_cmap{cid}")
            d = ParamTable()
            for p in ("phi", "psi", "energy"):
                d.add_prop(p, float)
            if isinstance(grid, MaeArray):
                v = _floats(grid, ["ffio_ai", "ffio_aj", "ffio_c1"])
                d.add_params(len(v), phi=v[:, 0], psi=v[:, 1], energy=v[:, 2])
            ctx.s.aux_tables[f"cmap{cid}"] = d
        pids.append(ids[cid])
    ctx.add(t, _site_columns(blk), pids)


def _raw_ids(blk: MaeArray, keys) -> np.ndarray:
    # the fbhw blocks use 1-based system atom ids rather than site ids
    return _ints(blk, keys) - 1


def _angle_fbhw(ctx: _Ctx, blk: MaeArray) -> None:
    t = ctx.table("angle_fbhw")
    pids = _ParamMap(t.params).add(_floats(blk, ["ffio_sigma", "ffio_theta0", "ffio_fc"]))
    t.add_terms(_raw_ids(blk, ["ffio_ai", "ffio_aj", "ffio_ak"]), pids)


def _improper_fbhw(ctx: _Ctx, blk: MaeArray) -> None:
    t = ctx.table("improper_fbhw")
    pids = _ParamMap(t.params).add(_floats(blk, ["ffio_sigma", "ffio_phi0", "ffio_fc"]))
    t.add_terms(_raw_ids(blk, ["ffio_ai", "ffio_aj", "ffio_ak", "ffio_al"]), pids)


def _posre_fbhw(ctx: _Ctx, blk: MaeArray) -> None:
    t = ctx.table("posre_fbhw")
    pids = _ParamMap(t.params).add(_floats(blk, ["ffio_fc", "ffio_sigma"]))
    atoms = ctx.atoms[_raw_ids(blk, ["ffio_ai"])]
    xyz = _floats(blk, ["ffio_x0", "ffio_y0", "ffio_z0"])
    t.add_terms(atoms, pids, x0=xyz[:, 0], y0=xyz[:, 1], z0=xyz[:, 2])


def _stretch_fbhw(ctx: _Ctx, blk: MaeArray) -> None:
    s = ctx.s
    t = s.add_table("stretch_fbhw", 1, "bond")
    t.add_term_prop("group", int)
    cols = ["lower", "upper", "sigma", "beta", "fc"]
    for c in cols:
        t.params.add_prop(c, float)
    pids = _ParamMap(t.params, cols).add(_floats(blk, [f"ffio_{c}" for c in cols]))
    inter = ParamTable()
    for c in ("group1", "group2", "param"):
        inter.add_prop(c, int)
    s.aux_tables["stretch_fbhw_interaction"] = inter
    group = min([0, *t.values("group").tolist()])
    g1 = _column(blk, "ffio_group1", blk.size, "", "s").tolist()
    g2 = _column(blk, "ffio_group2", blk.size, "", "s").tolist()
    for k, p in enumerate(pids.tolist()):
        inter.add_param(group1=group, group2=group + 1, param=p)
        for members, g in ((g1[k], group), (g2[k], group + 1)):
            ids = [int(x) - 1 for x in members.split()]
            if ids:
                t.add_terms(np.array(ids).reshape(-1, 1), p, group=g)
        group += 2


def _inplanewags(ctx: _Ctx, blk: MaeArray) -> None:
    f = _funct(blk)
    _check(f == "harm", f, "ffio_inplanewags")
    t = ctx.table("inplanewag_harm")
    atoms = _ints(blk, ["ffio_ai", "ffio_aj", "ffio_ak", "ffio_al"])
    ctx.add(t, atoms, _ParamMap(t.params).add(_floats(blk, _default_columns(t))))


def _pseudopol(ctx: _Ctx, blk: MaeArray) -> None:
    f = _funct(blk)
    _check(f == "fermi", f, "ffio_pseudo_polarization")
    t = ctx.table("pseudopol_fermi")
    ctx.add(t, _site_columns(blk), _ParamMap(t.params).add(_floats(blk, _default_columns(t))))


def _restraints(ctx: _Ctx, blk: MaeArray) -> None:
    f = _funct(blk)
    _check(f == "harm", f, "ffio_restraints")
    t = ctx.table("posre_harm")
    pids = _ParamMap(t.params).add(_floats(blk, _default_columns(t)))
    atoms = ctx.atoms[_raw_ids(blk, ["ffio_ai"])]
    xyz = _floats(blk, ["ffio_t1", "ffio_t2", "ffio_t3"])
    t.add_terms(atoms, pids, x0=xyz[:, 0], y0=xyz[:, 1], z0=xyz[:, 2])


def _ignore(ctx: _Ctx, blk) -> None:
    pass


_SKIPPABLE = {"ffio_sites", "ffio_atoms", "viparr_info", "msys_forcefield", "ffio_pseudo"}
_HANDLERS = {
    "ffio_bonds": _bonds,
    "ffio_morsebonds": _morse,
    "ffio_angles": _angles,
    "ffio_dihedrals": _dihedrals,
    "ffio_dihedrals6atom": _dihedral6,
    "ffio_constraints": _constraints,
    "ffio_exclusions": _exclusions,
    "ffio_pairs": _pairs,
    "ffio_vdwtypes": _vdwtypes,
    "ffio_vdwtypes_combined": _ignore,
    "ffio_virtuals": _virtuals,
    "ffio_torsion_torsion": _cmap,
    "ffio_posre_fbhw": _posre_fbhw,
    "ffio_angle_fbhw": _angle_fbhw,
    "ffio_improper_fbhw": _improper_fbhw,
    "ffio_stretch_fbhw": _stretch_fbhw,
    "ffio_inplanewags": _inplanewags,
    "ffio_pseudo_polarization": _pseudopol,
    "ffio_restraints": _restraints,
}
