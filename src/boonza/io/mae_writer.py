"""Write Maestro MAE files the way msys does.

Each ct becomes an ``f_m_ct`` block holding ``m_atom`` (real atoms only),
``m_bond`` and, unless ``structure_only``, an ``ffio_ff`` block.  The whole ct
is written as a single set of sites, so no molecule compression is done.
Tables that have no MAE equivalent are skipped with a warning, as in msys.
"""

from __future__ import annotations

import bz2
import getpass
import gzip
import os
import sys
import time
import warnings

import numpy as np

from .._columns import STR, scalar

# atomic number -> (maestro color, mmod type)
_MMOD = {1: (21, 48), 3: (4, 11), 6: (2, 14), 7: (43, 40), 8: (70, 23), 9: (8, 56),
         11: (4, 66), 12: (4, 72), 14: (14, 60), 15: (15, 53), 16: (13, 52),
         17: (13, 102), 19: (4, 67), 20: (4, 70)}  # fmt: skip
_GROUPS = [("grp_temperature", "ffio_grp_thermostat"), ("grp_energy", "ffio_grp_energy"),
           ("grp_frozen", "ffio_grp_frozen"), ("grp_bias", "ffio_grp_cm_moi"),
           ("grp_ligand", "ffio_grp_ligand")]  # fmt: skip


def _quote(s: str) -> str:
    if s == "":
        return '""'
    if not any(c.isspace() or not c.isprintable() or c in '"<\\' for c in s):
        return s
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _format(kind: str, values) -> np.ndarray:
    """Format a column of values (None entries become ``<>``) as strings."""
    if isinstance(values, np.ndarray) and values.dtype != object:
        arr, null = values, None
    else:
        values = list(values)
        null = np.array([v is None for v in values], bool)
        fill = {"r": 0.0, "i": 0, "s": ""}[kind]
        arr = np.array([fill if v is None else v for v in values],
                       dtype=STR if kind == "s" else None)  # fmt: skip
    if kind == "s":
        arr = np.asarray(arr, dtype=STR)
        if len(arr):
            uniq, inv = np.unique(arr, return_inverse=True)
            out = np.array([_quote(u) for u in uniq.tolist()], dtype=STR)[inv.reshape(-1)]
        else:
            out = np.array([], dtype=STR)
    elif kind == "r":
        out = np.asarray(arr, dtype=np.float64).astype(STR)
    else:
        out = np.asarray(arr).astype(np.int64).astype(STR)
    if null is not None and null.any():
        out[null] = "<>"
    return out


class _Array:
    """An MAE array block built column-wise; missing cells are written as ``<>``."""

    def __init__(self, name: str):
        self.name = name
        self.n = 0
        self.cols: dict[str, tuple[str, list]] = {}

    def add_rows(self, n: int, cols: dict) -> None:
        """Append ``n`` rows; ``cols`` maps 'k_name' to values (array or scalar)."""
        for key in cols:
            if key[2:] not in self.cols:
                self.cols[key[2:]] = (key[0], [(self.n, None)] if self.n else [])
        for name, (kind, chunks) in self.cols.items():
            key = f"{kind}_{name}"
            if key in cols:
                v = cols[key]
                if np.ndim(v) == 0:
                    v = np.full(n, v, dtype=STR if kind == "s" else None)
                chunks.append((n, v))
            elif n:
                chunks.append((n, None))
        self.n += n

    def text(self, indent: str) -> str:
        lines = [f"{indent}{self.name}[{self.n}] {{"]
        lines += [f"{indent}  {kind}_{name}" for name, (kind, _) in self.cols.items()]
        lines.append(f"{indent}  :::")
        if self.n:
            row = np.arange(1, self.n + 1).astype(STR)
            row = np.strings.add(indent + "  ", row)
            for kind, chunks in self.cols.values():
                parts = [
                    np.full(n, "<>", dtype=STR) if v is None else _format(kind, v)
                    for n, v in chunks
                ]
                row = np.strings.add(np.strings.add(row, " "), np.concatenate(parts))
            lines += row.tolist()
        lines.append(f"{indent}  :::")
        lines.append(f"{indent}}}")
        return "\n".join(lines)


class _Block:
    def __init__(self, name: str):
        self.name = name
        self.values: list[tuple[str, object]] = []
        self.blocks: list = []

    def set(self, key: str, value) -> None:
        self.values.append((key, value))

    def array(self, name: str) -> _Array:
        for b in self.blocks:
            if isinstance(b, _Array) and b.name == name:
                return b
        arr = _Array(name)
        self.blocks.append(arr)
        return arr

    def text(self, indent: str = "") -> str:
        lines = [f"{indent}{self.name} {{"]
        lines += [f"{indent}  {k}" for k, _ in self.values]
        lines.append(f"{indent}  :::")
        for k, v in self.values:
            lines.append(f"{indent}  " + _format(k[0], [v])[0])
        lines += [b.text(indent + "  ") for b in self.blocks]
        lines.append(f"{indent}}}")
        return "\n".join(lines)


def _pad(s: str, width: int) -> str:
    for i in range(len(s), width):
        s = s + " " if i % 2 else " " + s
    return s


def _pad_name(s: str) -> str:
    return {0: "", 1: f" {s}  ", 2: f" {s} ", 3: f" {s}"}.get(len(s), s.strip())


def _kind(value) -> str:
    if isinstance(value, bool | int | np.integer):
        return "i"
    if isinstance(value, float | np.floating):
        return "r"
    return "s"


def save_mae(system, path, structure_only: bool = False, allow_reorder_atoms: bool = False,
             append: bool = False) -> None:  # fmt: skip
    """Write ``system`` as MAE (``.gz``/``.bz2`` suffixes are compressed); one block per ct."""
    path = os.fspath(path)
    parts = ["{\n  s_m_m2io_version\n  :::\n  2.0.0\n}\n"]
    for ct in range(system.ncts):
        ids = system.ct_atoms(ct)
        if len(ids) == 0:
            continue
        if ids.max() - ids.min() + 1 != len(ids):
            msg = f"atom ids in ct {ct} are noncontiguous; writing to MAE will reorder the atoms"
            if not allow_reorder_atoms:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)
        parts.append(_ct_block(system.clone(ids), structure_only).text() + "\n")
    text = "\n".join(parts)
    mode = "at" if append else "wt"
    if path.endswith(".gz"):
        opener = gzip.open
    elif path.endswith(".bz2"):
        opener = bz2.open
    else:
        opener = open
    with opener(path, mode) as f:
        f.write(text)


def _ct_block(s, structure_only: bool) -> _Block:
    ct = _Block("f_m_ct")
    _provenance(ct, s)
    if s.cell.any():
        for i, a in enumerate("abc"):
            for j, x in enumerate("xyz"):
                ct.set(f"r_chorus_box_{a}{x}", float(s.cell[i, j]))
    ct.set("s_m_title", s._ct_names[0])
    for key, value in s._ct_props[0].items():
        key = "".join("_" if c.isspace() else c for c in key)
        ct.set(f"{_kind(value)}_{key}", scalar(value))

    A, R, C = s._atoms, s._residues, s._chains
    anum = A.column("anum")
    real = np.flatnonzero(anum != 0)
    res = A.column("residue")
    chn = R.column("chain")[res]
    mapping = np.full(s.natoms, -1, np.int64)
    mapping[real] = np.arange(1, len(real) + 1)

    color = np.array([_MMOD.get(a, (2, 64))[0] for a in anum[real].tolist()], np.int64)
    mmod = np.array([_MMOD.get(a, (2, 64))[1] for a in anum[real].tolist()], np.int64)
    if "m_mmod_type" in A:
        override = A.column("m_mmod_type")[real]
        mmod = np.where(override != 0, override, mmod)
    rres, rchn = res[real], chn[real]
    pos, vel = A.column("pos")[real], A.column("vel")[real]
    cols = {
        "i_m_mmod_type": mmod,
        "r_m_x_coord": pos[:, 0], "r_m_y_coord": pos[:, 1], "r_m_z_coord": pos[:, 2],
        "i_m_residue_number": R.column("resid")[rres],
        "s_m_insertion_code": _padded(R.column("insertion")[rres], 1),
        "s_m_mmod_res": " ",
        "s_m_chain_name": _padded(C.column("name")[rchn], 1),
        "i_m_color": color,
        "r_m_charge1": 0.0, "r_m_charge2": 0.0,
        "s_m_pdb_residue_name": _padded(R.column("name")[rres], 4),
        "s_m_pdb_atom_name": _names(A.column("name")[real]),
        "s_m_grow_name": " ",
        "i_m_atomic_number": anum[real],
        "i_m_visibility": 1,
        "r_ffio_x_vel": vel[:, 0], "r_ffio_y_vel": vel[:, 1], "r_ffio_z_vel": vel[:, 2],
        "i_m_formal_charge": A.column("formal_charge")[real],
    }  # fmt: skip
    for prop, key in _GROUPS:
        if prop in A:
            cols[f"i_{key}"] = A.column(prop)[real]
    if "segid" in A:
        cols["s_m_pdb_segment_name"] = A.column("segid")[real].astype(STR)
    ct.array("m_atom").add_rows(len(real), cols)

    bi, bj = mapping[s._bonds.column("i")], mapping[s._bonds.column("j")]
    ok = (bi > 0) & (bj > 0)
    ct.array("m_bond").add_rows(int(ok.sum()), {
        "i_m_from": bi[ok], "i_m_to": bj[ok], "i_m_order": s._bonds.column("order")[ok],
    })  # fmt: skip
    if structure_only:
        return ct

    ff = _Block("ffio_ff")
    ct.blocks.append(ff)
    ff.set("s_ffio_name", "msys")
    ff.set("i_ffio_version", 1)
    ff.array("ffio_sites").add_rows(s.natoms, {
        "s_ffio_type": np.where(anum == 0, "pseudo", "atom").astype(STR),
        "r_ffio_charge": A.column("charge"),
        "r_ffio_mass": A.column("mass"),
    })  # fmt: skip
    pseudo = np.flatnonzero(anum == 0)
    ppos, pvel, pres = A.column("pos")[pseudo], A.column("vel")[pseudo], res[pseudo]
    pchain = C.column("name")[chn[pseudo]]
    ff.array("ffio_pseudo").add_rows(len(pseudo), {
        "r_ffio_x_coord": ppos[:, 0], "r_ffio_y_coord": ppos[:, 1], "r_ffio_z_coord": ppos[:, 2],
        "s_ffio_atom_name": A.column("name")[pseudo],
        "s_ffio_pdb_residue_name": _padded(R.column("name")[pres], 4),
        "s_ffio_chain_name": pchain,
        "s_ffio_pdb_segment_name": pchain,
        "i_ffio_residue_number": R.column("resid")[pres],
        "r_ffio_x_vel": pvel[:, 0], "r_ffio_y_vel": pvel[:, 1], "r_ffio_z_vel": pvel[:, 2],
    })  # fmt: skip
    for name in sorted(s.tables):
        table = s.tables[name]
        if name == "nonbonded":
            _nonbonded(s, table, ff)
        elif not _tuple_table(table, ff):
            warnings.warn(f"Failed to process dms table '{name}'", stacklevel=3)
    for name, aux in s.aux_tables.items():
        if name.startswith("cmap"):
            ff.array("ffio_cmap" + name[4:]).add_rows(len(aux), {
                "r_ffio_ai": aux["phi"], "r_ffio_aj": aux["psi"], "r_ffio_c1": aux["energy"],
            })  # fmt: skip
    info = s.aux_tables.get("forcefield")
    if info is not None:
        cols = {}
        if "path" in info.props:
            cols["s_path"] = info["path"].astype(STR)
        if "info" in info.props:
            texts = [t[:-1] if t.endswith("\n") else t for t in info["info"].tolist()]
            cols["s_info"] = np.array([t.replace("\n", "\\n") for t in texts], dtype=STR)
        ff.array("msys_forcefield").add_rows(len(info), cols)
    return ct


def _padded(values: np.ndarray, width: int) -> np.ndarray:
    uniq, inv = np.unique(np.asarray(values, dtype=STR), return_inverse=True)
    return np.array([_pad(u, width) for u in uniq.tolist()], dtype=STR)[inv.reshape(-1)]


def _names(values: np.ndarray) -> np.ndarray:
    uniq, inv = np.unique(np.asarray(values, dtype=STR), return_inverse=True)
    return np.array([_pad_name(u) for u in uniq.tolist()], dtype=STR)[inv.reshape(-1)]


def _provenance(ct: _Block, s) -> None:
    from .. import __version__

    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    rows = [*s.provenance, {
        "version": f"boonza {__version__}", "timestamp": time.asctime(), "user": user,
        "workdir": os.getcwd(), "cmdline": " ".join(sys.argv), "executable": sys.executable,
    }]  # fmt: skip
    fields = ("version", "timestamp", "user", "workdir", "cmdline", "executable")
    ct.array("msys_provenance").add_rows(
        len(rows), {f"s_{f}": np.array([r.get(f, "") for r in rows], dtype=STR) for f in fields}
    )


def _nonbonded(s, table, ff: _Block) -> None:
    info = s.nonbonded_info
    funct = info.vdw_funct.lower()
    if funct == "vdw_12_6":
        funct, props = "LJ12_6_sig_epsilon", ["sigma", "epsilon"]
    elif funct == "vdw_exp_6":
        funct, props = "exp_6x", ["alpha", "epsilon", "rmin"]
    else:
        raise ValueError(f"Unsupported vdw_funct '{info.vdw_funct}' for MAE")
    ff.set("s_ffio_comb_rule", info.vdw_rule)
    params = table.params
    names = None
    if "type" in params.props:
        types = params["type"].tolist()
        if len(set(types)) == len(types):
            names = types
    if names is None:
        names = [str(i + 1) for i in range(len(params))]
    cols = {"s_ffio_name": np.array(names, dtype=STR), "s_ffio_funct": funct}
    for k, p in enumerate(props):
        cols[f"r_ffio_c{k + 1}"] = params[p]
    ff.array("ffio_vdwtypes").add_rows(len(params), cols)

    pids = table.param_ids
    if (pids < 0).any():
        raise ValueError(f"Missing nonbonded param for particle {int(np.argmax(pids < 0))}")
    vdw = np.full(s.natoms, "", dtype=object)
    vdw[table.atoms[:, 0]] = np.array(names, dtype=object)[pids]
    sites = ff.array("ffio_sites")
    sites.cols["ffio_vdwtype"] = ("s", [(s.natoms, np.array(vdw.tolist(), dtype=STR))])

    if len(table.overrides):
        if props[0] != "sigma":
            raise ValueError(f"override params supported only for vdw_12_6; got {funct}")
        items = table.overrides.items()
        ff.array("ffio_vdwtypes_combined").add_rows(len(items), {
            "s_ffio_name1": np.array([names[a] for (a, _), _ in items], dtype=STR),
            "s_ffio_name2": np.array([names[b] for (_, b), _ in items], dtype=STR),
            "r_ffio_c1": np.array([v["sigma"] for _, v in items]),
            "r_ffio_c2": np.array([v["epsilon"] for _, v in items]),
        })  # fmt: skip


# dms table -> (mae block, funct, number of ffio_a* sites, params written as ffio_c1...)
_TUPLES = {
    "stretch_harm": [("ffio_bonds", "harm", 2, ["r0", "fc"])],
    "stretch_morse": [("ffio_morsebonds", "Morse", 2, ["r0", "d", "a"])],
    "angle_harm": [("ffio_angles", "harm", 3, ["theta0", "fc"])],
    "dihedral_trig": [("ffio_dihedrals", "proper_trig", 4, [f"fc{k}" for k in range(7)])],
    "improper_harm": [("ffio_dihedrals", "improper_harm", 4, ["fc"])],
    "pair_12_6_es": [("ffio_pairs", "coulomb_qij", 2, ["qij"]),
                     ("ffio_pairs", "lj12_6_sig_epsilon", 2, [])],
    "torsiontorsion_cmap": [("ffio_torsion_torsion", "cmap", 8, [])],
    "constraint_ah1": [("ffio_constraints", "ah1", 2, ["r1"])],
    "constraint_ah2": [("ffio_constraints", "ah2", 3, ["r1", "r2"])],
    "constraint_ah3": [("ffio_constraints", "ah3", 4, ["r1", "r2", "r3"])],
    "constraint_ah4": [("ffio_constraints", "ah4", 5, ["r1", "r2", "r3", "r4"])],
    "constraint_hoh": [("ffio_constraints", "hoh", 3, ["theta", "r1", "r2"])],
    "virtual_lc2": [("ffio_virtuals", "lc2", 0, ["c1"])],
    "virtual_lc3": [("ffio_virtuals", "lc3", 0, ["c1", "c2"])],
    "virtual_out3": [("ffio_virtuals", "out3", 0, ["c1", "c2", "c3"])],
    "posre_harm": [("ffio_restraints", "harm", 1, ["fcx", "fcy", "fcz"])],
    "exclusion": [("ffio_exclusions", None, 2, [])],
    "improper_anharm": [("ffio_dihedrals", "improper_anharm", 4, ["fc2", "fc4"])],
    "inplanewag_harm": [("ffio_inplanewags", "harm", 4, ["w0", "fc"])],
    "pseudopol_fermi": [("ffio_pseudo_polarization", "fermi", 4, ["a", "b", "cutoff"])],
}  # fmt: skip


def _tuple_table(table, ff: _Block) -> bool:
    entries = _TUPLES.get(table.name)
    if entries is None:
        return False
    atoms = table.atoms + 1
    n = len(table)

    def param(prop):
        return table.values(prop)

    for block, funct, nsites, props in entries:
        cols = {f"i_ffio_a{chr(ord('i') + k)}": atoms[:, k] for k in range(nsites)}
        if funct is not None:
            cols["s_ffio_funct"] = funct
            if "constrained" in table.term_props:
                flag = table.values("constrained") != 0
                cols["s_ffio_funct"] = np.where(flag, funct + "_constrained", funct).astype(STR)
        for k, p in enumerate(props):
            cols[f"r_ffio_c{k + 1}"] = param(p)
        name = table.name
        if name in ("dihedral_trig", "improper_harm"):
            cols["r_ffio_c0"] = param("phi0")
        elif name == "pair_12_6_es" and funct.startswith("lj"):
            a, b = param("aij"), param("bij")
            ok = (a != 0) & (b != 0)
            with np.errstate(all="ignore"):
                cols["r_ffio_c1"] = np.where(ok, np.power(a / b, 1.0 / 6.0), 1.0)
                cols["r_ffio_c2"] = np.where(ok, (b * b) / (4 * a), 0.0)
        elif name == "torsiontorsion_cmap":
            ids = [str(v) for v in param("cmapid").tolist()]
            if not all(v.startswith("cmap") and v[4:].isdigit() for v in ids):
                raise ValueError("Invalid cmapid in torsiontorsion_cmap")
            cols["i_ffio_c1"] = np.array([int(v[4:]) for v in ids], np.int64)
        elif name.startswith("virtual_"):
            cols["i_ffio_index"] = atoms[:, 0]
            for k in range(1, table.natoms):
                cols[f"i_ffio_a{chr(ord('i') + k - 1)}"] = atoms[:, k]
        elif name == "posre_harm":
            for col, prop in (("r_ffio_t1", "x0"), ("r_ffio_t2", "y0"), ("r_ffio_t3", "z0")):
                cols[col] = param(prop)
        ff.array(block).add_rows(n, cols)
    return True
