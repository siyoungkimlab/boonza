"""GAFF2 templates for ligands, non-standard residues and covalent adducts.

AmberTools types each molecule or residue group that the force fields cannot
parameterize with GAFF2 and charges it with AM1-BCC (antechamber, sqm,
parmchk2 and tleap; no OpenFF).  :func:`gaff2_patch` turns the result into a
viparr patch force field, merged onto an Amber protein force field with
:func:`boonza.merge_forcefields` so that :func:`boonza.parameterize` treats
the new residues like any other template.

Residues bonded to a protein are cut at their peptide bonds and capped with
ACE/NME built from the neighbouring atoms.  Atoms of an amino acid keep the
protein force field's types and charges as far as each atom and its
neighbours match the parent residue, so backbone and CB terms, protein
impropers and CMAP come from the protein force field; every term that
touches a GAFF2 atom takes the value tleap gave it.  Each residue's charge
is brought to its formal charge by shifting its GAFF2 atoms evenly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np

from .system import System
from .viparr import (
    ParamRow,
    Rules,
    Template,
    ViparrError,
    ViparrForcefield,
    ViparrWarning,
    _Parameterizer,
    load_forcefield,
)

__all__ = ["AmberToolsError", "find_amber_tools", "find_unmatched", "gaff2_patch", "run_gaff2"]

#: GAFF2 releases and their AmberTools parameter files.
GAFF_FILES = {"2.11": "gaff211.dat", "2.2": "gaff2.dat"}
_TOOLS = ("antechamber", "sqm", "parmchk2", "tleap")
_PLUGINS = ["exclusions", "mass", "bonds", "angles", "vdw1", "propers", "impropers"]
_VALENCE = {5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
_CH = 1.09  # C-H length of the hydrogens that close cap methyl groups (A)
_ZERO = {"phi0": 0.0, **{f"fc{k}": 0.0 for k in range(7)}}


class AmberToolsError(RuntimeError):
    """AmberTools is missing or one of its programs failed."""


def find_amber_tools(amberhome=None) -> Path:
    """The AmberTools installation: ``amberhome``, else ``$AMBERHOME``, else
    the one whose ``antechamber`` is on ``PATH``."""
    if amberhome is not None:
        home = Path(amberhome)
    elif os.environ.get("AMBERHOME"):
        home = Path(os.environ["AMBERHOME"])
    else:
        exe = shutil.which("antechamber")
        home = Path(exe).parent.parent if exe else None
    if home is not None and all((home / "bin" / t).exists() for t in _TOOLS):
        return home
    raise AmberToolsError(
        "AmberTools (antechamber, sqm, parmchk2, tleap) was not found: install it with "
        "'conda install -c conda-forge ambertools' and activate that environment, or set AMBERHOME"
    )


def _run(cmd: list[str], cwd: Path, env: dict, output: str) -> str:
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    if proc.returncode != 0 or not (cwd / output).exists():
        tail = "\n".join(log.strip().splitlines()[-20:])
        raise AmberToolsError(f"{Path(cmd[0]).name} failed in {cwd}:\n{tail}")
    return log


def run_gaff2(fragment: System, net_charge: int, *, gaff: str = "2.11",
              charge_method: str = "bcc", amberhome=None, workdir=None) -> System:  # fmt: skip
    """GAFF2 parameters for one molecule with all its hydrogens, from AmberTools.

    antechamber types the atoms and computes charges (``charge_method``,
    AM1-BCC by default, for a molecule of ``net_charge``), parmchk2 writes
    every parameter the molecule uses from the ``gaff`` release, and tleap
    builds the topology.  Returns the tleap prmtop read as a system, atoms
    in the order of ``fragment``.  ``workdir`` keeps the AmberTools files.
    """
    from .io.amber import load_prmtop
    from .io.sdf import save_sdf

    if gaff not in GAFF_FILES:
        raise ValueError(f"gaff must be one of {sorted(GAFF_FILES)}")
    home = find_amber_tools(amberhome)
    parm = home / "dat" / "leap" / "parm" / GAFF_FILES[gaff]
    if not parm.exists():
        raise AmberToolsError(f"{parm} is missing")
    env = {**os.environ, "AMBERHOME": str(home)}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(workdir) if workdir is not None else Path(tmp)
        d.mkdir(parents=True, exist_ok=True)
        save_sdf(fragment, d / "in.sdf")
        bin_ = home / "bin"
        _run([str(bin_ / "antechamber"), "-i", "in.sdf", "-fi", "sdf", "-o", "out.mol2",
              "-fo", "mol2", "-at", "gaff2", "-c", charge_method, "-nc", str(int(net_charge)),
              "-rn", "MOL", "-pf", "y", "-dr", "no"], d, env, "out.mol2")  # fmt: skip
        shutil.copy(parm, d / "gaff.dat")
        _run([str(bin_ / "parmchk2"), "-i", "out.mol2", "-f", "mol2", "-p", "gaff.dat",
              "-s", "2", "-a", "Y", "-o", "out.frcmod"], d, env, "out.frcmod")  # fmt: skip
        (d / "leap.in").write_text(
            "source leaprc.gaff2\nloadamberparams gaff.dat\nm = loadmol2 out.mol2\n"
            "loadamberparams out.frcmod\nsaveamberparm m out.prmtop out.inpcrd\nquit\n"
        )
        log = _run([str(bin_ / "tleap"), "-f", "leap.in"], d, env, "out.prmtop")
        if "Errors = 0" not in log:
            raise AmberToolsError(f"tleap reported errors in {d}:\n{log[-2000:]}")
        p = load_prmtop(d / "out.prmtop")
    if p.natoms != fragment.natoms or (p.atoms["anum"] != fragment.atoms["anum"]).any():
        raise AmberToolsError("antechamber changed the atoms of the molecule")
    return p


def find_unmatched(system: System, forcefields, path=None) -> list[list[int]]:
    """Residue groups that ``forcefields`` cannot parameterize.

    A group is a set of bonded residues without a matching template, plus the
    residues bonded to them other than through a peptide bond or disulfide
    (the residue a covalent ligand is bound to).  Returns residue indices.
    """
    return _Builder(system, _load(forcefields, path)).groups()


def gaff2_patch(system: System, forcefields=(), *, charges=None, parents=None,
                gaff: str = "2.11", charge_method: str = "bcc", amberhome=None,
                workdir=None, path=None) -> ViparrForcefield:  # fmt: skip
    """A viparr patch with GAFF2 templates for what ``forcefields`` cannot parameterize.

    ``forcefields`` is the list to be given to :func:`boonza.parameterize`;
    the first, an Amber force field, is the one to merge the patch onto::

        patch = boonza.gaff2_patch(system, ["aa.amber.ff14SB", "water.tip3p"])
        ff = boonza.merge_forcefields("aa.amber.ff14SB", patch)
        out = boonza.parameterize(system, [ff, "water.tip3p"])

    With no force fields, the patch is a complete GAFF2 force field.

    Each group from :func:`find_unmatched`, capped with ACE/NME where it is
    bonded to a protein, is typed and charged by :func:`run_gaff2`.  Atoms of
    an amino acid keep the types and charges of the parent residue template
    (the best match by atom names; ``parents`` maps residue names to
    template names) as far as each atom and its bonded neighbours match it.
    Each residue is brought to its formal charge (``charges`` maps residue
    names to charges where the input has none) by shifting its GAFF2 atoms
    evenly. GAFF2 types are suffixed per group (``c3~1``) so groups never
    share parameters; templates pin the elements of their external atoms, so
    a Cys bound to a ligand takes its own template rather than CYX.
    ``workdir`` keeps the AmberTools files, one directory per template.
    """
    ffs = _load(forcefields, path)
    b = _Builder(system, ffs, charges or {}, parents or {},
                 dict(gaff=gaff, charge_method=charge_method, amberhome=amberhome),
                 None if workdir is None else Path(workdir))  # fmt: skip
    return b.patch()


def _load(forcefields, path) -> list[ViparrForcefield]:
    if isinstance(forcefields, str | os.PathLike | ViparrForcefield):
        forcefields = [forcefields]
    return [ff if isinstance(ff, ViparrForcefield) else load_forcefield(ff, path)
            for ff in forcefields]  # fmt: skip


def _check_host(ff: ViparrForcefield) -> None:
    r = ff.rules
    ok = (r.vdw_func == "lj12_6_sig_epsilon" and r.vdw_comb_rule == "arithmetic/geometric"
          and len(r.es_scale) == 3 and r.es_scale[:2] == [0.0, 0.0] == r.lj_scale[:2]
          and abs(r.es_scale[2] - 1 / 1.2) < 1e-3 and r.lj_scale[2] == 0.5)  # fmt: skip
    if not ok:
        raise ViparrError(
            f"{Path(ff.name).name} is not an Amber force field (12-6 LJ, Lorentz-Berthelot, "
            "1-4 scales 1/1.2 and 1/2); GAFF2 cannot join it"
        )


class _Builder:
    def __init__(self, system, ffs, charges=None, parents=None, run=None, workdir=None):
        self.P = P = _Parameterizer(system, ffs, False, False, True)
        self.ffs, self.host = ffs, (ffs[0] if ffs else None)
        if self.host is not None:
            _check_host(self.host)
        self.charges, self.parents = charges or {}, parents or {}
        self.run_opts, self.workdir = run or {}, workdir
        s = P.s
        self.anum, self.nbrs, self.names = P.anum, P.nbrs, P.names
        self.residue, self.resnames = P.residue, P.resnames
        self.resid = s.residues["resid"].tolist()
        self.pos = P.pos
        self.formal = s.atoms["formal_charge"].tolist()
        self.res_atoms: list[list[int]] = [[] for _ in range(s.nresidues)]
        for a, r in enumerate(self.residue):
            if self.anum[a] > 0:
                self.res_atoms[r].append(a)
        self.typed: dict[int, dict[int, tuple[str, str]]] = {}
        self.templates: list[Template] = []
        self.rows: dict[str, dict[str, list[dict]]] = {}
        self.memo: dict[str, str] = {}
        # per group: (the system atom of each fragment atom, the tleap prmtop)
        self.runs: list[tuple[list[int], System]] = []

    # -- which residues need templates ---------------------------------------------
    def _label(self, a: int) -> str:
        r = self.residue[a]
        return f"{self.resnames[r]}{self.resid[r]}:{self.names[a]}"

    def _peptide(self, i: int, j: int) -> bool:
        if {self.anum[i], self.anum[j]} != {6, 7}:
            return False
        n, c = (i, j) if self.anum[i] == 7 else (j, i)
        return self.names[n] == "N" and self.names[c] == "C"

    def _link(self, i: int, j: int) -> bool:
        """A peptide bond or a disulfide: the bonds protein templates expect."""
        return self._peptide(i, j) or (
            self.anum[i] == self.anum[j] == 16 and self.names[i] == self.names[j] == "SG"
        )

    def _matched(self, r: int) -> dict[int, tuple[str, str]] | None:
        """Types of residue ``r``'s atoms from the first force field matching it."""
        if r not in self.typed:
            self.typed[r] = None
            for k, ff in enumerate(self.ffs):
                tpl, pairs = self.P._find(ff, k, self.res_atoms[r])
                if tpl is not None:
                    self.typed[r] = {a: (tpl.btype[ti], tpl.nbtype[ti]) for ti, a in pairs}
                    break
        return self.typed[r]

    def groups(self) -> list[list[int]]:
        todo = {r for r, atoms in enumerate(self.res_atoms) if atoms and self._matched(r) is None}
        changed = True
        while changed:  # residues bound to a new one by anything but a protein link join it
            changed = False
            for r in list(todo):
                for a in self.res_atoms[r]:
                    for b in self.nbrs[a]:
                        rb = self.residue[b]
                        if self.anum[b] > 0 and rb not in todo and not self._link(a, b):
                            todo.add(rb)
                            changed = True
        parent = {r: r for r in todo}

        def root(r):
            while parent[r] != r:
                parent[r] = parent[parent[r]]
                r = parent[r]
            return r

        for r in todo:
            for a in self.res_atoms[r]:
                for b in self.nbrs[a]:
                    if self.residue[b] in todo:
                        parent[root(self.residue[b])] = root(r)
        out: dict[int, list[int]] = {}
        for r in sorted(todo):
            out.setdefault(root(r), []).append(r)
        return list(out.values())

    # -- parent residue templates ------------------------------------------------
    def _parent(self, r: int):
        """(template, {atom: template atom}, atoms that keep the template's types)."""
        atoms = self.res_atoms[r]
        if self.host is None or not {"N", "CA", "C"} <= {self.names[a] for a in atoms}:
            return None, {}, set()
        wanted = self.parents.get(self.resnames[r])
        if wanted is not None:
            candidates = [self.host.template(wanted)]
        else:
            candidates = [t for t in self.host.templates if {"N", "CA", "C"} <= set(t.names)]
        best = None
        for t in candidates:
            m = self._map(atoms, t)
            ok = self._fits(atoms, t, m)
            key = (len(ok), t.name == self.resnames[r], -t.natoms)
            if best is None or key > best[0]:
                best = (key, t, m, ok)
        if best is None or not best[3]:
            return None, {}, set()
        return best[1], best[2], best[3]

    def _map(self, atoms, t: Template) -> dict[int, int]:
        """Atoms to template atoms: heavy atoms by name, then grown through
        bonds where an element is unique, and hydrogens after their atom."""
        by_name: dict[str, int] = {}
        for i, (nm, z) in enumerate(zip(t.names, t.anum, strict=True)):
            if z > 0:
                by_name.setdefault(nm, i)
        m: dict[int, int] = {}
        used: set[int] = set()
        for a in atoms:
            i = by_name.get(self.names[a])
            if self.anum[a] > 1 and i is not None and t.anum[i] == self.anum[a] and i not in used:
                m[a] = i
                used.add(i)
        inside = set(atoms)
        grew = True
        while grew:
            grew = False
            for a, i in list(m.items()):
                mine = [b for b in self.nbrs[a] if b in inside and b not in m and self.anum[b] > 1]
                theirs = [j for j in t._nbrs[i] if t.anum[j] > 1 and j not in used]
                for z in {self.anum[b] for b in mine}:
                    bs = [b for b in mine if self.anum[b] == z]
                    js = [j for j in theirs if t.anum[j] == z]
                    if len(bs) == 1 and len(js) == 1:
                        m[bs[0]] = js[0]
                        used.add(js[0])
                        grew = True
        for a, i in list(m.items()):
            hs = sorted(b for b in self.nbrs[a] if b in inside and self.anum[b] == 1)
            ths = [j for j in t._nbrs[i] if t.anum[j] == 1 and j not in used]
            for b in list(hs):
                j = next((j for j in ths if t.names[j] == self.names[b]), None)
                if j is not None:
                    m[b] = j
                    used.add(j)
                    hs.remove(b)
                    ths.remove(j)
            for b, j in zip(hs, ths, strict=False):
                m[b] = j
                used.add(j)
        return m

    def _fits(self, atoms, t: Template, m: dict[int, int]) -> set[int]:
        """Mapped atoms whose bonded neighbours are exactly the template atom's."""
        inside = set(atoms)
        ok = set()
        for a, i in m.items():
            t_in = [j for j in t._nbrs[i] if t.anum[j] > 0]
            t_out = [j for j in t._nbrs[i] if t.anum[j] == -1]
            mine_in = [b for b in self.nbrs[a] if b in inside and self.anum[b] > 0]
            mine_out = [b for b in self.nbrs[a] if b not in inside and self.anum[b] > 0]
            if len(mine_in) != len(t_in) or len(mine_out) != len(t_out):
                continue
            if any(m.get(b) not in t_in for b in mine_in):
                continue
            if any(not self._link(a, b) for b in mine_out):
                continue
            ok.add(a)
        return ok

    # -- one group ---------------------------------------------------------------
    def patch(self) -> ViparrForcefield:
        for k, residues in enumerate(self.groups(), start=1):
            if self._known(residues):
                continue
            self._group(residues, k)
        params = {}
        for table, rows in self.rows.items():
            params[table] = [ParamRow(key, p, self.memo[key]) for key, plist in rows.items()
                             for p in plist]  # fmt: skip
        if self.host is None:
            rules = Rules(info=[f"GAFF2 {self.run_opts.get('gaff', '2.11')} (AmberTools)"],
                          vdw_func="lj12_6_sig_epsilon", vdw_comb_rule="arithmetic/geometric",
                          es_scale=[0.0, 0.0, 1 / 1.2], lj_scale=[0.0, 0.0, 0.5],
                          plugins=list(_PLUGINS))  # fmt: skip
        else:
            rules = Rules(es_scale=[], lj_scale=[])
        return ViparrForcefield("gaff2-patch", rules, self.templates, params)

    def _known(self, residues) -> bool:
        """Whether templates made for an earlier group match every residue."""
        if not self.templates:
            return False
        ff = ViparrForcefield("made", Rules(), self.templates, {})
        return all(self.P._find_uncached(ff, self.res_atoms[r])[0] is not None for r in residues)

    def _group(self, residues: list[int], k: int) -> None:
        atoms = [a for r in residues for a in self.res_atoms[r]]
        inside = set(atoms)
        protein: dict[int, tuple[Template, int]] = {}
        parents = {}
        for r in residues:
            t, m, ok = parents[r] = self._parent(r)
            for a in ok:
                protein[a] = (t, m[a])
        frag, f = self._capped(atoms, residues)
        index = {a: i for i, a in enumerate(frag)}
        name = self._template_name(residues)
        workdir = None if self.workdir is None else self.workdir / name
        p = run_gaff2(f, int(f.atoms["formal_charge"].sum()), workdir=workdir, **self.run_opts)
        self.runs.append((frag, p))

        nb = p.table("nonbonded")
        pid = nb.param_ids[np.argsort(nb.atoms[:, 0])]
        gtype, sigma, epsilon = (
            nb.params["type"][pid],
            nb.params["sigma"][pid],
            nb.params["epsilon"][pid],
        )
        gcharge, gmass = p.atoms["charge"], p.atoms["mass"]
        tag = f"~{k}"
        memo = f"GAFF2 {self.run_opts.get('gaff', '2.11')} with AmberTools ({name})"
        types: list[tuple[str, str] | None] = [None] * f.natoms
        gaff_atoms = set()
        for i, a in enumerate(frag):
            if a in inside:
                if a in protein:
                    t, ti = protein[a]
                    types[i] = (t.btype[ti], t.nbtype[ti])
                else:
                    types[i] = (gtype[i] + tag,) * 2
                    gaff_atoms.add(i)
                    self._add(
                        "vdw1",
                        (types[i][1],),
                        [{"sigma": float(sigma[i]), "epsilon": float(epsilon[i])}],
                        memo,
                    )
                    self._add("mass", (types[i][0],), [{"amu": float(gmass[i])}], memo)
            else:
                matched = self._matched(self.residue[a])
                types[i] = matched[a]
        impropers = self._terms(p, f, types, gaff_atoms, len(frag), memo)

        parts = [self.resnames[r] or "LIG" for r in residues]
        for i, r in enumerate(residues):
            t, m, _ = parents[r]
            part = parts[i] if parts.count(parts[i]) == 1 else f"{parts[i]}{i + 1}"
            tname = name if len(residues) == 1 else f"{name}_{part}"
            self.templates.append(self._template(r, tname, types, index, protein, gcharge, f,
                                                 t, m, impropers))  # fmt: skip

    def _template_name(self, residues) -> str:
        base = "+".join(self.resnames[r] or "LIG" for r in residues) + "_gaff2"
        taken = {t.name for t in self.templates} | (
            {t.name for t in self.host.templates} if self.host else set())  # fmt: skip
        name, n = base, 1
        while name in taken or any(x.startswith(name + "_") for x in taken):
            n += 1
            name = f"{base}{n}"
        return name

    def _capped(self, atoms, residues):
        """The group with ACE/NME caps: the neighbouring residue's atoms across
        each peptide bond, carbons among them closed with hydrogens."""
        inside = set(atoms)
        frag = list(atoms)
        index = {a: i for i, a in enumerate(frag)}
        new_h: list[tuple[int, np.ndarray]] = []
        for a in atoms:
            for b in self.nbrs[a]:
                if b in inside or self.anum[b] == 0:
                    continue
                if not self._peptide(a, b):
                    raise ViparrError(f"cannot cap the bond {self._label(a)}-{self._label(b)}: "
                                      "only peptide bonds are capped")  # fmt: skip
                if self._matched(self.residue[b]) is None:
                    raise ViparrError(f"{self._label(b)} has no template to cap with")
                if b in index:
                    continue
                index[b] = len(frag)
                frag.append(b)
                for y in self.nbrs[b]:
                    if y in index or self.anum[y] == 0:
                        continue
                    index[y] = len(frag)
                    frag.append(y)
                    heavy = [o for o in self.nbrs[y] if self.anum[o] > 0]
                    if self.anum[y] == 1 or len(heavy) == 1:
                        continue
                    if self.anum[y] != 6:
                        raise ViparrError(f"cannot cap at {self._label(y)}: only carbons "
                                          "become methyl groups")  # fmt: skip
                    for z in heavy:
                        if z != b and z not in index:
                            v = self.pos[z] - self.pos[y]
                            new_h.append((index[y], self.pos[y] + _CH * v / np.linalg.norm(v)))
        n = len(frag)
        pos = np.vstack([self.pos[frag], np.array([q for _, q in new_h]).reshape(-1, 3)])
        anum = np.array([self.anum[a] for a in frag] + [1] * len(new_h), np.int64)
        f = System.from_arrays(pos, anum=anum, resnames="MOL")
        f.atoms["formal_charge"] = np.array([self.formal[a] for a in frag] + [0] * len(new_h))
        pairs, orders = [], []
        for i, a in enumerate(frag):
            for b in self.nbrs[a]:
                j = index.get(b)
                if j is not None and i < j:
                    pairs.append((i, j))
                    orders.append(self.P.order[(min(a, b), max(a, b))])
        pairs += [(i, n + h) for h, (i, _) in enumerate(new_h)]
        orders += [1] * len(new_h)
        f.add_bonds(np.array(pairs, np.int64), order=np.array(orders, np.int64))
        if _needs_orders(f):
            from .chem import assign_bond_orders

            total = 0
            for r in residues:
                q = self.charges.get(self.resnames[r])
                total += int(q) if q is not None else sum(self.formal[a] for a in self.res_atoms[r])
            assign_bond_orders(f, charge=total)
        return frag, f

    def _add(self, table: str, types: tuple, params: list[dict], memo: str) -> None:
        rows = self.rows.setdefault(table, {})
        key, rkey = " ".join(types), " ".join(reversed(types))
        for kk in (key, rkey):
            if kk in rows:
                if _canon(rows[kk]) != _canon(params):
                    raise ViparrError(f"conflicting GAFF2 parameters for {table} {kk}")
                return
        rows[key] = params
        self.memo[key] = memo

    def _terms(self, p, f, types, gaff_atoms, ncap, memo) -> list[tuple[int, ...]]:
        """Rows for every term that touches a GAFF2 atom; returns the impropers."""
        bonded = set(zip(f.bonds["i"].tolist(), f.bonds["j"].tolist(), strict=True))
        bonded |= {(j, i) for i, j in bonded}

        def wanted(t) -> bool:
            if not any(i in gaff_atoms for i in t):
                return False
            if any(i >= ncap for i in t):
                raise ViparrError(
                    "a GAFF2 atom is next to a cap: the backbone N and C of a residue in "
                    "a protein must match its parent residue"
                )
            return True

        def rows_of(table, cols):
            tab = p.table(table)
            prm = {c: tab.params[c][tab.param_ids] for c in cols}
            for n, t in enumerate(tab.atoms.tolist()):
                yield tuple(t), {c: float(prm[c][n]) for c in cols}

        for table, cols in (("stretch_harm", ("r0", "fc")), ("angle_harm", ("theta0", "fc"))):
            for t, row in rows_of(table, cols):
                if wanted(t):
                    self._add(table, tuple(types[i][0] for i in t), [row], memo)
        cols = ("phi0", *(f"fc{k}" for k in range(7)))
        propers: dict[tuple, list[dict]] = {}
        impropers: dict[tuple, list[dict]] = {}
        for t, row in rows_of("dihedral_trig", cols):
            proper = (t[0], t[1]) in bonded and (t[1], t[2]) in bonded and (t[2], t[3]) in bonded
            (propers if proper else impropers).setdefault(t, []).append(row)
        seen = {min(t, t[::-1]) for t in propers}
        nbrs = [[] for _ in range(f.natoms)]
        for i, j in bonded:
            nbrs[i].append(j)
        for j in range(f.natoms):  # proper dihedrals tleap left out get zero rows
            for k in nbrs[j]:
                if k < j:
                    continue
                for i in nbrs[j]:
                    for m in nbrs[k]:
                        if (
                            i != k
                            and m != j
                            and m != i
                            and min((i, j, k, m), (m, k, j, i)) not in seen
                        ):
                            propers[(i, j, k, m)] = [dict(_ZERO)]
                            seen.add(min((i, j, k, m), (m, k, j, i)))
        for t, plist in propers.items():
            if wanted(t):
                self._add("dihedral_trig", tuple(types[i][0] for i in t), plist, memo)
        out = []
        for t, plist in impropers.items():
            if wanted(t):
                if len(plist) != 1:
                    raise ViparrError("an improper with several terms is not supported")
                self._add("improper_trig", tuple(types[i][0] for i in t), plist, memo)
                out.append(t)
        return out

    def _template(self, r, name, types, index, protein, gcharge, f, parent, pmap, impropers):
        ratoms = self.res_atoms[r]
        local = {a: i for i, a in enumerate(ratoms)}
        anum = [self.anum[a] for a in ratoms]
        names = _unique_names([self.names[a] for a in ratoms], anum)
        btype = [types[index[a]][0] for a in ratoms]
        nbtype = [types[index[a]][1] for a in ratoms]
        charge = [protein[a][0].charge[protein[a][1]] if a in protein else float(gcharge[index[a]])
                  for a in ratoms]  # fmt: skip
        bonds, ext, pinned = [], {}, {}
        for a in ratoms:
            for b in self.nbrs[a]:
                if self.anum[b] == 0:
                    continue
                if b in local:
                    if local[a] < local[b]:
                        bonds.append((local[a], local[b]))
                    continue
                if b not in ext:
                    ext[b] = len(names)
                    names.append(f"${len(ext)}")
                    anum.append(-1)
                    charge.append(0.0)
                    btype.append("")
                    nbtype.append("")
                    pinned[ext[b]] = self.anum[b]
                bonds.append((local[a], ext[b]))
        where = {**local, **ext}

        # formal charge by shifting the GAFF2 atoms evenly
        gaff = [i for i, a in enumerate(ratoms) if a not in protein]
        q = self.charges.get(self.resnames[r])
        want = int(q) if q is not None else int(round(sum(
            int(f.atoms["formal_charge"][index[a]]) for a in ratoms)))  # fmt: skip
        have = sum(charge[: len(ratoms)])
        if gaff:
            shift = (want - have) / len(gaff)
            for i in gaff:
                charge[i] += shift
            if abs(shift) > 0.1:
                warnings.warn(f"{self.resnames[r]}{self.resid[r]}: GAFF2 charges shifted by "
                              f"{shift:+.3f} e on each of {len(gaff)} atoms to reach {want:+d}",
                              ViparrWarning, stacklevel=4)  # fmt: skip
        elif abs(want - have) > 1e-3:
            warnings.warn(f"{self.resnames[r]}{self.resid[r]}: charge {have:+.4f}, not {want:+d}",
                          ViparrWarning, stacklevel=4)  # fmt: skip

        # the parent's impropers and CMAP where all their atoms keep its types
        tuples = {"impropers": [], "cmap": []}
        if parent is not None:
            inv = {i: a for a, i in pmap.items()}

            def system_atom(ti):
                if parent.anum[ti] > 0:
                    a = inv.get(ti)
                    return a if a in protein else None
                inner = inv.get(parent._nbrs[ti][0])
                if inner is None or inner not in protein:
                    return None
                out = [b for b in self.nbrs[inner] if b not in local and self.anum[b] > 0]
                return out[0] if len(out) == 1 else None

            for key, src in (("impropers", parent.impropers), ("cmap", parent.cmaps)):
                for tup in src:
                    mapped = [system_atom(ti) for ti in tup]
                    if all(a is not None for a in mapped):
                        tuples[key].append(tuple(where[a] for a in mapped))
        # GAFF2 impropers centred on this residue's atoms
        frag_of = {i: a for a, i in index.items()}
        for t in impropers:
            atoms = [frag_of[i] for i in t]
            if atoms[2] in local and all(a in where for a in atoms):
                tuples["impropers"].append(tuple(where[a] for a in atoms))
        return Template(name, names, anum, charge, btype, nbtype, [""] * len(names), bonds,
                        tuples["impropers"], tuples["cmap"], [], [], pinned)  # fmt: skip


def _unique_names(names: list[str], anum: list[int]) -> list[str]:
    """``names``, or element + count (C1, C2, ..., H1, ...) when they repeat."""
    if all(names) and len(set(names)) == len(names):
        return names
    from .elements import symbol

    counts: dict[str, int] = {}
    out = []
    for z in anum:
        sym = symbol(z) or "X"
        counts[sym] = counts.get(sym, 0) + 1
        out.append(f"{sym}{counts[sym]}")
    return out


def _canon(params: list[dict]) -> list:
    return sorted(tuple(sorted((k, round(v, 6)) for k, v in p.items())) for p in params)


def _needs_orders(f: System) -> bool:
    """Whether bond orders are missing: an atom lacks valence (a structure file's
    single bonds, possibly next to a ligand that has its orders)."""
    order = f.bonds["order"]
    val = np.zeros(f.natoms, np.int64)
    np.add.at(val, f.bonds["i"], order)
    np.add.at(val, f.bonds["j"], order)
    for z, fc, v in zip(f.atoms["anum"].tolist(), f.atoms["formal_charge"].tolist(),
                        val.tolist(), strict=True):  # fmt: skip
        want = _VALENCE.get(z)
        if want is not None and v < want + (fc if z in (7, 8, 15, 16) else -abs(fc)):
            return True
    return False
