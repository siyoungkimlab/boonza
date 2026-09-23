"""Martinize a protein: atoms to Martini 3 beads, and the beads' topology.

This follows martinize2 (vermouth) step for step, with vermouth's own data
files: each residue's atoms are averaged into beads by its mapping file,
the residue's block gives bead types and internal terms, termini and
protonation states retype beads, links add the terms between residues
(chosen by secondary structure), and an optional elastic network holds the
fold.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np

from .ff import (
    DATA,
    Interaction,
    read_ff,
    read_map,
    read_modification_mappings,
    read_rtp_parents,
)
from .links import CGMolecule, apply_links

# vermouth's element masses (AttachMass) and bond radii (MakeBonds), nm
_MASS = {"H": 1, "C": 12, "N": 14, "O": 16, "S": 32, "P": 31}
_RADIUS = {"H": 0.120, "C": 0.170, "N": 0.155, "O": 0.152, "S": 0.180, "P": 0.180, "SE": 0.19}
_BOND_FUDGE = 1.2
_SS_CG = {"1": "H", "2": "H", "3": "H", "H": "H", "G": "H", "I": "H", "B": "E", "E": "E",
          "T": "T", "S": "S", "C": "C", " ": "C", "P": "C", "-": "C", "F": "F"}  # fmt: skip
_HELIX = [(".H.", ".3."), (".HH.", ".33."), (".HHH.", ".333."), (".HHHH.", ".3333."),
          (".HHHHH.", ".13332."), (".HHHHHH.", ".113322."), (".HHHHHHH.", ".1113222."),
          (".HHHH", ".1111"), ("HHHH.", "2222.")]  # fmt: skip
# PDB names the reference files spell differently
_ALIASES = {("ILE", "CD1"): "CD"}
_TERMINAL_O = ("OXT", "OT2", "O2", "OC2")
# residue names read as another block, and the .rtp residue giving hydrogen parents
_RESNAMES = {"CYX": "CYS", "CYM": "CYS"}
_RTP = {"HIS": "HIS", "HSE": "HSE", "HSD": "HSD", "HSP": "HSP", "HIE": "HSE", "HID": "HSD",
        "HIP": "HSP", "ASH": "ASPP", "GLH": "GLUP", "LYN": "LSN"}  # fmt: skip


def convert_dssp_to_martini(sequence: str) -> str:
    """DSSP codes to Martini's: helices split into start (1), end (2), short (3)."""
    cg = "".join(_SS_CG[c] for c in sequence)
    wild = "." + "".join("H" if c == "H" else "." for c in cg) + "."
    for pattern, replacement in _HELIX:
        while pattern in wild:
            wild = wild.replace(pattern, replacement)
    return "".join(w if w != "." else c for w, c in zip(wild[1:-1], cg, strict=True))


@cache
def force_field(name: str = "martini3001"):
    ff = read_ff(sorted((DATA / name).glob("*.ff")), name)
    ff.maps = {p.name.split(".")[0].upper(): read_map(p)
               for p in (DATA / "mappings").glob("*.map")}  # fmt: skip
    ff.parents = read_rtp_parents(DATA / "charmm" / "aminoacids.rtp")
    charmm = read_ff([DATA / "charmm" / "modifications.ff"], "charmm")
    ff.mod_parents = {name: {a: b for e in mod.edges for a, b in (tuple(e), tuple(e)[::-1])
                             if a[0] == "H" and b[0] != "H"}
                      for name, mod in charmm.modifications.items()}  # fmt: skip
    ff.mod_maps = read_modification_mappings(DATA / "mappings" / "modifications.charmm36.mapping")
    return ff


@dataclass
class Martinized:
    """The Martini beads of a system and their GROMACS topology.

    ``molecules`` holds one topology per molecule (chains joined by a
    disulfide are one molecule), ``solvent`` the (name, count) of water and
    ion beads after them (see :func:`boonza.martini.solvate`), ``positions``
    every bead in Å, ``cell`` the box.  ``itp``/``top`` give GROMACS text;
    ``save`` writes the files; ``system`` loads them into a boonza System for
    OpenMM.
    """

    molecules: list
    positions: np.ndarray
    cell: np.ndarray | None
    names: list = field(default_factory=list)
    ss: str = ""
    solvent: list = field(default_factory=list)

    @property
    def nbeads(self) -> int:
        return len(self.positions)

    def itp(self, k: int = 0) -> str:
        return _write_itp(self.molecules[k], self.names[k])

    def top(self, martini_itp: str = "martini_v3.0.0.itp") -> str:
        lines = [f'#include "{martini_itp}"']
        lines += [f'#include "{n}.itp"' for n in self.names]
        if self.solvent:
            lines.append('#include "solvent.itp"')
        lines += ["", "[ system ]", "Martini system", "", "[ molecules ]"]
        lines += [f"{n} 1" for n in self.names]
        lines += [f"{n} {c}" for n, c in self.solvent if c]
        return "\n".join(lines) + "\n"

    def save(self, directory, martini_itp: str = "martini_v3.0.0.itp") -> Path:
        """Write ``topol.top``, one ``.itp`` per molecule (``solvent.itp`` for water
        and ions) and ``cg.gro``; returns the top."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for k, name in enumerate(self.names):
            (d / f"{name}.itp").write_text(self.itp(k))
        if self.solvent:
            (d / "solvent.itp").write_text(SOLVENT_ITP)
        (d / "topol.top").write_text(self.top(martini_itp))
        (d / "cg.gro").write_text(self._gro())
        return d / "topol.top"

    def system(self, martini_itp):
        """The beads as a boonza System, with the Martini force field of
        ``martini_itp`` (e.g. ``martini_v3.0.0.itp``) for nonbonded terms."""
        import tempfile

        from ..io.gromacs import load_top

        martini_itp = Path(martini_itp).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            top = self.save(tmp, martini_itp.name)
            s = load_top(top, Path(tmp) / "cg.gro", include_dirs=[str(martini_itp.parent)])
        return s

    def _gro(self) -> str:
        labels = [(n["input_resid"], n["resname"], n["atomname"])
                  for mol in self.molecules for n in mol.nodes]  # fmt: skip
        for name, count in self.solvent:
            resname = "W" if name == "W" else "ION"
            labels += [(r, resname, name) for r in range(1, count + 1)]
        rows = []
        for k, ((resid, resname, name), x) in enumerate(zip(labels, self.positions / 10,
                                                            strict=True), start=1):  # fmt: skip
            rows.append(f"{resid % 100000:5d}{resname[:5]:<5s}{name[:5]:>5s}{k % 100000:5d}"
                        f"{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}")  # fmt: skip
        if self.cell is not None and np.any(self.cell):
            box = np.asarray(self.cell) / 10
            v = [box[0, 0], box[1, 1], box[2, 2], box[0, 1], box[0, 2], box[1, 0], box[1, 2],
                 box[2, 0], box[2, 1]]  # fmt: skip
            tail = " ".join(f"{x:.5f}" for x in (v[:3] if not any(v[3:]) else v))
        else:
            x = self.positions / 10
            tail = " ".join(f"{e:.5f}" for e in (x.max(0) - x.min(0) + 2.0))
        return f"Martini beads\n{len(rows)}\n" + "\n".join(rows) + f"\n{tail}\n"


# Martini 3's water and ion molecules, as martini_v3.0.0_solvents_v1.itp and
# martini_v3.0.0_ions_v1.itp define them; their bead types are martini_v3.0.0.itp's
SOLVENT_ITP = """[ moleculetype ]
W 1

[ atoms ]
1 W 1 W W 1 0

[ moleculetype ]
NA 1

[ atoms ]
1 TQ5 1 ION NA 1 1.0

[ moleculetype ]
CL 1

[ atoms ]
1 TQ5 1 ION CL 1 -1.0 35.453
"""


@dataclass
class _Residue:
    resname: str
    resid: int
    insertion: str
    chain: str
    names: list
    elements: list
    xyz: np.ndarray  # nm


def martinize(system, atoms: str = "protein", *, ss: str | None = None,
              elastic: bool = False, elastic_fc: float = 700.0, elastic_lower: float = 0.0,
              elastic_upper: float = 9.0, elastic_decay: float = 0.0, elastic_power: float = 0.0,
              elastic_min_fc: float = 0.0, res_min_dist: int | None = None,
              cys: str | float = "auto", neutral_termini: bool = False, scfix: bool = True,
              extdih: bool = False, forcefield: str = "martini3001") -> Martinized:  # fmt: skip
    """Martini 3 beads and topology for the proteins of ``system``, as martinize2 makes them.

    ``ss``: secondary structure, one DSSP code per residue of ``atoms``; by
    default boonza's DSSP is run on the structure.  ``elastic`` adds
    martinize2's elastic network between backbone beads ``elastic_lower`` to
    ``elastic_upper`` Å apart (``-el``/``-eu``), with force constant
    ``elastic_fc`` kJ/mol/nm² (``-ef``), decay ``elastic_decay`` and
    ``elastic_power`` (``-ea``/``-ep``), dropping those below
    ``elastic_min_fc`` (``-em``) and those within ``res_min_dist`` residues
    apart in the residue graph (default 2, as martinize2's).  ``cys``: ``"auto"``
    bonds cysteines whose sulfurs are bonded, ``"none"`` never, or a
    distance in Å.  ``neutral_termini`` makes neutral termini;
    ``scfix``/``extdih`` as martinize2's (side-chain corrections on by
    default).  Hydrogens present in the structure decide protonation:
    Asp/Glu with a carboxyl hydrogen and Lys with two amine hydrogens are
    neutral, and His is typed by which ring nitrogens carry one.
    """
    ff = force_field(forcefield)
    residues, local = _residues(system, atoms)
    if not residues:
        raise ValueError(f"no atoms in {atoms!r}")
    unknown = sorted({r.resname for r in residues
                      if _RESNAMES.get(r.resname, r.resname) not in ff.blocks})  # fmt: skip
    if unknown:
        raise ValueError(f"not Martini 3 protein residues: {', '.join(unknown)}; leave them "
                         f"out of atoms={atoms!r}")  # fmt: skip
    bonds = _inter_residue_bonds(residues, cys, _system_bonds(system, local, residues))
    groups = _molecules(len(residues), bonds)
    if ss is None:
        ss = _dssp(system, atoms)
    if len(ss) != len(residues):
        raise ValueError(f"ss has {len(ss)} codes for {len(residues)} residues")
    neighbours = defaultdict(set)
    for a, _, b, _ in bonds:
        neighbours[a].add(b)
        neighbours[b].add(a)
    # vermouth's termini: residues bonded to exactly one other residue
    nter = {r for r, nb in neighbours.items() if len(nb) == 1 and r < min(nb)}
    cter = {r for r, nb in neighbours.items() if len(nb) == 1 and r > max(nb)}
    molecules, names, positions, missing = [], [], [], []
    for m, members in enumerate(groups):
        cg_ss = convert_dssp_to_martini("".join(ss[r] for r in members))
        mol = _build_molecule(ff, residues, members, cg_ss, bonds, nter, cter, neutral_termini,
                              missing)  # fmt: skip
        mol.meta = {"scfix": scfix, "extdih": extdih, "idr": False}
        apply_links(mol, ff.links)
        if elastic:
            _rubber_bands(mol, elastic_fc, elastic_lower / 10, elastic_upper / 10,
                          elastic_decay, elastic_power, elastic_min_fc,
                          ff.variables.get("elastic_network_res_min_dist", 2)
                          if res_min_dist is None else res_min_dist,
                          int(ff.variables.get("elastic_network_bond_type", 1)))  # fmt: skip
        molecules.append(mol)
        names.append(f"molecule_{m}")
        positions.extend(mol.positions)
    if missing:
        shown = ", ".join(missing[:10]) + (f" and {len(missing) - 10} more" if len(missing) > 10
                                           else "")  # fmt: skip
        raise ValueError(f"{len(missing)} beads have no atoms to place them, so rebuild the "
                         f"missing atoms first: {shown}")  # fmt: skip
    cell = getattr(system, "cell", None)
    return Martinized(molecules, np.asarray(positions) * 10, None if cell is None else
                      np.asarray(cell), names, "".join(ss))  # fmt: skip


def _residues(system, atoms) -> tuple[list[_Residue], dict]:
    """The residues of ``atoms`` in file order, and system atom -> (residue, index in it)."""
    from ..elements import symbol

    ids = system.select(atoms).ids
    res_of = np.asarray(system.atoms["residue"])[ids]
    names = np.asarray(system.atoms["name"])[ids]
    anum = np.asarray(system.atoms["anum"])[ids]
    pos = np.asarray(system.positions)[ids] / 10
    rtab = system.residues
    chain_names = np.asarray(system.chains["name"])
    out, where, local = [], {}, {}
    for k, r in enumerate(res_of.tolist()):
        if r not in where:
            where[r] = len(out)
            chain = str(chain_names[rtab["chain"][r]]).strip()
            out.append(_Residue(str(rtab["name"][r]).strip().upper(), int(rtab["resid"][r]),
                                str(rtab["insertion"][r]).strip(), chain, [], [], []))  # fmt: skip
        res = out[where[r]]
        name = str(names[k]).strip().upper()
        if name in res.names:  # an alternate location: keep the first
            continue
        local[int(ids[k])] = (where[r], len(res.names))
        res.names.append(name)
        res.elements.append(symbol(int(anum[k])).upper())
        res.xyz.append(pos[k])
    for res in out:
        res.xyz = np.asarray(res.xyz, float).reshape(-1, 3)
    return out, local


def _system_bonds(system, local, residues) -> set:
    """The system's own bonds between heavy atoms of different residues
    (from CONECT and SSBOND records, for instance), as martinize2 reads CONECT."""
    out = set()
    for i, j in zip(np.asarray(system.bonds["i"]).tolist(), np.asarray(system.bonds["j"]).tolist(),
                    strict=True):  # fmt: skip
        a, b = local.get(i), local.get(j)
        if a is not None and b is not None and a[0] != b[0]:
            if residues[a[0]].elements[a[1]] != "H" and residues[b[0]].elements[b[1]] != "H":
                out.add((*a, *b) if a < b else (*b, *a))
    return out


def _inter_residue_bonds(residues, cys, known=frozenset()):
    """Heavy-atom bonds between residues: the system's (``known``) and those
    vermouth's distance rule finds; with ``cys`` "none" no S-S bonds, with a
    number (Å) S-S bonds up to it."""
    atom_res, atom_idx, xyz, elem = [], [], [], []
    for r, res in enumerate(residues):
        for i, e in enumerate(res.elements):
            if e != "H" and e in _RADIUS:
                atom_res.append(r)
                atom_idx.append(i)
                xyz.append(res.xyz[i])
                elem.append(e)

    def is_ss(a, i, b, j):  # a cysteine bridge
        return residues[a].names[i] == residues[b].names[j] == "SG" and {
            residues[a].resname, residues[b].resname} <= {"CYS", "CYX", "CYM"}  # fmt: skip

    found = {bond for bond in known if not (cys == "none" and is_ss(*bond))}
    if not xyz:
        return sorted(found)
    from ..spatial import pairs_within

    reach = max(cys, 0.0) / 10 if isinstance(cys, int | float) else 0.0
    cut = max(max(_RADIUS[e] for e in elem) * _BOND_FUDGE, reach)
    ii, jj, d2 = pairs_within(np.asarray(xyz), cut)
    for a, b, dd in zip(ii.tolist(), jj.tolist(), np.sqrt(d2).tolist(), strict=True):
        ra, rb = atom_res[a], atom_res[b]
        if ra == rb:
            continue
        ss_bond = is_ss(ra, atom_idx[a], rb, atom_idx[b])
        if ss_bond and cys == "none":
            continue
        if ss_bond and isinstance(cys, int | float):
            ok = dd <= cys / 10
        else:
            ok = dd <= 0.5 * (_RADIUS[elem[a]] + _RADIUS[elem[b]]) * _BOND_FUDGE
        if ok:
            bond = (ra, atom_idx[a], rb, atom_idx[b])
            found.add(bond if ra < rb else (rb, atom_idx[b], ra, atom_idx[a]))
    return sorted(found)


def _molecules(n, bonds) -> list[list[int]]:
    parent = list(range(n))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, _, b, _ in bonds:
        parent[root(a)] = root(b)
    groups = defaultdict(list)
    for r in range(n):
        groups[root(r)].append(r)
    return sorted(groups.values(), key=lambda g: g[0])


def _dssp(system, atoms) -> str:
    from ..secondary import dssp

    ids = system.select(atoms).ids
    res = list(dict.fromkeys(np.asarray(system.atoms["residue"])[ids].tolist()))
    codes = dssp(system)[0][res]
    return "".join("C" if c in ("NA", " ", "") else str(c) for c in codes)


def _hydrogen_parents(res: _Residue) -> dict[int, int]:
    heavy = [i for i, e in enumerate(res.elements) if e != "H"]
    out = {}
    if not heavy:
        return out
    hx = res.xyz[heavy]
    for i, e in enumerate(res.elements):
        if e == "H":
            d = np.linalg.norm(hx - res.xyz[i], axis=1)
            if d.min() < 0.13:
                out[i] = heavy[int(d.argmin())]
    return out


def _canonical(res: _Residue, block_name: str) -> list[str]:
    names = [_ALIASES.get((block_name, n), n) for n in res.names]
    if "O" not in names and "OT1" in names:
        names[names.index("OT1")] = "O"
    return names


def _modifications(res: _Residue, names, parents, block_name) -> list[str]:
    """Protonation modifications, from where hydrogens sit."""
    carriers = defaultdict(int)
    for p in parents.values():
        carriers[names[p]] += 1
    mods = []
    if block_name == "ASP":
        mods += [f"ASP-HD{k}" for k in (1, 2) if carriers[f"OD{k}"]]
    elif block_name == "GLU":
        mods += [f"GLU-HE{k}" for k in (1, 2) if carriers[f"OE{k}"]]
    elif block_name == "LYS":
        if carriers["NZ"] == 2:
            mods.append("LYS-LSN")
        elif carriers["NZ"] == 3:
            mods.append("LYS-HZ3")
    elif block_name == "HIS":
        nd, ne = carriers["ND1"] > 0, carriers["NE2"] > 0
        if nd and ne:
            mods.append("HIS-HP")
        elif nd:
            mods.append("HIS-HD")
        elif ne:
            mods.append("HIS-HE")
    return mods


def _weights(ff, res, names, parents, block_name, mods):
    """Per input atom, {bead: weight}, as vermouth maps a repaired residue.

    Hydrogens take the names of the reference hydrogens on the same heavy
    atom (the residue's and those its modifications add); a modification's
    own mapping then overrides the weights of the atoms it lists."""
    amap = ff.maps.get(block_name)
    if amap is None:
        raise ValueError(f"no Martini mapping for residue {block_name}")
    h_names = defaultdict(list)  # heavy atom -> reference hydrogen names
    for h, p in ff.parents.get(_RTP.get(block_name, block_name), {}).items():
        h_names[p].append(h)
    for m in mods:
        for h, p in ff.mod_parents.get(m, {}).items():
            if h not in h_names[p]:
                h_names[p].append(h)
    canon = list(names)
    taken = defaultdict(int)
    for i, p in sorted(parents.items()):
        pool = h_names.get(names[p], [])
        k = taken[names[p]]
        taken[names[p]] += 1
        canon[i] = pool[k] if k < len(pool) else None
    out, unknown = [], []
    for i, n in enumerate(canon):
        if res.elements[i] == "H":
            w = dict(amap.get(n, {})) if n else {}
        elif n in amap:
            w = dict(amap[n])
        elif n in _TERMINAL_O:
            w = {}
        else:
            w = {}
            unknown.append(res.names[i])
        for m in mods:
            w.update(ff.mod_maps.get(m, {}).get(n, {}))
        out.append(w)
    if unknown:
        raise ValueError(f"residue {block_name} {res.chain}{res.resid}{res.insertion}: no "
                         f"mapping for atom(s) {', '.join(unknown)}")  # fmt: skip
    return out


def _build_molecule(ff, residues, members, cg_ss, bonds, nter, cter, neutral,
                    missing) -> CGMolecule:  # fmt: skip
    mol = CGMolecule()
    beads_of_atom = {}
    for serial, r in enumerate(members, start=1):
        res = residues[r]
        block_name = _RESNAMES.get(res.resname, res.resname)
        block = ff.blocks.get(block_name)
        if block is None:
            raise ValueError(f"residue {res.resname} {res.chain}{res.resid}: not a Martini 3 "
                             f"protein residue")  # fmt: skip
        names = _canonical(res, block_name)
        parents = _hydrogen_parents(res)
        mods = _modifications(res, names, parents, block_name)
        if r in nter:
            mods.append("NH2-ter" if neutral else "N-ter")
        if r in cter:
            mods.append("COOH-ter" if neutral else "C-ter")
        weights = _weights(ff, res, names, parents, block_name, mods)
        mass = np.array([_MASS.get(e, 30) for e in res.elements], float)
        index = {}
        for name, attrs in block.atoms.items():
            w = np.array([wt.get(name, 0.0) for wt in weights]) * mass
            if w.sum() < 1e-7:
                missing.append(f"{res.resname} {res.chain}{res.resid}{res.insertion} ({name})")
                w = np.ones(len(w))
            node = {k: v for k, v in attrs.items() if k != "resid"}
            node.update(resid=serial, chain=res.chain, input_resid=res.resid,
                        insertion=res.insertion, modifications=[])  # fmt: skip
            if cg_ss:
                node["cgsecstruct"] = cg_ss[serial - 1]
            index[name] = mol.add_node(node, (w[:, None] * res.xyz).sum(0) / w.sum())
        for mname in mods:
            mod = ff.modifications[mname]
            for bead, mattrs in mod.nodes.items():
                if bead in index:
                    node = mol.nodes[index[bead]]
                    replace = dict(mattrs.get("replace", {}))
                    if "charge" in replace:
                        replace["charge"] = float(replace["charge"])
                    node.update(replace)
                    node["modifications"].append(mname)
        for e in block.edges:
            a, b = tuple(e)
            mol.add_edge(index[a], index[b])
        for kind, lst in block.interactions.items():
            for t in lst:
                atoms = tuple(index[a] for a in t.atoms)
                mol.add(kind, Interaction(atoms, list(t.params), dict(t.meta)))
        for i, wt in enumerate(weights):
            beads_of_atom[(r, i)] = [index[b] for b in wt if b in index]
    for a, i, b, j in bonds:
        if (a, i) in beads_of_atom and (b, j) in beads_of_atom:
            for x in beads_of_atom[(a, i)]:
                for y in beads_of_atom[(b, j)]:
                    mol.add_edge(x, y)
    return mol


def _rubber_bands(mol, fc, lower, upper, decay, power, min_fc, res_min_dist, bond_type):
    """vermouth's ApplyRubberBand on the BB beads of one molecule (lengths in nm)."""
    sel = [k for k, n in enumerate(mol.nodes) if n["atomname"] == "BB"]
    if len(sel) < 2:
        return
    from ..spatial import pairs_within

    x = np.asarray([mol.positions[k] for k in sel])
    ii, jj, d2 = pairs_within(x, upper)  # pairs i < j, in order, as vermouth's triu
    d = np.sqrt(d2)
    k = np.exp(-decay * (d**power)) if decay and power else np.ones_like(d)
    k *= fc
    k[k < min_fc] = 0
    k[k > fc] = fc
    k[(d > upper) | (d < lower)] = 0
    close = _residues_within(mol, res_min_dist)
    resid = [mol.nodes[s]["resid"] for s in sel]
    for a, b, dist, fk in zip(ii.tolist(), jj.tolist(), d.round(5).tolist(), k.tolist(),
                              strict=True):  # fmt: skip
        if fk > min_fc and resid[b] not in close[resid[a]]:
            mol.interactions["bonds"].append(Interaction(
                (sel[a], sel[b]), [str(bond_type), f"{dist:.5f}", _num(fk)],
                {"group": "Rubber band"}))  # fmt: skip


def _residues_within(mol, cutoff) -> dict[int, set]:
    """Residues within ``cutoff`` bonds of each residue in the residue graph."""
    nbr = defaultdict(set)
    for a, adj in enumerate(mol.adj):
        for b in adj:
            ra, rb = mol.nodes[a]["resid"], mol.nodes[b]["resid"]
            if ra != rb:
                nbr[ra].add(rb)
    out = {}
    for start in {n["resid"] for n in mol.nodes}:
        seen, frontier = {start}, {start}
        for _ in range(int(cutoff)):
            frontier = {y for x in frontier for y in nbr[x]} - seen
            seen |= frontier
        out[start] = seen
    return out


def _num(x) -> str:
    return f"{x:g}" if math.isfinite(x) else str(x)


_SECTIONS = ("bonds", "constraints", "pairs", "angles", "dihedrals", "impropers",
             "virtual_sitesn", "exclusions")  # fmt: skip


def _write_itp(mol: CGMolecule, name: str) -> str:
    out = ["[ moleculetype ]", f"{name} 1", "", "[ atoms ]"]
    for k, n in enumerate(mol.nodes, start=1):
        mass = f" {_num(n['mass'])}" if "mass" in n else ""
        out.append(f"{k:5d} {n['atype']:<6s} {n['input_resid']:5d} {n['resname']:<5s} "
                   f"{n['atomname']:<5s} {k:5d} {_num(float(n['charge'])):>6s}{mass}")  # fmt: skip
    for kind in _SECTIONS:
        lst = mol.interactions.get(kind, [])
        if not lst:
            continue
        out += ["", f"[ {kind} ]"]
        groups = defaultdict(list)
        for t in lst:
            cond = ("ifdef", t.meta["ifdef"]) if "ifdef" in t.meta else (
                ("ifndef", t.meta["ifndef"]) if "ifndef" in t.meta else None)  # fmt: skip
            groups[cond].append(t)
        for cond, ts in groups.items():
            if cond:
                out.append(f"#{cond[0]} {cond[1]}")
            for t in ts:
                idx = [str(a + 1) for a in t.atoms]
                params = [str(p) for p in t.params]
                if kind == "virtual_sitesn":
                    cols = [idx[0], params[0], *idx[1:]]
                else:
                    cols = [*idx, *params]
                comment = f" ; {t.meta['comment']}" if "comment" in t.meta else ""
                out.append(" ".join(cols) + comment)
            if cond:
                out.append("#endif")
    return "\n".join(out) + "\n"
