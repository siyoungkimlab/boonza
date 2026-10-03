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
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np

# what counts as holding a cofactor, and as burying one, are the same questions
# for either model, so they are asked in one place
from ..cofactors import (
    COFACTOR_FC,
    COFACTOR_NEIGHBOURS,
    COFACTOR_REACH,
    COFACTOR_TETHERS,
    COORDINATION,
    anchor_bands,
    shape_bands,
    warn_exposed,
)
from .ff import (
    DATA,
    Interaction,
    read_ff,
    read_map,
    read_modification_mappings,
    read_rtp_parents,
)
from .links import CGMolecule, apply_links

#: What Martini calls the backbone bead, and what a viewer wants it called: a
#: chain is traced through CA, and with BB most viewers draw beads and no more.
BACKBONE_BEAD = "BB"
#: What :meth:`Martinized.for_viewing` calls the cts of the system it returns,
#: so a file with its elastic network taken out says so and is not run by
#: mistake (:func:`boonza.md.prepare._already_built` reads it).
VIEWING_MARK = "boonza view: no elastic network"
VIEWING_BACKBONE = "CA"

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
#: What a force field calls a residue another names differently.  Martini 2.2
#: takes CHARMM's histidines, and at its resolution HSD and HSE are the same four
#: beads.  Its neutral acids are residues of their own (ASP0, GLU0) rather than
#: modifications, and no mapping reaches them from an all-atom structure -- not
#: in vermouth either -- so a protonated aspartate stays charged under Martini 2,
#: which is what martinize2 does with it.
INSTEAD_OF = {
    "martini22": {
        "residue": {
            "HIS": "HSD",
            "HID": "HSD",
            "HIE": "HSE",
            "HIP": "HSP",
            "ASH": "ASP0",
            "ASPP": "ASP0",
            "GLH": "GLU0",
            "GLUP": "GLU0",
            "LYN": "LSN",
            "HYP": "PRO",
            "GLYM": "GLY",
            "CYSF": "CYS",
            "CYSG": "CYS",
            "CYSP": "CYS",
        },  # fmt: skip
    },
}


def _instead_of(ff, what: str, name: str, fallback=None):
    """What ``ff`` calls ``name``, where it calls it something else."""
    return INSTEAD_OF.get(getattr(ff, "name", ""), {}).get(what, {}).get(name, fallback)


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
    maps = DATA / "mappings" / name  # which atoms make which bead is the version's own
    ff.maps = {p.name.split(".")[0].upper(): read_map(p) for p in maps.glob("*.map")}
    ff.parents = read_rtp_parents(DATA / "charmm" / "aminoacids.rtp")
    charmm = read_ff([DATA / "charmm" / "modifications.ff"], "charmm")
    ff.mod_parents = {name: {a: b for e in mod.edges for a, b in (tuple(e), tuple(e)[::-1])
                             if a[0] == "H" and b[0] != "H"}
                      for name, mod in charmm.modifications.items()}  # fmt: skip
    ff.mod_maps = read_modification_mappings(maps / "modifications.charmm36.mapping")
    return ff


def _nonbonded(martini_itp, martini: int = 3):
    """The Martini parameter file to include: the one given, or the one carried
    for that version of Martini."""
    if martini_itp is not None:
        return martini_itp
    from . import NONBONDED_FOR, parameters

    return parameters(NONBONDED_FOR[martini])[0]


def _solvent_include(martini: int = 3):
    """What defines water and ions: the ``solvent.itp`` boonza writes for
    Martini 3, whose parameter file holds no moleculetype, or Martini 2's own
    ion file (its water comes with ``martini_v2.2.itp``)."""
    if martini == 3:
        return "solvent.itp"
    from . import IONS_FOR, parameters

    return parameters(IONS_FOR[martini])[0]


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
    copies: list = field(default_factory=list)  # how many of each molecule (1 each by default)
    lipids: list = field(default_factory=list)  # (name, count, bead names), in order
    includes: list = field(default_factory=list)  # topology files the lipids come from
    martini: int = 3  # which Martini the beads are, for the parameters to include

    @property
    def nbeads(self) -> int:
        return len(self.positions)

    def itp(self, k: int = 0) -> str:
        return _write_itp(self.molecules[k], self.names[k])

    def top(self, martini_itp=None) -> str:
        lines = [f'#include "{_nonbonded(martini_itp, self.martini)}"']
        lines += [f'#include "{p}"' for p in self.includes]
        lines += [f'#include "{n}.itp"' for n in self.names]
        if any(c for _, c in self.solvent):  # a list of zero counts is no solvent
            lines.append(f'#include "{_solvent_include(self.martini)}"')
        lines += ["", "[ system ]", "Martini system", "", "[ molecules ]"]
        lines += [f"{n} {c}" for n, c in zip(self.names, self.molecule_copies, strict=True)]
        lines += [f"{n} {c}" for n, c, _ in self.lipids]
        lines += [f"{n} {c}" for n, c in self.solvent if c]
        return "\n".join(lines) + "\n"

    def save(self, directory, martini_itp=None, system=None) -> Path:
        """Write ``topol.top``, one ``.itp`` per molecule (``solvent.itp`` for
        water and ions), ``cg.dms`` and ``cg.gro``; returns the top.

        ``topol.top`` includes ``martini_itp``, the file boonza carries for
        this version of Martini when none is given, so that GROMACS can
        resolve it as written.  ``solvent.itp`` is written for Martini 3
        only: Martini 2 defines its water and ions upstream.

        The coordinates go out twice over: ``cg.dms`` holds them as they are,
        with the chains the beads came from, and is what boonza reads back;
        ``cg.gro`` is what GROMACS needs, and rounds them to 0.001 nm.
        ``system`` is the built system when the caller already has one, which
        saves building it again; without one, and with parameters named to be
        resolved somewhere else, only the ``.gro`` is written.
        """
        from ..io import GromacsError
        from ..io import save as save_structure

        top = self._write_topology(directory, martini_itp)
        if system is None:
            try:
                system = self.system(martini_itp)
            except (GromacsError, FileNotFoundError):
                # the topology names parameters to be resolved elsewhere, so
                # there is no system to write here; the .gro carries the
                # coordinates, as it does for GROMACS
                return top
        save_structure(system, top.parent / "cg.dms")
        return top

    def elastic_bonds(self) -> list[tuple[int, int]]:
        """The rubber bands, as pairs of bead indices in the built system.

        They are bonds of the topology like any other, so a viewer draws them
        and a protein comes out a hairball; :func:`for_viewing` leaves them
        out.
        """
        out, offset = [], 0
        for mol, count in zip(self.molecules, self.molecule_copies, strict=True):
            for _ in range(count):
                for bond in mol.interactions["bonds"]:
                    if bond.meta.get("group") == "Rubber band":
                        i, j = (int(a) + offset for a in bond.atoms)
                        out.append((min(i, j), max(i, j)))
                offset += len(mol.nodes)
        return out

    def for_viewing(self, system=None, martini_itp=None, backbone_as_ca: bool = False):
        """The system without its elastic network: what to open in a viewer.

        Everything else is kept, atom for atom and in the same order, so the
        trajectory still lines up with it.  Its cts are named ``VIEWING_MARK``,
        which is how a file that has had bonds taken out of it says so.

        ``backbone_as_ca`` names the backbone bead ``CA`` as well, which is
        what a viewer traces a chain through -- but it is off by default,
        because a viewer that knows amino acids reads ``GLU: CA SC1`` as a
        broken residue and draws its own bonds over it, which is worse than no
        trace at all.  Turn it on for a viewer that wants a CA trace and
        perceives no bonds of its own.
        """
        s = (system if system is not None else self.system(martini_itp)).clone()
        bands = self.elastic_bonds()
        if bands:
            ids = [s.find_bond(s.atom(i), s.atom(j)) for i, j in bands]
            s.delete_bonds([b for b in ids if b is not None])
        if backbone_as_ca:
            names = s.atoms["name"]
            for a in np.flatnonzero(np.asarray(names) == BACKBONE_BEAD).tolist():
                names[a] = VIEWING_BACKBONE
        for c in range(s.ncts):  # a ct name survives a .dms and a .mae alike
            s.ct(c).name = VIEWING_MARK
        return s

    def _write_topology(self, directory, martini_itp=None) -> Path:
        """The topology, its molecules and the .gro, without building a system:
        what :meth:`system` needs to read its parameters back."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for k, name in enumerate(self.names):
            (d / f"{name}.itp").write_text(self.itp(k))
        if any(c for _, c in self.solvent) and self.martini == 3:
            (d / "solvent.itp").write_text(SOLVENT_ITP)
        (d / "topol.top").write_text(self.top(martini_itp))
        (d / "cg.gro").write_text(self._gro())
        return d / "topol.top"

    def system(self, martini_itp=None):
        """The beads as a boonza System, with the Martini force field of
        ``martini_itp`` for nonbonded terms; the file boonza carries for this
        version of Martini when none is given.

        The topology is written out and read back, since that is where Martini
        keeps its parameters, but the positions are the ones held here: a .gro
        rounds them to 0.01 Å, and no coordinate file of GROMACS has a chain
        column, which the beads do carry.
        """
        import tempfile

        from ..io.gromacs import load_top

        martini_itp = Path(_nonbonded(martini_itp, self.martini)).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            top = self._write_topology(tmp, martini_itp.name)
            s = load_top(top, include_dirs=[str(martini_itp.parent)])
        if s.natoms != len(self.positions):
            raise ValueError(f"the topology built {s.natoms} beads where this system holds "
                             f"{len(self.positions)}: does one of its molecules take a name "
                             "Martini already uses (W, WF, ION, NA, CL), whose definition wins "
                             "over a later one?")  # fmt: skip
        s.positions = np.asarray(self.positions, float)
        if self.cell is not None and np.any(self.cell):
            s.cell = np.asarray(self.cell, float)
        self._name_chains(s)
        return s

    def _name_chains(self, s) -> None:
        """Give the beads' residues the chains they came from.

        A molecule's beads know their chain, and the lipids and the solvent
        have none of their own; a system built from a composition alone knows
        no chains at all, and keeps the one chain it was read with.
        """
        of_bead = []
        for mol, count in zip(self.molecules, self.molecule_copies, strict=True):
            mine = [str(n.get("chain", "") or "") for n in mol.nodes]
            of_bead += mine * count
        if not any(of_bead):
            return
        of_bead += [""] * (s.natoms - len(of_bead))  # lipids, water, ions
        residue = np.asarray(s.atoms["residue"])
        first = np.zeros(s.nresidues, np.int64)
        first[residue[::-1]] = np.arange(s.natoms)[::-1]  # the first bead of each residue
        want = [of_bead[int(a)] for a in first]
        chains = {str(s.chains["name"][c]): s.chain(c) for c in range(s.nchains)}
        for name in dict.fromkeys(want):
            if name not in chains:
                chains[name] = s.add_chain(name=name)
        for r, name in enumerate(want):
            s.residue(r).chain = chains[name]
        s._prune_hierarchy()  # the chains the topology was read with are empty now

    @property
    def molecule_copies(self) -> list:
        """How many of each molecule the system holds; one each unless set."""
        return self.copies or [1] * len(self.molecules)

    def _gro(self) -> str:
        labels = []
        for mol, count in zip(self.molecules, self.molecule_copies, strict=True):
            rows = [(n["input_resid"], n["resname"], n["atomname"]) for n in mol.nodes]
            for c in range(count):
                # copies of one molecule carry on its residue numbering
                labels += [(r + c * len(mol.nodes), name, bead) for r, name, bead in rows]
        resid = 0
        for name, count, beads in self.lipids:
            for _ in range(count):
                resid += 1
                labels += [(resid, name, bead) for bead in beads]
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


#: What an inert cofactor bead is, per force field: the smallest apolar bead each
#: version has, which is an alanine's side chain in Martini 3 and the apolar bead
#: Martini 2 builds one from.  Uncharged and unremarkable on purpose -- see
#: :func:`martinize`'s ``cofactors``.
COFACTOR_BEAD = {"martini3001": "TC3", "martini22": "C1"}


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
              elastic: bool = False, elastic_selection: str | None = None,
              elastic_fc: float = 700.0, elastic_lower: float = 0.0,
              elastic_upper: float = 9.0, elastic_decay: float = 0.0, elastic_power: float = 0.0,
              elastic_min_fc: float = 0.0, res_min_dist: int | None = None,
              cys: str | float = "auto", neutral_termini: bool = False, scfix: bool = True,
              extdih: bool = False, forcefield: str = "martini3001",
              cofactors: bool = False, cofactor_fc: float = COFACTOR_FC,
              cofactor_neighbours=COFACTOR_NEIGHBOURS, cofactor_tethers=COFACTOR_TETHERS,
              cofactor_reach: float = COFACTOR_REACH) -> Martinized:  # fmt: skip
    """Martini beads and topology for the proteins of ``system``, as martinize2 makes them.

    ``forcefield`` is the version: ``"martini3001"`` or ``"martini22"``, each
    read from vermouth's own files for it.  Martini 2.2 leans on secondary
    structure where Martini 3 does not, names its residues as CHARMM does
    (``HSD`` and not ``HIS``, which boonza renames to), and cuts a tryptophan's
    rings and reads a phenylalanine's along other lines, so it maps a residue
    onto beads by its own rules.

    ``cofactors`` maps what the force field has no residue for -- a structural
    zinc, an ADP, a heme -- as inert beads rather than refusing it: one uncharged
    apolar bead per heavy atom, every pair inside it excluded, its shape held by
    bands and the whole thing banded to whatever protein beads lie within
    :data:`COORDINATION` of it (``cofactor_fc`` kJ/mol/nm², as the elastic
    network's).  It is there to hold the room up, so that probes cannot enter
    space the cofactor occupies, and to say nothing else: no charge and no
    chemistry are invented, and nothing can distort the fold, which the bands and
    the elastic network hold.  That is honest only while the cofactor is buried,
    where nothing can reach it to read its chemistry wrongly, so one that is not
    is warned about rather than quietly approximated.  A cofactor with parameters
    of its own deserves them instead: include its ``.itp`` and leave it out of
    ``atoms``.

    ``ss``: secondary structure, one DSSP code per residue of ``atoms``; by
    default boonza's DSSP is run on the structure, and ``ss=False`` leaves it
    unassigned (the backbone then takes the coil terms, as martinize2's
    links give a residue with no secondary structure).  ``elastic`` adds
    martinize2's elastic network between backbone beads ``elastic_lower`` to
    ``elastic_upper`` Å apart (``-el``/``-eu``), over every molecule or only
    the residues ``elastic_selection`` picks -- a receptor held rigid while a
    peptide bound to it stays free, say; a band never joins two molecules
    either way, since a molecule's bonds are its own, with force constant
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
    networked = None
    if elastic_selection is not None:
        if not elastic:
            raise ValueError("elastic_selection has no network to trim: give elastic=True")
        networked = _residue_keys(system, elastic_selection)
        if not networked:
            raise ValueError(f"elastic_selection {elastic_selection!r} selects no residues")
    if not residues:
        raise ValueError(f"no atoms in {atoms!r}")

    def named(resname: str) -> str:
        plain = _RESNAMES.get(resname, resname)
        return _instead_of(ff, "residue", plain, plain)

    bead = COFACTOR_BEAD.get(ff.name) if cofactors else None
    if cofactors and bead is None:
        raise ValueError(f"no inert cofactor bead is known for {ff.name}")
    nameless = [r for r in residues if named(r.resname) not in ff.blocks]
    # a residue the force field has no block for is not therefore a cofactor: a
    # D-amino acid or a modified one is part of the chain, and mapping it inert
    # would throw away a side chain in the middle of a protein.  What the
    # structure's own reading calls a polymer is refused rather than approximated
    inchain = sorted({r.resname for r in nameless
                      if r.resname in _polymer_resnames(system)}) if cofactors else []  # fmt: skip
    if inchain:
        raise ValueError(f"not {ff.name} residues, and part of a chain rather than cofactors: "
                         f"{', '.join(inchain)}.  Rename them to the residues they are, or leave "
                         "them out of atoms")  # fmt: skip
    unknown = sorted({r.resname for r in nameless}) if not cofactors else []
    if cofactors:
        warn_exposed(system, [(f"{r.resname} {r.chain}{r.resid}",
                               np.asarray(r.xyz, float) * 10) for r in nameless])  # fmt: skip
    if unknown:
        raise ValueError(f"not {ff.name} protein residues: {', '.join(unknown)}; leave them "
                         f"out of atoms={atoms!r}, give them parameters of their own, or "
                         "cofactors=True to hold them as inert beads")  # fmt: skip
    known = _system_bonds(system, local, residues)
    bonds = _inter_residue_bonds(residues, cys, known)
    _check_links_across_chains(residues, bonds, known)
    if cofactors:
        # what holds a cofactor has to be a bond before the molecules are split,
        # or the thing lands in a moleculetype of its own, where nothing in
        # GROMACS can bond it to the protein it belongs to and it floats away
        bonds = sorted(set(bonds) | _coordination_bonds(residues, nameless))
    groups = _molecules(len(residues), bonds)
    if ss is None:
        ss = _dssp(system, atoms)
    elif ss is False:  # no secondary structure: the coil terms, and no more
        ss = ""
    if ss and len(ss) != len(residues):
        raise ValueError(f"ss has {len(ss)} codes for {len(residues)} residues")
    neighbours = defaultdict(set)
    for a, _, b, _ in bonds:
        neighbours[a].add(b)
        neighbours[b].add(a)
    # vermouth's termini: residues bonded to exactly one other residue
    nter = {r for r, nb in neighbours.items() if len(nb) == 1 and r < min(nb)}
    cter = {r for r, nb in neighbours.items() if len(nb) == 1 and r > max(nb)}
    molecules, names, positions, missing = [], [], [], []
    unmodified: set[str] = set()
    for m, members in enumerate(groups):
        cg_ss = convert_dssp_to_martini("".join(ss[r] for r in members)) if ss else ""
        mol = _build_molecule(ff, residues, members, cg_ss, bonds, nter, cter, neutral_termini,
                              missing, unmodified, bead, cofactor_fc,
                              cofactor_neighbours, cofactor_tethers,
                              cofactor_reach)  # fmt: skip
        mol.meta = {"scfix": scfix, "extdih": extdih, "idr": False}
        apply_links(mol, ff.links)
        if elastic:
            _rubber_bands(mol, elastic_fc, elastic_lower / 10, elastic_upper / 10,
                          elastic_decay, elastic_power, elastic_min_fc,
                          ff.variables.get("elastic_network_res_min_dist", 2)
                          if res_min_dist is None else res_min_dist,
                          int(ff.variables.get("elastic_network_bond_type", 1)),
                          networked)  # fmt: skip
        molecules.append(mol)
        names.append(f"molecule_{m}")
        positions.extend(mol.positions)
    if missing:
        shown = ", ".join(missing[:10]) + (f" and {len(missing) - 10} more" if len(missing) > 10
                                           else "")  # fmt: skip
        raise ValueError(f"{len(missing)} beads have no atoms to place them, so rebuild the "
                         f"missing atoms first: {shown}")  # fmt: skip
    if unmodified:
        warnings.warn(f"{ff.name} has no {', '.join(sorted(unmodified))}: those residues keep "
                      "the form it does have, Martini 2 having no neutral aspartate or "
                      "glutamate and one histidine", stacklevel=2)  # fmt: skip
    cell = getattr(system, "cell", None)
    from . import FORCEFIELD_FOR

    version = next((v for v, name in FORCEFIELD_FOR.items() if name == forcefield), 3)
    return Martinized(molecules, np.asarray(positions) * 10, None if cell is None else
                      np.asarray(cell), names, "".join(ss), martini=version)  # fmt: skip


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


def _check_links_across_chains(residues, bonds, known) -> None:
    """Refuse a covalent link the distance rule invented between two chains.

    Heavy atoms of different residues closer than they can be unbonded are
    taken as bonded, as martinize2 does, which is how a peptide bond or a
    disulfide is found.  Between two chains that is a disulfide often enough --
    insulin, an antibody -- and those are left alone; anything else is almost
    always a clash in the structure, and it would quietly join a receptor and
    its ligand into one molecule, one elastic network over both, with the
    ligand unable to leave.  A link that is truly there belongs in the file's
    own connectivity, which is taken as given.
    """
    guilty = []
    for a, i, b, j in bonds:
        if (a, i, b, j) in known or residues[a].chain == residues[b].chain:
            continue
        if residues[a].names[i] == residues[b].names[j] == "SG":
            continue  # a disulfide between two chains is ordinary: insulin, an antibody
        d = 10 * float(np.linalg.norm(residues[a].xyz[i] - residues[b].xyz[j]))
        guilty.append(f"{residues[a].chain}/{residues[a].resname}{residues[a].resid} "
                      f"{residues[a].names[i]} - {residues[b].chain}/{residues[b].resname}"
                      f"{residues[b].resid} {residues[b].names[j]} ({d:.2f} A)")  # fmt: skip
    if guilty:
        more = f" and {len(guilty) - 5} more" if len(guilty) > 5 else ""
        raise ValueError(
            "two chains come close enough to look bonded, which would make them one molecule "
            f"with one elastic network over both: {'; '.join(guilty[:5])}{more}.  Separate them, "
            "or give the bond in the structure's own connectivity if it is real"
        )


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
                    missing, unmodified=None, cofactors=None,
                    cofactor_fc: float = COFACTOR_FC,
                    neighbours=COFACTOR_NEIGHBOURS, tethers=COFACTOR_TETHERS,
                    reach: float = COFACTOR_REACH) -> CGMolecule:  # fmt: skip
    mol = CGMolecule()
    beads_of_atom = {}
    inert: list = []  # (residue, its beads) for every cofactor mapped as inert
    for serial, r in enumerate(members, start=1):
        res = residues[r]
        block_name = _RESNAMES.get(res.resname, res.resname)
        block_name = _instead_of(ff, "residue", block_name, block_name)
        block = ff.blocks.get(block_name)
        if block is None and cofactors is not None:
            # not a residue of the force field, and asked for as a cofactor: one
            # bead per heavy atom, holding the room up and saying nothing
            heavy = [i for i, e in enumerate(res.elements) if e != "H"]
            index = {}
            for i in heavy:
                node = dict(atype=cofactors, resname=res.resname[:5],
                            atomname=res.names[i][:5], charge=0.0,
                            mass=float(_MASS.get(res.elements[i], 30)), resid=serial,
                            chain=res.chain, input_resid=res.resid,
                            insertion=res.insertion, modifications=[],
                            cofactor=True)  # fmt: skip
                index[res.names[i]] = mol.add_node(node, res.xyz[i])
                beads_of_atom[(r, i)] = [index[res.names[i]]]
            inert.append((r, [index[res.names[i]] for i in heavy]))
            continue
        if block is None:
            raise ValueError(f"residue {res.resname} {res.chain}{res.resid}: not a "
                             f"{ff.name} protein residue; cofactors=True maps it as inert "
                             "beads instead")  # fmt: skip
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
            mod = ff.modifications.get(mname)
            if mod is None:
                # the structure says this residue is in a form the force field
                # does not have: Martini 2 has no neutral aspartate or glutamate
                # and one histidine, where Martini 3 has the tautomers.  It keeps
                # the form the model knows, which is all the model can say
                if unmodified is not None:
                    unmodified.add(mname)
                continue
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
    if inert:
        _hold_cofactors(mol, residues, inert, cofactor_fc, neighbours, tethers, reach)
    return mol


def _hold_cofactors(mol: CGMolecule, residues, inert, fc: float,
                    neighbours=COFACTOR_NEIGHBOURS, tethers=COFACTOR_TETHERS,
                    reach: float = COFACTOR_REACH) -> None:  # fmt: skip
    """Hold each inert cofactor in shape and in place, and let nothing inside it
    feel anything.

    A bead per heavy atom sits where its atom did, which is about 1.5 A from the
    next one, and a bead is 3.4 A wide: left to the ordinary nonbonded terms they
    would fly apart on the first step, so every pair inside a cofactor is excluded
    and the shape is held by bands instead -- ``neighbours`` of them per bead, as a
    molecule's own bonds run.

    ``tethers`` bands each bead to that many of the nearest backbone beads, which
    is what keeps the body where the structure put it: tethering every bead of it
    holds far better than banding a few of them hard, the bands being soft.  Any
    coordination the structure shows is banded too, whatever the counting says.
    """
    protein = [k for k, n in enumerate(mol.nodes) if not n.get("cofactor")]
    pos = np.asarray(mol.positions, float) * 10  # the helpers measure in angstroms
    for _, beads in inert:
        for a in range(len(beads)):
            for b in range(a + 1, len(beads)):
                mol.add("exclusions", Interaction((beads[a], beads[b]), [], {}))
        for a, b, d in shape_bands(pos[beads], neighbours):
            mol.add("bonds", Interaction((beads[a], beads[b]), [1, round(d / 10, 4), fc],
                                         {"comment": "cofactor shape"}))  # fmt: skip
        spine = np.array([mol.nodes[k].get("atomname") == "BB" for k in protein])
        bands = {(beads[i], protein[j]): d
                 for i, j, d in anchor_bands(pos[beads], pos[protein], spine, reach,
                                             None, tethers)}  # fmt: skip
        for i in beads:  # and the coordination, which is a bond the structure shows
            for j in sorted(mol.adj[i]):
                if j in protein:
                    bands[(i, j)] = float(np.linalg.norm(pos[i] - pos[j]))
        for (i, j), d in sorted(bands.items()):
            mol.add("bonds", Interaction((i, j), [1, round(d / 10, 4), fc],
                                         {"comment": "cofactor held"}))  # fmt: skip


def _coordination_bonds(residues, cofactors) -> set:
    """``(residue, atom, residue, atom)`` wherever a cofactor's heavy atom is close
    enough to another residue's to be what holds it.

    A metal coordinates at 1.8 to 2.3 A and a passing contact comes no nearer than
    3, so :data:`COORDINATION` separates them without knowing any chemistry.  The
    structure's own bonds are used where it has them; these are for the files that
    leave a metal unbonded, which is most of them.
    """
    which = {id(r) for r in cofactors}
    heavy, out = [], set()
    for r, res in enumerate(residues):
        for i, e in enumerate(res.elements):
            if e != "H":
                heavy.append((r, i, res.xyz[i], id(res) in which))
    for a, (ra, ia, xa, is_a) in enumerate(heavy):
        for rb, ib, xb, is_b in heavy[a + 1 :]:
            if ra == rb or not (is_a or is_b):
                continue
            if float(np.linalg.norm(np.asarray(xa) - np.asarray(xb))) <= COORDINATION / 10:
                out.add((ra, ia, rb, ib) if ra < rb else (rb, ib, ra, ia))
    return out


def _polymer_resnames(system) -> set:
    """The residue names the structure's own reading calls a polymer: anything with
    a backbone, whatever the force field happens to have a block for."""
    ids = system.select("protein or nucleic").ids
    if not len(ids):
        return set()
    res = np.asarray(system.residues["name"])
    return {str(res[r]).strip().upper()
            for r in np.unique(np.asarray(system.atoms["residue"])[ids]).tolist()}  # fmt: skip


def _residue_keys(system, selection: str) -> set:
    """(chain, resid, insertion) of every residue ``selection`` touches, which
    is how a bead says where it came from."""
    ids = system.select(selection).ids
    res = system.atoms["residue"][ids]
    chains, resids = system.chains["name"], system.residues["resid"]
    of_chain, insertion = system.residues["chain"], system.residues["insertion"]
    return {(str(chains[of_chain[r]]), int(resids[r]), str(insertion[r]))
            for r in sorted(set(res.tolist()))}  # fmt: skip


def _rubber_bands(mol, fc, lower, upper, decay, power, min_fc, res_min_dist, bond_type,
                  networked=None):  # fmt: skip
    """vermouth's ApplyRubberBand on the BB beads of one molecule (lengths in nm).

    ``networked``, when given, is the residues that may carry bands, so the
    network can hold one molecule and leave another free.
    """
    sel = [k for k, n in enumerate(mol.nodes) if n["atomname"] == "BB"
           and (networked is None
                or (str(n.get("chain", "") or ""), int(n["input_resid"]),
                    str(n.get("insertion", "") or "")) in networked)]  # fmt: skip
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
