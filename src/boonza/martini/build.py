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
    ATOM_MASS,
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
#: beads.
#:
#: Its neutral acids and its neutral lysine are residues of its own (ASP0, GLU0,
#: LSN) rather than modifications, and no mapping reaches them from an all-atom
#: structure -- not in vermouth either, which is the set these come from.  So a
#: protonated aspartate stays charged under Martini 2, which is what martinize2
#: does with it, and naming it the Amber or CHARMM way asks for the same thing:
#: these point at the charged residue, whose mapping exists, not at the neutral
#: block, whose mapping does not.  The hydrogens then come back as a
#: modification the force field has no mapping for either (ASP-HD2), which is
#: warned about and skipped -- the same path a structure takes when it arrives
#: named ASP with the proton still on it, so the two spellings of one chemistry
#: do the same thing.  Martini 3 has the mappings and keeps such a residue
#: neutral; so does SIRAH, which has a residue of its own for it.
INSTEAD_OF = {
    "martini22": {
        "residue": {
            "HIS": "HSD",
            "HID": "HSD",
            "HIE": "HSE",
            "HIP": "HSP",
            "ASH": "ASP",
            "ASPP": "ASP",
            "GLH": "GLU",
            "GLUP": "GLU",
            "LYN": "LYS",
            "HYP": "PRO",
            "GLYM": "GLY",
            "CYSF": "CYS",
            "CYSG": "CYS",
            "CYSP": "CYS",
        },  # fmt: skip
    },
}


#: A neutral residue Martini 2.2 has no bead for, and the charged one whose
#: parameters stand in for it.  Its own neutral blocks (ASP0, GLU0, LSN) have no
#: mapping from an all-atom structure -- not in vermouth either -- so the charged
#: residue is built and the side chain's charge is then set to zero: the bead
#: keeps the type, the size and the bonded terms of the charged form, and carries
#: no charge, which is the part of the chemistry a neutral residue actually
#: differs in.  martinize2 leaves such a residue charged instead; this is a
#: deliberate difference, and the one that answers what the structure says.
NEUTRAL_RESIDUE = {"ASH": "ASP", "ASPP": "ASP", "GLH": "GLU", "GLUP": "GLU", "LYN": "LYS"}

#: Acids whose neutral form Martini 2.2 has no block for, so it builds the
#: charged one even from a structure that says otherwise.
#:
#: Whether the structure says otherwise is read from the hydrogens, not from a
#: name: the proton is spelled HD2, HD1 or 2HD depending on who wrote the file,
#: and a residue can be named ASP and still hold it.  Nor does it look for OD1
#: or OD2 -- in a charged aspartate no oxygen carries a hydrogen at all, and
#: neither does the backbone carbonyl, so a hydrogen on any oxygen of the
#: residue says the side chain holds its proton, whatever the file calls either
#: of them.
#:
#: Lysine needs none of this: Martini 2.2 maps a neutral one to C3-P1 and a
#: charged one to C3-Qd from the hydrogens itself, so it is already right.
PROTONATABLE_ACIDS = ("ASP", "GLU")
#: How close a hydrogen is to the atom it sits on, in nm, which is what
#: _Residue.xyz is written in -- a covalent O-H is near 0.10 nm.
_BONDED_H = 0.13


def _holds_its_proton(res) -> bool:
    """Whether an acid arrived with a proton its charged form would not have.

    Read from the elements and where they are, so a file that names its atoms
    unusually is read the same as one that does not.  ``False`` where there are
    no hydrogens to count, the name being the only answer left.
    """
    if res.resname.upper() not in PROTONATABLE_ACIDS:
        return False
    elements = list(res.elements)
    oxygens = [i for i, e in enumerate(elements) if e == "O"]
    hydrogens = [i for i, e in enumerate(elements) if e == "H"]
    if not oxygens or not hydrogens:
        return False
    xyz = np.asarray(res.xyz, float)
    near = np.sqrt(((xyz[hydrogens][:, None] - xyz[oxygens][None]) ** 2).sum(-1))
    return bool((near <= _BONDED_H).any())


def _instead_of(ff, what: str, name: str, fallback=None):
    """What ``ff`` calls ``name``, where it calls it something else."""
    out = INSTEAD_OF.get(getattr(ff, "name", ""), {}).get(what, {}).get(name, fallback)
    # a residue read as another charge state is a change to the chemistry, not a
    # spelling, so it is said out loud -- once per name, however many there are
    if what == "residue" and NEUTRAL_RESIDUE.get(name) == out != name:
        said = getattr(ff, "_said_protonation", None)
        if said is None:
            said = ff._said_protonation = set()
        if name not in said:
            said.add(name)
            warnings.warn(
                f"{name} built from {out} with its side chain's charge set to zero: "
                f"{getattr(ff, 'name', 'this force field')} has no bead for a neutral "
                f"{out.title()}, so it takes the charged residue's parameters without "
                f"its charge",
                stacklevel=2,
            )
    return out


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

    def itp(self, k: int = 0, serial_resids: bool = False) -> str:
        return _write_itp(self.molecules[k], self.names[k], serial_resids)

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
        return self._marked_bonds(lambda meta: meta.get("group") == "Rubber band")

    def restraint_bonds(self) -> list[tuple[int, int]]:
        """Every band, as pairs of bead indices in the built system.

        The rubber bands; what holds a cofactor, the bands inside one that keep
        its shape and the ones tying it to whatever coordinates it; and Martini
        2.2's own "short" and "long elastic bonds for extended regions", which
        hold a beta sheet at 0.64 and 0.97 nm.  Those last are written as real
        bonds rather than as the rubber bands are, so a viewer draws every one
        of them: a protease comes out under a lattice of 127 of them.

        All of them are bonds of the topology, so a viewer draws them -- a
        hundred and fifty from one benzamidine to the protein around it, which
        is a star rather than a ligand -- and :func:`for_viewing` leaves them
        all out.  The force field names them elastic; what holds a thing rather
        than saying what it is does not belong in a picture of it.

        Whatever holds a thing rather than saying what it is belongs here.  A
        disulfide does not: it is chemistry, and stays.
        """
        return self._marked_bonds(
            lambda meta: (
                "elastic" in str(meta.get("group", "")).lower()
                or meta.get("group") == "Rubber band"
                or str(meta.get("comment", "")).startswith("cofactor")
            )
        )

    def _marked_bonds(self, wanted) -> list[tuple[int, int]]:
        out, offset = [], 0
        for mol, count in zip(self.molecules, self.molecule_copies, strict=True):
            for _ in range(count):
                for bond in mol.interactions["bonds"]:
                    if wanted(bond.meta):
                        i, j = (int(a) + offset for a in bond.atoms)
                        out.append((min(i, j), max(i, j)))
                offset += len(mol.nodes)
        return out

    def for_viewing(self, system=None, martini_itp=None, backbone_as_ca: bool = False):
        """The system without its bands: what to open in a viewer.

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
        bands = self.restraint_bonds()
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

    def _write_topology(self, directory, martini_itp=None, serial_resids: bool = False) -> Path:
        """The topology, its molecules and the .gro, without building a system:
        what :meth:`system` needs to read its parameters back."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for k, name in enumerate(self.names):
            (d / f"{name}.itp").write_text(self.itp(k, serial_resids))
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
            # a topology numbers its residues and has no column for an
            # insertion code, so 170 GLN and 170A GLN read back as one residue
            # with two backbone beads in it; the numbering here is one per
            # residue, and _name_chains puts the structure's own back
            top = self._write_topology(tmp, martini_itp.name, serial_resids=True)
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
        """Give the beads' residues the chain, number and insertion code they
        came from.

        A molecule's beads know their chain, and the lipids and the solvent
        have none of their own; a system built from a composition alone knows
        no chains at all, and keeps the one chain it was read with.

        A GROMACS topology has no column for an insertion code and renumbers
        each molecule's residues after the one before it, so a structure
        numbered 60, 60A, 60B comes back as three residues all called 60 --
        one residue as far as any viewer is concerned, with three backbone
        beads in it and a trace that knots itself drawing them.  The beads
        carry the numbering they were mapped from, so it is put back here.
        """
        of_bead = []
        for mol, count in zip(self.molecules, self.molecule_copies, strict=True):
            # every copy of a molecule would claim the one residue number its
            # nodes carry, so copies keep the numbering the topology gave them
            mine = [
                (
                    str(n.get("chain", "") or ""),
                    None if count > 1 else n.get("input_resid"),
                    str(n.get("insertion", "") or "").strip(),
                )
                for n in mol.nodes
            ]
            of_bead += mine * count
        of_bead += [("", None, "")] * (s.natoms - len(of_bead))  # lipids, water, ions
        residue = np.asarray(s.atoms["residue"])
        first = np.zeros(s.nresidues, np.int64)
        first[residue[::-1]] = np.arange(s.natoms)[::-1]  # the first bead of each residue
        want = [of_bead[int(a)] for a in first]
        for r, (_name, resid, insertion) in enumerate(want):
            if resid is not None:
                s.residue(r).resid = int(resid)
                s.residue(r).insertion = insertion
        if not any(name for name, _resid, _insertion in want):
            return
        chains = {str(s.chains["name"][c]): s.chain(c) for c in range(s.nchains)}
        for name in dict.fromkeys(name for name, _resid, _insertion in want):
            if name not in chains:
                chains[name] = s.add_chain(name=name)
        for r, (name, _resid, _insertion) in enumerate(want):
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


#: Single-atom cofactors worth banding to what coordinates them.  A zinc holds a
#: fold together and nothing else in the model says so; the ions that merely sit
#: in a site are left free, where a band would pin them to the crystal's guess.
#: SIRAH draws the same line, in :data:`boonza.sirah.build.COORDINATED_IONS`.
COORDINATED_IONS = frozenset({"ZN", "FE", "FE2", "FE3", "MN", "CU", "CU1", "NI", "CO"})

#: What an inert cofactor bead is, per force field: the smallest apolar bead each
#: version has, which is an alanine's side chain in Martini 3 and the apolar bead
#: Martini 2 builds one from.  Uncharged and unremarkable on purpose -- see
#: :func:`martinize`'s ``cofactors``.
COFACTOR_BEAD = {"martini3001": "TC3", "martini22": "C1"}


@cache
def ion_beads(version: int) -> dict:
    """``{residue: (bead, charge, mass)}`` for the ions that version of Martini has.

    A cofactor of one atom that Martini has an ion for deserves that ion rather
    than a stand-in: a calcium is ``SD`` and +2 under Martini 3 and ``Qd`` and +2
    under Martini 2, and those bead types are in the parameter file the topology
    already includes, so borrowing the type and the charge needs nothing else.
    Martini has no magnesium either, and a magnesium is the calcium's bead and
    charge (with its own mass) rather than an inert bead: a neutral, apolar bead
    left free drifts into the hydrophobic pockets a run is meant to find.
    Martini has no zinc; that falls back to the inert bead.
    """
    from . import parameters

    # Martini 3 keeps its ions in a file of their own, where Martini 2's come
    # with the parameters; IONS_FOR names only the one the topology must include
    files = {3: ("martini_v3.0.0_ions_v1.itp",), 2: ("martini_v2.0_ions.itp",)}
    out: dict = {}
    for name in files.get(version, ()):
        where, section = None, None
        for line in parameters(name)[0].read_text().splitlines():
            s = line.split(";")[0].strip()
            if not s:
                continue
            if s.startswith("["):
                section = s.strip("[] ").strip()
                continue
            cols = s.split()
            if section == "moleculetype":
                where = cols[0].upper()
            elif section == "atoms" and where and len(cols) >= 7:
                mass = float(cols[7]) if len(cols) > 7 else None
                out.setdefault(where, (cols[1], float(cols[6]), mass))
                where = None  # one atom is an ion; more than one is not
    if "CA" in out and "MG" not in out:
        out["MG"] = (out["CA"][0], out["CA"][1], 24.305)
    return out


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
              cofactor_reach: float = COFACTOR_REACH,
              cofactor_anchors: bool = True,
              cofactor_side_chains: bool = False) -> Martinized:  # fmt: skip
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
    neutral, and His is typed by which ring nitrogens carry one.  A capping
    group at a chain end (ACE, NME) is left out, and the end it capped made
    neutral, as the cap had it.
    """
    ff = force_field(forcefield)
    atoms, capped, notes = _without_caps(system, atoms)
    for note in notes:
        warnings.warn(note, stacklevel=2)
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
    # would throw away a side chain in the middle of a protein.  What is bonded
    # into the chain is refused rather than approximated; a free SAH or ATP, read
    # as protein or nucleic by its atom names alone, is not
    linked = _chain_keys(system) if cofactors else set()
    inchain = sorted({r.resname for r in nameless
                      if (r.chain, r.resid, r.insertion) in linked})  # fmt: skip
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
    if cofactors and cofactor_anchors:
        inert_of = {r for r, res in enumerate(residues) if named(res.resname) not in ff.blocks}
        # a host is a molecule with a backbone in it: the bands tie a cofactor to
        # backbone beads, and a lipid or an ion has none to tie it to
        spine_of = {r for r, res in enumerate(residues)
                    if "BB" in getattr(ff.blocks.get(named(res.resname)), "atoms", {})}  # fmt: skip
        groups = _join_loose_cofactors(groups, residues, inert_of, spine_of, cofactor_reach)
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
    neutral = (set(range(len(residues))) if neutral_termini else
               {r for r, res in enumerate(residues) if (res.chain, res.resid, res.insertion)
                in capped})  # fmt: skip
    molecules, names, positions, missing = [], [], [], []
    unmodified: set[str] = set()
    for m, members in enumerate(groups):
        cg_ss = convert_dssp_to_martini("".join(ss[r] for r in members)) if ss else ""
        mol = _build_molecule(ff, residues, members, cg_ss, bonds, nter, cter, neutral,
                              missing, unmodified, bead, cofactor_fc,
                              cofactor_neighbours, cofactor_tethers,
                              cofactor_reach, cofactor_anchors,
                              cofactor_side_chains)  # fmt: skip
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


def _join_loose_cofactors(groups, residues, inert, spine, reach: float) -> list[list[int]]:
    """Fold a molecule of nothing but cofactor into the nearest one with a
    protein in it, so that the bands holding it have something to hold on to.

    :func:`_hold_cofactors` tethers a cofactor's beads to the backbone beads of
    its own molecule, and a cofactor that bonds to nothing is a molecule by
    itself, whose protein beads number none: the reach is then searched against
    an empty set and no band is written, however wide it is or however near the
    protein actually is.  A benzamidine 4.1 A from the nearest backbone bead,
    with 36 of them inside 12 A, came out tethered by nothing and walked 8.8 A
    out of its pocket in half a nanosecond.

    What used to decide this was :func:`_coordination_bonds`, which bonds at 2.6
    A -- a test for a metal's coordination, and no test at all for whether
    something is bound.  In one structure a sulfate made it by 0.04 A and a
    benzamidine missed by 0.8, and nothing about either was a decision.

    A cofactor is folded in when it is close enough to be held, which is the
    same ``reach`` the bands use, and when it is something :func:`_hold_cofactors`
    would hold: more than one bead, or an ion that coordinates.  A lone calcium
    in the solvent is left the free molecule it is.  The residues keep the order
    the structure gave them, so the beads come out in it too.

    ``spine`` is the residues with a backbone bead, which is what a host has to
    have: the bands tie a cofactor to backbone beads, and a bilayer's lipids or
    an ion are no more a place to tie one than open water is.
    """
    hosts = [g for g in groups if any(r in spine for r in g)]
    if not hosts:
        return groups
    held = {id(g): list(g) for g in hosts}
    out, free = [], []
    for g in groups:
        if any(r in spine for r in g):
            continue
        if any(r not in inert for r in g):
            free.append(g)  # no cofactor of ours: a lipid, a nucleic acid
            continue
        beads = sum(1 for r in g for e in residues[r].elements if str(e).upper() != "H")
        if beads <= 1 and not any(residues[r].resname.upper() in COORDINATED_IONS for r in g):
            free.append(g)  # a free ion, which banding would only pin
            continue
        mine = np.vstack([np.asarray(residues[r].xyz, float) for r in g])
        near, closest = None, np.inf
        for h in hosts:
            theirs = np.vstack([np.asarray(residues[r].xyz, float) for r in h
                                if r in spine])  # fmt: skip
            d = float(np.linalg.norm(mine[:, None] - theirs[None], axis=2).min()) * 10
            if d < closest:
                near, closest = h, d
        if near is None or closest > reach:
            free.append(g)  # too far from any protein to be bound to one
            continue
        held[id(near)] += g
    for g in groups:
        if any(r in spine for r in g):
            out.append(sorted(held[id(g)]))
        elif g in free:
            out.append(g)
    return out


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
                    reach: float = COFACTOR_REACH, anchors: bool = True,
                    side_chains: bool = False) -> CGMolecule:  # fmt: skip
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
            # one atom that Martini has an ion for gets that ion, charge and all;
            # anything else gets the inert bead, which claims nothing
            ion = ion_beads(2 if ff.name == "martini22" else 3).get(res.resname.upper())
            ion = ion if len(heavy) == 1 else None
            index = {}
            for i in heavy:
                node = dict(atype=ion[0] if ion else cofactors, resname=res.resname[:5],
                            atomname=res.names[i][:5], charge=ion[1] if ion else 0.0,
                            mass=float(ion[2]) if ion and ion[2] else
                            float(ATOM_MASS.get(res.elements[i],
                                                _MASS.get(res.elements[i], 30))),
                            resid=serial,
                            chain=res.chain, input_resid=res.resid,
                            insertion=res.insertion, modifications=[],
                            cofactor=True, ion=bool(ion))  # fmt: skip
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
            mods.append("NH2-ter" if r in neutral else "N-ter")
        if r in cter:
            mods.append("COOH-ter" if r in neutral else "C-ter")
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
        if res.resname in NEUTRAL_RESIDUE or (
            getattr(ff, "name", "") == "martini22" and _holds_its_proton(res)
        ):
            # the structure says this side chain holds its proton, and the model
            # has no bead for one: the charge is what it would differ in, so the
            # charge is what goes.  Only the side chain -- a terminus puts its
            # own charge on BB and has nothing to do with this
            for name, k in index.items():
                if name.startswith("SC"):
                    mol.nodes[k]["charge"] = 0.0
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
    # a cofactor of one atom has no shape to hold, so only its coordination is in
    # question, and that is worth banding for one of them: a zinc is what holds a
    # zinc finger's loops together and the model says so nowhere else.  A calcium,
    # a magnesium, a sodium is left the free ion it is -- banding one pins it
    # where the crystal happened to catch it
    held = [(r, beads) for r, beads in inert
            if len(beads) > 1 or residues[r].resname.upper() in COORDINATED_IONS]  # fmt: skip
    if held:
        _hold_cofactors(mol, residues, held, cofactor_fc, neighbours, tethers, reach,
                        anchors, side_chains)  # fmt: skip
    return mol


def _hold_cofactors(mol: CGMolecule, residues, inert, fc: float,
                    neighbours=COFACTOR_NEIGHBOURS, tethers=COFACTOR_TETHERS,
                    reach: float = COFACTOR_REACH, anchors: bool = True,
                    side_chains: bool = False) -> None:  # fmt: skip
    """Hold each inert cofactor in shape and in place, and let nothing inside it
    feel anything.

    A bead per heavy atom sits where its atom did, which is about 1.5 A from the
    next one, and a bead is 3.4 A wide: left to the ordinary nonbonded terms they
    would fly apart on the first step, so every pair inside a cofactor is excluded
    and the shape is held by bands instead -- ``neighbours`` of them per bead, as a
    molecule's own bonds run.  That shape is held whatever else is asked for.

    ``tethers`` bands each bead to that many of the nearest backbone beads, which
    is what keeps the body where the structure put it: tethering every bead of it
    holds far better than banding a few of them hard, the bands being soft.  Any
    coordination the structure shows is banded too, whatever the counting says.

    Without ``anchors`` no tether is added and the cofactor is free to move: its
    shape is still held, and so is any coordination the structure shows, which
    is a bond rather than a guess at one.

    With ``side_chains`` the tethers reach side-chain beads as well, which is
    what a ligand is actually in contact with.  It holds the cofactor better and
    holds the pocket with it -- measured over three replicates of a benzamidine
    in a thrombin, 1.20 A of drift against 1.96, and the pocket's side chains 40
    per cent less mobile -- so it is off unless asked for: a positive control
    wants the ligand still, and a measurement of what the pocket does does not
    want its walls banded to the thing in it.
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
        # one bead cannot deform, so the coordination alone fixes it; a body needs
        # the tethers, having a shape to hold as well as a place to stay
        spine = None if side_chains else np.array(
            [mol.nodes[k].get("atomname") == "BB" for k in protein])  # fmt: skip
        bands = (
            {}
            if len(beads) == 1 or not anchors
            else {
                (beads[i], protein[j]): d
                for i, j, d in anchor_bands(pos[beads], pos[protein], spine, reach, None, tethers)
            }
        )
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


def _chain_keys(system) -> set:
    """(chain, resid, insertion) of every residue bonded into a chain, as
    ``_residues`` names them."""
    from ..analyze import chain_residues

    res = system.residues
    chains = np.asarray(system.chains["name"])
    return {(str(chains[res["chain"][r]]).strip(), int(res["resid"][r]),
             str(res["insertion"][r]).strip())
            for r in chain_residues(system).tolist()}  # fmt: skip


def _without_caps(system, atoms: str) -> tuple[str, set, list[str]]:
    """``atoms`` without the capping groups at chain ends in it, the ends they
    capped as (chain, resid, insertion), and a line saying so for each.

    A cap is no residue of Martini's or SIRAH's, and the group it stands for is
    neutral: what it capped is a chain end that carries no terminal charge.
    """
    from ..analyze import end_caps

    caps = end_caps(system)
    if not caps:
        return atoms, set(), []
    chosen = set(system.select(atoms).ids.tolist())
    res, residue = system.residues, np.asarray(system.atoms["residue"])
    chains = np.asarray(system.chains["name"])

    def key(r):
        return (str(chains[res["chain"][r]]).strip(), int(res["resid"][r]),
                str(res["insertion"][r]).strip())  # fmt: skip

    def name(r):
        chain, resid, insertion = key(r)
        return f"{str(res['name'][r]).strip()} {chain}{resid}{insertion}"

    drop, ends, notes = [], set(), []
    for cap, (host, end) in sorted(caps.items()):
        if not chosen & set(np.flatnonzero(residue == host).tolist()):
            continue  # whether or not the cap was asked for, the end it capped is neutral
        mine = [a for a in np.flatnonzero(residue == cap).tolist() if a in chosen]
        drop += mine
        ends.add(key(host))
        notes.append(f"{name(cap)} left out: {name(host)} is a neutral {end}-terminus")
    if drop:
        atoms = f"({atoms}) and not index {' '.join(map(str, drop))}"
    return atoms, ends, notes


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


def _write_itp(mol: CGMolecule, name: str, serial_resids: bool = False) -> str:
    """The molecule as a GROMACS .itp.

    Its residues are numbered as the structure numbered them, which is what
    martinize2's .gro and boonza's own say too.  With ``serial_resids`` they
    are numbered one apiece instead: a topology has no insertion code, so two
    residues the structure told apart as 170 and 170A are one residue to
    anything reading it back -- which :meth:`Martinized.system` does.
    """
    out = ["[ moleculetype ]", f"{name} 1", "", "[ atoms ]"]
    for k, n in enumerate(mol.nodes, start=1):
        mass = f" {_num(n['mass'])}" if "mass" in n else ""
        resid = n["resid"] if serial_resids else n["input_resid"]
        out.append(f"{k:5d} {n['atype']:<6s} {resid:5d} {n['resname']:<5s} "
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
