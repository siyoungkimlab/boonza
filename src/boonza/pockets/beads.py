"""What fpocket reads of a bead: where it is, and whether it is polar.

fpocket knows two things about an atom, its position and its element.  The
element decides whether the atom is apolar (Pauling electronegativity below
2.8, i.e. C or S), and hydrogens are dropped before the Voronoi tessellation.
A coarse-grained bead has no element, so the PDB written here gives it one that
carries the bead's polarity instead: ``C`` for an apolar bead, ``O`` for a polar
one.

* Martini beads go by the class letter of their type: C and X apolar; P, Q and
  D polar; the intermediate N class is a choice (``n_polar``), which the tuned
  presets make per model.
* SIRAH's backbone goes by name (GN and GO polar, GC apolar); a SIRAH side-chain
  bead is polar when it carries a partial charge (|q| >= 0.1).  The name alone
  misleads there: lysine's BCG and BCE, arginine's BCZ, the carboxylate carbons
  of Asp and Glu and tyrosine's hydroxyl beads are named after carbons but are
  charged.

Residue names are written as the standard three-letter amino acids (SIRAH's
``sA``, ``sHe``; Martini's ``HSD``, ``CYX``), which fpocket reads for its
polarity score.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import numpy as np

#: what a structure can be searched as: all-atom, or one of the bead models
MODELS = ("aa", "martini2", "martini3", "sirah")
#: the boonza force field of each Martini model
FORCEFIELDS = {"martini2": "martini22", "martini3": "martini3001"}

APOLAR, POLAR = "C", "O"
#: a SIRAH side-chain bead with at least this much partial charge is polar
SIRAH_CHARGED = 0.1

SIRAH_RESNAMES = {
    "sA": "ALA", "sR": "ARG", "sN": "ASN", "sD": "ASP", "sC": "CYS", "sX": "CYS",
    "sQ": "GLN", "sE": "GLU", "sG": "GLY", "sHe": "HIS", "sHd": "HIS", "sHp": "HIS",
    "sI": "ILE", "sL": "LEU", "sK": "LYS", "sM": "MET", "sF": "PHE", "sP": "PRO",
    "sS": "SER", "sT": "THR", "sW": "TRP", "sY": "TYR", "sV": "VAL",
}  # fmt: skip
VARIANT_RESNAMES = {
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
}  # fmt: skip
#: residue names of protein beads, for CG systems the ``protein`` keyword misses
AMINO_ACIDS = frozenset(SIRAH_RESNAMES) | frozenset(VARIANT_RESNAMES) | {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
    "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}  # fmt: skip

__all__ = ["bead_elements", "bead_types", "guess_model", "martini_polar", "sirah_charges",
           "sirah_polar", "standard_resname", "write_fpocket_pdb"]  # fmt: skip


def martini_polar(bead_type: str, n_polar: bool = False) -> bool:
    """Whether a Martini 2 or 3 bead type is polar.

    The size prefix (S, T), Martini 2's amino-acid prefix (A, in the AC1 and AC2
    side chains of Leu, Ile and Val) and the label suffixes (d, a, h, e, r, q, p,
    n) are ignored: the class letter decides.  A type that is none of these is
    taken as polar.
    """
    m = re.match(r"^(?:[ST]|A(?=C))?([PNCQXD])", str(bead_type))
    if m is None:
        return True
    cls = m.group(1)
    if cls == "N":
        return n_polar
    return cls not in "CX"


def sirah_polar(bead_name: str, charge: float) -> bool:
    """Whether a SIRAH bead is polar: the backbone by name, a side chain by its charge."""
    name = str(bead_name).strip().upper()
    if name in ("GN", "GO"):
        return True
    if name == "GC":
        return False
    return abs(charge) >= SIRAH_CHARGED - 1e-6


def sirah_charges(names, resnames) -> list[float]:
    """SIRAH's partial charge of each bead, from its residue library."""
    from ..sirah.build import read_residues

    library = read_residues()[0]
    charges = []
    for name, resname in zip(names, resnames, strict=True):
        entry = library.get(str(resname).strip())
        by_name = {n: q for n, _kind, q in entry.atoms} if entry else {}
        charges.append(float(by_name.get(str(name).strip(), 0.0)))
    return charges


def standard_resname(resname: str) -> str:
    """The standard three-letter amino acid a SIRAH or protonation-variant name stands for."""
    resname = str(resname).strip()
    if resname in SIRAH_RESNAMES:
        return SIRAH_RESNAMES[resname]
    return VARIANT_RESNAMES.get(resname.upper(), resname.upper()[:3])


def guess_model(names, resnames) -> str:
    """``sirah``, ``martini`` or ``aa``, from bead names and residue names."""
    names = {str(n).strip() for n in names}
    if {"GN", "GC", "GO"} & names and any(str(r).startswith("s") for r in resnames):
        return "sirah"
    if "BB" in names:
        return "martini"
    return "aa"


@cache
def _block_types(model: str) -> dict[str, dict[str, str]]:
    from ..martini.build import force_field

    ff = force_field(FORCEFIELDS[model])
    return {res: {name: str(a.get("atype", "")) for name, a in block.atoms.items()}
            for res, block in ff.blocks.items()}  # fmt: skip


def bead_types(names, resnames, model: str) -> list[str]:
    """The Martini type of each bead, from its residue's block in the force field.

    For a structure read without its topology (a bare .gro or .pdb), which has
    bead names and no types.  A backbone bead's type follows the secondary
    structure in Martini 2, and the block holds the coil's (P5, or P4 for Ala
    and Pro); helix and strand types are N-class there, which the Martini 2
    preset counts as polar too, so the polarity fpocket reads is the same.  A
    bead the force field has no block for comes back as an empty string.
    """
    blocks = _block_types(model)
    out = []
    for name, resname in zip(names, resnames, strict=True):
        resname, name = str(resname).strip(), str(name).strip()
        for candidate in (resname, standard_resname(resname), "HSD" if
                          standard_resname(resname) == "HIS" else ""):  # fmt: skip
            if name in blocks.get(candidate, {}):
                out.append(blocks[candidate][name])
                break
        else:
            out.append("")
    return out


def _residue_columns(system, ids):
    res = np.asarray(system.atoms["residue"])[ids]
    resnames = np.asarray(system.residues["name"])[res]
    resids = np.asarray(system.residues["resid"])[res]
    ins = np.asarray(system.residues["insertion"])[res]
    chain_of = np.asarray(system.residues["chain"])[res]
    chains = np.asarray(system.chains["name"])[chain_of]
    return resnames, resids, ins, chains


def bead_elements(system, ids=None, model: str | None = None, n_polar: bool = False) -> list:
    """The element to write for each of ``ids``: the real one for all-atom, C or O for beads.

    ``model`` is ``aa``, ``sirah`` or a Martini model (guessed from the names when
    not given).  A Martini system read without its topology takes its bead types
    from the force field (:func:`bead_types`).
    """
    ids = np.arange(system.natoms) if ids is None else np.asarray(ids)
    names = np.asarray(system.atoms["name"])[ids]
    resnames = _residue_columns(system, ids)[0]
    model = model or guess_model(names, resnames)
    if model == "aa":
        from ..elements import symbol

        return [symbol(int(a)) for a in np.asarray(system.atoms["anum"])[ids]]
    if model == "sirah":
        charges = np.asarray(system.atoms["charge"])[ids]
        if not np.any(charges):  # loaded without its topology: take the library's
            charges = sirah_charges(names, resnames)
        return [POLAR if sirah_polar(n, q) else APOLAR for n, q in zip(names, charges, strict=True)]
    types = [str(t) for t in np.asarray(system.atoms["type"])[ids]]
    if not all(types):
        known = bead_types(names, resnames, model if model in FORCEFIELDS else "martini3")
        types = [t or k for t, k in zip(types, known, strict=True)]
    return [POLAR if martini_polar(t, n_polar) else APOLAR for t in types]


def write_pdb(path, names, resnames, chains, resids, insertions, positions, elements) -> None:
    """A plain PDB of ATOM records with the given elements (columns 77-78)."""
    lines = []
    for k, (name, resname, chain, resid, ins, xyz, el) in enumerate(
        zip(names, resnames, chains, resids, insertions, positions, elements, strict=True)
    ):
        name = str(name)[:4]
        name = f" {name:<3s}" if len(name) < 4 else name
        lines.append(
            f"ATOM  {(k + 1) % 100000:5d} {name:4s} {str(resname)[:3]:>3s} "
            f"{(str(chain) or 'A')[:1]:1s}{int(resid) % 10000:4d}{(str(ins) or ' ')[:1]:1s}   "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {el:>2s}"
        )
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def write_fpocket_pdb(system, path, ids=None, model: str | None = None,
                      n_polar: bool = False) -> None:  # fmt: skip
    """Write ``system`` (or the atoms ``ids``) as a PDB for fpocket or mdpocket."""
    ids = np.arange(system.natoms) if ids is None else np.asarray(ids)
    names = np.asarray(system.atoms["name"])[ids]
    resnames, resids, ins, chains = _residue_columns(system, ids)
    elements = bead_elements(system, ids, model, n_polar)
    write_pdb(path, names, [standard_resname(r) for r in resnames], chains, resids, ins,
              np.asarray(system.positions)[ids], elements)  # fmt: skip
