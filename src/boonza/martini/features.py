"""Pharmacophore features of Martini beads, by what the bead stands for.

RDKit types a ligand's atoms, and a bead has no element or valence to read.
It does not need one: a bead stands for a known piece of a known residue, so
its features are known too.  Lys's charged bead is a cation and a donor, Phe's
ring is aromatic, Ser's hydroxyl bead both donates and accepts.

Where a group both donates and accepts, the bead carries both families --
except in histidine, where Martini gives the two ring nitrogens their own
beads, so the donor and the acceptor are told apart.
"""

from __future__ import annotations

import numpy as np

from .probes import LETTERS

#: bead -> families, per residue.  A hydroxyl (Ser, Thr, Tyr) and an amide
#: (Asn, Gln) both donate and accept; the charged beads also donate or accept,
#: as their groups do.
SIDE_CHAINS: dict[str, dict[str, tuple[str, ...]]] = {
    "ARG": {"SC1": ("Hydrophobe",), "SC2": ("PosIonizable", "Donor")},
    "LYS": {"SC1": ("Hydrophobe",), "SC2": ("PosIonizable", "Donor")},
    "GLU": {"SC1": ("NegIonizable", "Acceptor")},
    "ASP": {"SC1": ("NegIonizable", "Acceptor")},
    "GLN": {"SC1": ("Donor", "Acceptor")},
    "ASN": {"SC1": ("Donor", "Acceptor")},
    "SER": {"SC1": ("Donor", "Acceptor")},
    "THR": {"SC1": ("Donor", "Acceptor")},
    "CYS": {"SC1": ("Hydrophobe",)},
    "MET": {"SC1": ("Hydrophobe",)},
    "ALA": {"SC1": ("Hydrophobe",)},
    "VAL": {"SC1": ("Hydrophobe",)},
    "ILE": {"SC1": ("Hydrophobe",)},
    "LEU": {"SC1": ("Hydrophobe",)},
    "PRO": {"SC1": ("Hydrophobe",)},
    # the rings are below; these are the beads that are more than ring
    "HIS": {"SC2": ("Donor",), "SC3": ("Acceptor",)},  # ND1-H and NE2, each its own bead
    "TRP": {"SC2": ("Donor",)},  # the indole NH
    "TYR": {"SC4": ("Donor", "Acceptor")},  # the phenol OH
    "PHE": {},
    "GLY": {},
}
#: aromatic rings: the beads whose centre the feature sits at, and what it is.
#: Phe's and Trp's rings are greasy as well as aromatic; His's and Tyr's carry
#: the polar groups above, so they count as aromatic only.
RINGS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "PHE": (("SC1", "SC2", "SC3"), ("Aromatic", "Hydrophobe")),
    "TRP": (("SC1", "SC2", "SC3", "SC4", "SC5"), ("Aromatic", "Hydrophobe")),
    "TYR": (("SC1", "SC2", "SC3"), ("Aromatic",)),
    "HIS": (("SC1", "SC2", "SC3"), ("Aromatic",)),
}
#: the backbone bead of a martinized residue: an amide, so a donor and an
#: acceptor.  Left out by default: every probe carries the same one, and it
#: would mark every place any probe went.
BACKBONE = ("Donor", "Acceptor")
MARTINI_NAMES = {"BB", *(f"SC{k}" for k in range(1, 10))}


def martini_beads(system, atoms) -> bool:
    """Whether ``atoms`` of ``system`` are Martini protein beads."""
    names = {str(n) for n in np.asarray(system.atoms["name"])[atoms]}
    return bool(names) and names <= MARTINI_NAMES


def _residues_of(system, atoms) -> list[tuple[str, list[int]]]:
    """(residue name, its atoms) for each residue the molecule is made of, in order."""
    res = np.asarray(system.atoms["residue"])[atoms]
    names = np.asarray(system.residues["name"])
    out = []
    for r in dict.fromkeys(res.tolist()):
        out.append((str(names[r]), [int(a) for a, rr in zip(atoms, res.tolist(),
                                                            strict=True) if rr == r]))  # fmt: skip
    return out


def _amino_acid(resname: str, place: int) -> str:
    """The residue a bead's own residue name stands for: a martinized protein
    keeps the three-letter name, a probe carries its two-letter code (``EK``)."""
    if resname in SIDE_CHAINS:
        return resname
    if len(resname) == 2 and place < 2:
        letter = resname[place]
        for three, one in LETTERS.items():
            if one == letter:
                return three
    return ""


def bead_features(system, atoms, families=(), backbone: bool = False):
    """``[(family, bead indices), ...]`` for one molecule of Martini beads.

    A feature sits at the centre of its beads, so an aromatic ring counts once
    at the middle of the ring, as RDKit's do.
    """
    found = []
    for place, (resname, beads) in enumerate(_residues_of(system, atoms)):
        residue = _amino_acid(resname, place)
        if not residue:
            continue
        by_name = {str(np.asarray(system.atoms["name"])[a]): a for a in beads}
        for bead, fams in SIDE_CHAINS.get(residue, {}).items():
            if bead in by_name:
                found += [(f, np.array([by_name[bead]])) for f in fams if f in families]
        ring, fams = RINGS.get(residue, ((), ()))
        present = [by_name[b] for b in ring if b in by_name]
        if len(present) >= 3:
            found += [(f, np.array(present)) for f in fams if f in families]
        if backbone and "BB" in by_name:
            found += [(f, np.array([by_name["BB"]])) for f in BACKBONE if f in families]
    return found
