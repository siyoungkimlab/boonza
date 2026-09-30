"""Pharmacophore features of SIRAH beads, from what the force field calls them.

SIRAH names every bead type after the group it stands for, and its own
comments say which group that is: ``C7Nk`` is "-[N]H3(+) hydrophilic amine
nitrogen (K)", ``P1O`` is "-[O]H hydroxylic oxygen", ``P2P`` is the "-O[H]
hydroxylic hydrogen".  So the features come from the bead type, with no
guessing and no atom typing.

The names are SIRAH 2.2's, as its GROMACS force field writes them, plus the
protonated histidine SIRAH 2.4 added (``sHp``: ``X3``, ``ZD``, ``ZE``).  A
topology built by tleap from the AMBER distribution carries that release's own
names throughout, which are not all in this table yet.

SIRAH is finer than Martini here.  A hydroxyl is two beads -- the oxygen and
the hydrogen -- so a serine gives an acceptor and a donor a bond apart, where
Martini has one bead that has to be called both.  The same holds for
tryptophan's indole (``A7N`` the nitrogen, ``A8P`` its hydrogen) and
tyrosine's phenol (``A4O`` and ``A3P``).
"""

from __future__ import annotations

import numpy as np

#: bead type -> families, from SIRAH's own description of each type.
BY_TYPE: dict[str, tuple[str, ...]] = {
    # backbone: the amide N donates, the carbonyl O accepts
    "GN": ("Donor",),
    "GO": ("Acceptor",),
    "GC": (),
    "Y2Ca": ("Hydrophobe",),  # alanine's alpha carbon
    # termini
    "GNn": ("Donor",),  # -[N]H2, neutral
    "GNz": ("Donor", "PosIonizable"),  # -[N]H3(+)
    "GOn": ("Acceptor",),  # -C[O]OH, neutral
    "GOz": ("Acceptor", "NegIonizable"),  # -[C]OO(-)
    # charged side chains
    "C5N": ("Donor", "PosIonizable"),  # Arg, N=C=[N](+)
    "C3Cr": ("PosIonizable",),  # Arg's central carbon
    "C7Nk": ("Donor", "PosIonizable"),  # Lys, -[N]H3(+)
    "C1Ck": ("Hydrophobe",),  # Lys, the carbon next to it
    "C6O": ("Acceptor", "NegIonizable"),  # Asp and Glu, O=C=[O](-)
    "C4Cd": ("NegIonizable",),  # Asp's carboxyl carbon
    "C4Ce": ("NegIonizable",),  # Glu's carboxyl carbon
    # polar side chains, where the hydrogen is its own bead
    "P1O": ("Acceptor",),  # -[O]H, Ser and Thr
    "P2P": ("Donor",),  # -O[H] and -S[H], the hydrogen itself
    "P4O": ("Acceptor",),  # [O]=C=N, Asn and Gln
    "P5N": ("Donor",),  # O=C=[N], Asn and Gln
    "P3Cn": ("Hydrophobe",),
    "P3Cq": ("Hydrophobe",),
    "P1S": ("Acceptor",),  # -[S]H, Cys
    "Y5Sz": ("Acceptor", "NegIonizable"),  # thiolate
    "Y5Sx": ("Hydrophobe",),  # cystine's oxidated sulfur
    # aromatics: the rings below, and the groups hanging off them
    "A1C": ("Aromatic", "Hydrophobe"),  # Phe
    "A1Cw": ("Aromatic", "Hydrophobe"),  # Trp
    "A2C": ("Aromatic", "Hydrophobe"),  # His, Phe, Tyr, Trp
    "A4O": ("Acceptor",),  # Tyr's phenol oxygen
    "A3P": ("Donor",),  # Tyr's phenol hydrogen
    "A7N": ("Acceptor",),  # Trp's indole nitrogen
    "A8P": ("Donor",),  # Trp's indole hydrogen
    # His: the same two types in both neutral tautomers, told apart by their
    # charge -- the ring nitrogen that carries the hydrogen is the positive one
    "A5D": ("BY_CHARGE",),
    "A5E": ("BY_CHARGE",),
    # a protonated His, which SIRAH 2.4 added as sHp with names of its own:
    # both ring nitrogens carry a hydrogen, so both donate, and it is a cation
    "ZD": ("Donor", "PosIonizable"),
    "ZE": ("Donor", "PosIonizable"),
    "X3": ("Aromatic", "PosIonizable"),
    # greasy side chains
    "Y1C": ("Hydrophobe",),  # Ile and Leu
    "Y2C": ("Hydrophobe",),
    "Y3C": ("Hydrophobe",),
    "Y4C": ("Hydrophobe",),
    "Y3Sm": ("Hydrophobe",),  # Met's sulfur
    "Y4Cv": ("Hydrophobe",),  # Val
    "Y6Cp": ("Hydrophobe",),  # Pro
    "C2Cr": ("Hydrophobe",),  # Arg's chain
    # ions, which stand for solvated ones
    "NaW": ("PosIonizable",),
    "KW": ("PosIonizable",),
    "MgX": ("PosIonizable",),
    "CaX": ("PosIonizable",),
    "ZnX": ("PosIonizable",),
    "ClW": ("NegIonizable",),
}
#: The bead types that are ring atoms: their centre is where the Aromatic
#: feature sits, once per residue, as RDKit places one for a ring.  How many
#: there are differs by residue -- three for Phe, four for Trp, two for His,
#: and one for Tyr, whose other two beads are the phenol's oxygen and hydrogen
#: -- so one is enough to place the ring.
RING_TYPES = ("A1C", "A1Cw", "A2C", "A5D", "A5E", "A7N", "X3", "ZD", "ZE")
#: The backbone, left out unless asked for: every residue has the same one.
BACKBONE_TYPES = ("GN", "GC", "GO", "GNn", "GNz", "GOn", "GOz", "Y2Ca")


def bead_types(system, atoms) -> list[str]:
    """The force field's type of each of ``atoms``, from the nonbonded table."""
    if "nonbonded" not in system.tables:
        return []
    table = system.table("nonbonded")
    of_atom = {int(a): k for k, (a,) in enumerate(table.atoms.tolist())}
    types = table.values("type")
    return [str(types[of_atom[int(a)]]) if int(a) in of_atom else "" for a in atoms]


def sirah_beads(system, atoms) -> bool:
    """Whether ``atoms`` hold SIRAH beads, by the types they carry.

    One is enough: a selection is mostly water in a solvated system, and
    SIRAH's water and ions have no features of their own.  Martini's beads are
    told apart before this, by their names.
    """
    return any(t in BY_TYPE for t in bead_types(system, atoms))


def bead_features(system, atoms, families=(), backbone: bool = False):
    """``[(family, bead indices), ...]`` for one molecule of SIRAH beads.

    A ring's Aromatic feature sits at the centre of its beads, once per
    residue; everything else sits on the bead that stands for it.
    """
    atoms = [int(a) for a in atoms]
    types = bead_types(system, atoms)
    residue = np.asarray(system.atoms["residue"])
    charge = np.asarray(system.atoms["charge"], float)
    found, rings = [], {}
    for atom, atype in zip(atoms, types, strict=True):
        if atype in RING_TYPES:
            rings.setdefault(int(residue[atom]), []).append(atom)
        if not backbone and atype in BACKBONE_TYPES:
            continue
        for family in BY_TYPE.get(atype, ()):
            if family == "BY_CHARGE":  # histidine's ring nitrogens
                family = "Donor" if charge[atom] > 0 else "Acceptor"
            if family in families and family != "Aromatic":
                found.append((family, np.array([atom])))
    if "Aromatic" in families:
        for beads in rings.values():  # one bead is enough: Tyr's ring is one
            found.append(("Aromatic", np.array(sorted(beads))))
    return found
