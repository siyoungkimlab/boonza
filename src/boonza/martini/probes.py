"""Dipeptide probes for mapping a protein's surface in Martini.

Martini has no general way to parameterize a small molecule, but every amino
acid is already parameterized, so a dipeptide is a probe that needs nothing
new: :func:`boonza.peptide` builds it and :func:`boonza.martinize` gives it
its topology.  A probe carries no secondary structure, no side-chain
corrections and no elastic network, so it is free to bend, and both its ends
are neutral, so only its side chains carry charge.
"""

from __future__ import annotations

import copy
from functools import cache

#: The residues probes are made of: the small ones (Ala, Gly, Val) are left
#: out, as are the ones another residue already stands for (Asp for Glu, Asn
#: for Gln), and the ones that would react (Cys, Sec).
PROBE_RESIDUES = ("ARG", "GLN", "GLU", "HIS", "ILE", "LEU", "LYS", "MET",
                  "PHE", "PRO", "SER", "THR", "TRP", "TYR")  # fmt: skip
LETTERS = {"ARG": "R", "GLN": "Q", "GLU": "E", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
           "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
           "TYR": "Y"}  # fmt: skip

#: What a probe's molecule is called in the topology.  Its beads keep the
#: sequence as their residue name, which is what selects them, but the
#: ``[ moleculetype ]`` cannot be ``W``: Martini's water is already called that,
#: and the second definition of a name silently loses to the first -- a
#: tryptophan probe would come out as one water bead.
PROBE_PREFIX = "probe_"


#: The amino acids of the single-residue library: every one but alanine and
#: glycine, which are too small to report anything -- in Martini both map onto
#: one bead, and in SIRAH alanine's side chain is part of its alpha carbon and
#: glycine has none at all.  A whole residue, not a side chain cut off its
#: backbone: the beads were parameterized with the backbone attached, and a
#: fragment of one is a molecule neither force field has ever seen.
SINGLE_RESIDUES = ("ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "ILE", "LEU", "LYS",
                   "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL")  # fmt: skip


def single_sequences(residues=SINGLE_RESIDUES) -> list[str]:
    """Each of ``residues`` as a one-letter sequence: the probes of a library of
    single amino acids.

    They carry a quarter to a half of a dipeptide's dipole -- a SIRAH dipeptide
    is 11 to 12 Debye where a single residue is 5 to 7 -- which is what a probe
    that reports chemistry rather than its own backbone needs.  Their charges
    are their side chains': +1 for arginine and lysine, -1 for aspartate and
    glutamate, zero for the rest.
    """
    letters = {**LETTERS, "ASN": "N", "ASP": "D", "CYS": "C", "VAL": "V"}
    return [letters.get(r, r) for r in residues]


def probe_sequences(residues=PROBE_RESIDUES) -> list[str]:
    """Every dipeptide of ``residues`` as a two-letter sequence, XY and YX once:
    the 14 default residues give 105 probes rather than 196.

    That folding is Martini's.  There both backbone beads are the same type with
    no charge, so KE is EK with its two residues exchanged -- the same molecule,
    bead for bead.  It does not hold in SIRAH, whose terminus entries make the
    first residue's GN and the last one's GO types of their own: EK carries the
    glutamate at the positive end of the backbone's dipole and KE at the negative
    end, which is 31.7 D against 44.0 D.  :func:`ordered_sequences` keeps both.
    """
    letters = [LETTERS[r] if r in LETTERS else r for r in residues]
    return [a + b for k, a in enumerate(letters) for b in letters[k:]]


def ordered_sequences(residues=PROBE_RESIDUES) -> list[str]:
    """Every dipeptide of ``residues`` both ways round: 196 from the 14 default.

    Which end of a probe carries which side chain is a difference SIRAH can tell
    -- its two chain ends are different bead types -- so a library that folds XY
    onto YX leaves 91 molecules of its own chemistry unswum.  In Martini the two
    are one molecule and 91 of these are duplicates.
    """
    letters = [LETTERS[r] if r in LETTERS else r for r in residues]
    return [a + b for a in letters for b in letters]


@cache
def _built(sequence: str, forcefield: str = "martini3001"):
    from .. import peptide
    from .build import martinize

    # the builder's peptides carry neutral side chains (a hydrogen on Glu's
    # carboxyl, two on Lys's amine); mapping the heavy atoms alone gives the
    # charged forms of pH 7, with His neutral
    m = martinize(peptide(sequence, conformation="extended"), "protein and not element H",
                  ss=False, scfix=False, neutral_termini=True,
                  forcefield=forcefield)  # fmt: skip
    if len(m.molecules) != 1:
        raise ValueError(f"{sequence} did not martinize into one molecule")
    for node in m.molecules[0].nodes:  # the probe's own residue name, not the protein's
        node["resname"] = sequence
    m.names = [PROBE_PREFIX + sequence]
    m.positions = m.positions - m.positions.mean(0)
    return m


def probe(sequence: str, forcefield: str = "martini3001"):
    """The martinized probe ``sequence`` (``"EK"``, or ``"E"``), centered on its beads.

    ``forcefield`` is the Martini the probe is made of, and it has to be the one
    the protein is made of: the two meet only through their bead types.

    Built once a sequence and version, and copied, so the caller may move it.
    """
    return copy.deepcopy(_built(sequence.upper(), forcefield))


def probe_charge(sequence: str, forcefield: str = "martini3001") -> float:
    """The probe's charge: its side chains', since both ends are neutral."""
    return sum(float(n["charge"])
               for n in _built(sequence.upper(), forcefield).molecules[0].nodes)  # fmt: skip
