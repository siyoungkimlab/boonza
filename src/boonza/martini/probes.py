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


def probe_sequences(residues=PROBE_RESIDUES) -> list[str]:
    """Every dipeptide of ``residues`` as a two-letter sequence, XY and YX once.

    In Martini the two differ only in which backbone bead carries which side
    chain, so the 14 default residues give 105 probes rather than 196.
    """
    letters = [LETTERS[r] if r in LETTERS else r for r in residues]
    return [a + b for k, a in enumerate(letters) for b in letters[k:]]


@cache
def _built(sequence: str):
    from .. import peptide
    from .build import martinize

    # the builder's peptides carry neutral side chains (a hydrogen on Glu's
    # carboxyl, two on Lys's amine); mapping the heavy atoms alone gives the
    # charged forms of pH 7, with His neutral
    m = martinize(peptide(sequence, conformation="extended"), "protein and not element H",
                  ss=False, scfix=False, neutral_termini=True)  # fmt: skip
    if len(m.molecules) != 1:
        raise ValueError(f"{sequence} did not martinize into one molecule")
    for node in m.molecules[0].nodes:  # the probe's own residue name, not the protein's
        node["resname"] = sequence
    m.names = [sequence]
    m.positions = m.positions - m.positions.mean(0)
    return m


def probe(sequence: str):
    """The martinized dipeptide ``sequence`` (e.g. ``"EK"``), centered on its beads.

    Built once and copied, so the caller may move it.
    """
    return copy.deepcopy(_built(sequence.upper()))


def probe_charge(sequence: str) -> float:
    """The probe's charge: its side chains', since both ends are neutral."""
    return sum(float(n["charge"]) for n in _built(sequence.upper()).molecules[0].nodes)
