"""Dipeptide probes in SIRAH, as :mod:`boonza.martini.probes` builds them for Martini.

The reasoning is the same: SIRAH parameterizes every amino acid, so a
dipeptide is a probe that needs nothing new, while an arbitrary small
molecule would need work SIRAH has not done.  The probes are the same 105 of
the same 14 residues, so a surface mapped in one model can be read against
the other.

SIRAH is the finer of the two here -- its side chains are two to five beads
where Martini's are one or two -- so a probe's chemistry sits where the
groups are rather than at their centre.
"""

from __future__ import annotations

import copy
from functools import cache

from ..martini.probes import (
    LETTERS,
    PROBE_RESIDUES,
    SINGLE_RESIDUES,
    probe_sequences,
    single_sequences,
)
from .build import sirahize

__all__ = ["LETTERS", "PROBE_RESIDUES", "PROBE_PREFIX", "SINGLE_RESIDUES", "probe",
           "probe_charge", "probe_sequences", "single_sequences"]  # fmt: skip

#: What a probe's molecule is called in the topology.  Its beads keep the
#: sequence as their residue name, which is what selects them, but the
#: ``[ moleculetype ]`` cannot be ``KW``: SIRAH's potassium is already called
#: that, and the second definition of a name silently loses to the first.
PROBE_PREFIX = "probe_"


@cache
def _built(sequence: str):
    from .. import peptide

    aa = peptide(sequence, conformation="extended")
    m = sirahize(aa, "protein", termini="Neutral")
    if len(m.molecules) != 1:
        raise ValueError(f"{sequence} did not map into one molecule")
    for bead in m.molecules[0].beads:  # the probe's own name, not the protein's
        bead.residue = sequence
    m.molecules[0].name = PROBE_PREFIX + sequence
    m.positions = m.positions - m.positions.mean(0)
    return m


def probe(sequence: str):
    """The dipeptide ``sequence`` (``"EK"``) mapped onto SIRAH beads, centered.

    Built once and copied, so the caller may move it.  Its ends are neutral,
    as a probe's should be: only the side chains carry charge.
    """
    return copy.deepcopy(_built(sequence.upper()))


def probe_charge(sequence: str) -> float:
    """The probe's charge, which is its side chains'."""
    return sum(_built(sequence.upper()).molecules[0].charges)
