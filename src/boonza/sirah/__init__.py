"""SIRAH: a coarse-grained force field that keeps its parameters in its topology.

SIRAH maps a protein to a few beads a residue, keeps explicit charges on them
and runs with PME, unlike Martini's reaction field.  boonza reads a SIRAH
topology and runs it; the settings here are the ones SIRAH's own mdp files
use (``tutorial/7/md_CGPROT.mdp`` of the GROMACS distribution):

* a 20 fs step,
* PME with a 1.2 nm cut-off for both Coulomb and Lennard-Jones,
* Lennard-Jones shifted to zero at the cut-off, as GROMACS's Verlet scheme
  does by default, and no dispersion correction,
* the medium's own permittivity, since the charges are explicit.

Reference: Machado, Barrera, Klein, Sóñora, Silva and Pantano, J. Chem.
Theory Comput. 15, 2719-2733 (2019), and https://www.sirahff.com.
"""

from __future__ import annotations

#: What :func:`boonza.to_openmm` needs to reproduce GROMACS for SIRAH.
OPENMM_OPTIONS = {
    "nonbonded_method": "PME",
    "cutoff": 12.0,
    "dispersion_correction": False,
    "lj_shift": True,
}
#: SIRAH's water: four beads standing for about eleven waters.
WATER = "WT4"
#: Its electrolytes, which stand for solvated ions.
IONS = ("NaW", "KW", "ClW")
#: One ion pair for this many waters is about 0.15 M (Machado et al. 2019).
WATERS_PER_ION_PAIR = 34
#: The backbone beads of a SIRAH residue: amide N, alpha C and carbonyl O.
BACKBONE = ("GN", "GC", "GO")

__all__ = ["BACKBONE", "IONS", "OPENMM_OPTIONS", "WATER", "WATERS_PER_ION_PAIR"]
