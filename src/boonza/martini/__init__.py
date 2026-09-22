"""Martini 3 coarse-graining, after martinize2 (vermouth), with its data files."""

from .build import Martinized, convert_dssp_to_martini, force_field, martinize
from .md import equilibrate
from .solvate import solvate

# to_openmm settings for Martini 3: GROMACS's reaction field (epsilon_r 15,
# epsilon_rf infinite) and potential-shifted Lennard-Jones, both cut at 11 Å
OPENMM_OPTIONS = {
    "nonbonded_method": "CutoffPeriodic",
    "cutoff": 11.0,
    "dispersion_correction": False,
    "epsilon_r": 15.0,
    "epsilon_rf": 0.0,
    "lj_shift": True,
}

__all__ = ["OPENMM_OPTIONS", "Martinized", "convert_dssp_to_martini", "equilibrate",
           "force_field", "martinize", "solvate"]  # fmt: skip
