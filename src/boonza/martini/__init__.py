"""Martini 3 coarse-graining, after martinize2 (vermouth), with its data files."""

from pathlib import Path

from .build import Martinized, convert_dssp_to_martini, force_field, martinize
from .md import equilibrate
from .membrane import bilayer, lipid_templates
from .solvate import solvate

PARAMETERS = Path(__file__).resolve().parent.parent / "data" / "martini" / "params"
NONBONDED = "martini_v3.0.0.itp"  # particle definitions and Lennard-Jones interactions
LIPIDS = ("martini_v3.0.0_phospholipids_v1.itp", "martini_v3.0_sterols_v1.0.itp")


def parameters(*names) -> list[Path]:
    """Parameter files that came with boonza, by name; all of them with none.

    The Martini 3.0.0 release from cgmartini.nl, unchanged, so that martinizing
    and building a membrane need no separate download, and the Martini 2 files
    for what has not been ported to Martini 3.  The two must not be mixed in
    one system.  Give a path of your own instead wherever one of these is
    taken, to use another version or a lipid boonza does not carry.
    """
    if not names:
        return sorted(PARAMETERS.glob("*.itp"))
    out = []
    for name in names:
        here = PARAMETERS / Path(name).name
        if not here.is_file():
            have = ", ".join(p.name for p in sorted(PARAMETERS.glob("*.itp")))
            raise FileNotFoundError(f"boonza does not carry {name}; it has {have}")
        out.append(here)
    return out


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

__all__ = ["LIPIDS", "NONBONDED", "OPENMM_OPTIONS", "PARAMETERS", "Martinized", "bilayer",
           "convert_dssp_to_martini", "equilibrate", "force_field", "lipid_templates",
           "martinize", "parameters", "solvate"]  # fmt: skip
