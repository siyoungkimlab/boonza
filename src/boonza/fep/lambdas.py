"""The lambda schedule of a hybrid-topology transformation.

One lambda runs from 0 (ligand A) to 1 (ligand B); each part of the hybrid
follows its own function of it, as perses and openfe's "default" protocol:
the atoms shared by A and B (the core) change linearly; A's own atoms lose
their charges over the first half while B's own atoms gain their
Lennard-Jones, then A's lose their Lennard-Jones while B's gain their charges.
So no atom ever carries a charge without its Lennard-Jones.
"""

from __future__ import annotations

import numpy as np

#: the global parameters of a hybrid system (boonza.fep.hybrid)
PARAMETERS = (
    "lambda_sterics_core",
    "lambda_electrostatics_core",
    "lambda_sterics_insert",
    "lambda_sterics_delete",
    "lambda_electrostatics_insert",
    "lambda_electrostatics_delete",
    "lambda_bonds",
    "lambda_angles",
    "lambda_torsions",
)


def _first_half(x: float) -> float:
    return min(2.0 * x, 1.0)


def _second_half(x: float) -> float:
    return max(2.0 * (x - 0.5), 0.0)


def values(x: float) -> dict[str, float]:
    """Each global parameter at master lambda ``x`` (0 to 1)."""
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"lambda must lie in [0, 1], not {x}")
    return {
        "lambda_sterics_core": x,
        "lambda_electrostatics_core": x,
        "lambda_sterics_insert": _first_half(x),
        "lambda_sterics_delete": _second_half(x),
        "lambda_electrostatics_insert": _second_half(x),
        "lambda_electrostatics_delete": _first_half(x),
        "lambda_bonds": x,
        "lambda_angles": x,
        "lambda_torsions": x,
    }


def windows(n: int = 11) -> np.ndarray:
    """``n`` evenly spaced master lambdas from 0 to 1 (openfe's default: 11)."""
    if n < 2:
        raise ValueError("a schedule needs at least the two end states")
    return np.linspace(0.0, 1.0, n)
