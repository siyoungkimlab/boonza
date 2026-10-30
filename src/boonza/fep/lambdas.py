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


#: the master lambdas between which a large group's sterics are switched densely
#: (boonza.fep.lambdas.dense): B's own atoms reach full Lennard-Jones at 0.5 and A's
#: start to lose theirs there, the steepest stretch of the schedule
DENSE_FROM, DENSE_TO, DENSE_STEP = 0.3, 0.7, 0.025


def dense() -> np.ndarray:
    """23 master lambdas for an edge that adds or removes a large group: 0.1 apart
    outside [DENSE_FROM, DENSE_TO], DENSE_STEP apart inside, where its atoms'
    Lennard-Jones is switched.  With 11 evenly spaced, the step from 0.5 to 0.6
    left a benzyl ring that had drifted into the core, once its Lennard-Jones was
    mostly off, at 50 to 10^7 kT when it came back on: neighbouring windows shared
    no configuration and MBAR returned nonsense."""
    n = int(round((DENSE_TO - DENSE_FROM) / DENSE_STEP))
    inside = DENSE_FROM + DENSE_STEP * np.arange(n + 1)
    outside = np.arange(0.0, 1.0 + 1e-9, 0.1)
    keep = (outside < DENSE_FROM - 1e-9) | (outside > DENSE_TO + 1e-9)
    return np.round(np.sort(np.concatenate([outside[keep], inside])), 6)


def windows(n: int = 11) -> np.ndarray:
    """``n`` evenly spaced master lambdas from 0 to 1 (openfe's default: 11)."""
    if n < 2:
        raise ValueError("a schedule needs at least the two end states")
    return np.linspace(0.0, 1.0, n)
