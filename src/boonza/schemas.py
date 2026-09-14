"""Term-table schemas understood by DMS readers and writers.

The schema list is taken from msys (src/schema/schema.cxx),
Copyright 2018-2020, D. E. Shaw Research.  See NOTICE for the msys license.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Schema:
    name: str
    category: str
    natoms: int
    params: tuple[tuple[str, str], ...] = ()
    term_props: tuple[tuple[str, str], ...] = ()


def _cols(names, kind="float"):
    return tuple((n, kind) if isinstance(n, str) else n for n in names)


def _s(name, category, natoms, params=(), term_props=()):
    return Schema(name, category, natoms, _cols(params), _cols(term_props))


def _fc(prefix, lo, hi, suffix=""):
    return [f"{prefix}{i}{suffix}" for i in range(lo, hi + 1)]


def _xyz(n):
    return [f"{c}{i}" for i in range(n) for c in "xyz"]


_CONSTRAINED = (("constrained", "int"),)
_SCHEDULE = (("schedule", "str"),)

_TERM = [
    _s("angle_harm", "bond", 3, ["theta0", "fc"], _CONSTRAINED),
    _s("angle_fbhw", "bond", 3, ["sigma", "theta0", "fc"]),
    _s("dihedral_trig", "bond", 4, ["phi0", *_fc("fc", 0, 6)]),
    _s("dihedral_fourier", "bond", 4, _fc("fc", 0, 12)),
    _s("dihedral6_trig", "bond", 6, ["phi0", "fc0", "fc2", "fc4"]),
    _s("improper_anharm", "bond", 4, ["fc2", "fc4"]),
    _s("improper_fbhw", "bond", 4, ["sigma", "phi0", "fc"]),
    _s("improper_harm", "bond", 4, ["phi0", "fc"]),
    _s("inplanewag_harm", "bond", 4, ["w0", "fc"]),
    _s("pair_12_6_es", "bond", 2, ["aij", "bij", "qij"]),
    _s("pair_softcore_es", "bond", 2, ["aij", "bij", "qij", "lambda"]),
    _s("pair_exp_6_es", "bond", 2, ["aij", "bij", "cij", "qij"]),
    _s("posre_harm", "bond", 1, ["fcx", "fcy", "fcz"], ["x0", "y0", "z0"]),
    _s("posre_fbhw", "bond", 1, ["fc", "sigma"], ["x0", "y0", "z0"]),
    _s("stretch_harm", "bond", 2, ["r0", "fc"], _CONSTRAINED),
    _s("stretch_morse", "bond", 2, ["r0", "d", "a"]),
    _s("softstretch_harm", "bond", 2, ["r0", "fc", "alpha"]),
    _s("softened_stretch_harm", "bond", 2, ["r0", "fc", "alpha", "lambda"]),
    _s("torsiontorsion_cmap", "bond", 8, [("cmapid", "str")]),
    _s("pseudopol_fermi", "bond", 4, ["a", "b", "cutoff"]),
    _s("alchemical_stretch_harm", "bond", 2, ["r0A", "fcA", "r0B", "fcB"], _SCHEDULE),
    _s(
        "alchemical_stretch_morse",
        "bond",
        2,
        ["r0A", "dA", "aA", "r0B", "dB", "aB"],
        _SCHEDULE,
    ),
    _s(
        "alchemical_softstretch_harm",
        "bond",
        2,
        ["r0A", "fcA", "alphaA", "r0B", "fcB", "alphaB"],
        _SCHEDULE,
    ),
    _s("alchemical_angle_harm", "bond", 3, ["theta0A", "fcA", "theta0B", "fcB"], _SCHEDULE),
    _s(
        "alchemical_angle_harm_soft",
        "bond",
        3,
        ["theta0A", "fcA", "epsilonA", "theta0B", "fcB", "epsilonB"],
        _SCHEDULE,
    ),
    _s(
        "alchemical_dihedral_trig",
        "bond",
        4,
        ["phi0A", *_fc("fc", 0, 6, "A"), "phi0B", *_fc("fc", 0, 6, "B")],
        _SCHEDULE,
    ),
    _s(
        "alchemical_dihedral_trig_soft",
        "bond",
        4,
        ["phi0A", *_fc("fc", 0, 6, "A"), "epsilonA", "phi0B", *_fc("fc", 0, 6, "B"), "epsilonB"],
        _SCHEDULE,
    ),
    _s(
        "alchemical_pair_12_6_es",
        "bond",
        2,
        ["aijA", "bijA", "qijA", "aijB", "bijB", "qijB"],
        _SCHEDULE,
    ),
    _s(
        "alchemical_pair_exp_6_es",
        "bond",
        2,
        ["aijA", "bijA", "cijA", "qijA", "aijB", "bijB", "cijB", "qijB"],
        _SCHEDULE,
    ),
    _s("alchemical_improper_harm", "bond", 4, ["phi0A", "fcA", "phi0B", "fcB"], _SCHEDULE),
    _s(
        "alchemical_improper_harm_soft",
        "bond",
        4,
        ["phi0A", "fcA", "epsilonA", "phi0B", "fcB", "epsilonB"],
        _SCHEDULE,
    ),
    _s(
        "alchemical_torsiontorsion_cmap",
        "bond",
        8,
        [("cmapidA", "str"), ("cmapidB", "str")],
        _SCHEDULE,
    ),
    _s("constraint_hoh", "constraint", 3, ["theta", "r1", "r2"]),
    *[_s(f"constraint_ah{n}", "constraint", n + 1, _fc("r", 1, n)) for n in range(1, 9)],
    _s("constraint_ah1R", "constraint", 2, _fc("r", 1, 1)),
    _s("constraint_ah2R", "constraint", 3, _fc("r", 1, 3)),
    _s("constraint_ah3R", "constraint", 4, _fc("r", 1, 6)),
    _s("constraint_ah4R", "constraint", 5, _fc("r", 1, 10)),
    *[_s(f"rigid_explicit{n}", "constraint", n, _xyz(n)) for n in range(2, 10)],
    _s("virtual_fdat3", "virtual", 4, ["c1", "c2", "c3"]),
    _s("virtual_lc1", "virtual", 2),
    *[
        _s(f"virtual_lc{n}{sfx}", "virtual", n + 1, _fc("c", 1, n - 1))
        for n in range(2, 8)
        for sfx in ("", "n")
    ],
    _s("virtual_midpoint", "virtual", 3, ["c1"]),
    _s("virtual_out3", "virtual", 4, ["c1", "c2", "c3"]),
    _s("virtual_out3n", "virtual", 4, ["c1", "c2", "c3"]),
    _s("virtual_sp3", "virtual", 4, ["c1", "c2"]),
    _s("exclusion", "exclusion", 2),
]

_NONBONDED = [
    _s("vdw_12_6", "nonbonded", 1, ["sigma", "epsilon"]),
    _s("vdw_exp_6", "nonbonded", 1, ["alpha", "epsilon", "rmin"]),
    _s("vdw_exp_6s", "nonbonded", 1, ["sigma", "epsilon", "lne"]),
    _s("polynomial_cij", "nonbonded", 1, _fc("c", 1, 16)),
]

TERM_SCHEMAS: dict[str, Schema] = {s.name: s for s in _TERM}
NONBONDED_SCHEMAS: dict[str, Schema] = {s.name: s for s in _NONBONDED}
