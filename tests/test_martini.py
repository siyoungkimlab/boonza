"""Martini in OpenMM: the terms GROMACS gives it, checked against GROMACS's own numbers.

The reference energies below were computed with GROMACS 2024.6, so these tests pin
boonza to GROMACS without needing it installed.
"""

import math

import numpy as np
import pytest

import boonza
from boonza.io.gromacs import load_top

mm = pytest.importorskip("openmm")
unit = pytest.importorskip("openmm.unit")

F = 138.935458  # kJ nm / (mol e^2)
MARTINI = {
    "nonbonded_method": "CutoffPeriodic",
    "cutoff": 11.0,
    "dispersion_correction": False,
    "epsilon_r": 15.0,
    "epsilon_rf": 0.0,
    "lj_shift": True,
}


def _gro(*beads):
    """A .gro with one bead per (resname, name, x, y) in nm, in a 5 nm box."""
    rows = [
        f"{1:5d}{r:<5s}{n:>5s}{k + 1:5d}{x:8.3f}{y:8.3f}{0:8.3f}"
        for k, (r, n, x, y) in enumerate(beads)
    ]
    return "t\n{}\n{}\n   5.00000   5.00000   5.00000\n".format(len(beads), "\n".join(rows))


def _load(tmp_path, top, gro):
    (tmp_path / "t.top").write_text(top)
    (tmp_path / "t.gro").write_text(gro)
    return load_top(tmp_path / "t.top", tmp_path / "t.gro")


def _energies(s, **kw):
    """Energy of every force, by name, in kJ/mol on the Reference platform."""
    _top, system, pos = boonza.to_openmm(s, **kw)
    for k, f in enumerate(system.getForces()):
        f.setForceGroup(k)
    platform = mm.Platform.getPlatformByName("Reference")
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), platform)
    ctx.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    ctx.setPositions(pos)
    out = {}
    for k, f in enumerate(system.getForces()):
        e = ctx.getState(getEnergy=True, groups={k}).getPotentialEnergy()
        out[f.getName()] = out.get(f.getName(), 0.0) + e.value_in_unit(unit.kilojoule_per_mole)
    return out, system, ctx


def _coulomb(e):
    return sum(v for k, v in e.items() if k.startswith("nonbonded") and k != "nonbonded_vdw")


TWO_CHARGES = """[ defaults ]
1 2
[ atomtypes ]
Q  72.0 0.0 A 0.0 0.0
[ moleculetype ]
ION 1
[ atoms ]
1 Q 1 ION Q 1  1.0
[ moleculetype ]
CNT 1
[ atoms ]
1 Q 1 CNT Q 1 -1.0
[ system ]
two charges
[ molecules ]
ION 1
CNT 1
"""


def _pair(r):
    return _gro(("ION", "Q", 0.0, 0.0), ("CNT", "Q", r, 0.0))


@pytest.mark.parametrize(
    ("r", "gromacs"),
    [(0.47, -20.475773), (0.80, -13.804800), (1.00, -12.741800), (1.09, -12.631500)],
)
def test_the_reaction_field_is_gromacss(tmp_path, r, gromacs):
    """epsilon_rf = 0 is infinity, and with GROMACS's self and exclusion terms the
    energy is GROMACS's -- which is why it does not reach zero at the cutoff."""
    e, _, _ = _energies(_load(tmp_path, TWO_CHARGES, _pair(r)), **MARTINI)
    assert _coulomb(e) == pytest.approx(gromacs, abs=1e-3)


def test_the_reaction_field_is_not_a_plain_cutoff(tmp_path):
    """The difference that matters for dynamics: the force reaches zero at the cutoff."""
    _, _, ctx = _energies(_load(tmp_path, TWO_CHARGES, _pair(1.099)), **MARTINI)
    forces = ctx.getState(getForces=True).getForces(asNumpy=True)
    near_cutoff = np.linalg.norm(forces.value_in_unit(unit.kilojoule_per_mole / unit.nanometer)[1])
    plain = F / 15 / 1.099**2  # what a cutoff would still be pushing with
    assert near_cutoff < 0.01 * plain


@pytest.mark.parametrize(("r", "gromacs"), [(0.47, -19.707157), (1.00, -9.262364)])
def test_epsilon_rf_equal_to_epsilon_r_is_the_cutoff(tmp_path, r, gromacs):
    """The one setting that really is a plain cutoff, in GROMACS as here."""
    e, _, _ = _energies(_load(tmp_path, TWO_CHARGES, _pair(r)), **{**MARTINI, "epsilon_rf": 15.0})
    assert _coulomb(e) == pytest.approx(gromacs, abs=1e-3)
    assert gromacs == pytest.approx(-F / 15 / r, abs=1e-3)


LJ_PAIR = """[ defaults ]
1 2
[ atomtypes ]
C 72.0 0.0 A 0.0 0.0
[ nonbond_params ]
C C 1 0.47 3.5
[ moleculetype ]
B 1
[ atoms ]
1 C 1 B C 1 0.0
[ system ]
two beads
[ molecules ]
B 2
"""


def test_lj_is_shifted_to_zero_at_the_cutoff(tmp_path):
    def lj(r):
        return 4 * 3.5 * ((0.47 / r) ** 12 - (0.47 / r) ** 6)

    def boonza_lj(r, shift):
        gro = _gro(("B", "C", 0.0, 0.0), ("B", "C", r, 0.0))
        e, _, _ = _energies(_load(tmp_path, LJ_PAIR, gro), **{**MARTINI, "lj_shift": shift})
        return e["nonbonded_vdw"]

    assert boonza_lj(0.6, False) == pytest.approx(lj(0.6), abs=1e-5)
    assert boonza_lj(0.6, True) == pytest.approx(lj(0.6) - lj(1.1), abs=1e-5)
    assert abs(boonza_lj(1.099, True)) < 0.01 * abs(lj(1.1))  # almost nothing left at 1.099


ANGLES = """[ defaults ]
1 2
[ atomtypes ]
C 72.0 0.0 A 0.0 0.0
[ moleculetype ]
A 1
[ atoms ]
1 C 1 A B1 1 0.0
2 C 1 A B2 1 0.0
3 C 1 A B3 1 0.0
[ angles ]
1 2 3 {funct} 120.0 25.0
[ system ]
angle
[ molecules ]
A 1
"""


def _bent(deg):
    """Three beads 0.3 nm apart at the given angle, and the angle the .gro keeps."""
    th = math.radians(deg)
    x, y = round(0.3 * math.cos(th), 3), round(0.3 * math.sin(th), 3)
    gro = _gro(("A", "B1", 0.3, 0.0), ("A", "B2", 0.0, 0.0), ("A", "B3", x, y))
    return gro, math.atan2(y, x)


def _cosine(th):
    return 0.5 * 25 * (math.cos(th) - math.cos(math.radians(120))) ** 2


@pytest.mark.parametrize(
    ("funct", "table", "form"),
    [
        (2, "angle_cosine_harm", _cosine),
        (10, "angle_restricted", lambda th: _cosine(th) / math.sin(th) ** 2),
    ],
)
def test_martini_angles(tmp_path, funct, table, form):
    """GROMACS's cosine angles (type 2) and restricted bending (type 10)."""
    gro, theta = _bent(95.0)
    s = _load(tmp_path, ANGLES.format(funct=funct), gro)
    assert len(s.tables[table]) == 1
    e, _, _ = _energies(s, **MARTINI)
    assert e[table] == pytest.approx(form(theta), rel=1e-5)


def test_restricted_bending_resists_straightening(tmp_path):
    """Why Martini uses it: the energy grows without bound toward 180 degrees."""

    def energy(deg):
        gro, _ = _bent(deg)
        e, _, _ = _energies(_load(tmp_path, ANGLES.format(funct=10), gro), **MARTINI)
        return e["angle_restricted"]

    assert energy(175) > 10 * energy(150) > 0


VSITE = """[ defaults ]
1 2
[ atomtypes ]
C 36.0 0.0 A 0.0 0.0
V 0.0 0.0 A 0.0 0.0
[ moleculetype ]
R 1
[ atoms ]
1 C 1 R A 1 0.0 36.0
2 C 1 R B 1 0.0 72.0
3 C 1 R C 1 0.0 36.0
4 C 1 R D 1 0.0 36.0
5 V 1 R S 1 0.0 0.0
[ virtual_sitesn ]
5 {funct} {parents}
[ system ]
ring
[ molecules ]
R 1
"""
RING = np.array([[0.0, 0.0], [0.4, 0.0], [0.4, 0.3], [0.0, 0.3]])


@pytest.mark.parametrize(
    ("funct", "parents", "weights"),
    [
        (1, "1 2 3 4", [1, 1, 1, 1]),  # centre of geometry
        (2, "1 2 3 4", [36, 72, 36, 36]),  # centre of mass
        (3, "1 1.0 2 3.0 3 0.0 4 0.0", [1, 3, 0, 0]),  # explicit weights
    ],
)
def test_virtual_sites_from_any_number_of_atoms(tmp_path, funct, parents, weights):
    beads = [("R", n, x, y) for n, (x, y) in zip("ABCD", RING, strict=True)]
    gro = _gro(*beads, ("R", "S", 0.0, 0.0))
    s = _load(tmp_path, VSITE.format(funct=funct, parents=parents), gro)
    assert len(s.tables["virtual_lc4"]) == 1
    _, system, ctx = _energies(s, nonbonded_method="CutoffPeriodic", cutoff=11.0)
    assert system.isVirtualSite(4)
    ctx.computeVirtualSites()
    site = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    w = np.array(weights, float)
    assert site[4][:2] == pytest.approx((RING * w[:, None]).sum(0) / w.sum(), abs=1e-6)


CONSTRAINED = """[ defaults ]
1 2
[ atomtypes ]
C 72.0 0.0 A 0.0 0.0
[ constrainttypes ]
C C 1 0.30
[ moleculetype ]
M 1
[ atoms ]
1 C 1 M A 1 0.0
2 C 1 M B 1 0.0
3 C 1 M C 1 0.0
[ constraints ]
1 2 1 0.25
2 3 1
[ system ]
t
[ molecules ]
M 1
"""


def test_constraints_are_kept_as_constraints(tmp_path):
    """They used to be read only as connectivity, which left the beads bonded in the
    topology and held by nothing; a length missing from the line comes from
    [ constrainttypes ]."""
    gro = _gro(("M", "A", 0.0, 0.0), ("M", "B", 0.25, 0.0), ("M", "C", 0.55, 0.0))
    s = _load(tmp_path, CONSTRAINED, gro)
    assert sorted(np.asarray(s.tables["constraint_ah1"].params["r1"])) == pytest.approx([2.5, 3.0])
    _, system, _ = _energies(s, nonbonded_method="CutoffPeriodic", cutoff=11.0)
    assert system.getNumConstraints() == 2


def test_what_the_options_refuse(tmp_path):
    s = _load(tmp_path, TWO_CHARGES, _pair(0.5))
    with pytest.raises(ValueError, match="epsilon_rf"):
        boonza.to_openmm(s, nonbonded_method="NoCutoff", epsilon_rf=0.0)


RIGID = """[ defaults ]
1 2
[ atomtypes ]
C 72.0 0.0 A 0.0 0.0
V 0.0 0.0 A 0.0 0.0
P 72.0 0.0 A 0.0 0.0
[ nonbond_params ]
C C 1 0.47 0.5
C V 1 0.47 0.5
V V 1 0.47 0.5
C P 1 0.47 0.5
V P 1 0.40 4.0
P P 1 0.47 0.5
[ moleculetype ]
RIG 1
[ atoms ]
1 C 1 RIG A 1 0.0 72.0
2 C 1 RIG B 2 0.0 72.0
3 C 1 RIG D 3 0.0 72.0
4 V 1 RIG O 4 0.5 0.0
5 V 1 RIG T 5 0.0 0.0
6 V 1 RIG L 6 -0.5 0.0
[ constraints ]
1 2 1 0.40
1 3 1 0.45
2 3 1 0.35
[ virtual_sites3 ]
4 1 2 3 4 0.6 0.3 1.8
5 1 2 3 1 0.3 0.3
[ virtual_sites2 ]
6 2 3 1 0.3
[ exclusions ]
1 2 3 4 5 6
2 3 4 5 6
3 4 5 6
4 5 6
5 6
[ moleculetype ]
PRB 1
[ atoms ]
1 P 1 PRB P 1 0.0
[ system ]
vsites
[ molecules ]
RIG 1
PRB 3
"""
RIGID_GRO = """vsites
9
    1RIG      A    1   2.000   2.000   2.000
    1RIG      B    2   2.400   2.000   2.000
    1RIG      D    3   2.150   2.417   2.000
    1RIG      O    4   0.000   0.000   0.000
    1RIG      T    5   0.000   0.000   0.000
    1RIG      L    6   0.000   0.000   0.000
    2PRB      P    7   2.285   2.125   2.750
    3PRB      P    8   2.165   2.125   1.550
    4PRB      P    9   2.800   2.125   2.000
   5.00000   5.00000   5.00000
"""


def test_virtual_sites_as_gromacs_builds_them(tmp_path):
    """virtual_sites3 (3 and 3out, as Martini 3's cholesterol uses) and
    virtual_sites2, placed by OpenMM from their parents: the energy is
    GROMACS's with the sites it builds (a 0-step md run, not -rerun, which
    takes sites as given).  The charged sites are excluded from each other,
    so the reaction field's exclusion term counts too."""
    s = _load(tmp_path, RIGID, RIGID_GRO)
    assert {t for t in s.table_names if t.startswith("virtual")} == {
        "virtual_lc2", "virtual_lc3", "virtual_out3"}  # fmt: skip
    _, system, ctx = _energies(s, **MARTINI)
    ctx.computeVirtualSites()
    energy = (
        ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    )
    assert energy == pytest.approx(-16.233511, abs=2e-4)  # GROMACS 2024.6
    x = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    a, b, c = x[0], x[1], x[2]
    assert x[3] == pytest.approx(a + 0.6 * (b - a) + 0.3 * (c - a) + 1.8 * np.cross(b - a, c - a))
    assert x[4] == pytest.approx(a + 0.3 * (b - a) + 0.3 * (c - a))
    assert x[5] == pytest.approx(0.7 * b + 0.3 * c)
