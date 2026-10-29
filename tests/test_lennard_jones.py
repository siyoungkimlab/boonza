"""Lennard-Jones at the cutoff: to_openmm's switches, and boonza md's choice of
them (and of the dispersion correction) by force field, recorded in final.toml."""

import warnings

import pytest

import boonza
from boonza.io.gromacs import load_top
from boonza.md import config
from boonza.md.config import parse_arguments

mm = pytest.importorskip("openmm")
unit = pytest.importorskip("openmm.unit")

SIGMA, EPS = 0.35, 0.4  # nm, kJ/mol
RON, ROFF = 1.0, 1.2  # nm

# two Lennard-Jones beads combined by Lorentz-Berthelot, no pair overrides
PAIR = f"""[ defaults ]
1 2
[ atomtypes ]
C 12.0 0.0 A {SIGMA} {EPS}
[ moleculetype ]
B 1
[ atoms ]
1 C 1 B C 1 0.0
[ system ]
two beads
[ molecules ]
B 2
"""


def _pair(tmp_path, r):
    (tmp_path / "t.top").write_text(PAIR)
    (tmp_path / "t.gro").write_text(
        f"t\n2\n    1B        C    1   0.000   0.000   0.000\n"
        f"    2B        C    2{r:8.3f}   0.000   0.000\n   5.00000   5.00000   5.00000\n")
    return load_top(tmp_path / "t.top", tmp_path / "t.gro")


def _lj_energy(tmp_path, r, **kw) -> float:
    """Lennard-Jones energy (kJ/mol) of the pair at ``r`` nm, every force but none
    else carrying it (the beads are uncharged)."""
    _, system, pos = boonza.to_openmm(_pair(tmp_path, r), nonbonded_method="CutoffPeriodic",
                                      cutoff=10.0 * ROFF, dispersion_correction=False, **kw)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def _lj(r):
    return 4 * EPS * ((SIGMA / r) ** 12 - (SIGMA / r) ** 6)


def _force_switched(r):
    """Steinbach & Brooks's force switch, written out independently."""
    a, b = 4 * EPS * SIGMA**12, 4 * EPS * SIGMA**6
    if r <= RON:
        return a * (r**-12 - (RON * ROFF) ** -6) - b * (r**-6 - (RON * ROFF) ** -3)
    if r < ROFF:
        return (a * ROFF**6 / (ROFF**6 - RON**6) * (r**-6 - ROFF**-6) ** 2
                - b * ROFF**3 / (ROFF**3 - RON**3) * (r**-3 - ROFF**-3) ** 2)  # fmt: skip
    return 0.0


@pytest.mark.parametrize("r", [0.38, 0.6, 0.95, 1.05, 1.15, 1.19, 1.3])
def test_force_switch(tmp_path, r):
    e = _lj_energy(tmp_path, r, lj_switch="force", switch_distance=10.0 * RON)
    assert e == pytest.approx(_force_switched(r), abs=1e-7)


def test_force_switch_is_smooth(tmp_path):
    def e(r):
        return _lj_energy(tmp_path, r, lj_switch="force", switch_distance=10.0 * RON)

    h = 1e-4
    for r in (RON, ROFF - 2 * h):  # energy and force continuous at the switch, both nil at the end
        assert e(r - h) == pytest.approx(e(r + h), abs=1e-6)
    force = lambda r: -(e(r + h) - e(r - h)) / (2 * h)  # noqa: E731
    assert force(RON - 2 * h) == pytest.approx(force(RON + 2 * h), rel=1e-3)
    assert abs(force(ROFF - 2 * h)) < 1e-4 and abs(e(ROFF - 2 * h)) < 1e-7
    # below the switch, the force is Lennard-Jones's own: the energy only moves by a constant
    assert e(0.5) - e(0.6) == pytest.approx(_lj(0.5) - _lj(0.6), abs=1e-7)


@pytest.mark.parametrize("r", [0.6, 1.05, 1.15])
def test_potential_switch(tmp_path, r):
    x = min(max((r - RON) / (ROFF - RON), 0.0), 1.0)
    s = 1 - 6 * x**5 + 15 * x**4 - 10 * x**3  # OpenMM's switching function
    e = _lj_energy(tmp_path, r, lj_switch="potential", switch_distance=10.0 * RON)
    assert e == pytest.approx(_lj(r) * s, abs=1e-7)


def test_switches_refused(tmp_path):
    s = _pair(tmp_path, 0.6)
    with pytest.raises(ValueError, match="no dispersion correction"):
        boonza.to_openmm(s, nonbonded_method="PME", cutoff=12.0, lj_switch="force",
                         switch_distance=10.0)  # fmt: skip
    with pytest.raises(ValueError, match="inside the cutoff"):
        boonza.to_openmm(s, nonbonded_method="PME", cutoff=12.0, dispersion_correction=False,
                         lj_switch="force", switch_distance=12.0)  # fmt: skip
    with pytest.raises(ValueError, match="NoCutoff"):
        boonza.to_openmm(s, lj_switch="potential", switch_distance=8.0)
    with pytest.raises(ValueError, match="shifted or switched"):
        boonza.to_openmm(s, nonbonded_method="PME", cutoff=12.0, lj_shift=True,
                         lj_switch="potential", switch_distance=10.0)  # fmt: skip


CHARMM = ["x", "-f", "aa.charmm.c36m", "-f", "water.tip3p_charmm"]


def _quiet(argv):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a force field's own convention warns about nothing
        return parse_arguments(argv)


def test_conventions():
    a = _quiet(["x"])
    assert (a.dispersion_correction, a.lj_switch, a.lj_switch_nm, a.cutoff_nm) == (True, "none",
                                                                                   None, 0.9)
    c = _quiet(CHARMM)
    assert (c.dispersion_correction, c.lj_switch, c.lj_switch_nm, c.cutoff_nm) == (False, "force",
                                                                                   1.0, 1.2)
    x = _quiet(["x", "-f", "charmm36_2024.xml", "-f", "charmm36_2024/water.xml"])
    assert (x.dispersion_correction, x.lj_switch) == (False, "force")
    for model in ("martini3", "sirah"):
        cg = _quiet(["x", "--model", model])
        assert (cg.dispersion_correction, cg.lj_switch, cg.lj_switch_nm) == (False, "shift", None)


def test_options_reach_openmm():
    from boonza.md.run import all_atom_nonbonded

    assert all_atom_nonbonded(_quiet(CHARMM)) == {
        "nonbonded_method": "PME", "cutoff": 12.0, "dispersion_correction": False,
        "lj_switch": "force", "switch_distance": 10.0}  # fmt: skip
    assert all_atom_nonbonded(_quiet(["x"])) == {
        "nonbonded_method": "PME", "cutoff": 9.0, "dispersion_correction": True,
        "lj_switch": "none", "switch_distance": None}  # fmt: skip


def test_departures_warn():
    with pytest.warns(UserWarning, match="CHARMM36 was parameterized without a dispersion"):
        c = parse_arguments(CHARMM + ["--cutoff-nm", "1.0"])
    assert c.lj_switch_nm == pytest.approx(0.8)  # 1.0 no longer fits inside the cutoff
    with pytest.warns(UserWarning, match="dispersion_correction = false"):
        parse_arguments(["x", "--no-dispersion-correction"])
    with pytest.warns(UserWarning, match="lj_switch"):
        f = parse_arguments(["x", "--lj-switch", "force"])
    # a force switch takes no correction, so Amber's default gives way to it
    assert (f.dispersion_correction, f.lj_switch_nm) == (False, pytest.approx(0.7))
    with pytest.warns(UserWarning, match="under 0.8"):
        parse_arguments(["x", "--cutoff-nm", "0.7"])


def test_contradictions_refused():
    with pytest.raises(SystemExit):
        parse_arguments(CHARMM + ["--dispersion-correction"])  # force switch with a correction
    with pytest.raises(SystemExit):
        parse_arguments(["x", "--lj-switch-nm", "0.8"])  # a start without a switch
    with pytest.raises(SystemExit):
        parse_arguments(["x", "--model", "martini3", "--dispersion-correction"])
    with pytest.raises(SystemExit):
        parse_arguments(CHARMM + ["--lj-switch-nm", "1.3"])  # outside the cutoff


def test_final_toml_records_it(tmp_path):
    for argv, line in ((CHARMM, "force-switched from 1.0 to 1.2 nm, without a long-range "
                                "dispersion correction"),
                       (["x"], "cut at 0.9 nm, with a long-range dispersion correction"),
                       (["x", "--model", "martini3"],
                        "potential-shifted to zero at 1.1 nm, without")):  # fmt: skip
        args = _quiet(argv)
        path = tmp_path / "final.toml"
        config.write_settings(path, config.settings_of(args))
        text = path.read_text()
        assert f"# Lennard-Jones as run: {line}" in text
        assert f'lj_switch = "{args.lj_switch}"' in text
        assert f"dispersion_correction = {str(args.dispersion_correction).lower()}" in text
        if args.model != "aa":
            continue  # a coarse-grained final.toml is read back by a restart, not --config
        # and an all-atom one is a settings file still: --config reads it back
        again = parse_arguments(["x", "--config", str(path)])
        assert (again.lj_switch, again.dispersion_correction) == (args.lj_switch,
                                                                  args.dispersion_correction)
