"""REM: soft, bounded nonbonded interactions for a minimization from a clash."""

from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza import rem

DATA = Path(__file__).parent / "data"
openmm = pytest.importorskip("openmm")


def _peptide():
    g = DATA / "gmx_amber"
    s = boonza.load(g / "topol.top", coordinates=g / "conf.gro")
    return s.clone(s.select("protein").ids)


def _energy(system, xyz_nm, groups=-1):
    import openmm.unit as u

    c = openmm.Context(system, openmm.VerletIntegrator(0.001),
                       openmm.Platform.getPlatformByName("Reference"))  # fmt: skip
    c.setPositions(np.asarray(xyz_nm) * u.nanometer)
    c.computeVirtualSites()
    e = c.getState(getEnergy=True, groups=groups).getPotentialEnergy()
    return e.value_in_unit(u.kilojoule_per_mole)


def test_soft_pairs_are_bounded_and_the_force_field_beyond_sigma():
    s = _peptide()
    _, system, _ = boonza.to_openmm(s, nonbonded_method="CutoffNonPeriodic", cutoff=12.0,
                                    constraints=False)  # fmt: skip
    soft = rem.soften(system)
    names = [f.getName() for f in soft.getForces()]
    assert "nonbonded" not in names and {"rem_pairs", "rem_exceptions"} <= set(names)
    assert not any(isinstance(f, openmm.NonbondedForce) for f in soft.getForces())
    assert len(names) == system.getNumForces() + 1  # bonded terms kept
    xyz = np.asarray(s.positions) / 10
    # one atom on top of another: astronomical for the force field, bounded when soft
    bad = xyz.copy()
    bad[0] = bad[20] + 0.001
    assert _energy(system, bad) > 1e10
    group = soft.getForce(names.index("rem_pairs")).getForceGroup()
    assert _energy(soft, bad, {group}) < 1e4
    # two atoms far apart (beyond sigma, within the cutoff): exactly Lennard-Jones
    two = openmm.System()
    for _ in range(2):
        two.addParticle(12.0)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(nb.CutoffNonPeriodic)
    nb.setCutoffDistance(1.2)
    nb.addParticle(0.0, 0.3, 0.5)
    nb.addParticle(0.0, 0.3, 0.5)
    two.addForce(nb)
    for r in (0.35, 0.5, 0.8):
        x = [[0, 0, 0], [r, 0, 0]]
        assert _energy(rem.soften(two), x) == pytest.approx(_energy(two, x), rel=1e-6)
    near = [[0, 0, 0], [1e-4, 0, 0]]  # within sigma: never above A
    assert _energy(rem.soften(two), near) == pytest.approx(rem.A_KJ, rel=1e-3)
    # the Coulomb term: bounded, zero at the cutoff
    nb.setParticleParameters(0, 1.0, 0.3, 0.0)
    nb.setParticleParameters(1, -1.0, 0.3, 0.0)
    assert _energy(rem.soften(two), [[0, 0, 0], [1e-4, 0, 0]]) == pytest.approx(-rem.C_KJ, rel=1e-3)
    assert _energy(rem.soften(two), [[0, 0, 0], [1.19999, 0, 0]]) == pytest.approx(0, abs=1e-6)


def test_relax_untangles_a_clash_and_holds_only_what_is_named():
    """Two copies of a peptide 1 A apart (atoms on atoms): REM pulls them apart.  No
    torsion is held unless named; a named one ends at the angle given."""
    one = _peptide()
    s = one.clone()
    other = one.clone()
    other.positions = other.positions + [1.0, 0, 0]
    s.append(other)
    _, system, _ = boonza.to_openmm(s, nonbonded_method="CutoffNonPeriodic", cutoff=12.0,
                                    constraints=False)  # fmt: skip
    xyz = np.asarray(s.positions) / 10
    new, report = rem.relax(system, xyz)
    assert report["rem_energy_after"] < report["rem_energy_before"]
    assert _energy(system, new) < _energy(system, xyz) / 1e3
    # a named improper at an alpha carbon, asked for as its mirror image: a stiff
    # enough hold gets it there (the default one is for holding, not turning)
    ca = int(s.select("resid 2 and name CA").ids[0])
    n, c, cb = (int(s.select(f"resid 2 and name {x}").ids[0]) for x in ("N", "C", "CB"))
    now = np.degrees(rem._dihedral(xyz, ca, n, c, cb))
    held, _ = rem.relax(system, xyz, torsions=[(ca, n, c, cb, -now)], hold_kJ=5000)
    assert np.degrees(rem._dihedral(held, ca, n, c, cb)) == pytest.approx(-now, abs=10)
    # held where it is unless an angle is given: zero energy at the start
    probe = openmm.System()
    for _ in range(s.natoms):
        probe.addParticle(1.0)
    probe.addForce(rem.hold_torsions([(ca, n, c, cb)], xyz))
    assert _energy(probe, xyz) == pytest.approx(0, abs=1e-8)


def test_martini_softens_its_tables_and_drops_the_reaction_field():
    from boonza.martini import OPENMM_OPTIONS, martinize

    cg = martinize(boonza.load(DATA / "1TEN.pdb"), elastic=True).system()
    _, system, _ = boonza.to_openmm(cg, **OPENMM_OPTIONS)
    names = {f.getName() for f in system.getForces()}
    assert "nonbonded_vdw" in names
    soft = rem.soften(system)
    left = {f.getName() for f in soft.getForces()}
    assert not left & {"nonbonded", "nonbonded_vdw", "nonbonded_rf_exclusions", "nonbonded_rf_self"}
    xyz = np.asarray(cg.positions) / 10
    assert np.isfinite(_energy(soft, xyz))
    bad = xyz.copy()
    bad[10] = bad[40]
    new, report = rem.relax(system, bad)  # two beads at one point
    assert np.isfinite(report["rem_energy_after"]) and np.isfinite(_energy(system, new))
    assert np.linalg.norm(new[10] - new[40]) > 0.1


def test_sirah_too():
    from boonza.sirah import OPENMM_OPTIONS, sirahize

    cg = sirahize(boonza.load(DATA / "1TEN.pdb")).system()
    _, system, _ = boonza.to_openmm(cg, **{**OPENMM_OPTIONS, "cutoff": 12.0})
    bad = np.asarray(cg.positions) / 10
    bad[10] = bad[40]
    new, report = rem.relax(system, bad)
    assert report["rem_energy_after"] < report["rem_energy_before"]
    assert np.isfinite(_energy(system, new)) and np.linalg.norm(new[10] - new[40]) > 0.1


def test_the_viewer_minimizes_with_rem_first():
    """Minimize with rem: the soft stage, then the force field's, one path."""
    from boonza import viewer

    session = viewer.Session(_peptide(), atoms="protein", cg=False)
    heavy = session.system.select("not hydrogen").ids
    out = session.minimize(fixed=[], tethered=heavy, restraint=10, rem=True, path=True)
    assert out["rem_energy_after"] < out["rem_energy_before"]
    assert out["energy_after"] < out["energy_before"]
    stages = out["path"]["stages"]
    assert [name for name, _ in stages] == ["REM", "force field"] and stages[1][1] > 0
    its = [k for k, _ in out["path"]["energy"]]
    assert its == sorted(its) and stages[1][1] in its
