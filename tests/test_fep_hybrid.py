"""The hybrid topology: exact at its ends, and the identity when nothing changes.

Ala -> Ser in the middle of a tripeptide (Amber ff14SB, no AmberTools): the
backbone and CB are the core, Ala's HB1 is its own atom, Ser's OG and HG are.
"""

import numpy as np
import pytest

import boonza
from boonza.fep import hybrid, lambdas

openmm = pytest.importorskip("openmm")


def _system(seq, method="NoCutoff"):
    from boonza.viewer.edit import protonate

    s = boonza.peptide(seq)
    s = s.clone(np.flatnonzero(s.atoms["anum"] > 1))
    protonate(s)
    p = boonza.parameterize(s, ["aa.amber.ff14SB"])
    if method != "NoCutoff":
        p.cell = np.eye(3) * 40.0
    _, system, _ = boonza.to_openmm(p, nonbonded_method=method, cutoff=12.0, constraints=False,
                                    dispersion_correction=False)  # fmt: skip
    return p, system, np.asarray(p.positions) / 10.0


def _mapping(a, b):
    """Every atom by (residue, name): the middle residue's are the core."""
    where = {(int(b.atoms["residue"][k]), str(b.atoms["name"][k])): k for k in range(b.natoms)}
    core, env = {}, {}
    for k in range(a.natoms):
        key = (int(a.atoms["residue"][k]), str(a.atoms["name"][k]))
        if key in where:
            (core if key[0] == 1 else env)[k] = where[key]
    return core, env


def _energy(system, xyz):
    c = openmm.Context(system, openmm.VerletIntegrator(0.001),
                       openmm.Platform.getPlatformByName("Reference"))  # fmt: skip
    c.setPositions(np.asarray(xyz) * openmm.unit.nanometer)
    return c, c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        openmm.unit.kilojoule_per_mole)  # fmt: skip


def _hybrid_energy(h, x):
    c, _ = _energy(h.system, h.positions)
    h.set_lambda(c, x)
    return c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        openmm.unit.kilojoule_per_mole)  # fmt: skip


def _valence(system, atoms, xyz):
    """The bonds, angles and torsions of ``system`` touching any of ``atoms``."""
    out = openmm.System()
    for k in range(system.getNumParticles()):
        out.addParticle(system.getParticleMass(k))
    atoms = set(atoms)
    for f in system.getForces():
        if isinstance(f, openmm.HarmonicBondForce):
            g = openmm.HarmonicBondForce()
            for k in range(f.getNumBonds()):
                i, j, r, kk = f.getBondParameters(k)
                if {i, j} & atoms:
                    g.addBond(i, j, r, kk)
        elif isinstance(f, openmm.HarmonicAngleForce):
            g = openmm.HarmonicAngleForce()
            for k in range(f.getNumAngles()):
                i, j, m, a, kk = f.getAngleParameters(k)
                if {i, j, m} & atoms:
                    g.addAngle(i, j, m, a, kk)
        elif isinstance(f, openmm.PeriodicTorsionForce):
            g = openmm.PeriodicTorsionForce()
            for k in range(f.getNumTorsions()):
                i, j, m, n, per, ph, kk = f.getTorsionParameters(k)
                if {i, j, m, n} & atoms:
                    g.addTorsion(i, j, m, n, per, ph, kk)
        else:
            continue
        out.addForce(g)
    return _energy(out, xyz)[1]


@pytest.mark.parametrize("method", ["NoCutoff", "PME"])
def test_a_residue_mapped_onto_itself_changes_nothing(method):
    p, system, xyz = _system("ASA", method)
    core, env = _mapping(p, p)
    h = hybrid.build(system, system, xyz, xyz, core, env)
    assert h.system.getNumParticles() == system.getNumParticles() and not h.unique_new
    _, e = _energy(system, xyz)
    for x in (0.0, 0.25, 0.5, 0.8, 1.0):
        assert _hybrid_energy(h, x) == pytest.approx(e, rel=1e-6, abs=1e-4)


@pytest.mark.parametrize("method", ["NoCutoff", "PME"])
def test_the_end_states_are_the_two_systems_and_their_dummies(method):
    a, sa, xa = _system("AAA", method)
    b, sb, xb = _system("ASA", method)
    core, env = _mapping(a, b)
    h = hybrid.build(sa, sb, xa, xb, core, env)
    names = lambda s, ks: sorted(str(s.atoms["name"][k]) for k in ks)  # noqa: E731
    assert names(a, h.unique_old) == ["HB1"]
    unique_new = [k for k in range(b.natoms) if h.new_to_hybrid[k] >= a.natoms]
    assert names(b, unique_new) == ["HG", "OG"]
    # each end state's own coordinates, as the hybrid holds them
    x_old = h.positions[h.old_to_hybrid]
    x_new = h.positions[h.new_to_hybrid]
    _, e_old = _energy(sa, x_old)
    _, e_new = _energy(sb, x_new)
    dummies_new = _valence(sb, unique_new, x_new)
    dummies_old = _valence(sa, h.unique_old, x_old)
    assert _hybrid_energy(h, 0.0) == pytest.approx(e_old + dummies_new, rel=1e-6, abs=1e-3)
    assert _hybrid_energy(h, 1.0) == pytest.approx(e_new + dummies_old, rel=1e-6, abs=1e-3)
    # in between: finite and smooth
    es = [_hybrid_energy(h, x) for x in lambdas.windows(11)]
    assert np.all(np.isfinite(es)) and np.max(np.abs(np.diff(es))) < 1e4


def test_masses_and_order():
    a, sa, xa = _system("AAA")
    b, sb, xb = _system("ASA")
    core, env = _mapping(a, b)
    h = hybrid.build(sa, sb, xa, xb, core, env)
    n = sa.getNumParticles()
    assert h.system.getNumParticles() == n + 2
    for k, j in core.items():  # the mean of the two
        m = h.system.getParticleMass(k).value_in_unit(openmm.unit.dalton)
        want = 0.5 * (sa.getParticleMass(k) + sb.getParticleMass(j))
        want = want.value_in_unit(openmm.unit.dalton)
        assert m == pytest.approx(want)
    np.testing.assert_allclose(h.positions[:n], xa)
    with pytest.raises(ValueError, match="both"):
        hybrid.build(sa, sb, xa, xb, core, {**env, next(iter(core)): 0})
    # a long-range dispersion correction would miss the ligand: refused
    _, with_lrc, _ = boonza.to_openmm(a, nonbonded_method="PME", cutoff=12.0, constraints=False)
    if any(isinstance(f, openmm.NonbondedForce) and f.getUseDispersionCorrection()
           for f in with_lrc.getForces()):  # fmt: skip
        with pytest.raises(ValueError, match="dispersion"):
            hybrid.build(with_lrc, with_lrc, xa, xa, *_mapping(a, a))
