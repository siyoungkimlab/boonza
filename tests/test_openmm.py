import numpy as np
import pytest
from conftest import msys_file, run_msys

import boonza
from boonza.omm import KCAL, from_openmm, openmm_energies, to_openmm

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

GROUPS = {
    "HarmonicBondForce": "bond", "stretch_harm": "bond",
    "HarmonicAngleForce": "angle", "angle_harm": "angle",
    "PeriodicTorsionForce": "torsion", "dihedral_trig": "torsion",
    "dihedral_trig_constant": "torsion",
    "NonbondedForce": "nonbonded", "nonbonded": "nonbonded", "pair_12_6_es": "nonbonded",
    "nonbonded_vdw": "nonbonded",
}  # fmt: skip


def _energies(omm_system, positions):
    """kcal/mol per category, total included."""
    for k, f in enumerate(omm_system.getForces()):
        f.setForceGroup(k)
    ctx = openmm.Context(omm_system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))  # fmt: skip
    ctx.setPositions(positions)
    ctx.computeVirtualSites()
    out = {"total": 0.0}
    for k, f in enumerate(omm_system.getForces()):
        e = ctx.getState(getEnergy=True, groups={k}).getPotentialEnergy()
        e = e.value_in_unit(unit.kilojoule_per_mole) / KCAL
        name = GROUPS.get(f.getName(), GROUPS.get(type(f).__name__, type(f).__name__))
        out[name] = out.get(name, 0.0) + e
        out["total"] += e
    return out


def _amber(**kwargs):
    prm = app.AmberPrmtopFile(str(msys_file("sys.prmtop")))
    crd = app.AmberInpcrdFile(str(msys_file("eq.rst")))
    omm = prm.createSystem(nonbondedMethod=app.NoCutoff, removeCMMotion=False, **kwargs)
    return prm.topology, omm, crd.positions


def _assert_close(ours, ref, rtol=1e-6):
    assert ours.keys() == ref.keys()
    for k in ref:
        assert ours[k] == pytest.approx(ref[k], rel=rtol, abs=1e-6), k


def test_dms_from_prmtop_matches_amber(tmp_path):
    out = tmp_path / "sys.dms"
    run_msys("prmtop", msys_file("sys.prmtop"), msys_file("eq.rst"), out)
    s = boonza.load(out)
    top, omm, pos = to_openmm(s, constraints=False)
    _, ref_sys, ref_pos = _amber(constraints=None, rigidWater=False)
    _assert_close(_energies(omm, pos), _energies(ref_sys, ref_pos), rtol=2e-5)


@pytest.mark.parametrize("via_dms", [False, True])
def test_openmm_roundtrip(via_dms, tmp_path):
    top, ref_sys, pos = _amber(constraints=None, rigidWater=False)
    s = from_openmm(top, ref_sys, pos)
    if via_dms:
        boonza.save(s, tmp_path / "rt.dms")
        s = boonza.load(tmp_path / "rt.dms")
    _, omm, opos = to_openmm(s, constraints=False)
    _assert_close(_energies(omm, opos), _energies(ref_sys, pos), rtol=1e-9)
    e = openmm_energies(s, constraints=False)
    assert e["total"] == pytest.approx(_energies(ref_sys, pos)["total"], rel=1e-9)


def test_constraints_and_rigid_water():
    top, ref_sys, pos = _amber(constraints=app.HBonds, rigidWater=True)
    s = from_openmm(top, ref_sys, pos)
    assert "constraint_hoh" in s.tables
    _, omm, _ = to_openmm(s, constraints=True)

    def cons(system):
        out = {}
        for k in range(system.getNumConstraints()):
            i, j, d = system.getConstraintParameters(k)
            out[(min(i, j), max(i, j))] = d.value_in_unit(unit.nanometer)
        return out

    ours, ref = cons(omm), cons(ref_sys)
    assert ours.keys() == ref.keys()
    for key in ref:
        assert ours[key] == pytest.approx(ref[key], rel=1e-9)
    assert omm.getForce(0).getNumBonds() < ref_sys.getForce(0).getNumBonds() + 1


def test_tip4p_virtual_sites():
    ff = app.ForceField("tip4pew.xml")
    pdb = """HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
HETATM    2  H1  HOH A   1       0.957   0.000   0.000  1.00  0.00           H
HETATM    3  H2  HOH A   1      -0.240   0.927   0.000  1.00  0.00           H
HETATM    4  O   HOH A   2       3.000   0.000   0.500  1.00  0.00           O
HETATM    5  H1  HOH A   2       3.957   0.000   0.500  1.00  0.00           H
HETATM    6  H2  HOH A   2       2.760   0.927   0.500  1.00  0.00           H
END
"""
    import io

    struct = app.PDBFile(io.StringIO(pdb))
    modeller = app.Modeller(struct.topology, struct.positions)
    modeller.addExtraParticles(ff)
    ref_sys = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, rigidWater=False,
                              removeCMMotion=False)  # fmt: skip
    s = from_openmm(modeller.topology, ref_sys, modeller.positions)
    assert "virtual_out3" in s.tables or "virtual_lc3" in s.tables
    _, omm, pos = to_openmm(s, constraints=False)
    _assert_close(_energies(omm, pos), _energies(ref_sys, modeller.positions), rtol=1e-9)


def _lj_coulomb(pos, q, sig, eps, rule, overrides, types):
    e = 0.0
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            key = tuple(sorted((types[i], types[j])))
            if key in overrides:
                s, ep = overrides[key]
            elif rule == "geometric":
                s, ep = np.sqrt(sig[i] * sig[j]), np.sqrt(eps[i] * eps[j])
            else:
                s, ep = 0.5 * (sig[i] + sig[j]), np.sqrt(eps[i] * eps[j])
            e += 4 * ep * ((s / r) ** 12 - (s / r) ** 6) + 332.0636930 * q[i] * q[j] / r
    return e


@pytest.mark.parametrize("rule", ["geometric", "arithmetic/geometric"])
def test_combining_rules_and_nbfix(rule):
    rng = np.random.default_rng(3)
    s = boonza.System()
    s.add_atoms(6, name=[f"A{i}" for i in range(6)], anum=6, mass=12.0,
                charge=[0.3, -0.3, 0.2, -0.2, 0.1, -0.1],
                pos=rng.uniform(0, 8, (6, 3)))  # fmt: skip
    nb = s.add_nonbonded_from_schema("vdw_12_6", rule)
    p = [nb.params.add_param(sigma=3.0 + 0.2 * k, epsilon=0.1 + 0.05 * k) for k in range(3)]
    types = [0, 1, 2, 0, 1, 2]
    nb.add_terms(np.arange(6).reshape(-1, 1), [p[t] for t in types])
    nb.overrides.set(p[0], p[2], sigma=3.3, epsilon=0.4)
    e = openmm_energies(s)
    sig = [3.0 + 0.2 * t for t in types]
    eps = [0.1 + 0.05 * t for t in types]
    ref = _lj_coulomb(s.positions, s.atoms["charge"], sig, eps, rule, {(0, 2): (3.3, 0.4)}, types)
    # OpenMM's Coulomb constant differs from 332.0637 in the 7th digit
    assert e["total"] == pytest.approx(ref, rel=1e-5)
