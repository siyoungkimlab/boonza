import math

import numpy as np
import pytest
from conftest import msys_file, run_msys

import boonza
from boonza.describe import describe


@pytest.fixture(scope="module")
def amber(tmp_path_factory):
    out = tmp_path_factory.mktemp("ff") / "sys.dms"
    run_msys("prmtop", msys_file("sys.prmtop"), msys_file("eq.rst"), out)
    return boonza.load(out)


def test_atoms_match_openmm_prmtop(amber):
    openmm = pytest.importorskip("openmm")
    from openmm import app, unit

    prm = app.AmberPrmtopFile(str(msys_file("sys.prmtop")))
    omm = prm.createSystem()  # keep the System alive: forces are proxies into it
    nb = next(f for f in omm.getForces() if isinstance(f, openmm.NonbondedForce))
    rep = describe(amber, "residue 0")
    assert [r["index"] for r in rep.atoms] == amber.select("residue 0").ids.tolist()
    for row in rep.atoms:
        q, sig, eps = nb.getParticleParameters(row["index"])
        assert row["charge"] == pytest.approx(q.value_in_unit(unit.elementary_charge), abs=1e-6)
        assert row["epsilon"] == pytest.approx(
            eps.value_in_unit(unit.kilocalorie_per_mole), rel=1e-5
        )
        if row["epsilon"] > 0:
            assert row["sigma"] == pytest.approx(sig.value_in_unit(unit.angstrom), rel=1e-5)


def test_amber_14_scaling(amber):
    rep = describe(amber, "residue 0", pairs=True)
    n = len(rep.atoms)
    assert len(rep.pairs) == n * (n - 1) // 2
    scaled = [p for p in rep.pairs if "qij" in p]
    assert scaled and all(p["excluded"] for p in scaled)
    for p in scaled:
        if p["es_scale"] is not None:
            assert p["es_scale"] == pytest.approx(1 / 1.2, rel=1e-4)
        if p["lj_scale"] is not None:
            assert p["lj_scale"] == pytest.approx(0.5, rel=1e-4)
            assert p["sigma_pair"] == pytest.approx(p["sigma"], rel=1e-4)
    # 1-2 and 1-3 pairs are excluded without a pair term
    ca = amber.select("residue 0 and name CA").ids[0]
    for p in rep.pairs:
        if ca in (p["i"], p["j"]) and amber.find_bond(p["i"], p["j"]) is not None:
            assert p["excluded"] and "qij" not in p


def test_terms_any_and_all(amber):
    ca = int(amber.select("residue 0 and name CA").ids[0])
    rep = describe(amber, [ca])
    for name in ("stretch_harm", "angle_harm", "dihedral_trig"):
        assert len(rep.terms[name]) == len(amber.tables[name].find_with_any([ca]))
        assert all(ca in r["atoms"] for r in rep.terms[name])
    partners = {a for r in rep.terms["stretch_harm"] for a in r["atoms"]} - {ca}
    assert sorted(partners) == sorted(amber.bonded_atoms(ca).tolist())
    residue = set(amber.select("residue 0").ids.tolist())
    rep_all = describe(amber, "residue 0", terms="all")
    for rows in rep_all.terms.values():
        assert all(set(r["atoms"]) <= residue for r in rows)
    assert len(rep_all.terms["stretch_harm"]) < len(
        describe(amber, "residue 0").terms["stretch_harm"]
    )
    text = str(rep)
    assert "stretch_harm" in text and "E = fc (r - r0)^2" in text and "ARG1:CA" in text
    assert "fc6" not in text  # all-zero Fourier columns are hidden


def test_nbfix_and_geometric():
    s = boonza.System()
    s.add_atoms(3, name=["A", "B", "C"], anum=6, mass=12.0, charge=[0.5, -0.5, 0.2])
    nb = s.add_nonbonded_from_schema("vdw_12_6", "geometric")
    p = [nb.params.add_param(sigma=sg, epsilon=ep) for sg, ep in ((3, 0.1), (4, 0.4), (5, 0.9))]
    nb.add_terms([[0], [1], [2]], p)
    nb.overrides.set(p[0], p[1], sigma=3.3, epsilon=0.25)
    rep = s.all.describe()
    assert rep.nonbonded_info["vdw_rule"] == "geometric"
    pairs = {(r["i"], r["j"]): r for r in rep.pairs}
    fix = pairs[(0, 1)]
    assert fix["nbfix"] and (fix["sigma"], fix["epsilon"]) == (3.3, 0.25)
    assert not pairs[(0, 2)]["nbfix"]
    assert pairs[(0, 2)]["sigma"] == pytest.approx(math.sqrt(15))
    assert pairs[(0, 2)]["epsilon"] == pytest.approx(0.3)
    assert pairs[(1, 2)]["qq"] == pytest.approx(-0.1)
    assert not any(r["excluded"] for r in rep.pairs)
    assert "nbfix" in str(rep) and "combining rule geometric" in str(rep)


def test_system_method_and_large_selection(amber):
    rep = amber.describe("water")
    assert len(rep.atoms) == len(amber.select("water")) and rep.pairs == []
    assert np.all([r["param"] >= 0 for r in rep.atoms])
    with pytest.raises(ValueError):
        describe(amber, "water", terms="some")
