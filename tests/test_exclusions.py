import numpy as np
import pytest
from conftest import msys_file, run_msys

import boonza
from boonza.diff import diff
from boonza.exclusions import update_exclusions


@pytest.fixture(scope="module")
def amber(tmp_path_factory):
    out = tmp_path_factory.mktemp("excl") / "sys.dms"
    run_msys("prmtop", msys_file("sys.prmtop"), msys_file("eq.rst"), out)
    return boonza.load(out)


def _pairs(t):
    return set(map(tuple, np.sort(t.atoms, axis=1).tolist()))


def test_regenerates_amber_exclusions_and_14_pairs(amber):
    s = amber.copy()
    s.del_table("exclusion")
    s.del_table("pair_12_6_es")
    update_exclusions(s, pair_scales=(0.5, 1 / 1.2))
    assert diff(amber, s, rtol=1e-5, tables=["exclusion", "pair_12_6_es"]) == []
    pytest.importorskip("openmm")
    from boonza.omm import openmm_energies

    ref = openmm_energies(amber, constraints=False)["total"]
    assert openmm_energies(s, constraints=False)["total"] == pytest.approx(ref, rel=1e-6)


def test_exclusions_follow_bond_edits(amber):
    s = amber.clone(structure_only=True)
    update_exclusions(s)
    assert _pairs(s.tables["exclusion"]) == _pairs(amber.tables["exclusion"])
    assert "pair_12_6_es" not in s.tables
    water = s.select("water").ids[:6]  # two waters, O H H O H H
    s.add_bond(water[0], water[3])
    update_exclusions(s, separation=2)
    new = _pairs(s.tables["exclusion"])
    assert (int(water[0]), int(water[3])) in new  # 1-2
    assert (int(water[1]), int(water[3])) in new  # 1-3 through the new bond
    assert (int(water[1]), int(water[4])) not in new  # 1-4 excluded only with separation 3


def test_virtual_sites_excluded_like_their_host():
    openmm = pytest.importorskip("openmm")
    import io

    from openmm import app

    from boonza.omm import from_openmm

    pdb = """HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
HETATM    2  H1  HOH A   1       0.957   0.000   0.000  1.00  0.00           H
HETATM    3  H2  HOH A   1      -0.240   0.927   0.000  1.00  0.00           H
HETATM    4  O   HOH A   2       3.000   0.000   0.500  1.00  0.00           O
HETATM    5  H1  HOH A   2       3.957   0.000   0.500  1.00  0.00           H
HETATM    6  H2  HOH A   2       2.760   0.927   0.500  1.00  0.00           H
END
"""
    ff = app.ForceField("tip4pew.xml")
    struct = app.PDBFile(io.StringIO(pdb))
    model = app.Modeller(struct.topology, struct.positions)
    model.addExtraParticles(ff)
    system = ff.createSystem(model.topology, nonbondedMethod=app.NoCutoff, rigidWater=False)
    assert any(isinstance(f, openmm.NonbondedForce) for f in system.getForces())
    s = from_openmm(model.topology, system, model.positions)
    want = _pairs(s.tables["exclusion"])
    assert len(want) == 12  # all six pairs within each water, M sites included
    update_exclusions(s)
    assert _pairs(s.tables["exclusion"]) == want
