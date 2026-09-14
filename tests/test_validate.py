import numpy as np
import pytest
from conftest import msys_file, run_msys

import boonza
from boonza.validate import find_knots, line_intersects_triangle, validate


def _canon(knots):
    return sorted([list(ring), list(bond), idx] for ring, bond, idx in knots)


def test_line_intersects_triangle():
    t0, t1, t2 = [0.0, 0.0, 0.0], [0.0, 3.0, 3.0], [0.0, 2.0, 0.0]
    assert not line_intersects_triangle([-1.0, 1.0, 1.0], [1.0, 0.5, 1.0], t0, t1, t2)
    assert line_intersects_triangle([-1.0, 1.0, 1.0], [1.0, 1.5, 1.0], t0, t1, t2)


@pytest.mark.parametrize("name,count", [("knot.mae", 2), ("jandor.sdf", 0),
                                        ("excluded_knot.dms", 1)])  # fmt: skip
def test_knots_match_msys(name, count):
    path = msys_file(name)
    s = boonza.load(path, structure_only=True) if name.endswith(".dms") else boonza.load(path)
    ours = find_knots(s)
    assert len(ours) == count
    assert _canon(ours) == _canon(run_msys("knots", path, "0", "all", "0"))


@pytest.mark.parametrize("name", ["knot.mae", "ww.dms"])
def test_wrap_matches_msys_wrapper(name, tmp_path):
    from boonza.validate import _msys_wrap

    path = msys_file(name)
    s = boonza.load(path)
    pos = s.positions + np.random.default_rng(1).uniform(-30, 30, 3)
    np.save(tmp_path / "pos.npy", pos)
    ref = np.array(run_msys("wrap", path, tmp_path / "pos.npy", "", "[]"))
    np.testing.assert_allclose(_msys_wrap(s)(pos, s.cell), ref, atol=1e-4)


def test_ignore_excluded_knots(tmp_path):
    s = boonza.load(msys_file("knot.mae"))
    with pytest.raises(ValueError):
        find_knots(s, ignore_excluded_knots=True)
    s.add_table_from_schema("exclusion").add_terms(np.column_stack(np.triu_indices(s.natoms, 1)))
    assert find_knots(s, ignore_excluded_knots=True) == []
    assert len(find_knots(s)) == 2
    path = tmp_path / "knot.dms"
    boonza.save(s, path)
    assert run_msys("knots", path, "0", "all", "1") == []
    assert {p.check for p in validate(boonza.load(msys_file("knot.mae")))} >= {"knot"}


def test_validate_amber(tmp_path):
    out = tmp_path / "sys.dms"
    run_msys("prmtop", msys_file("sys.prmtop"), msys_file("eq.rst"), out)
    s = boonza.load(out)
    assert validate(s) == []
    strict = {p.check for p in validate(s, strict=True)}
    assert "constraints" in strict  # a prmtop conversion has no constraint tables
    assert strict.isdisjoint({"exclusions", "contacts", "masses", "waters", "cell"})
    s.tables["exclusion"].delete_terms([0])
    assert "exclusions" in {p.check for p in validate(s, strict=True)}


def test_synthetic_problems():
    s = boonza.System()
    s.add_atoms(4, name=["C", "H", "V", "X"], anum=[6, 1, 0, 6], mass=[12.0, 1.008, 0.0, 0.0],
                pos=[[0, 0, 0], [1.09, 0, 0], [0.5, 0, 0], [0.3, 0, 0]])  # fmt: skip
    s.cell = np.diag([20.0, 20.0, 20.0])
    s.add_bond(0, 1)
    s.add_bond(0, 2)
    nb = s.add_nonbonded_from_schema("vdw_12_6", "arithmetic/geometric")
    nb.add_terms([[0], [1], [2]], nb.params.add_param(sigma=3.0, epsilon=0.1))
    for name in ("virtual_lc2", "virtual_midpoint"):
        t = s.add_table_from_schema(name)
        t.add_term([2, 0, 1], t.params.add_param(c1=0.5))
    checks = {p.check: p for p in validate(s)}
    assert set(checks) == {"nonbonded", "mass", "virtual", "stretch"}
    assert checks["stretch"].atoms == (0, 1)  # the C-V bond involves a pseudo particle
    assert checks["nonbonded"].atoms == (3,) and checks["mass"].atoms == (3,)
    assert checks["virtual"].atoms == (2,)
    strict = {p.check: p for p in validate(s, strict=True)}
    assert {"constraints", "constrained_hydrogens", "contacts", "masses"} <= set(strict)
    assert strict["contacts"].atoms == (0, 1, 2, 3)  # H-V at 0.59 Å is not bonded
    assert "3" in str(strict["contacts"])


def _amber(tmp_path):
    out = tmp_path / "sys.dms"
    run_msys("prmtop", msys_file("sys.prmtop"), msys_file("eq.rst"), out)
    return boonza.load(out)


def test_force_field_consistency(tmp_path):
    s = _amber(tmp_path)
    assert validate(s) == []
    strict = {p.check for p in validate(s, strict=True)}
    assert strict.isdisjoint({"term_topology", "extra_exclusions", "stretch", "charge"})

    c = s.copy()  # a bond loses its stretch term
    t = c.tables["stretch_harm"]
    lost = tuple(sorted(t.atoms[0].tolist()))
    t.delete_terms([0])
    found = {p.check: p for p in validate(c)}
    assert found["stretch"].atoms == lost

    c = s.copy()  # one charge off by 0.1 e
    q = c.atoms["charge"].copy()
    q[5] += 0.1
    c.atoms["charge"] = q
    found = {p.check: p for p in validate(c)}
    assert 5 in found["charge"].atoms and "+0.1" in found["charge"].message

    c = s.copy()  # an exclusion between distant atoms, an angle over non-bonded atoms
    prot = c.select("protein").ids
    c.tables["exclusion"].add_terms([[int(prot[0]), int(prot[-1])]])
    angle = c.tables["angle_harm"]
    angle.add_term([int(prot[0]), int(prot[10]), int(prot[20])], angle.param_ids[0])
    found = {p.check: p for p in validate(c, strict=True)}
    assert found["extra_exclusions"].atoms == (int(prot[0]), int(prot[-1]))
    assert "angle_harm: 1 terms" in found["term_topology"].message


def test_virtual_site_charge_counts_with_host():
    pytest.importorskip("openmm")
    import io

    from openmm import app

    from boonza.omm import from_openmm

    pdb = """HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
HETATM    2  H1  HOH A   1       0.957   0.000   0.000  1.00  0.00           H
HETATM    3  H2  HOH A   1      -0.240   0.927   0.000  1.00  0.00           H
END
"""
    ff = app.ForceField("tip4pew.xml")
    struct = app.PDBFile(io.StringIO(pdb))
    model = app.Modeller(struct.topology, struct.positions)
    model.addExtraParticles(ff)
    system = ff.createSystem(model.topology, nonbondedMethod=app.NoCutoff, rigidWater=False)
    s = from_openmm(model.topology, system, model.positions)
    assert s.nfragments == 2  # the M site is not bonded
    assert "charge" not in {p.check for p in validate(s)}
