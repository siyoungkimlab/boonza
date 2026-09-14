"""solvate, neutralize and hydrogen mass repartitioning vs msys's own tools."""

import numpy as np
import pytest
from conftest import msys_file, run_msys

import boonza
from boonza.build import WATER_BOX
from boonza.elements import MSYS_MASSES


def _solute(tmp_path):
    # structure only: msys cannot save a system where the added water lacks the
    # solute's force field
    protein = boonza.load(msys_file("ww.dms"), structure_only=True).clone("protein")
    path = tmp_path / "solute.dms"
    boonza.save(protein, path)
    return path


def _msys_solvated(tmp_path, box="46,48,50"):
    out = tmp_path / "msys_solvated.dms"
    run_msys("solvate", _solute(tmp_path), WATER_BOX, box, out)
    return out


def _no_differences(ours, ref):
    return [str(d) for d in boonza.diff(ours, ref)]


def test_solvate_matches_msys(tmp_path):
    ref = boonza.load(_msys_solvated(tmp_path))
    ours = boonza.solvate(boonza.load(tmp_path / "solute.dms"), box=(46, 48, 50))
    assert _no_differences(ours, ref) == []
    assert ours.natoms > 3000 and np.allclose(np.diag(ours.cell), [46, 48, 50])
    names = ours.chains["name"]
    assert names[-1].startswith("W") and ours.residues["resid"][-1] >= 1


def test_solvate_box_from_thickness():
    protein = boonza.load(msys_file("ww.dms")).clone("protein")
    extent = np.ptp(protein.positions, axis=0).max()
    s = boonza.solvate(protein, thickness=8.0)
    np.testing.assert_allclose(np.diag(s.cell), extent + 16.0)
    oxygens = s.positions[s.select("water and oxygen").ids]
    solute = s.positions[s.select("protein").ids]
    d = boonza.pbc.distances(oxygens, solute, s.cell).min()
    assert d > 2.4


@pytest.mark.parametrize(("concentration", "charge"), [(0.0, "charge"), (0.15, "charge")])
def test_neutralize_matches_msys(tmp_path, concentration, charge):
    solvated = _msys_solvated(tmp_path)
    out = tmp_path / "msys_ions.dms"
    run_msys("neutralize", solvated, concentration, 3, charge, out)
    ref = boonza.load(out)
    ions = ref.select(f"ct {ref.ncts - 1}").ids
    mass = ref.atoms["mass"].copy()
    mass[ions] = MSYS_MASSES[ref.atoms["anum"][ions]]  # msys leaves ions massless
    ref.atoms["mass"] = mass
    ours = boonza.neutralize(boonza.load(solvated), concentration=concentration, charge=charge,
                             random_seed=3)  # fmt: skip
    assert _no_differences(ours, ref) == []
    q = ours.atoms["charge"].sum()
    assert abs(q) < 0.5  # neutral up to the solute's fractional charge
    if concentration:
        assert len(ours.select("name Na")) > 1 and len(ours.select("name Cl")) > 1


def test_repartition_matches_msys(tmp_path):
    path = msys_file("ww.dms")
    out = tmp_path / "msys_hmr.dms"
    run_msys("hmr", path, "not water", 3.024, "1", out)
    s = boonza.load(path)
    ours = boonza.repartition_hydrogen_masses(s, "not water", 3.024)
    np.testing.assert_allclose(ours.atoms["mass"], boonza.load(out).atoms["mass"], atol=1e-9)
    assert ours.atoms["mass"].sum() == pytest.approx(s.atoms["mass"].sum())
    deut = boonza.repartition_hydrogen_masses(s, "all", 2.014, repartition=False)
    h = s.select("hydrogen").ids
    assert np.allclose(deut.atoms["mass"][h], 2.014)
