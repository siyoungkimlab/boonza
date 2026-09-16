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


def _cage(cavities, spacing=2.2, half_z=6.0, edge=28.0):
    """A carbon lattice slab (bonded into rows, so every atom is apolar) with
    spherical cavities carved out of it: the voids a membrane's tails leave."""
    s = boonza.System("cage")
    chain = s.add_chain()
    residue = s.add_residue(chain, name="CAG")
    axis = np.arange(-edge / 2 + spacing / 2, edge / 2, spacing)
    zs = np.arange(-half_z, half_z + 1e-9, spacing)
    for y in axis:
        for z in zs:
            previous = None
            for x in axis:
                p = np.array([x, y, z])
                if any(np.linalg.norm(p - c) < r for c, r in cavities):
                    previous = None
                    continue
                a = s.add_atom(residue, name="C", anum=6, pos=p)
                if previous is not None:
                    s.add_bond(previous, a)
                previous = a
    s.cell = np.diag([edge, edge, 40.0])
    return s


def test_buried_water_leaves_greasy_voids_dry():
    bare, pocket = np.array([-7.0, 0.0, 0.0]), np.array([7.0, 0.0, 0.0])
    peptide = boonza.peptide("AA")
    peptide.positions = peptide.positions - peptide.positions.mean(0) + pocket
    cage = _cage([(bare, 4.5), (pocket, 9.0)])
    cage.append(peptide)  # a protein in the second cavity: its water is its business
    box = np.diag(np.asarray(cage.cell, float)).copy()
    wet = boonza.solvate(cage, box=box, center_selection="none")
    dry = boonza.solvate(cage, box=box, center_selection="none", remove_buried=True)

    def near(s, point, r):
        o = s.positions[s.select("water and oxygen").ids]
        return int((np.linalg.norm(o - point, axis=1) < r).sum())

    assert near(wet, bare, 4.5) > 0  # water is tiled into the bare void ...
    assert near(dry, bare, 4.5) == 0  # ... and taken out again
    assert near(dry, pocket, 9.0) == near(wet, pocket, 9.0) > 0  # the peptide keeps its water
    far = lambda s: (np.abs(s.positions[s.select("water and oxygen").ids][:, 2]) > 10).sum()  # noqa: E731
    assert far(dry) == far(wet) > 100  # bulk water is untouched


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
