"""GROMACS topologies: energies vs OpenMM's GromacsTopFile, structure vs MDAnalysis's ITPParser.

tests/data/gmx_amber was written by ParmEd from msys's sys.prmtop and eq.rst, so the
same system can also be compared with its Amber original.
"""

from pathlib import Path

import numpy as np
import pytest
from conftest import msys_file

import boonza
from boonza.io.amber import read_amber_coordinates
from boonza.io.gromacs import GromacsError, load_top

HERE = Path(__file__).parent
GMX = HERE / "data" / "gmx_amber"
MDA_DATA = Path("~/mdanalysis/testsuite/MDAnalysisTests/data").expanduser()


def _openmm_total(omm_system, positions_nm):
    mm = pytest.importorskip("openmm")
    unit = pytest.importorskip("openmm.unit")
    ctx = mm.Context(omm_system, mm.VerletIntegrator(0.001),
                     mm.Platform.getPlatformByName("Reference"))  # fmt: skip
    ctx.setPositions(positions_nm * unit.nanometer)
    e = ctx.getState(getEnergy=True).getPotentialEnergy()
    return e.value_in_unit(unit.kilocalorie_per_mole)


def test_energies_match_openmm_gromacs_reader():
    app = pytest.importorskip("openmm.app")
    s = load_top(GMX / "topol.top", coordinates=GMX / "conf.gro")
    assert s.natoms == 2681 and s.nresidues == 879
    pos, _, _ = read_amber_coordinates(msys_file("eq.rst"), s.natoms)  # full precision
    s.positions = pos
    ours = boonza.openmm_energies(s, nonbonded_method="NoCutoff", constraints=False)["total"]
    top = app.GromacsTopFile(str(GMX / "topol.top"))
    omm = top.createSystem(nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False,
                           removeCMMotion=False)  # fmt: skip
    assert ours == pytest.approx(_openmm_total(omm, pos / 10.0), rel=1e-7)


def test_same_system_as_the_amber_original():
    pytest.importorskip("openmm")
    gmx = load_top(GMX / "topol.top")
    amber = boonza.load(msys_file("sys.prmtop"), coordinates=msys_file("eq.rst"))
    gmx.positions = amber.positions
    np.testing.assert_allclose(gmx.atoms["charge"], amber.atoms["charge"], atol=1e-7)
    np.testing.assert_array_equal(gmx.atoms["anum"], amber.atoms["anum"])
    assert gmx.nbonds == amber.nbonds
    e_gmx = boonza.openmm_energies(gmx, nonbonded_method="NoCutoff", constraints=False)
    e_amber = boonza.openmm_energies(amber, nonbonded_method="NoCutoff", constraints=False)
    # ParmEd writes fudgeQQ as 0.83333333 and water without angle terms (settles), so
    # compare everything but the rigid-water bonded terms at a loose tolerance
    assert e_gmx["total"] == pytest.approx(e_amber["total"] - _water_bonded(amber), rel=1e-5)


def _water_bonded(amber):
    """Bond and angle energy of the waters in the Amber system (GROMACS makes them rigid)."""
    water = amber.clone("water")
    e = boonza.openmm_energies(water, nonbonded_method="NoCutoff", constraints=False)
    return sum(v for k, v in e.items() if k in ("stretch_harm", "angle_harm"))


def test_defines_select_flexible_water():
    rigid = load_top(GMX / "topol.top")
    flexible = load_top(GMX / "topol.top", defines={"FLEXIBLE": ""})
    assert "constraint_hoh" in rigid.tables and "constraint_hoh" not in flexible.tables
    water_bonds = flexible.nbonds - rigid.nbonds  # the H-H bond of each water
    assert water_bonds == 876 == len(rigid.select("water and name O"))


def test_gromos_structure_matches_mdanalysis():
    mda = pytest.importorskip("MDAnalysis")
    top = MDA_DATA / "gromacs_ala10.top"
    if not top.exists():
        pytest.skip(f"{top} not found")
    include = [MDA_DATA / "gromacs"]
    s = load_top(top, include_dirs=include, structure_only=True)
    u = mda.Universe(str(top), topology_format="ITP", include_dir=str(include[0]))
    a = u.atoms
    assert s.natoms == len(a) == 135
    np.testing.assert_array_equal(s.atoms["name"], a.names)
    np.testing.assert_array_equal(s.atoms["type"], a.types)
    np.testing.assert_allclose(s.atoms["charge"], a.charges, atol=1e-6)
    np.testing.assert_allclose(s.atoms["mass"], a.masses, atol=1e-6)
    res = s.atoms["residue"]
    np.testing.assert_array_equal(s.residues["resid"][res], a.resids)
    np.testing.assert_array_equal(s.residues["name"][res], a.resnames)
    ours = {
        tuple(sorted(p)) for p in zip(s.bonds["i"].tolist(), s.bonds["j"].tolist(), strict=True)
    }
    assert ours == {tuple(sorted(map(int, p))) for p in u.bonds.indices}
    assert len(set(s.fragids)) == len(set(a.molnums))
    with pytest.raises(GromacsError, match="bond function type 2"):
        load_top(top, include_dirs=include)  # GROMOS quartic bonds have no msys table
