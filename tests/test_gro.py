import warnings
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.io.pdb import lengths_angles_from_cell

mda = pytest.importorskip("MDAnalysis")

DATA = Path("~/mdanalysis/testsuite/MDAnalysisTests/data").expanduser()
FILES = [
    "adk_oplsaa.gro",
    "sample_velocity_file.gro",
    "two_water_gro.gro",
    "two_water_gro_widebox.gro",
    "residwrap.gro",
    "residwrap_0base.gro",
    "grovels.gro",
    "sameresid_diffresname.gro",
    "coordinates/test.gro",
]


def _data(name):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"MDAnalysis test file {name} not found")
    return path


def _universe(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mda.Universe(str(path))


@pytest.mark.parametrize("name", FILES)
def test_matches_mdanalysis(name):
    path = _data(name)
    s = boonza.load(path)
    u = _universe(path)
    assert s.natoms == len(u.atoms)
    np.testing.assert_allclose(s.positions, u.atoms.positions, rtol=1e-6, atol=1e-4)
    assert s.atoms["name"].tolist() == u.atoms.names.tolist()
    assert s.residues["name"][s.atoms["residue"]].tolist() == u.atoms.resnames.tolist()
    assert s.residues["resid"][s.atoms["residue"]].tolist() == u.atoms.resids.tolist()
    assert s.nresidues == len(u.residues)
    if u.dimensions is not None:
        np.testing.assert_allclose(lengths_angles_from_cell(s.cell), u.dimensions,
                                   rtol=1e-5, atol=1e-3)  # fmt: skip
    else:
        assert not s.cell.any()
    if hasattr(u.atoms, "velocities"):
        try:
            v = u.atoms.velocities
        except mda.exceptions.NoDataError:
            v = np.zeros((s.natoms, 3))
        np.testing.assert_allclose(s.velocities, v, rtol=1e-6, atol=1e-4)


@pytest.mark.parametrize("name", ["adk_oplsaa.gro", "grovels.gro", "two_water_gro_widebox.gro"])
def test_write_roundtrip(name, tmp_path):
    src = boonza.load(_data(name))
    out = tmp_path / "out.gro"
    boonza.save(src, out)
    back = boonza.load(out)
    np.testing.assert_allclose(back.positions, src.positions, atol=1e-6)
    np.testing.assert_allclose(back.velocities, src.velocities, atol=1e-6)
    np.testing.assert_allclose(back.cell, src.cell, atol=1e-6)
    assert back.atoms["name"].tolist() == src.atoms["name"].tolist()
    assert back.residues["resid"].tolist() == src.residues["resid"].tolist()
    assert back.nbonds == src.nbonds
    u = _universe(out)  # MDAnalysis reads our file too
    np.testing.assert_allclose(u.atoms.positions, src.positions, rtol=1e-6, atol=1e-4)


def test_bonds_make_molecules():
    s = boonza.load(_data("two_water_gro.gro"))
    assert s.nfragments == 2
