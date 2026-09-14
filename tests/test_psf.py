"""PSF topologies vs MDAnalysis's PSFParser (and OpenMM's CharmmPsfFile)."""

from pathlib import Path

import numpy as np
import pytest

import boonza

MDA_DATA = Path("~/mdanalysis/testsuite/MDAnalysisTests/data").expanduser()
FILES = ["adk.psf", "namd_cgenff.psf", "1a2c_ins_code.psf", "nosegid.psf", "parmed_ala3.psf",
         "tip125_tric_C36.psf", "SiN_tric_namd.psf", "watdyn.psf", "adk_notop.psf"]  # fmt: skip


def _need(name):
    path = MDA_DATA / name
    if not path.exists():
        pytest.skip(f"{path} not found")
    return path


@pytest.mark.parametrize("name", FILES)
def test_matches_mdanalysis(name):
    mda = pytest.importorskip("MDAnalysis")
    path = _need(name)
    u = mda.Universe(str(path))
    s = boonza.load(path)
    a = u.atoms
    assert s.natoms == len(a)
    np.testing.assert_array_equal(s.atoms["name"], a.names)
    # MDAnalysis cuts types to the 4 columns of the standard layout (5-letter CGenFF
    # types in a NAMD-written file without the NAMD flag); boonza keeps the whole type
    ours_t, theirs_t = s.atoms["type"].tolist(), a.types.astype(str).tolist()
    assert all(o == m or (len(m) == 4 and o.startswith(m))
               for o, m in zip(ours_t, theirs_t, strict=True))  # fmt: skip
    np.testing.assert_allclose(s.atoms["charge"], a.charges, atol=1e-6)
    np.testing.assert_allclose(s.atoms["mass"], a.masses, atol=1e-6)
    res = s.atoms["residue"]
    np.testing.assert_array_equal(s.residues["resid"][res], a.resids)
    np.testing.assert_array_equal(s.residues["name"][res], a.resnames)
    segids = s.chains["segid"][s.residues["chain"][res]]
    np.testing.assert_array_equal(np.where(segids == "", "SYSTEM", segids), a.segids)
    if hasattr(a, "icodes"):
        np.testing.assert_array_equal(s.residues["insertion"][res], a.icodes)
    ours = {
        tuple(sorted(p)) for p in zip(s.bonds["i"].tolist(), s.bonds["j"].tolist(), strict=True)
    }
    theirs = {tuple(sorted(map(int, p))) for p in u.bonds.indices} if hasattr(u, "bonds") else set()
    assert ours == theirs
    # a new residue wherever segid, resid, insertion code or name changes; MDAnalysis
    # merges residues that differ only by insertion code (1H and 1 in thrombin)
    # (MDAnalysis reads no insertion codes from PSF files, so boonza's are used here)
    icodes = a.icodes if hasattr(a, "icodes") else s.residues["insertion"][res]
    key = list(zip(a.segids, a.resids, icodes, a.resnames, strict=True))
    assert s.nresidues == sum(k == 0 or key[k] != key[k - 1] for k in range(len(key)))
    assert s.nchains == len(u.segments)


def test_insertion_codes_and_openmm():
    s = boonza.load(_need("1a2c_ins_code.psf"))
    first = s.residue(0)
    assert (first.resid, first.insertion, first.chain.name) == (1, "H", "PROA")
    app = pytest.importorskip("openmm.app")
    psf = app.CharmmPsfFile(str(_need("adk.psf")))
    s = boonza.load(_need("adk.psf"))
    assert s.natoms == psf.topology.getNumAtoms() and s.nbonds == psf.topology.getNumBonds()
    assert [c.name for c in s.chains] == [c.id for c in psf.topology.chains()]


def test_coordinates_from_another_file(tmp_path):
    s = boonza.load(_need("adk.psf"))
    coords = np.arange(s.natoms * 3, dtype=float).reshape(-1, 3) / 7.0
    t = boonza.load(_need("adk.psf"), coordinates=coords)
    np.testing.assert_allclose(t.positions, coords)
    t.cell = np.diag([80.0, 81.0, 82.0])
    boonza.save(t, tmp_path / "adk.pdb")
    u = boonza.load(_need("adk.psf"), coordinates=tmp_path / "adk.pdb")
    np.testing.assert_allclose(u.positions, coords, atol=1e-3)
    np.testing.assert_allclose(u.cell, t.cell, atol=1e-3)
