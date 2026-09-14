"""boonza.analysis against MDAnalysis (RMSD, RMSF, Rg, RDF) and mdtraj (contacts, SASA, Rg)."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_hbonds import _index_sel

import boonza
from boonza.analysis import (
    radius_of_gyration,
    rdf,
    residue_contacts,
    rmsd_trajectory,
    rmsf,
    sasa,
)

HERE = Path(__file__).parent
DATA = Path("~/mdanalysis/testsuite/MDAnalysisTests/data").expanduser()
GRO, XTC = DATA / "adk_oplsaa.gro", DATA / "adk_oplsaa.xtc"


@pytest.fixture(scope="module")
def adk():
    if not GRO.exists():
        pytest.skip(f"{GRO} not found")
    mda = pytest.importorskip("MDAnalysis")
    s = boonza.load(GRO)
    frames = boonza.open_trajectory(XTC, s).read()
    return s, frames, mda.Universe(str(GRO), str(XTC))


def test_rmsd_rmsf_rg_match_mdanalysis(adk):
    from MDAnalysis.analysis.rms import RMSD, RMSF

    s, frames, u = adk
    ca = s.select("protein and name CA").ids
    sel = _index_sel(ca)
    ref = RMSD(u, select=sel).run().results.rmsd[:, 2]
    np.testing.assert_allclose(rmsd_trajectory(s, frames, ca), ref, atol=1e-4)
    ref_mass = RMSD(u, select=sel, weights="mass").run().results.rmsd[:, 2]
    np.testing.assert_allclose(rmsd_trajectory(s, frames, ca, weights="mass"), ref_mass,
                               atol=1e-4)  # fmt: skip
    got = rmsf(s, frames, ca)
    want = RMSF(u.select_atoms(sel)).run().results.rmsf
    np.testing.assert_allclose(got, want, atol=1e-3)
    prot = s.select("protein").ids
    group = u.select_atoms(_index_sel(prot))
    want_rg = [group.radius_of_gyration() for _ in u.trajectory]
    np.testing.assert_allclose(radius_of_gyration(s, frames, prot), want_rg, rtol=1e-5)
    assert rmsd_trajectory(s, frames, ca)[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("norm", ["rdf", "density", "none"])
def test_rdf_matches_mdanalysis(adk, norm):
    from MDAnalysis.analysis.rdf import InterRDF

    s, frames, u = adk
    ow = s.select("water and atomicnumber 8").ids
    g = u.select_atoms(_index_sel(ow))
    ref = InterRDF(g, g, nbins=40, range=(0.0, 8.0), norm=norm,
                   exclusion_block=(1, 1)).run(stop=3)  # fmt: skip
    first3 = boonza.trajectory.Frames(frames.indices[:3], frames.positions[:3], frames.boxes[:3],
                                      frames.times[:3], frames.steps[:3])  # fmt: skip
    centers, g_r, edges, counts = rdf(s, ow, ow, first3, nbins=40, range=(0.0, 8.0), norm=norm,
                                      exclusion_block=(1, 1))  # fmt: skip
    np.testing.assert_allclose(centers, ref.results.bins)
    # float32 distances in MDAnalysis may move a pair across a bin edge
    assert np.abs(counts - ref.results.count).max() <= 2
    np.testing.assert_allclose(g_r, ref.results.rdf, rtol=2e-3, atol=1e-9)
    if norm == "rdf":
        first_shell = centers[np.argmax(g_r)]
        assert 2.6 < first_shell < 3.0  # water O-O peak


def _mdtraj(path):
    from test_dssp import MDTRAJ_PYTHON, _mdtraj_available

    if not _mdtraj_available():
        pytest.skip("mdtraj not available; set BOONZA_MDTRAJ_PYTHON")
    r = subprocess.run([MDTRAJ_PYTHON, str(HERE / "mdtraj_oracle.py"), "analysis", str(path)],
                       capture_output=True, text=True)  # fmt: skip
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.fixture(scope="module")
def lysozyme():
    path = HERE / "data" / "1LYZ.pdb"
    return boonza.load(path), _mdtraj(path)


def test_contacts_match_mdtraj(lysozyme):
    s, ref = lysozyme
    assert s.natoms == ref["natoms"]
    for scheme, key in (("closest-heavy", "heavy"), ("ca", "ca")):
        d, pairs = residue_contacts(s, scheme=scheme)
        assert pairs.tolist() == ref[f"pairs_{key}"]
        np.testing.assert_allclose(d[0], np.array(ref[f"contacts_{key}"]) * 10, atol=2e-3)


def test_sasa_matches_mdtraj(lysozyme):
    s, ref = lysozyme
    atom = sasa(s)[0]
    want = np.array(ref["sasa"]) * 100  # nm^2 -> A^2
    point = 4 * np.pi * (1.70 + 1.4) ** 2 / 960  # one sphere point on a carbon, A^2
    off = np.abs(atom - want) > 1e-3 * np.maximum(want, 1)
    assert off.sum() <= 0.01 * len(atom)  # float rounding flips a few sphere points
    assert np.abs(atom - want).max() <= 3 * point
    assert atom.sum() == pytest.approx(want.sum(), rel=1e-4)
    res = sasa(s, mode="residue")[0]
    np.testing.assert_allclose(res.sum(), atom.sum(), rtol=1e-9)
    assert res.sum() == pytest.approx(np.sum(ref["residue_sasa"]) * 100, rel=1e-4)
    sub = sasa(s, atoms="resid 1 to 5")[0]
    ids = s.select("resid 1 to 5").ids
    np.testing.assert_allclose(sub[ids], atom[ids], atol=1e-6)
    assert (np.delete(sub, ids) == -1).all()


def test_rg_matches_mdtraj(lysozyme):
    s, ref = lysozyme
    assert radius_of_gyration(s, weights=None)[0] == pytest.approx(ref["rg"] * 10, rel=1e-5)
