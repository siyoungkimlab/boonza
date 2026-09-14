"""Amber prmtop/rst7 reading vs msys's LoadPrmTop, and energies vs OpenMM's AmberPrmtopFile."""

import bz2
from pathlib import Path

import numpy as np
import pytest
from conftest import msys_file, run_msys

import boonza
from boonza.io.amber import load_prmtop, read_amber_coordinates

MDA_AMBER = Path("~/mdanalysis/testsuite/MDAnalysisTests/data/Amber").expanduser()


def _msys(tmp_path, prmtop, crd="-"):
    out = tmp_path / "msys.dms"
    run_msys("prmtop", prmtop, crd, out)
    return boonza.load(out)


def test_matches_msys_conversion(tmp_path):
    ours = boonza.load(msys_file("sys.prmtop"), coordinates=msys_file("eq.rst"))
    ref = _msys(tmp_path, msys_file("sys.prmtop"), msys_file("eq.rst"))
    assert [str(d) for d in boonza.diff(ours, ref)] == []


@pytest.mark.parametrize("name", ["bala.prmtop", "ache.prmtop", "tz2.truncoct.parm7.bz2",
                                  "ace_mbondi3.parm7", "ala.ff19SB.OPC.parm7.bz2"])  # fmt: skip
def test_more_prmtops_match_msys(tmp_path, name):
    src = MDA_AMBER / name
    if not src.exists():
        pytest.skip(f"{src} not found")
    plain = src
    if name.endswith(".bz2"):  # msys reads plain files only
        plain = tmp_path / name[:-4]
        plain.write_bytes(bz2.decompress(src.read_bytes()))
    ours = load_prmtop(src)
    ref = _msys(tmp_path, plain)
    assert [str(d) for d in boonza.diff(ours, ref, positions=False)] == []
    if "ff19SB" in name:
        assert "torsiontorsion_cmap" in ours.tables and ours.aux_tables  # CMAP grids


def test_energies_match_openmm():
    pytest.importorskip("openmm")
    from test_openmm import _amber, _assert_close, _energies

    s = boonza.load(msys_file("sys.prmtop"), coordinates=msys_file("eq.rst"))
    _, omm, pos = boonza.to_openmm(s, constraints=False)
    _, ref_sys, ref_pos = _amber(constraints=None, rigidWater=False)
    _assert_close(_energies(omm, pos), _energies(ref_sys, ref_pos), rtol=2e-5)


def test_restart_files(tmp_path):
    s = boonza.load(msys_file("sys.prmtop"), coordinates=msys_file("eq.rst"))
    n = s.natoms
    pos, vel, cell = read_amber_coordinates(msys_file("eq.rst"), n)
    assert vel is not None and cell is not None
    np.testing.assert_allclose(s.positions, pos)

    def rst(path, blocks):
        lines = ["title", f"{n:6d}"]
        for block in blocks:
            flat = np.asarray(block).reshape(-1)
            lines += [
                "".join(f"{v:12.7f}" for v in flat[k : k + 6]) for k in range(0, len(flat), 6)
            ]
        path.write_text("\n".join(lines) + "\n")
        return path

    box = [[30.0, 31.0, 32.0, 90.0, 95.0, 90.0]]
    # inpcrd with a box and no velocities (msys would misread this one)
    p, v, c = read_amber_coordinates(rst(tmp_path / "a.rst7", [pos, box]), n)
    assert v is None
    np.testing.assert_allclose(p, pos, atol=1e-6)
    np.testing.assert_allclose(boonza.io.pdb.lengths_angles_from_cell(c), box[0], atol=1e-6)
    p, v, c = read_amber_coordinates(rst(tmp_path / "b.rst7", [pos]), n)
    assert v is None and c is None
    # an Amber NetCDF restart works too
    with boonza.open_writer(tmp_path / "r.nc", n) as w:
        w.write(pos, box=s.cell)
    p, _, c = read_amber_coordinates(tmp_path / "r.nc", n)
    np.testing.assert_allclose(p, pos, atol=1e-4)
    np.testing.assert_allclose(c, s.cell, atol=1e-4)
