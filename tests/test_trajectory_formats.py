"""TRR and Amber NetCDF vs MDAnalysis's readers; Desmond DTR/STK vs msys's molfile reader."""

from pathlib import Path

import numpy as np
import pytest
from conftest import msys_file, run_msys

import boonza
from boonza.io.dtr import DTRError
from boonza.io.pdb import cell_from_lengths_angles, lengths_angles_from_cell

MDA_DATA = Path("~/mdanalysis/testsuite/MDAnalysisTests/data").expanduser()


def _need(path):
    if not path.exists():
        pytest.skip(f"{path} not found")
    return path


def _mda_frames(reader):
    pos, dims, times = [], [], []
    for ts in reader:
        pos.append(ts.positions.copy())
        # copy: MDAnalysis reuses one array and rewinds to frame 0 after iterating
        dims.append(np.zeros(6) if ts.dimensions is None else ts.dimensions.copy())
        times.append(ts.time)
    return np.array(pos), np.array(dims), np.array(times)


def _dims(boxes):
    return np.array([lengths_angles_from_cell(b) if b.any() else np.zeros(6) for b in boxes])


def _compare(ours, pos, dims, times, atol=1e-4):
    np.testing.assert_allclose(ours.positions, pos, atol=atol)
    np.testing.assert_allclose(_dims(ours.boxes), dims, atol=1e-3)
    np.testing.assert_allclose(ours.times, times, atol=1e-4)


def _random_frames(n=6, natoms=9, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(0, 10, (n, natoms, 3)).astype(np.float32)
    # a truncated octahedron and a slightly changing monoclinic cell
    boxes = np.array([cell_from_lengths_angles(40 + k, 40 + k, 40 + k, 109.47, 109.47, 109.47)
                      if k % 2 else cell_from_lengths_angles(35, 36 + k, 37, 90, 100, 90)
                      for k in range(n)])  # fmt: skip
    return pos, boxes, np.arange(n) * 2.5


@pytest.mark.parametrize("name", ["adk_oplsaa.trr", "trr_test_only_10_frame_10_atoms.trr"])
def test_trr_matches_mdanalysis(name):
    pytest.importorskip("MDAnalysis")
    from MDAnalysis.coordinates.TRR import TRRReader

    path = _need(MDA_DATA / name)
    traj = boonza.open_trajectory(path)
    _compare(traj.read(), *_mda_frames(TRRReader(str(path))))
    np.testing.assert_array_equal(traj[3].positions, traj.read()[3].positions)


@pytest.mark.parametrize("name", ["tz2.truncoct.nc", "bala.ncdf", "ace_tip3p.nc", "posfor.ncdf"])
def test_amber_netcdf_matches_mdanalysis(name):
    pytest.importorskip("MDAnalysis")
    from MDAnalysis.coordinates.TRJ import NCDFReader

    path = _need(MDA_DATA / "Amber" / name)
    _compare(boonza.open_trajectory(path).read(), *_mda_frames(NCDFReader(str(path))), atol=1e-5)


@pytest.mark.parametrize("fmt", ["nc", "trr"])
def test_writers_round_trip(tmp_path, fmt):
    pos, boxes, times = _random_frames()
    path = tmp_path / f"out.{fmt}"
    with boonza.open_writer(path, pos.shape[1]) as w:
        for p, b, t in zip(pos, boxes, times, strict=True):
            w.write(p, box=b, time=t)
    back = boonza.open_trajectory(path).read()
    np.testing.assert_allclose(back.positions, pos, atol=1e-5)
    np.testing.assert_allclose(back.boxes, boxes, atol=1e-4)
    np.testing.assert_allclose(back.times, times, atol=1e-5)
    mda = pytest.importorskip("MDAnalysis")
    from MDAnalysis.coordinates.TRJ import NCDFReader
    from MDAnalysis.coordinates.TRR import TRRReader

    reader = NCDFReader if fmt == "nc" else TRRReader
    _compare(back, *_mda_frames(reader(str(path))), atol=1e-5)  # MDAnalysis reads our files
    del mda
    if fmt == "nc":
        scipy_io = pytest.importorskip("scipy.io")
        with scipy_io.netcdf_file(path, "r", mmap=False) as nc:
            assert nc.Conventions == b"AMBER"
            np.testing.assert_allclose(nc.variables["coordinates"][:], pos, atol=1e-5)


def test_dtr_matches_msys_reader():
    for path in (msys_file("ch4.dtr"), msys_file("relative.stk")):
        ref = run_msys("dtrread", path)
        ours = boonza.open_trajectory(path).read()
        np.testing.assert_array_equal(ours.positions, np.array(ref["pos"], np.float32))
        np.testing.assert_array_equal(ours.boxes, np.array(ref["box"]))
        np.testing.assert_array_equal(ours.times, ref["time"])
    with pytest.raises(DTRError, match="POSITION"):  # per-term energies and forces only
        boonza.open_trajectory(msys_file("force_groups.dtr"))


def _msys_dtr(tmp_path, name, pos, boxes, times, per_file):
    npz = tmp_path / f"{name}.npz"
    np.savez(npz, pos=pos, boxes=boxes, times=times)
    path = tmp_path / f"{name}.dtr"
    assert run_msys("dtrwrite", npz, path, per_file)["nframes"] == len(pos)
    return path


@pytest.mark.parametrize("double", [False, True])
def test_dtr_written_by_msys(tmp_path, double):
    pos, boxes, times = _random_frames(n=8, natoms=7, seed=1)
    pos = pos.astype(np.float64 if double else np.float32)
    path = _msys_dtr(tmp_path, "run", pos, boxes, times, per_file=3)  # frames in 3 files
    traj = boonza.open_trajectory(path)
    assert (traj.natoms, len(traj)) == (7, 8)
    frames = traj.read()
    assert frames.positions.dtype == pos.dtype
    np.testing.assert_array_equal(frames.positions, pos)
    np.testing.assert_allclose(frames.boxes, boxes, atol=1e-6 if double else 1e-4)
    np.testing.assert_allclose(frames.times, times)
    np.testing.assert_array_equal(traj[5].positions, pos[5])  # random access across files
    np.testing.assert_array_equal(traj[::3].read().positions, pos[::3])


def test_stk_later_runs_supersede(tmp_path):
    pos, boxes, times = _random_frames(n=10, natoms=4, seed=2)
    _msys_dtr(tmp_path, "a", pos, boxes, times, per_file=4)  # times 0 ... 22.5
    later = pos + 100.0
    _msys_dtr(tmp_path, "b", later[:5], boxes[:5], times[:5] + 10.0, per_file=4)  # 10 ... 20
    stk = tmp_path / "run.stk"
    stk.write_text("a.dtr\nb.dtr\n")
    ours = boonza.open_trajectory(stk).read()
    np.testing.assert_allclose(ours.times, [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0])
    np.testing.assert_array_equal(ours.positions[:4], pos[:4])  # before b starts
    np.testing.assert_array_equal(ours.positions[4:], later[:5])
    ref = run_msys("dtrread", stk)
    np.testing.assert_allclose(ours.times, ref["time"])
    np.testing.assert_array_equal(ours.positions, np.array(ref["pos"], np.float32))
