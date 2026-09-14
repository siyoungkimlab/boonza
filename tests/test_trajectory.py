import warnings
from pathlib import Path

import numpy as np
import pytest

from boonza.io.dcd import DCDWriter
from boonza.io.pdb import lengths_angles_from_cell
from boonza.trajectory import open_trajectory, open_writer

mda = pytest.importorskip("MDAnalysis")
from MDAnalysis.coordinates.DCD import DCDReader  # noqa: E402
from MDAnalysis.coordinates.XTC import XTCReader  # noqa: E402

DATA = Path("~/mdanalysis/testsuite/MDAnalysisTests/data").expanduser()
MSYS = Path("~/msys/tests/files").expanduser()
DCDS = [
    DATA / "adk_dims.dcd",
    DATA / "adk_dims2.dcd",
    DATA / "SiN_tric_namd.dcd",
    DATA / "tip125_tric_C36.dcd",
    DATA / "coordinates/test.dcd",
    DATA / "watdyn.dcd",
    DATA / "lammps/ifabp_apo_100mM.dcd",
    MSYS / "alanin.dcd",
]
XTCS = [
    DATA / "xtc_test_only_10_frame_10_atoms.xtc",
    DATA / "coordinates/test.xtc",
    DATA / "adk_oplsaa.xtc",
    DATA / "cobrotoxin.xtc",
]


def _need(path):
    if not path.exists():
        pytest.skip(f"{path} not found")
    return path


def _reader(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DCDReader(str(path)) if path.suffix == ".dcd" else XTCReader(str(path))


def _assert_box(box, dims):
    if dims is None or not np.any(dims[:3]):
        assert not box.any()
    else:
        np.testing.assert_allclose(lengths_angles_from_cell(box), dims, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("path", DCDS + XTCS, ids=lambda p: p.name)
def test_matches_mdanalysis(path):
    traj = open_trajectory(_need(path))
    ref = _reader(path)
    assert len(traj) == ref.n_frames and traj.natoms == ref.n_atoms
    for ts, frame in zip(ref, traj, strict=True):
        np.testing.assert_allclose(frame.positions, ts.positions, rtol=1e-6, atol=1e-5)
        _assert_box(frame.box, ts.dimensions)
        assert frame.time == pytest.approx(ts.time, rel=1e-6, abs=1e-9)
    ref.close()


def test_views_and_chunks():
    traj = open_trajectory(_need(DATA / "adk_dims.dcd"))
    sub = traj[2:20:3]
    assert len(sub) == 6
    assert sub[1].index == 5
    np.testing.assert_array_equal(sub[1].positions, traj[5].positions)
    np.testing.assert_array_equal(traj[-1].positions, traj[len(traj) - 1].positions)
    block = traj.read([0, 3], atoms=[0, 10, 20])
    assert block.positions.shape == (2, 3, 3)
    np.testing.assert_array_equal(block.positions[1], traj[3].positions[[0, 10, 20]])
    sizes = [len(b) for b in traj.chunks(40)]
    assert sizes == [40, 40, 18]
    assert sum(1 for _ in traj) == len(traj)
    with pytest.raises(IndexError):
        traj[len(traj)]


@pytest.mark.parametrize("fmt", ["dcd", "xtc"])
def test_write_roundtrip(fmt, tmp_path):
    src = open_trajectory(_need(DATA / "tip125_tric_C36.dcd"))
    out = tmp_path / f"out.{fmt}"
    with open_writer(out, src.natoms) as w:
        w.write_frames(src)
    back = open_trajectory(out)
    ref = _reader(out)
    tol = 1e-5 if fmt == "dcd" else 6e-3  # XTC keeps 0.001 nm
    assert len(back) == len(src) == ref.n_frames
    for a, b, ts in zip(src, back, ref, strict=True):
        np.testing.assert_allclose(b.positions, a.positions, atol=tol)
        np.testing.assert_allclose(ts.positions, a.positions, atol=tol)
        # DCD stores lengths and angles only, so the cell may come back reoriented
        np.testing.assert_allclose(lengths_angles_from_cell(b.box),
                                   lengths_angles_from_cell(a.box), atol=1e-4)  # fmt: skip
        _assert_box(a.box, ts.dimensions)
    ref.close()


def test_fixed_atoms(tmp_path):
    """A CHARMM DCD with fixed atoms stores only the free atoms after frame 0."""
    natoms, free = 4, np.array([2, 4])  # 1-based free atoms
    i4 = lambda *v: np.array(v, "<i4").tobytes()  # noqa: E731
    ic = [3, 0, 1, 0, 0, 0, 0, 0, natoms - len(free), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24]
    head = b"CORD" + np.array(ic, "<i4").tobytes()
    head = head[:40] + np.array([0.5], "<f4").tobytes() + head[44:]
    data = i4(84) + head + i4(84) + i4(84, 1) + b"x" * 80 + i4(84) + i4(4, natoms, 4)
    data += i4(4 * len(free)) + i4(*free) + i4(4 * len(free))
    first = np.arange(12, dtype=np.float32).reshape(3, natoms)  # x row, y row, z row
    for row in first:
        data += i4(4 * natoms) + row.tobytes() + i4(4 * natoms)
    moves = []
    for f in range(2):
        m = np.full((3, len(free)), 100.0 + f, np.float32)
        moves.append(m)
        for row in m:
            data += i4(4 * len(free)) + row.tobytes() + i4(4 * len(free))
    path = tmp_path / "fixed.dcd"
    path.write_bytes(data)
    traj = open_trajectory(path)
    assert len(traj) == 3
    np.testing.assert_array_equal(traj[0].positions, first.T)
    for f in range(2):
        expect = first.T.copy()
        expect[free - 1] = moves[f].T
        np.testing.assert_array_equal(traj[f + 1].positions, expect)


def test_writer_counts_frames(tmp_path):
    path = tmp_path / "count.dcd"
    with DCDWriter(path, 2) as w:
        for k in range(5):
            w.write(np.full((2, 3), k, np.float32))
    traj = open_trajectory(path)
    assert traj.header["nframes_header"] == 5
    assert [f.positions[0, 0] for f in traj] == [0, 1, 2, 3, 4]
