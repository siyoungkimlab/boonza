"""Native XTC/TRR vs GROMACS's xdrfile library (as compiled in MDAnalysis):
exact reads of real files, byte-identical writes."""

from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.io import _xdr

libmdaxdr = pytest.importorskip("MDAnalysis.lib.formats.libmdaxdr")
DATA = Path("~/mdanalysis/testsuite/MDAnalysisTests/data").expanduser()


def _need(name):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{path} not found")
    return path


@pytest.mark.parametrize("name", ["adk_oplsaa.xtc", "cobrotoxin.xtc"])
def test_xtc_reads_exactly(name):
    path = _need(name)
    buf = np.fromfile(path, np.uint8)
    offsets = _xdr.frame_offsets(buf, "xtc")
    with libmdaxdr.XTCFile(str(path)) as f:
        ref = [f.read() for _ in range(len(f))]
        np.testing.assert_array_equal(offsets, f.offsets)
    for off, theirs in zip(offsets, ref, strict=True):
        _, step, time, box, xyz = _xdr.read_xtc_frame(buf, int(off))
        np.testing.assert_array_equal(xyz, theirs.x)  # bit-identical decompression
        np.testing.assert_allclose(box, theirs.box, rtol=0, atol=0)
        assert step == theirs.step and np.float32(time) == np.float32(theirs.time)
    traj = boonza.open_trajectory(path)
    np.testing.assert_array_equal(traj[len(traj) - 1].positions, ref[-1].x * np.float32(10.0))


@pytest.mark.parametrize("name", ["adk_oplsaa.trr", "trr_test_only_10_frame_10_atoms.trr"])
def test_trr_reads_exactly(name):
    path = _need(name)
    buf = np.fromfile(path, np.uint8)
    offsets = _xdr.frame_offsets(buf, "trr")
    with libmdaxdr.TRRFile(str(path)) as f:
        ref = [f.read() for _ in range(len(f))]
    assert len(offsets) == len(ref)
    for off, theirs in zip(offsets, ref, strict=True):
        _, step, time, box, x, v, force = _xdr.read_trr_frame(buf, int(off))
        assert step == theirs.step and time == pytest.approx(theirs.time)
        for mine, has, other in ((x, theirs.hasx, theirs.x), (v, theirs.hasv, theirs.v),
                                 (force, theirs.hasf, theirs.f)):  # fmt: skip
            assert (mine is not None) == bool(has)
            if mine is not None:
                np.testing.assert_array_equal(mine, other)


def _water_like(rng, natoms):
    xyz = rng.uniform(0, 5.0, (natoms, 3)).astype(np.float32)
    xyz[1::3] = xyz[0::3][: len(xyz[1::3])] + 0.1  # close pairs: swaps and runs
    return xyz


@pytest.mark.parametrize(("natoms", "precision"), [(500, 1000.0), (3000, 100.0), (7, 1000.0)])
def test_xtc_writes_byte_identical(tmp_path, natoms, precision):
    rng = np.random.default_rng(natoms)
    xyz = _water_like(rng, natoms)
    box = np.diag([5.0, 5.1, 5.2]).astype(np.float32)
    ref = tmp_path / "ref.xtc"
    with libmdaxdr.XTCFile(str(ref), "w") as f:
        for k in range(3):
            f.write(xyz + np.float32(0.01 * k), box, k, 2.0 * k, precision)
    ours = b"".join(_xdr.xtc_frame_bytes(xyz + np.float32(0.01 * k), box, k, 2.0 * k, precision)
                    for k in range(3))  # fmt: skip
    assert ours == ref.read_bytes()
    # the writer class produces the same bytes from Å coordinates
    path = tmp_path / "ours.xtc"
    with boonza.open_writer(path, natoms, precision=precision) as w:
        for k in range(3):
            w.write((xyz + np.float32(0.01 * k)) * np.float32(10.0), box=box * 10.0,
                    time=2.0 * k, step=k)  # fmt: skip
    back = boonza.open_trajectory(path).read()
    expected = xyz[None] + 0.01 * np.arange(3)[:, None, None]
    np.testing.assert_allclose(back.positions / 10.0, expected, atol=1.0 / precision)


def test_trr_writes_byte_identical(tmp_path):
    rng = np.random.default_rng(1)
    xyz = rng.normal(0, 2, (50, 3)).astype(np.float32)
    box = np.diag([3.0, 3.0, 3.0]).astype(np.float32)
    ref = tmp_path / "ref.trr"
    with libmdaxdr.TRRFile(str(ref), "w") as f:
        for k in range(2):
            f.write(xyz + k, None, None, box, k, 1.5 * k, 0.0, 50)
    ours = b"".join(_xdr.trr_frame_bytes(50, k, 1.5 * k, box, xyz + k) for k in range(2))
    assert ours == ref.read_bytes()
