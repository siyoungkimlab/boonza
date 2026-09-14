import numpy as np
import pytest
from conftest import msys_file

import boonza
from boonza.align import kabsch, rmsd, superpose


def _rotation(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    return q if np.linalg.det(q) > 0 else -q


def test_kabsch_recovers_transform():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(50, 3)) * 10
    R, t = _rotation(1), np.array([3.0, -2.0, 7.0])
    Q = P @ R.T + t
    R2, t2 = kabsch(P, Q)
    np.testing.assert_allclose(R2, R, atol=1e-10)
    np.testing.assert_allclose(t2, t, atol=1e-9)
    assert rmsd(P, Q, superpose=True) < 1e-9
    assert rmsd(P, Q) > 1


def test_superpose_by_sequence():
    ref = boonza.load(msys_file("2f4k.dms"))
    mobile = ref.clone(ref.select("protein and not resid 1 2 3"))
    mobile.residues["resid"] = mobile.residues["resid"] + 100  # numbering differs
    R, t = _rotation(2), np.array([10.0, 0.0, -5.0])
    mobile.positions = mobile.positions @ R.T + t
    result = superpose(mobile, ref)
    assert result.n_matched == ref.select("protein and name CA and not resid 1 2 3").ids.size
    assert result.rmsd < 1e-6
    moved = mobile.select("name CA").positions
    expected = ref.select("protein and name CA and not resid 1 2 3").positions
    np.testing.assert_allclose(moved, expected, atol=1e-6)


def test_superpose_rejects_outliers():
    ref = boonza.load(msys_file("2f4k.dms"))
    mobile = ref.clone(ref.select("protein"))
    ca = mobile.select("name CA").ids
    pos = mobile.positions.copy()
    pos[ca[:3]] += 25.0  # three residues moved far away
    mobile.positions = pos
    result = superpose(mobile, ref, cutoff=2.0)
    assert result.n_used == result.n_matched - 3
    assert result.rmsd < 1e-6


def test_superpose_order_needs_equal_counts():
    ref = boonza.load(msys_file("2f4k.dms"))
    mobile = ref.clone(ref.select("protein and not resid 1"))
    with pytest.raises(ValueError):
        superpose(mobile, ref, match="order")
