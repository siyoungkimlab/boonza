import numpy as np
import pytest

from boonza import pbc
from boonza.io.pdb import cell_from_lengths_angles

mdd = pytest.importorskip("MDAnalysis.lib.distances")

BOXES = [None, [20.0, 25.0, 30.0, 90.0, 90.0, 90.0], [20.0, 25.0, 30.0, 70.0, 80.0, 100.0]]
IDS = ["nobox", "ortho", "triclinic"]
# MDAnalysis computes in float32; boonza in float64
TOL = {"rtol": 2e-6, "atol": 2e-5}


def _data(n, seed):
    # float32-representable coordinates, so both libraries start from the same numbers
    return np.random.default_rng(seed).uniform(-5, 35, (n, 3)).astype(np.float32).astype(float)


@pytest.mark.parametrize("dims", BOXES, ids=IDS)
def test_distances_match_mdanalysis(dims):
    a, b = _data(60, 1), _data(45, 2)
    box = None if dims is None else cell_from_lengths_angles(*dims)
    kw = {} if dims is None else {"box": np.array(dims)}
    np.testing.assert_allclose(pbc.distances(a, b, box), mdd.distance_array(a, b, **kw), **TOL)
    np.testing.assert_allclose(pbc.self_distances(a, box), mdd.self_distance_array(a, **kw), **TOL)
    np.testing.assert_allclose(pbc.paired_distances(a[:45], b, box),
                               mdd.calc_bonds(a[:45], b, **kw), **TOL)  # fmt: skip
    c, d = _data(45, 3), _data(45, 4)
    np.testing.assert_allclose(pbc.angles(a[:45], b, c, box),
                               mdd.calc_angles(a[:45], b, c, **kw), **TOL)  # fmt: skip
    np.testing.assert_allclose(pbc.dihedrals(a[:45], b, c, d, box),
                               mdd.calc_dihedrals(a[:45], b, c, d, **kw), **TOL)  # fmt: skip
    if dims is not None:
        v = b - a[:45]
        np.testing.assert_allclose(pbc.minimum_image(v, box),
                                   mdd.minimize_vectors(v, np.array(dims)), **TOL)  # fmt: skip


@pytest.mark.parametrize("dims", BOXES, ids=IDS)
def test_capped_distances_match_mdanalysis(dims):
    a, b = _data(300, 5), _data(250, 6)
    box = None if dims is None else cell_from_lengths_angles(*dims)
    i, j, d = pbc.capped_distances(a, b, 6.0, box)
    kw = {} if dims is None else {"box": np.array(dims)}
    pairs, dist = mdd.capped_distance(a, b, 6.0, **kw)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    assert (i.tolist(), j.tolist()) == (pairs[order, 0].tolist(), pairs[order, 1].tolist())
    np.testing.assert_allclose(d, dist[order], **TOL)


@pytest.mark.parametrize("dims", BOXES[1:] + [[40.0, 42.0, 45.0, 70.0, 80.0, 100.0]],
                         ids=["ortho", "small-triclinic", "triclinic"])  # fmt: skip
def test_pairs_within_matches_brute_force(dims):
    from boonza.spatial import pairs_within

    box = cell_from_lengths_angles(*dims)
    pos = _data(400, 7)
    i, j, d2 = pairs_within(pos, 5.0, box)
    full = pbc.distances(pos, pos, box)
    bi, bj = np.nonzero(np.triu(full <= 5.0, 1))
    assert (i.tolist(), j.tolist()) == (bi.tolist(), bj.tolist())
    np.testing.assert_allclose(np.sqrt(d2), full[bi, bj], rtol=1e-12)
    ci, cj, cd = pbc.capped_distances(pos[:150], pos, 5.0, box)
    bi, bj = np.nonzero(full[:150] <= 5.0)
    assert (ci.tolist(), cj.tolist()) == (bi.tolist(), bj.tolist())


def test_box_forms():
    dims = [20.0, 25.0, 30.0, 70.0, 80.0, 100.0]
    np.testing.assert_allclose(pbc.as_box(dims), cell_from_lengths_angles(*dims))
    assert pbc.as_box(np.zeros((3, 3))) is None
