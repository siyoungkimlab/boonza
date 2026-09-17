"""Representative poses: the linkage, the symmetry correction, and what comes back."""

import numpy as np
import pytest

import boonza
from boonza.poses import _cut, _nn_chain
from boonza.symmetry import DEFAULT_LIGAND


def _partition(labels) -> set:
    groups: dict[int, list[int]] = {}
    for i, g in enumerate(labels.tolist()):
        groups.setdefault(g, []).append(i)
    return {frozenset(v) for v in groups.values()}


def _slow_linkage(d):
    """Average linkage the obvious way: join the closest pair, average the rows, repeat."""
    n = len(d)
    D, alive, size, out = d.copy(), list(range(n)), np.ones(n), []
    while len(alive) > 1:
        best, pair = np.inf, None
        for i in alive:
            for j in alive:
                if j > i and D[i, j] < best:
                    best, pair = D[i, j], (i, j)
        a, b = pair
        out.append((a, b, best))
        for x in alive:
            if x not in (a, b):
                D[a, x] = D[x, a] = (size[a] * D[a, x] + size[b] * D[b, x]) / (size[a] + size[b])
        size[a] += size[b]
        alive.remove(b)
    return np.array(out)


@pytest.fixture(scope="module")
def two_sites():
    """Eight alanines and a benzene that sits in one place, then another; every third
    frame has the ring rotated, which is a relabelling, not a move."""
    rng = np.random.default_rng(3)
    s = boonza.peptide("AAAAAAAA")
    s.positions = s.positions - s.positions.mean(0)
    s.append(boonza.from_smiles("c1ccccc1"))
    s.cell = np.diag([40.0, 40.0, 40.0])
    ring = s.select(DEFAULT_LIGAND).ids
    base = s.positions.copy()
    base[ring] -= base[ring].mean(0)
    frames = []
    for k in range(100):
        x = base.copy()
        where = np.array([4.0, 0.0, 0.0]) if k < 60 else np.array([4.0, 6.0, 0.0])
        x[ring] = base[ring] + where + rng.normal(scale=0.15, size=(len(ring), 3))
        if k % 3 == 0:
            x[ring] = np.roll(x[ring], 1, axis=0)
        frames.append(x)
    return s, np.array(frames)


def test_average_linkage_matches_the_obvious_way():
    rng = np.random.default_rng(0)
    for _ in range(4):
        n = int(rng.integers(8, 45))
        x = rng.normal(size=(n, 3))
        d = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
        fast, slow = _nn_chain(d), _slow_linkage(d)
        assert np.allclose(np.sort(fast[:, 2]), np.sort(slow[:, 2]))
        for h in np.linspace(0, fast[:, 2].max() * 1.05, 20):  # and every cut agrees
            assert _partition(_cut(fast, n, h)) == _partition(_cut(slow, n, h))


def test_distances_are_the_drmsd_of_every_pair(two_sites):
    s, frames = two_sites
    d = boonza.pose_distances(s, frames)
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0)
    assert np.allclose(d[0], boonza.drmsd(s, positions=frames).drmsd)  # row 0: against frame 0
    middle = s.clone()
    middle.positions = frames[50]
    assert np.allclose(d[50], boonza.drmsd(s, reference=middle, positions=frames).drmsd)


def test_poses_are_the_two_sites(two_sites):
    s, frames = two_sites
    p = boonza.poses(s, frames, cutoff=1.5)
    assert len(p) == 2
    assert [len(x) for x in p] == [60, 40]  # the sites, not the relabelled frames
    assert set(p[0].frames.tolist()) == set(range(60))
    assert p[0].population == 0.6 and p[0].center in p[0].frames
    assert (p.labels >= 0).all() and set(p.labels.tolist()) == {0, 1}
    rotated = np.flatnonzero(np.arange(100) % 3 == 0)  # spread over both, not a pose of their own
    assert 0 < len(set(rotated.tolist()) & set(p[1].frames.tolist())) < len(p[1])


def test_symmetry_absorbs_a_relabelled_ring(two_sites):
    s, frames = two_sites
    tight = boonza.poses(s, frames, cutoff=1.5)
    loose = boonza.poses(s, frames, cutoff=1.5, symmetry=False)
    assert [len(x) for x in loose] == [60, 40]  # the sites are far enough apart either way
    assert tight[0].spread < 0.5 * loose[0].spread  # but the ring stops looking like motion


def test_sweep_reads_every_cutoff_off_the_one_tree(two_sites):
    s, frames = two_sites
    p = boonza.poses(s, frames, cutoff=1.5)
    cutoffs, share, count = p.sweep(np.linspace(0.5, 6.0, 12))
    assert np.all(np.diff(share) >= 0) and np.all(np.diff(count) <= 0)  # both are monotone
    assert share[0] == 0.6 and share[-1] == 1.0  # the two sites hold up, then merge
    assert p.sweep()[1].shape == (12,)  # a default range of cutoffs


def test_what_it_refuses(two_sites):
    s, frames = two_sites
    with pytest.raises(ValueError, match="at least two frames"):
        boonza.poses(s, frames[0])
    with pytest.raises(ValueError, match="within 0.1 A"):
        boonza.poses(s, frames, pocket_cutoff=0.1)


def test_a_trajectory_reads_the_same_as_an_array(two_sites, tmp_path):
    """Frames come from a file chunk by chunk, and only the atoms the pocket needs."""
    s, frames = two_sites
    path = tmp_path / "poses.dcd"
    with boonza.open_writer(path, s.natoms) as w:
        for x in frames:
            w.write(x, box=s.cell)
    traj = boonza.open_trajectory(path, s)
    assert len(traj) == len(frames)
    assert np.allclose(boonza.pose_distances(s, traj), boonza.pose_distances(s, frames), atol=1e-3)
    from_file, from_memory = boonza.poses(s, traj), boonza.poses(s, frames)
    assert [len(x) for x in from_file] == [len(x) for x in from_memory] == [60, 40]
    assert [x.center for x in from_file] == [x.center for x in from_memory]
    every_fourth = boonza.poses(s, traj[::4])  # a slice, for a run too long to hold at once
    assert [len(x) for x in every_fourth] == [15, 10]
