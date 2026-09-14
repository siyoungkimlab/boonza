"""Native contacts and PCA vs MDAnalysis; contact frequencies and blocking vs direct sums."""

from pathlib import Path

import numpy as np
import pytest

import boonza

mda = pytest.importorskip("MDAnalysis")

DATA = Path("~/mdanalysis/testsuite/MDAnalysisTests/data").expanduser()
GRO, XTC = DATA / "adk_oplsaa.gro", DATA / "adk_oplsaa.xtc"
SEL = ("resid 1 to 40 and noh", "resid 41 to 214 and noh")  # boonza
MDA_SEL = ("resid 1-40 and not name H*", "resid 41-214 and not name H*")


@pytest.fixture(scope="module")
def adk():
    if not GRO.exists():
        pytest.skip(f"{GRO} not found")
    s = boonza.load(GRO)
    frames = boonza.open_trajectory(XTC, s).read()
    return s, frames, mda.Universe(str(GRO), str(XTC))


@pytest.mark.parametrize("method", ["hard_cut", "soft_cut", "radius_cut"])
def test_native_contacts_match_mdanalysis(adk, method):
    from MDAnalysis.analysis import contacts

    s, frames, u = adk
    assert len(s.select(SEL[0])) == len(u.select_atoms(MDA_SEL[0]))
    ref = s.copy()
    ref.positions, ref.cell = frames.positions[0], frames.boxes[0]
    q = boonza.native_contacts(s, *SEL, positions=frames, reference=ref, method=method)
    u.trajectory[0]
    groups = (u.select_atoms(MDA_SEL[0]), u.select_atoms(MDA_SEL[1]))
    theirs = contacts.Contacts(u, select=MDA_SEL, refgroup=groups, method=method,
                               radius=4.5).run()  # fmt: skip
    assert q[0] == pytest.approx(1.0) or method == "soft_cut"
    np.testing.assert_allclose(q, theirs.results.timeseries[:, 1], atol=1e-6)


def test_contact_frequency_matches_direct_count(adk):
    s, frames, _ = adk
    a, b = s.select("resid 1 to 30").ids, s.select("resid 150 to 214").ids
    rows, cols, freq = boonza.contact_frequency(s, "resid 1 to 30", "resid 150 to 214",
                                                positions=frames, cutoff=6.0)  # fmt: skip
    res = s.atoms["residue"]
    want = np.zeros((len(rows), len(cols)))
    for X, box in zip(frames.positions, frames.boxes, strict=True):
        d = boonza.pbc.distances(X[a], X[b], box) <= 6.0
        hit = np.zeros_like(want, bool)
        ii, jj = np.nonzero(d)
        hit[np.searchsorted(rows, res[a][ii]), np.searchsorted(cols, res[b][jj])] = True
        want += hit
    np.testing.assert_allclose(freq, want / len(frames.positions))
    assert freq.max() == 1.0

    rows, cols, self_freq = boonza.contact_frequency(s, "resid 1 to 30", positions=frames)
    assert (rows == cols).all() and np.allclose(self_freq, self_freq.T)
    assert (np.diag(self_freq) == 0).all() and self_freq[0, 1] == 1.0  # neighbours touch
    atoms_level = boonza.contact_frequency(s, "resid 1 to 3", positions=frames, level="atom")
    assert atoms_level[2].shape == (len(s.select("resid 1 to 3")),) * 2


def test_pca_matches_mdanalysis(adk):
    from MDAnalysis.analysis.pca import PCA as MDAPCA

    s, frames, u = adk
    ours = boonza.pca(s, frames, sel="name CA")
    theirs = MDAPCA(u, select="name CA", align=True).run()
    k = 4
    np.testing.assert_allclose(ours.variance[:k], theirs.results.variance[:k], rtol=1e-3)
    for c in range(k):
        assert abs(ours.components[c] @ theirs.results.p_components[:, c]) > 0.99
    np.testing.assert_allclose(ours.transform(s, frames), ours.projections, atol=1e-8)
    assert ours.cumulated_variance[-1] == pytest.approx(1.0)
    assert ours.projections.shape == (len(frames.positions), len(ours.variance))
    np.testing.assert_allclose(ours.projections.var(axis=0, ddof=1), ours.variance,
                               rtol=1e-8, atol=1e-12 * ours.variance[0])  # fmt: skip


def test_block_average_on_a_correlated_series():
    rng = np.random.default_rng(0)
    n, phi = 2**15, 0.9
    noise = rng.normal(size=n) * np.sqrt(1 - phi * phi)
    x = np.empty(n)
    x[0] = rng.normal()
    for k in range(1, n):
        x[k] = phi * x[k - 1] + noise[k]
    b = boonza.block_average(x)
    assert b.block_sizes[:4].tolist() == [1, 2, 4, 8]
    assert b.sem[0] == pytest.approx(x.std(ddof=1) / np.sqrt(n))
    size = int(b.block_sizes[3])
    means = x[: n // size * size].reshape(-1, size).mean(axis=1)
    assert b.sem[3] == pytest.approx(means.std(ddof=1) / np.sqrt(len(means)))
    theory = np.sqrt((1 + phi) / (1 - phi) / n)  # AR(1) with unit variance
    assert b.estimate == pytest.approx(theory, rel=0.25)
    assert b.statistical_inefficiency == pytest.approx((1 + phi) / (1 - phi), rel=0.5)
    iid = boonza.block_average(rng.normal(size=4096))
    assert iid.statistical_inefficiency == pytest.approx(1.0, abs=0.5)
