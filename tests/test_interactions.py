"""Interaction fingerprints: comparing molecules that have no atoms in common."""

import numpy as np
import pytest
from test_sites import swimming  # noqa: F401 - a fixture, used by the test below

import boonza
from boonza.symmetry import DEFAULT_LIGAND

HERE, THERE = np.array([5.0, 0.0, 0.0]), np.array([-3.0, 6.0, 1.0])


@pytest.fixture(scope="module")
def unlike():
    """A peptide and two different molecules: benzene, and something half again
    as large. The small one stays put; the large one starts beside it and moves."""
    rng = np.random.default_rng(0)
    s = boonza.peptide("AAAAAAAAAA")
    s.positions = s.positions - s.positions.mean(0)
    s.append(boonza.from_smiles("c1ccccc1"))
    s.append(boonza.from_smiles("c1ccccc1CCO"))
    s.cell = np.diag([40.0, 40.0, 40.0])
    lig = s.select(DEFAULT_LIGAND).ids
    frag = np.unique(np.asarray(s.fragids)[lig])
    copies = [lig[np.asarray(s.fragids)[lig] == f] for f in frag]
    base = s.positions.copy()
    for c in copies:
        base[c] -= base[c].mean(0)
    frames = []
    for k in range(60):
        x = base.copy()
        x[copies[0]] = base[copies[0]] + HERE + rng.normal(scale=0.3, size=3)
        x[copies[1]] = base[copies[1]] + (HERE if k < 30 else THERE) + rng.normal(scale=0.3, size=3)
        frames.append(x)
    return s, np.array(frames), [int(f) for f in frag], [len(c) for c in copies]


def test_molecules_of_different_size_are_comparable(unlike):
    """The thing a pose cannot do: the two have no atom-to-atom correspondence."""
    s, frames, frag, sizes = unlike
    assert sizes[0] != sizes[1]  # 6 heavy atoms against 9
    small = boonza.interaction_fingerprints(s, frames, ligand=f"fragid {frag[0]} and noh")
    beside = boonza.interaction_fingerprints(s, frames[:30], ligand=f"fragid {frag[1]} and noh",
                                             residues=small.residues)  # fmt: skip
    away = boonza.interaction_fingerprints(s, frames[30:], ligand=f"fragid {frag[1]} and noh",
                                           residues=small.residues)  # fmt: skip
    assert small.values.shape[1] == beside.values.shape[1] == away.values.shape[1]
    assert len(small) == 60 and len(beside) == 30  # one row per frame and copy
    assert ((small.values >= 0) & (small.values <= 1)).all()

    together = boonza.similarity(small.mean(), beside.mean())
    apart = boonza.similarity(small.mean(), away.mean())
    assert together > 0.95  # unlike molecules, same place: the same residues
    assert together > apart + 0.2  # and the measure can still say no
    # the larger molecule covers what the smaller touches and reaches further, so
    # ask about the smaller one's residues: how much further depends on the
    # conformer RDKit embeds, which is not the same on every platform
    here, there, gone = (set(x.touched(s)) for x in (small, beside, away))
    assert len(here & there) >= 0.75 * len(here)  # the same place
    assert len(here & gone) < 0.75 * len(here)  # and somewhere else is not


def test_what_similarity_refuses(unlike):
    s, frames, frag, _ = unlike
    a = boonza.interaction_fingerprints(s, frames[:5], ligand=f"fragid {frag[0]} and noh")
    assert boonza.similarity(a.mean(), a.mean()) == pytest.approx(1.0)
    assert boonza.similarity(a.mean(), np.zeros_like(a.mean())) == 0.0  # nothing touched
    with pytest.raises(ValueError, match="they must share residues"):
        boonza.similarity(a.mean(), a.mean()[:-1])
    with pytest.raises(ValueError, match="no heavy atoms"):
        boonza.interaction_fingerprints(s, frames, ligand="resname NOPE")


def test_the_matrix_behaves(unlike):
    s, frames, frag, _ = unlike
    f = boonza.interaction_fingerprints(s, frames[:10], ligand=f"fragid {frag[0]} and noh")
    m = boonza.similarity_matrix(f.values)
    assert m.shape == (10, 10)
    assert np.allclose(np.diag(m), 1.0) and np.allclose(m, m.T)
    assert (m <= 1.0).all() and (m >= -1.0).all()


def test_softening_is_what_stops_flicker(unlike):
    """A hard cutoff makes a residue come and go as the ligand breathes."""
    s, frames, frag, _ = unlike
    f = boonza.interaction_fingerprints(s, frames, ligand=f"fragid {frag[0]} and noh")
    per_frame = f.values[:, f.mean() > 0.2]
    assert per_frame.size  # residues it is genuinely near
    wobble = per_frame.std(0).max()
    hard = (per_frame >= 0.5).astype(float).std(0).max()  # the same thing, thresholded
    assert wobble < hard  # the softened value varies less frame to frame


def test_a_sites_ligands_can_be_compared(swimming):  # noqa: F811
    s, runs = swimming
    found = boonza.sites(s, runs)
    f = boonza.site_interactions(s, runs, found, 0)
    assert f.values.shape[1] == len(f.residues)
    assert len(f) == len(f.where) and f.where.shape[1] == 2  # a row per (run, copy)
    assert set(f.where[:, 0].tolist()) <= set(range(len(runs)))
    within = boonza.similarity_matrix(f.values)
    other = boonza.site_interactions(s, runs, found, 1)
    between = boonza.similarity(f.values.mean(0), other.values.mean(0))
    iu = np.triu_indices(len(f), 1)
    assert within[iu].mean() > between  # a site agrees with itself more than with another


def test_it_paints_the_structure(tmp_path, unlike):
    """The fingerprint in the B-factor column: what it touches, on the structure."""
    s, frames, frag, _ = unlike
    f = boonza.interaction_fingerprints(s, frames, ligand=f"fragid {frag[0]} and noh")
    path = tmp_path / "touched.pdb"
    f.write_structure(s, path)
    back = boonza.load(path)
    assert back.natoms == s.natoms
    painted = back.atoms["bfactor"]
    for residue, value in zip(f.residues.tolist(), f.mean().tolist(), strict=True):
        atoms = np.flatnonzero(back.atoms["residue"] == residue)
        assert np.allclose(painted[atoms], value, atol=1e-2)  # the PDB column is 2 decimals
    ligand = back.select(DEFAULT_LIGAND).ids  # the ligand itself is not a residue it touches
    assert np.allclose(painted[ligand], 0.0)
    with pytest.raises(ValueError, match="not one per residue"):
        f.write_structure(s, path, values=f.mean()[:-1])


def test_the_table_is_named(unlike):
    s, frames, frag, _ = unlike
    f = boonza.interaction_fingerprints(s, frames[:5], ligand=f"fragid {frag[0]} and noh")
    table = f.table(s)
    assert table.shape == (5, len(f.residues))
    assert list(table.columns) == f.names(s) and "ALA5" in table.columns
    assert table.index.names == ["frame", "copy"]


def test_the_heatmap_puts_like_beside_like(tmp_path, unlike):
    from boonza.interactions import _like_beside_like

    s, frames, frag, _ = unlike
    rng = np.random.default_rng(0)
    a = np.tile([1.0, 1.0, 0.0, 0.0], (4, 1)) + rng.normal(scale=0.01, size=(4, 4))
    b = np.tile([0.0, 0.0, 1.0, 1.0], (4, 1)) + rng.normal(scale=0.01, size=(4, 4))
    order = _like_beside_like(np.vstack([a[0], b[0], a[1], b[1], a[2], b[2]]))
    kinds = [i % 2 for i in order.tolist()]  # the two kinds were interleaved
    assert kinds in ([0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0])  # and come back apart

    f = boonza.interaction_fingerprints(s, frames[:6], ligand=f"fragid {frag[0]} and noh")
    path = tmp_path / "fingerprints.png"
    drawn = boonza.plot_interactions(f, s, path)
    if drawn:  # matplotlib is optional, as for the restraint plot
        assert path.stat().st_size > 1000
    assert boonza.plot_interactions(f, s, path, share=2.0) is False  # nothing that close
