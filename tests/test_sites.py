"""Binding sites pooled over runs and copies, and how they feed the pose level."""

import numpy as np
import pytest

import boonza
from boonza.symmetry import DEFAULT_LIGAND

TRUE = (np.array([6.0, 0.0, 0.0]), np.array([-2.0, 7.0, 1.0]))


@pytest.fixture(scope="module")
def swimming():
    """A protein with three benzenes: two settle into two sites after frame 30,
    the third wanders; four independent runs, the protein tumbling in each."""
    s = boonza.peptide("AAAAAAAAAA")
    s.positions = s.positions - s.positions.mean(0)
    for _ in range(3):
        s.append(boonza.from_smiles("c1ccccc1"))
    s.cell = np.diag([40.0, 40.0, 40.0])
    lig = s.select(DEFAULT_LIGAND).ids
    frag = np.asarray(s.fragids)[lig]
    copies = [lig[frag == f] for f in np.unique(frag)]
    base = s.positions.copy()
    for c in copies:
        base[c] -= base[c].mean(0)

    def run(seed, nframes=150):
        r = np.random.default_rng(seed)
        out = []
        for k in range(nframes):
            x = base.copy()
            for ci, c in enumerate(copies):
                bound = ci < 2 and k > 30
                x[c] = base[c] + (TRUE[ci] + r.normal(scale=0.4, size=3) if bound
                                  else r.uniform(-18, 18, size=3))  # fmt: skip
            angle = r.uniform(0, 2 * np.pi)
            ca, sa = np.cos(angle), np.sin(angle)
            spin = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
            out.append(x @ spin.T + r.normal(scale=0.05, size=3))
        return np.array(out)

    return s, [run(seed) for seed in (1, 2, 3, 4)]


def test_sites_are_found_where_they_were_put(swimming):
    s, runs = swimming
    found = boonza.sites(s, runs)
    assert len(found) == 2
    for site in found:
        assert min(np.linalg.norm(site.center - t) for t in TRUE) < 0.5
        assert site.runs == 4  # every run agrees, which is the evidence that counts
        assert 0.2 < site.occupancy < 0.35  # 2 of 3 copies, for 120 of 150 frames
        assert site.spread < 2.0
        assert site.arrivals >= 4
    assert len(found.labels) == 4 * 150 * 3


def test_bulk_is_bulk_and_not_a_site(swimming):
    s, runs = swimming
    found = boonza.sites(s, runs)
    bulk = (found.labels < 0).mean()
    assert 0.35 < bulk < 0.6  # the wandering copy, plus the frames before the others settle
    rng = np.random.default_rng(0)
    nowhere = []  # nothing binds: every copy everywhere, every frame
    base = runs[0][0]
    lig = s.select(DEFAULT_LIGAND).ids
    frag = np.asarray(s.fragids)[lig]
    for _ in range(200):
        x = base.copy()
        for f in np.unique(frag):
            c = lig[frag == f]
            x[c] = base[c] - base[c].mean(0) + rng.uniform(-18, 18, size=3)
        nowhere.append(x)
    assert len(boonza.sites(s, np.array(nowhere))) == 0


def test_enrichment_is_a_threshold_over_bulk(swimming):
    s, runs = swimming
    assert len(boonza.sites(s, runs, enrichment=200.0)) <= len(boonza.sites(s, runs))
    assert len(boonza.sites(s, runs, min_occupancy=0.9)) == 0  # nothing holds 90%


def test_a_site_hands_its_frames_to_the_pose_level(swimming):
    """Phase 2 says where, phase 1 says how: the composition has to line up."""
    s, runs = swimming
    found = boonza.sites(s, runs)
    rows = found.frames(0, run=0)
    assert rows[:, 0].tolist() == [0] * len(rows)  # only that run
    copy = int(np.bincount(rows[:, 1]).argmax())
    frames = rows[rows[:, 1] == copy][:, 2]
    assert len(frames) > 50
    lig = s.select(DEFAULT_LIGAND).ids  # copies share resname and resid; fragid tells them apart
    one = f"fragid {int(np.asarray(s.fragids)[lig][0]) + copy} and noh"
    assert len(s.select(one).ids) == 6
    p = boonza.poses(s, runs[0][frames], ligand=one, pocket_cutoff=12.0)
    assert len(p) >= 1 and p[0].center in p[0].frames
    assert p[0].population > 0.5  # the copy sat still in that site, so one pose dominates


def test_a_sites_pocket_comes_from_its_frames(swimming):
    """An atom earns its place by being there for the ligand, not by being close once."""
    s, runs = swimming
    found = boonza.sites(s, runs)
    pocket = boonza.site_pocket(s, runs, found, 0, protein="protein and noh", share=0.3)
    assert len(pocket) >= 4
    assert not set(pocket.tolist()) & set(s.select(DEFAULT_LIGAND).ids.tolist())
    strict = boonza.site_pocket(s, runs, found, 0, protein="protein and noh", share=0.99)
    assert set(strict.tolist()) <= set(pocket.tolist())  # a harder test keeps fewer atoms

    rows = found.frames(0, run=0)  # and it feeds the pose level with no reference at all
    copy = int(np.bincount(rows[:, 1]).argmax())
    frames = rows[rows[:, 1] == copy][:, 2]
    lig = s.select(DEFAULT_LIGAND).ids
    one = f"fragid {int(np.asarray(s.fragids)[lig][0]) + copy} and noh"
    p = boonza.poses(s, runs[0][frames], ligand=one, pocket=pocket)
    assert len(p) == 1 and p[0].population > 0.9


def test_the_sites_command(tmp_path, swimming, capsys):
    import json

    from boonza.cli import main

    s, runs = swimming
    structure = tmp_path / "s.dms"
    boonza.save(s, structure)
    paths = []
    for r, run in enumerate(runs[:2]):
        path = tmp_path / f"run{r}.dcd"
        with boonza.open_writer(path, s.natoms) as w:
            for x in run:
                w.write(x, box=s.cell)
        paths.append(str(path))
    out = tmp_path / "out"
    assert main(["sites", str(structure), "--traj", *paths, "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "in bulk" in printed and "arrivals" in printed
    doc = json.loads((out / "sites.json").read_text())
    assert len(doc["sites"]) == 2
    assert all(site["runs"] == 2 for site in doc["sites"])  # both runs, pooled
    assert doc["bulk"] > 0


def test_the_density_map_agrees_with_the_clusters(tmp_path, swimming):
    """The grid is a second opinion: it finds the sites without clustering at all."""
    s, runs = swimming
    found = boonza.sites(s, runs)
    grid = found.density
    assert grid.counts.sum() == len(found.centroids)  # every frame lands somewhere
    assert grid.enrichment.max() > 100  # a site is far above what bulk explains

    cells = np.array(np.nonzero(grid.enrichment > found.enrichment)).T
    xyz = cells * grid.spacing + grid.origin + 0.5 * grid.spacing
    for site in found:  # every site centre has a dense cell within one spacing
        assert np.linalg.norm(xyz - site.center, axis=1).min() <= grid.spacing

    path = tmp_path / "density.dx"
    grid.write_dx(path)
    head = path.read_text().splitlines()
    nx, ny, nz = grid.counts.shape
    assert head[0] == f"object 1 class gridpositions counts {nx} {ny} {nz}"
    assert head[1].startswith("origin ")  # at cell centres, half a spacing in
    written = np.array([float(x) for line in head[7:-1] for x in line.split()])
    assert written.size == grid.counts.size
    assert np.allclose(written.max(), grid.enrichment.max(), rtol=1e-3)


def test_the_box_is_measured_not_assumed(swimming):
    """Under a barostat the box is not what the structure file says, and the
    concentration a rate is measured against depends on it."""
    s, runs = swimming
    stored = abs(float(np.linalg.det(np.asarray(s.cell, float))))
    rng = np.random.default_rng(0)

    class Breathing:  # an NPT run: equilibrated smaller than it was built, and fluctuating
        positions = runs[0]
        boxes = np.array([np.diag([38.6 + rng.normal(scale=0.15)] * 3) for _ in runs[0]])

    found = boonza.sites(s, Breathing())
    seen = Breathing.boxes[:, 0, 0] ** 3
    assert found.volume == pytest.approx(seen.mean(), rel=1e-6)  # the frames, not the file
    assert found.volume < 0.95 * stored  # and here they differ by a tenth
    _, _, volumes = boonza.ligand_centroids(s, Breathing())
    assert volumes.shape == (len(runs[0]) * 3,)  # one per row: frame and copy


def test_runs_of_different_lengths_pool_by_time(swimming):
    """Nothing is padded or truncated: a run counts for as long as it ran."""
    s, runs = swimming
    ragged = [runs[0][:50], runs[1][:150], runs[2]]
    found = boonza.sites(s, ragged)
    assert len(found.centroids) == 3 * sum(len(r) for r in ragged)  # three copies each
    rows = found.frames(0)
    per_run = np.array([int((rows[:, 0] == r).sum()) for r in range(3)])
    assert (np.diff(per_run) > 0).all()  # the longer the run, the more it contributes
    assert found[0].runs == 3  # but every run is credited once, however long it ran
    assert found.where[:, 2].max() == max(len(r) for r in ragged) - 1
