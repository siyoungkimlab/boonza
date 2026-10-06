"""Binding sites pooled over runs and copies, and how they feed the pose level."""

import csv
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.symmetry import DEFAULT_LIGAND

DATA = Path(__file__).parent / "data"
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
    found = boonza.sites(s, runs, pocket_protein="protein")
    assert len(found) == 2
    for site in found:
        assert min(np.linalg.norm(site.center - t) for t in TRUE) < 0.5
        assert site.runs == 4  # every run agrees, which is the evidence that counts
        # held in 120 of 150 frames, by one copy of the three in the box: the
        # frames are what a site is occupied for, reported and not gated on
        assert 0.75 < site.occupancy < 0.85
        assert 0.2 < site.copy_frames < 0.35
        assert site.spread < 2.0
        assert site.arrivals >= 4
    assert len(found.labels) == 4 * 150 * 3


def test_bulk_is_bulk_and_not_a_pocket(swimming):
    """Bulk gathers by chance here and there; what it never does is make a pocket.

    Without a protein to measure against there is nothing to tell a chance
    gathering from a site, so every one is returned.  With one, the chance
    gatherings fail the shell and the enclosure and are not reported.
    """
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
    # nothing binds, so nothing is enclosed by the protein: no pockets at all
    assert len(boonza.sites(s, np.array(nowhere), pocket_protein="protein")) == 0
    loose = boonza.sites(s, np.array(nowhere))  # and with nothing to measure against
    assert all(x.volume == 0.0 for x in loose)  # they are gatherings, not pockets


def test_enrichment_is_a_threshold_over_bulk(swimming):
    s, runs = swimming
    assert len(boonza.sites(s, runs, enrichment=200.0)) <= len(boonza.sites(s, runs))
    assert len(boonza.sites(s, runs, enrichment=2.0)) >= len(boonza.sites(s, runs))


def test_what_the_occupancy_map_is_made_of(swimming):
    """``radius`` says what a molecule is to the map a pocket is cut from: a point,
    a set of points, or a volume.  One point per molecule per frame, one per bead,
    or a sphere per bead -- so the map grows with each, and so do its pockets."""
    s, runs = swimming
    sizes = {}
    for mode in ("point", "beads", "sigma"):
        found = boonza.sites(s, runs, pocket_protein="protein", radius=mode)
        sizes[mode] = found.occupancy.total
    # a benzene of six heavy atoms: six times the deposits of its centre alone
    # (within a little, since a molecule at the grid's edge can have its centre
    # outside it and beads inside), and a sphere apiece more again
    assert 5.9 < sizes["beads"] / sizes["point"] < 6.1
    assert sizes["sigma"] > 5 * sizes["beads"]


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


def test_runs_may_bring_their_own_system(swimming):
    """Only the protein has to match: the ligands can be different molecules,
    in different numbers, from one run to the next."""
    s, runs = swimming
    lig = s.select(DEFAULT_LIGAND).ids  # drop two whole copies, hydrogens and all
    frag = np.asarray(s.fragids)
    drop = np.isin(frag, np.unique(frag[lig])[1:])
    keep = np.flatnonzero(~drop)
    fewer = s.clone(keep)
    trimmed = runs[1][:, keep]
    assert len(fewer.select(DEFAULT_LIGAND).ids) == len(lig) // 3  # one copy left
    assert len(fewer.select("protein").ids) == len(s.select("protein").ids)  # same protein

    mixed = boonza.sites(s, [runs[0], (fewer, trimmed)], pocket_protein="protein")
    assert len(mixed) == 2  # the same two sites, pooled over unlike systems
    assert mixed.systems[0] is s and mixed.systems[1] is fewer
    # the copy that was kept visits one site in both runs; the other site only run 0 reaches
    assert sorted(x.runs for x in mixed) == [1, 2]
    both = next(x for x in mixed if x.runs == 2)
    rows = mixed.frames(mixed.sites.index(both))
    assert set(rows[:, 0].tolist()) == {0, 1}
    assert rows[rows[:, 0] == 1][:, 1].max() == 0  # run 1 has a single copy
    pocket = boonza.site_pocket(s, [runs[0], (fewer, trimmed)], mixed,
                                mixed.sites.index(both), protein="protein and noh",
                                share=0.3)  # fmt: skip
    assert len(pocket) >= 4


def _run_directory(tmp_path, s, run):
    """A directory shaped like one boonza md wrote: what --workdir reads."""
    d = tmp_path / "md"
    d.mkdir(parents=True)
    boonza.save(s, d / "solvated.dms")
    with boonza.open_writer(d / "trajectory.dcd", s.natoms) as w:
        for x in run:
            w.write(x, box=s.cell)
    return d


#: A cavity of 1FNA: the open point most enclosed by the protein, found by
#: taking the grid points 3.4-5.0 A from the nearest atom and keeping the one
#: with the most atoms within 8 A of it -- 120 of them, which is a pocket.
CAVITY = np.array([-7.43, 27.22, 12.99])


@pytest.fixture(scope="module")
def buried():
    """A benzene turning over in a cavity of a protein, which is what a pocket
    looks like: enriched cells, against the protein, enclosed by it.  A second
    copy wanders the box, and is bulk."""
    s = boonza.load(DATA / "1FNA.pdb").clone("protein")
    middle = np.asarray(s.positions).mean(0)
    for _ in range(2):
        s.append(boonza.from_smiles("c1ccccc1"))
    s.cell = np.diag([60.0, 60.0, 60.0])
    lig = s.select(DEFAULT_LIGAND).ids
    frag = np.asarray(s.fragids)[lig]
    copies = [lig[frag == f] for f in np.unique(frag)]
    base = s.positions.copy()
    for c in copies:
        base[c] -= base[c].mean(0)
    r = np.random.default_rng(7)
    frames = []
    for _ in range(60):
        x = base.copy()
        # a molecule held in a pocket still turns and rattles, and it is the
        # cells its atoms reach that give the pocket its shape
        u, _, v = np.linalg.svd(r.normal(size=(3, 3)))
        x[copies[0]] = base[copies[0]] @ (u @ v).T + CAVITY + r.normal(scale=0.6, size=3)
        x[copies[1]] = base[copies[1]] + middle + r.uniform(-25, 25, size=3)
        frames.append(x)
    return s, np.array(frames)


def test_the_sites_command_puts_the_scores_in_one_table(tmp_path, buried, capsys):
    """--features is a second pass over the frames, so it is taken before the table
    rather than after it: one table carries how often a site was held, its pocket
    and its score, and nothing a reader is waiting for arrives later."""
    from boonza.cli import main

    s, frames = buried
    structure = tmp_path / "s.dms"
    boonza.save(s, structure)
    path = tmp_path / "run.dcd"
    with boonza.open_writer(path, s.natoms) as w:
        for x in frames:
            w.write(x, box=s.cell)
    assert main(["sites", str(structure), "--traj", str(path), "--features"]) == 0
    printed = capsys.readouterr().out
    head = next(line for line in printed.splitlines() if line.split()[:2] == ["site", "occupied"])
    # one run, so no runs column; no --interval-ns, so no rates
    assert head.split() == ["site", "occupied", "pocket", "burial", "philic", "score",
                            "Dscore", "copies", "arrivals", "centre"]  # fmt: skip
    rows = [line.split() for line in printed.splitlines()
            if line[:4].strip().isdigit() and "%" in line]  # fmt: skip
    assert len(rows) == 1  # the parked copy; the wanderer is bulk
    held, volume, unit, burial, philic, score, drug = rows[0][1:8]
    assert unit == "A^3" and float(volume) >= 20.0  # a pocket, not a handful of cells
    assert float(held.rstrip("%")) > 75.0
    assert float(burial) > 0.9  # in a cavity: every way out of it meets the protein
    assert float(philic) == 0.0  # a benzene asks for nothing polar
    assert float(score) > 0.0 and float(drug) > 0.0
    # and the table comes first, with the hotspots the same pass found after it
    assert printed.index(head) < printed.index("hotspots")
    assert "SiteMap's shape" in printed

    capsys.readouterr()
    assert main(["sites", str(structure), "--traj", str(path)]) == 0
    plain = capsys.readouterr().out
    assert "score" not in plain.split("centre")[0]  # no second pass, no scores
    assert "--features adds" in plain


def test_the_sites_command_says_how_to_look_at_a_run(tmp_path, swimming, capsys):
    """One directory, named once, and a line per viewer with the session command
    on it.  view.dms is what to open, whatever the model wrote it."""
    import shutil

    from boonza.cli import _still_named, main

    s, runs = swimming
    d = _run_directory(tmp_path, s, runs[0])
    out = tmp_path / "out"
    assert main(["sites", "--workdir", str(d), "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    # the paths as they were given, so a relative workdir stays relative
    assert f"vizard {d / 'solvated.dms'} {d / 'trajectory.dcd'}" in printed  # no view file
    assert "D=" not in printed
    assert f"# then: source {out / 'sites.tcl'}" in printed
    assert f"# then: @{out / 'sites.pml'}" in printed
    # the maps are named beside the script, and the script finds itself, so the
    # directory can be copied off the machine that wrote it -- which is how a
    # cluster's results are read.  An absolute path would name the cluster
    tcl, pml = ((out / f"sites.{x}").read_text() for x in ("tcl", "pml"))
    assert "mol new density.dx type dx" in tcl
    assert "load density.dx, density" in pml
    assert "info script" in tcl and "__script__" in pml  # each finds its own directory
    for text in (tcl, pml):
        named = [ln for ln in text.splitlines()
                 if str(out.resolve()) in ln and not ln.lstrip().startswith("#")]  # fmt: skip
        assert not named, named  # nowhere but the comment that says where it was written

    shutil.copy2(d / "solvated.dms", d / "view.dms")  # what a run writes for a viewer
    capsys.readouterr()
    assert main(["sites", "--workdir", str(d), "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    assert f"pizard {d / 'view.dms'} {d / 'trajectory.dcd'}" in printed
    # an all-atom run renames nothing, so its view file is always the one to open;
    # a coarse-grained view written before boonza stopped renaming the backbone
    # holds beads a viewer mis-bonds, and such a run is sent back to solvated.dms
    assert _still_named(d / "view.dms", [])
    assert not _still_named(d / "view.dms", ["BB", "GC"])
    assert _still_named(d / "nothing.dms", ["BB"])  # unreadable: let the viewer say so


def test_the_strip_hint_is_the_same_line_for_every_model(tmp_path, swimming, capsys):
    """One list, Martini's and SIRAH's and an all-atom box's together, so a line
    copied from one run still strips the right things in another.  Naming only
    what this box holds would be shorter and would quietly keep 2 216 waters the
    moment it was pasted into a run of another model."""
    from boonza.cli import _VIEWER_SOLVENT, main

    s, runs = swimming
    d = _run_directory(tmp_path, s, runs[0])
    assert main(["sites", "--workdir", str(d), "-o", str(tmp_path / "out")]) == 0
    printed = capsys.readouterr().out
    assert f'--strip "resname {" ".join(_VIEWER_SOLVENT)}"' in printed
    assert f'--strip "resn {"+".join(_VIEWER_SOLVENT)}"' in printed
    # Martini's water and ions, SIRAH's, and an all-atom box's, though this one
    # is an all-atom system with no solvent in it at all
    assert {"W", "ION"} <= set(_VIEWER_SOLVENT)  # Martini
    assert {"WT4", "NaW", "ClW"} <= set(_VIEWER_SOLVENT)  # SIRAH
    assert {"HOH", "TIP3", "NA", "CL"} <= set(_VIEWER_SOLVENT)  # all-atom
    # a structural metal is not solvent: a run that holds one wants to see it
    assert not {"MG", "ZN", "CAL", "CA"} & set(_VIEWER_SOLVENT)


def test_a_hole_in_a_pocket_is_filled_but_a_channel_is_not():
    """A cell no atom happened to visit, in the middle of a pocket, is still inside
    the pocket.  A gap that still opens to the box is a way out, not a hole, so
    two regions with a channel between them stay two regions."""
    from boonza.sites import _fill_enclosed

    dims = np.array([9, 9, 9])
    free = np.ones(dims.prod(), bool)
    flat = lambda i, j, k: (i * dims[1] + j) * dims[2] + k  # noqa: E731

    shell = [flat(i, j, k) for i in (3, 4, 5) for j in (3, 4, 5) for k in (3, 4, 5)
             if (i, j, k) != (4, 4, 4)]  # fmt: skip
    filled = _fill_enclosed(np.array(sorted(shell)), dims, free)
    assert set(filled.tolist()) == set(shell) | {flat(4, 4, 4)}  # the one cell inside it

    # the same shell with a cell of its wall missing: the middle now reaches the
    # box, so nothing is sealed and nothing is taken
    leaky = [c for c in shell if c != flat(3, 4, 4)]
    assert set(_fill_enclosed(np.array(sorted(leaky)), dims, free).tolist()) == set(leaky)

    # and a cell the protein sits in is never taken, however enclosed it is
    walled = free.copy()
    walled[flat(4, 4, 4)] = False
    assert set(_fill_enclosed(np.array(sorted(shell)), dims, walled).tolist()) == set(shell)


def test_a_pocket_map_is_a_mask_of_the_volume_reported(tmp_path, buried, capsys):
    """An isosurface of enrichment inside a pocket shows the cells the probes
    visited most, which looks like scatter; the question a pocket map answers is
    where the pocket is.  So it holds one value, drawn at half of it."""
    from boonza.cli import main
    from boonza.sites import POCKET_LEVEL

    s, frames = buried
    structure = tmp_path / "s.dms"
    boonza.save(s, structure)
    path = tmp_path / "run.dcd"
    with boonza.open_writer(path, s.natoms) as w:
        for x in frames:
            w.write(x, box=s.cell)
    out = tmp_path / "out"
    assert main(["sites", str(structure), "--traj", str(path), "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    volume = float(printed.split("A^3")[0].split()[-1])

    lines = (out / "pocket0.dx").read_text().splitlines()
    values = np.array([float(x) for line in lines[7:-1] for x in line.split()])
    assert set(np.unique(values).tolist()) == {0.0, 1.0}
    assert float((values == 1.0).sum()) == volume  # at 1 A spacing, a cell is a cubic A
    assert not (out / "pocket1.dx").exists()  # the wanderer has no pocket

    # and the viewer scripts draw it at half of that one value, as a surface
    tcl = (out / "sites.tcl").read_text()
    pml = (out / "sites.pml").read_text()
    assert f"mol modstyle 0 top Isosurface {POCKET_LEVEL:g}" in tcl
    assert f"isosurface pocket0, pocket0_map, {POCKET_LEVEL:g}" in pml
    assert "isomesh density_mesh, density, 50" in pml  # the other maps keep their level


def test_a_pocket_is_held_together_by_faces_not_corners():
    """Two cells meeting at a corner share no volume and are no way through for a
    molecule, and a surface drawn through them comes out as two pieces pinched at
    a point -- which is what a pocket looked like in a viewer."""
    from boonza.sites import _join_neighbours

    dims = np.array([6, 6, 6])
    flat = lambda i, j, k: (i * dims[1] + j) * dims[2] + k  # noqa: E731
    corner = np.array(sorted([flat(1, 1, 1), flat(2, 2, 2)]))
    _, by_corner = _join_neighbours(corner, dims)
    _, by_face = _join_neighbours(corner, dims, faces=True)
    assert by_corner == 1 and by_face == 2

    touching = np.array(sorted([flat(1, 1, 1), flat(2, 1, 1)]))  # a shared face
    assert _join_neighbours(touching, dims, faces=True)[1] == 1


def test_the_pocket_a_site_claims_is_one_solid(buried):
    """Every cell of it can be reached from every other through shared faces, so a
    surface drawn at the mask's half value is one closed object."""
    from boonza.sites import _join_neighbours

    s, frames = buried
    found = boonza.sites(s, [frames], pocket_protein="protein")
    site = found[0]
    assert site.volume >= 20.0 and site.cells is not None
    group, pieces = _join_neighbours(site.cells, site.grid_dims, faces=True)
    assert pieces == 1
    assert len(site.cells) * site.grid_spacing**3 == site.volume


def test_a_column_with_nothing_in_it_keeps_its_width():
    """A site with no pocket has no burial and a site nothing left has no rate, and
    a dash as wide as the number would have been is what keeps the table a table."""
    from boonza.cli import _cell

    for spec in ("6.2f", "+6.2f", "9.1e", "5d", "7.1f"):
        wide = len(_cell(1.5 if spec[-1] in "fe" else 2, spec))
        assert _cell(None, spec) == "-".rjust(wide)
        assert len(_cell(float("nan"), spec)) == wide
        assert len(_cell(float("inf"), spec)) == wide


def test_a_virtual_site_is_part_of_its_molecule_not_a_copy_of_its_own():
    """Martini 3's tryptophan carries a virtual site, one bead of its ring, placed
    from the others rather than bonded to them.  No bond holds it, so counting
    molecules by bonds alone makes it a probe copy that does not exist: its own
    centroid, its own features, and one more copy against every site it is near.
    """
    from boonza.martini.probes import probe
    from boonza.symmetry import molecules_of

    s = probe("EW").system()
    ids = np.arange(s.natoms)
    anum = np.asarray(s.atoms["anum"])
    assert len(np.unique(np.asarray(s.fragids))) == 2  # the bonds say two
    assert (anum == 0).sum() == 1  # the one with no mass and no bonds
    molecules = molecules_of(s, ids)
    assert len(np.unique(molecules)) == 1  # one probe, as it was built
    loose = int(np.flatnonzero(anum == 0)[0])
    ring = [int(a) for a in ids if str(s.atoms["name"][a]).startswith("SC")]
    assert molecules[loose] == molecules[ring[0]]  # with the ring it sits in

    whole = probe("FF").system()  # a probe with no virtual site is untouched
    assert np.array_equal(molecules_of(whole, np.arange(whole.natoms)),
                          np.asarray(whole.fragids))  # fmt: skip


def test_two_clusters_in_one_pocket_are_one_site(buried):
    """A pocket big enough holds a molecule in two spots a few angstroms apart, and
    the centroids cluster twice.  That is one site: the pocket is what a ligand
    would occupy, so the clusters in it are merged rather than reported as two
    sites with the same volume, the same burial and the same score."""
    s, frames = buried
    lig = s.select(DEFAULT_LIGAND).ids
    frag = np.asarray(s.fragids)[lig]
    copies = [lig[frag == f] for f in np.unique(frag)]
    # the parked copy sits in two spots of its cavity, half the frames in each
    moved = frames.copy()
    for k in range(0, len(moved), 2):
        moved[k, copies[0]] = moved[k, copies[0]] + np.array([2.4, 0.0, 0.0])
    found = boonza.sites(s, [moved], pocket_protein="protein")
    pockets = [tuple(site.cells.tolist()) for site in found if site.cells is not None]
    assert len(pockets) == len(set(pockets))  # no pocket is reported twice
    held = [site for site in found if site.volume]
    assert len(held) == 1  # one pocket, one site
    assert held[0].occupancy > 0.75  # and it holds the frames of both spots


def test_sites_are_ranked_by_their_pocket_not_by_dwell(buried):
    """Dwell is the tempting ranking and the wrong one: a sticky patch of surface
    holds something for most of a run without being anywhere a ligand fits.  On the
    one protein where the answer is known, ranking by the pocket put the crystal
    ligand's site first where dwell put it second of two and twelfth of 25."""
    from boonza.sites import _RANKS

    assert set(_RANKS) == {"pocket", "occupied", "agreement", "burial"}
    s, frames = buried
    found = boonza.sites(s, [frames], pocket_protein="protein")
    assert found.sites  # the default is the pocket, and it is a real ordering
    keys = [_RANKS["pocket"](site) for site in found]
    assert keys == sorted(keys)
    # the key is room times enclosure, so a bigger, more enclosed pocket comes first
    a, b = _RANKS["pocket"], _RANKS["occupied"]
    one, two = found[0], found[0]
    assert a(one)[0] == -one.volume * one.burial
    assert b(two)[0] == -two.occupancy


def test_how_far_the_protein_moved_is_reported(tmp_path, swimming, capsys):
    """Every site and every pocket is measured in the reference's frame, so a run
    whose protein changes shape measures them against a shape it has left behind.
    The fit is done anyway, so what it cost to make the frames line up is free to
    report -- on screen, and frame by frame in rmsd.csv."""
    import csv

    from boonza.cli import main

    s, runs = swimming
    bent = runs[0].copy()
    # the protein drifts apart over the run, as a coarse-grained one without a
    # network does: every backbone atom moves out from the centre
    prot = s.select("protein").ids
    middle = np.asarray(s.positions)[prot].mean(0)
    for k in range(len(bent)):
        push = 0.06 * k  # a few per cent a frame: by the end it has swollen
        bent[k, prot] = bent[k, prot] + push * (bent[k, prot] - middle) / 10.0
    d = _run_directory(tmp_path, s, bent)
    out = tmp_path / "out"
    assert main(["sites", "--workdir", str(d), "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "the protein moved" in printed and "shape the run no longer has" in printed
    assert "--dihedral-restraint ss" in printed  # and what to do about it

    rows = list(csv.DictReader((out / "rmsd.csv").read_text().splitlines()))
    assert len(rows) == len(bent)
    assert [r["run"] for r in rows] == ["0"] * len(bent)
    assert [int(r["frame"]) for r in rows] == list(range(len(bent)))
    walk = [float(r["rmsd_A"]) for r in rows]
    assert walk[0] < walk[-1] and walk[-1] > 2.0  # it wanders, and the warning fired

    # a run that holds its shape says so instead
    steady = _run_directory(tmp_path / "steady", s, runs[0])
    capsys.readouterr()
    assert main(["sites", "--workdir", str(steady), "-o", str(tmp_path / "out2")]) == 0
    printed = capsys.readouterr().out
    assert "the protein stayed within" in printed
    assert "no longer has" not in printed


@pytest.mark.parametrize(("version", "forcefield", "radii"), [
    (2, "martini22", [2.15, 2.35]),       # Martini 2's small bead is 0.43 nm, the rest 0.47
    (3, "martini3001", [1.7, 2.05, 2.35]),  # Martini 3's are 0.34, 0.41 and 0.47
])  # fmt: skip
def test_a_bead_is_as_wide_as_its_own_pair(version, forcefield, radii):
    """How much room a bead takes, for the map the pockets are measured on.

    Both Martinis write their Lennard-Jones per pair of types rather than per
    type, so a bead's own size is the pair it makes with itself; neither carries
    a size in ``[ atomtypes ]`` at all.  They write it differently, though --
    Martini 2.2 declares combination rule 1 and gives C6 and C12, Martini 3
    rule 2 and gives sigma and epsilon -- and a radius must come out the same
    either way, which is the bead sizes the papers name.
    """
    from boonza.martini import NONBONDED_FOR, martinize, parameters
    from boonza.sites import RMIN, particle_radii

    s = boonza.load(DATA / "2TRX.pdb").select("protein and chain A").clone()
    cg = martinize(s, forcefield=forcefield).system(parameters(NONBONDED_FOR[version])[0])
    ids = np.arange(cg.natoms)
    sigma = particle_radii(cg, ids, "sigma")
    assert sorted(np.unique(np.round(sigma, 4)).tolist()) == radii
    assert np.allclose(particle_radii(cg, ids, "rmin"), sigma * RMIN)


def test_what_counts_as_a_pocket_can_be_asked_for(swimming, tmp_path):
    """``buried`` and ``min_volume`` decide what is a pocket rather than a dent,
    and so how long a list a run comes back with -- which is what a hit has to be
    found in, so they are the analysis's to set and not constants in it."""
    import inspect

    from boonza.sites import BURIED, MIN_VOLUME, sites

    args = inspect.signature(sites).parameters
    assert args["buried"].default == BURIED
    assert args["min_volume"].default == MIN_VOLUME
    s, runs = swimming
    loose = sites(s, runs, pocket_protein="protein", min_volume=1.0, buried=0.0)
    tight = sites(s, runs, pocket_protein="protein", min_volume=400.0, buried=0.9)
    # a pocket has to clear both, so asking for more leaves fewer sites with one
    assert sum(1 for x in loose if x.volume) >= sum(1 for x in tight if x.volume)
    assert all(x.volume >= 400.0 for x in tight if x.volume)
    assert all(x.burial >= 0.9 for x in tight if x.volume)


def test_how_a_pocket_is_scored_against_a_known_ligand():
    """DCA, DCC and coverage, which is how a pocket is judged against a crystal.

    DCA and DCC are the benchmarks' two measures -- the pocket's centre to the
    nearest ligand atom, and to its centroid -- and both pass a pocket that only
    touches one end of the ligand.  Coverage is the one that asks whether the
    pocket holds the thing: the share of its atoms inside the pocket.
    """
    from boonza.sites import coverage, dca, dcc

    # a slab of pocket cells, and a ligand lying along it with one atom off the end
    cells = np.array([[x, y, 0.0] for x in range(-4, 5) for y in (-1.0, 0.0, 1.0)])
    ligand = np.array([[x, 0.0, 0.0] for x in range(-3, 4)] + [[20.0, 0.0, 0.0]])
    assert dca(cells, ligand) == pytest.approx(0.0, abs=1e-9)  # a cell sits on an atom
    assert dcc(cells, ligand) == pytest.approx(np.linalg.norm(ligand.mean(0)))
    assert coverage(cells, ligand) == pytest.approx(7 / 8)  # all but the far one
    assert coverage(cells, ligand, within=0.1) == pytest.approx(7 / 8)
    # a pocket beside the ligand rather than around it passes DCA and little else
    beside = np.array([[-4.0, 2.0, 0.0], [-4.0, 2.0, 1.0]])
    assert dca(beside, ligand) < 3.0
    assert coverage(beside, ligand) == 0.0
    assert coverage(np.empty((0, 3)), ligand) == 0.0
    assert np.isnan(dca(cells, np.empty((0, 3))))


def test_a_site_hands_back_the_cells_of_its_pocket():
    """The pocket's cells as coordinates, which is what scoring one needs."""
    from boonza.sites import Site

    site = Site(center=np.zeros(3), points=np.empty(0, np.int64), occupancy=0.0,
                copy_frames=0.0, runs=1, copies=1, arrivals=0, spread=0.0)  # fmt: skip
    assert not len(site.pocket_points())  # no pocket, no cells
    assert np.allclose(site.pocket_center(), site.center)  # falls back on the site's own
    dims = np.array([4, 4, 4])
    site.cells = np.array([int(np.ravel_multi_index((1, 1, 1), tuple(dims)))])
    site.grid_dims, site.grid_origin, site.grid_spacing = dims, np.zeros(3), 2.0
    assert np.allclose(site.pocket_points(), [[3.0, 3.0, 3.0]])  # (1 + 0.5) * 2
    assert np.allclose(site.pocket_center(), [3.0, 3.0, 3.0])


def test_which_edge_a_pocket_is_measured_from_is_asked_for(swimming):
    """``shell`` is off by default, because it is not free.

    Measuring from the surfaces keeps a pocket out of the beads, which measuring
    from the centers does not -- but over 181 coarse-grained runs the two measures
    of a hit disagreed about it, ligand coverage falling (top-1 51 to 48) where
    DCA rose (50 to 54), so the default stays where it was and the correctness is
    asked for.
    """
    import inspect

    from boonza.sites import SHELL, SHELL_SURFACE, particle_radii, sites

    assert inspect.signature(sites).parameters["shell"].default == "center"
    assert SHELL[0] > SHELL_SURFACE[0]  # a center is further out than a surface
    s, runs = swimming
    with pytest.raises(ValueError, match="center"):
        sites(s, runs, pocket_protein="protein", shell="from the surface, please")
    plain = sites(s, runs, pocket_protein="protein")
    asked = sites(s, runs, pocket_protein="protein", shell="surface")
    # the same sites either way: what the shell changes is the pocket cut around
    # them, and in particular whether any of it is inside the protein
    assert {tuple(x.center.round(3)) for x in asked} == \
        {tuple(x.center.round(3)) for x in plain}  # fmt: skip

    ids = np.asarray(s.select("protein"), int)
    xyz, radii = np.asarray(s.positions)[ids], particle_radii(s, ids, "sigma")
    cells = {}
    for name, found in (("center", plain), ("surface", asked)):
        pts = np.concatenate([x.pocket_points() for x in found if len(x.pocket_points())])
        gap = (np.linalg.norm(pts[:, None] - xyz[None], axis=2) - radii[None]).min(axis=1)
        assert (gap >= 0.0).all()  # no cell inside a particle either way, here
        cells[name] = len(pts)
    # it does something, and which way is the protein's own business: this one is
    # all-atom, where a surface lies 1.1 to 1.7 A out and a band from the centers
    # at 2 A is the cautious one.  A Martini bead reaches 2.35 and it is the other
    # way around, which is the whole point -- see the test below.
    assert cells["surface"] != cells["center"]


def test_a_near_edge_can_be_asked_for_not_at_all(swimming):
    """``shell="none"`` keeps every enriched cell near the protein, however close.

    A cell's counts say a probe was there, and the protein they are measured
    against is one structure out of a run that moved, so a near edge of any kind
    throws sampling away for the reference's convenience.  Without one nothing is
    thrown away -- and pockets merge through a thin wall of protein, cells inside
    a bead touching the open cells on either side of it, which is the price.
    """
    from boonza.sites import SHELL, SHELL_ANY, sites

    assert SHELL_ANY[0] == -np.inf  # no near edge at all
    assert SHELL_ANY[1] == SHELL[1]  # the far edge is the one thing it keeps
    s, runs = swimming
    plain = sites(s, runs, pocket_protein="protein")
    kept = sites(s, runs, pocket_protein="protein", shell="none")
    # nothing a near edge would have dropped is dropped: every pocket cell of
    # the default run is still in a pocket, and there are more of them
    assert set(np.concatenate([x.cells for x in plain if len(x.cells)]).tolist()) <= \
        set(np.concatenate([x.cells for x in kept if len(x.cells)]).tolist())  # fmt: skip
    assert sum(x.volume for x in kept) > sum(x.volume for x in plain)
    # and the far edge still holds: nothing out in bulk
    ids = np.asarray(s.select("protein"), int)
    xyz = np.asarray(s.positions)[ids]
    for site in kept:
        pts = site.pocket_points()
        far = np.linalg.norm(pts[:, None] - xyz[None], axis=2).min(axis=1)
        assert (far <= SHELL_ANY[1]).all()


def test_a_pocket_keeps_out_of_the_beads_it_is_measured_against():
    """A pocket is room a probe could occupy, so none of it lies inside the protein.

    The near edge of the band a pocket lies in has to know how wide a particle
    is.  Measured from the centers, as it was, 2 A clears a heavy atom's 1.7 but
    lies 0.2 inside a Martini bead's 2.35 -- and the filling that closes a
    pocket's holes was bounded by the same figure, so it walked inside the beads
    and the pocket came out drawn over the protein.  That is visible even with
    the probes binned as points, where nothing else can put a cell there.
    """
    from boonza.sites import SHELL_SURFACE as SHELL
    from boonza.sites import Occupancy

    # two rows of beads with a groove between them, and a probe that sat in it
    rows = np.array([[x, y, 0.0] for x in np.arange(-12.0, 12.1, 4.7) for y in (-4.0, 4.0)])
    radii = np.full(len(rows), 2.35)
    groove = np.array([[x, 0.0, 0.0] for x in np.arange(-8.0, 8.1, 1.0)])
    grid = Occupancy(rows, spacing=1.0, margin=10.0)
    for _ in range(40):  # enriched, and often enough to beat the floor of 2
        grid.add(groove)
    cells, _, _, _, _ = grid.pockets(2.0, 1e5, rows, shell=SHELL, radii=radii,
                                     buried=0.0, min_volume=4.0)  # fmt: skip
    assert len(cells)
    clear = grid.clearance(rows, radii, SHELL[1])
    assert clear[cells].min() >= SHELL[0]  # nothing inside a bead, nor hugging one
    # and what a band from the centers would have allowed: cells inside the beads
    from boonza.spatial import min_dist2

    centers = np.sqrt(min_dist2(grid.cell_centres(), rows, SHELL[1] + 1.0, cell=None))
    assert (clear[centers >= 2.0] < 0.0).any()  # 2 A from a center is inside a 2.35 A bead
    # measured from a particle's surface, which is what clearance is
    assert clear[np.argmin(np.linalg.norm(grid.cell_centres() - rows[0], axis=1))] \
        == pytest.approx(-2.35, abs=grid.spacing)  # fmt: skip


@pytest.mark.parametrize(
    "frames,interval,first,until,every,want",
    [
        (2000, 0.1, None, 50, None, (0, 500, 1)),         # the first quarter
        (2000, 0.1, 100, None, None, (999, 2000, 1)),     # from 100 ns on, inclusive
        (2000, 0.1, 50, 100, None, (499, 1000, 1)),       # a middle window
        (2000, 0.1, None, None, 1, (0, 2000, 10)),        # every tenth frame
        (200, 1.0, None, None, 1, (0, 200, 1)),           # already that coarse
        (2000, 0.1, None, 1e9, None, (0, 2000, 1)),       # past the end is the end
    ],
)  # fmt: skip
def test_which_frames_a_window_asks_for(frames, interval, first, until, every, want):
    """``--from-ns``, ``--until-ns`` and ``--every-ns`` in frames.

    A frame is written every interval, the first at the interval and not at zero,
    so the frame holding time t is t/interval - 1.  Asking for 100 ns of a run
    reported every 0.1 is 1000 frames, not 1001, and asking from 100 ns starts at
    the frame that holds 100.0.
    """
    from boonza.cli import _window_of

    got = _window_of(frames, interval, first, until, every)
    assert (got.start or 0, got.stop, got.step or 1) == want
    assert _window_of(frames, interval, None, None, None) == slice(None)  # nothing cut


def test_a_window_needs_to_know_the_interval():
    """Frames are numbered and the flags are in nanoseconds, so without the one
    the other cannot be honoured -- and guessing it would be wrong by whatever
    the run actually used."""
    from boonza.cli import _window_of

    with pytest.raises(ValueError, match="--interval-ns"):
        _window_of(2000, None, None, 50, None)
    with pytest.raises(ValueError, match="finer than the run"):
        _window_of(200, 1.0, None, None, 0.1)  # one frame every 0.1 of a 1 ns run
    with pytest.raises(ValueError, match="no frames between"):
        _window_of(200, 1.0, 150, 100, None)  # the window is backwards


def test_sites_reads_part_of_a_run(tmp_path, swimming, capsys):
    """``--until-ns`` reads the frames it asked for and no others.

    The point is a run analysed over part of itself: what 200 ns found that 50
    would not have.  The trajectory slices lazily, so this is also the cheapest
    thing in the analysis -- a quarter of the frames is a quarter of the work.
    """
    from boonza.cli import main

    s, runs = swimming
    structure = tmp_path / "s.dms"
    boonza.save(s, structure)
    path = tmp_path / "run.dcd"
    with boonza.open_writer(path, s.natoms) as w:
        for x in runs[0]:
            w.write(x, box=s.cell)
    whole = len(runs[0])
    part = str(0.1 * (whole // 4))
    assert main(["sites", str(structure), "--traj", str(path), "--interval-ns", "0.1",
                 "--until-ns", part, "-o", str(tmp_path / "part")]) == 0  # fmt: skip
    assert f"over {whole // 4} of {whole} frames" in capsys.readouterr().out


def test_a_stride_moves_the_interval_the_rates_are_measured_against(tmp_path, swimming):
    """``--every-ns`` is in nanoseconds, not frames, because the spacing is what
    the dwell and the rates are divided by: a stride that left the interval where
    it was would report a residence time short by exactly the stride.  So the
    flag is the new spacing, and reading one frame in five makes a 0.1 ns run a
    0.5 ns one as far as everything downstream is concerned.
    """
    from argparse import Namespace

    from boonza.cli import _over_ns

    s, runs = swimming
    path = tmp_path / "run.dcd"
    with boonza.open_writer(path, s.natoms) as w:
        for x in runs[0]:
            w.write(x, box=s.cell)
    whole = len(runs[0])
    traj = boonza.open_trajectory(str(path), s)
    args = Namespace(from_ns=None, until_ns=None, every_ns=0.5, interval_ns=0.1)
    (cut,) = _over_ns([traj], args)
    assert args.interval_ns == 0.5  # what the rates will be measured against
    assert len(cut) == len(range(0, whole, 5))  # one frame in five, and lazily
    # and a window and a stride together: half the run, one frame in five of it
    args = Namespace(from_ns=None, until_ns=0.1 * (whole // 2), every_ns=0.5, interval_ns=0.1)
    (cut,) = _over_ns([traj], args)
    assert args.interval_ns == 0.5
    assert len(cut) == len(range(0, whole // 2, 5))


def test_one_temperature_for_running_and_for_scoring(tmp_path, swimming, capsys):
    """A dG is in kT, so the analysis has to use the temperature the probes swam
    at.  The simulation's default was 298 K and the analysis's was 310, which is
    four percent of every dG and KD reported, so the analysis now takes the run's
    own and falls back on the same default rather than a different one."""
    import inspect

    from boonza.cli import _temperature_of, main
    from boonza.kinetics import kinetics
    from boonza.md.config import DEFAULTS, TEMPERATURE

    assert DEFAULTS["temperature"] == TEMPERATURE  # boonza md and boonza sites agree
    assert inspect.signature(kinetics).parameters["temperature"].default == TEMPERATURE

    # a run says what it was at, and the analysis reads it rather than being told
    run = tmp_path / "md"
    run.mkdir()
    (run / "md.toml").write_text("temperature = 310.0\nproduction_report_interval_ns = 0.1\n")
    assert _temperature_of([str(run)]) == 310.0
    assert "310 K" in capsys.readouterr().out
    # two runs that disagree are not guessed at
    other = tmp_path / "md2"
    other.mkdir()
    (other / "md.toml").write_text("temperature = 298.0\n")
    assert _temperature_of([str(run), str(other)]) is None
    assert "different temperatures" in capsys.readouterr().out

    # and with no run to ask, the default is the one the simulation uses
    s, runs = swimming
    structure = tmp_path / "s.dms"
    boonza.save(s, structure)
    path = tmp_path / "run.dcd"
    with boonza.open_writer(path, s.natoms) as w:
        for x in runs[0]:
            w.write(x, box=s.cell)
    out = tmp_path / "out"
    assert main(["sites", str(structure), "--traj", str(path), "--interval-ns", "0.1",
                 "-o", str(out)]) == 0  # fmt: skip
    # nothing to read it from, so nothing is claimed about where it came from
    assert "their own temperature" not in capsys.readouterr().out
    rows = list(csv.DictReader((out / "sites.csv").open()))
    assert all(float(r["interval_ns"]) == 0.1 for r in rows)


def test_a_holo_ligand_in_a_chain_of_its_own():
    """Maestro writes a ligand as a chain of its own, so no chain of the holo
    structure has both it and a protein to fit with.  The copy of the protein it
    touches is the one to fit on -- the other copy would put it somewhere else.
    """
    from boonza.sites import known_ligand

    holo = boonza.load(str(DATA / "1HHO.pdb"))  # two copies, a HEM bound to each
    held = next(r for r in holo.residues
                if r.name.strip() == "HEM" and r.chain.name == "A")  # fmt: skip
    apart = holo.add_chain()
    apart.name = "X"
    held.chain, held.name = apart, "LIG"
    reference = boonza.load(str(DATA / "1HHO.pdb")).select("chain A and protein").clone()

    where, how = known_ligand(holo, reference, ligand="resname LIG")
    assert how["chain"] == "A"  # the copy it sits in, not the other one
    assert how["ligand_chain"] == "X"  # and it was carried from there
    assert how["atoms"] == len(held.atoms)
    assert how["fit_rmsd"] < 1e-3  # the reference is that chain itself
    assert len(where) == how["atoms"]


def test_a_holo_ligand_still_fits_in_the_protein_s_own_chain():
    """The ordinary case is untouched: a ligand written in the protein's chain
    is paired with it without looking at what else the structure holds."""
    from boonza.sites import known_ligand

    holo = boonza.load(str(DATA / "1HHO.pdb"))
    for r in holo.residues:
        if r.name.strip() == "HEM" and r.chain.name == "A":
            r.name = "LIG"
    reference = boonza.load(str(DATA / "1HHO.pdb")).select("chain A and protein").clone()

    where, how = known_ligand(holo, reference, ligand="resname LIG")
    assert how["chain"] == "A"
    assert "ligand_chain" not in how  # it never had to go looking


def test_the_peak_of_a_site_is_where_the_ligand_sat_most():
    """A site's centre is the mean of everywhere the ligand sat, which a long
    pocket pulls off the place it actually sat.  DPA measures from the mode."""
    from boonza.sites import Site, _densest, dca, dpa

    rng = np.random.default_rng(0)
    # a crowded lobe at the origin and a thin tail running out to x = 20: the
    # mean lands in the middle of the tail, the mode stays on the lobe
    lobe = rng.normal(0.0, 0.5, size=(400, 3))
    tail = np.stack([np.linspace(2.0, 25.0, 150), np.zeros(150), np.zeros(150)], axis=1)
    xyz = np.vstack([lobe, tail])
    peak = _densest(xyz, 1.0)
    assert np.linalg.norm(peak) < 1.0  # on the lobe
    assert np.linalg.norm(xyz.mean(0)) > 3.0  # the mean is dragged down the tail

    ligand = np.zeros((1, 3))
    site = Site(center=xyz.mean(0), peak=peak, points=np.arange(len(xyz)), occupancy=1.0,
                copy_frames=1.0, runs=1, copies=1, arrivals=1, spread=1.0)  # fmt: skip
    assert dpa(site, ligand) < dca(site, ligand)
    assert dpa(site, ligand) < 1.0


def test_a_site_with_no_peak_measures_from_its_centre():
    """DPA is reported beside DCA, never instead of it, so a site built without
    a peak still answers rather than returning nothing."""
    from boonza.sites import Site, dca, dpa

    site = Site(center=np.array([3.0, 0.0, 0.0]), points=np.arange(2), occupancy=1.0,
                copy_frames=1.0, runs=1, copies=1, arrivals=1, spread=1.0)  # fmt: skip
    ligand = np.zeros((1, 3))
    assert dpa(site, ligand) == pytest.approx(dca(site, ligand))


def test_the_table_and_the_csv_carry_both_measures(tmp_path, capsys):
    """`--holo` prints a row per site and writes one too, and both carry DPA
    beside DCA.  The printed table reads what the scoring holds, so a measure
    added to one and not the other is a crash rather than a missing column."""
    import json

    from boonza.cli import main

    # long enough to fit a holo structure on: the fit needs 20 alpha carbons
    s = boonza.peptide("A" * 24)
    s.positions = np.asarray(s.positions) - np.asarray(s.positions).mean(0)
    s.append(boonza.from_smiles("c1ccccc1"))
    s.cell = np.diag([40.0, 40.0, 40.0])
    lig = np.asarray(s.select(DEFAULT_LIGAND).ids)
    base = np.asarray(s.positions).copy()
    base[lig] -= base[lig].mean(0)
    here = base[: lig[0]].mean(0) + np.array([9.0, 0.0, 0.0])  # beside the peptide

    rng = np.random.default_rng(0)
    structure = tmp_path / "s.dms"
    boonza.save(s, structure)
    path = tmp_path / "run.dcd"
    with boonza.open_writer(path, s.natoms) as w:
        for _ in range(120):
            x = base.copy()
            x[lig] = base[lig] + here + rng.normal(scale=0.4, size=3)
            w.write(x, box=s.cell)

    # the same protein with the ligand left where it sat, which is what a
    # benchmark gives --holo
    holo = boonza.peptide("A" * 24)
    holo.positions = np.asarray(holo.positions) - np.asarray(holo.positions).mean(0)
    held = boonza.from_smiles("c1ccccc1")
    held.positions = np.asarray(held.positions) - np.asarray(held.positions).mean(0) + here
    holo.append(held)
    holo_path = tmp_path / "holo.dms"
    boonza.save(holo, holo_path)

    out = tmp_path / "out"
    assert main(["sites", str(structure), "--traj", str(path), "--holo", str(holo_path),
                 "--holo-ligand", DEFAULT_LIGAND, "-o", str(out)]) == 0  # fmt: skip
    printed = capsys.readouterr().out
    assert "DCA" in printed and "DPA" in printed

    rows = list(csv.DictReader((out / "sites.csv").open()))
    assert rows and "dpa_A" in rows[0] and "dca_A" in rows[0]
    doc = json.loads((out / "sites.json").read_text())
    scored = [v for v in doc["sites"] if "dca_A" in v]
    assert scored and all("dpa_A" in v for v in scored)
    # the ligand never moved, so both measures land on it
    best = min(rows, key=lambda r: float(r["dca_A"]))
    assert float(best["dca_A"]) <= 4.0 and float(best["dpa_A"]) <= 4.0
