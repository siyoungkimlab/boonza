"""SiteMap-style sites of coarse-grained proteins: pair tables, beads, the grid and the CLI."""

import csv
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.cavity import beads as B
from boonza.cavity import grid as G
from boonza.cavity.cli import main
from boonza.cavity.forcefield import pair_table

DATA = Path(__file__).parent / "data"


def _system(model):
    s = boonza.load(str(DATA / "1TEN.pdb"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model == "sirah":
            return boonza.sirah.sirahize(s, "protein").system()
        ff = {"martini2": "martini22", "martini3": "martini3001"}[model]
        return boonza.martinize(s, "protein", forcefield=ff).system()


@pytest.mark.parametrize("model", ["martini2", "martini3", "sirah"])
def test_pairs_match_boonza(model):
    """Every pair of types in a parameterized protein has the sigma boonza gave it."""
    cg = _system(model)
    nb = cg.table("nonbonded")
    ptype = list(nb.values("type"))  # per atom
    pid = np.asarray(nb.param_ids)
    sig = nb.values("sigma")
    own = {int(p): float(s) for p, s in zip(pid, sig, strict=True)}
    name = {int(p): str(t) for p, t in zip(pid, ptype, strict=True)}
    table = pair_table(model)
    checked = 0
    for (a, b), p in nb.overrides.items():
        assert table.pair(name[int(a)], name[int(b)]) == pytest.approx(float(p["sigma"]), abs=1e-3)
        checked += 1
    if model == "sirah":  # pairs without an override follow the combining rule
        over = {tuple(sorted((int(a), int(b)))) for (a, b), _ in nb.overrides.items()}
        ids = sorted(name)
        for i in ids:
            for j in ids:
                if tuple(sorted((i, j))) in over:
                    continue
                mean = (own[i] + own[j]) / 2
                assert table.pair(name[i], name[j]) == pytest.approx(mean, abs=1e-3)
                checked += 1
    assert checked > 40


@pytest.mark.parametrize("model", ["martini2", "martini3", "sirah"])
def test_beads_typed_like_boonza(model):
    s = boonza.load(str(DATA / "1TEN.pdb"))
    beads = B.coarse_grain(s, model)
    assert len(beads) > 150 and np.isfinite(beads.radius).all()
    assert 1.5 < np.median(beads.radius) < 3.0
    if model == "sirah":  # the N-terminal backbone bead takes sirahize's own type
        assert beads.types[0] == "GNz"


@pytest.mark.parametrize("model", ["martini2", "sirah"])
def test_rays_match_direct_search(model):
    """The lattice rays equal the nearest-bead search, overlapping beads included."""
    rng = np.random.default_rng(1)
    types = {"martini2": ["P5", "SC4", "Qa"], "sirah": ["GC", "GN", "Y2Ca", "A5D"]}[model]
    xyz = rng.uniform(0, 14, (80, 3))
    xyz[40:] = xyz[:40] + rng.normal(0, 1.5, (40, 3))  # pairs that overlap
    beads = B.from_arrays(model, xyz, rng.choice(types, 80))
    groups = G._by_radius(beads)
    pts = rng.uniform(-4, 18, (600, 3))
    np.testing.assert_array_equal(G._rays(pts, groups), G._rays_direct(pts, groups))
    np.testing.assert_array_equal(
        G._rays(pts, groups, step=2.0), G._rays_direct(pts, groups, step=2.0)
    )


def test_structure_cli(tmp_path):
    out = tmp_path / "sites"
    assert main(["structure", str(DATA / "1MBN.pdb"), "--model", "martini3", "-o", str(out)]) == 0
    rows = list(csv.DictReader(open(out / "pockets.csv")))
    assert rows and rows[0]["rank"] == "1"
    p = [float(r["p"]) for r in rows]
    assert p == sorted(p, reverse=True)  # best first
    view = (out / "view.pml").read_text()
    assert "load structure.pdb, structure" in view and "pocket_1" in view
    assert not (out / "ligand.json").exists() and "create ligand" not in view
    assert "scene overview, store" in view


def test_named_presets(tmp_path):
    """volume200 is the one preset and every command's default; a presets file is used by its
    path, and its preset by the command line."""
    import json

    from boonza.cavity import presets as P

    assert P.names() == ["volume200"] and P.DEFAULT == "volume200"
    assert P.presets() == P.presets("volume200")
    v = P.presets("volume200")
    mine = {**v, "martini3": {**v["martini3"], "spacing": 3.0}}
    (tmp_path / "mine.json").write_text(json.dumps(mine))
    assert P.preset("martini3", str(tmp_path / "mine.json"))["spacing"] == 3.0
    with pytest.raises(ValueError, match="no presets"):
        P.presets("no-such-presets")
    out = tmp_path / "sites"
    assert main(["structure", str(DATA / "1MBN.pdb"), "--model", "martini3", "-o", str(out),
                 "--preset", str(tmp_path / "mine.json")]) == 0  # fmt: skip
    assert list(csv.DictReader(open(out / "pockets.csv")))
    assert main(["structure", str(DATA / "1MBN.pdb"), "--model", "martini3", "-o", str(out),
                 "--preset", "no-such-presets"]) == 1  # fmt: skip


@pytest.mark.parametrize("model", ["martini2", "martini3", "sirah"])
def test_structure_cli_finds_the_heme_pocket(model, tmp_path):
    """Myoglobin without its heme: the heme pocket is among the first sites, and --ligand
    leaves the heme out of the protein and measures every site against it."""
    out = tmp_path / "sites"
    assert main(["structure", str(DATA / "1MBN.pdb"), "--model", model, "-o", str(out),
                 "--ligand", "resname HEM"]) == 0  # fmt: skip
    rows = list(csv.DictReader(open(out / "pockets.csv")))
    assert {"PPc", "MOc", "LVC", "PVN"} <= set(rows[0])
    first = next(int(r["rank"]) for r in rows if r["PPc"] == "True")
    assert first <= 3
    # the heme stays in the structure, which ligand.json points at
    assert json.loads((out / "ligand.json").read_text()) == {
        "file": "structure.pdb",
        "selection": "(chain A and resid 155 and resname HEM)",
    }
    view = (out / "view.pml").read_text()
    assert "create ligand, structure and ((chain A and resi 155 and resn HEM))" in view
    assert not (out / "ligand.pdb").exists()


def _run(tmp_path, pdb, selection="protein"):
    """A small Martini 3 run of ``pdb``'s ``selection``: its system and three jittered
    frames."""
    from boonza.trajectory import open_writer

    run = tmp_path / "md_solute"
    run.mkdir()
    s = boonza.load(str(DATA / pdb))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cg = boonza.martinize(s, selection, forcefield="martini3001").system()
    boonza.save(cg, run / "solvated.dms")
    rng = np.random.default_rng(2)
    with open_writer(str(run / "trajectory.dcd"), cg.natoms) as w:
        for _ in range(3):
            w.write(np.asarray(cg.positions, np.float32) + rng.normal(0, 0.2, (cg.natoms, 3)), None)
    return run


def test_traj_cli(tmp_path):
    """Myoglobin's beads through three frames: sites grouped into pockets, then regrouped
    from the saved sites."""
    run = _run(tmp_path, "1MBN.pdb")
    out = tmp_path / "cavity"
    assert main(["traj", "--workdir", str(run), "--model", "martini3", "-o", str(out),
                 "-j", "1"]) == 0  # fmt: skip
    rows = list(csv.DictReader(open(out / "pockets.csv")))
    assert rows and 0 < float(rows[0]["occupancy"]) <= 1
    assert (out / "sites.pkl").exists() and (out / "view.pml").exists()
    assert {"md_frame", "best_md_frame", "centre_x", "centre_y", "centre_z"} <= set(rows[0])
    timing = json.loads((out / "timing.json").read_text())
    assert {"load_and_fit", "site_finding", "grouping", "ranking", "report"} <= set(
        timing["stages_s"]
    )
    assert timing["frames"] == 3 and not timing["site_finding"]["from_cache"]
    assert timing["site_finding"]["grid_cpu"] > 0 and timing["site_finding"]["workers"] == 1
    assert "site_finding" in (out / "timing.txt").read_text()
    # ranked by each pocket's best frame (p_max: with volume200, its largest site)
    assert [r["rank_p_max"] for r in rows] == [str(k) for k in range(1, len(rows) + 1)]
    assert all(float(r["p_q90_open"]) <= float(r["p_max"]) for r in rows)
    # a second run reuses the frames' sites, here grouped by the residues lining them
    assert main(["traj", "--workdir", str(run), "--model", "martini3", "-o", str(out),
                 "-j", "1", "--top", "3", "--grouping", "consensus"]) == 0  # fmt: skip
    assert list(csv.DictReader(open(out / "pockets.csv")))
    assert json.loads((out / "timing.json").read_text())["site_finding"]["from_cache"]


def _frame_site(frame, centre, score, residues=(1, 2, 3)):
    from boonza.cavity.traj import FrameSite

    xyz = np.asarray(centre, float) + np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], float)
    return FrameSite(frame, 1, score, {"n": 24.0}, xyz.astype(np.float32), frozenset(residues))


def test_centroid_groups():
    """Sites gather by their centres: two places 20 A apart are two pockets, whatever residues
    line them; a pocket's centre is its frames' mean and its medoid the frame site nearest it."""
    from boonza.cavity import traj as T

    sites = [_frame_site(k, (0.5 * k, 0, 0), 1.0 - 0.01 * k) for k in range(5)]
    sites += [_frame_site(k, (20, 0, 0), 2.0, residues=(1, 2, 3)) for k in range(3)]
    pockets = T.group(sites)
    assert sorted(len(p.sites) for p in pockets) == [3, 5]
    near = next(p for p in pockets if len(p.sites) == 5)
    assert np.allclose(near.centre, np.mean([s.xyz.mean(0) for s in sites[:5]], axis=0))
    assert near.medoid is sites[2]  # its centre is the mean of the five
    # by the residues instead, the two places are one pocket
    assert len(T.group(sites, "consensus")) == 1
    with pytest.raises(ValueError, match="grouping"):
        T.group(sites, "nonsense")


def test_pocket_counts_every_site():
    """A frame can give a pocket two sites, and both count for its centre, its medoid and
    its median volume; occupancy and p_max still go by frame."""
    from boonza.cavity import traj as T

    a = _frame_site(0, (0, 0, 0), 2.0)
    b = _frame_site(1, (4, 0, 0), 1.5)
    c = _frame_site(0, (2, 0, 0), 1.0)  # frame 0's second, smaller site
    for s, n in ((a, 400.0), (b, 300.0), (c, 40.0)):
        s.props["n"] = n
    p = T.Pocket()
    for s in (a, b, c):
        p.add(s)
    assert np.allclose(p.centre, np.mean([s.xyz.mean(0) for s in (a, b, c)], axis=0))
    assert p.medoid is c  # nearest the centre, though not its frame's best
    s = p.summary(2)
    assert s["median_volume"] == 300.0 and s["max_volume"] == 400.0
    assert s["frames"] == 2 and s["occupancy"] == 1.0
    assert s["p_max"] == pytest.approx(1 / (1 + np.exp(-2.0)))


def test_traj_cli_without_sites(tmp_path):
    """A protein with no site in any frame (1TEN's first 28 residues) gives an empty
    table, not an error."""
    run = _run(tmp_path, "1TEN.pdb", "protein and resid < 830")
    out = tmp_path / "cavity"
    assert main(["traj", "--workdir", str(run), "--model", "martini3", "-o", str(out),
                 "-j", "1"]) == 0  # fmt: skip
    assert list(csv.DictReader(open(out / "pockets.csv"))) == []


def test_traj_cli_with_holo(tmp_path):
    """--holo carries a holo ligand onto the run; every pocket is measured against it, and the
    heme pocket of myoglobin is among the right ones."""
    run = _run(tmp_path, "1MBN.pdb")
    out = tmp_path / "cavity"
    holo = ["--holo", str(DATA / "1MBN.pdb"), "--holo-ligand", "resname HEM"]
    assert main(["traj", "--workdir", str(run), "--model", "martini3", "-o", str(out),
                 "-j", "1", *holo]) == 0  # fmt: skip
    rows = list(csv.DictReader(open(out / "pockets.csv")))
    assert {"PPc", "MOc", "LVC", "PVN", "frames_PPc"} <= set(rows[0])
    assert any(r["PPc"] == "True" for r in rows)
    view = (out / "view.pml").read_text()
    assert "load holo.pdb, holo" in view and "create ligand, holo and (" in view
    assert json.loads((out / "ligand.json").read_text())["file"] == "holo.pdb"
    assert not (out / "ligand.pdb").exists()


def test_grid_keeps_its_ray_step(tmp_path):
    """A grid saves and loads its ray step; enclosure at 10 A reads hits in that step."""
    from boonza.cavity import sites as S

    rng = np.random.default_rng(2)
    beads = B.from_arrays("martini3", rng.uniform(0, 20, (150, 3)), ["P2"] * 150)
    g1 = G.compute(beads, 2.0)
    g2 = G.compute(beads, 2.0, ray_step=2.0)
    assert g1.ray_step == 1.0 and g2.ray_step == 2.0
    assert g2.hit[g2.hit != G.NO_HIT].max() <= 5  # 2, 4, 6, 8, 10 A
    g2.save(tmp_path / "g.npz")
    assert G.Grid.load(tmp_path / "g.npz").ray_step == 2.0
    # a 2 A walk can only miss what a 1 A walk sees (grazing hits), never the reverse
    assert np.all(S._shut(g2.hit, 10.0, 2.0) <= S._shut(g1.hit, 10.0, 1.0) + 1e-9)


def test_volume200_ranks_by_volume():
    """volume200: rays every 2 A out to 10 A, no contact test, a score ordering by size."""
    from boonza.cavity import presets as P

    for model in ("martini2", "martini3", "sirah"):
        p = P.preset(model, "volume200")
        assert p["ray_step"] == 2.0 and p["params"]["ray_length"] == 10.0
        assert set(p["score"]) == {"intercept", "sqrt_n"} and "probe" not in p
        assert P.score({"n": 400.0}, p) > P.score({"n": 40.0}, p)


def test_sites_carry_only_what_finding_computes():
    """A site's properties are its volume and its own points' enclosure, nothing more."""
    from boonza.cavity import sites as S

    rng = np.random.default_rng(3)
    beads = B.from_arrays("martini3", rng.uniform(0, 24, (220, 3)), ["P2"] * 220)
    g = G.compute(beads, 2.0, ray_step=2.0)
    found = S.find_sites(g, {"outside": 2.75, "ray_length": 10.0, "enclosure": 0.3})
    assert found and all(set(s.props) == {"n", "enclosure_site"} for s in found)


def test_an_old_grid_with_energies_loads(tmp_path):
    """A grid saved with contact energies, before they were dropped, loads without them."""
    rng = np.random.default_rng(4)
    g = G.compute(B.from_arrays("martini3", rng.uniform(0, 12, (40, 3)), ["P2"] * 40), 2.0)
    n = len(g.ijk)
    np.savez_compressed(tmp_path / "old.npz", spacing=g.spacing, origin=g.origin, ijk=g.ijk,
                        ratio=g.ratio, dmin=g.dmin, hit=g.hit, probes=np.array(["SC3"]),
                        e_polar=np.zeros((1, n)), e_apolar=np.zeros((1, n)),
                        c_polar=np.zeros((1, n)), c_apolar=np.zeros((1, n)))  # fmt: skip
    old = G.Grid.load(tmp_path / "old.npz")
    np.testing.assert_array_equal(old.hit, g.hit)
    np.testing.assert_array_equal(old.ijk, g.ijk)


def test_a_ballooned_site_is_left_out_of_grouping():
    """A frame whose site balloons past MAX_VOLUME does not join, or set, its pocket."""
    from boonza.cavity import traj as T

    xyz = np.zeros((4, 3), np.float32)
    sites = [T.FrameSite(f, 1, 0.5, {"n": 300.0}, xyz, frozenset({1})) for f in range(4)]
    sites.append(T.FrameSite(4, 1, 3.0, {"n": 5000.0}, xyz, frozenset({1})))
    (pocket,) = T.group(sites, "centroid")
    assert len(pocket.sites) == 4 and pocket.best.score == 0.5
    (pocket,) = T.group(sites, "centroid", max_volume=0)  # 0 keeps every site
    assert len(pocket.sites) == 5 and pocket.best.score == 3.0


def test_a_pocket_huge_in_every_frame_is_ranked_last():
    """A pocket whose median site is over MAX_MEDIAN_VOLUME goes after the rest under
    every ranking, however high its value; 0 ranks by value alone."""
    from boonza.cavity import traj as T

    def pocket(volumes, score, x):
        p = T.Pocket()
        for f, n in enumerate(volumes):
            p.add(T.FrameSite(f, 1, score, {"n": n}, np.full((2, 3), x, np.float32),
                              frozenset({x})))  # fmt: skip
        return p

    groove = pocket([1600.0, 1800.0, 1400.0], 5.0, 0.0)  # huge in every frame, best p_max
    small = pocket([200.0, 300.0], 1.0, 30.0)
    balloon = pocket([150.0, 150.0, 1900.0], 2.0, 60.0)  # one big frame, median small
    pockets = [groove, small, balloon]
    summaries = {id(p): p.summary(3) for p in pockets}
    for by in T.RANKINGS:
        assert T.ranked(pockets, summaries, by)[-1] is groove
    assert T.ranked(pockets, summaries, "p_max") == [balloon, small, groove]
    assert T.ranked(pockets, summaries, "p_max", 0) == [groove, balloon, small]


def test_the_old_name_still_loads():
    """boonza sitemap was renamed boonza cavity: sites.pkl caches pickled under the old
    module name load, and the old command runs."""
    import pickle

    from boonza.cavity.traj import FrameSite

    s = FrameSite(0, 1, 0.5, {"n": 10.0}, np.zeros((1, 3), np.float32), frozenset({1}))
    old = pickle.dumps(s, protocol=0).replace(b"boonza.cavity.traj", b"boonza.sitemap.traj")
    assert type(pickle.loads(old)) is FrameSite
    from boonza.cli import main as boonza_main

    with pytest.raises(SystemExit) as e:
        boonza_main(["sitemap", "--help"])
    assert e.value.code == 0


def test_frames_are_drawn_from_a_view_only_atom_for_atom():
    """cavity traj draws its frames with the run's view.dms only when that is the
    system atom for atom; one in another order (a Go run's, copied before the run
    gathered it) is left out, and the frames are the system's own."""
    from boonza.cavity.cli import _same_atoms
    from boonza.martini import martinize

    s = martinize(boonza.load(DATA / "1TEN.pdb").clone("protein")).system()
    assert _same_atoms(s.clone(), s)
    other = s.clone()
    other.reorder_atoms(np.r_[np.arange(1, s.natoms), 0])
    assert not _same_atoms(other, s)
    assert not _same_atoms(s.clone(np.arange(s.natoms - 1)), s)
