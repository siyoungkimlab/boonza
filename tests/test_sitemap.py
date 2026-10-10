"""SiteMap-style sites of coarse-grained proteins: pair tables, beads, the grid and the CLI."""

import csv
import warnings
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.sitemap import beads as B
from boonza.sitemap import grid as G
from boonza.sitemap.cli import main
from boonza.sitemap.forcefield import pair_table

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
    """Every pair of types in a parameterized protein has the sigma and epsilon boonza gave it."""
    cg = _system(model)
    nb = cg.table("nonbonded")
    ptype = list(nb.values("type"))  # per atom
    pid = np.asarray(nb.param_ids)
    sig, eps = nb.values("sigma"), nb.values("epsilon")
    own = {int(p): (float(s), float(e)) for p, s, e in zip(pid, sig, eps, strict=True)}
    name = {int(p): str(t) for p, t in zip(pid, ptype, strict=True)}
    table = pair_table(model)
    checked = 0
    for (a, b), p in nb.overrides.items():
        s, e = table.pair(name[int(a)], name[int(b)])
        assert s == pytest.approx(float(p["sigma"]), abs=1e-3)
        assert e == pytest.approx(float(p["epsilon"]), abs=1e-4)
        checked += 1
    if model == "sirah":  # pairs without an override follow the combining rule
        over = {tuple(sorted((int(a), int(b)))) for (a, b), _ in nb.overrides.items()}
        ids = sorted(name)
        for i in ids:
            for j in ids:
                if tuple(sorted((i, j))) in over:
                    continue
                s, e = table.pair(name[i], name[j])
                assert s == pytest.approx((own[i][0] + own[j][0]) / 2, abs=1e-3)
                assert e == pytest.approx(np.sqrt(own[i][1] * own[j][1]), abs=1e-4)
                checked += 1
    assert checked > 40


@pytest.mark.parametrize("model", ["martini2", "martini3", "sirah"])
def test_beads_typed_like_boonza(model):
    s = boonza.load(str(DATA / "1TEN.pdb"))
    beads = B.coarse_grain(s, model)
    assert len(beads) > 150 and np.isfinite(beads.radius).all()
    assert 1.5 < np.median(beads.radius) < 3.0
    assert 0.1 < beads.polar.mean() < 0.9
    if model == "sirah":  # the N-terminal backbone bead takes sirahize's own type
        assert beads.types[0] == "GNz"


@pytest.mark.parametrize("model", ["martini3", "sirah"])
def test_add_probes(model, tmp_path):
    """Probes added to a saved grid match the same probes computed with it."""
    rng = np.random.default_rng(0)
    types = {"martini3": ["SC3", "P2", "TC5"], "sirah": ["GC", "GN", "Y2Ca"]}[model]
    beads = B.from_arrays(model, rng.uniform(0, 12, (30, 3)), rng.choice(types, 30),
                          rng.random(30) < 0.5)  # fmt: skip
    probes = G.PROBES[model]
    whole = G.compute(beads, 2.0, probes)
    path = tmp_path / "g.npz"
    G.compute(beads, 2.0, probes[:2]).save(path)
    grown = G.add_probes(G.Grid.load(path), beads, probes)
    assert grown.probes == whole.probes
    for name in ("e_polar", "e_apolar", "c_polar", "c_apolar"):
        np.testing.assert_allclose(getattr(grown, name), getattr(whole, name), rtol=1e-5)
    assert not list(tmp_path.glob("*.partial.npz"))


@pytest.mark.parametrize("model", ["martini2", "sirah"])
def test_rays_match_direct_search(model):
    """The lattice rays equal the nearest-bead search, overlapping beads included."""
    rng = np.random.default_rng(1)
    types = {"martini2": ["P5", "SC4", "Qa"], "sirah": ["GC", "GN", "Y2Ca", "A5D"]}[model]
    xyz = rng.uniform(0, 14, (80, 3))
    xyz[40:] = xyz[:40] + rng.normal(0, 1.5, (40, 3))  # pairs that overlap
    beads = B.from_arrays(model, xyz, rng.choice(types, 80), rng.random(80) < 0.5)
    groups = G._by_radius(beads)
    pts = rng.uniform(-4, 18, (600, 3))
    np.testing.assert_array_equal(G._rays(pts, groups), G._rays_direct(pts, groups))


def test_structure_cli(tmp_path):
    out = tmp_path / "sites"
    assert main(["structure", str(DATA / "1MBN.pdb"), "--model", "martini3", "-o", str(out)]) == 0
    rows = list(csv.DictReader(open(out / "pockets.csv")))
    assert rows and rows[0]["rank"] == "1"
    p = [float(r["p"]) for r in rows]
    assert p == sorted(p, reverse=True)  # best first
    view = (out / "view.pml").read_text()
    assert "load structure.pdb, structure" in view and "pocket_1" in view
    assert "scene overview, store" in view


def test_named_presets(tmp_path):
    """static200 is kept by name; a presets file is used by its path, and its preset by the
    command line."""
    import json

    from boonza.sitemap import presets as P

    assert "static200" in P.names()
    static = P.presets("static200")
    assert {static[m]["probe"] for m in static} == {"AC2", "SC3", "Y4Cv"}
    mine = {**static, "martini3": {**static["martini3"], "spacing": 3.0}}
    (tmp_path / "mine.json").write_text(json.dumps(mine))
    assert P.preset("martini3", str(tmp_path / "mine.json"))["spacing"] == 3.0
    with pytest.raises(ValueError, match="no presets"):
        P.presets("no-such-presets")
    out = tmp_path / "sites"
    assert main(["structure", str(DATA / "1MBN.pdb"), "--model", "martini3", "-o", str(out),
                 "--preset", "static200"]) == 0  # fmt: skip
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
    assert (out / "ligand.pdb").exists()


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
    out = tmp_path / "sitemap"
    assert main(["traj", "--workdir", str(run), "--model", "martini3", "-o", str(out),
                 "-j", "1"]) == 0  # fmt: skip
    rows = list(csv.DictReader(open(out / "pockets.csv")))
    assert rows and 0 < float(rows[0]["occupancy"]) <= 1
    assert (out / "sites.pkl").exists() and (out / "view.pml").exists()
    # a second run reuses the frames' sites
    assert main(["traj", "--workdir", str(run), "--model", "martini3", "-o", str(out),
                 "-j", "1", "--top", "3"]) == 0  # fmt: skip


def test_traj_cli_without_sites(tmp_path):
    """A protein with no site in any frame (1TEN's first 28 residues) gives an empty
    table, not an error."""
    run = _run(tmp_path, "1TEN.pdb", "protein and resid < 830")
    out = tmp_path / "sitemap"
    assert main(["traj", "--workdir", str(run), "--model", "martini3", "-o", str(out),
                 "-j", "1"]) == 0  # fmt: skip
    assert list(csv.DictReader(open(out / "pockets.csv"))) == []


def test_traj_cli_with_holo(tmp_path):
    """--holo carries a holo ligand onto the run; every pocket is measured against it, and the
    heme pocket of myoglobin is among the right ones."""
    run = _run(tmp_path, "1MBN.pdb")
    out = tmp_path / "sitemap"
    holo = ["--holo", str(DATA / "1MBN.pdb"), "--holo-ligand", "resname HEM"]
    assert main(["traj", "--workdir", str(run), "--model", "martini3", "-o", str(out),
                 "-j", "1", *holo]) == 0  # fmt: skip
    rows = list(csv.DictReader(open(out / "pockets.csv")))
    assert {"PPc", "MOc", "LVC", "PVN", "frames_PPc"} <= set(rows[0])
    assert any(r["PPc"] == "True" for r in rows)
    view = (out / "view.pml").read_text()
    assert "load holo.pdb, holo" in view and (out / "ligand.pdb").exists()
