"""CTMD: the drmsd coordinate, the reversible work c(t), and the scoring."""

from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.ctmd import (
    CVError,
    Settings,
    bedroc,
    drmsd_of,
    drmsd_pairs,
    enrichment_factor,
    escape_index,
    rank,
    read_colvar,
    replica_score,
    reversible_work,
    score_pose,
)

DATA = Path(__file__).parent / "data"


def _complex():
    """1TEN with a phenol parked beside one of its alpha carbons."""
    s = boonza.load(DATA / "1TEN.pdb")
    lig = boonza.from_smiles("c1ccccc1O", seed=3, conformers=1)
    ca = s.select("protein and name CA").ids
    anchor = np.asarray(s.positions)[ca[10]]
    pos = np.asarray(lig.positions, float)
    lig.positions = pos - pos.mean(0) + anchor + [4.0, 0.0, 0.0]
    s.append(lig)
    return s


def _colvar(path, time, drmsd, work):
    with open(path, "w", encoding="utf-8") as f:
        f.write("#! FIELDS time d metad.bias metad.rct\n")
        for t, d, w in zip(time, drmsd, work, strict=True):
            f.write(f" {t:f} {d:f} {0.0:f} {w:f}\n")


# --- the reversible work ----------------------------------------------------------------


def test_reversible_work_of_a_flat_bias_is_that_bias():
    # c(t) is a ratio of two integrals of the bias; where the bias is the same
    # everywhere the ratio collapses and the work is the bias itself
    assert reversible_work(np.zeros(50), 10.0, 300.0) == pytest.approx(0.0)
    for height in (0.5, 5.0, 50.0):
        assert reversible_work(np.full(50, height), 10.0, 300.0) == pytest.approx(height)
    assert reversible_work([], 10.0, 300.0) == 0.0


def test_reversible_work_lies_under_the_peak_of_a_hill():
    s = np.linspace(0, 1.5, 200)
    bias = 12.0 * np.exp(-0.5 * ((s - 0.3) / 0.05) ** 2)
    work = reversible_work(bias, 10.0, 300.0)
    assert 0.0 < work < bias.max()
    # a taller hill is more work, and a wider one at the same height is too
    assert reversible_work(2 * bias, 10.0, 300.0) > work
    wide = 12.0 * np.exp(-0.5 * ((s - 0.3) / 0.2) ** 2)
    assert reversible_work(wide, 10.0, 300.0) > work


def test_reversible_work_matches_the_ratio_it_is_defined_as():
    rng = np.random.default_rng(0)
    bias = rng.uniform(0, 3, 64)
    gamma, temperature = 8.0, 310.0
    kT = 0.008314462618 * temperature
    want = kT * np.log(np.exp(gamma / (gamma - 1) / kT * bias).sum()
                       / np.exp(1 / (gamma - 1) / kT * bias).sum())  # fmt: skip
    assert reversible_work(bias, gamma, temperature) == pytest.approx(want)
    with pytest.raises(ValueError, match="bias factor"):
        reversible_work(bias, 1.0, temperature)


# --- the coordinate ---------------------------------------------------------------------


def test_the_drmsd_coordinate_is_boonzas_own():
    s = _complex()
    # the same pocket on both sides, and wide enough for either: 8 A, what both default
    # to, leaves this small fixture short of the 8 alpha carbons ctmd asks for
    same = {"cutoff": 10.0, "periodic": False}
    pocket, ligand, dref = drmsd_pairs(s, **same)
    assert dref.shape == (len(pocket), len(ligand)) and dref.size
    assert not np.isin(pocket, ligand).any()
    assert drmsd_of(s, **same) == pytest.approx(0.0, abs=1e-6)  # at its reference
    moved = np.asarray(s.positions, float).copy()
    moved[s.select("not (polymer or water or ions)").ids] += [1.0, 0.0, 0.0]
    mine = drmsd_of(s, moved, **same) * 10.0  # nm -> A
    theirs = boonza.drmsd(s, reference=s, positions=moved, symmetry=False, cutoff=10.0,
                          periodic=False)  # fmt: skip
    assert mine == pytest.approx(theirs.drmsd, rel=1e-5)


def test_the_drmsd_coordinate_ignores_moving_the_whole_system():
    # the point of it: no superposition, so the protein is free to move and only the
    # ligand's motion against its pocket is seen
    s = _complex()
    moved = np.asarray(s.positions, float).copy()
    moved[s.select("not (polymer or water or ions)").ids] += [0.7, -0.2, 0.1]
    before = drmsd_of(s, moved, cutoff=10.0, periodic=False)
    assert before > 0.01
    turn = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))[0]
    turn *= np.sign(np.linalg.det(turn))
    rigid = moved @ turn.T + [31.0, -7.0, 4.0]
    assert drmsd_of(s, rigid, cutoff=10.0, periodic=False) == pytest.approx(before, rel=1e-6)


def test_the_coordinate_says_what_it_cannot_be_built_from():
    s = _complex()
    with pytest.raises(CVError, match="no ligand atoms"):
        drmsd_of(s, ligand="resname NOPE", periodic=False)
    with pytest.raises(CVError, match="no pocket atom within"):
        drmsd_of(s, cutoff=0.01, periodic=False)
    # 5 A, what boonza.drmsd used to default to, leaves a handful of alpha carbons: enough
    # to measure with and too thin to bias along.  How many exactly depends on the
    # conformer RDKit built for the fixture's ligand, so it is not asserted
    with pytest.raises(CVError, match="too few to measure against"):
        drmsd_of(s, cutoff=5.0, periodic=False)
    with pytest.raises(CVError, match="pairs them by index"):
        drmsd_of(s, reference=boonza.load(DATA / "1TEN.pdb"), periodic=False)


# --- when the ligand has left -----------------------------------------------------------


def test_the_ligand_has_left_only_once_it_stays_out():
    time = np.arange(0, 1000, 10.0)
    inside = np.full(time.shape, 0.2)
    assert escape_index(time, inside, 0.6, 200.0) is None
    # a brief excursion is not a departure
    brief = inside.copy()
    brief[20:28] = 0.9
    assert escape_index(time, brief, 0.6, 200.0) is None
    # out at 400 ps and never back
    gone = inside.copy()
    gone[40:] = 0.9
    assert escape_index(time, gone, 0.6, 200.0) == 40
    # out, back, then out for good: the second window is the one
    twice = inside.copy()
    twice[10:20] = 0.9
    twice[60:] = 0.9
    assert escape_index(time, twice, 0.6, 200.0) == 60


def test_a_replica_is_scored_where_the_ligand_left_it():
    time = np.arange(0, 1000, 10.0)
    work = time / 10.0  # 0, 1, 2, ... kJ/mol
    stayed = np.full(time.shape, 0.2)
    assert replica_score(time, stayed, work, 0.6, 200.0) == (work[-1], None)
    gone = stayed.copy()
    gone[40:] = 0.9
    score, when = replica_score(time, gone, work, 0.6, 200.0)
    assert score == work[40] and when == 400.0  # the work it took, not the work after
    assert np.isnan(replica_score([], [], [], 0.6, 200.0)[0])


# --- a pose's score ---------------------------------------------------------------------


def test_a_poses_score_is_the_lowest_of_its_replicas(tmp_path):
    folder = tmp_path / "ZINC1"
    folder.mkdir()
    time = np.arange(0, 600, 10.0)
    inside = np.full(time.shape, 0.2)
    for k, top in enumerate([40.0, 25.0, 60.0], start=1):
        _colvar(folder / f"colvar_{k}.dat", time, inside, np.linspace(0, top, len(time)))
    pose = score_pose(folder)
    assert pose.name == "ZINC1" and pose.score == pytest.approx(25.0) and pose.error == 0.0
    assert len(pose.replicas) == 3 and not pose.filtered
    assert "25.0" in str(pose)
    # the files the authors' runs write are read the same way
    their = tmp_path / "theirs"
    their.mkdir()
    for k, top in enumerate([40.0, 25.0, 60.0], start=1):
        _colvar(their / f"COLVAR_{k}", time, inside, np.linspace(0, top, len(time)))
    assert score_pose(their).score == pytest.approx(25.0)


def test_more_replicas_than_asked_for_are_bootstrapped(tmp_path):
    folder = tmp_path / "many"
    folder.mkdir()
    time = np.arange(0, 600, 10.0)
    inside = np.full(time.shape, 0.2)
    tops = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    for k, top in enumerate(tops, start=1):
        _colvar(folder / f"colvar_{k}.dat", time, inside, np.linspace(0, top, len(time)))
    pose = score_pose(folder, replicas=3, bootstrap=400, seed=1)
    # the mean of the lowest of three draws out of ten: above the lowest, far below
    # the median, and with a spread of its own
    assert 10.0 < pose.score < 55.0 and pose.error > 0.0
    assert len(pose.replicas) == 10
    assert score_pose(folder, replicas=10).score == pytest.approx(10.0)
    # reading less of each replica is reading a shorter run
    assert score_pose(folder, replicas=10, until=300.0).score < 10.0


def test_a_ligand_that_leaves_unbiased_scores_zero(tmp_path):
    folder = tmp_path / "escapee"
    folder.mkdir()
    time = np.arange(0, 1000, 10.0)
    out = np.full(time.shape, 0.2)
    out[40:] = 0.9
    _colvar(folder / "filter.dat", time, out, np.zeros(len(time)))
    pose = score_pose(folder)
    assert pose.filtered and pose.score == 0.0 and not pose.replicas
    assert "left the site" in str(pose)
    with pytest.raises(ValueError, match="no COLVAR file"):
        score_pose(tmp_path / "nothing" if False else tmp_path)


def test_an_empty_colvar_is_not_a_replica_that_scored_zero(tmp_path):
    folder = tmp_path / "part"
    folder.mkdir()
    time = np.arange(0, 600, 10.0)
    inside = np.full(time.shape, 0.2)
    _colvar(folder / "colvar_1.dat", time, inside, np.linspace(0, 40.0, len(time)))
    (folder / "colvar_2.dat").write_text("", encoding="utf-8")  # a replica that never ran
    pose = score_pose(folder)
    assert len(pose.replicas) == 1 and pose.score == pytest.approx(40.0)
    assert read_colvar(folder / "colvar_2.dat")[0].size == 0


# --- ranking and enrichment -------------------------------------------------------------


def test_poses_are_ranked_by_work_then_by_how_long_they_lasted():
    from boonza.ctmd import Pose

    poses = [Pose("a", 10.0, exit_time=100.0), Pose("b", 30.0, exit_time=50.0),
             Pose("c", 10.0, exit_time=900.0), Pose("d", 0.0, filtered=True)]  # fmt: skip
    assert [p.name for p in rank(poses)] == ["b", "c", "a", "d"]


def test_enrichment_factor_counts_the_hits_in_the_top_share():
    hits = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    best = np.arange(10, 0, -1.0)  # both hits first
    # the top 30% is 3 ligands and holds both hits: 2/3 against 2/10 by chance
    assert enrichment_factor(hits, best, 0.3) == pytest.approx((2 / 3) / 0.2)
    worst = np.arange(10, dtype=float)
    assert enrichment_factor(hits, worst, 0.3) == 0.0
    assert enrichment_factor(hits, best, 1.0) == pytest.approx(1.0)  # everything is chance
    with pytest.raises(ValueError, match="no hits"):
        enrichment_factor(np.zeros(5), np.arange(5.0), 0.3)
    with pytest.raises(ValueError, match="same non-empty shape"):
        enrichment_factor(hits, best[:5], 0.3)


def test_bedroc_is_one_for_a_perfect_ranking_and_zero_for_its_reverse():
    hits = np.array([1] * 4 + [0] * 36)
    best = np.arange(40, 0, -1.0)
    assert bedroc(hits, best, 20.0) == pytest.approx(1.0, abs=0.02)
    assert bedroc(hits, -best, 20.0) == pytest.approx(0.0, abs=0.02)
    middling = np.zeros(40)
    middling[:4] = [10, 9, -9, -10]  # two early, two late
    assert 0.2 < bedroc(hits, middling, 20.0) < 0.8
    with pytest.raises(ValueError, match="both hits and non-hits"):
        bedroc(np.ones(5), np.arange(5.0))


# --- the whole thing --------------------------------------------------------------------


def test_ctmd_runs_the_filter_and_the_replicas(tmp_path):
    # a tripeptide in water, already parameterized: residue 3 is pushed away from the
    # other two, which exercises everything without a force field to build
    g = DATA / "gmx_amber"
    s = boonza.load(g / "topol.top", coordinates=g / "conf.gro")
    cfg = Settings(
        replicas=2, nanoseconds=0.001, filter_nanoseconds=0.0005, stride=0.1, pace=0.1,
        bins=40, cutoff=5.0, commit=50.0, seed=7, pressure=None,
        cv={"ligand": "resid 3 and noh", "pocket": "protein and noh and resid 1 2",
            "cutoff": 12.0},
    )  # fmt: skip
    said = []
    pose = boonza.ctmd.run(s, tmp_path / "lig", settings=cfg, log=said.append)
    assert not pose.filtered and len(pose.replicas) == 2
    assert np.isfinite(pose.score)
    assert any("pocket-ligand pairs" in line for line in said)
    for k in (1, 2):
        time, drmsd, bias, work = read_colvar(tmp_path / "lig" / f"colvar_{k}.dat")
        assert len(time) and time[-1] == pytest.approx(1.0)  # 0.001 ns of it
        assert (drmsd >= 0).all()
        assert work[0] >= 0.0 and work[-1] >= work[0]  # the work only accumulates
        assert (np.diff(time) > 0).all()
        assert (tmp_path / "lig" / f"bias_{k}.npy").exists()
        assert np.load(tmp_path / "lig" / f"bias_{k}.npy").shape == (40,)
    # the work the bias did is positive once hills have gone down
    assert pose.score > 0.0
    # running it again leaves the replicas it already has alone
    again = boonza.ctmd.run(s, tmp_path / "lig", settings=cfg, log=said.append)
    assert again.score == pytest.approx(pose.score)
    assert any("is there already" in line for line in said)


def test_the_command_scores_folders_and_reports_enrichment(tmp_path, capsys):
    from boonza.cli import main

    time = np.arange(0, 600, 10.0)
    inside = np.full(time.shape, 0.2)
    for name, top in (("good", 60.0), ("fair", 30.0), ("poor", 5.0)):
        folder = tmp_path / name
        folder.mkdir()
        for k in (1, 2, 3):
            _colvar(folder / f"colvar_{k}.dat", time, inside, np.linspace(0, top, len(time)))
    (tmp_path / "hits.csv").write_text("name,hit\ngood,1\nfair,0\npoor,0\n", encoding="utf-8")
    out = tmp_path / "ranking.csv"
    code = main(["ctmd", "score", str(tmp_path / "good"), str(tmp_path / "poor"),
                 str(tmp_path / "fair"), "--hits", str(tmp_path / "hits.csv"),
                 "--fraction", 0.34.__str__(), "-o", str(out)])  # fmt: skip
    assert code == 0
    printed = capsys.readouterr().out
    assert printed.index("good") < printed.index("fair") < printed.index("poor")
    assert "EF34 = 2.94" in printed or "EF34 = 3.00" in printed
    assert "BEDROC" in printed
    rows = out.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("pose,score") and rows[1].startswith("good,60")
    # a ligand the hit list says nothing about is an error, not a silent 0
    assert main(["ctmd", "score", str(tmp_path / "good"), "--hits", str(tmp_path / "hits.csv"),
                 "--replicas", "3"]) == 0  # fmt: skip
    (tmp_path / "half.csv").write_text("good,1\n", encoding="utf-8")
    assert main(["ctmd", "score", str(tmp_path / "good"), str(tmp_path / "fair"),
                 "--hits", str(tmp_path / "half.csv")]) == 1  # fmt: skip
    assert "says nothing about fair" in capsys.readouterr().err


def test_a_hit_list_can_be_the_authors_own_file(tmp_path):
    from boonza.ctmd.cli import _read_hits

    # their hit_data_mol2.spc: name.mol2, whether it binds, and the ligand's charge
    theirs = tmp_path / "hit_data_mol2.spc"
    theirs.write_text(
        "ZINC000547933290.mol2 True -1\nZINC000130612584.mol2 False 0\n", encoding="utf-8"
    )
    assert _read_hits(theirs) == {"ZINC000547933290": 1, "ZINC000130612584": 0}
    # and the plainer forms, with a header and comments
    mine = tmp_path / "hits.csv"
    mine.write_text("name,hit\n# a note\nlig1,1\nlig2,0\nlig3,yes\nlig4\tno\n", encoding="utf-8")
    assert _read_hits(mine) == {"lig1": 1, "lig2": 0, "lig3": 1, "lig4": 0}
    bad = tmp_path / "bad.csv"
    bad.write_text("lig1,maybe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a yes or a no"):
        _read_hits(bad)
    short = tmp_path / "short.csv"
    short.write_text("lig1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="a name and a 0 or 1"):
        _read_hits(short)


def test_a_pose_with_no_data_is_left_out_rather_than_ranked(tmp_path, capsys):
    from boonza.cli import main

    time = np.arange(0, 600, 10.0)
    inside = np.full(time.shape, 0.2)
    good = tmp_path / "good"
    good.mkdir()
    for k in (1, 2, 3):
        _colvar(good / f"colvar_{k}.dat", time, inside, np.linspace(0, 40.0, len(time)))
    empty = tmp_path / "empty"
    empty.mkdir()
    for k in (1, 2, 3):
        (empty / f"colvar_{k}.dat").write_text("", encoding="utf-8")
    assert main(["ctmd", "score", str(good), str(empty)]) == 0
    printed = capsys.readouterr().out
    assert "not scored, and left out" in printed and "empty" in printed
    assert "good" in printed
    # nothing to rank at all is an error
    assert main(["ctmd", "score", str(empty)]) == 1
    assert "none of these folders" in capsys.readouterr().err


def test_the_coordinate_moves_less_than_one_for_one_with_the_ligand():
    from boonza.ctmd import drmsd_response

    # a mean over pairs, most of them off the direction of travel: the coordinate moves
    # well under 1 nm per 10 A, which is what the published widths assume
    s = _complex()
    per_angstrom = drmsd_response(s, cutoff=10.0, periodic=False)
    assert 0.0 < per_angstrom < 0.1  # nm per A, i.e. under one-for-one
    # and close enough to linear that one factor carries
    far = drmsd_response(s, displacement=3.0, cutoff=10.0, periodic=False)
    assert far == pytest.approx(per_angstrom, rel=0.35)
