"""fpocket on coarse-grained proteins: bead polarity, the measures, consensus and the CLI.

The tests that run fpocket itself are skipped where the siyoungkimlab build is
not installed (``FPOCKET_HOME`` or ``PATH``).
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.pockets import beads, consensus, core, overlap, prepare, run
from boonza.pockets.cli import main, structure_format

DATA = Path(__file__).parent / "data"


def _fpocket():
    try:
        return run.find_fpocket()
    except run.PocketError:
        pytest.skip("the siyoungkimlab fpocket build is not installed")


@pytest.mark.parametrize(
    ("bead_type", "polar"),
    [("C1", False), ("SC4", False), ("TC5", False), ("X2", False), ("C6v", False),
     ("P1", True), ("SP2", True), ("TP1dq", True), ("Q5", True), ("SQ4p", True), ("Qda", True),
     ("D", True), ("N0", False), ("Nda", False), ("TN6d", False), ("AC1", False), ("AC2", False)],
)  # fmt: skip
def test_martini_polarity(bead_type, polar):
    assert beads.martini_polar(bead_type) is polar


def test_martini_n_class_is_a_choice():
    assert beads.martini_polar("TN6d", n_polar=True)
    assert not beads.martini_polar("TN6d", n_polar=False)


@pytest.mark.parametrize(
    ("name", "charge", "polar"),
    [("GN", 0.13, True), ("GO", -0.23, True), ("GC", 0.10, False), ("GN", 0.4, True),
     ("BCG", 0.0, False), ("BSD", 0.0, False), ("BCE1", 0.0, False),
     ("BCE", 0.6, True), ("BCG", 0.4, True), ("BCD", -0.3, True), ("BSG", -0.2, True),
     ("BCE2", -0.1, True), ("BNE", 0.1, True), ("BOG", -0.4, True)],
)  # fmt: skip
def test_sirah_polarity(name, charge, polar):
    """Backbone by name; a side-chain bead is polar when it is charged."""
    assert beads.sirah_polar(name, charge) is polar


def test_sirah_library_charges():
    q = beads.sirah_charges(["BCG", "BCE", "BCG", "BCZ", "BSD", "BCE2"],
                            ["sK", "sK", "sL", "sR", "sM", "sY"])  # fmt: skip
    assert q == pytest.approx([0.4, 0.6, 0.0, 0.3, 0.0, -0.1])


def test_bead_types_from_the_force_field():
    """A structure without its topology takes its types from the force field's blocks."""
    got = beads.bead_types(["BB", "SC1", "SC2", "SC1"], ["ARG", "ARG", "ARG", "XYZ"], "martini3")
    assert got == ["P2", "SC3", "SQ3p", ""]
    # Martini 2's backbone type follows the secondary structure; the block's is the coil's,
    # which is as polar under the preset as the helix's Nda
    bb = beads.bead_types(["BB", "BB"], ["LEU", "HIS"], "martini2")
    assert all(bb) and all(beads.martini_polar(t, n_polar=True) for t in bb)


@pytest.mark.parametrize(
    ("resname", "standard"),
    [("sA", "ALA"), ("sHe", "HIS"), ("sX", "CYS"), ("HSD", "HIS"), ("CYX", "CYS"), ("GLY", "GLY")],
)
def test_standard_resnames(resname, standard):
    assert beads.standard_resname(resname) == standard


def test_guess_model():
    assert beads.guess_model(["GN", "GC", "GO"], ["sA"]) == "sirah"
    assert beads.guess_model(["BB", "SC1"], ["ALA"]) == "martini"
    assert beads.guess_model(["N", "CA", "C", "O"], ["ALA"]) == "aa"


def test_pdb_columns(tmp_path):
    path = tmp_path / "x.pdb"
    beads.write_pdb(path, ["BB", "SC1"], ["LEU", "LEU"], ["A", "A"], [1001, 1001], ["", ""],
                    [(1.0, -2.0, 3.0), (10.5, 20.25, -30.125)], ["O", "C"])  # fmt: skip
    lines = [x for x in path.read_text().splitlines() if x.startswith("ATOM")]
    assert lines[0][12:16] == " BB "
    assert lines[1][17:20] == "LEU" and lines[1][21] == "A" and int(lines[1][22:26]) == 1001
    assert float(lines[1][30:38]) == 10.5 and float(lines[1][46:54]) == -30.125
    assert lines[0][76:78].strip() == "O" and lines[1][76:78].strip() == "C"


def test_flags_after_the_preset_override_it():
    base = ["-m", "4.9", "-M", "7.5", "-i", "9"]
    assert run.merge_flags(base, ["-i", "20"]) == ["-m", "4.9", "-M", "7.5", "-i", "20"]
    assert run.merge_flags(base, []) == base


def test_presets_carry_flags_and_coefficients():
    """Every bead model's preset passes its refitted score to fpocket."""
    for model in ("martini2", "martini3", "sirah"):
        p = run.preset(model)
        coeffs = next(f for f in p["flags"] if f.startswith("--score_coefficients="))
        assert len(coeffs.split("=")[1].split(",")) == len(p["score_coefficients"])
        assert p["mdpocket_density_iso"] > 0
    assert run.preset("aa")["flags"] == []


def test_unknown_residues_from_mapper_messages():
    msg = "not martini3001 protein residues: DAS, MSE; leave them out of atoms='protein'"
    assert prepare._unknown_residues(msg) == ["DAS", "MSE"]
    assert prepare._unknown_residues("SIRAH's map has no LIG; leave them out") == ["LIG"]


def test_probes_in_chain_lig_are_not_protein():
    """Chain LIG holds a swim's probes (ligand copies); they never enter the search."""
    s = boonza.System("probe")
    for chain in ("A", "LIG"):
        r = s.add_residue(s.add_chain(name=chain), name="ALA", resid=1)
        for k, name in enumerate(("N", "CA", "C", "O", "CB")):
            r.add_atom(name=name, anum={"N": 7, "O": 8}.get(name, 6), pos=(k, 0.0, 0.0))
    assert prepare.probe_ids(s).tolist() == [5, 6, 7, 8, 9]
    assert prepare.protein_ids(s, "all").tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    ("path", "fmt"),
    [("apo.mae", "mae"), ("apo.maegz", "mae"), ("apo.mae.gz", "mae"), ("apo.dms", "mae"),
     ("apo.cif", "cif"), ("apo.pdb", "pdb"), ("apo.gro", "gro"), ("apo.xyz", "pdb")],
)  # fmt: skip
def test_structures_keep_their_format(path, fmt):
    """The view writes each structure in the format it was given; DMS as MAE."""
    assert structure_format(path) == fmt


def test_ppc_and_moc():
    lig = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    on = lig + [0.0, 1.0, 0.0]
    assert overlap.ppc(on, lig) and overlap.moc(on, lig)
    assert overlap.ligand_atoms_within(on, lig) == 1.0 and overlap.spheres_within(on, lig) == 1.0
    beside = lig + [0.0, 4.5, 0.0]  # centre 4.5 A from the nearest atom
    assert not overlap.ppc(beside, lig) and not overlap.moc(beside, lig)
    # a pocket that reaches the ligand with one sphere of twenty passes neither share
    far = np.vstack([lig[:1] + [0.0, 1.0, 0.0], np.full((19, 3), 30.0)])
    assert overlap.spheres_within(far, lig) == pytest.approx(0.05) and not overlap.moc(far, lig)


def test_volume_overlap_of_a_pocket_on_its_ligand():
    lig = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    same = overlap.ligand_voxels(lig)
    got = overlap.volume_overlap(same, lig)
    assert got["ligand_volume_covered"] == 1.0 and got["DVO"] == 1.0
    apart = overlap.volume_overlap(overlap.ligand_voxels(lig + 20.0), lig)
    assert apart["ligand_volume_covered"] == 0.0 and apart["pocket_volume_near_ligand"] == 0.0
    cube = overlap.grid_pocket(np.zeros((1, 3)), 2.0)
    assert len(cube) * overlap.SPACING**3 == pytest.approx(8.0 * (5 / 4) ** 3)


def _shell(radius=7.0, n=400, seed=0):
    """Beads on a sphere with a gap at +z: a closed cavity with one mouth."""
    r = np.random.default_rng(seed).normal(size=(n, 3))
    r = radius * r / np.linalg.norm(r, axis=1)[:, None]
    return r[r[:, 2] < radius * 0.8]


def test_enclosed_core_fills_a_cavity_and_not_the_outside():
    shell = _shell()
    radii = np.full(len(shell), 2.35)
    # one big alpha sphere filling the cavity and one outside the protein
    inside = core.enclosed_core([[0.0, 0.0, 0.0]], [5.0], shell, radii)
    outside = core.enclosed_core([[0.0, 0.0, 20.0]], [5.0], shell, radii)
    assert len(inside) > 5 and len(outside) == 0
    assert np.linalg.norm(inside, axis=1).max() < 7.0 - core.OUTSIDE * 2.35 + core.SPACING


def test_bead_radii_from_the_tables():
    got = core.bead_radii(["P2", "SC3", "TC5", "nonsense"], "martini3")
    assert got[:3] == pytest.approx([2.35, 2.05, 1.70], abs=0.01) and np.isnan(got[3])


def _frame_pocket(frame, rank, centre, score, burial=0.5):
    p = run.Pocket(np.asarray(centre, float)[None] + np.zeros((3, 3)), np.full(3, 4.0), score)
    return consensus.FramePocket(frame, rank, p, burial)


def test_consensus_merges_by_place_and_ranks_three_ways():
    """A pocket in every frame at one place, a rare strong one, and a rare weak one."""
    frames = []
    for f in range(40):
        here = [_frame_pocket(f, 1, [0.0, 0.0, 0.1 * (f % 3)], 0.0, burial=0.4)]
        if f < 4:  # open in 10% of frames, scored high and buried
            here.append(_frame_pocket(f, 2, [20.0, 0.0, 0.0], 3.0, burial=0.9))
        if f == 0:  # open once: below the occupancy floor
            here.append(_frame_pocket(f, 3, [-20.0, 0.0, 0.0], 5.0))
        frames.append(here)
    ranked = consensus.consensus_pockets(frames, crystal=[[0.0, 0.0, 0.0]])
    assert len(ranked) == 3
    common, rare, once = (min(ranked, key=lambda q: np.linalg.norm(q.centre - c))
                          for c in ([0, 0, 0], [20, 0, 0], [-20, 0, 0]))  # fmt: skip
    assert common.occupancy == 1.0 and len(common.open) == 40
    assert common.rank_persistence == 1  # always there
    assert rare.rank_quality == 1 and rare.rank_quality_burial == 1  # opens rarely, opens well
    assert once.rank_quality == 3  # below MIN_OCCUPANCY, however well scored
    assert not common.cryptic and rare.cryptic


def test_burial_of_points_is_the_occupancy_burial():
    """sites.burial at a grid's cell centres is Occupancy.burial, on a real protein in beads."""
    from boonza.sites import Occupancy, burial

    cg = prepare.coarse_grain(boonza.load(str(DATA / "1TEN.pdb")), "martini3")
    protein = np.asarray(cg.positions, float)
    grid = Occupancy(protein, spacing=1.0, margin=6.0)
    cells = np.arange(0, grid.counts.size, 37)
    assert np.array_equal(grid.burial(cells, protein), burial(grid.cell_centres()[cells], protein))
    centre = protein.mean(0)
    assert burial([centre], protein)[0] > burial([centre + [40.0, 0.0, 0.0]], protein)[0]


def test_find_fpocket_names_the_build(monkeypatch, tmp_path):
    monkeypatch.setenv("FPOCKET_HOME", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(run.PocketError, match="siyoungkimlab"):
        run.find_fpocket()


def test_flags_command(capsys):
    assert main(["flags", "--model", "sirah", "--", "-i", "20"]) == 0
    printed = capsys.readouterr().out.split()
    assert printed[printed.index("-i") + 1] == "20" and "--score_coefficients" in " ".join(printed)


def test_run_on_a_mapped_structure(tmp_path):
    """An all-atom structure, mapped onto Martini 3 beads and searched with the preset."""
    _fpocket()
    assert main(["run", str(DATA / "2TRX.pdb"), "--model", "martini3", "-o", str(tmp_path)]) == 0
    rows = list(csv.DictReader(open(tmp_path / "pockets.csv")))
    doc = json.loads((tmp_path / "pockets.json").read_text())
    assert rows and len(rows) == len(doc["pockets"])
    assert doc["settings"]["model"] == "martini3" and doc["settings"]["fpocket_flags"]
    assert {"rank", "fpocket_score", "fpocket_volume", "center_x"} <= set(rows[0])
    view = (tmp_path / "view.pml").read_text()
    assert "pocket_1" in view and "__script__" in view
    assert all(";" not in line for line in view.splitlines() if line.startswith("#"))


@pytest.mark.parametrize("fit", ["whole", "chain"])
def test_holo_ligand_follows_the_protein(tmp_path, fit):
    """--holo-fit whole superposes every chain of the holo protein, chain only the one the
    ligand sits in; either way the ligand moves with the fit."""
    from boonza.pockets.cli import _holo_on

    ref = boonza.load(str(DATA / "1HHO.pdb"))
    holo = ref.clone()
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    holo.positions = np.asarray(holo.positions) @ turn.T + [30.0, -5.0, 12.0]
    boonza.save(holo, tmp_path / "holo.pdb")
    lig, info, moved = _holo_on(tmp_path / "holo.pdb", ref, "resname HEM and chain B", fit=fit)
    want = np.asarray(ref.positions)[ref.select("resname HEM and chain B and not element H").ids]
    assert info["fit"] == fit and np.abs(lig - want).max() < 0.01
    assert info["paired"] > (250 if fit == "whole" else 120)
    moved_lig = moved.select("resname HEM and chain B and not element H").ids
    assert np.abs(np.asarray(moved.positions)[moved_lig] - want).max() < 0.01


@pytest.mark.parametrize(
    ("model", "found"), [("sirah", "sirah"), ("martini3", "martini"), ("aa", "aa")]
)
def test_coarse_grained_input_is_recognized(model, found):
    """A SIRAH system has no BB bead: its residue names (sA, sK, ...) say what it is."""
    from boonza.pockets.cli import _model_of

    s = prepare.coarse_grain(boonza.load(str(DATA / "1TEN.pdb")), model)
    assert _model_of(s, np.arange(s.natoms)) == found


def test_traj_checks_fpocket_before_reading_frames(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FPOCKET_HOME", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path))
    code = main(["traj", "--workdir", str(tmp_path / "nowhere"), "--model", "sirah",
                 "-o", str(tmp_path / "out")])  # fmt: skip
    assert code == 1 and "siyoungkimlab" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
