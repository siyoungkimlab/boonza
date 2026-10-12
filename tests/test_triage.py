"""Triaging many runs of one system: the fingerprint, the grouping, the commands."""

import json
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.interactions import interaction_fingerprints
from boonza.triage import cosine_distance, modes, pharmacophore_fingerprint, triage

DATA = Path(__file__).parent / "data"
LIGAND = "resname HEM and noh"
PROTEIN = "protein and noh"


def _one_chain():
    return boonza.load(str(DATA / "1HHO.pdb")).select("chain A").clone()


def _turned(system, ligand, degrees, axis):
    """The ligand turned about its own centre, the protein left where it is."""
    xyz = np.asarray(system.positions).copy()
    ids = np.asarray(system.select(ligand).ids, np.int64)
    centre = xyz[ids].mean(0)
    th = np.radians(degrees)
    r = np.eye(3)
    a, b = (axis + 1) % 3, (axis + 2) % 3
    r[a, a] = r[b, b] = np.cos(th)
    r[a, b], r[b, a] = -np.sin(th), np.sin(th)
    xyz[ids] = (xyz[ids] - centre) @ r.T + centre
    return xyz


def test_modes_groups_by_direction_and_reports_a_real_member():
    """Average linkage on cosine distance, each mode named by its medoid."""
    v = np.array([[1, 0, 0], [0.99, 0.1, 0], [0, 1, 0], [0, 0.98, 0.1], [0, 0, 1.0]])
    found = modes(v, cutoff=0.25)
    assert len(found) == 3
    assert found.members == [[0, 1], [2, 3], [4]]  # biggest first
    assert all(m in members for m, members in zip(found.medoids, found.members, strict=True))
    assert found.spread[-1] == 0.0  # a mode of one has no width
    # one mode when everything is the same direction, however long the vectors
    assert len(modes(np.array([[1.0, 0], [5.0, 0], [0.2, 0]]), cutoff=0.01)) == 1


def test_a_mode_that_merges_two_says_so_in_its_spread():
    """The cutoff is a guess; the widest pair inside a mode is how it is checked."""
    v = np.array([[1, 0, 0], [0, 1, 0]], float)
    loose = modes(v, cutoff=2.0)
    assert len(loose) == 1
    assert loose.spread[0] == pytest.approx(1.0, abs=1e-6)  # orthogonal: nothing alike
    assert len(modes(v, cutoff=0.25)) == 2


def test_the_fingerprint_sees_a_flip_the_plain_one_misses():
    """A ligand turned over touches the same residues, so a contact fingerprint
    barely moves; typing it by feature family is what notices, because the flip
    puts a donor where an acceptor was."""
    pytest.importorskip("rdkit")
    s = _one_chain()
    both = np.stack([np.asarray(s.positions), _turned(s, LIGAND, 180, axis=1)])
    plain = np.asarray(interaction_fingerprints(s, positions=both, ligand=LIGAND,
                                                protein=PROTEIN).values)  # fmt: skip
    typed = pharmacophore_fingerprint(s, both, ligand=LIGAND, protein=PROTEIN).values
    assert typed.shape[1] == 6 * plain.shape[1]  # one block per feature family
    apart_plain = cosine_distance(*plain)
    apart_typed = cosine_distance(*typed)
    assert apart_plain < 0.15  # the residues touched hardly changed
    assert apart_typed > 3 * apart_plain


def test_a_ligand_with_no_atoms_of_a_family_gets_a_column_of_zeros_not_a_gap():
    """The columns have to line up across runs, so a missing family is still a block."""
    pytest.importorskip("rdkit")
    s = _one_chain()
    fp = pharmacophore_fingerprint(s, ligand=LIGAND, protein=PROTEIN)
    width = fp.values.shape[1] // 6
    blocks = fp.values[0].reshape(6, width)
    assert fp.values.shape[1] % 6 == 0
    assert (blocks.sum(1) > 0).any()  # at least one family is there
    assert len(fp.residues) == width


def _trajectory(system, ligand, frames, drift):
    """A run whose ligand slides `drift` A over the first half and then holds."""
    xyz = np.asarray(system.positions)
    ids = np.asarray(system.select(ligand).ids, np.int64)
    out = []
    for k in range(frames):
        step = xyz.copy()
        how_far = drift * min(1.0, 2.0 * k / max(frames - 1, 1))
        step[ids] = step[ids] + np.array([how_far, 0.0, 0.0])
        out.append(step)
    return np.array(out)


def test_triage_tells_a_run_that_moved_and_settled_from_one_that_never_moved():
    """The class worth finding: a pose that leaves a bad start and holds a good one.

    Everything measured against the first frame scores it badly, because it was
    right to move -- so the second half is measured on its own as well.
    """
    pytest.importorskip("rdkit")
    s = _one_chain()
    still = triage(s, _trajectory(s, LIGAND, 8, 0.0), ligand=LIGAND, protein=PROTEIN,
                   pocket="protein and name CA", fingerprint=False)  # fmt: skip
    moved = triage(s, _trajectory(s, LIGAND, 8, 6.0), ligand=LIGAND, protein=PROTEIN,
                   pocket="protein and name CA", fingerprint=False)  # fmt: skip
    assert still.frames == moved.frames == 8
    assert still.drifted < 0.1 and still.population == pytest.approx(1.0)
    assert not still.settled  # it never left: nothing to settle into
    assert moved.drifted > still.drifted  # it is far from where it began
    assert moved.late_population > 0.9  # and held one pose once it got there
    assert moved.settled


def test_triage_keeps_a_run_whose_ligand_left_the_pocket():
    """dRMSD against a frame with no pocket around the ligand is undefined, and
    losing the row over it throws away exactly the runs worth looking at."""
    pytest.importorskip("rdkit")
    s = _one_chain()
    gone = triage(s, _trajectory(s, LIGAND, 6, 60.0), ligand=LIGAND, protein=PROTEIN,
                  pocket="protein and name CA", fingerprint=False)  # fmt: skip
    assert gone.frames == 6
    assert np.isnan(gone.from_end) or gone.from_end >= 0  # whichever, the row survives
    assert gone.n_poses >= 0


def _run_dir(where: Path, system, positions, name: str) -> Path:
    from boonza.trajectory import open_writer

    run = where / name
    run.mkdir(parents=True, exist_ok=True)
    boonza.save(system, run / "solvated.dms")
    with open_writer(run / "trajectory.dcd", system.natoms) as w:
        for frame in positions:
            w.write(frame, system.cell)
    (run / "status.json").write_text(json.dumps({"outcome": "target_reached"}))
    return run


def test_both_commands_write_what_they_promise(tmp_path):
    """boonza triage then boonza modes: a summary, fingerprints, medoids, sessions."""
    pytest.importorskip("rdkit")
    from boonza.cli import main

    s = _one_chain()
    for k, drift in enumerate((0.0, 0.3, 7.0)):
        _run_dir(tmp_path / "md", s, _trajectory(s, LIGAND, 6, drift), f"run{k}")
    out = tmp_path / "triage"
    assert main(["triage", *[str(p) for p in sorted((tmp_path / "md").iterdir())],
                 "-o", str(out), "--ligandsel", LIGAND, "--proteinsel", PROTEIN,
                 "--pocketsel", "protein and name CA"]) == 0  # fmt: skip
    rows = (out / "summary.csv").read_text().splitlines()
    assert len(rows) == 4 and rows[0].startswith("run,")
    loaded = np.load(out / "fingerprints.npz")
    assert len(loaded.files) == 3
    assert len({loaded[f].shape for f in loaded.files}) == 1  # one width: one system
    assert (out / "structures" / "run0_pose.mae").exists()  # mae: bond orders kept
    assert json.loads((out / "triage.json").read_text())["ligandsel"] == LIGAND

    review = tmp_path / "review"
    # no --ligandsel: the session shows the ligand triage was given
    assert main(["modes", str(out), "-o", str(review)]) == 0
    assert (review / "modes.csv").exists()
    assert (review / "session.pml").exists() and (review / "session.vmd").exists()
    assert "save session.pse" in (review / "session.pml").read_text()
    assert len(list((review / "structures").glob("mode*_pose.mae"))) >= 1


def _moved(system):
    """The same system turned and shifted, as another run's box leaves it."""
    s = system.clone()
    th = np.radians(40.0)
    r = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    s.positions = np.asarray(s.positions) @ r.T + np.array([30.0, -12.0, 5.0])
    return s


def test_modes_writes_its_medoids_in_one_frame(tmp_path):
    """Every medoid comes out fitted onto the largest mode's, or onto --reference."""
    from boonza.cli import _align_medoids

    s = _one_chain()
    (tmp_path / "structures").mkdir()
    boonza.save(s, tmp_path / "structures" / "mode00_n2_a_pose.mae")
    boonza.save(_moved(s), tmp_path / "structures" / "mode01_n1_b_pose.mae")
    poses = _align_medoids(tmp_path, [(0, "a", 2), (1, "b", 1)])
    assert [p.name for p, _ in poses] == ["mode00_n2_a_pose.mae", "mode01_n1_b_pose.mae"]
    ca = s.select("name CA").ids
    back = boonza.load(str(tmp_path / "structures" / "mode01_n1_b_pose.mae"))
    assert np.allclose(np.asarray(back.positions)[ca], np.asarray(s.positions)[ca], atol=1e-3)

    # a reference from elsewhere moves the largest mode too
    _align_medoids(tmp_path, [(0, "a", 2), (1, "b", 1)], reference=_moved(s))
    back = boonza.load(str(tmp_path / "structures" / "mode00_n2_a_pose.mae"))
    assert np.allclose(np.asarray(back.positions)[ca], np.asarray(_moved(s).positions)[ca],
                       atol=1e-3)  # fmt: skip


def test_the_sessions_show_the_same_atoms_in_pymol_and_vmd(tmp_path):
    """boonza picks the ligand and its pocket and both viewers get them by index,
    so a selection only one viewer could parse can no longer break the other."""
    from boonza.cli import _index_ranges, _write_sessions

    s = _one_chain()
    pose = tmp_path / "structures" / "mode00_n1_a_pose.mae"
    pose.parent.mkdir()
    boonza.save(s, pose)
    _write_sessions(tmp_path, [(pose, s)], "resname HEM")
    carbons = s.select("resname HEM and carbon").ids
    assert len(carbons) > 0
    vmd = (tmp_path / "session.vmd").read_text()
    pml = (tmp_path / "session.pml").read_text()
    assert f"mol selection {{index {_index_ranges(carbons, 0, ' to ', ' ')}}}" in vmd
    assert f"index {_index_ranges(carbons, 1, '-', '+')})" in pml
    assert "resname HEM" not in vmd + pml  # nothing left for a viewer to parse
    assert _index_ranges([3, 4, 5, 9], 1, "-", "+") == "4-6+10"
    assert _index_ranges([3, 4, 5, 9], 0, " to ", " ") == "3 to 5 9"


def test_the_pocket_shows_whole_residues_with_polar_hydrogens_only(tmp_path):
    """Whole residues near the ligand, and of their hydrogens only those on N, O or S."""
    from boonza.cli import _write_sessions

    s = boonza.peptide("AAFAA")
    assert len(s.select("hydrogen").ids), "the test needs explicit hydrogens"
    pose = tmp_path / "structures" / "mode00_n1_a_pose.mae"
    pose.parent.mkdir()
    boonza.save(s, pose)
    _write_sessions(tmp_path, [(pose, s)], "resid 3")
    line = (tmp_path / "session.vmd").read_text().split("mol selection {index ")[1]
    shown = set()
    for part in line.split("}")[0].replace(" to ", ":").split():
        a, _, b = part.partition(":")
        shown.update(range(int(a), int(b or a) + 1))
    near = set(s.select("(same residue as (not resid 3 and within 5 of resid 3)) and noh").ids)
    assert near and near <= shown  # every heavy atom of the pocket residues
    on_carbon = set(s.select("hydrogen and withinbonds 1 of carbon").ids)
    assert not shown & on_carbon  # and no hydrogen on a carbon
    assert shown & set(s.select("hydrogen and withinbonds 1 of nitrogen").ids)
