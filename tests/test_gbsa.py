"""MM/GBSA: the GB force against OpenMM's own, the bookkeeping, the command."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.cli import main
from boonza.gbsa import MODELS, TERMS, gb_force, gbsa, kappa
from boonza.omm import KCAL, NM, to_openmm

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def protein():
    """1TEN with hydrogens and Amber14 parameters, no water: OpenMM's Modeller
    beside boonza's copy of the same system."""
    pdb = app.PDBFile(str(DATA / "1TEN.pdb"))
    model = app.Modeller(pdb.topology, pdb.positions)
    model.deleteWater()
    backbone = {"N", "CA", "C"}
    model.delete([r for r in model.topology.residues()
                  if not backbone <= {a.name for a in r.atoms()}])  # fmt: skip
    ff = app.ForceField("amber14-all.xml")
    model.addHydrogens(ff)
    omm = ff.createSystem(model.topology, nonbondedMethod=app.NoCutoff, constraints=None,
                          removeCMMotion=False)  # fmt: skip
    return model, boonza.from_openmm(model.topology, omm, model.positions)


def _alone(force, positions_A):
    """kcal/mol of one force by itself on these positions (A); a copy, since a
    force that already sits in a System cannot be given to another."""
    import copy

    system = openmm.System()
    for _ in range(len(positions_A)):
        system.addParticle(1.0)
    system.addForce(copy.deepcopy(force))
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))  # fmt: skip
    ctx.setPositions(unit.Quantity(np.asarray(positions_A) * NM, unit.nanometer))
    e = ctx.getState(getEnergy=True).getPotentialEnergy()
    return e.value_in_unit(unit.kilojoule_per_mole) / KCAL


def test_obc2_matches_openmm_s_built_in_force(protein):
    """The same model two ways: boonza's CustomGBForce against OpenMM's GBSAOBCForce,
    written in C++ and given mbondi2 radii by element the way the viewer assigns
    them.  Same dielectrics, same ACE surface term, no cutoff on either."""
    from boonza.viewer.edit import GB_DEFAULT, GB_RADII

    model, s = protein
    ours = gb_force(s, model="OBC2", surface_area="ACE")
    ref = openmm.GBSAOBCForce()
    ref.setSolventDielectric(78.5)
    ref.setSoluteDielectric(1.0)
    ref.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)
    anum = np.asarray(s.atoms["anum"])
    on_n = set()
    for i, j in zip(np.asarray(s.bonds["i"]), np.asarray(s.bonds["j"]), strict=True):
        if anum[i] == 1 and anum[j] == 7:
            on_n.add(int(i))
        if anum[j] == 1 and anum[i] == 7:
            on_n.add(int(j))
    for k, (z, q) in enumerate(zip(anum.tolist(), np.asarray(s.atoms["charge"]).tolist(),
                                   strict=True)):  # fmt: skip
        radius, scale = GB_RADII.get(z, GB_DEFAULT)
        if k in on_n:
            radius = 1.3
        ref.addParticle(q, radius / 10.0, scale)
    assert _alone(ours, s.positions) == pytest.approx(_alone(ref, s.positions), rel=1e-5)


@pytest.mark.parametrize("salt", [0.0, 0.15])
def test_gbn2_matches_openmm_s_xml_force_field(protein, salt):
    """GBn2 against OpenMM's other implementation of it, the ``implicit/gbn2.xml``
    force field, whose radii are per atom type rather than read off the bonds.
    That route takes the screening as kappa, so it checks the energy the Debye
    term gives; the conversion from a concentration is Amber's, tested apart."""
    model, s = protein
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    kwargs = {"implicitSolventKappa": kappa(salt) / unit.nanometer} if salt else {}
    omm = ff.createSystem(model.topology, nonbondedMethod=app.NoCutoff, constraints=None,
                          removeCMMotion=False, **kwargs)  # fmt: skip
    ref = next(f for f in omm.getForces() if isinstance(f, openmm.CustomGBForce))
    ours = gb_force(s, model="GBn2", salt=salt, surface_area="ACE")
    assert _alone(ours, s.positions) == pytest.approx(_alone(ref, s.positions), rel=1e-5)


def test_kappa_is_amber_s():
    """0.15 M at 298.15 K in water: 0.0930 1/A, Debye-Huckel's 0.1274 times Amber's
    0.73 -- the conversion AmberPrmtopFile.createSystem makes from a salt
    concentration, in 1/nm."""
    assert kappa(0.0) == 0.0
    assert kappa(0.15) == pytest.approx(50.33355 * (0.15 / 78.5 / 298.15) ** 0.5 * 7.3)
    assert kappa(0.15) == pytest.approx(0.930, abs=0.001)
    assert kappa(0.15, temperature=310.0) < kappa(0.15)  # warmer screens less
    with pytest.raises(ValueError, match="not negative"):
        kappa(-1.0)


def test_every_model_builds(protein):
    _, s = protein
    for model in MODELS:
        assert gb_force(s, model=model).getNumParticles() == s.natoms
    with pytest.raises(ValueError, match="GB model is one of"):
        gb_force(s, model="igb8")


def _dimer(s, shift):
    """Two copies of the protein, the second moved by ``shift`` A along x: the first
    is the receptor and the second the ligand, and they share no bond."""
    other = s.clone()
    other.positions = np.asarray(other.positions) + np.array([shift, 0.0, 0.0])
    two = s.clone()
    two.append(other)
    # 1TEN's PDB carries its crystal cell, under which a copy 80 A away is imaged
    # straight back onto the first: no box, unless a test gives one
    two.cell = np.zeros((3, 3))
    return two, f"index {s.natoms} to {two.natoms - 1}"


def _touching(s, gap: float = 3.2):
    """How far along x to move a copy for its nearest atom to sit ``gap`` A from
    the original: in contact, with no two atoms overlapping."""
    xyz = np.asarray(s.positions)
    lo, hi = 0.0, float(np.ptp(xyz[:, 0])) + gap + 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        d = np.sqrt(((xyz[None] + [mid, 0.0, 0.0] - xyz[:, None]) ** 2).sum(-1)).min()
        lo, hi = (lo, mid) if d > gap else (mid, hi)
    return hi


def test_the_pieces_add_up_and_a_far_ligand_feels_only_the_screened_charge(protein):
    """Bonded terms cancel to the last digit and the terms sum to the total.  A
    copy 80 A away touches nothing and buries nothing; what is left is the two
    net charges through water, q^2 / (78.5 r), which is what GB tends to."""
    _, s = protein
    two, ligand = _dimer(s, 80.0)
    far = gbsa(two, ligand=ligand)
    assert len(far) == 1 and far.frames.tolist() == [0]
    assert far.terms["bonded"][0] == pytest.approx(0.0, abs=1e-6)
    assert far.total[0] == pytest.approx(sum(far.terms[t][0] for t in TERMS))
    assert far.terms["sa"][0] == pytest.approx(0.0, abs=1e-6)
    assert far.terms["vdw"][0] == pytest.approx(0.0, abs=1e-3)
    q = float(np.asarray(s.atoms["charge"]).sum())
    assert abs(q) > 1  # the check below is empty for a neutral protein
    screened = 332.0636 * q * q / (78.5 * 80.0)
    assert far.terms["elec"][0] + far.terms["gb"][0] == pytest.approx(screened, rel=0.05)
    assert far.areas["complex"][0] == pytest.approx(far.areas["receptor"][0]
                                                    + far.areas["ligand"][0], rel=1e-6)  # fmt: skip


def test_a_ligand_in_contact_buries_surface_and_is_scored_as_a_whole(protein):
    """Pushed into contact, the copy buries area and its van der Waals term goes
    negative; the summary carries what a run is ranked by."""
    _, s = protein
    two, ligand = _dimer(s, _touching(s))
    near = gbsa(two, ligand=ligand)
    assert near.terms["sa"][0] < -0.5  # surface buried: the nonpolar term is favourable
    assert near.terms["vdw"][0] < -1.0
    assert near.terms["bonded"][0] == pytest.approx(0.0, abs=1e-6)
    out = near.summary()
    assert out["frames"] == 1 and out["dg"] == pytest.approx(near.total[0])
    assert out["best_frame"] == 0 and out["sem"] == 0.0
    assert set(TERMS) <= set(out)


def test_a_covalent_ligand_is_refused(protein):
    _, s = protein
    first = s.residues["resid"][0]
    with pytest.raises(ValueError, match="join the ligand to the receptor"):
        gbsa(s, ligand=f"resid {int(first)}")
    with pytest.raises(ValueError, match="picks no atom"):
        gbsa(s, ligand="resname XYZ")


def test_a_wrapped_frame_is_imaged_before_it_is_scored(protein):
    """A trajectory is written wrapped: a ligand on the far side of the box is
    the same pose as one beside the protein, and scores the same."""
    _, s = protein
    two, ligand = _dimer(s, _touching(s))
    width = float(np.ptp(np.asarray(two.positions)[:, 0]))
    two.cell = np.diag([width + 60.0, 400.0, 400.0])
    here = np.asarray(two.positions)
    wrapped = here.copy()
    wrapped[s.natoms :] -= two.cell[0]  # the ligand one box over
    both = gbsa(two, np.stack([here, wrapped]), ligand=ligand)
    assert both.frames.tolist() == [0, 1]
    # the same pose, give or take the single precision the CPU platform sums in
    assert both.total[1] == pytest.approx(both.total[0], rel=1e-4, abs=1e-3)
    assert both.terms["sa"][1] == pytest.approx(both.terms["sa"][0], rel=1e-6)
    assert both.terms["sa"][0] < -0.5  # and it is a contact, not two strangers


def test_stride_and_frames_blocks(protein):
    _, s = protein
    two, ligand = _dimer(s, 80.0)
    xyz = np.repeat(np.asarray(two.positions)[None], 5, axis=0)
    out = gbsa(two, xyz, ligand=ligand, stride=2)
    assert out.frames.tolist() == [0, 2, 4]
    with pytest.raises(ValueError, match="stride"):
        gbsa(two, xyz, ligand=ligand, stride=0)


def _run(where, system, frames, outcome="target_reached"):
    """A run directory as boonza md leaves one: solvated.dms, trajectory.dcd, status.json."""
    where.mkdir()
    boonza.save(system, where / "solvated.dms")
    with boonza.open_writer(where / "trajectory.dcd", natoms=system.natoms) as w:
        for xyz in frames:
            w.write(xyz, box=system.cell)
    (where / "status.json").write_text(json.dumps({"outcome": outcome}))


def test_the_command_scores_runs_and_structures_and_skips_detached(tmp_path, protein, capsys):
    _, s = protein
    xyz = np.asarray(s.positions)
    width = float(xyz[:, 0].max() - xyz[:, 0].min())
    two, ligand = _dimer(s, width + 3.5)
    two.cell = np.diag([400.0, 400.0, 400.0])
    here = np.asarray(two.positions)
    frames = [here, here + 0.1, here - 0.1]
    _run(tmp_path / "run_a", two, frames)
    _run(tmp_path / "run_b", two, frames[:1], outcome="detached")
    boonza.save(two, tmp_path / "pose.mae")
    out = tmp_path / "gbsa"
    assert main(["gbsa", str(tmp_path / "run_a"), str(tmp_path / "run_b"),
                 str(tmp_path / "pose.mae"), "-o", str(out), "--ligandsel", ligand,
                 "--stride", "2"]) == 0  # fmt: skip
    printed = capsys.readouterr().out
    assert "run_b: detached" in printed and "3 run(s), 2 scored with GBn2" in printed
    rows = {r["run"]: r for r in csv.DictReader(open(out / "summary.csv"))}
    assert set(rows) == {"run_a", "run_b", "pose"}
    assert rows["run_b"]["note"] == "detached" and rows["run_b"]["frames"] == ""
    assert rows["run_a"]["frames"] == "2" and rows["run_a"]["outcome"] == "target_reached"
    assert rows["pose"]["frames"] == "1" and rows["pose"]["model"] == "GBn2"
    assert float(rows["run_a"]["bonded"]) == 0.0
    assert float(rows["pose"]["dg"]) == pytest.approx(float(rows["run_a"]["best"]), abs=0.5)
    per_frame = list(csv.DictReader(open(out / "frames" / "run_a.csv")))
    assert [r["frame"] for r in per_frame] == ["0", "2"]
    assert set(per_frame[0]) == {"frame", "total", *TERMS, "bonded"}
    assert not (out / "frames" / "run_b.csv").exists()
    # --traj belongs to one structure, not to a run
    assert main(["gbsa", str(tmp_path / "run_a"), "--traj", "x.dcd", "-o", str(out)]) == 1
    assert "--traj goes with one structure file" in capsys.readouterr().err


def test_the_bonded_terms_the_check_relies_on_are_really_there(protein):
    """The full potential goes into every piece: constrained stretches included, so
    the bonded energy is the molecule's own and not a number that happens to cancel."""
    _, s = protein
    _, omm, _ = to_openmm(s, nonbonded_method="NoCutoff", constraints=False)
    names = {f.getName() for f in omm.getForces()}
    assert "stretch_harm" in names and omm.getNumConstraints() == 0


# --- the viewer: Analyze | Binding Energy ------------------------------------------------


def _water_ligand(session):
    """The water molecule nearest the protein, as the page's Set as Ligand names it."""
    s = session.system
    near = s.select("same residue as (water and within 3.5 of protein)").ids
    res = int(np.asarray(s.atoms["residue"])[near[0]])
    atoms = [int(a) for a in s.residue_atoms(res)]
    session.set_ligand(atoms)
    return atoms


def test_the_viewer_scores_the_ligand_without_moving_it():
    """The force field the file carries, the ligand the page chose -- a water
    molecule, kept though it is solvent -- and nothing moved or edited."""
    from boonza import viewer

    g = DATA / "gmx_amber"
    s = boonza.load(g / "topol.top", coordinates=g / "conf.gro")
    session = viewer.Session(s, cg=False)
    atoms = _water_ligand(session)
    before = np.asarray(session.system.positions).copy()
    out = session.gbsa()
    assert out["frames"] == 1 and out["frame_indices"] == [0]
    assert out["ligand_atoms"] == len(atoms) == 3
    assert out["receptor_atoms"] == len(session.system.select("protein"))
    assert out["bonded"] == pytest.approx(0.0, abs=1e-6)
    assert out["dg"] == pytest.approx(out["vdw"] + out["elec"] + out["gb"] + out["sa"], abs=0.02)
    assert out["per_frame"] == [out["dg"]] and out["best_frame"] == 0
    assert out["how"].startswith("GBn2, no salt")
    assert np.allclose(session.system.positions, before) and session.version == 0
    assert "version" not in out  # nothing was prepared, so nothing to reload
    with_salt = session.gbsa(model="OBC2", salt=0.15)
    assert with_salt["how"].startswith("OBC2, 0.15 M salt") and with_salt["dg"] != out["dg"]
    session.ligand = "resname XYZ"
    with pytest.raises(ValueError, match="choose the ligand"):
        session.gbsa()


def test_the_viewer_scores_every_frame_of_a_trajectory(tmp_path):
    from boonza import viewer

    g = DATA / "gmx_amber"
    s = boonza.load(g / "topol.top", coordinates=g / "conf.gro")
    xyz = np.asarray(s.positions)
    with boonza.open_writer(tmp_path / "run.dcd", natoms=s.natoms) as w:
        for k in range(3):
            w.write(xyz + 0.05 * k, box=s.cell)
    session = viewer.Session(s, trajectory=boonza.open_trajectory(tmp_path / "run.dcd", s),
                             cg=False)  # fmt: skip
    _water_ligand(session)
    one = session.gbsa(frame=2)
    assert one["frames"] == 1 and one["frame_indices"] == [2] and one["best_frame"] == 2
    every = session.gbsa(all_frames=True, stride=2)
    assert every["frames"] == 2 and every["frame_indices"] == [0, 2]
    assert every["best_frame"] in (0, 2) and len(every["per_frame"]) == 2
    assert every["dg"] == pytest.approx(np.mean(every["per_frame"]), abs=0.02)


def test_the_page_asks_for_a_binding_energy():
    """POST /api/gbsa, with the page's token, on a served session."""
    import threading
    import urllib.request

    from boonza import viewer

    g = DATA / "gmx_amber"
    s = boonza.load(g / "topol.top", coordinates=g / "conf.gro")
    session = viewer.Session(s, cg=False)
    _water_ligand(session)
    server = viewer.make_server(session, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}/"
        body = json.dumps({"m": 0, "model": "GBn2", "frame": 0}).encode()
        req = urllib.request.Request(
            base + "api/gbsa",
            body,
            method="POST",
            headers={"X-Boonza-Token": server.RequestHandlerClass.token},
        )
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        assert set(TERMS) <= set(out) and out["ligand_atoms"] == 3
        assert out["dg"] == pytest.approx(session.gbsa()["dg"], abs=0.02)
    finally:
        server.shutdown()
