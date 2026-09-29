"""Dipeptide probes and the Martini side of ``boonza swim``."""

from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.martini.probes import LETTERS, PROBE_RESIDUES, probe, probe_charge, probe_sequences
from boonza.md.cgswim import build, groups_of, place, prepare
from boonza.md.config import parse_arguments

DATA = Path(__file__).parent / "data"


def test_the_probe_library():
    """Every pair of the 14 residues, once; the small and the doubled-up ones
    (Ala, Gly, Val, Asp, Asn) and the reactive ones (Cys) are left out."""
    seqs = probe_sequences()
    assert len(seqs) == 105 == 14 * 15 // 2
    assert len(set(seqs)) == len(seqs)
    assert not {"A", "G", "V", "D", "N", "C", "U"} & set("".join(seqs))
    assert set("".join(seqs)) == {LETTERS[r] for r in PROBE_RESIDUES}
    assert "RE" in seqs and "ER" not in seqs  # XY and YX are one probe


@pytest.mark.parametrize(("sequence", "charge"), [
    ("EK", 0.0), ("RR", 2.0), ("EE", -2.0), ("HH", 0.0), ("WY", 0.0), ("RE", 0.0),
])  # fmt: skip
def test_probes_carry_the_charges_of_ph_7(sequence, charge):
    """Arg and Lys +1, Glu -1, His neutral: the ends are capped, so only the
    side chains count (boonza's own peptides are built neutral, which is why
    the probe is mapped from the heavy atoms)."""
    assert probe_charge(sequence) == charge
    ends = [n for n in probe(sequence).molecules[0].nodes if n["atomname"] == "BB"]
    assert [(n["atype"], n["charge"]) for n in ends] == [("P6", 0.0)] * 2


def test_probes_are_free_to_bend():
    """No secondary structure, no side-chain corrections, no elastic network:
    a probe is a small molecule, not a piece of folded protein."""
    m = probe("WY")
    mol = m.molecules[0]
    assert not any("cgsecstruct" in n for n in mol.nodes)
    assert not [t for t in mol.interactions["dihedrals"]
                if t.meta.get("group", "").startswith("SC-BB-BB-SC")]  # fmt: skip
    assert not [t for t in mol.interactions["bonds"] if t.meta.get("group") == "Rubber band"]
    assert {n["resname"] for n in mol.nodes} == {"WY"}  # its own name, not TRP/TYR
    assert m.names == ["WY"]


def test_probes_are_built_once():
    assert probe("EK") is not probe("EK")  # a copy the caller may move
    assert probe("EK").positions == pytest.approx(probe("EK").positions)


def test_probes_are_dealt_round_robin():
    groups = groups_of(probe_sequences(), 10)
    assert len(groups) == 10  # 105 probes, about 10 a simulation
    assert sorted(s for g in groups for s in g) == sorted(probe_sequences())
    assert {len(g) for g in groups} <= {10, 11}
    # every simulation holds a spread of side chains, not RR RQ RE ... together
    assert min(len(set("".join(g))) for g in groups) >= 8


def test_probes_are_placed_clear_of_everything():
    from boonza.spatial import min_dist2

    box = np.full(3, 60.0)
    protein = np.array([[30.0, 30.0, 30.0]])
    probes = [probe("EK"), probe("LL")]
    rng = np.random.default_rng(0)
    placed = place(protein, probes, 3, box, rng, clearance=5.0)
    assert [len(p) for p in placed] == [3 * probes[0].nbeads, 3 * probes[1].nbeads]
    molecules = [chunk for p, m in zip(placed, probes, strict=True)
                 for chunk in np.split(p, 3)]  # every copy of every probe  # fmt: skip
    for k, mine in enumerate(molecules):
        others = np.vstack([protein, *molecules[:k], *molecules[k + 1 :]])
        assert (min_dist2(mine, others, 5.0, cell=np.diag(box)) >= 25.0 - 1e-6).all()


def test_a_prepared_simulation(tmp_path):
    """One directory per group: the built topology, settings `boonza md` accepts,
    and the probes written down for the analysis."""
    args = parse_arguments([str(DATA / "1TEN.pdb"), "--model", "martini3",
                            "--workdir", str(tmp_path / "swim"), "--production-ns", "10",
                            "--seed", "1"])  # fmt: skip
    sims = prepare(args, ["EK", "LL", "RR"], types=2, copies=2, log=lambda *_: None)
    assert len(sims) == 2  # 3 probes, about 2 a simulation
    d = sims[0]
    assert (d / "probes.json").is_file() and (d / "martini" / "topol.top").is_file()
    import json

    written = json.loads((d / "probes.json").read_text())
    assert written["copies"] == 2 and written["align"] == "name BB"
    top = (d / "martini" / "topol.top").read_text()
    for name in written["probes"]:
        assert f"{name} 2" in top  # two copies of each probe
        assert f'#include "{name}.itp"' in top
    settings = parse_arguments(["--config", str(d / "md.toml")])
    assert settings.model == "martini3" and settings.solvate == "none"
    assert settings.repulsion_selection == "resname " + " ".join(written["probes"])
    assert settings.production_ns == 10.0
    assert not settings.forcefields  # nothing all-atom came along
    assert Path(settings.input_structure) == (d / "martini" / "topol.top").resolve()
    assert (tmp_path / "swim" / "assignment.csv").is_file()
    assert (tmp_path / "swim" / "simulations.txt").read_text().count("boonza md") == 2


def test_the_built_system_is_neutral_and_runs(tmp_path):
    """The protein, its probes and water in one box, with the ions the charges ask for."""
    from boonza.martini import OPENMM_OPTIONS, martinize

    pytest.importorskip("openmm")
    protein = martinize(boonza.load(DATA / "1TEN.pdb"), elastic=True)
    probes = [probe("RR"), probe("EE")]
    box = np.full(3, float(np.ptp(protein.positions, axis=0).max()) + 20.0)
    m = build(protein, probes, 2, box, np.random.default_rng(0), salt=0.15)
    counts = dict(zip(m.names, m.molecule_copies, strict=True))
    assert counts["RR"] == counts["EE"] == 2
    charge = sum(c * sum(float(n["charge"]) for n in mol.nodes)
                 for mol, c in zip(m.molecules, m.molecule_copies, strict=True))  # fmt: skip
    (_, _), (_, na), (_, cl) = m.solvent
    assert charge + na - cl == pytest.approx(0)
    s = m.system()
    assert s.natoms == m.nbeads
    assert {"RR", "EE"} <= {str(n) for n in s.residues["name"]}
    assert np.isfinite(boonza.openmm_energies(s, **OPENMM_OPTIONS)["total"])


def test_the_analysis_knows_a_coarse_grained_run(tmp_path):
    """`boonza sites` aligns on BB beads and takes the probes from the run."""
    from boonza.cli import DEFAULT_ALIGN, _cg_selections, _coarse_grained
    from boonza.symmetry import DEFAULT_LIGAND

    aa = boonza.load(DATA / "1TEN.pdb")
    assert not _coarse_grained(aa)
    from boonza.martini import martinize

    cg = martinize(aa).system()
    assert _coarse_grained(cg)
    (tmp_path / "probes.json").write_text('{"probes": ["EK", "LL"]}')
    run = tmp_path / "md"
    run.mkdir()

    class Args:
        alignsel = DEFAULT_ALIGN
        ligandsel = DEFAULT_LIGAND

    args = Args()
    _cg_selections(args, cg, [str(run)])  # probes.json sits beside the run
    assert args.alignsel == "name BB"
    assert args.ligandsel == "resname EK LL"
    unchanged = Args()
    _cg_selections(unchanged, aa, [str(run)])  # all-atom: left alone
    assert unchanged.alignsel == DEFAULT_ALIGN and unchanged.ligandsel == DEFAULT_LIGAND


def test_swim_refuses_a_ligand_library_in_martini(tmp_path):
    from boonza.md.swim import main

    code = main([str(DATA / "1TEN.pdb"), "--model", "martini3", "--ligands", "library.sdf",
                 "--workdir", str(tmp_path / "swim")])  # fmt: skip
    assert code == 1


def _martinized(name="1MBN.pdb", **kw):
    from boonza.martini import martinize

    m = martinize(boonza.load(DATA / name), **kw)
    return m, m.system()


def test_coarse_grained_backbone_torsions():
    """BB-BB-BB-BB over four residues in a row, the torsion Martini's own
    helix term acts on; the chain's last three residues start no torsion."""
    from boonza.md.restraints import bead_torsions, coarse_grained

    m, s = _martinized()
    assert coarse_grained(s) and not coarse_grained(boonza.load(DATA / "1MBN.pdb"))
    torsions = bead_torsions(s)
    assert len(torsions) == len(m.ss) - 3
    residue, kind, beads = torsions[0]
    names = [str(s.atoms["name"][b]) for b in beads]
    assert kind == "bb" and names == ["BB"] * 4
    of = [int(s.atoms["residue"][b]) for b in beads]
    assert of == [residue, residue + 1, residue + 2, residue + 3]


def test_restraining_a_coarse_grained_backbone():
    """`bb` holds every torsion, `ss` only the helices and sheets, which needs
    the secondary structure boonza writes when it builds the system."""
    pytest.importorskip("openmm")
    from boonza.martini import OPENMM_OPTIONS
    from boonza.md.restraints import add_dihedral_restraints

    m, s = _martinized(elastic=False)
    _top, system, _pos = boonza.to_openmm(s, **OPENMM_OPTIONS)
    records, what = add_dihedral_restraints(system, s, "bb", 20.0)
    assert len(records) == len(m.ss) - 3 and "protein residues" in what
    assert {r["angle"] for r in records} == {"bb"}
    assert [f.getName() for f in system.getForces()].count("DihedralRestraint") == 1

    _top, system, _pos = boonza.to_openmm(s, **OPENMM_OPTIONS)
    ss_records, what = add_dihedral_restraints(system, s, "ss", 20.0, secondary=m.ss)
    assert 0 < len(ss_records) < len(records) and "helices and sheets" in what
    # myoglobin is helical, and Martini's helix dihedral has its minimum at +60
    angles = np.array([r["reference_degrees"] for r in ss_records])
    assert abs(np.median(angles) - 60) < 15

    _top, system, _pos = boonza.to_openmm(s, **OPENMM_OPTIONS)
    with pytest.raises(ValueError, match="secondary.txt"):
        add_dihedral_restraints(system, s, "ss", 20.0)


def test_the_secondary_structure_is_written_beside_the_topology(tmp_path):
    from boonza.md.prepare import build_martini_system, secondary_beside

    args = parse_arguments([str(DATA / "1TEN.pdb"), "--model", "martini3",
                            "--workdir", str(tmp_path / "run")])  # fmt: skip
    build_martini_system(args, tmp_path, log=lambda *_: None)
    written = (tmp_path / "martini" / "secondary.txt").read_text().strip()
    assert set(written) <= set("HBEGITSC ")
    assert secondary_beside(tmp_path / "martini" / "topol.top") == written
    assert secondary_beside(tmp_path / "topol.top") is None


def test_probes_swim_free_of_the_backbone_restraints(tmp_path):
    args = parse_arguments([str(DATA / "1TEN.pdb"), "--model", "martini3",
                            "--dihedral-restraint", "bb",
                            "--workdir", str(tmp_path / "swim")])  # fmt: skip
    sims = prepare(args, ["EK", "LL"], types=2, copies=1, log=lambda *_: None)
    settings = parse_arguments(["--config", str(sims[0] / "md.toml")])
    assert settings.dihedral_restraint == "bb"
    assert settings.dihedral_restraint_selection == "not (resname EK LL)"


def _probe_run(tmp_path, frames=6):
    """A tiny coarse-grained run: a martinized protein, two probes, and frames
    in which one probe sits on a chosen residue while the other stays away."""
    from boonza.martini import martinize

    protein = martinize(boonza.load(DATA / "1TEN.pdb"), elastic=True)
    probes = [probe("EK"), probe("LL")]
    box = np.full(3, float(np.ptp(protein.positions, axis=0).max()) + 30.0)
    m = build(protein, probes, 1, box, np.random.default_rng(0), salt=0.0)
    s = m.system()
    res = np.asarray(s.atoms["residue"])
    names = np.array([str(x) for x in np.asarray(s.residues["name"])])
    ek = np.flatnonzero(names[res] == "EK")
    target = int(res[np.flatnonzero(names[res] == "LEU")[0]])  # a residue of the protein
    on_target = np.flatnonzero(res == target)
    xyz = np.asarray(s.positions)
    positions = []
    for _ in range(frames):
        frame = xyz.copy()
        frame[ek] = xyz[on_target].mean(0) + np.linspace(0, 2, len(ek))[:, None]  # EK sits on it
        positions.append(frame)
    return s, np.array(positions), int(np.asarray(s.residues["resid"])[target])


def test_probe_contacts(tmp_path):
    """Each residue's share of frames touching each probe."""
    from boonza.probemap import probe_contacts

    s, frames, resid = _probe_run(tmp_path)
    m = probe_contacts(s, [(s, frames)], ["EK", "LL"], cutoff=6.0)
    assert m.frames == len(frames)
    assert m.probes == ["EK", "LL"]
    assert len(m.residues) == len([r for r in np.unique(np.asarray(s.atoms["residue"]))
                                   if str(np.asarray(s.residues["name"])[r]) not in
                                   ("W", "ION", "EK", "LL")])  # fmt: skip
    row = [k for k, (_, r, _) in enumerate(m.residues) if r == resid][0]
    assert m.contacts[row, 0] == 1.0  # EK touches it in every frame
    assert m.contacts[:, 1].max() < 1.0  # LL was left where it was placed
    letters, pooled = m.side_chains()
    assert letters == ["E", "K", "L"]
    assert pooled[row, letters.index("E")] == pooled[row, letters.index("K")] == 1.0
    top = m.top(n=1, by="side-chain")
    assert ("E", m.residues[row], 1.0) in top
    m.to_csv(tmp_path / "probes.csv")
    text = (tmp_path / "probes.csv").read_text().splitlines()
    assert text[0] == "chain,resid,residue,EK,LL"
    assert len(text) == len(m.residues) + 1


def test_the_probes_command(tmp_path, capsys):
    """It reads the probes of each run and writes the table."""
    from boonza.cli import main as cli_main

    s, frames, resid = _probe_run(tmp_path)
    run = tmp_path / "md"
    run.mkdir()
    boonza.save(s, run / "solvated.dms")
    with boonza.open_writer(run / "trajectory.dcd", s.natoms) as w:
        for frame in frames:
            w.write(frame, s.cell)
    (tmp_path / "probes.json").write_text('{"probes": ["EK", "LL"]}')
    assert cli_main(["probes", "--workdir", str(run), "-o", str(tmp_path / "out.csv")]) == 0
    printed = capsys.readouterr().out
    assert "2 probes" in printed and f"LEU{resid}" in printed
    assert (tmp_path / "out.csv").is_file()


def test_feature_maps_run_on_beads(tmp_path, capsys):
    """`--features` types beads by what they stand for, so it works on a
    coarse-grained run rather than quietly finding nothing."""
    from boonza.cli import main as cli_main

    s, frames, _ = _probe_run(tmp_path, frames=4)
    run = tmp_path / "md"
    run.mkdir()
    boonza.save(s, run / "solvated.dms")
    with boonza.open_writer(run / "trajectory.dcd", s.natoms) as w:
        for frame in frames:
            w.write(frame, s.cell)
    (tmp_path / "probes.json").write_text('{"probes": ["EK", "LL"]}')
    assert cli_main(["sites", "--workdir", str(run), "--features"]) == 0
    assert "hotspots" in capsys.readouterr().out


def test_poses_ask_which_probe(tmp_path, capsys):
    """A coarse-grained pocket is BB beads, and the probe has to be named."""
    from boonza.cli import main as cli_main

    s, frames, _ = _probe_run(tmp_path, frames=3)
    boonza.save(s, tmp_path / "system.dms")
    with boonza.open_writer(tmp_path / "traj.dcd", s.natoms) as w:
        for frame in frames:
            w.write(frame, s.cell)
    assert (
        cli_main(["poses", str(tmp_path / "system.dms"), "--traj", str(tmp_path / "traj.dcd")]) == 1
    )
    assert cli_main(["poses", str(tmp_path / "system.dms"), "--traj", str(tmp_path / "traj.dcd"),
                     "--ligandsel", "resname EK"]) == 0  # fmt: skip
    assert "name BB" in capsys.readouterr().out


def _named(system, atoms):
    return [str(np.asarray(system.atoms["name"])[a]) for a in atoms]


def test_beads_are_typed_by_what_they_stand_for():
    """No element to read, but the bead is a known piece of a known residue."""
    from boonza.pharmacophore import ligand_features

    s = probe("FK").system()
    found = {(f, tuple(_named(s, at))) for f, at in ligand_features(s, "all")[0]}
    assert ("Aromatic", ("SC1", "SC2", "SC3")) in found  # the Phe ring, once, at its centre
    assert ("Hydrophobe", ("SC1", "SC2", "SC3")) in found
    assert ("PosIonizable", ("SC2",)) in found and ("Donor", ("SC2",)) in found  # Lys
    assert not [f for f, _ in found if f == "NegIonizable"]

    s = probe("EQ").system()
    found = {(f, tuple(_named(s, at))) for f, at in ligand_features(s, "all")[0]}
    assert ("NegIonizable", ("SC1",)) in found and ("Acceptor", ("SC1",)) in found  # Glu
    assert ("Donor", ("SC1",)) in found  # Gln's amide both donates and accepts


def test_histidine_tells_its_nitrogens_apart():
    """Martini gives ND1-H and NE2 their own beads, so the donor and the
    acceptor are separate features, not one bead that is both."""
    from boonza.pharmacophore import ligand_features

    s = probe("HH").system()
    found = {(f, tuple(_named(s, at))) for f, at in ligand_features(s, "all")[0]}
    assert ("Donor", ("SC2",)) in found
    assert ("Acceptor", ("SC3",)) in found
    assert ("Aromatic", ("SC1", "SC2", "SC3")) in found


@pytest.mark.parametrize(("sequence", "bead", "families"), [
    ("SS", "SC1", {"Donor", "Acceptor"}),   # hydroxyls donate and accept
    ("TT", "SC1", {"Donor", "Acceptor"}),
    ("YY", "SC4", {"Donor", "Acceptor"}),   # the phenol
    ("WW", "SC2", {"Donor"}),               # the indole NH only
    ("RR", "SC2", {"PosIonizable", "Donor"}),
    ("LL", "SC1", {"Hydrophobe"}),
])  # fmt: skip
def test_which_families_a_side_chain_bead_carries(sequence, bead, families):
    from boonza.pharmacophore import ligand_features

    s = probe(sequence).system()
    found = {f for f, at in ligand_features(s, "all")[0] if _named(s, at) == [bead]}
    assert found == families


def test_the_backbone_is_left_out_unless_asked():
    """Every probe carries the same backbone; typing it would mark everywhere a
    probe went."""
    from boonza.pharmacophore import ligand_features

    s = probe("LL").system()
    plain = ligand_features(s, "all")[0]
    assert not [f for f, at in plain if _named(s, at) == ["BB"]]
    with_bb = ligand_features(s, "all", backbone=True)[0]
    bb = {f for f, at in with_bb if _named(s, at) == ["BB"]}
    assert bb == {"Donor", "Acceptor"}


def test_all_atom_ligands_still_go_to_rdkit():
    from boonza.pharmacophore import ligand_features

    s = boonza.from_smiles("c1ccccc1O", name="PHN")
    families = {f for f, _ in ligand_features(s, "all")[0]}
    assert {"Aromatic", "Donor", "Acceptor"} <= families


def test_feature_maps_of_a_coarse_grained_run(tmp_path):
    from boonza.pharmacophore import feature_maps

    s, frames, _ = _probe_run(tmp_path, frames=4)
    maps = feature_maps(s, [(s, frames)], ligand="resname EK LL", align="name BB", spacing=2.0)
    assert {"Donor", "Acceptor", "PosIonizable", "NegIonizable", "Hydrophobe"} <= set(maps)
    assert len(maps["PosIonizable"].places) > 0  # EK carries Lys
    assert len(maps["NegIonizable"].places) > 0  # and Glu
    assert len(maps["Aromatic"].places) == 0  # neither probe has a ring


def test_a_martini_run_holds_the_fold_by_default():
    """Martini does not keep a fold without an elastic network, so `boonza md
    --model martini3` turns one on; all-atom runs never have one."""
    assert parse_arguments([str(DATA / "1TEN.pdb"), "--model", "martini3"]).elastic is True
    assert parse_arguments([str(DATA / "1TEN.pdb"), "--model", "martini2"]).elastic is True
    assert parse_arguments([str(DATA / "1TEN.pdb")]).elastic is False
    off = parse_arguments([str(DATA / "1TEN.pdb"), "--model", "martini3", "--no-elastic"])
    assert off.elastic is False


def test_an_all_atom_run_still_refuses_an_elastic_network(capsys):
    with pytest.raises(SystemExit):
        parse_arguments([str(DATA / "1TEN.pdb"), "--elastic"])
    assert "needs a Martini model" in capsys.readouterr().err


def _bonds_of(directory):
    itp = next((directory / "martini").glob("molecule_*.itp")).read_text()
    section = itp.split("[ bonds ]")[1].split("[", 1)[0]
    return [ln for ln in section.splitlines() if ln.strip()]


def test_the_network_reaches_the_built_topology(tmp_path):
    """Not only the setting: the rubber bands are in the topology that runs."""
    from boonza.md.prepare import build_martini_system

    counts = {}
    for label, extra in (("on", []), ("off", ["--no-elastic"])):
        where = tmp_path / label
        args = parse_arguments([str(DATA / "1TEN.pdb"), "--model", "martini3", *extra,
                                "--workdir", str(where / "run")])  # fmt: skip
        build_martini_system(args, where, log=lambda *_: None)
        counts[label] = len(_bonds_of(where))
    assert counts["on"] > counts["off"] + 300  # 1TEN gets 354 rubber bands


def test_a_coarse_grained_swim_follows_the_same_setting(tmp_path):
    """swim used to carry its own --no-elastic; boonza md's now serves both."""
    from boonza.md.swim import main

    for label, extra in (("on", []), ("off", ["--no-elastic"])):
        root = tmp_path / label
        assert main([str(DATA / "1TEN.pdb"), "--model", "martini3", "--probes", "EK",
                     "--types", "1", "--copies", "1", *extra,
                     "--workdir", str(root)]) == 0  # fmt: skip
    assert len(_bonds_of(tmp_path / "on" / "sim_000")) > len(
        _bonds_of(tmp_path / "off" / "sim_000")
    ) + 300  # fmt: skip


def test_the_built_system_keeps_its_chains_and_positions():
    """`Martinized.system()` reads the topology back for its parameters, but a
    .gro has no chain column and rounds to 0.01 A, so the positions and chains
    come from the beads themselves."""
    from boonza.martini import martinize, solvate

    aa = boonza.load(DATA / "1TEN.pdb").clone("protein")
    peptide = boonza.peptide("KLVFF", conformation="extended")
    peptide.positions = peptide.positions + (aa.positions.max(0) - aa.positions.min(0)) + 20.0
    peptide.chains["name"][:] = "B"
    both = aa.clone()
    both.append(peptide)
    m = martinize(both, "protein")
    s = m.system()
    assert sorted({str(c) for c in s.chains["name"]}) == ["A", "B"]
    beads_of_peptide = len(s.select("chain B").ids)
    assert beads_of_peptide == 15  # 5 residues: a BB each, plus their side chains
    assert beads_of_peptide < len(s.select("chain A").ids)
    assert np.array_equal(np.asarray(s.positions), np.asarray(m.positions))

    wet = solvate(m, padding=12.0, salt=0.15).system()
    assert sorted({str(c) for c in wet.chains["name"]}) == ["", "A", "B"]  # solvent has none
    assert len(wet.select("chain A").ids) == len(s.select("chain A").ids)
    assert len(wet.select("resname W ION").ids) > 100


def test_a_martini_run_stretches_its_intervals():
    """0.01 ns is 500 steps at 20 fs against 5000 at 2 fs, so the checkpoint
    and the monitor move out with the step; anything asked for still wins."""
    cg = parse_arguments([str(DATA / "1TEN.pdb"), "--model", "martini3"])
    assert (cg.checkpoint_interval_ns, cg.monitor_interval_ns, cg.confirmation_checks) == (
        0.1,
        0.2,
        3,
    )
    aa = parse_arguments([str(DATA / "1TEN.pdb")])
    assert (aa.checkpoint_interval_ns, aa.monitor_interval_ns, aa.confirmation_checks) == (
        0.01,
        0.1,
        2,
    )
    asked = parse_arguments(
        [
            str(DATA / "1TEN.pdb"),
            "--model",
            "martini3",
            "--checkpoint-interval-ns",
            "0.01",
            "--confirmation-checks",
            "5",
        ]  # fmt: skip
    )
    assert asked.checkpoint_interval_ns == 0.01 and asked.confirmation_checks == 5
    assert asked.monitor_interval_ns == 0.2


def _bands(m):
    return [
        sum(1 for b in mol.interactions["bonds"] if b.meta.get("group") == "Rubber band")
        for mol in m.molecules
    ]


def _receptor_and_peptide(conformation="helix", touching: float = 3.2):
    """A receptor with a peptide laid against it as chain B, its closest heavy
    atom ``touching`` Å away: near enough to be bound, far enough that nothing
    looks covalently bonded."""
    from boonza.spatial import min_dist2

    receptor = boonza.load(DATA / "1TEN.pdb").clone("protein")
    peptide = boonza.peptide("KLVFFAEDVG", conformation=conformation)
    middle = receptor.positions.mean(0)
    edge = receptor.positions[np.argmax(np.linalg.norm(receptor.positions - middle, axis=1))]
    out = (edge - middle) / np.linalg.norm(edge - middle)
    peptide.positions = peptide.positions - peptide.positions.mean(0) + edge
    for _ in range(200):  # push it out until it only touches
        gap = np.sqrt(min_dist2(peptide.positions, receptor.positions, 20.0).min())
        if gap >= touching:
            break
        peptide.positions = peptide.positions + out * (touching - gap + 0.2)
    peptide.chains["name"][:] = "B"
    both = receptor.clone()
    both.append(peptide)
    return both


def test_the_network_never_joins_two_molecules():
    """A band lives in a molecule's own topology, so a peptide bound to a
    receptor is never tethered to it and can always leave."""
    from boonza.martini import martinize

    both = _receptor_and_peptide()
    m = martinize(both, "protein", elastic=True)
    assert len(m.molecules) == 2
    s = m.system()
    of_atom = np.asarray(s.residues["chain"])[np.asarray(s.atoms["residue"])]
    names = [str(c) for c in s.chains["name"]]
    pairs = np.asarray(s.table("stretch_harm").atoms)[:, :2]
    assert not [1 for i, j in pairs if names[of_atom[i]] != names[of_atom[j]]]


def test_holding_the_receptor_and_leaving_the_peptide_free():
    """A folded peptide picks up a network of its own, which freezes its shape;
    elastic_selection keeps the receptor rigid and leaves it out."""
    from boonza.martini import martinize

    both = _receptor_and_peptide("helix")
    everywhere = _bands(martinize(both, "protein", elastic=True))
    assert everywhere[0] > 300 and everywhere[1] > 0  # the peptide is held too
    receptor_only = _bands(martinize(both, "protein", elastic=True, elastic_selection="chain A"))
    assert receptor_only[0] == everywhere[0]
    assert receptor_only[1] == 0  # and now it is free to change shape

    extended = _bands(martinize(_receptor_and_peptide("extended"), "protein", elastic=True))
    assert extended[1] == 0  # an extended peptide has no pair close enough anyway


def test_elastic_selection_says_when_it_cannot_work():
    from boonza.martini import martinize

    both = _receptor_and_peptide()
    with pytest.raises(ValueError, match="no network to trim"):
        martinize(both, "protein", elastic=False, elastic_selection="chain A")
    with pytest.raises(ValueError, match="selects no residues"):
        martinize(both, "protein", elastic=True, elastic_selection="chain Z")


def test_a_run_holds_only_what_it_is_told_to(tmp_path):
    """The same through `boonza md`'s settings, into the topology it writes."""
    from boonza.md.prepare import build_martini_system

    both = _receptor_and_peptide("helix")
    boonza.save(both, tmp_path / "complex.pdb")
    counts = {}
    for label, extra in (("all", []), ("receptor", ["--elastic-selection", "chain A"])):
        where = tmp_path / label
        args = parse_arguments([str(tmp_path / "complex.pdb"), "--model", "martini3", *extra,
                                "--workdir", str(where / "run")])  # fmt: skip
        build_martini_system(args, where, log=lambda *_: None)
        counts[label] = [
            len(_bonds_of_itp(where / "martini" / f"molecule_{k}.itp")) for k in (0, 1)
        ]
    assert counts["all"][0] == counts["receptor"][0] > 300  # receptor held either way
    assert counts["all"][1] > counts["receptor"][1]  # the peptide only in the first


def _bonds_of_itp(path):
    section = path.read_text().split("[ bonds ]")[1].split("[", 1)[0]
    return [ln for ln in section.splitlines() if ln.strip()]


def test_a_clash_between_chains_is_not_taken_for_a_bond():
    """Heavy atoms too close to be unbonded are taken as bonded, as martinize2
    does.  Between a receptor and its ligand that is a clash, and letting it
    pass would make them one molecule under one elastic network, with the
    ligand unable to leave."""
    from boonza.martini import martinize
    from boonza.martini.build import _check_links_across_chains, _Residue

    clashing = _receptor_and_peptide("extended", touching=0.9)
    with pytest.raises(ValueError, match="come close enough to look bonded"):
        martinize(clashing, "protein", elastic=True)

    def residue(chain, atoms, xyz):
        return _Residue("CYS", 1, "", chain, list(atoms), ["C"] * len(atoms), np.asarray(xyz))

    two = [residue("A", ["CA", "SG"], [[0, 0, 0], [0.2, 0, 0]]),
           residue("B", ["CA", "SG"], [[0.6, 0, 0], [0.4, 0, 0]])]  # fmt: skip
    _check_links_across_chains(two, [(0, 1, 1, 1)], known=frozenset())  # a disulfide: ordinary
    with pytest.raises(ValueError):
        _check_links_across_chains(two, [(0, 0, 1, 0)], known=frozenset())
    _check_links_across_chains(two, [(0, 0, 1, 0)], known={(0, 0, 1, 0)})  # the file says so


def test_a_ligand_is_not_quietly_left_behind(tmp_path):
    """Martini maps proteins and has nothing for a ligand beside one, so a
    complex would come out as the protein alone: a run of something other than
    what was given.  It is refused unless the selection says it is meant."""
    from boonza.md.prepare import build_martini_system

    protein = boonza.load(DATA / "1TEN.pdb").clone("protein")
    ligand = boonza.from_smiles("c1ccccc1O", name="LIG")
    ligand.positions = ligand.positions + protein.positions.max(0) + 6.0
    ligand.chains["name"][:] = "L"
    both = protein.clone()
    both.append(ligand)
    boonza.save(both, tmp_path / "complex.mae")

    args = parse_arguments([str(tmp_path / "complex.mae"), "--model", "martini3",
                            "--workdir", str(tmp_path / "run")])  # fmt: skip
    with pytest.raises(ValueError, match="chain L: LIG"):
        build_martini_system(args, tmp_path, log=lambda *_: None)

    said = []
    chosen = parse_arguments([str(tmp_path / "complex.mae"), "--model", "martini3",
                              "--cg-selection", "protein",
                              "--workdir", str(tmp_path / "run2")])  # fmt: skip
    s, _ = build_martini_system(chosen, tmp_path / "b", log=said.append)
    assert any("left out" in line and "LIG" in line for line in said)
    assert "LIG" not in {str(n) for n in s.residues["name"]}


def test_a_residue_too_small_to_map_is_only_a_note(tmp_path):
    """1TEN opens with an arginine of two atoms, which is a fragment of the
    structure rather than a molecule of its own: it is said, not refused."""
    from boonza.md.prepare import build_martini_system

    said = []
    args = parse_arguments([str(DATA / "1TEN.pdb"), "--model", "martini3",
                            "--workdir", str(tmp_path / "run")])  # fmt: skip
    build_martini_system(args, tmp_path, log=said.append)
    assert any("too little of a residue to map" in line and "ARG1" in line for line in said)


def test_a_view_of_the_system_without_its_rubber_bands(tmp_path):
    """The elastic network is bonds like any other, so a viewer draws it and a
    protein comes out a hairball.  view.dms and view.mae leave it out, keeping
    every atom in its place so a trajectory still lines up with them."""
    from boonza.martini import martinize

    m = martinize(boonza.load(DATA / "1TEN.pdb").clone("protein"), elastic=True)
    whole = m.system()
    bands = m.elastic_bonds()
    assert len(bands) > 300
    viewing = m.for_viewing(whole)
    assert viewing.natoms == whole.natoms
    assert viewing.nbonds == whole.nbonds - len(bands)
    # the backbone is named CA in the view, which is what a viewer traces a
    # chain through; the topology keeps Martini's BB
    assert len(viewing.select("name CA").ids) == len(whole.select("name BB").ids) > 50
    assert not len(viewing.select("name BB").ids)
    assert not len(whole.select("name CA").ids)
    kept = m.for_viewing(whole, backbone_as_ca=False)
    assert len(kept.select("name BB").ids) == len(whole.select("name BB").ids)
    side = [str(n) for n in whole.atoms["name"] if str(n) != "BB"]
    assert [str(n) for n in viewing.atoms["name"] if str(n) != "CA"] == side  # rest untouched
    for i, j in bands[:5]:
        assert viewing.find_bond(viewing.atom(i), viewing.atom(j)) is None

    without = martinize(boonza.load(DATA / "1TEN.pdb").clone("protein"), elastic=False)
    assert without.elastic_bonds() == []


def test_a_martini_run_writes_what_a_viewer_wants(tmp_path):
    from boonza.md.prepare import build_martini_system

    args = parse_arguments([str(DATA / "1TEN.pdb"), "--model", "martini3",
                            "--workdir", str(tmp_path / "run")])  # fmt: skip
    s, _ = build_martini_system(args, tmp_path, log=lambda *_: None)
    view = boonza.load(tmp_path / "view.dms")
    assert (tmp_path / "view.mae").is_file()
    assert view.natoms == s.natoms and view.nbonds < s.nbonds
    assert len(view.select("name CA").ids) == len(s.select("name BB").ids)  # for a viewer
    assert [str(r) for r in view.residues["name"]] == [str(r) for r in s.residues["name"]]
    # the coordinates it carries are the built ones, not a .gro's
    assert np.allclose(np.asarray(view.positions), np.asarray(s.positions))


def test_the_default_work_directories():
    assert parse_arguments(["x.pdb"]).workdir == "boonza_md"
    from boonza.md import swim

    source = swim.main.__code__.co_consts
    assert any(c == "boonza_swim" for c in source if isinstance(c, str))
