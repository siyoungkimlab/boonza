"""Dipeptide probes and the Martini side of ``boonza swim``."""

from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.martini.probes import (
    LETTERS,
    PROBE_PREFIX,
    PROBE_RESIDUES,
    probe,
    probe_charge,
    probe_sequences,
)
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
    # the molecule is prefixed: a probe named W would lose to Martini's water
    assert m.names == ["probe_WY"]


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
    args = parse_arguments([str(DATA / "1TEN.pdb"), "--model", "martini3", "--gromacs",
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
        # the molecule is prefixed, the residue is not: one would clash with
        # Martini's own water, the other is what selects the probe
        assert f"{PROBE_PREFIX}{name} 2" in top  # two copies of each probe
        assert f'#include "{PROBE_PREFIX}{name}.itp"' in top
    settings = parse_arguments(["--config", str(d / "md.toml")])
    assert settings.model == "martini3" and settings.solvate == "none"
    # the probes have a chain of their own, as an all-atom swim's ligands do
    assert settings.repulsion_selection == "chain LIG" == f"chain {written['chain']}"
    assert settings.production_ns == 10.0
    assert not settings.forcefields  # nothing all-atom came along
    # the run reads the built system, which carries its parameters; the topology
    # beside it is what GROMACS would read
    assert Path(settings.input_structure) == (d / "martini" / "cg.dms").resolve()
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
    assert counts["probe_RR"] == counts["probe_EE"] == 2
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
    # the probes are left out of the fit: a fit on them is a fit on what moves
    assert args.alignsel == "(name BB) and not resname EK LL"
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
    assert settings.dihedral_restraint_selection == "not chain LIG"  # the probes' chain


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
                                "--gromacs", "--workdir", str(where / "run")])  # fmt: skip
        build_martini_system(args, where, log=lambda *_: None)
        counts[label] = len(_bonds_of(where))
    assert counts["on"] > counts["off"] + 300  # 1TEN gets 354 rubber bands


def test_a_coarse_grained_swim_follows_the_same_setting(tmp_path):
    """swim used to carry its own --no-elastic; boonza md's now serves both."""
    from boonza.md.swim import main

    for label, extra in (("on", []), ("off", ["--no-elastic"])):
        root = tmp_path / label
        assert main([str(DATA / "1TEN.pdb"), "--model", "martini3", "--probes", "EK",
                     "--types", "1", "--copies", "1", "--gromacs", *extra,
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


def test_the_written_system_is_what_the_run_reads(tmp_path):
    """cg.dms is what a run started from a built simulation reads, so it has to
    carry the chains an early-stop target is named by, and the positions."""
    from boonza.martini import martinize, solvate

    aa = boonza.load(DATA / "1TEN.pdb").clone("protein")
    peptide = boonza.peptide("KLVFF", conformation="extended")
    peptide.positions = peptide.positions + (aa.positions.max(0) - aa.positions.min(0)) + 20.0
    peptide.chains["name"][:] = "L"
    both = aa.clone()
    both.append(peptide)
    m = solvate(martinize(both, "protein", elastic=True), padding=12.0, salt=0.15, seed=0)
    s = m.system()
    m.save(tmp_path / "martini", system=s)
    back = boonza.load(tmp_path / "martini" / "cg.dms")
    assert len(back.select("chain L").ids) == len(s.select("chain L").ids) == 15
    assert back.nbonds == s.nbonds  # the elastic network among them
    assert np.allclose(np.asarray(back.positions), np.asarray(s.positions))


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
                                "--gromacs", "--workdir", str(where / "run")])  # fmt: skip
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
    # the beads keep their names: a viewer that knows amino acids reads a
    # renamed "GLU: CA SC1" as a broken residue and draws its own bonds over it
    assert [str(n) for n in viewing.atoms["name"]] == [str(n) for n in whole.atoms["name"]]
    assert len(viewing.select("name BB").ids) > 50 and not len(viewing.select("name CA").ids)
    for i, j in bands[:5]:
        assert viewing.find_bond(viewing.atom(i), viewing.atom(j)) is None
    # which is what the cts say, so the file cannot be run by mistake
    from boonza.martini.build import VIEWING_MARK

    assert {str(viewing.ct(c).name) for c in range(viewing.ncts)} == {VIEWING_MARK}
    assert all(str(whole.ct(c).name) != VIEWING_MARK for c in range(whole.ncts))
    # a viewer that wants a CA trace and perceives no bonds of its own can ask
    traced = m.for_viewing(whole, backbone_as_ca=True)
    assert len(traced.select("name CA").ids) == len(whole.select("name BB").ids)

    without = martinize(boonza.load(DATA / "1TEN.pdb").clone("protein"), elastic=False)
    assert without.elastic_bonds() == []


@pytest.mark.parametrize("elastic", [[], ["--no-elastic"]])
def test_a_martini_run_writes_what_a_viewer_wants(tmp_path, elastic):
    """Written whether or not there is a network to leave out, and with the beads
    named as the model names them."""
    from boonza.martini.build import VIEWING_MARK
    from boonza.md.prepare import build_martini_system

    args = parse_arguments([str(DATA / "1TEN.pdb"), "--model", "martini3", *elastic,
                            "--workdir", str(tmp_path / "run")])  # fmt: skip
    s, _ = build_martini_system(args, tmp_path, log=lambda *_: None)
    view = boonza.load(tmp_path / "view.dms")
    assert (tmp_path / "view.mae").is_file()
    assert view.natoms == s.natoms
    assert [str(n) for n in view.atoms["name"]] == [str(n) for n in s.atoms["name"]]
    assert {str(view.ct(c).name) for c in range(view.ncts)} == {VIEWING_MARK}
    assert [str(r) for r in view.residues["name"]] == [str(r) for r in s.residues["name"]]
    # the rubber bands are left out where there are any; "elastic" here is the
    # flag that switches the network off, so an empty one means it is on
    assert view.nbonds == (s.nbonds if elastic else s.nbonds - 354)
    # the coordinates it carries are the built ones, not a .gro's
    assert np.allclose(np.asarray(view.positions), np.asarray(s.positions))
    # and the writer every model shares leaves this one alone: Martini's is the
    # one with the rubber bands taken out, and a copy of the run would put them back
    from boonza.md.run import RunPaths, _write_view

    _write_view(s, RunPaths(tmp_path), log=lambda *_: None)
    assert boonza.load(tmp_path / "view.dms").nbonds == view.nbonds


def test_the_default_work_directories():
    assert parse_arguments(["x.pdb"]).workdir == "boonza_md"
    from boonza.md import swim

    source = swim.main.__code__.co_consts
    assert any(c == "boonza_swim" for c in source if isinstance(c, str))


def test_a_coarse_grained_swim_needs_a_model_it_can_map(tmp_path):
    """boonza martinizes as Martini 3, so a Martini 2 swim would label a Martini 3
    system as Martini 2 rather than build one."""
    from boonza.md.swim import main

    code = main([str(DATA / "1TEN.pdb"), "--model", "martini2", "--probes", "EK",
                 "--workdir", str(tmp_path / "swim")])  # fmt: skip
    assert code == 1
    assert not any((tmp_path / "swim").glob("sim_*/martini"))


@pytest.mark.parametrize("model", ["martini3", "sirah"])
def test_an_all_atom_structure_is_still_mapped(model, tmp_path):
    """A structure has no force field on it, so it is coarse-grained as before;
    only a file that already carries beads and their parameters runs as it is."""
    from boonza.md.prepare import build_martini_system, build_sirah_system

    name = "1CRN_ph7.pdb" if model == "sirah" else "1TEN.pdb"
    where = DATA / "sirah" / name if model == "sirah" else DATA / name
    args = parse_arguments([str(where), "--model", model, "--no-solvate",
                            "--workdir", str(tmp_path / "run")])  # fmt: skip
    build = build_sirah_system if model == "sirah" else build_martini_system
    s, _ = build(args, tmp_path, log=lambda *_: None)
    beads = "GN GC GO" if model == "sirah" else "BB"
    assert len(s.select(f"name {beads}").ids) > 40  # mapped, not run as it is
    assert not len(s.select("hydrogen").ids)


def test_a_parameterized_structure_is_mapped_too(tmp_path):
    """Heavy atoms with a force field on them are still a structure to map: it is
    the model's own beads that say a file is one boonza built."""
    from boonza.md.prepare import build_martini_system

    aa = boonza.load(DATA / "1TEN.pdb")
    heavy = aa.clone(aa.select("not hydrogen").ids)
    table = heavy.add_nonbonded_from_schema()  # a force field, of a kind, and no beads
    param = table.params.add_param(sigma=3.0, epsilon=0.1)
    for atom in range(heavy.natoms):
        table.add_term([atom], param)
    boonza.save(heavy, tmp_path / "heavy.dms")
    args = parse_arguments([str(tmp_path / "heavy.dms"), "--model", "martini3", "--no-solvate",
                            "--workdir", str(tmp_path / "run")])  # fmt: skip
    s, _ = build_martini_system(args, tmp_path, log=lambda *_: None)
    assert len(s.select("name BB").ids) > 40


@pytest.mark.parametrize("model", ["martini3", "sirah"])
def test_the_gromacs_form_is_written_only_when_it_is_asked_for(model, tmp_path):
    """A run reads cg.dms, which carries the parameters; --gromacs adds the same
    system in GROMACS's form beside it, for running or checking it there."""
    from boonza.md.cgswim import prepare

    name = "sirah/1CRN_ph7.pdb" if model == "sirah" else "1TEN.pdb"
    built = "sirah" if model == "sirah" else "martini"
    for flag, wanted in ((None, False), ("--gromacs", True)):
        where = tmp_path / (flag or "default")
        args = parse_arguments([str(DATA / name), "--model", model, *([flag] if flag else []),
                                "--padding-nm", "1.0", "--workdir", str(where)])  # fmt: skip
        (d,) = prepare(args, ["EK"], types=1, copies=1, log=lambda *_: None)
        assert (d / built / "cg.dms").is_file()  # what the run reads, either way
        assert (d / built / "topol.top").is_file() is wanted
        assert (d / built / "cg.gro").is_file() is wanted
        if model == "sirah":
            assert (d / built / "sirah.ff").is_dir() is wanted
        settings = parse_arguments(["--config", str(d / "md.toml")])
        assert Path(settings.input_structure).name == "cg.dms"


def test_a_snapshot_is_not_a_system_to_run(tmp_path):
    """final.dms holds a run's coordinates and bonds, not its parameters, so it
    says so rather than being coarse-grained a second time."""
    from boonza.martini import martinize
    from boonza.md.prepare import build_martini_system, save_structure

    m = martinize(boonza.load(DATA / "1TEN.pdb").clone("protein"), elastic=True)
    save_structure(m.system(), tmp_path / "final.dms")  # as a run writes its snapshots
    args = parse_arguments([str(tmp_path / "final.dms"), "--model", "martini3", "--no-solvate",
                            "--workdir", str(tmp_path / "run")])  # fmt: skip
    with pytest.raises(ValueError, match="carries no parameters"):
        build_martini_system(args, tmp_path, log=lambda *_: None)


@pytest.mark.parametrize("model", ["martini3", "sirah"])
def test_the_probes_have_a_chain_of_their_own(model, tmp_path):
    """Chain LIG, free of the protein's, as an all-atom swim names its ligand
    library's: theirs was the protein's chain once, which left every chain-based
    selection ambiguous."""
    import json

    from boonza.md.cgswim import prepare

    name = "sirah/1CRN_ph7.pdb" if model == "sirah" else "1TEN.pdb"
    built = "sirah" if model == "sirah" else "martini"
    args = parse_arguments([str(DATA / name), "--model", model, "--padding-nm", "1.5",
                            "--workdir", str(tmp_path / "swim")])  # fmt: skip
    (d,) = prepare(args, ["EK", "WY"], types=2, copies=2, log=lambda *_: None)
    s = boonza.load(d / built / "cg.dms")
    probes = json.loads((d / "probes.json").read_text())
    assert probes["chain"] == "LIG"
    res = np.asarray(s.atoms["residue"])
    rn = np.asarray(s.residues["name"])
    lig = s.select("chain LIG").ids
    assert len(lig) and set(rn[res[lig]].tolist()) == set(probes["probes"])
    # and the protein keeps its own, so neither selection catches the other
    protein = s.select("not chain LIG and not resname W ION WT4 NaW ClW").ids
    assert len(protein) and not set(protein.tolist()) & set(lig.tolist())


def test_a_probe_chain_steps_aside_for_a_protein_that_uses_it():
    from boonza.martini import martinize
    from boonza.md.cgswim import probe_chain

    aa = boonza.load(DATA / "1TEN.pdb").clone("protein")
    aa.chains["name"][:] = "LIG"  # a protein whose own chain is called LIG
    assert probe_chain(martinize(aa, "protein")) == "LIG2"


def test_both_coarse_grained_models_write_frames_as_often():
    """How often a frame is written is a trade of disk against time resolution,
    not a force field's business: a bead arrives and leaves faster than an atom,
    and every rate `boonza sites` reports is dwell times counted in frames.  The
    same interval for both also lets runs of each pool into one analysis."""
    from boonza.md.config import DEFAULTS, parse_arguments

    for model in ("martini2", "martini3", "sirah"):
        args = parse_arguments(["x.top", "--model", model])
        assert args.production_report_interval_ns == 0.1
        assert args.checkpoint_interval_ns == 0.1  # and it still divides the interval
    assert DEFAULTS["production_report_interval_ns"] == 1.0  # all-atom boxes are bigger
    assert parse_arguments(["x.pdb"]).production_report_interval_ns == 1.0
    # and it is still a setting, not a rule
    assert parse_arguments(["x.top", "--model", "martini3",
                            "--production-report-interval-ns", "1"]
                           ).production_report_interval_ns == 1.0  # fmt: skip


def test_a_martini_run_needs_no_tables_to_be_typed(tmp_path):
    """Its bead names say what each one stands for, so an analysis reads the
    structure alone however many features it is asked for."""
    from boonza.cli import _run_system
    from boonza.martini import martinize
    from boonza.martini.features import martini_beads

    built = martinize(boonza.load(DATA / "1TEN.pdb").clone("protein")).system()
    path = tmp_path / "solvated.dms"
    boonza.save(built, path)
    for typed in (False, True):
        s = _run_system(path, typed=typed)
        assert "nonbonded" not in s.tables
        assert martini_beads(s, s.select("name BB").ids)


def test_copies_from_a_concentration():
    """Every probe type gets the same number of copies, so only some
    concentrations can be asked for and the answer is the nearest of them."""
    from boonza.md.cgswim import concentration_of, copies_for

    edge = 91.0  # a 9.1 nm box
    assert copies_for(450, edge, 105) == 2
    assert concentration_of(2, edge, 105) == pytest.approx(462.7, abs=0.5)
    assert copies_for(100, edge, 10) == 5  # fewer types a box, so more of each
    assert concentration_of(5, edge, 10) == pytest.approx(110.2, abs=0.5)
    for want in (50, 120, 300, 480, 900):
        got = copies_for(want, edge, 20)
        assert abs(concentration_of(got, edge, 20) - want) <= concentration_of(1, edge, 20) / 2
    assert copies_for(1e-6, edge, 105) == 1  # never none at all
    with pytest.raises(ValueError, match="positive"):
        copies_for(0, edge, 105)
    # twice the box edge is eight times the volume, so an eighth of the copies
    assert copies_for(450, 2 * edge, 105) == 8 * copies_for(450, edge, 105)


def test_a_swim_takes_a_concentration_instead_of_copies(tmp_path):
    """--conc-mM says what the box should hold; the copies follow, the same for
    every type, and each simulation writes down what it came to."""
    import json

    from boonza.md.cgswim import concentration_of, prepare

    said = []
    args = parse_arguments([str(DATA / "1TEN.pdb"), "--model", "martini3", "--padding-nm", "1.0",
                            "--workdir", str(tmp_path / "swim"), "--seed", "1"])  # fmt: skip
    sims = prepare(args, ["EK", "LL", "RR", "WW"], types=2, conc_mM=200.0, log=said.append)
    line = next(x for x in said if "copies of each type" in x)
    assert "200 mM asked" in line
    docs = [json.loads((d / "probes.json").read_text()) for d in sims]
    copies = docs[0]["copies"]
    assert copies >= 1 and line.startswith(f"{copies} copies of each type")
    assert 100.0 < docs[0]["concentration_mM"] < 400.0  # the nearest whole copies can get
    for doc in docs:  # the same copies everywhere, however many types a simulation holds
        assert doc["copies"] == copies
        assert doc["concentration_mM"] > 0
    # a simulation holding more types holds more probe, in the same box
    by_types = {len(doc["probes"]): doc["concentration_mM"] for doc in docs}
    if len(by_types) > 1:
        assert sorted(by_types) == sorted(by_types, key=by_types.get)
    assert concentration_of(copies, 100.0, 2) < concentration_of(copies, 100.0, 4)


def test_a_swim_refuses_a_concentration_and_copies_at_once(tmp_path):
    from boonza.md.swim import main

    code = main([str(DATA / "1TEN.pdb"), "--model", "martini3", "--conc-mM", "200",
                 "--copies", "3", "--workdir", str(tmp_path / "swim")])  # fmt: skip
    assert code == 1
    # and it is not only for probes: a concentration is a count over a volume,
    # so a library of ligands of every size takes it just the same
    from boonza.md.swim import concentration_of, copies_for

    assert copies_for(200, 60.0, 5) == copies_for(200, 60.0, 5)
    assert concentration_of(copies_for(200, 60.0, 5), 60.0, 5) == pytest.approx(200, rel=0.15)


def test_the_single_amino_acid_library():
    """Eighteen probes, one residue each: every amino acid but alanine and glycine,
    which are too small to report anything -- Martini maps both onto a single bead,
    and in SIRAH alanine's side chain is part of its alpha carbon while glycine has
    none at all.  A whole residue, not a side chain cut off its backbone: the beads
    were parameterized with the backbone attached."""
    from boonza.martini.probes import PROBE_RESIDUES, SINGLE_RESIDUES, single_sequences

    assert len(SINGLE_RESIDUES) == 18
    assert not {"ALA", "GLY"} & set(SINGLE_RESIDUES)
    assert set(PROBE_RESIDUES) < set(SINGLE_RESIDUES)  # the dipeptides' residues and more
    letters = single_sequences()
    assert len(letters) == 18 and all(len(x) == 1 for x in letters)
    assert len(set(letters)) == 18


@pytest.mark.parametrize("model", ["martini3", "sirah"])
def test_a_single_amino_acid_probe_carries_a_whole_charge(model):
    """Its charge is its side chain's, and nothing is left dangling: +1 for arginine
    and lysine, -1 for aspartate and glutamate, zero for the rest."""
    import importlib

    probes = importlib.import_module(
        "boonza.sirah.probes" if model == "sirah" else "boonza.martini.probes"
    )
    want = {"R": 1.0, "K": 1.0, "D": -1.0, "E": -1.0}
    for letter in probes.single_sequences():
        s = probes.probe(letter).system()
        q = float(np.asarray(s.atoms["charge"], float).sum())
        assert q == pytest.approx(want.get(letter, 0.0), abs=1e-6)
        assert {str(n) for n in s.residues["name"]} == {letter}  # its own name, to select by


def test_a_single_residue_probe_carries_less_dipole_than_a_dipeptide():
    """Which is the reason for the library: in SIRAH a dipeptide's two neutral
    termini sit a residue apart and make 11 to 12 Debye of it, where one residue's
    are the same bead pair and make half that."""
    from boonza.sirah.probes import probe

    def dipole(seq):
        s = probe(seq).system()
        q = np.asarray(s.atoms["charge"], float)
        x = np.asarray(s.positions)
        assert abs(q.sum()) < 1e-6  # only meaningful for a neutral probe
        return 4.803 * float(np.linalg.norm((q[:, None] * x).sum(0)))

    for one, two in (("L", "LL"), ("F", "FF"), ("S", "SS")):
        assert dipole(one) < 0.65 * dipole(two)


def test_a_swim_swims_the_library_it_is_asked_for(tmp_path):
    """The eighteen single residues dealt out instead of the 105 dipeptides."""
    import json

    from boonza.martini.probes import single_sequences
    from boonza.md.cgswim import prepare

    said = []
    args = parse_arguments([str(DATA / "1TEN.pdb"), "--model", "martini3", "--padding-nm", "1.0",
                            "--workdir", str(tmp_path / "swim"), "--seed", "1"])  # fmt: skip
    sims = prepare(args, single_sequences(), types=9, copies=2, log=said.append)
    assert any("18 amino acid probes" in line for line in said)  # not "dipeptide"
    names = []
    for d in sims:
        doc = json.loads((d / "probes.json").read_text())
        names += doc["probes"]
        assert all(len(q) == 1 for q in doc["probes"])
    assert sorted(names) == sorted(single_sequences())


def test_a_coarse_grained_swim_takes_a_library_by_name(tmp_path):
    """--ligands names the library for every model, so the same eighteen residues
    or the same 105 dipeptides run as beads or as all-atom ligands and the runs
    read against each other.  What a coarse-grained swim cannot take is a file of
    arbitrary ligands, which neither model can map."""
    from boonza.md.swim import library_sequences, main

    assert len(library_sequences("SingleAminoAcid18")) == 18
    assert len(library_sequences("Dipeptide105")) == 105
    assert library_sequences("dipeptide105")[:2] == ["RR", "RQ"]  # whatever its case

    code = main([str(DATA / "1TEN.pdb"), "--model", "martini3", "--ligands", "AstexMiniFrag",
                 "--workdir", str(tmp_path / "sdf")])  # fmt: skip
    assert code == 1  # a library of fragments is not something Martini can map
    code = main([str(DATA / "1TEN.pdb"), "--model", "sirah", "--ligands", "Dipeptide105",
                 "--probes", "EK", "--workdir", str(tmp_path / "both")])  # fmt: skip
    assert code == 1  # a library names its own probes


def test_a_probe_cannot_take_a_name_the_force_field_already_uses():
    """Martini's water is a molecule called W, and the second definition of a name
    silently loses to the first: a tryptophan probe named W would be built as one
    water bead.  The probes' molecules are prefixed, and their beads keep the
    sequence as the residue name, which is what selects them."""
    from boonza.martini import martinize
    from boonza.martini.probes import PROBE_PREFIX, probe

    m = probe("W")
    assert m.names == [PROBE_PREFIX + "W"]
    s = m.system()
    assert s.natoms == len(m.positions) == 6  # not the one bead water would give
    assert {str(n) for n in s.residues["name"]} == {"W"}

    # and where the water is, a clash says so rather than failing on a reshape
    # somewhere else: the topology would build the probe as one water bead
    protein = martinize(boonza.load(DATA / "1TEN.pdb").clone("protein"))
    box = np.full(3, float(np.ptp(protein.positions, axis=0).max()) + 20.0)
    system = build(protein, [m], 2, box, np.random.default_rng(0), salt=0.15)
    assert system.system(None).natoms == len(system.positions)  # prefixed: it builds
    system.names = [n if n != PROBE_PREFIX + "W" else "W" for n in system.names]
    with pytest.raises(ValueError, match="a name Martini already uses"):
        system.system(None)


def test_both_ways_round_is_a_library_of_its_own():
    """SIRAH can tell a probe from its reverse: its chain ends are bead types of
    their own, so EK carries the glutamate at the positive end of the backbone's
    dipole and KE at the negative end.  Martini cannot -- both its backbone beads
    are the same type with no charge -- so there the two are one molecule."""
    from boonza.martini.probes import ordered_sequences, probe_sequences
    from boonza.md.swim import library_sequences

    both = ordered_sequences()
    assert len(both) == 196 == len(set(both)) == 14**2
    assert {"EK", "KE"} <= set(both)
    assert set(probe_sequences()) < set(both)  # the folded library is half of it
    assert len(probe_sequences()) == 105 == 14 * 15 // 2
    assert library_sequences("Dipeptide196") == both

    def dipole(build, seq):
        s = build(seq).system()
        q = np.asarray(s.atoms["charge"], float)
        x = np.asarray(s.positions)
        return 4.803 * float(np.linalg.norm((q[:, None] * (x - x.mean(0))).sum(0)))

    from boonza.martini.probes import probe as martini_probe
    from boonza.sirah.probes import probe as sirah_probe

    # the beads themselves: Martini's two residues are interchangeable, SIRAH's not
    def blocks(m):
        s = m.system()
        nb, res = s.table("nonbonded"), np.asarray(s.atoms["residue"])
        out = {}
        for a in range(s.natoms):
            out.setdefault(int(res[a]), []).append(
                (str(nb.values("type")[a]), round(float(s.atoms["charge"][a]), 2))
            )
        return sorted(tuple(v) for v in out.values())

    assert blocks(martini_probe("EK")) == blocks(martini_probe("KE"))
    assert blocks(sirah_probe("EK")) != blocks(sirah_probe("KE"))
    assert dipole(sirah_probe, "EK") < 0.8 * dipole(sirah_probe, "KE")  # 31.7 D against 44.0
