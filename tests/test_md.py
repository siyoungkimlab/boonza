"""boonza md: settings, restraints, the detachment monitor, and short simulations."""

import json
import math

import numpy as np
import pytest
from test_gaff import AMBERHOME, _mapped, needs_amber

import boonza
from boonza.md import config, monitor, restraints
from boonza.md.config import parse_arguments
from boonza.md.prepare import build_system, components
from boonza.md.run import RunPaths, run_workflow

DIPEPTIDE = (
    "[CH3:1][C:2](=[O:3])[NH:4][C@@H:5]([CH3:6])[C:7](=[O:8])[NH:9][CH3:10]",
    ["ACE 1 CH3", "ACE 1 C", "ACE 1 O", "ALA 2 N", "ALA 2 CA", "ALA 2 CB", "ALA 2 C",
     "ALA 2 O", "NME 3 N", "NME 3 CH3"],
)  # fmt: skip
SHORT = ["--padding-nm", "0.8", "--platform", "CPU", "--equilibration-ns", "0.001",
         "--equilibration-report-interval-ns", "0.001", "--production-ns", "0.002",
         "--production-report-interval-ns", "0.001", "--checkpoint-interval-ns", "0.001",
         "--performance-interval-ns", "0.001"]  # fmt: skip


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def dipeptide(tmp_path_factory):
    path = tmp_path_factory.mktemp("in") / "dipeptide.dms"
    boonza.save(_mapped(*DIPEPTIDE, "A"), path)
    return path


@pytest.fixture(scope="module")
def two_peptides(tmp_path_factory):
    a, b = _mapped(*DIPEPTIDE, "A"), _mapped(*DIPEPTIDE, "B")
    b.positions = b.positions + np.array([7.0, 0.0, 0.0])
    a.append(b)
    path = tmp_path_factory.mktemp("in") / "two.dms"
    boonza.save(a, path)
    return path


# ---- settings --------------------------------------------------------------------


def test_settings_precedence(tmp_path):
    toml = tmp_path / "s.toml"
    toml.write_text('production_ns = 5.0\nsaltM = 0.1\nforcefields = ["aa.amber.ff14SB", '
                    '["water.tip3p"]]\n')  # fmt: skip
    a = parse_arguments(["x.pdb", "--config", str(toml), "--saltM", "0.2"])
    assert (a.production_ns, a.saltM) == (5.0, 0.2)
    assert a.forcefields == (("aa.amber.ff14SB",), ("water.tip3p",))
    assert {"production_ns", "saltM", "forcefields"} <= a.specified
    assert "temperature" not in a.specified
    d = parse_arguments(["x.pdb"])
    assert d.forcefields == config.DEFAULT_FORCEFIELDS
    assert (d.cutoff_nm, d.integration_fs, d.workdir) == (0.9, 2.0, "openmm_md")
    c = parse_arguments(["x", "-f", "aa.charmm.c36m", "-f", "water.tip3p_charmm"])
    assert c.cutoff_nm == 1.2
    m = parse_arguments(["x", "-f", "aa.amber.ff19SB", "-m", "aa.amber.phosaa19SB", "-f",
                         "water.tip3p"])  # fmt: skip
    assert m.forcefields == (("aa.amber.ff19SB", "aa.amber.phosaa19SB"), ("water.tip3p",))
    assert parse_arguments(["x", "--hmr"]).integration_fs == 4.0
    assert parse_arguments(["x", "--hmr", "--integration-fs", "3"]).integration_fs == 3.0
    x = parse_arguments(["x", "-f", "amber19/protein.ff19SB.xml", "-f", "amber19/opc.xml"])
    assert config.forcefield_kind(x.forcefields) == "xml" and x.cutoff_nm == 0.9
    x = parse_arguments(["x", "-f", "charmm36_2024.xml", "-f", "charmm36_2024/water.xml"])
    assert x.cutoff_nm == 1.2
    assert parse_arguments(["x", "--charge", "LIG=-1"]).ligand_charges == {"LIG": -1}
    assert parse_arguments(["x", "--parent", "MSE=MET"]).parents == {"MSE": "MET"}
    r = parse_arguments(["x", "--repulsion-selection", "chain L", "--repulsion-kJ", "800"])
    assert (r.repulsion_selection, r.repulsion_distance_nm, r.repulsion_kJ) == ("chain L", 0.5, 800)
    sel = parse_arguments(["x", "--dihedral-restraint-selection", "chain A"])
    assert sel.dihedral_restraint_selection == "chain A"
    toml.write_text('dihedral_restraint_selection = "chain A"\n')  # and in a settings file
    assert config.load_configuration(toml)["dihedral_restraint_selection"] == "chain A"


@pytest.mark.parametrize("argv", [
    ["x", "--proteinff", "amber19sb", "--waterff", "opc"],
    ["x", "-f", "amber19/protein.ff19SB.xml", "-f", "water.tip3p"],
    ["x", "-f", "amber19/protein.ff19SB.xml", "-m", "extra.xml"],
    ["x", "-f", "charmm36.xml", "-f", "amber14/tip3p.xml"],  # Amber's TIP3P for CHARMM
    ["x", "-f", "amber19-all.xml"],  # no water model
    ["x", "-f", "aa.charmm.c36m", "-f", "water.tip3p"],
    ["x", "-f", "aa.amber.ff14SB", "-f", "water.tip3p_charmm"],
    ["x", "--monitor-chain", "A", "--monitor-ligand", "ligand-0"],
    ["x", "--monitor-selection", "chain A", "--monitor-component", "component-0"],
    ["x", "-m", "aa.amber.phosaa19SB"],
    ["x", "--saltM", "-1"],
    ["x", "--charge", "LIG"],
    ["x", "--parent", "MSE"],
])  # fmt: skip
def test_settings_errors(argv):
    with pytest.raises(SystemExit):
        parse_arguments(argv)


def test_settings_file_errors(tmp_path):
    for text in ('temperatur = 300\n', 'hmr = "yes"\n', 'precision = "quad"\n',
                 'forcefields = [1]\n'):  # fmt: skip
        p = tmp_path / "bad.toml"
        p.write_text(text)
        with pytest.raises(ValueError):
            config.load_configuration(p)
    p.write_text('proteinff = "amber19sb"\nwaterff = "opc"\n')  # ommflow's, replaced
    with pytest.raises(ValueError, match="list OpenMM XML files in forcefields"):
        config.load_configuration(p)


def test_ion_water_mismatch():
    default = config.DEFAULT_FORCEFIELDS
    assert config.ion_water_mismatch(default) is None
    assert config.ion_water_mismatch([["water.tip3p-fb"], ["ions.amber1jc.tip3p"]]) is None
    assert config.ion_water_mismatch([["water.tip3p_charmm"], ["ions.charmm36"]]) is None
    note = config.ion_water_mismatch([["water.opc"], ["ions.amber1jc.tip3p"],
                                      ["ions.amber1lm_iod.all"]])  # fmt: skip
    assert note == "ions.amber1jc.tip3p: fitted for another water model than water.opc"


def test_water_model_mismatch():
    fits = [
        config.DEFAULT_FORCEFIELDS,
        [["aa.charmm.c36m"], ["water.tip3p_charmm"]],
        [["aa.charmm.c36m"], ["water.tip4p2005"]],  # only TIP3P differs by family
        [["amber14-all.xml"], ["amber14/opc.xml"]],
        [["amber19/protein.ff19SB.xml"], ["amber19/tip3p.xml"]],
        [["charmm36_2024.xml"], ["charmm36_2024/water.xml"]],
        [["amber99sbildn.xml"], ["tip3p.xml"]],  # the older Amber files: top-level water
        [["my_protein.xml"], ["my_water.xml"]],  # your own files are not checked
    ]
    for spec in fits:
        assert config.water_model_mismatch(spec) is None, spec
    wrong = config.water_model_mismatch([["charmm36.xml"], ["amber14/tip3p.xml"]])
    assert wrong.startswith("amber14/tip3p.xml is not a water model of charmm36.xml; use one "
                            "of: charmm36/water.xml, ")  # fmt: skip
    none = config.water_model_mismatch([["amber19-all.xml"], ["amber19/lipid21.xml"]])
    assert none.startswith("amber19-all.xml needs one of its water models: amber19/tip3p.xml, ")
    assert config.water_model_mismatch([["charmm36.xml"], ["amber14-all.xml"], ["tip3p.xml"]])
    charmm = "aa.charmm.c36m takes CHARMM's TIP3P, water.tip3p_charmm, not water.tip3p"
    assert config.water_model_mismatch([["aa.charmm.c36m"], ["water.tip3p"]]) == charmm
    assert (config.water_model_mismatch([["aa.amber.ff14SB"], ["water.tip3p_charmm"]])
            == "aa.amber.ff14SB takes water.tip3p, not CHARMM's water.tip3p_charmm")  # fmt: skip


def test_final_settings_round_trip(tmp_path):
    a = parse_arguments(["x.pdb", "--production-ns", "2", "--charge", "LIG=-1",
                         "--checkpoint-interval-ns", "0.02"])  # fmt: skip
    path = tmp_path / "final.toml"
    config.write_settings(path, config.settings_of(a))
    back = config.load_configuration(path)
    assert back["production_ns"] == 2 and back["ligand_charges"] == {"LIG": -1}
    assert back["forcefields"] == config.DEFAULT_FORCEFIELDS
    b = parse_arguments(["--production-ns", "3"])
    config.restore_settings(b, path)
    assert (b.production_ns, b.checkpoint_interval_ns) == (3.0, 0.02)  # given, then saved
    config.commit_settings(b, path)
    assert config.load_configuration(path)["production_ns"] == 3.0


def test_default_configuration_is_valid(tmp_path):
    path = tmp_path / "t.toml"
    config.write_default_configuration(path)
    assert config.load_configuration(path)["saltM"] == 0.15


# ---- restraints and monitor --------------------------------------------------------


def test_restraint_well_sits_on_the_reference():
    theta0 = 0.7

    def energy(theta):
        return sum(k * (1 + math.cos(n * theta - (n * (theta0 + math.pi)) % (2 * math.pi)))
                   for n, k in restraints.fourier_terms(20.0))  # fmt: skip

    xs = np.linspace(-math.pi, math.pi, 4001)
    e = np.array([energy(x) for x in xs])
    assert abs(xs[np.argmin(e)] - theta0) < 2e-3
    h = 1e-4
    curvature = (energy(theta0 + h) - 2 * energy(theta0) + energy(theta0 - h)) / h**2
    assert curvature == pytest.approx(restraints._curvature(20.0), rel=1e-4)
    assert list(restraints.fourier_terms(-20.0)) == list(restraints.fourier_terms(20.0))


def test_backbone_torsions(dipeptide):
    s = boonza.load(dipeptide)
    names = s.atoms["name"]
    got = [(k, tuple(str(names[a]) for a in t)) for _, k, t in restraints.backbone_torsions(s)]
    assert got == [("phi", ("C", "N", "CA", "C")), ("psi", ("N", "CA", "C", "N"))]


def test_restraints_bb_and_ss_on_a_helix():
    import openmm as mm

    s = boonza.peptide("AAAAAAAAAAAA", conformation="helix")
    counts = {}
    for mode in ("bb", "ss"):
        system = mm.System()
        for _ in range(s.natoms):
            system.addParticle(12.0)
        records, _ = restraints.add_dihedral_restraints(system, s, mode, 20.0)
        counts[mode] = len(records)
        assert system.getForce(0).getNumTorsions() == 6 * len(records)
    assert counts == {"bb": 22, "ss": 20}  # DSSP: CHHHHHHHHHHC; termini have one torsion
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(s.positions / 10)
    e0 = ctx.getState(getEnergy=True).getPotentialEnergy()._value
    ctx.setPositions(
        (s.positions + np.random.default_rng(0).normal(0, 0.3, s.positions.shape)) / 10
    )
    assert ctx.getState(getEnergy=True).getPotentialEnergy()._value > e0 + 10


@pytest.fixture(scope="module")
def two_ligands(tmp_path_factory):
    """The dipeptide with a benzene (chain L) and a toluene (chain M) beside it."""
    s = _mapped(*DIPEPTIDE, "A")
    for smiles, name, chain, dx in (("c1ccccc1", "BNZ", "L", 6.5), ("Cc1ccccc1", "TOL", "M", -6.5)):
        lig = boonza.from_smiles(smiles, name=name)
        lig.chains["name"] = np.array([chain])
        lig.positions = lig.positions - lig.positions.mean(0) + s.positions.mean(0) + [dx, 0, 0]
        s.append(lig)
    path = tmp_path_factory.mktemp("in") / "two_ligands.dms"
    boonza.save(s, path)
    return path


def test_early_stop_target_with_several_ligands(tmp_path, two_ligands):
    from boonza.gaff import find_unmatched
    from boonza.md.prepare import component_table, load_input

    s = load_input(two_ligands)
    info = components(s, find_unmatched(s, [boonza.load_forcefield("aa.amber.ff19SB")]))
    table = component_table(info)
    assert "--monitor-ligand ligand-0" in table and "--monitor-ligand ligand-1" in table
    assert [c["ligand_id"] for c in info["components"]] == [None, "ligand-0", "ligand-1"]

    def pick(*argv):
        return monitor.select_target(info, parse_arguments(["x", "--early-stop", *argv]))["id"]

    assert pick("--monitor-ligand", "ligand-1") == "component-2"
    assert pick("--monitor-chain", "M") == "component-2"
    assert pick("--monitor-component", "component-1") == "component-1"
    from boonza.md.prepare import select_atoms

    info["selection"] = select_atoms(s, "resname TOL and not hydrogen")
    got = monitor.select_target(
        info, parse_arguments(["x", "--early-stop", "--monitor-selection", "resname TOL"])
    )
    assert got["_kind"] == "selection" and len(got["input_atom_indices"]) == 7
    assert got["chains"] == ["M"]
    for bad in ("resname XYZ", "hydrogen and resname TOL"):
        with pytest.raises(ValueError, match="no heavy atoms"):
            select_atoms(s, bad)
    for bad in ((), ("--monitor-ligand", "ligand-5"), ("--monitor-chain", "Z")):
        with pytest.raises(ValueError):
            pick(*bad)
    # a missing choice stops the run before anything is built (no AmberTools needed)
    work = tmp_path / "run"
    with pytest.raises(ValueError, match="2 ligands"):
        run_workflow(parse_arguments([str(two_ligands), "--workdir", str(work), *SHORT,
                                      "--early-stop"]), log=quiet)  # fmt: skip
    assert not work.exists()  # a refused new run leaves nothing, so the fixed command starts afresh
    work.mkdir()  # an empty directory (as batch schedulers make) is left empty
    with pytest.raises(ValueError, match="2 ligands"):
        run_workflow(parse_arguments([str(two_ligands), "--workdir", str(work), *SHORT,
                                      "--early-stop"]), log=quiet)  # fmt: skip
    assert work.is_dir() and not any(work.iterdir())


def test_repulsion_keeps_molecules_apart():
    import openmm as mm

    a, b = boonza.from_smiles("CC", name="ETH"), boonza.from_smiles("CC", name="ETH")
    b.positions = b.positions - b.positions.mean(0) + a.positions.mean(0) + [3.0, 0.0, 0.0]
    a.append(b)
    system = mm.System()
    for _ in range(a.natoms):
        system.addParticle(12.0)
    assert restraints.add_repulsion(system, a, "all", 0.5, 500.0) == 2
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(a.positions / 10)
    energy = ctx.getState(getEnergy=True).getPotentialEnergy()._value
    heavy = np.flatnonzero(a.atoms["anum"] > 1)
    frag = np.asarray(a.fragids)
    want = 0.0
    for i in heavy:
        for j in heavy:
            if i < j and frag[i] != frag[j]:  # different molecules only: C-C inside is excluded
                r = np.linalg.norm(a.positions[i] - a.positions[j]) / 10
                want += 500.0 * max(0.0, 0.5 - r) ** 2
    assert want > 0 and energy == pytest.approx(want, rel=1e-5)
    one = mm.System()
    for _ in range(a.natoms):
        one.addParticle(12.0)
    assert restraints.add_repulsion(one, a, "index 0 to 7", 0.5, 500.0) == 1  # one molecule
    assert one.getNumForces() == 0


def test_monitor_measures_across_the_box(tmp_path):
    box = np.eye(3) * 3.0
    pos = np.array([[0.1, 0.0, 0.0], [2.9, 0.0, 0.0], [1.5, 1.5, 1.5]])
    dist, n = monitor.measure([0], [1, 2], pos, box, 0.3)
    assert dist == pytest.approx(0.2) and n == 1
    assert monitor.detachment_update(1.0, 0, 0.8, 0, 2) == (1, False)
    assert monitor.detachment_update(1.0, 0, 0.8, 1, 2) == (2, True)
    assert monitor.detachment_update(1.0, 3, 0.8, 1, 2) == (0, False)
    csv = tmp_path / "monitor.csv"
    row = dict.fromkeys(monitor.FIELDS, 0)
    for step in (10, 20, 20):
        monitor.append_row(csv, {**row, "step": step, "consecutive_detached_count": step // 10})
    assert len(csv.read_text().splitlines()) == 3  # the repeated step is not written twice
    assert monitor.restore_count(csv, 15) == 1
    with pytest.raises(ValueError):
        monitor.append_row(csv, {**row, "step": 5})


def test_components(two_peptides):
    s = boonza.load(two_peptides)
    info = components(s, [[1]])
    assert [(c["id"], c["classification"], c["chains"]) for c in info["components"]] == [
        ("component-0", "modified", ["A"]), ("component-1", "standard", ["B"])]  # fmt: skip
    info = components(s, [[3, 4, 5]])
    assert info["components"][1]["ligand_id"] == "ligand-0"


# ---- simulations -------------------------------------------------------------------


def test_new_run_and_restarts(tmp_path, dipeptide):
    work = tmp_path / "run"
    run_workflow(parse_arguments([str(dipeptide), "--workdir", str(work), *SHORT,
                                  "--dihedral-restraint", "bb"]), log=quiet)  # fmt: skip
    p = RunPaths(work)
    for f in ("solvated_dms", "solvated_pdb", "solvated_mae", "components_json", "system_xml",
              "integrator_xml", "checkpoint", "final_configuration", "equilibrated_pdb",
              "equilibrated_mae", "final_pdb", "final_mae", "equilibration_dcd",
              "equilibration_csv", "trajectory_dcd", "state_csv", "performance_csv",
              "dihedral_restraints_csv"):  # fmt: skip
        assert getattr(p, f).is_file(), f
    assert len(p.state_csv.read_text().splitlines()) == 3  # header, 1 and 2 ps
    s = boonza.load(p.solvated_dms)
    assert "DihedralRestraint" in p.system_xml.read_text()
    assert s.nresidues > 100 and "MonteCarloBarostat" in p.system_xml.read_text()
    # a larger target resumes and appends
    run_workflow(parse_arguments(["--workdir", str(work), "--production-ns", "0.003"]), log=quiet)
    assert len(p.state_csv.read_text().splitlines()) == 4
    assert config.load_configuration(p.final_configuration)["production_ns"] == 0.003
    run_workflow(parse_arguments(["--workdir", str(work)]), log=quiet)  # nothing left to do
    assert len(p.state_csv.read_text().splitlines()) == 4
    with pytest.raises(ValueError, match="integration_fs"):
        run_workflow(parse_arguments(["--workdir", str(work), "--integration-fs", "1"]),
                     log=quiet)  # fmt: skip


def test_early_stop_confirms_detachment(tmp_path, two_peptides):
    work = tmp_path / "es"
    argv = [str(two_peptides), "--workdir", str(work), *SHORT, "--production-ns", "0.004",
            "--early-stop", "--monitor-selection", "chain B", "--monitor-interval-ns", "0.001",
            "--pocket-cutoff-nm", "1.5", "--contact-cutoff-nm", "0.05",
            "--detach-cutoff-nm", "0.1"]  # fmt: skip
    run_workflow(parse_arguments(argv), log=quiet)
    p = RunPaths(work)
    status = json.loads(p.status_json.read_text())
    assert status["outcome"] == "detached" and status["final_production_step"] == 1000
    pocket = json.loads(p.pocket_json.read_text())
    assert (pocket["selector_kind"], pocket["selector_value"]) == ("selection", "chain B")
    assert len(pocket["ligand_atom_indices"]) == 22  # the whole second dipeptide
    with pytest.raises(ValueError, match="differs from the saved target"):
        run_workflow(parse_arguments(["--workdir", str(work), "--monitor-selection", "chain A"]),
                     log=quiet)  # fmt: skip
    rows = p.monitor_csv.read_text().splitlines()
    assert len(rows) == 3 and rows[-1].endswith("true")
    run_workflow(parse_arguments(["--workdir", str(work)]), log=quiet)  # stays stopped
    assert p.monitor_csv.read_text().splitlines() == rows


def test_solvate_false_runs_the_input_as_it_is(tmp_path, dipeptide):
    from boonza.build import neutralize, solvate

    built = neutralize(solvate(boonza.load(dipeptide), thickness=8.0), concentration=0.15)
    path = tmp_path / "built.dms"
    boonza.save(built, path)
    args = parse_arguments([str(path), "--no-solvate"])
    assert args.solvate == "none"  # --no-solvate is the alias
    assert parse_arguments(["x"]).solvate == "box"
    assert parse_arguments(["x", "--solvate", "fill"]).solvate == "fill"
    toml = tmp_path / "old.toml"
    toml.write_text("solvate = false\n")  # a boolean, from before fill was a mode
    assert config.load_configuration(toml)["solvate"] == "none"
    lines = []
    out, info = build_system(args, tmp_path, log=lines.append)
    assert out.natoms == built.natoms  # no water or ions added
    assert np.allclose(out.cell, built.cell)
    assert abs(out.atoms["charge"].sum()) < 1e-6
    assert any(line.startswith("System: ") for line in lines)
    assert info["components"][0]["production_atom_indices"][:2] == [0, 1]
    lines.clear()  # settings that no longer apply are named
    build_system(parse_arguments([str(path), "--no-solvate", "--saltM", "0.2"]), tmp_path,
                 log=lines.append)  # fmt: skip
    assert any("saltM is not used" in line for line in lines)
    with pytest.raises(ValueError, match="needs a periodic cell"):
        build_system(parse_arguments([str(dipeptide), "--no-solvate"]), tmp_path, log=quiet)


def test_solvate_fill_uses_the_inputs_own_box(tmp_path, dipeptide):
    s = boonza.load(dipeptide)
    s.cell = np.diag([34.0, 34.0, 34.0])
    s.positions = s.positions - s.positions.mean(0)
    path = tmp_path / "boxed.dms"
    boonza.save(s, path)
    lines = []
    out, _ = build_system(parse_arguments([str(path), "--solvate", "fill", "--saltM", "0"]),
                          tmp_path, log=lines.append)  # fmt: skip
    assert np.allclose(np.diag(out.cell), 34.0)  # the input's own box, not a new one
    assert len(out.select("water").ids) > 300  # its empty space is filled
    assert any("Filled the input's box" in line for line in lines)
    lines.clear()  # padding does not apply, and is named
    build_system(parse_arguments([str(path), "--solvate", "fill", "--padding-nm", "1.5"]),
                 tmp_path, log=lines.append)  # fmt: skip
    assert any("padding_nm is not used" in line for line in lines)
    with pytest.raises(ValueError, match="needs a periodic cell"):
        build_system(parse_arguments([str(dipeptide), "--solvate", "fill"]), tmp_path, log=quiet)
    s.cell = np.array([[34.0, 0, 0], [3.0, 34.0, 0], [0, 0, 34.0]])
    boonza.save(s, path)
    with pytest.raises(ValueError, match="rectangular cell"):
        build_system(parse_arguments([str(path), "--solvate", "fill"]), tmp_path, log=quiet)


def test_barostat_choices(tmp_path, dipeptide):
    a = parse_arguments(["x", "--barostat", "membrane", "--surface-tension", "20"])
    assert (a.barostat, a.surface_tension) == ("membrane", 20.0)
    assert parse_arguments(["x"]).barostat == "isotropic"
    with pytest.raises(SystemExit):  # a tension without the barostat that uses it
        parse_arguments(["x", "--surface-tension", "20"])
    work = tmp_path / "mb"
    run_workflow(parse_arguments([str(dipeptide), "--workdir", str(work), *SHORT,
                                  "--barostat", "membrane"]), log=quiet)  # fmt: skip
    xml = RunPaths(work).system_xml.read_text()
    assert "MonteCarloMembraneBarostat" in xml
    dry = tmp_path / "nvt"
    run_workflow(parse_arguments([str(dipeptide), "--workdir", str(dry), *SHORT,
                                  "--barostat", "none"]), log=quiet)  # fmt: skip
    assert "Barostat" not in RunPaths(dry).system_xml.read_text()


def test_openmm_xml_route_builds(tmp_path, dipeptide):
    args = parse_arguments([str(dipeptide), "-f", "amber19/protein.ff19SB.xml", "-f",
                            "amber19/tip3p.xml", "--padding-nm", "0.8"])  # fmt: skip
    lines = []
    s, info = build_system(args, tmp_path, log=lines.append)
    assert lines[0] == "Force fields: amber19/protein.ff19SB.xml, amber19/tip3p.xml"
    assert info["components"][0]["classification"] == "standard"
    assert {"stretch_harm", "angle_harm", "dihedral_trig", "nonbonded"} <= set(s.table_names)
    assert abs(s.atoms["charge"].sum()) < 1e-6


@needs_amber
def test_ligand_gets_gaff2_templates(tmp_path, dipeptide):
    s = boonza.load(dipeptide)
    bnz = boonza.from_smiles("c1ccccc1", name="BNZ")
    bnz.positions = bnz.positions + s.positions.mean(0) + np.array([6.5, 0.0, 0.0])
    s.append(bnz)
    path = tmp_path / "complex.dms"
    boonza.save(s, path)
    assert AMBERHOME is not None  # found through $AMBERHOME or PATH, as gaff2_patch finds it
    args = parse_arguments([str(path), "--padding-nm", "0.8"])
    out, info = build_system(args, tmp_path, log=quiet)
    assert [c["ligand_id"] for c in info["components"]] == [None, "ligand-0"]
    assert (tmp_path / "gaff2_patch" / "templates").is_file()
    lig = info["components"][1]["production_atom_indices"]
    assert abs(out.atoms["charge"][lig].sum()) < 1e-6


def test_load_input_gathers_split_residues(tmp_path):
    """Preparation tools write the hydrogens they add at the end of the file,
    which splits a residue in two; OpenMM refuses such a topology."""
    from boonza.md.prepare import load_input

    s = boonza.peptide("AAA")  # every heavy atom first, then every hydrogen
    res = s.atoms["residue"]
    assert 1 + np.count_nonzero(np.diff(res) != 0) > s.nresidues  # split as written
    path = tmp_path / "split.mae"
    boonza.save(s, path)
    with pytest.raises(ValueError, match="contiguous"):
        boonza.to_openmm(boonza.load(path))

    said = []
    got = load_input(path, log=said.append)
    out = got.atoms["residue"]
    assert 1 + np.count_nonzero(np.diff(out) != 0) == got.nresidues == s.nresidues
    assert got.natoms == s.natoms and got.nbonds == s.nbonds
    assert np.allclose(s.positions[got.atoms["md_index"] - 1], got.positions)
    assert said and "3 residue(s)" in said[0]
    top, _, _ = boonza.to_openmm(got)
    assert top.getNumAtoms() == s.natoms and top.getNumResidues() == s.nresidues


def test_split_residue_input_keeps_its_recorded_indices(tmp_path, dipeptide):
    """A file that keeps a residue in pieces is gathered, so the input's own
    order is no longer its order in the file; what components.json records
    must still point at the same atoms."""
    from boonza.md.prepare import load_input

    s = boonza.load(dipeptide)  # move the first residue's hydrogens to the end,
    first = s.atoms["residue"] == 0  # the way preparation tools write added ones
    moved = np.flatnonzero(first & (s.atoms["anum"] == 1))
    s.reorder_atoms(np.concatenate([np.setdiff1d(np.arange(s.natoms), moved), moved]))
    path = tmp_path / "split.mae"
    boonza.save(s, path)

    lines = []
    out, info = build_system(parse_arguments([str(path), "--saltM", "0"]), tmp_path,
                             log=lines.append)  # fmt: skip
    assert any("Gathered the atoms of 1 residue(s)" in line for line in lines)
    loaded = load_input(path)
    for c in info["components"]:
        a = np.asarray(c["input_atom_indices"])
        b = np.asarray(c["production_atom_indices"])
        assert (loaded.atoms["name"][a] == out.atoms["name"][b]).all()
        assert (loaded.atoms["anum"][a] == out.atoms["anum"][b]).all()


def test_every_run_says_how_far_it_got(tmp_path, dipeptide):
    """status.json is not only for watched runs: 'has this finished, and where is it'
    should be one file to read whether or not a ligand is being followed."""
    work = tmp_path / "plain"
    run_workflow(parse_arguments([str(dipeptide), "--workdir", str(work), *SHORT]), log=quiet)
    p = RunPaths(work)
    doc = json.loads(p.status_json.read_text())
    assert doc["outcome"] == "target_reached"
    assert doc["early_stop_enabled"] is False
    assert doc["final_production_time_ns"] == pytest.approx(0.002)
    assert doc["final_production_step"] == 1000
    assert "ligand_id" not in doc and "consecutive_detached_count" not in doc
    assert not p.pocket_json.exists() and not p.monitor_csv.exists()  # those stay for the monitor

    longer = [*SHORT]  # resuming to a further target says so, from the same file
    longer[longer.index("--production-ns") + 1] = "0.004"
    run_workflow(parse_arguments(["--workdir", str(work), *longer]), log=quiet)
    doc = json.loads(p.status_json.read_text())
    assert doc["final_production_time_ns"] == pytest.approx(0.004)
    assert doc["target_production_ns"] == 0.004


@needs_amber  # two_ligands is parameterized with GAFF2
def test_a_watched_run_resumed_unwatched_keeps_what_it_found(tmp_path, two_ligands):
    # seeded: without one the velocities differ every run, so whether the ligand
    # reads as detached in two picoseconds is a coin toss, and this asserts what
    # the record says afterwards
    work = tmp_path / "was_watched"
    argv = [str(two_ligands), "--workdir", str(work), *SHORT, "--seed", "1", "--early-stop",
            "--monitor-ligand", "ligand-0"]  # fmt: skip
    run_workflow(parse_arguments(argv), log=quiet)
    watched = json.loads(RunPaths(work).status_json.read_text())
    assert watched["early_stop_enabled"] is True and watched["ligand_id"] == "ligand-0"

    longer = [*SHORT]
    longer[longer.index("--production-ns") + 1] = "0.004"
    run_workflow(parse_arguments(["--workdir", str(work), *longer, "--no-early-stop"]), log=quiet)
    doc = json.loads(RunPaths(work).status_json.read_text())
    # the messages carry the record, so a failure here says what it found
    assert doc["early_stop_enabled"] is False, doc  # no longer watched
    assert doc["ligand_id"] == "ligand-0", doc  # but what it was watching is not forgotten
    assert doc["consecutive_detached_count"] == 0, doc
    assert doc["outcome"] in ("target_reached", "detached"), doc
