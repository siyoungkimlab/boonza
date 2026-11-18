"""OpenFF templates for ligands, through openmmforcefields (boonza.openff).

The tests that type a molecule skip without openmmforcefields and AmberTools; what
is refused, and where the setting goes, is tested everywhere."""

import importlib.util
import os
import warnings

import numpy as np
import pytest

import boonza
from boonza import gaff, openff, viparr
from boonza.md.config import LIGAND_FORCE_FIELDS, parse_arguments

try:
    openff.find_openff()
    AMBERHOME = gaff.find_amber_tools()
    HAVE = True
except (openff.OpenFFError, gaff.AmberToolsError):
    if os.environ.get("BOONZA_REQUIRE_OPENFF"):  # CI's AmberTools job must not skip
        raise
    HAVE = False
needs_openff = pytest.mark.skipif(not HAVE, reason="openmmforcefields or AmberTools is missing")
try:  # found, not imported: "import openff.nagl" would rename boonza's openff here
    NAGL = HAVE and all(importlib.util.find_spec(m) for m in ("openff.nagl", "openff.nagl_models"))
except ImportError:
    NAGL = False
if os.environ.get("BOONZA_REQUIRE_OPENFF") and not NAGL:
    raise ImportError("openff-nagl is missing, and BOONZA_REQUIRE_OPENFF asks for it")
needs_nagl = pytest.mark.skipif(not NAGL, reason="openff-nagl (or openmmforcefields) is missing")

LIGANDS = ("CC(=O)Nc1ccc(O)cc1", "O=C(N)c1ccc[n+](C)c1", "CS(=O)(=O)Nc1ccc(Cl)cc1C#N",
           "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)[O-]")  # fmt: skip


def quiet(*_):
    pass


def _reference(s, charge, name="openff-2.2.1"):
    """The molecule as openmmforcefields itself parameterizes it, an OpenMM system:
    with the partial charges ``charge``, or its own where that is None."""
    import openmm.app as app
    from openff.toolkit import Molecule
    from openff.units import unit
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    from boonza.chem import to_rdkit

    mol = Molecule.from_rdkit(to_rdkit(s, implicit_hydrogens=False),
                              allow_undefined_stereo=True, hydrogens_are_explicit=True)  # fmt: skip
    if charge is not None:
        mol.partial_charges = np.asarray(charge, float) * unit.elementary_charge
    ff = app.ForceField()
    ff.registerTemplateGenerator(
        SMIRNOFFTemplateGenerator(molecules=[mol], forcefield=name).generator)  # fmt: skip
    top = mol.to_topology().to_openmm()
    return top, ff.createSystem(top, nonbondedMethod=app.NoCutoff, constraints=None,
                                removeCMMotion=False)  # fmt: skip


@needs_openff
@pytest.mark.parametrize("smiles", LIGANDS)
def test_a_ligand_has_the_energies_openmmforcefields_gives_it(smiles):
    """Sage's bonds, angles, torsions and impropers to the last digit -- three impropers
    to a planar centre, where a patch used to keep one.  The nonbonded energy is within
    hundredths of a kcal/mol: the 1-4 electrostatics take the Amber host's 0.8333."""
    import openmm.unit as u

    s = boonza.from_smiles(smiles, name="LIG")
    ff = viparr.load_forcefield("aa.amber.ff19SB")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch = openff.openff_patch(s, [ff], charge_geometry="reference", amberhome=AMBERHOME)
        mine = viparr.parameterize(s, [viparr.merge_forcefields(ff, patch)], constraints=False)
        top, system = _reference(s, mine.atoms["charge"])
    assert patch.templates[0].name == "LIG_openff" and patch.templates[0].each_improper
    net = int(np.asarray(s.atoms["formal_charge"]).sum())
    assert float(np.asarray(mine.atoms["charge"]).sum()) == pytest.approx(net, abs=1e-6)
    theirs = boonza.from_openmm(top, system, np.asarray(s.positions) / 10.0 * u.nanometer)
    a, b = boonza.omm.openmm_energies(mine), boonza.omm.openmm_energies(theirs)
    for term in ("stretch_harm", "angle_harm", "dihedral_trig"):
        assert a[term] == pytest.approx(b[term], abs=1e-4), term
    assert a["nonbonded"] == pytest.approx(b["nonbonded"], abs=0.03)


@needs_openff
def test_two_copies_take_one_template_and_a_patch_round_trips(tmp_path):
    s = boonza.from_smiles("CC(=O)Nc1ccc(O)cc1", name="APA")
    other = s.copy()
    other.positions = other.positions + [20.0, 0.0, 0.0]
    s.append(other)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch = openff.openff_patch(s, [], charge_geometry="reference", amberhome=AMBERHOME,
                                    tag="L7")  # fmt: skip
        assert [t.name for t in patch.templates] == ["L7_APA_openff"]
        assert all(t.endswith("~L7") for t in patch.templates[0].btype)
        first = viparr.parameterize(s, [patch], constraints=False)
        # written and read again, the three impropers a centre has are still three
        viparr.write_forcefield(patch, tmp_path / "patch")
        again = viparr.load_forcefield(str(tmp_path / "patch"))
        assert again.templates[0].each_improper
        second = viparr.parameterize(s, [again], constraints=False)
    assert len(first.table("dihedral_trig")) == len(second.table("dihedral_trig"))
    half = s.natoms // 2
    assert np.allclose(first.atoms["charge"][:half], first.atoms["charge"][half:])


@needs_openff
def test_boonza_md_builds_with_an_openff_ligand(tmp_path):
    from test_md import DIPEPTIDE, _mapped

    from boonza.md.prepare import build_system

    s = _mapped(*DIPEPTIDE, "A")
    lig = boonza.from_smiles("CC(=O)Nc1ccc(O)cc1", name="LIG")
    lig.positions = lig.positions + s.positions.mean(0) + np.array([10.0, 0.0, 0.0])
    s.append(lig)
    boonza.save(s, tmp_path / "complex.dms")
    args = parse_arguments([str(tmp_path / "complex.dms"), "--ligandff", "openff-2.2.1",
                            "--charge-geometry", "reference", "--padding", "0.8"])  # fmt: skip
    said = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built, _ = build_system(args, tmp_path / "md", log=said.append)
    assert any("OpenFF openff-2.2.1 templates" in line for line in said)
    assert (tmp_path / "md" / "openff_patch" / "templates").is_file()
    assert not (tmp_path / "md" / "gaff2_patch").exists()
    types = {str(t) for t in built.table("nonbonded").params["type"]}
    assert any(t.endswith("~1") for t in types)  # the ligand's own types
    assert float(np.asarray(built.atoms["charge"]).sum()) == pytest.approx(0.0, abs=1e-4)


def test_openff_is_for_free_ligands():
    """A covalent adduct mixes protein types with the ligand's, which is GAFF2's: said
    before openmmforcefields is looked for, so it is said wherever boonza runs."""
    from test_gaff import _cys_adduct

    with pytest.raises(viparr.ViparrError, match="OpenFF templates are for free ligands"):
        openff.openff_patch(_cys_adduct(), ["aa.amber.ff19SB"])
    with pytest.raises(ValueError, match="named like openff-2.2.1"):
        openff.find_openff("sage")
    # nothing to make: no package is needed to say so
    water = boonza.from_smiles("O", name="HOH")
    assert openff.openff_patch(water, ["aa.amber.ff19SB", "water.tip3p"]).templates == []


def test_the_setting_reaches_md_swim_and_fep(tmp_path, monkeypatch):
    """ligandff = openff-... is what each command's ligands are built with: boonza md's
    build, each ligand's job in a swim (whose kept templates are made again when the
    force field changes), and boonza fep's one patch.  A pseudopeptide, which is GAFF2's
    to mix with protein types, is refused by all three."""
    from pathlib import Path

    from test_md import DIPEPTIDE, _mapped

    from boonza.cli import main
    from boonza.fep import grow
    from boonza.md import swim
    from boonza.md.prepare import build_system

    assert "openff-2.2.1" in LIGAND_FORCE_FIELDS and LIGAND_FORCE_FIELDS[0] == "gaff-2.11"
    receptor = _mapped(*DIPEPTIDE, "A")
    boonza.save(receptor, tmp_path / "receptor.dms")
    lig = boonza.from_smiles("c1ccccc1", name="BNZ")
    lig.positions = lig.positions + receptor.positions.mean(0) + np.array([8.0, 0.0, 0.0])
    s = receptor.copy()
    s.append(lig)
    boonza.save(s, tmp_path / "complex.dms")
    given = []

    class Stop(Exception):
        pass

    def patch(system, ff, **kw):
        given.append(kw)
        raise Stop

    monkeypatch.setattr(openff, "ligand_patch", patch)
    # boonza md
    args = parse_arguments(
        [
            str(tmp_path / "complex.dms"),
            "--ligandff",
            "openff-2.1.0",
            "--charge-geometry",
            "reference",
            "--charge-model",
            "nagl",
        ]
    )
    with pytest.raises(Stop):
        build_system(args, tmp_path / "md", log=quiet)
    assert given[-1]["ligandff"] == "openff-2.1.0" and given[-1]["charge_geometry"] == "reference"
    assert given[-1]["charge_model"] == "nagl"
    # boonza fep (grow's force field is the one function; prep calls the same)
    with pytest.raises(Stop):
        grow._forcefield(receptor, lig, swim._load_forcefields(args.forcefields), args)
    assert given[-1]["ligandff"] == "openff-2.1.0" and given[-1]["charge_geometry"] == "reference"
    # boonza swim: the job of each ligand, and what a kept patch was made with
    boonza.save(lig, tmp_path / "lig.sdf")
    jobs = []

    def job(work):
        d, _, _, opts = work
        jobs.append(opts["ligandff"])
        (Path(d) / "patch").mkdir(exist_ok=True)
        (Path(d) / "patch" / "templates").write_text("", encoding="utf-8")
        return str(Path(d) / "patch")

    monkeypatch.setattr(swim, "_gaff2_job", job)
    sargs = parse_arguments([str(tmp_path / "receptor.dms")])
    ffs = swim._load_forcefields(sargs.forcefields)
    ligands = swim.load_library(tmp_path / "lig.sdf", ffs)
    for ligandff in ("gaff-2.11", "openff-2.2.1", "openff-2.2.1", "openff-2.3.0", "gaff-2.11"):
        sargs.ligandff = ligandff
        swim.ligand_patches(ligands, sargs, ffs, tmp_path / "ligands", log=quiet)
    assert jobs == ["gaff-2.11", "openff-2.2.1", "openff-2.3.0", "gaff-2.11"]
    monkeypatch.undo()
    monkeypatch.setattr(openff, "ligand_patch", patch)
    with pytest.raises(Stop):  # the real job hands an OpenFF ligand to boonza.openff
        (tmp_path / "one").mkdir()
        boonza.save(lig, tmp_path / "one" / "ligand.dms")
        swim._gaff2_job((str(tmp_path / "one"), "L0", [list(e) for e in sargs.forcefields],
                         {"ligandff": "openff-2.2.1", "charge_geometry": "reference",
                          "pseudopeptide": "off"}))  # fmt: skip
    assert given[-1]["ligandff"] == "openff-2.2.1" and given[-1]["tag"] == "L0"
    assert given[-1]["charge_model"] == "auto"
    # NAGL charges are an OpenFF ligand's: GAFF2 has AM1-BCC alone, in every command
    gargs = parse_arguments([str(tmp_path / "complex.dms"), "--charge-model", "nagl"])
    with pytest.raises(ValueError, match="goes with an OpenFF ligandff, not gaff-2.11"):
        build_system(gargs, tmp_path / "md3", log=quiet)
    with pytest.raises(ValueError, match="goes with an OpenFF ligandff"):
        swim.ligand_patches(ligands, gargs, ffs, tmp_path / "ligands3", log=quiet)
    with pytest.raises(ValueError, match="goes with an OpenFF ligandff"):
        grow._forcefield(receptor, lig, ffs, gargs)
    with pytest.raises(SystemExit):
        parse_arguments([str(tmp_path / "complex.dms"), "--charge-model", "resp"])
    # a pseudopeptide is GAFF2's
    args.pseudopeptide = sargs.pseudopeptide = "protein"
    sargs.ligandff = "openff-2.2.1"
    with pytest.raises(ValueError, match="goes with ligandff = 'gaff-2.11'"):
        build_system(args, tmp_path / "md2", log=quiet)
    with pytest.raises(ValueError, match="goes with ligandff = 'gaff-2.11'"):
        swim.ligand_patches(ligands, sargs, ffs, tmp_path / "ligands2", log=quiet)
    with pytest.raises(ValueError, match="goes with ligandff = 'gaff-2.11'"):
        grow._forcefield(receptor, lig, ffs, args)
    with pytest.raises(SystemExit):
        parse_arguments([str(tmp_path / "complex.dms"), "--ligandff", "openff-9.9.9"])
    with pytest.raises(SystemExit, match="--openff is for free ligands"):
        main(["parameterize", str(tmp_path / "lig.sdf"), str(tmp_path / "o.dms"),
              "-f", "aa.amber.ff19SB", "--openff", "--gaff2"])  # fmt: skip


@needs_openff
def test_what_is_bonded_to_the_protein_takes_gaff2_beside_an_openff_ligand():
    """One system with a free ligand and a covalent adduct: OpenFF for the first, GAFF2
    for the Cys and what is bound to it, in one patch, and the log says which."""
    from test_gaff import _cys_adduct

    s = _cys_adduct()
    free = boonza.from_smiles("CC(=O)Nc1ccc(O)cc1", name="APA")
    free.positions = free.positions + [25.0, 0.0, 0.0]
    s.append(free)
    ff = viparr.load_forcefield("aa.amber.ff19SB")
    said = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch = openff.ligand_patch(s, [ff], charge_geometry="reference", log=said.append)
        out = viparr.parameterize(s, [viparr.merge_forcefields(ff, patch)])
    names = sorted(t.name for t in patch.templates)
    assert names[0] == "APA_openff" and all("_gaff2" in n for n in names[1:]) and len(names) == 3
    assert len(said) == 1 and "GAFF2 templates for CYS3+LIG4" in said[0]
    types = [str(t) for t in out.table("nonbonded").params["type"]]
    assert any(t.startswith("C1~") for t in types) and any(t.startswith("c3~") for t in types)
    assert float(np.asarray(out.atoms["charge"]).sum()) == pytest.approx(0.0, abs=1e-4)


@needs_openff
def test_an_openff_pair_makes_a_hybrid_exact_at_its_ends():
    """boonza fep between two OpenFF ligands, toluene to phenol on one ring: each end of
    the hybrid is that ligand as parameterized alone, plus the other's dummies -- three
    impropers to a ring carbon and all."""
    import test_fep_hybrid as H

    from boonza.chem import to_rdkit
    from boonza.fep import hybrid, mapping

    ligs = [boonza.from_smiles(smi, name=n) for smi, n in (("Cc1ccccc1", "TOL"),
                                                            ("Oc1ccccc1", "PHE"))]  # fmt: skip
    both = ligs[0].copy()
    far = ligs[1].copy()
    far.positions = far.positions + [30.0, 0.0, 0.0]
    both.append(far)
    ff = viparr.load_forcefield("aa.amber.ff19SB")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch = openff.ligand_patch(both, [ff], charge_geometry="reference")
        ff = viparr.merge_forcefields(ff, patch)
        made = [viparr.parameterize(s, [ff]) for s in ligs]
    assert sorted(t.name for t in patch.templates) == ["PHE_openff", "TOL_openff"]
    assert {t.btype[0].partition("~")[2] for t in patch.templates} == {"1", "2"}
    sa, sb = (boonza.to_openmm(p, nonbonded_method="NoCutoff", constraints=False,
                               dispersion_correction=False)[1] for p in made)  # fmt: skip
    ra, rb = (to_rdkit(s, implicit_hydrogens=False) for s in ligs)
    core = mapping.map_graph(ra, rb)
    h = hybrid.build(sa, sb, np.asarray(ligs[0].positions) / 10.0,
                     mapping.fit_on_core(ra, rb, core) / 10.0, core, {})  # fmt: skip
    unique_new = [k for k in range(made[1].natoms) if h.new_to_hybrid[k] >= made[0].natoms]
    # the methyl carbon becomes the oxygen, and two of its hydrogens have nowhere to go
    assert len(core) >= 10 and (unique_new or h.unique_old)
    x_old, x_new = h.positions[h.old_to_hybrid], h.positions[h.new_to_hybrid]
    e_old = H._energy(sa, x_old)[1] + H._valence(sb, unique_new, x_new)
    e_new = H._energy(sb, x_new)[1] + H._valence(sa, h.unique_old, x_old)
    assert H._hybrid_energy(h, 0.0) == pytest.approx(e_old, rel=1e-6, abs=1e-3)
    assert H._hybrid_energy(h, 1.0) == pytest.approx(e_new, rel=1e-6, abs=1e-3)


@needs_openff
def test_boonza_fep_preps_runs_and_extends_with_openff_ligands(tmp_path, capsys):
    """The whole of boonza fep on OpenFF ligands: prep, a leg of each kind run, a ligand
    added later whose atoms are not named as an earlier ligand's were (C1~3 beside C1~1),
    and a ligand added under GAFF2 refused -- one network, one ligand force field."""
    from pathlib import Path

    import test_fep_pipeline as T

    from boonza.fep import cli
    from boonza.fep.run import run_leg

    tiny = ["--windows", "3", "--repeats", "1", "--iteration-ps", "0.02",
            "--equilibration-ns", "0.00002", "--production-ns", "0.0001"]  # fmt: skip
    work = tmp_path / "work"
    sage = ["--ligandff", "openff-2.2.1"]
    first = T._sdf(tmp_path / "first.sdf", ["benzene", "toluene"], {})
    assert cli.main(["prep", "--ligands", str(first), "--workdir", str(work), *tiny, *sage]) == 0
    assert "OpenFF openff-2.2.1 (openmmforcefields) templates" in capsys.readouterr().out
    made = viparr.load_forcefield(str(work / "forcefield"), require_rules=False)
    assert [t.name.endswith("_openff") and t.each_improper for t in made.templates] == [True] * 2
    for line in (work / "commands.txt").read_text(encoding="utf-8").splitlines():
        run_leg(Path(line.split()[3]), int(line.split()[-1]), platform="CPU", minimize=20,
                log=lambda *_: None)  # fmt: skip
    assert len(list((work / "edges").glob("*/*/run_1.npz"))) == 2  # solvent and vacuum
    more = T._sdf(tmp_path / "more.sdf", ["benzene", "toluene", "anisole"], {})
    assert cli.main(["prep", "--ligands", str(more), "--workdir", str(work), "--extend",
                     *tiny, *sage]) == 0  # fmt: skip
    made = viparr.load_forcefield(str(work / "forcefield"), require_rules=False)
    tags = [t.btype[0].partition("~")[2] for t in made.templates]
    assert len(made.templates) == 3 and len(set(tags)) == 3
    assert cli.main(["result", str(work)]) == 0
    capsys.readouterr()
    again = T._sdf(tmp_path / "again.sdf", ["benzene", "toluene", "anisole", "phenol"], {})
    assert cli.main(["prep", "--ligands", str(again), "--workdir", str(work), "--extend",
                     *tiny]) == 1  # fmt: skip
    assert "one network, one ligand force field" in capsys.readouterr().err


@needs_nagl
def test_openff_2_3_takes_the_nagl_charges_it_specifies():
    """Sage 2.3.0 says its charges are NAGL's, a graph network trained on AM1-BCC, and
    so they are unless AM1-BCC is asked for: the charges openmmforcefields assigns when
    it is given none, and with them its energies, nonbonded included.  Releases before
    it say AM1-BCC, and take NAGL's when asked."""
    import openmm.unit as u

    assert openff.nagl_model("openff-2.2.1") is None
    assert openff.nagl_model("openff-2.3.0").startswith("openff-gnn-am1bcc")
    assert [openff.charge_model(*x) for x in (("openff-2.2.1", "auto"), ("openff-2.3.0", "auto"),
                                              ("gaff-2.11", "auto"))] == ["am1bcc", "nagl",
                                                                          "am1bcc"]  # fmt: skip
    s = boonza.from_smiles("CS(=O)(=O)Nc1ccc(Cl)cc1C#N", name="LIG")
    ff = viparr.load_forcefield("aa.amber.ff19SB")
    made = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, model in (("openff-2.3.0", "auto"), ("openff-2.3.0", "am1bcc"),
                            ("openff-2.2.1", "nagl"), ("openff-2.2.1", "auto")):  # fmt: skip
            patch = openff.openff_patch(s, [ff], ligandff=name, charge_model=model,
                                        charge_geometry="reference")  # fmt: skip
            made[name, model] = (patch, np.array(patch.templates[0].charge))
        top, system = _reference(s, None, "openff-2.3.0")
        mine = viparr.parameterize(
            s, [viparr.merge_forcefields(ff, made["openff-2.3.0", "auto"][0])], constraints=False
        )
    assert [made[k][0].charge_model for k in made] == ["nagl", "am1bcc", "nagl", "am1bcc"]
    theirs = boonza.from_openmm(top, system, np.asarray(s.positions) / 10.0 * u.nanometer)
    np.testing.assert_allclose(mine.atoms["charge"], theirs.atoms["charge"], atol=1e-5)
    a, b = boonza.omm.openmm_energies(mine), boonza.omm.openmm_energies(theirs)
    for term in ("stretch_harm", "angle_harm", "dihedral_trig"):
        assert a[term] == pytest.approx(b[term], abs=1e-4), term
    assert a["nonbonded"] == pytest.approx(b["nonbonded"], abs=0.03)
    # one model, whichever release asks for it; and not AM1-BCC's numbers
    np.testing.assert_allclose(made["openff-2.3.0", "auto"][1], made["openff-2.2.1", "nagl"][1],
                               atol=1e-6)  # fmt: skip
    assert np.abs(made["openff-2.3.0", "auto"][1] - made["openff-2.3.0", "am1bcc"][1]).max() > 0.01
    for _, q in made.values():
        assert q.sum() == pytest.approx(0.0, abs=1e-6)


@needs_nagl
def test_nagl_charges_a_peptide_sized_ligand_in_seconds():
    """What AM1 cannot do: a ligand of 278 atoms, charged without a quantum calculation
    and without AmberTools, with its net charge and the same charges on the same
    residue wherever it is in the chain."""
    import time

    unit = ("N[C@@H](Cc1ccc(F)cc1)C(=O)N[C@H](C)C(=O)N[C@@H](CCCC[NH3+])C(=O)"
            "N[C@H](CC(=O)[O-])C(=O)")  # fmt: skip
    s = boonza.from_smiles("CC(=O)" + unit * 4 + "NCc1ccccc1", name="LIG", optimize=False)
    assert s.natoms == 278
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch = openff.openff_patch(s, [], ligandff="openff-2.3.0", amberhome="/nowhere")
    assert time.time() - start < 60
    t = patch.templates[0]
    assert patch.charge_model == "nagl" and t.natoms == 278
    assert sum(t.charge) == pytest.approx(0.0, abs=1e-6)  # four Lys+ and four Asp-
    fluorine = [q for q, z in zip(t.charge, t.anum, strict=True) if z == 9]
    assert len(fluorine) == 4 and max(fluorine) - min(fluorine) < 0.02
