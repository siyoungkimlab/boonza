"""OpenFF templates for ligands, through openmmforcefields (boonza.openff).

The tests that type a molecule skip without openmmforcefields and AmberTools; what
is refused, and where the setting goes, is tested everywhere."""

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
    HAVE = False
needs_openff = pytest.mark.skipif(not HAVE, reason="openmmforcefields or AmberTools is missing")

LIGANDS = ("CC(=O)Nc1ccc(O)cc1", "O=C(N)c1ccc[n+](C)c1", "CS(=O)(=O)Nc1ccc(Cl)cc1C#N",
           "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)[O-]")  # fmt: skip


def quiet(*_):
    pass


def _reference(s, charge, name="openff-2.2.1"):
    """The molecule as openmmforcefields itself parameterizes it, an OpenMM system."""
    import openmm.app as app
    from openff.toolkit import Molecule
    from openff.units import unit
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    from boonza.chem import to_rdkit

    mol = Molecule.from_rdkit(to_rdkit(s, implicit_hydrogens=False),
                              allow_undefined_stereo=True, hydrogens_are_explicit=True)  # fmt: skip
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


def test_the_setting_is_md_and_parameterize_only(tmp_path, monkeypatch):
    """ligandff = openff-... reaches boonza md's build; swim and fep, which build their
    ligands with GAFF2, refuse it rather than build with GAFF2 and say nothing."""
    from test_md import DIPEPTIDE, _mapped

    from boonza.cli import main
    from boonza.md.prepare import build_system

    assert "openff-2.2.1" in LIGAND_FORCE_FIELDS and LIGAND_FORCE_FIELDS[0] == "gaff-2.11"
    s = _mapped(*DIPEPTIDE, "A")
    lig = boonza.from_smiles("c1ccccc1", name="BNZ")
    lig.positions = lig.positions + s.positions.mean(0) + np.array([8.0, 0.0, 0.0])
    s.append(lig)
    boonza.save(s, tmp_path / "complex.dms")
    given = []

    class Stop(Exception):
        pass

    def patch(system, ff, **kw):
        given.append(kw)
        raise Stop

    monkeypatch.setattr(openff, "openff_patch", patch)
    args = parse_arguments([str(tmp_path / "complex.dms"), "--ligandff", "openff-2.1.0",
                            "--charge-geometry", "reference"])  # fmt: skip
    with pytest.raises(Stop):
        build_system(args, tmp_path / "md", log=quiet)
    assert given[0]["ligandff"] == "openff-2.1.0" and given[0]["charge_geometry"] == "reference"
    args.pseudopeptide = "protein"
    with pytest.raises(ValueError, match="goes with ligandff = 'gaff-2.11'"):
        build_system(args, tmp_path / "md2", log=quiet)
    for command in ("boonza swim", "boonza fep"):
        with pytest.raises(ValueError, match=f"{command} builds its ligands with GAFF2"):
            gaff.only_gaff2(args, command)
    gaff.only_gaff2(parse_arguments([str(tmp_path / "complex.dms")]), "boonza swim")
    with pytest.raises(SystemExit):
        parse_arguments([str(tmp_path / "complex.dms"), "--ligandff", "openff-9.9.9"])
    boonza.save(lig, tmp_path / "lig.sdf")
    with pytest.raises(SystemExit, match="--openff is for free ligands"):
        main(["parameterize", str(tmp_path / "lig.sdf"), str(tmp_path / "o.dms"),
              "-f", "aa.amber.ff19SB", "--openff", "--gaff2"])  # fmt: skip
