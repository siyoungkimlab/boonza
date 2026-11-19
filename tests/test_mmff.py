"""MMFF94 ligands (boonza.mmff): the ligand's own energy is RDKit's MMFF94, every
term, through templates, viparr, a .dms file and OpenMM; and the setting reaches
boonza md and boonza swim, while boonza fep refuses it."""

import warnings

import numpy as np
import pytest

import boonza
from boonza import mmff, openff, viparr
from boonza.md.config import LIGAND_FORCE_FIELDS, parse_arguments

pytest.importorskip("openmm")
pytest.importorskip("rdkit")

# one of each thing MMFF94 treats in its own way: amide and phenol, a nitrile and an
# azide (angles linear at rest), small rings, a sulfonamide, a zwitterion, an allene,
# fused aromatics, an amidinium, a carboxylate, water (no torsion at all)
MOLECULES = ["CC(=O)Nc1ccc(O)cc1", "CC#N", "[N-]=[N+]=NCC", "C1CC1C(=O)N2CCC2",
             "CS(=O)(=O)Nc1ccccc1", "C[N+](C)(C)CCOP(=O)([O-])OC", "CC=C=C",
             "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "NC(=[NH2+])c1ccccc1", "CC(C)C(=O)[O-]",
             "O"]  # fmt: skip
_TERMS = {"bond": "SetMMFFBondTerm", "angle": "SetMMFFAngleTerm",
          "stretchbend": "SetMMFFStretchBendTerm", "oop": "SetMMFFOopTerm",
          "torsion": "SetMMFFTorsionTerm", "vdw": "SetMMFFVdWTerm",
          "ele": "SetMMFFEleTerm"}  # fmt: skip
_TABLES = {"bond": ("stretch_harm", "stretch_anharm"),
           "angle": ("angle_harm", "angle_anharm", "angle_linear"),
           "stretchbend": ("stretchbend_harm",), "oop": ("wilson_harm",),
           "torsion": ("dihedral_trig", "dihedral_trig_constant"),
           "nonbonded": ("pair_buf14_7_es", "nonbonded", "pair_12_6_es")}  # fmt: skip


def quiet(*_):
    pass


def _rdkit(s, variant="mmff94"):
    """RDKit's MMFF94 energy of ``s``, term by term."""
    from rdkit.Chem import AllChem

    from boonza.chem import to_rdkit

    mol = to_rdkit(s, implicit_hydrogens=False)
    out = {}
    for name in _TERMS:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=mmff._title(variant))
        for other, switch in _TERMS.items():
            getattr(props, switch)(other == name)
        out[name] = AllChem.MMFFGetMoleculeForceField(mol, props).CalcEnergy()
    out["nonbonded"] = out.pop("vdw") + out.pop("ele")
    return out


def _mine(parameterized):
    e = boonza.openmm_energies(parameterized, constraints=False)
    return {name: sum(e.get(t, 0.0) for t in tables) for name, tables in _TABLES.items()}


def _ligand(smiles, name="LIG", seed=3):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return boonza.from_smiles(smiles, name=name, seed=seed)


@pytest.mark.parametrize("variant", mmff.VARIANTS)
@pytest.mark.parametrize("smiles", MOLECULES)
def test_a_ligand_has_the_energy_rdkit_gives_it_term_by_term(smiles, variant):
    s = _ligand(smiles)
    out = boonza.parameterize(s, [boonza.mmff_patch(s, ligandff=variant)])
    mine, ref = _mine(out), _rdkit(s, variant)
    for name in ref:
        assert mine[name] == pytest.approx(ref[name], abs=2e-4), name
    charge = np.asarray(out.atoms["charge"])
    assert charge.sum() == pytest.approx(round(charge.sum()), abs=1e-6)


def test_mmff94s_is_another_force_field_where_nitrogen_is_delocalized():
    s = _ligand("CC(=O)Nc1ccc(O)cc1")
    a = _mine(boonza.parameterize(s, [boonza.mmff_patch(s)]))
    b = _mine(boonza.parameterize(s, [boonza.mmff_patch(s, ligandff="mmff94s")]))
    assert abs(sum(a.values()) - sum(b.values())) > 0.1


def test_the_energy_is_the_same_bent_out_of_shape():
    """Far from the minimum is where the cubic, quartic and coupling terms count."""
    s = _ligand("CC(=O)Nc1ccc(O)cc1")
    rng = np.random.default_rng(0)
    s.positions = s.positions + rng.normal(0.0, 0.12, s.positions.shape)
    mine = _mine(boonza.parameterize(s, [boonza.mmff_patch(s)]))
    ref = _rdkit(s)
    assert sum(ref.values()) > 50.0
    for name in ref:
        assert mine[name] == pytest.approx(ref[name], abs=1e-3), name


def test_two_copies_take_one_template_and_a_patch_round_trips(tmp_path):
    one = _ligand("CC(=O)Nc1ccc(O)cc1")
    other = one.copy()
    other.positions = other.positions + np.array([30.0, 0.0, 0.0])
    s = one.copy()
    s.append(other)
    patch = boonza.mmff_patch(s, tag="L7")
    assert len(patch.templates) == 1 and patch.templates[0].name == "L7_LIG_mmff94"
    assert all(t.endswith("~L7") for t in patch.templates[0].btype)
    first = viparr.parameterize(s, [patch])
    viparr.write_forcefield(patch, tmp_path / "patch")
    second = viparr.parameterize(s, [viparr.load_forcefield(str(tmp_path / "patch"))])
    for table in mmff.TABLES:
        if table == "angle_linear":
            assert table not in first.table_names
            continue
        assert len(first.table(table)) == len(second.table(table)) > 0
        assert len(first.table(table)) % 2 == 0  # the two copies, term for term
    # no pair of one copy with the other, and each copy's energy is one molecule's
    pairs = first.table("pair_buf14_7_es").atoms
    assert ((pairs < one.natoms).all(1) | (pairs >= one.natoms).all(1)).all()
    # a .dms keeps every table
    boonza.save(first, tmp_path / "two.dms")
    again = boonza.load(tmp_path / "two.dms")
    e1 = boonza.openmm_energies(first)["total"]
    assert boonza.openmm_energies(again)["total"] == pytest.approx(e1, abs=1e-6)


def test_inside_a_ligand_every_pair_is_mmff94s_and_nothing_else():
    """A 1-4 pair of the ligand has no scaled Amber pair beside its MMFF94 one, and
    every pair three bonds apart or more is taken out of the nonbonded sum."""
    s = _ligand("CC(=O)Nc1ccc(O)cc1")
    host = ["aa.amber.ff19SB", "water.tip3p"]
    patch = boonza.mmff_patch(s, host)
    out = boonza.parameterize(s, [viparr.merge_forcefields(host[0], patch), host[1]])
    assert "pair_12_6_es" not in out.table_names
    n = s.natoms
    assert len(out.table("exclusion")) == n * (n - 1) // 2
    mine, ref = _mine(out), _rdkit(s)
    for name in ref:
        assert mine[name] == pytest.approx(ref[name], abs=2e-4), name


def test_a_donor_hydrogen_meets_the_surroundings_without_lennard_jones():
    s = _ligand("CC(=O)NC")
    patch = boonza.mmff_patch(s)
    tpl = patch.templates[0]
    eps = {r.type: r.params["epsilon"] for r in patch.params["vdw1"]}
    nbrs = {i: [] for i in range(len(tpl.names))}
    for i, j in tpl.bonds:
        nbrs[i].append(j)
        nbrs[j].append(i)
    for i, z in enumerate(tpl.anum):
        on_nitrogen = z == 1 and tpl.anum[nbrs[i][0]] == 7
        assert (eps[tpl.btype[i]] == 0.0) == on_nitrogen


def test_mmff94_is_for_free_ligands_it_has_types_for():
    from test_gaff import _cys_adduct

    s = _cys_adduct()
    with pytest.raises(viparr.ViparrError, match="MMFF94 templates are for free ligands"):
        boonza.mmff_patch(s, ["aa.amber.ff19SB"])
    patch = boonza.mmff_patch(s, ["aa.amber.ff19SB"], bound="skip")
    assert not patch.templates and len(patch.skipped) == 1
    with pytest.raises(ValueError, match="one of"):
        boonza.mmff_patch(s, ligandff="mmff95")
    boron = _ligand("OB(O)c1ccccc1")
    with pytest.raises(mmff.MMFFError, match="no atom types"):
        boonza.mmff_patch(boron)
    with pytest.raises(mmff.MMFFError, match="formal charges sum to"):
        boonza.mmff_patch(_ligand("CC(=O)[O-]"), charges={"LIG": 0})


def test_boonza_md_builds_and_runs_with_an_mmff94_ligand(tmp_path):
    import openmm as mm
    import openmm.unit as u
    from test_md import DIPEPTIDE, _mapped

    from boonza.md.prepare import build_system

    s = _mapped(*DIPEPTIDE, "A")
    lig = _ligand("CC(=O)Nc1ccc(O)cc1")
    lig.positions = lig.positions + s.positions.mean(0) + np.array([10.0, 0.0, 0.0])
    s.append(lig)
    boonza.save(s, tmp_path / "complex.dms")
    args = parse_arguments([str(tmp_path / "complex.dms"), "--ligandff", "mmff94",
                            "--padding", "0.8"])  # fmt: skip
    said = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built, _ = build_system(args, tmp_path / "md", log=said.append)
    assert any("MMFF94 (RDKit) templates (MMFF94 charges)" in line for line in said)
    assert (tmp_path / "md" / "mmff_patch" / "templates").is_file()
    assert not (tmp_path / "md" / "gaff2_patch").exists()
    assert set(mmff.TABLES) - {"angle_linear"} <= set(built.table_names)
    assert float(np.asarray(built.atoms["charge"]).sum()) == pytest.approx(0.0, abs=1e-4)
    # the ligand's bonds to hydrogen are constrained like the protein's
    top, system, pos = boonza.to_openmm(built, "PME", cutoff=9.0)
    first = int(np.flatnonzero(built.residues["name"][built.atoms["residue"]] == "LIG")[0])
    held = {system.getConstraintParameters(k)[0] for k in range(system.getNumConstraints())}
    held |= {system.getConstraintParameters(k)[1] for k in range(system.getNumConstraints())}
    hydrogens = [first + i for i in range(lig.natoms) if lig.atoms["anum"][i] == 1]
    assert set(hydrogens) <= held
    context = mm.Context(system, mm.LangevinMiddleIntegrator(300 * u.kelvin, 1 / u.picosecond,
                                                             0.002 * u.picoseconds),
                         mm.Platform.getPlatformByName("CPU"))  # fmt: skip
    context.setPositions(pos)
    mm.LocalEnergyMinimizer.minimize(context, 10.0, 200)
    context.getIntegrator().step(100)
    energy = context.getState(getEnergy=True).getPotentialEnergy()
    assert np.isfinite(energy.value_in_unit(u.kilojoule_per_mole))


def test_the_setting_reaches_md_and_swim_and_fep_refuses_it(tmp_path, monkeypatch):
    from pathlib import Path

    from test_md import DIPEPTIDE, _mapped

    from boonza.cli import main
    from boonza.fep import grow
    from boonza.md import swim
    from boonza.md.prepare import build_system

    assert {"mmff94", "mmff94s"} <= set(LIGAND_FORCE_FIELDS)
    assert openff.whole_molecule("mmff94s") and not openff.whole_molecule("gaff-2.11")
    assert openff.describe_charges("mmff94") == "MMFF94 charges"
    assert openff.charge_model("mmff94") == "mmff94"
    receptor = _mapped(*DIPEPTIDE, "A")
    lig = _ligand("c1ccccc1", name="BNZ")
    lig.positions = lig.positions + receptor.positions.mean(0) + np.array([8.0, 0.0, 0.0])
    s = receptor.copy()
    s.append(lig)
    boonza.save(s, tmp_path / "complex.dms")
    # its own charges only, and no pseudopeptide: boonza md says so before it builds
    for extra, why in ((["--charge-model", "nagl"], "has its own charges"),
                       (["--charge-model", "am1bcc"], "has its own charges"),
                       (["--pseudopeptide", "protein"], "gaff-2.11")):  # fmt: skip
        args = parse_arguments([str(tmp_path / "complex.dms"), "--ligandff", "mmff94", *extra])
        with pytest.raises(ValueError, match=why):
            build_system(args, tmp_path / "md", log=quiet)
    # a swim: each ligand's job makes an MMFF94 patch, and a kept GAFF2 one is made again
    args = parse_arguments([str(tmp_path / "complex.dms"), "--ligandff", "mmff94s"])
    ffs = [viparr.load_forcefield(f) for f in ("aa.amber.ff19SB", "water.tip3p")]
    args.forcefields = [["aa.amber.ff19SB"], ["water.tip3p"]]

    class Ligand:
        code, system, source = "BNZ", lig, None

    root = Path(tmp_path / "ligands")
    said = []
    made = swim.ligand_patches([Ligand], args, ffs, root, 1, said.append)
    assert any("MMFF94s (RDKit) templates (MMFF94 charges)" in line for line in said)
    patch = viparr.load_forcefield(str(made[0]), require_rules=False)
    assert patch.templates[0].name == "BNZ_BNZ_mmff94s" and patch.params["pair_buf14_7_es"]
    assert "mmff94s" in (root / "BNZ" / "charge_geometry").read_text()
    # boonza fep has no hybrid of MMFF94's terms
    args = parse_arguments([str(tmp_path / "complex.dms"), "--ligandff", "mmff94"])
    with pytest.raises(ValueError, match="boonza fep morphs"):
        grow._forcefield(receptor, lig, ffs, args)
    # boonza parameterize --mmff
    boonza.save(lig, tmp_path / "lig.dms")
    out = str(tmp_path / "out.dms")
    assert main(["parameterize", str(tmp_path / "lig.dms"), out, "--mmff"]) == 0
    assert "pair_buf14_7_es" in boonza.load(out).table_names
    with pytest.raises(SystemExit, match="one force field"):
        main(["parameterize", str(tmp_path / "lig.dms"), out, "--mmff", "--openff"])
    with pytest.raises(SystemExit, match="has its own charges"):
        main(["parameterize", str(tmp_path / "lig.dms"), out, "--mmff", "--charge-model", "nagl"])


@pytest.mark.skipif(__import__("test_gaff").AMBERHOME is None, reason="AmberTools is not installed")
def test_what_is_bonded_to_the_protein_takes_gaff2_beside_an_mmff94_ligand():
    """One system with a free ligand and a covalent adduct: MMFF94 for the first, GAFF2
    for the Cys and what is bound to it, in one patch, and the log says which."""
    from test_gaff import _cys_adduct

    s = _cys_adduct()
    free = _ligand("CC(=O)Nc1ccc(O)cc1", name="APA")
    free.positions = free.positions + [25.0, 0.0, 0.0]
    s.append(free)
    ff = viparr.load_forcefield("aa.amber.ff19SB")
    said = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch = openff.ligand_patch(s, [ff], ligandff="mmff94", charge_geometry="reference",
                                    log=said.append)  # fmt: skip
        out = viparr.parameterize(s, [viparr.merge_forcefields(ff, patch)])
    names = sorted(t.name for t in patch.templates)
    assert names[0] == "APA_mmff94" and all("_gaff2" in n for n in names[1:]) and len(names) == 3
    assert len(said) == 1 and "which MMFF94 templates are not for" in said[0]
    first = int(np.flatnonzero(out.residues["name"][out.atoms["residue"]] == "APA")[0])
    pairs = out.table("pair_buf14_7_es").atoms
    assert (pairs >= first).all()  # MMFF94's pairs are the free ligand's alone
    assert float(np.asarray(out.atoms["charge"]).sum()) == pytest.approx(0.0, abs=1e-4)
