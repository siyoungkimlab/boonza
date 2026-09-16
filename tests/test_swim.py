"""boonza swim: sizes, diverse dealing, ligand force fields, placement, simulations."""

import csv

import numpy as np
import pytest
from rdkit import Chem
from test_gaff import AMBERHOME, _mapped, needs_amber

import boonza
from boonza import gaff, viparr
from boonza.md import swim
from boonza.md.config import parse_arguments
from boonza.md.run import run_workflow

DIPEPTIDE = (
    "[CH3:1][C:2](=[O:3])[NH:4][C@@H:5]([CH3:6])[C:7](=[O:8])[NH:9][CH3:10]",
    ["ACE 1 CH3", "ACE 1 C", "ACE 1 O", "ALA 2 N", "ALA 2 CA", "ALA 2 CB", "ALA 2 C",
     "ALA 2 O", "NME 3 N", "NME 3 CH3"],
)  # fmt: skip
SERIES = ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCN", "CCCN", "CCCCN", "CCCCCN",
          "CC(=O)O", "CCC(=O)O", "CCCC(=O)O", "CCCCC(=O)O",
          "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1"]  # fmt: skip
SHORT = ["--padding-nm", "1.0", "--platform", "CPU", "--equilibration-ns", "0.001",
         "--equilibration-report-interval-ns", "0.001", "--production-ns", "0.002",
         "--production-report-interval-ns", "0.001", "--checkpoint-interval-ns", "0.001",
         "--performance-interval-ns", "0.001"]  # fmt: skip


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def protein(tmp_path_factory):
    path = tmp_path_factory.mktemp("in") / "protein.dms"
    boonza.save(_mapped(*DIPEPTIDE, "A"), path)
    return path


@pytest.fixture(scope="module")
def peptides(tmp_path_factory):
    """Four capped peptides in one SDF file (no residue information)."""
    lib = None
    for name, side in (("ala", "C"), ("gly", None), ("ser", "CO"), ("val", "C(C)C")):
        smi = f"CC(=O)N[C@@H]({side})C(=O)NC" if side else "CC(=O)NCC(=O)NC"
        s = boonza.from_smiles(smi, name=f"ace-{name}-nme")
        if lib is None:
            lib = s
        else:
            lib.append(s)
    path = tmp_path_factory.mktemp("in") / "peptides.sdf"
    boonza.save(lib, path)
    return path


# ---- which ligands go together ----------------------------------------------------


def test_simulation_sizes():
    assert swim.simulation_sizes(500, 5) == [5] * 100
    assert swim.simulation_sizes(501, 5) == [5] * 99 + [6]
    assert swim.simulation_sizes(502, 5) == [5] * 98 + [6, 6]
    assert swim.simulation_sizes(7, 5) == [7]
    assert swim.simulation_sizes(3, 5) == [3]
    with pytest.raises(ValueError):
        swim.simulation_sizes(10, 0)


def test_deal_spreads_similar_ligands():
    mols = [Chem.MolFromSmiles(s) for s in SERIES]
    groups = swim.deal(mols, 4)
    assert groups == swim.deal(mols, 4)  # deterministic
    series = [j // 4 for j in range(16)]  # alcohols, amines, acids, aromatics
    assert all(sorted(series[j] for j in g) == [0, 1, 2, 3] for g in groups)
    # the extra ligand of 17 goes to the last simulation
    more = swim.deal([*mols, Chem.MolFromSmiles("CCCCCCO")], 4)
    assert [len(g) for g in more] == [4, 4, 4, 5]
    # ligands with the same bond graph never share a simulation
    same = [Chem.MolFromSmiles(s) for s in ("CCO", "CCO", "CCO", "c1ccccc1", "CCN", "CCCl")]
    keys = [Chem.MolToSmiles(m) for m in same]
    for g in swim.deal(same, 2, keys=keys):
        assert len({keys[j] for j in g}) == len(g)


# ---- ligands ----------------------------------------------------------------------


def test_peptides_take_the_protein_force_field():
    ffs = [viparr.load_forcefield("aa.amber.ff14SB")]
    pep = boonza.from_smiles("CC(=O)N[C@@H](C)C(=O)NCC(=O)NC", name="PEP")
    split = swim.split_peptide(pep, ffs)
    assert split.residues["name"].tolist() == ["ACE", "ALA", "GLY", "NME"]
    assert (np.diff(split.atoms["residue"]) >= 0).all()  # residues contiguous
    assert gaff.find_unmatched(split, ffs) == []
    # a molecule with an amide that is not a peptide stays whole
    assert swim.split_peptide(boonza.from_smiles("CC(=O)NCc1ccccc1"), ffs) is None


def test_a_ligand_keeps_its_own_force_field():
    host = viparr.load_forcefield("aa.amber.ff14SB")
    ref = boonza.parameterize(_mapped(*DIPEPTIDE, "A"), [host])
    patch = viparr.patch_from_system(ref, "L000", "L000", rules=host.rules)
    one = swim._one_residue(ref.clone(structure_only=True), "L000")
    got = boonza.parameterize(one, [viparr.merge_forcefields(host, patch)])
    kinds = ("atoms", "residues", "chains", "cts", "bonds")
    assert [d for d in boonza.diff(got, ref, canonical=True) if d.kind not in kinds] == []
    assert np.allclose(got.atoms["charge"], ref.atoms["charge"])
    bad = ref.copy()
    bad.table("pair_12_6_es").params["qij"] = bad.table("pair_12_6_es").params["qij"] * 1.2
    with pytest.raises(viparr.ViparrError, match="1-4 pairs"):
        viparr.patch_from_system(bad, "L001", "L001", rules=host.rules)
    cmap = boonza.parameterize(_mapped(*DIPEPTIDE, "A"), ["aa.amber.ff19SB"])
    with pytest.raises(viparr.ViparrError, match="torsiontorsion_cmap"):
        viparr.patch_from_system(cmap, "L002", "L002")


def test_sdf_records_keep_their_own_name(tmp_path):
    lib = None
    for name, smi in (("Z359510198", "CCO"), ("Z104476730", "c1ccccc1"), ("", "CC(=O)N")):
        s = boonza.from_smiles(smi, name=name)
        if lib is None:
            lib = s
        else:
            lib.append(s)
    path = tmp_path / "frags.sdf"
    boonza.save(lib, path)
    ligands = swim.load_library(path, [viparr.load_forcefield("aa.amber.ff14SB")])
    assert [x.name for x in ligands[:2]] == ["Z359510198", "Z104476730"]
    # the record's own name becomes the residue name, whatever its length
    assert [x.system.residues["name"].tolist()[0] for x in ligands] == [
        "Z359510198", "Z104476730", "LIG",  # an unnamed record falls back to LIG
    ]  # fmt: skip
    assert swim._resname("ligand 7") == "LIG"  # the placeholder _entries makes up
    assert swim._resname("  two words ") == "two_words"


def test_bundled_fragment_libraries():
    assert set(swim.bundled_libraries()) >= {"AstexMiniFrag", "Essential320"}
    found = swim.find_library("astexminifrag")  # by name, whatever its case
    assert found.name == "AstexMiniFrag.sdf" and found.is_file()
    ligands = swim.load_library(found, [viparr.load_forcefield("aa.amber.ff14SB")])
    records = found.read_text().count("$$$$")  # every record becomes a ligand
    assert len(ligands) == records > 50
    assert all((x.system.atoms["anum"] == 1).any() for x in ligands)  # hydrogens, for GAFF2
    names = [x.system.residues["name"].tolist()[0] for x in ligands]
    assert all(n.startswith("Z") for n in names)  # Enamine's catalogue IDs, kept
    assert len(set(names)) == len(names)  # and distinct, so a selection picks one
    with pytest.raises(FileNotFoundError, match="AstexMiniFrag, Essential320"):
        swim.find_library("no-such-library")


def test_place_keeps_copies_apart(protein):
    prot = boonza.load(protein)
    ligs = [
        swim._one_residue(boonza.from_smiles(s), f"L00{k}")
        for k, s in enumerate(("CCO", "c1ccccc1"))
    ]
    edge = 30.0
    out = swim.place(prot, ligs, 3, edge, np.random.default_rng(0), clearance=3.0)
    assert out.natoms == prot.natoms + 3 * sum(lig.natoms for lig in ligs)
    heavy = out.atoms["anum"] > 1
    mol = np.asarray(out.fragids)
    pos = out.positions
    for f in np.unique(mol[prot.natoms :]):
        mine = heavy & (mol == f)
        rest = heavy & (mol != f)
        d = np.linalg.norm(pos[mine][:, None] - pos[rest][None], axis=-1)
        assert d.min() >= 3.0
    center = prot.positions.mean(0)
    assert (np.abs(pos - center) <= edge / 2 + 1e-6).all()
    again = swim.place(prot, ligs, 3, edge, np.random.default_rng(0), clearance=3.0)
    assert np.allclose(again.positions, pos)  # the same seed places them the same


# ---- simulations ------------------------------------------------------------------


def test_swim_prepares_and_runs(tmp_path, protein, peptides):
    work = tmp_path / "swim"
    args = parse_arguments([str(protein), "--workdir", str(work), *SHORT,
                            "--dihedral-restraint", "bb"])  # fmt: skip
    sims = swim.prepare(args, peptides, types=2, copies=2, log=quiet, repel=True)
    assert [d.name for d in sims] == ["sim_000", "sim_001"]
    rows = list(csv.DictReader((work / "assignment.csv").open()))
    assert sorted(r["name"] for r in rows) == ["ace-ala-nme", "ace-gly-nme", "ace-ser-nme",
                                               "ace-val-nme"]  # fmt: skip
    assert all((work / "ligands" / f"L00{k}" / "standard").is_file() for k in range(4))
    lines = (work / "simulations.txt").read_text().splitlines()
    assert len(lines) == 2 and lines[0].startswith("boonza md --config ")
    inp = boonza.load(sims[0] / "input.dms")
    assert int((inp.residues["name"] == "ACE").sum()) == 1 + 2 * 2  # protein + 2 x 2 copies
    settings = parse_arguments(["--config", str(sims[0] / "md.toml")])
    assert settings.repulsion_selection == "chain LIG"  # asked for, so on the ligands' chain
    assert settings.dihedral_restraint_selection == "not chain LIG"  # the ligands swim
    lig_rows = list(csv.DictReader((sims[0] / "ligands.csv").open()))
    assert len(lig_rows) == 4 and lig_rows[0]["first_resid"] == "1"  # 2 types x 2 copies
    assert set(inp.chains["name"].tolist()) == {"A", "LIG"}
    run_workflow(settings, log=quiet)
    state = sims[0] / "md" / "state.csv"
    assert len(state.read_text().splitlines()) == 3
    assert "LigandRepulsion" in (sims[0] / "md" / "system.xml").read_text()
    restrained = list(csv.DictReader((sims[0] / "md" / "dihedral_restraints.csv").open()))
    assert {r["chain_id"] for r in restrained} == {"A"}  # the peptide ligands are not held
    # prepared again, the started simulation is left alone
    swim.prepare(args, peptides, types=2, copies=2, log=quiet)
    assert len(state.read_text().splitlines()) == 3


def test_swim_with_a_parameterized_library(tmp_path, protein):
    host = viparr.load_forcefield("aa.amber.ff14SB")
    lig = boonza.parameterize(_mapped(*DIPEPTIDE, "B"), [host])
    lib = tmp_path / "ligands.dms"
    boonza.save(lig, lib)
    args = parse_arguments([str(protein), "--workdir", str(tmp_path / "swim"), "-f",
                            "aa.amber.ff14SB", "-f", "water.tip3p", "-f",
                            "ions.amber1jc.tip3p"])  # fmt: skip
    (sim,) = swim.prepare(args, lib, types=1, copies=2, log=quiet)
    placed = boonza.load(sim / "input.dms")
    lig = placed.residues["chain"] == int(np.flatnonzero(placed.chains["name"] == "LIG")[0])
    assert set(placed.residues["name"][lig].tolist()) == {"LIG"}  # told apart by resid
    settings = parse_arguments(["--config", str(sim / "md.toml")])
    # swim starts from boonza md's defaults: neither restraints nor repulsion
    assert settings.dihedral_restraint == "none"
    assert settings.repulsion_selection is None
    patch = tmp_path / "swim" / "ligands" / "L000" / "patch"
    assert (patch / "templates").is_file()
    from boonza.md.config import load_configuration
    from boonza.md.prepare import build_system

    settings = load_configuration(sim / "md.toml")
    assert str(patch.resolve()) in settings["forcefields"][0]
    built, info = build_system(parse_arguments(["--config", str(sim / "md.toml")]), sim,
                               log=quiet)  # fmt: skip
    assert not (sim / "gaff2_patch").exists()  # the ligand needed no GAFF2
    assert abs(built.atoms["charge"].sum()) < 1e-6


@needs_amber
def test_swim_parameterizes_small_molecules_once(tmp_path, protein):
    lib = None
    for smi in ("CCO", "Oc1ccccc1", "CC(=O)N"):
        s = boonza.from_smiles(smi)
        if lib is None:
            lib = s
        else:
            lib.append(s)
    path = tmp_path / "small.sdf"
    boonza.save(lib, path)
    assert AMBERHOME is not None
    args = parse_arguments([str(protein), "--workdir", str(tmp_path / "swim")])
    (sim,) = swim.prepare(args, path, types=3, copies=1, jobs=2, log=quiet)
    for k in range(3):
        tpl = viparr.load_forcefield(tmp_path / "swim" / "ligands" / f"L00{k}" / "patch",
                                     require_rules=False).templates  # fmt: skip
        assert all(t.name.startswith(f"L00{k}_") for t in tpl)
        assert all(b.endswith(f"~L00{k}") for t in tpl for b in t.btype if b)
    from boonza.md.prepare import load_input

    inp = load_input(sim / "input.dms")
    ffs = swim._load_forcefields(parse_arguments(["--config", str(sim / "md.toml")]).forcefields)
    assert gaff.find_unmatched(inp, ffs) == []  # the simulation needs no GAFF2 of its own
