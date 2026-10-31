"""boonza fep grow: the steps a ligand is grown from a reference in, and a growth."""

import json

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from test_fep_pipeline import _amber_tools

import boonza
from boonza.fep import grow


def _mol(smiles, name="x", seed=7):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    m.SetProp("_Name", name)
    return m


def _symbols(m, step):
    return "".join(sorted(m.GetAtomWithIdx(i).GetSymbol() for i in step))


def test_groups_and_rings_are_grown_whole():
    ref = _mol("c1ccccc1")
    for smiles, whole in [("OC(=O)Cc1ccccc1", None),  # a carboxylic acid: C, O, O together
                          ("[O-]C(=O)c1ccccc1", "COO"),
                          ("NC(=[NH2+])NCc1ccccc1", None),  # guanidinium: C and its 3 N
                          ("c1ccccc1-c1ncccn1", "CCCCNN")]:  # a pyrimidine, whole  # fmt: skip
        target = _mol(smiles)
        core = grow.core_map(ref, target)
        plan = grow.steps(target, core)
        rest = [a.GetIdx() for a in target.GetAtoms()
                if a.GetAtomicNum() > 1 and a.GetIdx() not in core]  # fmt: skip
        assert sorted(i for s in plan for i in s) == sorted(rest)
        if whole:
            assert any(_symbols(target, s) == whole for s in plan), (smiles, plan)
        if "C(=O)" in smiles or "(=O)" in smiles:  # the C=O never apart from its O-/OH
            for s in plan:
                syms = _symbols(target, s)
                assert not (syms.count("O") == 1 and "C" in syms and len(s) == 2)
        if "NH2+" in smiles:
            guan = next(s for s in plan if _symbols(target, s).count("N") >= 3)
            assert len(guan) >= 4
    # a chain: at most 3 heavy atoms a step
    target = _mol("CCCCCCc1ccccc1")
    plan = grow.steps(target, grow.core_map(ref, target))
    assert [len(s) for s in plan] == [3, 3]


def test_every_intermediate_is_a_molecule():
    ref = _mol("c1ccccc1")
    target = _mol("[O-]C(=O)CCc1ccc(-c2ncccn2)cc1")
    core = grow.core_map(ref, target)
    placed = set(core)
    for s in grow.steps(target, core):
        placed |= set(s)
        mol, index = grow.intermediate(target, placed)
        assert Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is not None
        assert all(i in index for i in placed)
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == Chem.MolToSmiles(Chem.RemoveHs(target))


@pytest.mark.skipif(not _amber_tools(), reason="AmberTools (GAFF2) not installed")
def test_growing_toluene_from_benzene_in_a_peptide(tmp_path):
    from boonza.fep.cli import FEP_DEFAULTS
    from boonza.md.config import build_parser, parse_arguments
    from boonza.md.prepare import forcefields
    from boonza.viewer.edit import protonate

    ref = _mol("c1ccccc1", "benzene")
    target = _mol("Cc1ccccc1", "toluene")
    s = boonza.peptide("GAG")
    s = s.clone(np.flatnonzero(s.atoms["anum"] > 1))
    protonate(s)
    pos = np.asarray(s.positions)
    s.positions = pos - pos.mean(0) + ref.GetConformer().GetPositions().mean(0) + [7.0, 0, 0]
    args = parse_arguments(["--workdir", str(tmp_path), "--platform", "CPU"],
                           build_parser("boonza fep grow"))  # fmt: skip
    for k, v in FEP_DEFAULTS.items():
        setattr(args, k, v)
    _, ff = forcefields(args)
    path = grow.grow(s, ref, target, "toluene", ff, args, tmp_path, beam=1, keep=1,
                     n_conformers=5, md_ps=2.0, log=lambda *_: None)  # fmt: skip
    out = boonza.load(path)
    assert out.natoms == s.natoms + target.GetNumAtoms()
    record = json.loads((tmp_path / "grow" / "toluene.json").read_text())
    assert record["core"] == 6 and record["steps"] and "score" in record["steps"][0]
    # the core stayed where the reference was
    lig = np.asarray(out.positions)[-target.GetNumAtoms():]
    core = grow.core_map(ref, target)
    ref_xyz = ref.GetConformer().GetPositions()
    d = [np.linalg.norm(lig[t] - ref_xyz[r]) for t, r in core.items()]
    assert max(d) < 1.0
