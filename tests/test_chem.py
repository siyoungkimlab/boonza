import functools
import json

import numpy as np
import pytest
from conftest import msys_file, run_msys

import boonza
from boonza.chem import assign_bond_orders, from_rdkit, to_rdkit

Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import AllChem  # noqa: E402

SDFS = ["lig.sdf", "methotrexate.sdf", "34106.sdf", "jandor.sdf", "stereo.sdf", "cofactors.sdf"]


def _smiles(mol):
    return Chem.MolToSmiles(mol, isomericSmiles=False)


@functools.cache
def _peptide():
    mol = Chem.AddHs(Chem.MolFromSequence("ACDEFHKWY"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    return mol


@pytest.mark.parametrize("name", SDFS)
def test_smiles_match_rdkit_reader(name):
    path = msys_file(name)
    s = boonza.load(path)
    refs = [
        None if m is None else _smiles(m) for m in Chem.SDMolSupplier(str(path), removeHs=False)
    ]
    checked = 0
    for ct, ref in enumerate(refs):
        if ref is not None:
            assert _smiles(to_rdkit(s, s.ct_atoms(ct))) == ref
            checked += 1
    assert checked


@pytest.mark.parametrize("name", ["lig.sdf", "methotrexate.sdf", "34106.sdf"])
def test_roundtrip(name):
    s = boonza.load(msys_file(name))
    mol = s.to_rdkit()
    back = from_rdkit(mol)
    assert back.atoms["anum"].tolist() == s.atoms["anum"].tolist()
    assert back.atoms["formal_charge"].tolist() == s.atoms["formal_charge"].tolist()
    np.testing.assert_allclose(back.positions, s.positions)
    assert Chem.MolToSmiles(back.to_rdkit()) == Chem.MolToSmiles(mol)
    assert back.ct(0).keys() == s.ct(0).keys()
    for key in s.ct(0).keys():
        assert (
            back.ct(0)[key] == pytest.approx(s.ct(0)[key])
            if isinstance(s.ct(0)[key], float)
            else back.ct(0)[key] == s.ct(0)[key]
        )


def test_heavy_atom_only_molecule_gets_implicit_hydrogens():
    s = boonza.load(msys_file("cofactors.sdf"))
    for ct in range(s.ncts):
        ids = s.ct_atoms(ct)
        if not np.any(s.atoms["anum"][ids] == 1):
            assert "[C]" not in Chem.MolToSmiles(to_rdkit(s, ids))
            return
    pytest.skip("no heavy-atom-only entry")


def test_peptide_roundtrip_and_residues():
    pep = _peptide()
    s = from_rdkit(pep)
    assert s.nresidues == 9
    assert s.residues["name"].tolist()[:3] == ["ALA", "CYS", "ASP"]
    assert len(s.select("protein and name CA")) == 9
    mol = s.atom(0).fragment.to_rdkit()
    assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(pep)
    assert [a.GetIntProp("boonza_index") for a in mol.GetAtoms()] == list(range(s.natoms))
    assert "ATOM" in Chem.MolToPDBBlock(mol)


def test_unsanitizable_gives_hint():
    s = boonza.load(msys_file("2f4k.dms"))  # no bond orders or formal charges
    with pytest.raises(ValueError, match="assign_bond_orders"):
        s.atom(0).fragment.to_rdkit()
    water = s.select("water").ids[0]
    assert Chem.MolToSmiles(s.atom(water).fragment.to_rdkit()) == "[H]O[H]"


def test_from_rdkit_hydrogens():
    with pytest.raises(ValueError):
        from_rdkit(Chem.MolFromSmiles("CCO"))
    s = from_rdkit(Chem.MolFromSmiles("CCO"), add_hydrogens=True)
    assert s.natoms == 9 and s.nbonds == 8


@pytest.mark.parametrize("name", ["methotrexate.sdf", "34106.sdf", "lig.sdf"])
def test_assign_bond_orders(name):
    s = boonza.load(msys_file(name))
    ref = _smiles(s.to_rdkit())
    total = int(s.atoms["formal_charge"].sum())
    s.bonds["order"] = np.ones(s.nbonds, np.int64)
    s.atoms["formal_charge"] = np.zeros(s.natoms, np.int64)
    assign_bond_orders(s, charge=total)
    assert _smiles(s.to_rdkit()) == ref


def test_smarts_selections_match_msys(tmp_path):
    path = tmp_path / "pep.dms"
    boonza.save(from_rdkit(_peptide()), path)
    s = boonza.load(path)
    sels = ["smarts 'c1ccccc1'", "smarts '[#8]'", "smarts 'C(=O)N'", "protein and smarts 'c'",
            "smarts '[NX3;H2]' or smarts '[SX2H]'", "smarts 'C(=O)[OH]'"]  # fmt: skip
    f = tmp_path / "sels.json"
    f.write_text(json.dumps(sels))
    for sel, want in zip(sels, run_msys("select", path, f), strict=True):
        assert s.select(sel).ids.tolist() == want, sel
