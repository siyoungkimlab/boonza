"""3D atom mappings between posed ligands, and their scores."""

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from boonza.fep import mapping

DATA = Path(__file__).parent / "data" / "fep"


@pytest.fixture(scope="module")
def benzenes():
    ms = Chem.SDMolSupplier(str(DATA / "benzene_modifications.sdf"), removeHs=False)
    return {m.GetProp("_Name"): m for m in ms}


def _elements(mol, atoms):
    return sorted(mol.GetAtomWithIdx(i).GetSymbol() for i in atoms)


def test_benzene_onto_toluene(benzenes):
    a, b = benzenes["benzene"], benzenes["toluene"]
    m = mapping.map_atoms(a, b)
    # the ring and five hydrogens; the sixth sits where the methyl carbon is
    assert _elements(a, m) == ["C"] * 6 + ["H"] * 5
    assert all(a.GetAtomWithIdx(i).GetSymbol() == b.GetAtomWithIdx(j).GetSymbol()
               for i, j in m.items())  # fmt: skip
    assert mapping.score(a, b, m) == pytest.approx(np.exp(-0.1))
    assert mapping.score(a, a, mapping.map_atoms(a, a)) == 1.0
    assert len(mapping.map_atoms(a, a)) == a.GetNumAtoms()


def test_bigger_changes_score_lower(benzenes):
    a = benzenes["benzene"]
    s = {n: mapping.score(a, benzenes[n], mapping.map_atoms(a, benzenes[n]))
         for n in ("toluene", "phenol", "anisole", "styrene")}  # fmt: skip
    assert s["toluene"] == s["phenol"] > s["anisole"] == s["styrene"]


def _posed(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    return m


def test_a_ring_maps_whole_or_not_at_all():
    """Cyclohexane onto cyclopentane, superposed: no ring atom may map."""
    a, b = _posed("C1CCCCC1"), _posed("C1CCCC1")
    from rdkit.Chem import rdMolAlign

    rdMolAlign.AlignMol(b, a, atomMap=[(0, 0), (1, 1), (2, 2)])
    m = mapping.map_atoms(a, b, max_distance=1.5)
    ring_a = set(a.GetRingInfo().AtomRings()[0])
    assert not set(m) & ring_a


def test_no_bond_breaks_and_hydrogens_follow_their_atoms(benzenes):
    a, b = benzenes["benzene"], benzenes["styrene"]
    m = mapping.map_atoms(a, b, max_distance=2.0)  # loose: tempt wrong pairs
    for bond in a.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in m and j in m:
            assert b.GetBondBetweenAtoms(m[i], m[j]) is not None
    for i, j in m.items():
        if a.GetAtomWithIdx(i).GetAtomicNum() == 1:
            hi = a.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx()
            assert m[hi] == b.GetAtomWithIdx(j).GetNeighbors()[0].GetIdx()


def test_a_centre_that_would_invert_is_left_out():
    """Neighbours of a tetrahedral centre placed as its mirror image in B."""
    a = _posed("C(F)(Cl)(Br)[H]")
    b = Chem.Mol(a)
    conf = b.GetConformer()
    xyz = conf.GetPositions()
    xyz[:, 0] *= -1  # the mirror image
    for k, p in enumerate(xyz):
        conf.SetAtomPosition(k, p.tolist())
    graph_a, graph_b = mapping._graph(a), mapping._graph(b)
    identity = {k: k for k in range(a.GetNumAtoms())}
    kept = mapping._chirality(identity, graph_a, graph_b, a.GetConformer().GetPositions(), xyz)
    assert 0 not in kept and len(kept) == a.GetNumAtoms() - 1


def test_a_change_of_charge_is_not_offered():
    a, b = _posed("CC(=O)O"), _posed("CC(=O)[O-]")
    assert mapping.score(a, b, {0: 0, 1: 1}) == 0.0
    # unless asked for, at a penalty
    m = {0: 0, 1: 1, 2: 2, 3: 3}
    assert mapping.score(a, b, m, charge_change=0.1) == pytest.approx(0.1)
