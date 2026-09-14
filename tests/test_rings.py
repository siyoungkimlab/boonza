import pytest
from conftest import msys_file, run_msys

import boonza
from boonza.rings import ring_systems, sssr

FILES = ["jandor.sdf", "cofactors.sdf", "methotrexate.sdf", "34106.sdf", "knot.mae",
         "2f4k.dms", "ww.dms"]  # fmt: skip


@pytest.mark.parametrize("relevant", [False, True])
@pytest.mark.parametrize("name", FILES)
def test_sssr_matches_msys(name, relevant):
    path = msys_file(name)
    want = run_msys("sssr", path, "all" if relevant else "sssr")
    assert sssr(boonza.load(path), all_relevant=relevant) == want


def test_cubane_fused_rings_and_ring_systems(tmp_path):
    Chem = pytest.importorskip("rdkit.Chem")
    cubane = boonza.from_rdkit(Chem.MolFromSmiles("C12C3C4C1C5C2C3C45"), add_hydrogens=True)
    assert sorted(map(len, sssr(cubane))) == [4] * 5
    assert sorted(map(len, sssr(cubane, all_relevant=True))) == [4] * 6
    path = tmp_path / "cubane.dms"
    boonza.save(cubane, path)
    for relevant in (False, True):
        want = run_msys("sssr", path, "all" if relevant else "sssr")
        assert sssr(boonza.load(path), all_relevant=relevant) == want

    mol = boonza.from_rdkit(Chem.MolFromSmiles("c1ccc2ccccc2c1C1CC1"), add_hydrogens=True)
    rings = sssr(mol)
    assert sorted(map(len, rings)) == [3, 6, 6]
    assert sorted(map(len, ring_systems(mol, rings))) == [1, 2]
    carbons = mol.select("element C").ids
    assert sssr(mol, "element H") == []
    assert sorted(map(len, sssr(mol, carbons))) == [3, 6, 6]
