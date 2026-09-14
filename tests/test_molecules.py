import time

import pytest
from conftest import msys_file, run_msys

import boonza
from boonza.molecules import distinct_fragments

FILES = ["ww.dms", "2f4k.dms", "cofactors.sdf", "jandor.sdf", "knot.mae", "3RYZ.pdb", "pro.dms"]


@pytest.mark.parametrize("name", FILES)
def test_matches_msys(name):
    path = msys_file(name)
    as_msys = {"conect": False, "ssbond": False, "link": False}  # msys ignores bond records
    kw = as_msys if name.endswith(".pdb") else {}
    ours = distinct_fragments(boonza.load(path, **kw))
    assert [[k, v] for k, v in ours.items()] == run_msys("distinct", path)


def test_isomers_split_and_stereoisomers_merge(tmp_path):
    Chem = pytest.importorskip("rdkit.Chem")
    s = boonza.System()
    for smiles in ("CCCO", "CC(C)O", "CCCO", "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "CC(C)O"):
        s.append(boonza.from_rdkit(Chem.MolFromSmiles(smiles), add_hydrogens=True))
    groups = distinct_fragments(s)
    assert list(groups.values()) == [[0, 2], [1, 5], [3, 4]]
    path = tmp_path / "isomers.dms"
    boonza.save(s, path)
    assert [[k, v] for k, v in distinct_fragments(boonza.load(path)).items()] == run_msys(
        "distinct", path
    )


def test_many_waters_are_fast():
    s = boonza.load(msys_file("ww.dms"))
    t0 = time.perf_counter()
    groups = distinct_fragments(s)
    assert time.perf_counter() - t0 < 5.0
    nwater = len(set(s.fragids[s.select("water").ids].tolist()))
    biggest = max(groups.values(), key=len)
    assert len(biggest) == nwater > 100
    assert s.distinct_fragments() == groups
