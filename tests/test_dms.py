import json

import numpy as np
import pytest
from canonical import assert_same, canon
from conftest import msys_file, run_msys

import boonza

FILES = [
    "2f4k.dms",
    "3.dms",
    "cdk2-ligand-Amber14EHT.dms",
    "ch4.dms",
    "cubane.dms",
    "desresff420.dms",
    "memo.dms",
    "methane-pdff.dms",
    "pro.dms",
    "pseudo.dms",
    "spatial_hash_test.dms",
    "stable-hash-6896670165215326854.dms",
    "ww.dms",
]


@pytest.mark.parametrize("name", FILES)
def test_load_matches_msys(name):
    path = msys_file(name)
    assert_same(canon(boonza.load(path)), run_msys("load", path))


@pytest.mark.parametrize("name", FILES)
def test_saved_file_matches_msys(name, tmp_path):
    path = msys_file(name)
    s = boonza.load(path)
    out = tmp_path / "out.dms"
    boonza.save(s, out)
    assert_same(run_msys("load", out), run_msys("load", path))
    assert_same(canon(boonza.load(out)), canon(s))


def test_alchemical_not_supported_yet():
    with pytest.raises(NotImplementedError):
        boonza.load(msys_file("excluded_knot.dms"))


@pytest.mark.parametrize("suffix", [".dms.gz", ".dms.bz2"])
def test_compressed_roundtrip(tmp_path, suffix):
    s = boonza.load(msys_file("ww.dms"))
    out = tmp_path / f"ww{suffix}"
    boonza.save(s, out)
    assert_same(canon(boonza.load(out)), canon(s))


def test_clone_matches_msys(tmp_path):
    path = msys_file("3.dms")
    s = boonza.load(path)
    ids = np.random.default_rng(7).permutation(s.natoms)[: s.natoms // 3]
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps(ids.tolist()))
    assert_same(canon(s.clone(ids)), run_msys("clone", path, ids_file))


def test_append_matches_msys():
    a_path, b_path = msys_file("ww.dms"), msys_file("desresff420.dms")
    a = boonza.load(a_path)
    a.append(boonza.load(b_path))
    assert_same(canon(a), run_msys("append", a_path, b_path))


def test_without_tables_and_structure_only():
    path = msys_file("methane-pdff.dms")
    full = boonza.load(path)
    bare = boonza.load(path, without_tables=True)
    assert bare.natoms == full.natoms and not bare.tables
    heavy = boonza.load(path, structure_only=True)
    assert heavy.natoms == int((full.atoms["anum"] > 0).sum())
