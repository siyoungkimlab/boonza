import pytest
from canonical import as_msys_reads_mae, assert_same, canon
from conftest import msys_file, run_msys

import boonza

FILES = [
    "1vcc.mae",
    "boron.mae",
    "colzuy.mae.gz",
    "fbhw.mae",
    "kanzoo.mae.gz",
    "knot.mae",
    "m2io_at_end.mae",
    "noFused1.mae",
    "noFused2.mae",
    "noe.mae",
    "small.mae",
    "struct.mae",
    "syntax.mae",
    "t2-original.cms.gz",
    "tip5p.mae",
    "two.mae",
]


@pytest.mark.parametrize("name", FILES)
def test_load_matches_msys(name):
    path = msys_file(name)
    ref = run_msys("load", path)
    assert_same(*as_msys_reads_mae(canon(boonza.load(path)), ref))


@pytest.mark.parametrize("name", ["alchemical_restraint.mae", "morse.mae"])
def test_alchemical_not_supported_yet(name):
    with pytest.raises(NotImplementedError):
        boonza.load(msys_file(name))


def test_structure_only():
    path = msys_file("tip5p.mae")
    full = boonza.load(path)
    bare = boonza.load(path, structure_only=True)
    assert not bare.tables
    assert bare.natoms == int((full.atoms["anum"] > 0).sum())
