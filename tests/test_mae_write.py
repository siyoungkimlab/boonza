import warnings

import numpy as np
import pytest
from canonical import assert_same, assert_same_content, canon
from conftest import msys_file, run_msys

import boonza

# (source file, tables whose content should survive a trip through MAE)
CASES = [
    ("ww.dms", None),
    ("cdk2-ligand-Amber14EHT.dms", ["stretch_harm", "angle_harm", "dihedral_trig",
                                    "pair_12_6_es", "exclusion", "nonbonded"]),
    ("desresff420.dms", None),
    ("3.dms", None),
    ("small.mae", None),
    ("tip5p.mae", None),
    ("noe.mae", ["stretch_harm", "angle_harm", "stretch_morse", "constraint_hoh", "exclusion"]),
]  # fmt: skip


def _pseudos_last(s):
    """MAE lists real atoms first, then pseudo particles: new index of each old atom."""
    order = np.argsort(s.atoms["anum"] == 0, kind="stable")
    new = np.empty(s.natoms, np.int64)
    new[order] = np.arange(s.natoms)
    return new


@pytest.mark.parametrize(("name", "tables"), CASES)
def test_mae_roundtrip(name, tables, tmp_path):
    src = boonza.load(msys_file(name))
    out = tmp_path / "out.mae"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boonza.save(src, out)
    back = boonza.load(out)
    assert_same(canon(back), run_msys("load", out))  # msys reads our file identically
    if tables is None:
        tables = [t for t in src.tables if t != "nonbonded"]
    nb = "nonbonded" in tables
    assert_same_content(src, back, _pseudos_last(src), [t for t in tables if t != "nonbonded"])
    if nb or "nonbonded" in src.tables:
        m = _pseudos_last(src)
        for p in ("sigma", "epsilon"):
            got = _per_atom(back, p)[m]
            np.testing.assert_array_equal(got, _per_atom(src, p), err_msg=p)


def _per_atom(s, prop):
    t = s.table("nonbonded")
    out = np.zeros(s.natoms)
    out[t.atoms[:, 0]] = t.values(prop)
    return out


def test_mae_gz_and_structure_only(tmp_path):
    src = boonza.load(msys_file("ww.dms"))
    out = tmp_path / "ww.mae.gz"
    boonza.save(src, out, structure_only=True)
    back = boonza.load(out)
    assert not back.tables
    np.testing.assert_array_equal(back.positions, src.positions)
