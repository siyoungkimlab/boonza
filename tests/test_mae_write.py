import warnings
from pathlib import Path

import numpy as np
import pytest
from canonical import assert_same, assert_same_content, canon
from conftest import msys_file, run_msys

import boonza

HERE = Path(__file__).parent

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


def test_atom_properties_survive_mae(tmp_path):
    """MAE used to drop every atom property: a structure written and read back
    came home with no B-factors, no occupancies and none of the properties put
    on it by hand."""
    src = boonza.load(HERE / "data" / "1LYZ.pdb")
    assert {"bfactor", "occupancy"} <= set(src.atoms.props)  # the PDB reader's
    out = tmp_path / "out.mae"
    boonza.save(src, out)
    back = boonza.load(out)
    assert set(src.atoms.props) <= set(back.atoms.props)
    for prop in ("bfactor", "occupancy"):
        np.testing.assert_allclose(back.atoms[prop], src.atoms[prop], rtol=1e-6)
    # under the names Maestro keeps for them, so Maestro shows them as its own
    text = out.read_text(errors="replace")
    assert "r_m_pdb_tfactor" in text and "r_m_pdb_occupancy" in text


def test_properties_of_ones_own_survive_mae(tmp_path):
    """A property of any scalar type travels as <kind>_user_<name>.  nan has to
    come back as nan: boonza's selection grammar resolves an atom property as a
    keyword, and a missing value read as 0 would quietly pass `drifted < 1.5`."""
    s = boonza.load(HERE / "data" / "1LYZ.pdb")
    drifted = np.arange(s.natoms, dtype=float) / 10
    drifted[:5] = np.nan
    s.atoms["drifted"] = drifted
    s.atoms["nposes"] = np.arange(s.natoms) % 7
    s.atoms["tag"] = np.array(["aa", "bb"] * (s.natoms // 2) + ["cc"] * (s.natoms % 2))
    out = tmp_path / "out.mae"
    boonza.save(s, out)
    back = boonza.load(out)

    np.testing.assert_allclose(back.atoms["drifted"], drifted, rtol=1e-6, equal_nan=True)
    assert np.isnan(np.asarray(back.atoms["drifted"], float)).sum() == 5
    np.testing.assert_array_equal(back.atoms["nposes"], s.atoms["nposes"])
    np.testing.assert_array_equal(back.atoms["tag"], s.atoms["tag"])

    # and they select, with the unmeasured atoms left out rather than counted
    below = int((drifted < 1.5).sum())
    assert len(back.select("drifted < 1.5")) == below
    assert len(back.select("nposes == 3")) == int((s.atoms["nposes"] == 3).sum())


def test_atom_properties_in_every_ct(tmp_path):
    """MAE writes one f_m_ct block per ct and the reader walks them in turn, so
    a property has to be read for each.  Reading it once leaves every ct after
    the first on the column default, which reads as a uniform value rather than
    as an error."""
    s = boonza.load(HERE / "data" / "1LYZ.pdb")
    s.append(s.clone("name CA"))
    s.append(s.clone("backbone"))
    assert s.ncts > 1
    marker = np.arange(s.natoms, dtype=float)
    marker[:3] = np.nan
    s.atoms["marker"] = marker
    out = tmp_path / "out.mae"
    boonza.save(s, out)
    back = boonza.load(out)

    assert back.ncts == s.ncts
    np.testing.assert_allclose(back.atoms["marker"], marker, rtol=1e-6, equal_nan=True)
