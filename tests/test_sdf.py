import numpy as np
import pytest
from canonical import assert_same, canon
from conftest import msys_file, run_msys

import boonza

FILES = [
    "34106.sdf",
    "cofactors.sdf",
    "extra_line_m_end.sdf",
    "fused.sdf",
    "isotope.sdf",
    "jandor-bad.sdf",
    "jandor.sdf",
    "lig.sdf",
    "methotrexate.sdf",
    "multiline.sdf",
    "no-delim.sdf",
    "no-delim2.sdf",
    "stereo.sdf",
]


def _canon(s):
    c = canon(s)
    c["atom_props"].pop("stereo_parity", None)  # msys keeps stereo outside its properties
    return c


@pytest.mark.parametrize("name", FILES)
def test_load_matches_msys(name):
    path = msys_file(name)
    assert_same(_canon(boonza.load(path)), run_msys("load", path))


@pytest.mark.parametrize("name", FILES)
def test_write_roundtrip(name, tmp_path):
    src = boonza.load(msys_file(name))
    out = tmp_path / "out.sdf"
    boonza.save(src, out)
    back = boonza.load(out)
    assert_same(_canon(back), run_msys("load", out))  # msys reads our file the same way
    ours, ref = canon(back), canon(src)
    for a, b in zip(ours.pop("cts"), ref.pop("cts"), strict=True):
        assert a[0] == b[0] and a[1].keys() == b[1].keys()
        for key, value in b[1].items():
            assert (
                a[1][key] == pytest.approx(value, rel=1e-15)
                if isinstance(value, float)
                else (a[1][key] == value)
            )
    assert_same(ours, ref)
    for prop in ("stereo", "aromatic"):
        if prop in src.bonds:
            assert back.bonds[prop].tolist() == src.bonds[prop].tolist()


@pytest.mark.parametrize("name", FILES)
def test_v3000_roundtrip(name, tmp_path):
    src = boonza.load(msys_file(name))
    out = tmp_path / "v3.sdf"
    boonza.save(src, out, v3000=True)
    assert "V3000" in out.read_text()
    ours, ref = canon(boonza.load(out)), canon(src)
    # V2000 coordinates go through msys' float32 parse; V3000 ones are read exactly
    for a, b in zip(ours.pop("atoms"), ref.pop("atoms"), strict=True):
        np.testing.assert_allclose(a[5:8], b[5:8], atol=1e-6)
        assert a[:5] + a[8:] == b[:5] + b[8:]
    assert_same(ours, ref)


@pytest.mark.parametrize("name", ["methotrexate.sdf", "lig.sdf", "cofactors.sdf", "isotope.sdf"])
def test_reads_rdkit_v3000(name, tmp_path):
    Chem = pytest.importorskip("rdkit.Chem")
    path = msys_file(name)
    out = tmp_path / "rdkit.sdf"
    with Chem.SDWriter(str(out)) as w:
        w.SetForceV3000(True)
        for mol in Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False):
            w.write(mol)
    ref, got = boonza.load(path), boonza.load(out)
    assert got.atoms["anum"].tolist() == ref.atoms["anum"].tolist()
    assert got.atoms["formal_charge"].tolist() == ref.atoms["formal_charge"].tolist()
    np.testing.assert_allclose(got.positions, ref.positions, atol=1e-4)
    assert canon(got)["bonds"] == canon(ref)["bonds"]
    if "isotope" in ref.atoms:
        assert got.atoms["isotope"].tolist() == ref.atoms["isotope"].tolist()


def test_large_system_writes_v3000(tmp_path):
    src = boonza.load(msys_file("pro.dms"))  # more than 999 atoms
    out = tmp_path / "big.sdf"
    boonza.save(src, out)
    back = boonza.load(out)
    assert back.natoms == src.natoms and back.nbonds == src.nbonds
    np.testing.assert_allclose(back.positions, src.positions, atol=5e-5)
    assert back.atoms["anum"].tolist() == src.atoms["anum"].tolist()


def test_charges_and_fields(tmp_path):
    s = boonza.load(msys_file("no-delim.sdf"))
    assert s.atoms["formal_charge"][[29, 30]].tolist() == [1, -1]
    assert np.count_nonzero(s.atoms["formal_charge"]) == 2
    lig = boonza.load(msys_file("lig.sdf"))
    props = lig.ct(0).keys()
    assert "i_epik_Tot_Q" in props and isinstance(lig.ct(0)["i_epik_Tot_Q"], int)
