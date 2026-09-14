import functools
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from canonical import assert_same, canon

import boonza
from boonza.io.cif import parse_cif
from boonza.io.pdb import lengths_angles_from_cell

HERE = Path(__file__).parent
GEMMI_PYTHON = os.path.expanduser(
    os.environ.get("BOONZA_GEMMI_PYTHON", "~/miniforge3/envs/ommflow/bin/python")
)
HOME = Path("~").expanduser()
FILES = [
    "mdanalysis/testsuite/MDAnalysisTests/data/4x8u.pdbx",
    "runs-n-poses/examples/ground_truth/8c3u__1__1.A__1.C/system.cif",
    "runs-n-poses/examples/ground_truth/8c3u__1__1.A__1.C/receptor.cif",
    "runs-n-poses/examples/outputs/af3/8c3u__1__1.a__1.c/8c3u__1__1.a__1.c_model.cif",
    "runs-n-poses/examples/outputs/protenix/8c3u__1__1.A__1.C/seed_2242028199/predictions/"
    "8c3u__1__1.A__1.C_seed_2242028199_sample_0.cif",
]
NO_CELL = [1.0, 1.0, 1.0, 90.0, 90.0, 90.0]  # gemmi's placeholder when there is no _cell


@functools.cache
def _gemmi_available() -> bool:
    try:
        r = subprocess.run([GEMMI_PYTHON, "-c", "import gemmi"], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def run_gemmi(path):
    if not _gemmi_available():
        pytest.skip("gemmi not available; set BOONZA_GEMMI_PYTHON")
    r = subprocess.run([GEMMI_PYTHON, str(HERE / "gemmi_oracle.py"), str(path)],
                       capture_output=True, text=True)  # fmt: skip
    if r.returncode:
        raise RuntimeError(f"gemmi oracle failed:\n{r.stderr}")
    return json.loads(r.stdout)


def _file(rel):
    path = HOME / rel
    if not path.exists():
        pytest.skip(f"{rel} not found")
    return path


def assert_matches_gemmi(s, ref):
    atoms = ref["atoms"]
    assert s.natoms == len(atoms)
    res = s.atoms["residue"]
    chn = s.residues["chain"][res]
    col = list(zip(*atoms, strict=True)) if atoms else [[]] * 16
    assert s.chains["ct"][chn].tolist() == list(col[0])
    assert s.chains["name"][chn].tolist() == list(col[2])
    assert s.chains["segid"][chn].tolist() == list(col[3])
    assert s.residues["name"][res].tolist() == list(col[4])
    assert s.residues["resid"][res].tolist() == list(col[5])
    assert s.residues["insertion"][res].tolist() == list(col[6])
    assert s.atoms["name"].tolist() == list(col[7])
    assert s.atoms["anum"].tolist() == list(col[8])
    altloc = s.atoms["altloc"].tolist() if "altloc" in s.atoms else [""] * s.natoms
    assert altloc == list(col[9])
    np.testing.assert_allclose(s.positions, np.column_stack(col[10:13]), atol=1e-9)
    np.testing.assert_allclose(s.atoms["occupancy"], col[13], rtol=1e-6)
    np.testing.assert_allclose(s.atoms["bfactor"], col[14], rtol=1e-6)
    assert s.atoms["formal_charge"].tolist() == list(col[15])
    if s.cell.any():
        np.testing.assert_allclose(lengths_angles_from_cell(s.cell), ref["cell"], atol=1e-6)
    else:
        assert ref["cell"] == NO_CELL


@pytest.mark.parametrize("rel", FILES, ids=[Path(f).name for f in FILES])
def test_matches_gemmi(rel):
    path = _file(rel)
    assert_matches_gemmi(boonza.load(path, format="cif"), run_gemmi(path))


@pytest.mark.parametrize("rel", FILES, ids=[Path(f).name for f in FILES])
def test_write_roundtrip(rel, tmp_path):
    src = boonza.load(_file(rel), format="cif")
    out = tmp_path / "out.cif"
    boonza.save(src, out)
    back = boonza.load(out)
    assert_matches_gemmi(back, run_gemmi(out))  # gemmi reads our file the same way
    ours, ref = canon(back), canon(src)
    np.testing.assert_allclose(ours.pop("cell"), ref.pop("cell"), atol=1e-3)
    for c in ours["cts"] + ref["cts"]:
        c[0] = ""  # the data block name is not kept per model
    assert_same(ours, ref)


def test_tokenizer():
    text = """data_test
_cell.length_a 10.5(2)
_struct.title
;A multi-line
title
;
_quoted 'it''s "fine"'
loop_
_atom_site.id
_atom_site.label_atom_id
_atom_site.auth_comp_id
_atom_site.note
1 "O5'" DA 'a b' # trailing comment
2 C#1 ? .
#
"""
    (blk,) = parse_cif(text)
    assert blk.name == "test"
    assert blk.value("_cell.length_a") == "10.5(2)"
    assert blk.value("_struct.title") == "A multi-line\ntitle"
    assert blk.value("_quoted") == "it''s \"fine\""
    site = blk.category("_atom_site")
    assert site["label_atom_id"].tolist() == ["O5'", "C#1"]
    assert site["note"].tolist() == ["a b", "."]
    assert site["auth_comp_id"].tolist() == ["DA", "?"]
