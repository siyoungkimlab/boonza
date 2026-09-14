import functools
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.ce import ce_align

HERE = Path(__file__).parent
DATA = HERE / "data"
BIO_PYTHON = os.path.expanduser(
    os.environ.get("BOONZA_BIOPYTHON", "~/miniforge3/envs/ommflow/bin/python")
)
PAIRS = [("1LYZ.pdb", "", "1LZ1.pdb", ""), ("1MBN.pdb", "", "1HHO.pdb", "A"),
         ("1TEN.pdb", "", "1FNA.pdb", ""), ("1HHO.pdb", "A", "1HHO.pdb", "B")]  # fmt: skip


@functools.cache
def _bio_available() -> bool:
    try:
        r = subprocess.run([BIO_PYTHON, "-c", "import Bio.PDB.cealign"], capture_output=True,
                           timeout=120)  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def _ca(name, chain):
    s = boonza.load(DATA / name)
    if s.ncts > 1:
        s = s.clone(s.ct_atoms(0))
    sel = "protein and name CA" + (f" and chain {chain}" if chain else "")
    return s.positions[s.select(sel).ids]


@pytest.mark.parametrize("ref,rc,mob,mc", PAIRS, ids=lambda x: str(x).replace(".pdb", ""))
def test_matches_biopython(ref, rc, mob, mc, tmp_path):
    if not _bio_available():
        pytest.skip("Biopython not available; set BOONZA_BIOPYTHON")
    A, B = _ca(ref, rc), _ca(mob, mc)
    np.save(tmp_path / "a.npy", A)
    np.save(tmp_path / "b.npy", B)
    r = subprocess.run([BIO_PYTHON, str(HERE / "ce_oracle.py"), str(tmp_path / "a.npy"),
                        str(tmp_path / "b.npy")], capture_output=True, text=True)  # fmt: skip
    assert r.returncode == 0, r.stderr
    want = json.loads(r.stdout)
    ia, ib, rms, R, t, z = ce_align(A, B)
    assert rms == pytest.approx(want["rms"], rel=1e-6)
    # Biopython applies coords @ rot + tran
    np.testing.assert_allclose(B @ R.T + t, B @ np.array(want["rot"]) + want["tran"], atol=1e-4)
    assert len(ia) == len(ib) and len(set(ia)) == len(ia)


def test_cealign_systems():
    ref = boonza.load(DATA / "1TEN.pdb")
    mob = boonza.load(DATA / "1FNA.pdb")
    before = mob.positions.copy()
    r = boonza.cealign(mob, ref)
    ia, ib, rms, R, t, z = ce_align(_ca("1TEN.pdb", ""), _ca("1FNA.pdb", ""))
    assert r.rmsd == pytest.approx(rms) and r.z_score == pytest.approx(z)
    np.testing.assert_allclose(mob.positions, before @ R.T + t, atol=1e-9)
    fitted = np.linalg.norm(
        mob.positions[r.mobile_atoms] - ref.positions[r.reference_atoms], axis=1
    )
    assert np.sqrt((fitted**2).mean()) == pytest.approx(rms, rel=1e-6)
    assert (ref.atoms["name"][r.reference_atoms] == "CA").all()


def test_remote_homologs_align_well():
    A, B = _ca("1TEN.pdb", ""), _ca("1FNA.pdb", "")
    ia, ib, rms, R, t, z = ce_align(A, B)
    assert len(ia) >= 72 and rms < 3.0 and z >= 3.5
    assert (np.diff(ia) > 0).all() and (np.diff(ib) > 0).all()  # a sequential alignment
