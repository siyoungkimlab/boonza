"""boonza.matchmaker and chimerax_ss against UCSF ChimeraX itself (run headless)."""

import functools
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import boonza
from boonza.matchmaker import chain_sequence, matchmaker, needleman_wunsch
from boonza.secondary import chimerax_ss

HERE = Path(__file__).parent
DATA = HERE / "data"
CHIMERAX = os.path.expanduser(
    os.environ.get("BOONZA_CHIMERAX", "/Applications/ChimeraX-1.11.app/Contents/bin/ChimeraX")
)
# (reference, chain, mobile, chain): close homologs, remote homologs, a self pair
PAIRS = [("1LYZ", "A", "1LZ1", "A"), ("1TEN", "A", "1FNA", "A"), ("1MBN", "A", "1HHO", "A"),
         ("1HHO", "A", "1HHO", "B"), ("1MBN", "A", "1HHO", "B")]  # fmt: skip


@functools.cache
def _oracle(ref, rc, mob, mc, tmpdir):
    if not os.path.exists(CHIMERAX):
        pytest.skip("ChimeraX not available; set BOONZA_CHIMERAX")
    out = Path(tmpdir) / f"{ref}{rc}_{mob}{mc}.json"
    script = f"{HERE / 'chimerax_oracle.py'} {DATA / ref}.pdb {rc} {DATA / mob}.pdb {mc} {out}"
    r = subprocess.run([CHIMERAX, "--nogui", "--exit", "--script", script], capture_output=True,
                       text=True, timeout=600)  # fmt: skip
    if not out.exists():
        raise RuntimeError(f"ChimeraX oracle failed:\n{r.stdout}\n{r.stderr}")
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("chimerax"))
    return lambda *pair: _oracle(*pair, tmp)


def _load(name):
    s = boonza.load(DATA / f"{name}.pdb")
    return s.clone(s.ct_atoms(0)) if s.ncts > 1 else s


def _chain_index(s, name):
    return [c for c in range(s.nchains) if s.chains["name"][c] == name][0]


def _residue_keys(s, atoms):
    res = s.atoms["residue"][atoms]
    return [[int(s.residues["resid"][r]), str(s.residues["insertion"][r])] for r in res]


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: "-".join(p))
def test_sequences_and_ss_match_chimerax(pair, oracle):
    want = oracle(*pair)
    for (name, chain), key in (((pair[0], pair[1]), "ref"), ((pair[2], pair[3]), "mobile")):
        s = _load(name)
        seq, res, _ = chain_sequence(s, _chain_index(s, chain))
        assert seq == want["chains"][key]["characters"]
        ss = "".join(chimerax_ss(s)[res].tolist())
        assert ss == "".join(r[3] for r in want["chains"][key]["residues"]), name


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: "-".join(p))
def test_matchmaker_matches_chimerax(pair, oracle):
    want = oracle(*pair)
    ref, mob = _load(pair[0]), _load(pair[2])
    before = mob.positions.copy()
    r = matchmaker(mob, ref, mobile_chain=pair[3], reference_chain=pair[1])
    assert r.aligned_reference == want["aligned_ref"]
    assert r.aligned_mobile == want["aligned_mobile"]
    assert _residue_keys(ref, r.reference_atoms) == want["full_ref"]
    assert _residue_keys(mob, r.mobile_atoms) == want["full_mobile"]
    assert sorted(_residue_keys(ref, r.reference_atoms[r.kept])) == sorted(want["final_ref"])
    assert r.rmsd == pytest.approx(want["final_rmsd"], abs=1e-6)
    assert r.full_rmsd == pytest.approx(want["full_rmsd"], abs=1e-6)
    M = np.array(want["matrix"])
    np.testing.assert_allclose(r.rotation, M[:, :3], atol=1e-6)
    np.testing.assert_allclose(r.translation, M[:, 3], atol=1e-5)
    np.testing.assert_allclose(mob.positions, r.apply(before), atol=1e-9)


def test_best_chain_pair_is_chosen():
    ref, mob = _load("1MBN"), _load("1HHO")
    r = matchmaker(mob, ref, apply=False)
    assert ref.chains["name"][r.reference_chain] == "A"
    # myoglobin is closer to the hemoglobin beta chain than to alpha by sequence score
    scores = {c: matchmaker(mob, ref, mobile_chain=c, apply=False).score for c in "AB"}
    assert mob.chains["name"][r.mobile_chain] == max(scores, key=scores.get)


def test_homologs_beat_identity_only_alignment():
    """Myoglobin vs hemoglobin alpha (~25% identity): matchmaker pairs far more residues."""
    ref, mob = _load("1MBN"), _load("1HHO")
    mm = matchmaker(mob.copy(), ref, mobile_chain="A", reference_chain="A", apply=False)
    old = boonza.superpose(mob.copy(), ref, sel="chain A and name CA", apply=False)
    assert len(mm.kept) > old.n_used and len(mm.kept) >= 100 and mm.rmsd < 2.0


def test_needleman_wunsch_basics():
    score, i, j = needleman_wunsch("ACDEFGHIK", "ACDFGHIK")
    assert list(zip(i.tolist(), j.tolist(), strict=True))[:3] == [(0, 0), (1, 1), (2, 2)]
    assert len(i) == 8 and score > 0
