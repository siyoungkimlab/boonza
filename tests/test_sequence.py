import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_ce import BIO_PYTHON, _bio_available

import boonza
from boonza.sequence import align_sequences, alignment_score, identity_matrix, sequence

HERE = Path(__file__).parent
DATA = HERE / "data"


def _chain_seq(name, chain):
    s = boonza.load(DATA / f"{name}.pdb")
    s = s.clone(s.ct_atoms(0)) if s.ncts > 1 else s
    return sequence(s, chain)


@pytest.fixture(scope="module")
def pairs():
    rng = np.random.default_rng(4)
    aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    rand = ["".join(rng.choice(aa, size=int(rng.integers(20, 90)))) for _ in range(6)]
    return [
        (_chain_seq("1LYZ", "A"), _chain_seq("1LZ1", "A")),  # hen vs human lysozyme
        (_chain_seq("1MBN", "A"), _chain_seq("1HHO", "A")),  # myoglobin vs hemoglobin alpha
        (_chain_seq("1TEN", "A"), _chain_seq("1FNA", "A")),  # remote FN3 homologs
        (_chain_seq("1HHO", "A"), _chain_seq("1HHO", "B")),
        (rand[0], rand[1]), (rand[2], rand[3]), (rand[4], rand[4][10:50]),
    ]  # fmt: skip


@pytest.mark.parametrize("mode,end_gaps", [("global", True), ("global", False), ("local", True)])
def test_scores_match_biopython(pairs, mode, end_gaps, tmp_path):
    if not _bio_available():
        pytest.skip("Biopython not available; set BOONZA_BIOPYTHON")
    cases = [{"a": a, "b": b, "mode": mode, "open": 10.0, "extend": 0.5, "end_gaps": end_gaps}
             for a, b in pairs]  # fmt: skip
    f = tmp_path / "cases.json"
    f.write_text(json.dumps(cases))
    r = subprocess.run([BIO_PYTHON, str(HERE / "bio_align_oracle.py"), str(f)],
                       capture_output=True, text=True)  # fmt: skip
    assert r.returncode == 0, r.stderr
    for (a, b), want in zip(pairs, json.loads(r.stdout), strict=True):
        aln = align_sequences(a, b, mode=mode, end_gaps=end_gaps)
        assert aln.score == pytest.approx(want["score"], abs=1e-9)
        # our alignment really scores what we report, and spells out both sequences
        if mode == "global":
            assert aln.aligned_a.replace("-", "") == a and aln.aligned_b.replace("-", "") == b
            got = alignment_score(aln.aligned_a, aln.aligned_b, end_gaps=end_gaps)
            assert got == pytest.approx(aln.score, abs=1e-9)
            if (aln.aligned_a, aln.aligned_b) == (want["a"], want["b"]):
                assert aln.identical == sum(x == y != "-" for x, y in zip(want["a"], want["b"],
                                                                          strict=True))  # fmt: skip
        else:
            assert aln.aligned_a.replace("-", "") in a and aln.aligned_b.replace("-", "") in b
            assert alignment_score(aln.aligned_a, aln.aligned_b) == pytest.approx(aln.score)


def test_statistics_and_systems():
    same = align_sequences("MKTAYIAKQR", "MKTAYIAKQR")
    assert same.identity == 1.0 and same.similarity == 1.0 and same.gaps == 0
    one = align_sequences("MKTAYIAKQR", "MKTAHIAKQR")
    assert one.identical == 9 and one.identity == pytest.approx(0.9)
    assert one.positive >= 9 and "|||| |" not in str(one) and "Y" in str(one)
    ref = boonza.load(DATA / "1LYZ.pdb")
    mob = boonza.load(DATA / "1LZ1.pdb")
    aln = align_sequences(ref, mob)
    assert 0.55 < aln.identity < 0.65  # hen and human lysozyme are ~60% identical
    assert aln.similarity > aln.identity
    m = identity_matrix([sequence(ref), sequence(mob), sequence(ref)])
    assert m[0, 2] == 1.0 and m[0, 1] == pytest.approx(aln.identity) and (m == m.T).all()
    two = boonza.load(DATA / "1HHO.pdb")
    with pytest.raises(ValueError, match="pass chain="):
        sequence(two)
    assert len(sequence(two, "A")) > 100
