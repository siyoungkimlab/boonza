"""Pairwise sequence alignment with identity and similarity (Biopython / EMBOSS conventions).

    aln = boonza.align_sequences("MKTAYIAK", "MKTAHIAK")        # global, BLOSUM-62, 10 / 0.5
    aln = boonza.align_sequences(sys_a, sys_b, chain_a="A", chain_b="B", mode="local")
    aln.identity, aln.similarity, aln.gap_fraction, aln.score
    print(aln)                                                  # 60-column blocks
    boonza.identity_matrix([seq1, seq2, seq3])
    boonza.sequence(system, "A")                                # one-letter chain sequence

Scoring follows Biopython's PairwiseAligner and EMBOSS needle/water: a gap of
length L scores ``-(gap_open + (L - 1) * gap_extend)``.  ``mode="global"``
scores end gaps like internal gaps unless ``end_gaps=False`` (EMBOSS needle's
default); ``mode="local"`` is Smith-Waterman.  Identity and similarity are
EMBOSS's: identical (or positively scoring) aligned pairs over the number of
alignment columns, gaps included.  For superposition driven by an alignment
use ``boonza.matchmaker``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .matchmaker import BLOSUM62, _lookup, chain_sequence

_NEG = -1e300


@njit(cache=True)
def _gotoh(S, go, ge, local, end_gaps):
    """Affine-gap DP: values and traceback pointers per state.

    States: 0 match, 1 gap in b (consumes a), 2 gap in a (consumes b); 3 starts a local alignment.
    """
    n, m = S.shape
    M = np.full((n + 1, m + 1), _NEG)
    X = np.full((n + 1, m + 1), _NEG)
    Y = np.full((n + 1, m + 1), _NEG)
    tM = np.zeros((n + 1, m + 1), np.int8)
    tX = np.zeros((n + 1, m + 1), np.int8)
    tY = np.zeros((n + 1, m + 1), np.int8)
    M[0, 0] = 0.0
    if not local:
        for i in range(1, n + 1):
            X[i, 0] = (go + (i - 1) * ge) if end_gaps else 0.0
            tX[i, 0] = 1 if i > 1 else 0
        for j in range(1, m + 1):
            Y[0, j] = (go + (j - 1) * ge) if end_gaps else 0.0
            tY[0, j] = 2 if j > 1 else 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            b, t = M[i - 1, j - 1], 0
            if X[i - 1, j - 1] > b:
                b, t = X[i - 1, j - 1], 1
            if Y[i - 1, j - 1] > b:
                b, t = Y[i - 1, j - 1], 2
            if local and b < 0.0:
                b, t = 0.0, 3
            M[i, j] = b + S[i - 1, j - 1]
            tM[i, j] = t
            b, t = M[i - 1, j] + go, 0
            v = X[i - 1, j] + ge
            if v > b:
                b, t = v, 1
            v = Y[i - 1, j] + go
            if v > b:
                b, t = v, 2
            X[i, j] = b
            tX[i, j] = t
            b, t = M[i, j - 1] + go, 0
            v = X[i, j - 1] + go
            if v > b:
                b, t = v, 1
            v = Y[i, j - 1] + ge
            if v > b:
                b, t = v, 2
            Y[i, j] = b
            tY[i, j] = t
    return M, X, Y, tM, tX, tY


def _score_matrix(a, b, matrix) -> np.ndarray:
    ca, ia = np.unique(list(a), return_inverse=True)
    cb, ib = np.unique(list(b), return_inverse=True)
    table = np.array([[_lookup(matrix, x, y) for y in cb] for x in ca], dtype=np.float64)
    return np.ascontiguousarray(table[ia][:, ib])


@dataclass
class SequenceAlignment:
    """A pairwise alignment: gapped strings, score, paired positions and statistics."""

    seq_a: str
    seq_b: str
    aligned_a: str
    aligned_b: str
    score: float
    mode: str
    pairs: np.ndarray  # (k, 2) aligned residue positions in seq_a and seq_b
    positive: int  # aligned pairs scoring > 0 (identities included)

    @property
    def length(self) -> int:
        """Number of alignment columns, gaps included."""
        return len(self.aligned_a)

    @property
    def identical(self) -> int:
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        return int(sum(self.seq_a[x] == self.seq_b[y] for x, y in zip(a.tolist(), b.tolist(),
                                                                       strict=True)))  # fmt: skip

    @property
    def gaps(self) -> int:
        return self.length - len(self.pairs)

    @property
    def identity(self) -> float:
        """Identical pairs over alignment columns (EMBOSS "Identity")."""
        return self.identical / self.length if self.length else 0.0

    @property
    def similarity(self) -> float:
        """Positively scoring pairs over alignment columns (EMBOSS "Similarity")."""
        return self.positive / self.length if self.length else 0.0

    @property
    def gap_fraction(self) -> float:
        return self.gaps / self.length if self.length else 0.0

    @property
    def identity_shorter(self) -> float:
        """Identical pairs over the length of the shorter sequence."""
        short = min(len(self.seq_a), len(self.seq_b))
        return self.identical / short if short else 0.0

    def __str__(self) -> str:
        marks = []
        for x, y in zip(self.aligned_a, self.aligned_b, strict=True):
            if x == "-" or y == "-":
                marks.append(" ")
            elif x == y:
                marks.append("|")
            else:
                marks.append(":" if _lookup(BLOSUM62, x, y) > 0 else ".")
        mark = "".join(marks)
        head = (f"{self.mode} alignment, score {self.score:g}, length {self.length}: identity "
                f"{self.identical}/{self.length} ({100 * self.identity:.1f}%), similarity "
                f"{self.positive}/{self.length} ({100 * self.similarity:.1f}%), gaps "
                f"{self.gaps}/{self.length} ({100 * self.gap_fraction:.1f}%)")  # fmt: skip
        blocks = [head]
        for k in range(0, self.length, 60):
            blocks.append("\n".join((self.aligned_a[k : k + 60], mark[k : k + 60],
                                     self.aligned_b[k : k + 60])))  # fmt: skip
        return "\n\n".join(blocks)


def sequence(system, chain=None) -> str:
    """One-letter sequence of a chain (index or name) of amino acids and nucleotides."""
    if chain is None:
        chains = [c for c in range(system.nchains) if chain_sequence(system, c)[0]]
        if len(chains) != 1:
            names = [str(system.chains["name"][c]) for c in chains]
            raise ValueError(f"the system has {len(chains)} chains with a sequence ({names}); "
                             "pass chain=")  # fmt: skip
        chain = chains[0]
    elif isinstance(chain, str):
        found = [c for c in range(system.nchains) if system.chains["name"][c] == chain]
        if not found:
            raise ValueError(f"no chain {chain!r}")
        chain = found[0]
    return chain_sequence(system, int(chain))[0]


def align_sequences(a, b, mode: str = "global", matrix=None, gap_open: float = 10.0,
                    gap_extend: float = 0.5, end_gaps: bool = True, chain_a=None,
                    chain_b=None) -> SequenceAlignment:  # fmt: skip
    """Align two sequences (strings, or Systems with ``chain_a``/``chain_b``).

    ``matrix``: {(x, y): score} (default BLOSUM-62).  ``gap_open`` and
    ``gap_extend`` are positive penalties.  See the module docstring for the
    scoring and the statistics.
    """
    if mode not in ("global", "local"):
        raise ValueError("mode must be 'global' or 'local'")
    if not isinstance(a, str):
        a = sequence(a, chain_a)
    if not isinstance(b, str):
        b = sequence(b, chain_b)
    a, b = a.upper(), b.upper()
    if not a or not b:
        raise ValueError("both sequences must be non-empty")
    m = BLOSUM62 if matrix is None else matrix
    S = _score_matrix(a, b, m)
    local = mode == "local"
    M, X, Y, tM, tX, tY = _gotoh(S, -float(gap_open), -float(gap_extend), local, end_gaps)
    n, k = len(a), len(b)
    tail_a, tail_b = "", ""
    if local:
        i, j = np.unravel_index(int(np.argmax(M)), M.shape)
        state, score = 0, float(M[i, j])
    elif end_gaps:
        vals = (M[n, k], X[n, k], Y[n, k])
        state = int(np.argmax(vals))
        i, j, score = n, k, float(vals[state])
    else:  # trailing gaps are free: end anywhere on the last row or column
        best = np.maximum(np.maximum(M, X), Y)
        cands = [(best[n, k], n, k)] + [(best[n, jj], n, jj) for jj in range(k)] + \
            [(best[ii, k], ii, k) for ii in range(n)]  # fmt: skip
        score, i, j = max(cands, key=lambda c: c[0])
        score = float(score)
        state = int(np.argmax((M[i, j], X[i, j], Y[i, j])))
        tail_a = a[i:] + "-" * (k - j)
        tail_b = "-" * (n - i) + b[j:]
    cols_a, cols_b, pairs = [], [], []
    i, j = int(i), int(j)
    while i > 0 or j > 0:
        if state == 0:
            t = tM[i, j]
            cols_a.append(a[i - 1])
            cols_b.append(b[j - 1])
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
            if t == 3:
                break
            state = int(t)
        elif state == 1:
            t = tX[i, j]
            cols_a.append(a[i - 1])
            cols_b.append("-")
            i -= 1
            state = int(t)
        else:
            t = tY[i, j]
            cols_a.append("-")
            cols_b.append(b[j - 1])
            j -= 1
            state = int(t)
        if local and state == 0 and (i == 0 or j == 0):
            break
    aligned_a = "".join(reversed(cols_a)) + tail_a
    aligned_b = "".join(reversed(cols_b)) + tail_b
    pairs = np.array(pairs[::-1], np.int64).reshape(-1, 2)
    positive = sum(_lookup(m, a[x], b[y]) > 0 for x, y in pairs.tolist())
    return SequenceAlignment(a, b, aligned_a, aligned_b, score, mode, pairs, int(positive))


def alignment_score(aligned_a: str, aligned_b: str, matrix=None, gap_open: float = 10.0,
                    gap_extend: float = 0.5, end_gaps: bool = True) -> float:  # fmt: skip
    """Score of a given gapped alignment under the same scheme (for checking alignments)."""
    m = BLOSUM62 if matrix is None else matrix
    total, run_a, run_b = 0.0, 0, 0
    cols = list(zip(aligned_a, aligned_b, strict=True))
    first = next(k for k, (x, y) in enumerate(cols) if x != "-" and y != "-") if cols else 0
    last = max((k for k, (x, y) in enumerate(cols) if x != "-" and y != "-"), default=-1)
    for k, (x, y) in enumerate(cols):
        end = k < first or k > last
        if x == "-" or y == "-":
            gap_in_a = x == "-"
            opening = (run_a == 0) if gap_in_a else (run_b == 0)
            if end_gaps or not end:
                total -= gap_open if opening else gap_extend
            run_a, run_b = (run_a + 1, 0) if gap_in_a else (0, run_b + 1)
        else:
            total += _lookup(m, x, y)
            run_a = run_b = 0
    return total


def identity_matrix(sequences, metric: str = "identity", **kwargs) -> np.ndarray:
    """Pairwise ``metric`` ("identity", "similarity" or "identity_shorter") of sequences."""
    seqs = list(sequences)
    out = np.eye(len(seqs))
    for x in range(len(seqs)):
        for y in range(x + 1, len(seqs)):
            out[x, y] = out[y, x] = getattr(align_sequences(seqs[x], seqs[y], **kwargs), metric)
    return out
