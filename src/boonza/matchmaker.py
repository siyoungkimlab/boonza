"""Sequence-guided superposition, a port of UCSF ChimeraX matchmaker.

    result = boonza.matchmaker(mobile, reference)          # moves ``mobile``
    result = boonza.matchmaker(mobile, reference, mobile_chain="B", reference_chain="A")

The two chains are aligned by Needleman-Wunsch with ChimeraX's defaults:
BLOSUM-62 mixed with a secondary-structure score (fraction 0.3), gap opening
12 (18 inside helices and strands, 6 elsewhere), extension 1, and free end
gaps.  Aligned residues are paired by their principal atom (CA, or C4' for
nucleotides) and superposed, repeatedly discarding the worst-fitting pairs
(at most 10% per round, and at most half of those beyond ``cutoff``) until
all remaining pairs lie within ``cutoff`` (2 Å).  With several chains and
no chain given, every chain pair is aligned and the best-scoring pair is
used, as ChimeraX's default "best-best" pairing.

Ported from ChimeraX (alignment_algs/_nw/nw.cpp, match_maker/match.py,
std_commands/align.py), Copyright Regents of the University of California,
LGPL 2.1 for the ported C++ (see NOTICE).  Secondary structure comes from
``boonza.chimerax_ss``, a port of ChimeraX's own DSSP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

from .align import THREE2ONE, kabsch

_BLOSUM62 = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X  *
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0 -4
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1  0 -1 -4
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1 -4
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1 -4
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1 -4
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1 -4
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1 -4
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1 -4
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1 -4
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1 -4
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1 -4
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1 -4
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -1 -2 -4
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0  0  0 -4
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1  0 -4
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2 -4
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1 -4
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3 -2 -1 -4
B -2 -1  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1 -4
Z -1  0  0  1 -3  3  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
X  0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2  0  0 -2 -1 -1 -1 -1 -1 -4
* -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1
"""

SS_SCORES = {("H", "H"): 6.0, ("S", "S"): 6.0, ("O", "O"): 4.0, ("S", "H"): -9.0,
             ("H", "S"): -9.0, ("S", "O"): -6.0, ("O", "S"): -6.0, ("H", "O"): -6.0,
             ("O", "H"): -6.0}  # fmt: skip
NUCLEIC = {"A": "A", "C": "C", "G": "G", "U": "U", "T": "T", "DA": "A", "DC": "C",
           "DG": "G", "DT": "T", "DU": "U", "ADE": "A", "CYT": "C", "GUA": "G", "URA": "U",
           "THY": "T"}  # fmt: skip


def _read_matrix(text) -> dict:
    """ChimeraX sim_matrices.read_matrix_file for a protein matrix."""
    lines = [ln.split() for ln in text.strip().splitlines()]
    chars = lines[0]
    m = {}
    for row in lines[1:]:
        for c, v in zip(chars, row[1:], strict=True):
            m[(row[0], c)] = float(v)
    for c in chars:
        m[(c, "?")] = m[("?", c)] = 0.0
    m[("?", "?")] = 0.0
    for c in [*chars, "?"]:  # pyrrolysine as X, selenocysteine as C
        m[(c, "O")] = m[("O", c)] = m[(c, "X")]
        m[(c, "U")] = m[("U", c)] = m[(c, "C")]
    m[("O", "U")] = m[("U", "O")] = m[("X", "C")]
    m[("O", "O")] = m[("X", "X")]
    m[("U", "U")] = m[("C", "C")]
    return m


BLOSUM62 = _read_matrix(_BLOSUM62)


def _lookup(m, c1, c2):
    """align_algs::matrix_lookup: exact pair, then '*' wildcards."""
    for key in ((c1, c2), ("*", c2), (c1, "*"), ("*", "*")):
        if key in m:
            return m[key]
    raise KeyError(f"no similarity score for {c1!r}/{c2!r}")


def _score_matrix(seq1, seq2, ss1, ss2, sim, ss_scores, ss_fraction) -> np.ndarray:
    chars1, inv1 = np.unique(list(seq1), return_inverse=True)
    chars2, inv2 = np.unique(list(seq2), return_inverse=True)
    table = np.array([[_lookup(sim, a, b) for b in chars2] for a in chars1])
    score = table[inv1][:, inv2]
    if ss_fraction:
        ssm = {**ss_scores}
        for c in "HSO ":  # missing residues have a blank SS type
            ssm[(c, " ")] = ssm[(" ", c)] = 0.0
        s1, i1 = np.unique(list(ss1), return_inverse=True)
        s2, i2 = np.unique(list(ss2), return_inverse=True)
        sst = np.array([[_lookup(ssm, a, b) for b in s2] for a in s1])
        score = ss_fraction * sst[i1][:, i2] + (1.0 - ss_fraction) * score
    return np.ascontiguousarray(score, dtype=np.float64)


def _gap_opens(ss, gap_open, helix, strand, other, ss_specific) -> np.ndarray:
    """Per-position gap-open scores (negative), index k for a gap before residue k."""
    g = np.full(len(ss) + 1, gap_open)
    g[0] = 0.0  # end gaps are free
    if ss_specific:
        for i in range(len(ss) - 1):
            a, b = ss[i], ss[i + 1]
            g[i + 1] = helix if a == b == "H" else strand if a == b == "S" else other
    return g


@njit(cache=True)
def _nw(score, go1, go2, gap_extend):
    """ChimeraX nw.cpp ``match`` with free end gaps; returns (score, i1 list, i2 list)."""
    rows, cols = score.shape[0] + 1, score.shape[1] + 1
    m = np.zeros((rows, cols))
    bt = np.zeros((rows, cols), np.int64)
    bt[:, 0] = 1
    bt[0, :] = 2
    col_gap_starts = np.zeros(cols - 1, np.int64)
    base_col = 0.0
    base_row = 0.0
    for i1 in range(rows - 1):
        row_gap_pos = 0
        for i2 in range(cols - 1):
            best = m[i1, i2] + score[i1, i2]
            bt_type = 0
            if i2 + 1 < cols - 1:
                cgp = col_gap_starts[i2]
                size = i1 + 1 - cgp
                base_col = m[cgp, i2 + 1] + size * gap_extend
                skip = base_col + go2[i2 + 1]
            else:
                size = 1
                skip = m[i1, i2 + 1]
            if skip > best:
                best = skip
                bt_type = size
            if i1 + 1 < rows - 1:
                size = i2 + 1 - row_gap_pos
                base_row = m[i1 + 1, row_gap_pos] + size * gap_extend
                skip = base_row + go1[i1 + 1]
            else:
                size = 1
                skip = m[i1 + 1, i2]
            if skip > best:
                best = skip
                bt_type = -size
            m[i1 + 1, i2 + 1] = best
            bt[i1 + 1, i2 + 1] = bt_type
            if bt_type >= 0 and best > base_row:
                row_gap_pos = i2 + 1
            if bt_type <= 0 and best > base_col:
                col_gap_starts[i2] = i1 + 1
    out1 = np.empty(min(rows, cols), np.int64)
    out2 = np.empty(min(rows, cols), np.int64)
    k = 0
    i1, i2 = rows - 1, cols - 1
    while i1 > 0 and i2 > 0:
        t = bt[i1, i2]
        if t == 0:
            out1[k], out2[k] = i1 - 1, i2 - 1
            k += 1
            i1 -= 1
            i2 -= 1
        elif t > 0:
            i1 -= t
        else:
            i2 += t
    return m[rows - 1, cols - 1], out1[:k][::-1].copy(), out2[:k][::-1].copy()


def needleman_wunsch(seq1, seq2, ss1=None, ss2=None, matrix=None, gap_open=12.0,
                     gap_extend=1.0, ss_fraction=0.3, ss_scores=None, gap_open_helix=18.0,
                     gap_open_strand=18.0, gap_open_other=6.0):  # fmt: skip
    """ChimeraX Needleman-Wunsch: (score, matched positions in seq1, in seq2).

    ``ss1``/``ss2`` are strings of H (helix), S (strand), O (other) or blank
    (unknown), one per residue; without them only the similarity matrix and
    the plain gap penalties are used.  Penalties are given as positive numbers.
    """
    use_ss = ss_fraction not in (None, False) and ss1 is not None and ss2 is not None
    ssf = float(ss_fraction) if use_ss else 0.0
    score = _score_matrix(seq1, seq2, ss1 or " " * len(seq1), ss2 or " " * len(seq2),
                          matrix or BLOSUM62, ss_scores or SS_SCORES, ssf)  # fmt: skip
    args = (-gap_open, -gap_open_helix, -gap_open_strand, -gap_open_other, use_ss)
    go1 = _gap_opens(ss1 or "", *args) if use_ss else _gap_opens(" " * len(seq1), *args)
    go2 = _gap_opens(ss2 or "", *args) if use_ss else _gap_opens(" " * len(seq2), *args)
    total, i1, i2 = _nw(score, go1, go2, -float(gap_extend))
    return float(total), i1, i2


def align_and_prune(mobile, reference, cutoff):
    """ChimeraX std_commands.align.align_and_prune: (rotation, translation, rmsd, kept indices)."""
    keep = np.arange(len(mobile))
    cut2 = cutoff * cutoff
    while True:
        R, t = kabsch(mobile[keep], reference[keep])
        d2 = ((mobile[keep] @ R.T + t - reference[keep]) ** 2).sum(axis=1)
        order = np.argsort(d2, kind="stable")
        if d2[order[-1]] <= cut2:
            return R, t, float(np.sqrt(d2.mean())), keep
        index = max(int(len(d2) * 0.9), int(((d2 <= cut2).sum() + len(d2)) / 2))
        keep = keep[order[:index]]
        if len(keep) < 3:
            raise ValueError(f"pruning distances > {cutoff:g} left fewer than 3 atom pairs")


@dataclass
class MatchResult:
    """Outcome of ``matchmaker``: the transform and the residue and atom pairing."""

    rotation: np.ndarray
    translation: np.ndarray
    rmsd: float  # over the pairs kept after pruning
    full_rmsd: float  # over all paired atoms, after the final fit
    score: float  # sequence-alignment score
    mobile_atoms: np.ndarray  # all paired atoms
    reference_atoms: np.ndarray
    kept: np.ndarray  # indices into the paired atoms that survived pruning
    mobile_chain: int
    reference_chain: int
    aligned_mobile: str = ""
    aligned_reference: str = ""
    extra: dict = field(default_factory=dict)

    def apply(self, positions) -> np.ndarray:
        return np.asarray(positions, dtype=np.float64) @ self.rotation.T + self.translation


def chain_sequence(system, chain):
    """(one-letter sequence, residue indices, principal atom per residue or -1) of a chain.

    Residues are the chain's amino acids and nucleotides in order (as in a
    ChimeraX chain), whether or not their backbone is complete.
    """
    res = system.chain_residues(chain)
    names = system.residues["name"][res].tolist()
    letters, keep = [], []
    for k, nm in enumerate(names):
        key = nm.upper()
        if key in THREE2ONE:
            letters.append(THREE2ONE[key])
        elif key in NUCLEIC:
            letters.append(NUCLEIC[key])
        else:
            continue
        keep.append(res[k])
    keep = np.array(keep, dtype=np.int64)
    atom_res = system.atoms["residue"]
    atom_names = system.atoms["name"]
    principal = np.full(len(keep), -1, np.int64)
    where = {int(r): k for k, r in enumerate(keep.tolist())}
    for target in ("CA", "C4'", "P"):  # amino acids, then nucleotides, then P-only traces
        hits = np.flatnonzero(atom_names == target)
        for a, r in zip(hits.tolist(), atom_res[hits].tolist(), strict=True):
            k = where.get(r)
            if k is not None and principal[k] < 0:
                principal[k] = a
    return "".join(letters), keep, principal


def _gapped(seq, idx, other_len, other_idx, first=True):
    """Gapped strings from match lists (only used for reporting)."""
    out1, out2, p1, p2 = [], [], 0, 0
    s1, s2 = seq
    for a, b in zip(*idx, strict=True):
        while p1 < a:
            out1.append(s1[p1])
            out2.append(".")
            p1 += 1
        while p2 < b:
            out1.append(".")
            out2.append(s2[p2])
            p2 += 1
        out1.append(s1[a])
        out2.append(s2[b])
        p1, p2 = a + 1, b + 1
    out1 += list(s1[p1:]) + ["."] * (len(s2) - p2)
    out2 += ["."] * (len(s1) - p1) + list(s2[p2:])
    return "".join(out1), "".join(out2)


def matchmaker(mobile, reference, mobile_chain=None, reference_chain=None, cutoff=2.0,
               ss=True, apply: bool = True, **nw_options) -> MatchResult:  # fmt: skip
    """Superpose ``mobile`` onto ``reference`` (Systems) by sequence alignment, ChimeraX style.

    ``mobile_chain``/``reference_chain``: chain index or chain name; by
    default every pair of chains with sequence is tried and the best
    alignment score wins.  ``ss``: use secondary structure from
    ``chimerax_ss`` (True), none (False), or a dict {system id: per-residue
    H/S/O array}.  ``cutoff=None`` fits all pairs without pruning.  Extra
    keywords go to ``needleman_wunsch`` (``gap_open``, ``ss_fraction``, ...).
    ``apply`` moves ``mobile`` (positions and cell).
    """
    from .secondary import chimerax_ss

    def chains_of(s, which):
        if which is None:
            return list(range(s.nchains))
        if isinstance(which, str):
            found = [c for c in range(s.nchains) if s.chains["name"][c] == which]
            if not found:
                raise ValueError(f"no chain {which!r}")
            return found
        return [int(which)]

    def ss_of(s):
        if ss is True:
            return chimerax_ss(s)
        if isinstance(ss, dict):
            return np.asarray(ss[id(s)])
        return None

    ss_mob, ss_ref = ss_of(mobile), ss_of(reference)
    best = None
    for rc in chains_of(reference, reference_chain):
        rseq, rres, ratom = chain_sequence(reference, rc)
        if len(rseq) < 3:
            continue
        for mc in chains_of(mobile, mobile_chain):
            mseq, mres, matom = chain_sequence(mobile, mc)
            if len(mseq) < 3:
                continue
            rss = None if ss_ref is None else "".join(ss_ref[rres].tolist())
            mss = None if ss_mob is None else "".join(ss_mob[mres].tolist())
            score, i1, i2 = needleman_wunsch(rseq, mseq, rss, mss,
                                             ss_fraction=0.3 if ss is not False else None,
                                             **nw_options)  # fmt: skip
            if best is None or score > best[0]:
                best = (score, rc, mc, rseq, mseq, rres, mres, ratom, matom, i1, i2)
    if best is None:
        raise ValueError("no chains with at least 3 residues to match")
    score, rc, mc, rseq, mseq, rres, mres, ratom, matom, i1, i2 = best
    ok = (ratom[i1] >= 0) & (matom[i2] >= 0)
    ref_atoms, mob_atoms = ratom[i1][ok], matom[i2][ok]
    if len(ref_atoms) < 3:
        raise ValueError("fewer than 3 residues aligned")
    P, Q = mobile.positions[mob_atoms], reference.positions[ref_atoms]
    if cutoff is None:
        R, t = kabsch(P, Q)
        keep = np.arange(len(P))
        rmsd = float(np.sqrt(((P @ R.T + t - Q) ** 2).sum(axis=1).mean()))
    else:
        R, t, rmsd, keep = align_and_prune(P, Q, float(cutoff))
    full = float(np.sqrt(((P @ R.T + t - Q) ** 2).sum(axis=1).mean()))
    ga, gb = _gapped((rseq, mseq), (i1, i2), len(mseq), i2)
    result = MatchResult(R, t, rmsd, full, score, mob_atoms, ref_atoms, np.sort(keep), mc, rc,
                         aligned_mobile=gb, aligned_reference=ga)  # fmt: skip
    if apply:
        mobile.positions = result.apply(mobile.positions)
        if mobile.cell.any():
            mobile.cell = mobile.cell @ R.T
    return result
