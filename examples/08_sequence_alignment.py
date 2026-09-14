"""Pairwise sequence alignment, identity and similarity.

Global (Needleman-Wunsch) and local (Smith-Waterman) alignment with
BLOSUM-62 and affine gaps, EMBOSS-style statistics, and an identity matrix
over several chains.

    python examples/08_sequence_alignment.py
"""

import numpy as np
from _common import DATA

import boonza

hen = boonza.load(DATA / "1LYZ.pdb")
human = boonza.load(DATA / "1LZ1.pdb")

# Systems can be passed directly; their chain sequence is used.
aln = boonza.align_sequences(hen, human)
print(aln)
print(f"\nidentity {aln.identity:.1%}, similarity {aln.similarity:.1%}, "
      f"gaps {aln.gap_fraction:.1%}")  # fmt: skip

# Local alignment finds the best-matching stretch only.
motif = "CAKKIVSDGNGMNAWVAWR"
local = boonza.align_sequences(boonza.sequence(hen), motif, mode="local")
print(f"\nlocal alignment of a motif: score {local.score:g}, starts at residue "
      f"{local.pairs[0, 0] + 1} of hen lysozyme")  # fmt: skip
print(local.aligned_a)
print(local.aligned_b)

# Free end gaps (EMBOSS needle's default) for a fragment against a full chain.
frag = boonza.sequence(hen)[20:60]
semi = boonza.align_sequences(boonza.sequence(hen), frag, end_gaps=False)
print(f"\nfragment 21-60 with free end gaps: score {semi.score:g}, "
      f"identity over the fragment {semi.identical / len(frag):.0%}")  # fmt: skip

# Identity matrix over several chains.
hemo = boonza.load(DATA / "1HHO.pdb")
chains = {
    "Hb alpha": boonza.sequence(hemo, "A"),
    "Hb beta": boonza.sequence(hemo, "B"),
    "myoglobin": boonza.sequence(boonza.load(DATA / "1MBN.pdb")),
    "lysozyme": boonza.sequence(hen),
}
m = boonza.identity_matrix(list(chains.values()))
print("\npairwise identity (%)")
print(" " * 11 + "".join(f"{k:>11s}" for k in chains))
for k, row in zip(chains, m, strict=True):
    print(f"{k:>11s}" + "".join(f"{100 * v:11.0f}" for v in row))
names = list(chains)
i, j = max(((i, j) for i in range(len(names)) for j in range(i + 1, len(names))),
           key=lambda ij: m[ij])  # fmt: skip
print(f"closest pair: {names[i]} and {names[j]} ({m[i, j]:.0%} identical)")
assert np.allclose(np.diag(m), 1)
