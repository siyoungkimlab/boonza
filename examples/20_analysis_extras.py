"""More analysis: native contacts, contact frequencies, principal components,
secondary structure along a path, and block averaging.

A peptide is unfolded from a helix to an extended strand in 20 steps.  The
path is made up (straight-line interpolation, not dynamics), but it is
enough to see what each tool reports.

    python examples/20_analysis_extras.py
"""

import numpy as np
from _common import require

require("rdkit")

import boonza  # noqa: E402

seq = "AEAAAKEAAAKA"
helix = boonza.peptide(seq, "helix", optimize=False)
strand = boonza.peptide(seq, "extended", optimize=False)
rot, shift = boonza.kabsch(strand.positions, helix.positions)
extended = strand.positions @ rot.T + shift
steps = np.linspace(0.0, 1.0, 21)
frames = np.array([(1 - t) * helix.positions + t * extended for t in steps])

# Fraction of native contacts: pairs of heavy atoms between the two halves
# that are within 4.5 Å in the helix.
q = boonza.native_contacts(helix, "resid 1 to 6 and noh", "resid 7 to 12 and noh",
                           positions=frames, method="soft_cut")  # fmt: skip
print("native contacts Q along the path:", np.round(q[::5], 2).tolist())

# Which residue pairs touch, and how often (neighbors in sequence always do).
rows, cols, freq = boonza.contact_frequency(helix, "all", positions=frames, cutoff=4.5)
far = np.abs(rows[:, None] - cols[None, :]) >= 3
often = int((freq[far] > 0.5).sum() // 2)
print(f"residue pairs 3+ apart in contact in over half the frames: {often}")

# Principal components of the C-alpha motion: one direction carries the unfolding.
p = boonza.pca(helix, positions=frames, sel="name CA")
print(f"first principal component: {p.cumulated_variance[0]:.0%} of the variance; "
      f"projections {np.round(p.projections[::5, 0], 1).tolist()}")  # fmt: skip

# DSSP frame by frame (chains are cut at C-N gaps, as the DSSP program does)
codes = boonza.dssp(helix, frames, simplified=True)
for k in (0, 5, 10, 20):
    print(f"step {k:2d}: {''.join(codes[k])}")

# Block averaging: the error of the mean of a correlated series (an AR(1) process).
rng = np.random.default_rng(0)
x = np.empty(20000)
x[0] = 0.0
for k in range(1, len(x)):
    x[k] = 0.95 * x[k - 1] + rng.normal()
b = boonza.block_average(x)
print(f"\nmean {b.mean:.3f}: naive error {b.sem[0]:.4f}, from blocks {b.estimate:.4f} "
      f"(statistical inefficiency {b.statistical_inefficiency:.0f})")  # fmt: skip
