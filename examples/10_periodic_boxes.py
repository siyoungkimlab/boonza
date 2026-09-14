"""Periodic boundaries: minimum-image distances, making molecules whole, wrapping.

Uses a small water box.  We break molecules across the box on purpose,
then repair them, and compare orthorhombic and triclinic cells.

    python examples/10_periodic_boxes.py
"""

import numpy as np
from _common import water_box

import boonza
from boonza import pbc

s = water_box(n=4)  # 64 waters in a 12.4 A cube
L = s.cell[0, 0]
print(s, "| cell edge", L, "A")

# Minimum-image distances: two atoms near opposite faces are actually close.
a, b = np.array([[0.5, 5.0, 5.0]]), np.array([[L - 0.5, 5.0, 5.0]])
print("plain distance:", round(float(np.linalg.norm(a - b)), 2),
      "| periodic distance:", round(float(pbc.distances(a, b, s.cell)[0, 0]), 2))  # fmt: skip

# The same helpers accept (a, b, c, alpha, beta, gamma) and triclinic cells.
tric = (L, L, L, 60.0, 60.0, 90.0)
print("periodic distance in a 60/60/90 triclinic cell:",
      round(float(pbc.distances(a, b, tric)[0, 0]), 2))  # fmt: skip

# All O-O pairs within 3.2 A (cell list, periodic).
oxygens = s.positions[s.select("name O").ids]
i, j, d = pbc.capped_distances(oxygens, oxygens, 3.2, s.cell)
nonself = i != j
print("O-O pairs within 3.2 A:", int(nonself.sum() // 2), "| shortest", d[nonself].min().round(2))

# Break molecules: push the last layer of waters so their H atoms cross the x face,
# then wrap every atom into the box on its own.
shifted = s.positions + [2.6, 0.0, 0.0]
wrapped = shifted - np.floor(shifted / L) * L  # atom-wise wrapping splits molecules
oh = pbc.paired_distances(wrapped[s.bonds["i"]], wrapped[s.bonds["j"]])  # no box: raw lengths
print("longest O-H bond after atom-wise wrapping:", oh.max().round(2), "A (broken molecules)")

whole = boonza.make_whole(s, wrapped, s.cell)
oh = np.linalg.norm(whole[s.bonds["i"]] - whole[s.bonds["j"]], axis=1)
print("after make_whole:", oh.max().round(3), "A")

# Glue: make whole and wrap every molecule into the box around a center.
fix = boonza.Glue(s, center="resid 1")
fixed, box = fix(wrapped, s.cell)
center = fixed[s.select("resid 1").ids].mean(axis=0)
offsets = fixed.reshape(-1, 3, 3).mean(axis=1) - center
print("after Glue, every water is within half a box of residue 1:",
      bool((np.abs(offsets) <= L / 2 + 1e-6).all()))  # fmt: skip

# Periodic selections use the cell too.
print("waters within 3.5 A of residue 1, plain vs periodic:",
      len(s.select("same residue as within 3.5 of resid 1", pos=wrapped).ids) // 3,
      len(s.select("same residue as pbwithin 3.5 of resid 1", pos=wrapped).ids) // 3)  # fmt: skip
