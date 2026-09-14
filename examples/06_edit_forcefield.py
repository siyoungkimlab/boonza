"""Edit force-field parameters: copy-on-write terms, NBFIX overrides, exclusions.

Many terms share one parameter row.  Changing a parameter through a term
copies the row first, so only that term changes.  Pair-specific Lennard-
Jones overrides (NBFIX) change one type pair.  ``diff`` shows exactly what
changed, and ``update_exclusions`` rebuilds exclusions and 1-4 pairs from
the bonds.

    python examples/06_edit_forcefield.py
"""

import numpy as np  # noqa: E402
from _common import amber_system

import boonza  # noqa: E402

original = amber_system("1TEN.pdb")
s = original.copy()

# --- copy-on-write: change one bond, not every bond of that type ----------
stretch = s.table("stretch_harm")
same_type = np.flatnonzero(stretch.param_ids == stretch.param_ids[0])
print(f"stretch term 0 shares its parameter row with {len(same_type) - 1} other terms")
print("fc before:", stretch[0]["fc"])
stretch[0]["fc"] = 500.0
fc = stretch.values("fc")
print(f"fc after:  term 0 = {fc[0]}, term {same_type[1]} (same type) = {fc[same_type[1]]}")

# --- NBFIX: a special Lennard-Jones pair between two atom types ------------
nb = s.table("nonbonded")
o_type = int(nb.param_ids[s.select("name O").ids[0]])  # backbone carbonyl O
n_type = int(nb.param_ids[s.select("name N").ids[0]])  # backbone amide N
print(f"\nLJ type of O: {nb.params.row(o_type)}")
print(f"LJ type of N: {nb.params.row(n_type)}")
nb.overrides.set(o_type, n_type, sigma=2.9, epsilon=0.25)
print("override stored:", nb.overrides.get(o_type, n_type))

pairs = s.describe("resid 810 and name N O", pairs=True).pairs
for p in pairs:
    print(f"  {p['label_i']} - {p['label_j']}: sigma {p['sigma']:.3f}, "
          f"epsilon {p['epsilon']:.4f}, nbfix {p['nbfix']}")  # fmt: skip

# --- what changed?  diff compares every table as a set of terms -------------
for d in boonza.diff(original, s):
    print("diff:", str(d)[:100])
e0, e1 = boonza.openmm_energies(original), boonza.openmm_energies(s)
print(f"total energy: {e0['total']:.2f} -> {e1['total']:.2f} kcal/mol")

# --- exclusions and 1-4 pairs rebuilt from bonds ----------------------------
rebuilt = original.copy()
rebuilt.del_table("exclusion")
rebuilt.del_table("pair_12_6_es")
boonza.update_exclusions(rebuilt, pair_scales=(0.5, 1 / 1.2))  # Amber 1-4 scaling
changes = boonza.diff(original, rebuilt, rtol=1e-5, tables=["exclusion", "pair_12_6_es"])
print("\nrebuilt exclusions and 1-4 pairs vs original:", changes or "identical")
