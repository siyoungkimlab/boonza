"""Catch problems before simulating: validate, find_knots, diff.

We take a clean parameterized protein, break it in several typical ways,
and let ``validate`` find each problem.  Then a hand-made "knot": a bond
threaded through a benzene ring.

    python examples/12_validate_and_diff.py
"""

import numpy as np  # noqa: E402
from _common import amber_system

import boonza  # noqa: E402

clean = amber_system("1TEN.pdb")
clean.cell = np.eye(3) * 60.0  # give it a box so the box check passes
print("clean system, basic checks:", boonza.validate(clean) or "no problems")

broken = clean.copy()
broken.table("stretch_harm").delete_terms([0])  # a bond loses its stretch term
q = broken.atoms["charge"].copy()
q[10] += 0.25  # a charge typo
broken.atoms["charge"] = q
ca = broken.select("name CA").ids
broken.table("exclusion").add_terms([[int(ca[0]), int(ca[-1])]])  # a stray exclusion
pos = broken.positions
pos[ca[5]] = pos[ca[6]] + 0.3  # two atoms on top of each other
broken.positions = pos

print("\nbroken system, strict checks:")
for p in boonza.validate(broken, strict=True):
    if p.check in ("constraints", "constrained_hydrogens"):
        continue  # this example system has no constraints by design
    print(f"  [{p.check}] {p.message[:100]}")

print("\nwhat differs between the clean and broken systems:")
for d in boonza.diff(clean, broken):
    print("  " + str(d)[:110])

# A knot: a C-C bond passing through the middle of a benzene ring.
s = boonza.System("knot")
angles = np.radians(np.arange(6) * 60.0)
ring = np.column_stack([1.39 * np.cos(angles), 1.39 * np.sin(angles), np.zeros(6)])
s.add_atoms(8, name=[f"C{k}" for k in range(8)], anum=6, mass=12.011,
            pos=np.vstack([ring, [[0.0, 0.0, -0.77], [0.0, 0.0, 0.77]]]))  # fmt: skip
s.add_bonds([[k, (k + 1) % 6] for k in range(6)] + [[6, 7]])
for ring_atoms, bond, _ in boonza.find_knots(s):
    print(f"\nknot: bond {bond} passes through ring {list(ring_atoms)}")
