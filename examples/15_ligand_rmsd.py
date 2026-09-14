"""Symmetry-corrected ligand RMSD: equivalent atoms should not count as errors.

A plain RMSD pairs atoms by name or order.  When a ligand has equivalent
atoms (the two oxygens of a carboxylate, a flipped phenyl ring, the methyls
of a t-butyl group) the same pose can get a large RMSD just because atoms
were named the other way round.  ``symmetry_rmsd`` takes the smallest RMSD
over every mapping that preserves elements and bonds.  ``ligand_rmsd`` is
the docking workflow: superpose the proteins, then compare the ligands.
``drmsd`` needs no fitting: it compares pocket-ligand distances.

Here: the two hemes of hemoglobin, in the alpha and the beta subunit, after
superposing the subunits.

    python examples/15_ligand_rmsd.py
"""

import numpy as np
from _common import DATA

import boonza

hb = boonza.load(DATA / "1HHO.pdb")

# Superpose the beta subunit onto the alpha subunit (different sequences: matchmaker).
beta_on_alpha = hb.copy()
fit = boonza.matchmaker(beta_on_alpha, hb, mobile_chain="B", reference_chain="A")
print(f"beta onto alpha: {len(fit.kept)} C-alpha pairs, RMSD {fit.rmsd:.2f} A")

# Heme of beta (moved) vs heme of alpha, in place: a pose comparison.
r = boonza.symmetry_rmsd(beta_on_alpha, hb, atoms="resname HEM and chain B",
                         reference_atoms="resname HEM and chain A")  # fmt: skip
print(f"heme RMSD, atoms paired by order:  {r.plain_rmsd:.3f} A")
print(f"heme RMSD, symmetry-corrected:     {r.rmsd:.3f} A")

swapped = [(hb.atom(int(a)).name, beta_on_alpha.atom(int(b)).name)
           for a, b in zip(r.reference_atoms, r.mapping, strict=True)
           if hb.atom(int(a)).name != beta_on_alpha.atom(int(b)).name]  # fmt: skip
print("atoms matched to a differently named partner:", swapped)

# Two files can name equivalent atoms the other way round.  Swap the names of the
# carboxylate oxygens of both propionates (O1A <-> O2A, O1D <-> O2D) in the beta heme.
renamed = beta_on_alpha.copy()
pos = renamed.positions
for a, b in (("O1A", "O2A"), ("O1D", "O2D")):
    i = renamed.select(f"resname HEM and chain B and name {a}").ids
    j = renamed.select(f"resname HEM and chain B and name {b}").ids
    pos[i], pos[j] = pos[j].copy(), pos[i].copy()
renamed.positions = pos
swap = boonza.symmetry_rmsd(renamed, hb, atoms="resname HEM and chain B",
                            reference_atoms="resname HEM and chain A")  # fmt: skip
print(f"\noxygen names swapped: plain {swap.plain_rmsd:.3f} A, "
      f"symmetry-corrected {swap.rmsd:.3f} A (unchanged)")  # fmt: skip

# The answer does not depend on atom order: shuffle the beta heme's atoms.
shuffled = beta_on_alpha.copy()
heme_b = shuffled.select("resname HEM and chain B").ids
order = np.arange(shuffled.natoms)
order[heme_b] = np.random.default_rng(0).permutation(heme_b)
shuffled.reorder_atoms(order)
again = boonza.symmetry_rmsd(shuffled, hb, atoms="resname HEM and chain B",
                             reference_atoms="resname HEM and chain A")  # fmt: skip
print(f"after shuffling the atom order: symmetry-corrected {again.rmsd:.3f} A "
      f"(plain RMSD no longer meaningful: {again.plain_rmsd})")  # fmt: skip

# Shape only (conformer comparison): superpose the two hemes themselves.
shape = boonza.symmetry_rmsd(beta_on_alpha, hb, atoms="resname HEM and chain B",
                             reference_atoms="resname HEM and chain A", superpose=True)  # fmt: skip
print(f"heme shape difference (fitted): {shape.rmsd:.3f} A over {shape.isomorphisms} "
      "equivalent atom mappings")  # fmt: skip


def random_rotation(rng):
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# The docking workflow in one call.  Here the "docked" system is the crystal
# structure moved somewhere else, so the ligand RMSD should be zero.
rng = np.random.default_rng(1)
docked = hb.copy()
docked.positions = docked.positions @ random_rotation(rng).T + [40.0, -10.0, 5.0]
res = boonza.ligand_rmsd(docked, hb, ligand="resname HEM and chain A")
print(f"\nligand_rmsd: protein fit {res.fit_rmsd:.2e} A, heme RMSD {res.rmsd:.2e} A")

# Trajectories: pass frames (an array, a Frames block, or an open trajectory,
# e.g. boonza.open_trajectory("md.xtc", system)).  Every frame is fitted on the
# protein on its own, the heme moves with it, and the RMSDs come back per frame.
frames = np.stack([hb.positions @ random_rotation(rng).T + rng.normal(0, 10, 3)
                   + rng.normal(0, 0.1 * k, hb.positions.shape) for k in range(6)])  # fmt: skip
per = boonza.ligand_rmsd(hb, hb, ligand="resname HEM and chain A", positions=frames)
print("frames of a tumbling, increasingly jiggled hemoglobin:")
print("  protein fit RMSD (A):", np.round(per.fit_rmsd, 2))
print("  heme RMSD (A):       ", np.round(per.rmsd, 2))

# dRMSD: the RMS change of every pocket-atom-to-ligand-atom distance, no fitting.
# The pocket is the C-alpha atoms within 5 A of the ligand in the reference (the
# first frame by default).  Equivalent ligand atoms are matched in each frame.
d = boonza.drmsd(hb, ligand="resname HEM and chain A", positions=frames)
print(f"\ndRMSD over {len(d.pocket)} pocket C-alphas and {len(d.reference_ligand)} heme atoms:")
print("  symmetry-corrected (A):", np.round(d.drmsd, 2))
print("  atoms in order (A):    ", np.round(d.plain_drmsd, 2))
