"""Superpose proteins: ChimeraX-style matchmaker, PyMOL-style cealign, plain fits.

* myoglobin (1MBN) onto hemoglobin alpha (1HHO chain A): ~25% sequence
  identity, so a sequence alignment guided by secondary structure is needed;
* fibronectin FN3 domains 1FNA onto 1TEN: remote homologs, where the
  structure-only CE method shines.

    python examples/07_superposition.py
"""

from _common import DATA, OUT

import boonza

hemo = boonza.load(DATA / "1HHO.pdb")
myo = boonza.load(DATA / "1MBN.pdb")

# matchmaker moves `myo` onto `hemo` and reports what it paired.
r = boonza.matchmaker(myo, hemo, reference_chain="A")
print("matchmaker, myoglobin -> hemoglobin alpha")
print(f"  {len(r.kept)} of {len(r.mobile_atoms)} aligned residue pairs kept, RMSD {r.rmsd:.3f} A "
      f"(all pairs {r.full_rmsd:.3f} A)")  # fmt: skip
print("  reference:", r.aligned_reference[:60])
print("  mobile:   ", r.aligned_mobile[:60])
boonza.save(myo, OUT / "1MBN_on_1HHO.pdb")

# Without choosing chains, every chain pair is tried and the best one is used.
best = boonza.matchmaker(boonza.load(DATA / "1MBN.pdb"), hemo, apply=False)
print("  best-scoring hemoglobin chain:", hemo.chains["name"][best.reference_chain])

# A naive identity-only alignment pairs far fewer residues.
naive = boonza.superpose(boonza.load(DATA / "1MBN.pdb"), hemo, ref_sel="chain A and name CA",
                         apply=False)  # fmt: skip
print(f"  (identity-only alignment: {naive.n_used} of {naive.n_matched} pairs, "
      f"RMSD {naive.rmsd:.3f} A)")  # fmt: skip

# cealign needs no sequence similarity at all.
ten, fna = boonza.load(DATA / "1TEN.pdb"), boonza.load(DATA / "1FNA.pdb")
ce = boonza.cealign(fna, ten)
print("\ncealign, 1FNA -> 1TEN")
print(f"  {len(ce.mobile_atoms)} C-alpha pairs, RMSD {ce.rmsd:.3f} A, z-score {ce.z_score:.2f}")
seq_id = boonza.align_sequences(boonza.sequence(ten), boonza.sequence(fna)).identity
print(f"  sequence identity only {100 * seq_id:.0f}%")
boonza.save(fna, OUT / "1FNA_on_1TEN.pdb")

# The same number by hand: a Kabsch fit of the pairs matchmaker kept, then the RMSD.
mob_xyz = boonza.load(DATA / "1MBN.pdb").positions[r.mobile_atoms[r.kept]]
ref_xyz = hemo.positions[r.reference_atoms[r.kept]]
R, t = boonza.kabsch(mob_xyz, ref_xyz)  # mob_xyz @ R.T + t best matches ref_xyz
print(f"\nby hand: RMSD {boonza.rmsd(mob_xyz, ref_xyz):.1f} A before fitting, "
      f"{boonza.rmsd(mob_xyz @ R.T + t, ref_xyz):.3f} A after the Kabsch fit")  # fmt: skip
