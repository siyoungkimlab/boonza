"""Analyze one structure: surface area, contacts, secondary structure, angles.

Hen egg-white lysozyme (1LYZ).

    python examples/13_structure_analysis.py
"""

import numpy as np
from _common import DATA

import boonza

s = boonza.load(DATA / "1LYZ.pdb")
protein = s.clone("protein")  # crystal waters would bury the surface
print(protein)

# Solvent accessible surface area (Shrake-Rupley, 1.4 A probe).
area = boonza.sasa(protein, mode="residue")[0]
names = [f"{n}{r}" for n, r in zip(protein.residues["name"], protein.residues["resid"],
                                   strict=True)]  # fmt: skip
print(f"total SASA {area.sum():.0f} A^2")
exposed = np.argsort(area)[::-1][:5]
print("most exposed residues:", [(names[k], round(float(area[k]))) for k in exposed])
print("buried residues (< 5 A^2):", int((area < 5).sum()))

# Residue contacts: closest heavy-atom distance for every residue pair.
dist, pairs = boonza.residue_contacts(protein)
close = pairs[dist[0] < 4.0]
print(f"\n{len(close)} residue pairs in contact (< 4 A heavy-atom distance)")
long_range = close[np.abs(close[:, 1] - close[:, 0]) > 20]
print("long-range contacts (> 20 residues apart), first five:",
      [(names[a], names[b]) for a, b in long_range[:5]])  # fmt: skip

# Secondary structure: DSSP and the ChimeraX assignment.
dssp = boonza.dssp(protein)[0]
print("\nDSSP:        ", "".join(c if c.strip() else "-" for c in dssp))
print("ChimeraX H/S:", "".join(boonza.chimerax_ss(protein)))
counts = {c: int((dssp == c).sum()) for c in "HGIEBTS"}
print("DSSP counts:", {k: v for k, v in counts.items() if v})

# Backbone dihedrals: a Ramachandran summary.
phi, psi, omega = (x[0] for x in boonza.backbone_dihedrals(protein))
helical = (phi > -100) & (phi < -30) & (psi > -80) & (psi < -10)
print(f"residues in the helical phi/psi region: {int(helical.sum())}; "
      f"cis peptide bonds: {int((np.abs(omega) < 30).sum())}")  # fmt: skip

# Backbone hydrogen bonds (Kabsch-Sander energies) and disulfides.
donor, acceptor, energy = boonza.backbone_hbonds(protein)
print(f"backbone H-bonds: {len(donor)}, strongest {energy.min():.2f} kcal/mol")
# Disulfides: PDB bonds are guessed from distance with msys's rule (S-S under 2.16 A),
# then the file's SSBOND and CONECT records are applied.  1LYZ is an old 2 A structure
# with stretched S-S geometry: only one S-S is short enough to guess, SSBOND gives all four.
sg = protein.select("name SG").ids
d = boonza.pbc.distances(protein.positions[sg], protein.positions[sg])
for a, b in zip(*[x.tolist() for x in np.nonzero(np.triu(d < 2.5, 1))], strict=True):
    i, j = int(sg[a]), int(sg[b])
    print(f"  CYS{protein.atom(i).residue.resid}-CYS{protein.atom(j).residue.resid}: "
          f"S-S {d[a, b]:.2f} A, bonded: {protein.find_bond(i, j) is not None}")  # fmt: skip
print(f"radius of gyration: {boonza.radius_of_gyration(protein)[0]:.2f} A")
