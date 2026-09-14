"""Build a simulation box: solvate a protein, add salt, repartition hydrogen masses.

These follow msys's dms-solvate, dms-neutralize and dms-hmr tools step by
step and give the same systems.  The water is the TIP3P box that ships
with msys and boonza.

    python examples/17_solvate_and_ions.py
"""

import numpy as np
from _common import DATA, OUT, water_box

import boonza

protein = boonza.load(DATA / "1LYZ.pdb").clone("protein")
box = boonza.solvate(protein, thickness=8.0)
waters = len(box.select("water and oxygen"))
print(f"lysozyme in water: {box.natoms} atoms, {waters} waters, "
      f"box {' x '.join(f'{x:.1f}' for x in np.diag(box.cell))} Å")  # fmt: skip
oxygens = box.positions[box.select("water and oxygen").ids]
solute = box.positions[box.select("protein").ids]
closest = boonza.pbc.distances(oxygens, solute, box.cell).min()
print(f"closest water oxygen to the protein: {closest:.2f} Å (solvate keeps at least 2.4 Å)")
print("water chains:", ", ".join(box.chains["name"][1:4].tolist()), "...")

# The PDB file carries no charges, so only the 150 mM salt is added here; with a
# force field (charge="charge") the counterions for the protein's charge come first.
salted = boonza.neutralize(box, concentration=0.15, random_seed=1)
na, cl = len(salted.select("name Na")), len(salted.select("name Cl"))
print(f"\nafter neutralize: {na} Na+ and {cl} Cl- replace waters; "
      f"{len(salted.select('water and oxygen'))} waters left")  # fmt: skip
boonza.save(salted, OUT / "lysozyme_solvated.pdb")

# Hydrogen mass repartitioning keeps the total mass: hydrogens get 3.024 u,
# taken from the atom each is bonded to.
w = water_box(3)
hmr = boonza.repartition_hydrogen_masses(w, "all", 3.024)
print(f"\nwater masses before: {w.atoms['mass'][:3].round(3).tolist()}, "
      f"after: {hmr.atoms['mass'][:3].round(3).tolist()}")  # fmt: skip
print(f"total mass {w.atoms['mass'].sum():.3f} -> {hmr.atoms['mass'].sum():.3f} u")
print("\nwrote", OUT / "lysozyme_solvated.pdb")
