"""viparr force fields: parameterize, set priorities, patch, and handle D residues.

viparr force fields are directories of JSON files; DESRES's viparr-ffpublic
has about 70 (Amber, CHARMM, DES-Amber, lipids, nucleic acids, ions, waters),
and boonza ships a copy.  boonza parameterizes with them as viparr does:
residues match templates by their bond graph, and every molecule takes the
first force field that matches it.

    python examples/22_viparr_forcefields.py
"""

import warnings

import numpy as np
from _common import OUT, amber_system, water_box

import boonza

# The viparr-ffpublic force fields ship with boonza; $VIPARR_FFPATH can add others
names = boonza.viparr.list_forcefields()
print(f"{len(names)} force fields, e.g. {', '.join(names[:3])}, ...")
print("bundled:", boonza.viparr.bundled_version())

# A peptide (RDKit lists hydrogens last; keep each residue's atoms together)
pep = boonza.peptide("AVLSKEF", conformation="helix")
pep = pep.clone(np.argsort(pep.atoms["residue"], kind="stable"))
waters = water_box(3)
waters.positions = waters.positions + [25.0, 0.0, 0.0]
system = pep.copy()
system.append(waters)

# 1. Parameterize: the peptide with CHARMM36m, the waters with TIP3P
c36m = boonza.load_forcefield("aa.charmm.c36m")
print(c36m)
p = boonza.parameterize(system, [c36m, "water.tip3p_charmm"])
print(", ".join(f"{n} {len(p.table(n))}" for n in p.table_names))
energy = boonza.openmm_energies(p)
print(f"energy {energy['total']:.1f} kcal/mol, CMAP {energy['torsiontorsion_cmap']:.2f}")
boonza.save(p, OUT / "peptide_c36m.dms")

# 2. Priority: both Amber force fields match a protein; the first one listed wins.
#    (Amber has charged termini only, so use 1TEN with hydrogens from OpenMM.)
protein = amber_system("1TEN.pdb")
for order in (["aa.amber.ff14SB", "aa.amber.ff99SB"], ["aa.amber.ff99SB", "aa.amber.ff14SB"]):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        q = boonza.parameterize(protein, order)
    dih = boonza.openmm_energies(q)["dihedral_trig"]
    print(f"{' then '.join(order)}: dihedral energy {dih:.2f}; warning: {caught[0].message}")

# 3. Patching: ff99SB with ff99SB-ILDN's side-chain torsions (viparr's -m)
ildn = boonza.merge_forcefields("aa.amber.ff99SB", "aa.amber.ff99SB-ILDN")
e = boonza.openmm_energies(boonza.parameterize(protein, [ildn]))["dihedral_trig"]
print(f"ff99SB patched with ILDN: dihedral energy {e:.2f}")

# 4. D amino acids: the mirror image of the peptide should have the same energy
mirror = pep.copy()
mirror.positions = pep.positions * [-1.0, 1.0, 1.0]
L = boonza.openmm_energies(boonza.parameterize(pep, [c36m]))["torsiontorsion_cmap"]
for chiral in (True, False):
    D = boonza.openmm_energies(boonza.parameterize(mirror, [c36m], cmap_chirality=chiral))
    how = "mirrored CMAP for D residues" if chiral else "L CMAP for D residues (viparr)"
    print(f"CMAP energy: L peptide {L:.3f}, D peptide with {how} {D['torsiontorsion_cmap']:.3f}")
