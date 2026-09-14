"""Parameterize a protein with OpenMM's Amber14 force field and save it as DMS.

OpenMM builds the force field; ``boonza.from_openmm`` turns it into DMS
tables (stretches, angles, dihedrals, 1-4 pairs, Lennard-Jones, charges).
Then we inspect the tables, report the parameters of a few atoms, check
the system, compute energies, and make sure a DMS round trip is lossless.

    python examples/05_parameterize_with_openmm.py
"""

from _common import OUT, amber_system

import boonza  # noqa: E402

s = amber_system("1TEN.pdb")  # waters removed, hydrogens added, Amber14 parameters
print(s)
print("nonbonded:", s.nonbonded_info.vdw_funct, s.nonbonded_info.vdw_rule)
for name, table in sorted(s.tables.items()):
    print(f"  {name:14s} {len(table):6d} terms  {len(table.params):5d} parameter sets")

# Parameters of the backbone of residue 810, with atom labels and energy formulas.
report = s.describe("resid 810 and name N CA")
text = str(report).splitlines()
print("\n".join(text[:14]))
print("  ...")

# Basic checks (example 12 shows what they catch).
print("validate:", boonza.validate(s) or "no problems")

# Energies of each table through OpenMM, in kcal/mol.
for name, value in boonza.openmm_energies(s).items():
    print(f"  E[{name}] = {value:10.2f}")

# Save as DMS and confirm nothing changes on reading it back.
path = OUT / "1TEN_amber14_copy.dms"
boonza.save(s, path)
print("DMS round trip differences:", boonza.diff(s, boonza.load(path)))
