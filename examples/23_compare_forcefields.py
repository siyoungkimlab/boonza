"""Compare two force fields term by term: viparr's ff19SB and OpenMM's amber19.

boonza parameterizes one structure with a viparr force field and with an
OpenMM XML force field, then compares the two systems.  ``canonical=True``
brings both to one form per interaction first, so only differences that
change the energy remain.

    python examples/23_compare_forcefields.py
"""

import warnings

from _common import amber_system

import boonza

warnings.simplefilter("ignore", boonza.viparr.ViparrWarning)
protein = amber_system("1TEN.pdb")  # 1TEN with hydrogens, prepared with OpenMM

viparr_ff = boonza.parameterize(protein, ["aa.amber.ff19SB"], constraints=False)
openmm_ff = boonza.parameterize_openmm(protein, ["amber19-all.xml"], rigid_water=False)
print(boonza.load_openmm_forcefield("amber19-all.xml"))

# 1. Table by table, in canonical form
print("\nviparr ff19SB vs OpenMM amber19, table by table:")
for d in boonza.diff(viparr_ff, openmm_ff, positions=False, canonical=True):
    print(" ", str(d)[:150])

# 2. Energy per term, through OpenMM
a, b = boonza.openmm_energies(viparr_ff), boonza.openmm_energies(openmm_ff)
dih = {
    k: v.get("dihedral_trig", 0.0) + v.get("dihedral_trig_constant", 0.0)
    for k, v in (("a", a), ("b", b))
}
print("\nenergy differences (kcal/mol):")
for term in ("stretch_harm", "angle_harm", "torsiontorsion_cmap", "nonbonded"):
    print(f"  {term:20s} {a[term] - b[term]:+.4f}")
print(f"  {'dihedrals':20s} {dih['a'] - dih['b']:+.4f}")

# 3. The 1-4 difference is viparr's rounded scale factor (0.8333 for 1/1.2)
ff = boonza.load_forcefield("aa.amber.ff19SB")
ff.rules.es_scale = [0.0, 0.0, 1 / 1.2]
exact = boonza.parameterize(protein, [ff], constraints=False)
left = [
    str(d).split(":")[1].strip()
    for d in boonza.diff(exact, openmm_ff, positions=False, canonical=True)
]
print("\nwith viparr's 1-4 scale set to exactly 1/1.2, what still differs:", "; ".join(left))
gap = boonza.openmm_energies(exact)["nonbonded"] - b["nonbonded"]
print(f"nonbonded difference: {gap:+.4f} kcal/mol")
