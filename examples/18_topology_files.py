"""Read simulation topologies: a GROMACS .top with its .gro coordinates.

examples/data/gmx_amber holds a GROMACS topology that ParmEd wrote from an
Amber system (a peptide in 876 waters).  Amber prmtop and CHARMM PSF files
load the same way (``boonza.load("x.prmtop", coordinates="x.rst7")``,
``boonza.load("x.psf", coordinates="x.pdb")``).

    python examples/18_topology_files.py
"""

import importlib.util

import numpy as np
from _common import DATA

import boonza

top = DATA / "gmx_amber"
s = boonza.load(top / "topol.top", coordinates=top / "conf.gro")
print(s)
print("force-field tables:", ", ".join(sorted(s.tables)))
info = s.nonbonded_info
print(f"van der Waals: {info.vdw_funct}, combining rule {info.vdw_rule}")

# #ifdef blocks follow the defines: rigid water (settles) by default.
flexible = boonza.load(top / "topol.top", defines={"FLEXIBLE": ""})
print(f"rigid water: {'constraint_hoh' in s.tables}; with FLEXIBLE defined, "
      f"{flexible.nbonds - s.nbonds} more bonds (one H-H per water)")  # fmt: skip

# The force field of a few atoms, with each pair's bonds apart and bare energy.
report = boonza.describe(s, "resid 1 and name N CA C O", pairs=True)
print("\npairs in the first residue's backbone:")
for p in report.pairs[:5]:
    print(f"  {p['label_i']:>12} - {p['label_j']:<12} {p['bonds']} bonds apart, "
          f"r {p['r']:.2f} Å, energy {p['energy']:+.3f} kcal/mol")  # fmt: skip
d = boonza.topological_distances(s, "resid 1 and name N", "resid 1 to 4 and name CA")
print("bonds from residue 1's N to the first four CA atoms:", d[0].tolist())

if importlib.util.find_spec("openmm"):
    e = boonza.openmm_energies(s, nonbonded_method="NoCutoff", constraints=False)
    print(f"\npotential energy (OpenMM, no cutoff): {e['total']:.1f} kcal/mol")
    print("by table:", {k: round(v, 1) for k, v in e.items() if k != "total"})
else:
    print("\n(install OpenMM to compute the energy)")
print(f"net charge {s.atoms['charge'].sum():+.3f} e over {np.unique(s.fragids).size} molecules")
