"""Load a structure and look around: hierarchy, columns, selections, sequences.

Hemoglobin (PDB 1HHO) has two protein chains, heme groups and crystal
waters.  Everything boonza knows about it is stored as NumPy columns, and
handles let you walk from an atom to its residue, chain and molecule.

    python examples/01_load_and_inspect.py
"""

import numpy as np
from _common import DATA

import boonza

s = boonza.load(DATA / "1HHO.pdb")
print(s)
print("chains:", s.chains["name"].tolist(), "(protein chains A and B, then their waters)")
print("residues:", s.nresidues, "| molecules:", s.nfragments)

# Every attribute is a NumPy column: slice, mask and compute on it directly.
print("first atom names:", s.atoms["name"][:6].tolist())
print("atomic numbers present:", sorted(set(s.atoms["anum"].tolist())))
print("residue names (first 8):", s.residues["name"][:8].tolist())

# Walk the hierarchy from one atom with handles.
atom = s.atom(100)
res = atom.residue
print(f"atom 100: {atom.name} ({atom.element}) in {res.name}{res.resid}, chain {res.chain.name}")
print(f"  bonded to {[s.atom(i).name for i in atom.bonded_atoms]}; its molecule has "
      f"{len(atom.fragment)} atoms")  # fmt: skip

# Selections use the msys/VMD language and return AtomSel objects.
ca = s.select("protein and name CA")
heme = s.select("resname HEM")
print("C-alpha atoms:", len(ca))
print("heme atoms:", len(heme), "in", len(heme.residues), "heme residues")
# Bond guessing (msys rules) links each heme iron to a histidine, so the hemes
# are part of the protein molecules here; example 04 shows how to separate them.
iron = s.select("element Fe").ids
partners = [s.atom(int(j)) for i in iron for j in s.atom(int(i)).bonded_atoms]
print("iron bonded to:", sorted({f"{a.residue.name}{a.residue.resid}:{a.name}" for a in partners
                                 if a.residue.name != "HEM"}))  # fmt: skip
print("waters within 3.5 A of a heme:", len(s.select("water and within 3.5 of resname HEM")))

# Sequences of the protein chains.
for chain in ("A", "B"):
    seq = boonza.sequence(s, chain)
    print(f"chain {chain}: {len(seq)} residues  {seq[:30]}...")

# Geometry is plain NumPy on the positions (Angstrom).
center = s.positions[ca.ids].mean(axis=0)
extent = np.ptp(s.positions[ca.ids], axis=0)
print("protein center:", np.round(center, 1), " extent:", np.round(extent, 1))
