"""Small molecules through RDKit: SMILES to 3D, SMARTS selections, bond orders.

Aspirin goes from a SMILES string to a 3D boonza System and back.  SMARTS
patterns select atoms, and bond orders are recovered from geometry alone,
as needed for ligands read from PDB, GRO or mmCIF files.

    python examples/11_rdkit_bridge.py        (needs rdkit)
"""

from _common import OUT, require

require("rdkit")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

import boonza  # noqa: E402

# SMILES -> RDKit 3D -> boonza (aspirin)
mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
AllChem.EmbedMolecule(mol, randomSeed=1)
s = boonza.from_rdkit(mol, name="aspirin")
print(s)
print("bond orders:", sorted(set(s.bonds["order"].tolist())), "| formal charges:",
      int(abs(s.atoms["formal_charge"]).sum()))  # fmt: skip

# SMARTS patterns work inside selections.
for label, smarts in (("aromatic carbons", "c"), ("carboxylic acid", "C(=O)[OH]"),
                      ("ester", "[CX3](=O)[OX2][#6]"), ("ring atoms", "[R]")):  # fmt: skip
    print(f"  {label:17s} {len(s.select(f'smarts {smarts!r}')):3d} atoms")

# Rings found by boonza itself (no RDKit needed for this part).
print("rings:", [len(r) for r in boonza.sssr(s)])

# boonza -> RDKit keeps atom order, names and a back-reference to boonza indices.
back = s.to_rdkit()
print("round-trip SMILES:", Chem.MolToSmiles(Chem.RemoveHs(back)))
print("boonza index of RDKit atom 5:", back.GetAtomWithIdx(5).GetIntProp("boonza_index"))

# Structures from PDB/GRO have no bond orders; perceive them from geometry.
plain = s.copy()
plain.bonds["order"] = [1] * plain.nbonds
plain.atoms["formal_charge"] = [0] * plain.natoms
boonza.assign_bond_orders(plain)
same = Chem.MolToSmiles(Chem.RemoveHs(plain.to_rdkit())) == Chem.MolToSmiles(Chem.RemoveHs(back))
print("bond orders recovered from geometry:", same)

# Write an SDF with a data field, and read it back.
s.ct(0)["source"] = "boonza example"
boonza.save(s, OUT / "aspirin.sdf")
print("SDF data field:", boonza.load(OUT / "aspirin.sdf").ct(0)["source"])
