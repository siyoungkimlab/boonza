"""From text to 3D: a molecule from SMILES and a peptide from its sequence.

One line of text becomes a physically sensible structure (RDKit embedding
and an MMFF94 relaxation), and boonza's own tools read it back to check that
it is what was asked for: the same SMILES, the same sequence, the helix that
was requested.

    python examples/16_build_from_text.py
"""

import numpy as np
from _common import OUT, require

require("rdkit")

from rdkit import Chem  # noqa: E402

import boonza  # noqa: E402

# A small molecule: the lowest-energy of five MMFF94-relaxed conformers.
aspirin = boonza.from_smiles("CC(=O)Oc1ccccc1C(=O)O", name="AIN", conformers=5, seed=1)
print(f"aspirin: {aspirin.natoms} atoms, {aspirin.nbonds} bonds, "
      f"atoms named {', '.join(aspirin.atoms['name'][:5].tolist())}, ...")  # fmt: skip
print("read back as SMILES:", Chem.MolToSmiles(Chem.RemoveHs(aspirin.to_rdkit())))
ch = [np.linalg.norm(aspirin.positions[i] - aspirin.positions[j])
      for i, j in zip(aspirin.bonds["i"], aspirin.bonds["j"], strict=True)
      if {aspirin.atoms["anum"][i], aspirin.atoms["anum"][j]} == {1, 6}]  # fmt: skip
print(f"C-H bond lengths {min(ch):.3f} to {max(ch):.3f} Å")
boonza.save(aspirin, OUT / "aspirin.sdf")

# A peptide: every peptide bond trans, phi/psi set, side chains relaxed with MMFF94.
seq = "AEAAAKEAAAKA"
helix = boonza.peptide(seq, "helix")
phi, psi, omega = (x[0] for x in boonza.backbone_dihedrals(helix))
print(f"\npeptide {seq}: {helix.natoms} atoms in {helix.nresidues} residues")
print("sequence read back:", boonza.sequence(helix))
print("phi of the inner residues:", np.round(phi[1:-1]).astype(int).tolist())
print("psi of the inner residues:", np.round(psi[1:-1]).astype(int).tolist())
print("DSSP of the helix:  ", "".join(boonza.dssp(helix, simplified=True)[0]))
strand = boonza.peptide(seq, "sheet")
print("DSSP of the strand: ", "".join(boonza.dssp(strand, simplified=True)[0]))
boonza.save(helix, OUT / "helix.pdb")
print("\nwrote", OUT / "aspirin.sdf", "and", OUT / "helix.pdb")
