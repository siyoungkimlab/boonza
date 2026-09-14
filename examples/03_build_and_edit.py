"""Build a system from scratch, edit it, and write it in several formats.

Covers: adding residues, atoms and bonds (one at a time and in bulk),
custom columns, deleting atoms (and the old-to-new index map), stale
handles, cloning a subset, appending systems, and file round trips.

    python examples/03_build_and_edit.py
"""

import numpy as np
from _common import OUT, water_box

import boonza

# --- one molecule, atom by atom --------------------------------------------
m = boonza.System("methanol")
res = m.add_residue(name="MOH", resid=1)  # creates a ct and a chain as needed
c = res.add_atom(name="C", anum=6, mass=12.011, pos=(0.0, 0.0, 0.0))
o = res.add_atom(name="O", anum=8, mass=15.999, pos=(1.43, 0.0, 0.0))
h = res.add_atom(name="H", anum=1, mass=1.008, pos=(1.75, 0.9, 0.0))
m.add_bond(c, o)
m.add_bond(o, h)
print(m, "| bonds:", m.bonds["i"].tolist(), m.bonds["j"].tolist())

# --- many molecules at once: a water box (see _common.water_box) ------------
box = water_box(n=5)  # 125 waters, built with add_atoms/add_bonds in bulk
print(box, "| molecules:", box.nfragments, "| cell:", np.diag(box.cell))

# Custom columns live next to the built-in ones.
box.atoms["depth"] = box.positions[:, 2]
print("custom atom columns:", box.atoms.props)

# --- deleting atoms renumbers them ----------------------------------------------
keep_me = box.atom(100)
old_to_new = box.delete_atoms("same residue as x < 2")  # whole waters: first layer
print("after deleting the first layer of waters:", box.natoms, "atoms,", box.nfragments,
      "molecules")  # fmt: skip
print("old atom 100 is now atom", old_to_new[100], "| deleted atoms map to -1:",
      int((old_to_new == -1).sum()))  # fmt: skip
try:
    print(keep_me.name)  # the handle was made before the deletion
except boonza.StaleHandleError as e:
    print("stale handle caught:", e)
fresh = box.atom(int(old_to_new[100]))
print("fresh handle:", fresh.name, fresh.residue.resid)

# --- cloning, appending ------------------------------------------------------
first10 = box.clone("resid 26 to 35")  # a self-contained copy of part of the system
print("clone of 10 waters:", first10.natoms, "atoms")
mixture = first10.copy()
m.positions = m.positions + [0.0, 0.0, 20.0]  # place it clear of the waters
new_ids = mixture.append(m)  # add the methanol; returns its new atom indices
print("water + methanol:", mixture.natoms, "atoms; methanol atoms are", new_ids.tolist())

# --- writing and reading back ----------------------------------------------------
for suffix in (".dms", ".pdb", ".gro", ".cif", ".mae"):
    path = OUT / f"mixture{suffix}"
    boonza.save(mixture, path)
    back = boonza.load(path)
    print(f"{suffix:5s} round trip: {back.natoms} atoms, {back.nbonds} bonds")

# DMS keeps custom columns exactly; PDB bonds are re-guessed from geometry.
back = boonza.load(OUT / "mixture.dms")
print("depth column survived in DMS:", "depth" in back.atoms.props)
