# Systems: structure, columns, editing

A `boonza.System` holds:

- **structure**: atoms, bonds, residues, chains and cts ("connection
  tables", msys's name for top-level entries such as SDF records or MAE
  blocks);
- **force field**: term tables, parameter tables, `nonbonded_info` and
  auxiliary tables (see [Force fields](forcefield.md));
- **coordinates**: positions, velocities and the unit cell.

The hierarchy is atom → residue → chain → ct. Independently of it, every atom
belongs to one **fragment** (molecule): the atoms it is bonded to, directly
or indirectly.

## Columns

Each level is stored as NumPy columns, one per attribute. `s.atoms`,
`s.bonds`, `s.residues` and `s.chains` behave like dictionaries of arrays:

```python
s.natoms, s.nbonds, s.nresidues, s.nchains, s.ncts
s.atoms["name"]  # names, NumPy string array
s.atoms["anum"]  # atomic numbers (0 for virtual sites / pseudo particles)
s.atoms["charge"], s.atoms["mass"], s.atoms["residue"]
s.residues["resid"], s.residues["name"], s.residues["chain"]
s.chains["name"], s.chains["segid"]
s.bonds["i"], s.bonds["j"], s.bonds["order"]

s.positions  # (natoms, 3) float64 view, Å
s.velocities
s.cell  # (3, 3), rows are box vectors

s.atoms["buried"] = s.atoms["charge"] < 0  # add a column of your own
s.atoms.props  # user-defined columns
```

Built-in atom columns include `name`, `anum`, `mass`, `charge`,
`formal_charge`, `residue`, `pos` and `vel`. Extra columns survive DMS and
MAE round trips. ct properties are free-form key/value pairs:
`s.ct(0)["title"]`.

## Handles

For object-style access, lightweight handles wrap an index:

```python
atom = s.atom(10)
atom.name, atom.element, atom.charge, atom.pos
atom.residue.name, atom.residue.chain.name, atom.ct
atom.bonded_atoms, atom.bonds
atom.fragment  # AtomSel of the whole molecule
res = s.residue(3)
res.atoms, res.select_atom("CA")
chain = s.chain(0)
chain.residues
bond = s.bond(0)
bond.first, bond.second, bond.order
```

Handles are views, not copies. After an edit that renumbers atoms (deleting
or reordering), older handles raise `StaleHandleError`, so fetch fresh ones.

## Atom selections

`s.select(...)` returns an `AtomSel`, an ordered set of atom indices. It
accepts a selection string (see [Selections](selections.md)), indices, a
boolean mask or handles:

```python
sel = s.select("protein and name CA")
sel.ids  # NumPy int array
sel.positions
sel.expand("residue")  # whole residues (also "chain", "ct", "fragment")
sel.fragments  # one AtomSel per molecule touched
s.select(s.atoms["anum"] == 1)
s.all  # every atom
```

## Molecules

```python
s.fragids  # molecule index of every atom (numbered by first atom)
s.nfragments
s.fragment_atoms(5)  # atoms of molecule 5
s.fragments()  # list of AtomSel
groups = s.distinct_fragments()  # {representative: [identical molecules]}
```

`distinct_fragments` groups molecules whose bond graphs are identical, with
atoms matched by element and number of bonds. It gives exactly msys's
`FindDistinctFragments` result and is fast: 0.3 s for 333,000 molecules
(1 M atoms).

## Building and editing

```python
s = boonza.System("water")
res = s.add_residue(name="HOH", resid=1)  # creates a ct and a chain as needed
o = res.add_atom(name="O", anum=8, mass=15.999, charge=-0.834, pos=(0.0, 0.0, 0.0))
h1 = res.add_atom(name="H1", anum=1, mass=1.008, charge=0.417, pos=(0.957, 0.0, 0.0))
h2 = res.add_atom(name="H2", anum=1, mass=1.008, charge=0.417, pos=(-0.240, 0.927, 0.0))
s.add_bond(o, h1)
s.add_bond(o, h2)

# many at once (fast): columns are arrays, one value per atom
new = s.add_atoms(1000, residue=res, name="X", anum=6, pos=xyz)
s.add_bonds(pairs, order=1)

# delete: bonds and force-field terms of removed atoms go too; empty residues/chains are pruned
old_to_new = s.delete_atoms("water and not within 5 of protein")  # -1 for removed atoms
s.delete_bonds([0, 1])
s.reorder_atoms(order)  # new atom k is old atom order[k]

sub = s.clone("protein or resname LIG")  # copy of a subset; terms fully inside are kept
s.append(other)  # merge another system, force field included
s.copy()
s.guess_bonds(periodic=True)  # msys bond-guessing rules
```

`clone` keeps the atoms in the order you give. `append` merges parameter
tables and checks that nonbonded settings (functional form, combining rule)
are compatible.

Editing a parameter through a term is **copy-on-write**. Changing one term's
`fc` does not change other terms that shared the parameter row (see
[Force fields](forcefield.md)).

Batches and whole systems from arrays:

```python
chains = s.add_chains(2, name=["A", "B"], segid="PROT")
residues = s.add_residues(
    4, chain=[chains[0]] * 2 + [chains[1]] * 2, name="ALA", resid=[1, 2, 1, 2]
)
s.add_atoms(8, residue=np.repeat(residues, 2), name=["N", "CA"] * 4, anum=[7, 6] * 4)

s = boonza.System.from_arrays(
    positions,
    names=names,
    elements=elements,
    resnames=resnames,
    resids=resids,
    chains=chains,
    bonds=pairs,
    charge=charges,
)
```

`from_arrays` groups residues and chains the way a PDB reader does. Elements
come from `anum`, or from `elements` symbols, or are guessed from the names.

## Tables as pandas DataFrames

```python
df = s.to_pandas()  # one row per atom: name, element, x, y, z, resid, resname, chain, fragid, ...
df = s.to_pandas("resname LIG")
s.atoms.to_pandas()  # the raw atom table (pos as pos_x, pos_y, pos_z)
s.residues.to_pandas()
s.table("stretch_harm").to_pandas()  # terms: atom0, atom1, param, r0, fc, ...
```

## Viewing in a notebook

```python
s.view()  # cartoon for polymers, sticks for ligands, spheres for ions, water hidden
s.view("protein or resname LIG", style="sticks")
boonza.view(s, positions=frames[::10])  # an animation
boonza.view(s).save("complex.html")  # a standalone page
```

Views are drawn with 3Dmol.js, which the page loads from a CDN, so no Python
package is needed.

## Secondary structure and rings

```python
boonza.dssp(s)  # DSSP codes per residue (mdtraj DSSP 2.2)
boonza.backbone_dihedrals(s)  # phi, psi, omega per residue
boonza.sssr(s)  # smallest set of smallest rings, as msys
```

See [Analysis](analysis.md).
