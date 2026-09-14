# Summaries for AI

A language model reads text, not coordinates. `boonza.summarize` states
in words and numbers what a 3D structure holds, so a model (or a person)
can reason about it without the atom list:

```python
import boonza

s = boonza.load("1HHO.pdb")
print(boonza.summarize(s))  # Markdown text
print(boonza.summarize(s, focus="resname HEM and chain A"))  # one site in detail
data = boonza.summarize(s).to_dict()  # the same facts as plain data (JSON-ready)
```

```bash
boonza summarize 1HHO.pdb --focus "resname HEM and chain A"
boonza summarize 1HHO.pdb --json
```

## What a summary contains

- **Composition**: the numbers of atoms, residues, chains and molecules; the
  periodic box; the polymer chains; the numbers of waters and ions; other
  molecules by residue name; the net formal charge; the force-field tables
  and net partial charge, if there is a force field.
- **Chains**: for each polymer chain, its residue range, its one-letter
  sequence, and its secondary structure as a DSSP string with helix and
  strand fractions. Also chain breaks (C-N over 2.5 Å) and disulfide bonds.
- **Other molecules and their sites**: for each ligand, cofactor or other
  molecule:
  - its Hill formula, heavy atoms and charge, and its SMILES (with RDKit)
    when the file gives hydrogens or bond orders; without them a SMILES
    would describe a different, saturated molecule, so none is given;
  - the polymer residues within the cutoff (4 Å), closest first, with the
    closest atom pair;
  - other molecules and waters nearby;
  - bonds to the rest (covalent links, metal coordination);
  - polar contacts;
  - how much of its solvent-accessible surface is buried.
- **Contacts between chains**: residue pairs within the cutoff for each pair
  of chains.
- **Focus**: the same site description for any selection you name.
- **Checks**: clashes (nonbonded heavy atoms closer than 2.2 Å and more
  than two bonds apart) and the problems `boonza.validate` finds.

Every number comes from the structure. Crystal structures usually have no
hydrogens, so hydrogen bonds cannot be tested by angle. The summary then
reports polar contacts (N/O pairs within 3.5 Å) and says so. With
`boonza.hbonds` you can compute real hydrogen bonds once hydrogens are
present.

## Why this matters

With `from_smiles` and `peptide` (text to 3D), `load` (files to a graph and
positions) and `summarize` (3D back to text), a model can go from a
description to a structure and back. It can check that what it built is
what it meant: the right sequence, the helix it asked for, the ligand in
the pocket, no clashes.
