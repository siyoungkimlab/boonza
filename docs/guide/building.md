# Building systems

Molecules and peptides can be built from text (SMILES, sequences). Solvating a solute, adding ions and repartitioning hydrogen masses follow
msys's `dms-solvate`, `dms-neutralize` and `dms-hmr` tools step by step, so
the same inputs give the same systems. The tests compare each with msys.

```python
import boonza

protein = boonza.load("protein.dms")
s = boonza.solvate(protein, thickness=10.0)  # TIP3P, 10 Å around the protein
s = boonza.neutralize(s, concentration=0.15)  # Na+/Cl-: neutral, then 150 mM
s = boonza.repartition_hydrogen_masses(s, "not water", 3.024)
boonza.save(s, "solvated.dms")
```

## From text

A molecule or a peptide can be built in 3D from one line of text (RDKit is
needed):

```python
lig = boonza.from_smiles("CC(=O)Oc1ccccc1C(=O)O", name="ASP", conformers=10)
pep = boonza.peptide("ACDEFGHIKLMNPQRSTVWY", "helix")  # or "sheet", "extended",
pep = boonza.peptide("GPPG", [(-75, 145)] * 4)  # "polyproline", or (phi, psi)
```

- `from_smiles` adds hydrogens and embeds 3D conformers with RDKit's ETKDG
  (reproducible with `seed`). It minimizes them with MMFF94, or UFF where
  MMFF has no parameters, and keeps the lowest in energy. Bond orders and
  formal charges come from the SMILES. The atoms are named C1, C2, ..., H1, ...
  in one residue.
- `peptide` builds the chain with PDB atom and residue names and makes every
  peptide bond trans. It sets phi/psi residue by residue, then relaxes the
  structure with MMFF94 while holding phi/psi. Proline's phi is left to its
  ring. The termini are a free amine and a free acid.

From the command line:

```bash
boonza build --smiles 'CC(=O)Oc1ccccc1C(=O)O' -o aspirin.sdf
boonza build --sequence ACDEFGHIK --conformation helix -o peptide.pdb
```

## A picture of them

`draw` writes a 2D picture of one molecule or a list of them, as PNG or SVG:

```python
boonza.draw("CC(=O)Oc1ccccc1C(=O)O", "aspirin.png")
d = boonza.draw(series, "series.png", names=names, columns=3, mcs=True, align=True)
d.core  # the SMARTS that was marked, or None if they share nothing
d.failures  # the SMILES RDKit could not read
```

```bash
boonza draw 'CCO' 'c1ccccc1' -o two.png
boonza draw --input library.smi -o series.png --columns 4 --rows 3 --mcs --align
```

`--input` reads a file of SMILES, one per line, each with an optional name
after it; `#` starts a comment. The names become the labels, so a `.smi`
file needs nothing else to make a legible sheet.

- **Grid.** `--columns` sets how many go across and `--rows` how many down.
  Without `--rows` every molecule lands on one page, which for a few hundred
  is a picture nothing can display; with it, the rest go to `series-2.png`,
  `series-3.png` and so on.
- **`--mcs`** finds the largest scaffold the molecules share and marks it,
  which is the one to reach for when you do not already know what they have
  in common. `--highlight SMARTS` marks something you name instead. When the
  molecules share nothing, that is said rather than drawn as a picture with
  no highlight.
- **`--align`** lays every molecule out with the core the same way up, which
  is what makes a series comparable at a glance. A molecule without the core
  keeps its own depiction rather than being forced onto one.

A SMILES that cannot be read is reported and the others are still drawn, so
one bad line in a library does not cost the rest.

## Solvate

`solvate(solute, box=None, thickness=5.0, solvent=None, ...)` centers the
solute and tiles a solvent box over the requested box. `box` is one edge
length or three. By default the box is a cube of the solute's largest extent
plus `thickness` on each side. The default solvent is the TIP3P box
distributed with msys (`boonza.build.WATER_BOX`); any system with a periodic
cell can be used instead.

A solvent molecule is removed when:

- one of its `solvent_selection` atoms (oxygens by default) is within
  `min_solute_dist` (2.4 Å) of the solute, counting periodic images;
- its center lies outside the box;
- any of its atoms is within `min_solvent_dist` (1.0 Å) of a periodic image
  of the solvent.

The water goes into its own ct, in chains named W1, W2, ... with residues
numbered from 1.

## Neutralize

`neutralize(system, cation="Na", anion="Cl", concentration=0.0, ...)`
replaces waters with ions:

- enough counterions to cancel the solute's charge (the sum of
  `formal_charge` by default; `charge="charge"` uses partial charges);
- then ion pairs up to `concentration` in mol/L, counted against the waters
  as msys counts them.

Waters within `solute_pad` (5 Å) of anything that is not water, or matching
`keep`, are never replaced. The rest are taken in a random order fixed by
`random_seed`. Ions sit at the replaced water's center of mass: counterions
in chain `ION`, the others in `ION2`, both in a new ct. Unlike msys, boonza
gives the ions their element masses.

## Hydrogen mass repartitioning

`repartition_hydrogen_masses(system, selection="not water", mass=3.024)`
sets the hydrogens' mass and takes the difference from the heavy atom each
is bonded to, so the total mass is unchanged. This allows 4 fs time steps.
With `repartition=False` only the hydrogens change, for example to deuterium
(2.014).

## Force fields

Added waters and ions have no force-field terms, as in msys. Parameterize
the finished system, or add their terms, before simulating. `boonza.validate`
reports atoms without nonbonded parameters.
