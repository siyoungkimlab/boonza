# Building systems

Solvating a solute, adding ions and repartitioning hydrogen masses follow
msys's `dms-solvate`, `dms-neutralize` and `dms-hmr` tools step by step, so
the same inputs give the same systems. The tests compare each with msys.

```python
import boonza

protein = boonza.load("protein.dms")
s = boonza.solvate(protein, thickness=10.0)       # TIP3P, 10 Å around the protein
s = boonza.neutralize(s, concentration=0.15)      # Na+/Cl-: neutral, then 150 mM
s = boonza.repartition_hydrogen_masses(s, "not water", 3.024)
boonza.save(s, "solvated.dms")
```

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
