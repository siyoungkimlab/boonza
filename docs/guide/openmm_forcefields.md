# OpenMM force fields, and comparing force fields

boonza reads OpenMM's XML force fields natively and builds the force field
that OpenMM's `ForceField.createSystem` would build, as boonza tables, without
calling OpenMM:

```python
import boonza

s = boonza.load("protein_with_hydrogens.dms")
p = boonza.parameterize_openmm(s, ["amber19-all.xml", "amber19/opc.xml"], constraints="hbonds")
boonza.save(p, "amber19.dms")
```

OpenMM's own XML files, from its 8.6.1 release, are bundled with boonza: a
file name that is not a path is looked up there first, then in the installed
OpenMM's data directory, so `amber19-all.xml` gives the same force field
whatever OpenMM is installed (OpenMM 8.2, for one, has no amber19 files).
`<Include>` files are followed, and your own XML files work too, by path.
`boonza.load_openmm_forcefield(...)` reads them once for reuse; its `files`
say where each came from, and `boonza.ffxml.list_openmm_forcefields()` lists
the bundled names.

## What it follows

The rules are OpenMM's own, and the test suite checks the result term by term
against `createSystem` for amber14, amber19 and CHARMM36 (2024), proteins with
disulfides, water models with virtual sites (TIP4P-Ew, OPC), ions, and H-bond
constraints with rigid water:

- residues match templates by element and bond graph, with OpenMM's search
  order; unmatched residues are tried with every combination of one-residue
  patches, then with multi-residue patches such as CHARMM's disulfide `DISU`;
- bonds and angles take the first matching row, torsions prefer rows without
  wildcards, impropers follow their section's `default`, `amber` or `charmm`
  ordering;
- 1-4 pairs use the force field's `coulomb14scale`/`lj14scale`; CHARMM's
  `LennardJonesForce` gives per-type Lennard-Jones, NBFIX and its own 1-4
  terms; Urey-Bradley terms join `stretch_harm`, CMAP maps become
  `torsiontorsion_cmap`;
- two force fields ship Python `<Script>`s: CHARMM36 (2024)'s impropers and
  Amber lipid21's per-torsion 1-4 scale factors. boonza reads their tables as
  literals (it never runs the script) and applies them the same way. The
  CHARMM impropers are keyed by the atom names of the matched templates (not
  the structure's names, which do not matter) and by the neighbouring
  residues in chain order, as in OpenMM.

Where the output differs from `createSystem`, on purpose:

- constrained bonds and angles stay in their tables, marked `constrained`,
  with the constraints in `constraint_ahN`/`constraint_hoh` (as
  `boonza.parameterize` does), where OpenMM leaves them out of its forces;
- extra particles a template has but the structure lacks (TIP4P/OPC sites)
  are added after their residue's atoms, placed from the force field's site
  weights (OpenMM needs `Modeller.addExtraParticles` for this).

Not supported: other custom forces and scripts, implicit solvent, AMOEBA and
Drude force fields, template generators (GAFF, SMIRNOFF) and templates that
span several residues.

## Two ways to compare force fields

**Through OpenMM, by energy.** Parameterize a structure with a viparr force
field in boonza, convert it with `boonza.to_openmm`, and compare its energy
per term with OpenMM's own parameterization of the same structure
(`createSystem`, converted back with `boonza.from_openmm`):

```python
a = boonza.openmm_energies(boonza.parameterize(s, ["aa.amber.ff19SB"], constraints=False))
b = boonza.openmm_energies(boonza.parameterize_openmm(s, ["amber19-all.xml"], rigid_water=False))
{k: a.get(k, 0) - b.get(k, 0) for k in a}  # kcal/mol per table
```

This tells you how much two force fields disagree, and in which kind of term.

**Table by table.** Parameterize the same structure with both and compare the
two systems term by term:

```python
viparr_ff = boonza.parameterize(s, ["aa.amber.ff19SB"], constraints=False)
openmm_ff = boonza.parameterize_openmm(s, ["amber19-all.xml"], rigid_water=False)
for d in boonza.diff(viparr_ff, openmm_ff, positions=False, canonical=True):
    print(d)
```

`canonical=True` first brings both force fields to one form per interaction
(`boonza.canonical_forcefield`): every periodic torsion term of an atom tuple
is summed into Fourier coefficients (so k with phase 0 equals -k with phase
180, and several terms per row equal one term per periodicity), 1-4 pair
terms are summed per atom pair, sigma is ignored where epsilon is 0, and
terms with no energy are dropped. What remains changes the energy. For
viparr's ff19SB against OpenMM's amber19 on 1TEN that is two terminal
charges rounded differently, 111 impropers whose equivalent atoms the two
programs list in a different order, and the 1-4 electrostatic scale that
viparr's file rounds to 0.8333; for viparr's CHARMM36m (2019 files) against
OpenMM's CHARMM36 (2024), the changes between the two releases: oxygen mass,
ten charges and seventeen NBFIX pairs.

Each reader is checked against the program it copies (viparr against viparr,
the XML reader against `createSystem`), so a difference found this way is a
difference between the force fields as those programs apply them.

## Command line

```bash
boonza parameterize in.dms amber19.dms -x amber19-all.xml -x amber19/opc.xml
boonza parameterize in.dms viparr.dms -f aa.amber.ff19SB -f water.opc
boonza diff viparr.dms amber19.dms --canonical --no-positions
```
