# viparr force fields

[viparr](https://github.com/DEShawResearch/viparr) force fields are
directories of JSON files: residue templates (atom types, charges, bonds,
impropers, CMAP tuples, virtual sites) and one table per functional form.
[viparr-ffpublic](https://github.com/DEShawResearch/viparr-ffpublic) has
about 70 of them: Amber (ff94 to ff19SB, DES-Amber, lipids, nucleic acids),
CHARMM (c22 to c36m, lipids, carbohydrates, ethers), ions and water models.
Together they cover many more residues than OpenMM's XML force fields.

boonza reads them and parameterizes a system the way viparr does, with no
viparr or msys installed:

```bash
git clone https://github.com/DEShawResearch/viparr-ffpublic
export VIPARR_FFPATH=$PWD/viparr-ffpublic/ff
```

```python
import boonza

s = boonza.load("protein_with_hydrogens.dms")  # needs bonds and hydrogens
p = boonza.parameterize(s, ["aa.charmm.c36m", "water.tip3p_charmm", "ions.charmm36"])
boonza.save(p, "parameterized.dms")
boonza.openmm_energies(p)  # kcal/mol per table
```

A force field can be a name in `$VIPARR_FFPATH` (or `path=`), a directory,
or a loaded `boonza.load_forcefield(...)`. The output keeps the input's atoms
and coordinates, takes charges from the templates, and has the force field as
DMS tables. Each parameter row records the atom types it matched (`type`)
and the force field's comment (`memo`), so `boonza.describe` shows where
every parameter came from.

## How molecules are matched

- A residue matches a template by its bond graph: the element and number of
  bonds of every atom, bonds to neighbouring residues included. Atom and
  residue names do not matter (copy the template names with
  `rename_atoms=True`, `rename_residues=True`).
- A molecule (bonded fragment) takes all its parameters from one force field:
  the **first one in the list** whose templates match every residue. A later
  force field that also matches it is skipped with a `ViparrWarning`. So list
  the force field you trust most first:

  ```python
  boonza.parameterize(s, ["aa.amber.ff14SB", "aa.amber.ff99SB", "water.tip3p"])
  # the protein gets ff14SB; ff99SB would only take molecules ff14SB cannot match
  ```

- Two templates of the *same* force field that both match a residue is an
  error, and so is a molecule that no force field matches (the error lists
  why each force field failed: no template with that formula, or one with
  that formula but different bonds).

This is viparr's own rule, and it is what makes mixing force fields easy:
OpenMM refuses two force fields with different parameters for the same
residue, while here the order of the list settles it.

## Patching a force field

`merge_forcefields(base, patch)` is viparr's `-m`: templates of the patch
replace templates of the same name, parameter rows of the patch replace rows
with the same atom types (new types go first), and CMAP grids are replaced.
`append_only=True` is `-a`, which refuses to replace anything.

```python
ildn = boonza.merge_forcefields("aa.amber.ff99SB", "aa.amber.ff99SB-ILDN")
p = boonza.parameterize(s, [ildn, "water.tip3p", "ions.amber1jc.tip3p"])
```

## How parameters are matched

- Bonds, angles and proper dihedrals come from the bond graph; impropers,
  CMAP tuples, extra exclusions and virtual sites from the templates.
- A row matches a tuple of atom types exactly or through `*` wildcards,
  forwards or backwards. An exact row wins, then the first wildcard row in
  file order. A proper dihedral takes every consecutive row with the same
  types (CHARMM's multi-term dihedrals).
- Pairs up to the rules' exclusion distance are excluded; scaled 1-4 pairs go
  to `pair_12_6_es` (with `vdw1_14` types where the force field has them,
  as CHARMM does), `vdw2` rows become NBFIX overrides, and Urey-Bradley and
  `improper_trig` terms join `stretch_harm` and `dihedral_trig`.
- Constraints, as viparr adds them by default: each heavy atom with its
  hydrogens becomes a `constraint_ahN` term and each water a rigid
  `constraint_hoh`, with the lengths and angle of the bonds and angle they
  replace; those terms are marked `constrained`, and `to_openmm` makes them
  OpenMM constraints. `constraints=False` (`--without-constraints`) leaves
  them out; `boonza.build_constraints` adds them to any parameterized system.
- Masses come from the atom types; then, as viparr does by default, all atoms
  of an element get the median of their masses (`fix_masses=False` keeps
  them). Virtual sites are appended after the real atoms
  (`reorder_ids=True` puts them after their parents).

## Where boonza differs from viparr

- **D amino acids.** CMAP grids are made for L residues. Templates match by
  graph, which cannot tell mirror images apart, so viparr gives D residues the
  L grid. boonza reads each CMAP residue's chirality from its coordinates and
  gives D residues the mirrored grid, E_D(phi, psi) = E_L(-phi, -psi). That is
  how CHARMM builds its D-amino-acid CMAPs (checked: its D grids equal the
  mirrored L grids exactly). The mirror image of a peptide then has the same
  energy as the peptide. `cmap_chirality=False` reproduces viparr.
- **Two CMAP force fields.** When two force fields in one run both have CMAP
  tables, viparr re-points the first one's CMAP terms to the second one's
  grids. boonza keeps each force field's own grids.
- Not done: parameterizing only a selection, prochiral atom renaming, and
  ligand force fields.

## What the checks found in the force fields

- The Amber force fields store the 1-4 electrostatic scale as 0.8333
  instead of 1/1.2. On a 1,375-atom protein that is 0.17 kcal/mol of 1-4
  electrostatics compared with OpenMM's ff14SB.
- Impropers on equivalent atoms (Arg NH2, Asn/Gln amide H, Asp OD1/OD2) list
  them in a different order from OpenMM's templates, a few hundredths of a
  kcal/mol.
- Proper dihedrals, backbone ones included, agree exactly with CHARMM's
  `par_all36m_prot.prm`, with Amber's `parm10.dat` + `frcmod.ff14SB`, and
  with OpenMM's `charmm36_2024.xml` and `amber14-all.xml` on 1TEN.

## Command line

```bash
boonza parameterize in.dms out.dms -f aa.charmm.c36m -f water.tip3p_charmm -f ions.charmm36
boonza parameterize in.dms out.dms -f aa.amber.ff99SB -m aa.amber.ff99SB-ILDN -f water.tip3p
```

`-f` adds a force field (in priority order), `-m`/`-a` patch the previous
one, `--ffpath` sets where names are looked up, and `--viparr-cmap` keeps
viparr's CMAP behaviour.

## Running simulations

The parameterized system is a complete force field, so it goes straight to
OpenMM:

```python
topology, system, positions = boonza.to_openmm(p, nonbonded_method="PME", cutoff=9.0)
```

Solvate and neutralize first (`boonza.solvate`, `boonza.neutralize`), then
parameterize the whole box, so the water and ions get the force field you
list for them.
