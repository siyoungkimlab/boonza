# Running simulations (`boonza md`)

`boonza md` prepares and runs explicit-solvent MD with OpenMM, and resumes it
after a wall-time limit. It replaces ommflow: its options, settings keys and
output files are ommflow's, so configuration files and job scripts carry
over.

```bash
boonza md protein.pdb --workdir protein_md                  # a new 100 ns run
boonza md --workdir protein_md                              # the same run, resumed
boonza md --workdir protein_md --production-ns 200          # a larger target
boonza md --write-default-config md.toml                    # a commented template
boonza md --config md.toml --production-ns 50               # options override the file
boonza md complex.dms --list-components                     # molecules and selectors
```

A new run:

1. reads the structure (PDB, DMS, MAE, mmCIF, SDF, GRO, ...), which must have
   all hydrogens and the protonation states you want;
2. applies the force fields, with GAFF2 templates for ligands, non-standard
   residues and covalent adducts ([Ligands](ligands.md));
3. solvates it (`padding_nm` from solute to box edge) and adds counterions
   and NaCl;
4. minimizes, then runs `equilibration_ns` of NVT and `equilibration_ns` of
   NPT (Langevin middle integrator, 1/ps, Monte Carlo barostat every 25
   steps), and NPT production toward `production_ns`.

## Force fields

The default is viparr's force fields, bundled with boonza, in this priority
order (the first one whose templates match a molecule parameterizes it):

1. `aa.amber.ff19SB` with `aa.amber.phosaa19SB` merged in. A phosphorylated
   chain is one molecule and must be matched by one force field, so the
   phosphorylated residues join ff19SB rather than being listed after it.
2. `water.tip3p`
3. `ions.amber1jc.tip3p`: Joung-Cheatham monovalent ions.
4. `ions.amber1234lm_anton.tip3p`: Li/Merz ions for the rest (Mg²⁺, Zn²⁺,
   Ca²⁺, ...). It also has the monovalent ions; listed after 1jc, it does not
   supply them.

Give your own with `-f` (and `-m` to patch the one before), or in the file:

```toml
forcefields = [["aa.amber.ff14SB", "aa.amber.phosaa10"], "water.tip3p", "ions.amber1jc.tip3p"]
```

ommflow's OpenMM XML families still work, applied by boonza's own XML reader
(`proteinff = "amber19sb"`, `waterff = "opc"`; `--proteinff`, `--waterff`).
GAFF2 ligands need viparr force fields.

The nonbonded cutoff defaults to 0.9 nm for Amber and 1.2 nm for CHARMM, with
PME; bonds to hydrogen and water are rigid.

## Water and ions

Water comes from a TIP3P box; 4- and 5-site models get their extra sites
from their templates. Counterions cancel the solute's charge, and `saltM`
adds NaCl as msys's `dms-neutralize` does: `int(saltM / 55.345 M × waters)`
pairs, the waters counted without those the counterions replace. OpenMM's
Modeller also counts against the waters but rounds to the nearest pair (with
55.4 M), so the two differ by at most about one pair.

## Options

| Setting | Default | |
|---|---|---|
| `forcefields` (`-f`, `-m`) | see above | viparr force fields |
| `proteinff`, `waterff` | none | OpenMM XML families instead |
| `ligand_mode` | `auto` | GAFF2 for what the force fields cannot match; `disabled` makes it an error |
| `ligand_charges` (`--charge LIG=-1`) | none | formal charges of ligands read from files without them |
| `parents` (`--parent MSE=MET`) | none | the standard residue a modified residue comes from, where the file has no `MODRES` and the PDB's dictionary does not know it; an unclear guess stops the run |
| `protein_extent` | `matched` | amino acids with GAFF2 atoms keep protein types as far as they match, or on the backbone and CB only (`cb`); see [Ligands](ligands.md) |
| `padding_nm`, `saltM` | 1.0, 0.15 | |
| `cutoff_nm` | 0.9 Amber, 1.2 CHARMM | |
| `temperature`, `pressure` | 298 K, 1 bar | |
| `equilibration_ns` | 0.1 | NVT, then the same of NPT |
| `production_ns` | 100 | absolute target |
| `*_report_interval_ns`, `checkpoint_interval_ns` | 0.01 / 1.0, 0.01 | checkpoints must divide the production report interval |
| `performance_interval_ns` | 1.0 | rows of `performance.csv` |
| `integration_fs`, `hmr` | 2, off | `hmr` repartitions hydrogens to 4 amu (water untouched) and makes the step 4 fs |
| `dihedral_restraint`, `dihedral_restraint_kJ` | `none`, 20 | hold phi/psi at the input (`bb`: all, `ss`: helices and sheets) |
| `seed`, `precision`, `platform` | 0, mixed, fastest | |
| `early_stop` and `monitor_*`, `*_cutoff_nm`, `confirmation_checks` | off | see below |

## Restarts

A work directory that exists and is not empty is resumed; an empty one (as
batch schedulers make) starts a new run. A restart loads `system.xml`,
`integrator.xml` and the checkpoint, runs only what is left of
`production_ns`, and appends to `trajectory.dcd`, `state.csv` and
`performance.csv`. Settings not given again come from `final.toml`, which is
rewritten only after the restart passes its checks. The time step cannot
change on a restart.

## Backbone restraints

`dihedral_restraint = "bb"` restrains every backbone phi and psi across an
existing peptide bond to its value in the input, with a six-term Fourier well
of depth set by `dihedral_restraint_kJ`; `"ss"` only those of residues DSSP
puts in helices or sheets (boonza's own DSSP; MDTraj is not needed). The
force is in `system.xml`, so restarts keep it; the torsions and their
reference angles are in `dihedral_restraints.csv`.

## Stopping when a binder leaves

With `early_stop`, a target is watched every `monitor_interval_ns`. It is the
only GAFF2 ligand, or, when there are several (then one must be chosen), the
one named by one of:

- `monitor_ligand = "ligand-1"`, `monitor_chain = "B"` or
  `monitor_component = "component-2"`: the IDs `--list-components` prints,
  numbered by lowest atom index, so stable for a given input file;
- `monitor_selection = "resname LIG and chain L"`: atoms of the input
  structure in boonza's selection language (msys's), with its atom and
  residue names; the selected atoms are the target.

A missing or wrong choice stops a new run before anything is built, and
leaves the work directory as it was. The pocket is the
receptor's heavy atoms within `pocket_cutoff_nm` at the start of production
(`pocket.json`). Production stops after `confirmation_checks` checks in a row
with no target-pocket pair within `contact_cutoff_nm` and a minimum distance
above `detach_cutoff_nm` (periodic distances). Each check is a line of
`monitor.csv`, and `status.json` says `running`, `target_reached` or
`detached`. A detached run stays stopped when resumed unless early stop is
turned off or a larger `production_ns` is given.

## Output files

| File | |
|---|---|
| `input.*` | the input structure |
| `solvated.dms` | the solvated system with its force field (inspect it with `boonza.describe`, compare it with `boonza diff`) |
| `solvated.pdb`, `solvated.mae` | the same structure, with bonds and box |
| `components.json` | the molecules of the input and their atoms in the simulation |
| `gaff2/`, `gaff2_patch/` | AmberTools files and the GAFF2 patch, when ligands were found |
| `covalent_*.png` | a 2D drawing of each covalent adduct: blue atoms keep protein types, orange ones are GAFF2 |
| `equilibration.dcd`, `equilibration.csv` | equilibration |
| `equilibrated.pdb`, `.mae` | the structure production starts from |
| `trajectory.dcd`, `state.csv` | production (step and time start at 0) |
| `final.pdb`, `.mae` | the latest coordinates |
| `checkpoint.chk`, `system.xml`, `integrator.xml`, `final.toml` | for restarts |
| `performance.csv` | where the wall time went |
| `dihedral_restraints.csv`, `.png` | with backbone restraints |
| `pocket.json`, `monitor.csv`, `status.json` | with early stop |

Each run prints the system (particles, waters, ions, constraints, forces,
box, nonbonded method) and the platform with its properties before it
integrates.

## Installing

OpenMM is a dependency of boonza. For GAFF2 ligands, AmberTools comes from
conda-forge; `environment.yml` makes an environment with both:

```bash
conda env create -f environment.yml
conda activate boonza
pip install -e .
```

numpy stays below 2.4, whose builds need x86-64-v2 CPUs that older HPC nodes
lack. `pip check` still reports that four Python tools bundled with AmberTools
(proprep, ndfes, fetkutils, edgembar) ask for numpy below 2; boonza does not
use them, and antechamber, sqm, parmchk2 and tleap do not use numpy.
