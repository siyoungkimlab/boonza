# Running simulations (`boonza md`)

`boonza md` prepares and runs explicit-solvent MD with OpenMM, and resumes it
after a wall-time limit. It replaces ommflow: its options, settings keys and
output files are ommflow's, so configuration files and job scripts carry
over, except for the force fields (`-f` replaces `--proteinff` and
`--waterff`; see below).

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
   all hydrogens and the protonation states you want (a residue whose atoms
   the file keeps in two pieces, as preparation tools do when they write the
   hydrogens they added at the end, is gathered back together, and
   `md_index` keeps pointing at the input file);
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

Give your own with `-f`, once per force field in priority order, and `-m` to
patch the one before; or in the file:

```bash
boonza md complex.pdb -f aa.amber.ff14SB -m aa.amber.phosaa10 -f water.tip3p -f ions.amber1jc.tip3p
```

```toml
forcefields = [["aa.amber.ff14SB", "aa.amber.phosaa10"], "water.tip3p", "ions.amber1jc.tip3p"]
```

`-f` replaces the whole default list, so give every force field the system
needs. A run prints the force fields it uses, and warns when an ion set was
fitted for another water model than the water's (`ions.amber1jc.tip3p` with
`water.opc`, say).

A protein force field must come with a water model of its own family, since
the same name can mean different water: CHARMM's TIP3P has Lennard-Jones
hydrogens and Amber's does not. A CHARMM protein (`aa.charmm.*`) takes
`water.tip3p_charmm` rather than `water.tip3p`, and an Amber one the reverse;
other water models are left to you.

OpenMM XML force fields are `-f` files too, applied by boonza's own XML
reader. OpenMM 8.6.1's files are bundled with boonza, so a name such as
`amber19/opc.xml` gives the same file whatever OpenMM is installed; a name
the bundle lacks is looked up in the installed OpenMM, and a path reads your
own file. The files of one run are all XML or all viparr, and GAFF2 ligands
need viparr force fields.

```bash
boonza md protein.pdb -f amber19/protein.ff19SB.xml -f amber19/opc.xml
boonza md protein.pdb -f charmm36_2024.xml -f charmm36_2024/water.xml
boonza md protein.pdb -f my_protein.xml -f my_water.xml
```

OpenMM keeps each family's water models beside its protein force field, and
a run takes water only from there: `amber14/` for `amber14-all.xml` and
ff14SB or ff15ipq, `amber19/` for `amber19-all.xml` and ff19SB, `charmm36/`
for `charmm36.xml` (`water.xml` is CHARMM's TIP3P), `charmm36_2024/` for
`charmm36_2024.xml`, and the top-level `tip3p.xml`, `opc.xml`, ... for the
older Amber files such as `amber99sbildn.xml`. `-f charmm36.xml -f
amber14/tip3p.xml` stops with the water models that fit. Files of your own,
given by path, are not checked.

ommflow's `--proteinff`/`--waterff` pairs are these files:

| ommflow | `-f` files |
|---|---|
| `amber14sb`, `amber15ipq`, `amber19sb` | `amber14/protein.ff14SB.xml`, `amber14/protein.ff15ipq.xml`, `amber19/protein.ff19SB.xml` |
| their water: `opc`, `tip3p`, `tip4pew`, ... | `amber14/<water>.xml` (`amber19/<water>.xml` for amber19sb) |
| `charmm36`, `charmm36_2024` | `charmm36.xml`, `charmm36_2024.xml` |
| their water: `tip3p`, others | `charmm36/water.xml`, `charmm36/<water>.xml` (`charmm36_2024/...`) |

The nonbonded cutoff defaults to 0.9 nm for Amber and 1.2 nm for CHARMM, with
PME; bonds to hydrogen and water are rigid.

## Water and ions

Water comes from a TIP3P box; 4- and 5-site models get their extra sites
from their templates. Counterions cancel the solute's charge, and `saltM`
adds NaCl as msys's `dms-neutralize` does: `int(saltM / 55.345 M × waters)`
pairs, the waters counted without those the counterions replace. OpenMM's
Modeller also counts against the waters but rounds to the nearest pair (with
55.4 M), so the two differ by at most about one pair.

A system that has its own box but no water — a membrane built elsewhere, say
— is filled with `solvate = "fill"`, which keeps that box and adds water only
where water belongs:

```bash
boonza md membrane.dms --solvate fill --barostat membrane --workdir run1
```

Water is tiled through the cell and then taken out again wherever it landed
in a hydrophobic void: where fewer than 5 waters are linked within 4.5 Å of
each other, at least 15 apolar heavy atoms lie within 6 Å, and no protein
atom is within 8 Å. The three tests together separate a lipid void from the
water-filled pore of a membrane protein, which looks the same to any one of
them alone: the cluster finds water sitting on its own, the apolar count says
its surroundings are greasy rather than polar, and the protein clause leaves
a protein's own cavities and channels alone. Nothing uses a plane or an axis,
so a vesicle, a tube or a micelle is treated like a flat bilayer. Ions follow
as usual, with `saltM` counted against the water that remains; `padding_nm`
and `box_nm` are unused, and the cell must be rectangular.

On a 38k-atom OmpF bilayer (POPC/POPE/POPS and cholesterol, 94 x 94 x 151 Å),
filling adds 30,405 waters in 2.0 s, and the void test removes 44 of them in
a further 0.4 s: every water in the lipid core, none from the porin's pore,
none from bulk.

A system that is already solvated and ionized runs as it is with
`solvate = "none"` (`--no-solvate`):

```bash
boonza md built.dms --no-solvate --workdir run1
```

Its water, ions and periodic cell are kept, so the file must have a cell (a
DMS, MAE, GRO or mmCIF file of a built system does); `padding_nm`, `box_nm`
and `saltM` are then unused, and a run says so if you give them. Nothing is
added to neutralize the system either: a net charge is reported as a
warning, not fixed. Everything else is unchanged, force fields and GAFF2
ligands included.

## Options

| Setting | Default | |
|---|---|---|
| `forcefields` (`-f`, `-m`) | see above | viparr force fields, or OpenMM XML files |
| `ligand_mode` | `auto` | GAFF2 for what the force fields cannot match; `disabled` makes it an error |
| `ligand_charges` (`--charge LIG=-1`) | none | formal charges of ligands read from files without them |
| `parents` (`--parent MSE=MET`) | none | the standard residue a modified residue comes from, where the file has no `MODRES` and the PDB's dictionary does not know it; an unclear guess stops the run |
| `protein_extent` | `matched` | amino acids with GAFF2 atoms keep protein types as far as they match, or on the backbone and CB only (`cb`); see [Ligands](ligands.md) |
| `solvate` | `box` | `fill` keeps the input's own cell and fills its empty space, leaving hydrophobic voids dry (a membrane); `none` (`--no-solvate`) runs the input as it is |
| `padding_nm`, `saltM` | 1.0, 0.15 | |
| `cutoff_nm` | 0.9 Amber, 1.2 CHARMM | |
| `temperature`, `pressure` | 298 K, 1 bar | |
| `barostat`, `surface_tension` | `isotropic`, 0 | `membrane` scales x and y together and z on its own (semi-isotropic, for a planar bilayer), held at `surface_tension` (bar nm; 0 is tensionless); `none` keeps the volume fixed |
| `equilibration_ns` | 0.1 | NVT, then the same of NPT |
| `production_ns` | 100 | absolute target |
| `*_report_interval_ns`, `checkpoint_interval_ns` | 0.01 / 1.0, 0.01 | checkpoints must divide the production report interval |
| `performance_interval_ns` | 1.0 | rows of `performance.csv` |
| `integration_fs`, `hmr` | 2, off | `hmr` repartitions hydrogens to 4 amu (water untouched) and makes the step 4 fs |
| `dihedral_restraint`, `dihedral_restraint_kJ` | `none`, 20 | hold phi/psi at the input (`bb`: all, `ss`: helices and sheets) |
| `dihedral_restraint_selection` | every peptide chain | with `dihedral_restraint`, hold only the torsions whose atoms this selects, e.g. `chain A` (`boonza swim` leaves out its ligands); it does nothing on its own |
| `seed`, `precision`, `platform` | 0, mixed, fastest | `seed` goes to the initial velocities, the integrator and the barostat; 0 means choose one |
| `repulsion_selection`, `repulsion_distance_nm`, `repulsion_kJ` | none, 0.5, 500 | keep the molecules a selection picks from sticking together: E = k (d0 - r)^2 between heavy atoms of different ones closer than d0 (k in kJ/mol/nm^2); `boonza swim` sets it for its ligands |
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
`monitor.csv`, and `status.json` says `detached` as well as `running` and
`target_reached`. A detached run stays stopped when resumed unless early stop
is turned off or a larger `production_ns` is given.

### status.json

Every run writes it, watched or not, beside each checkpoint — so it is never
newer than the checkpoint it describes, and one file answers "has this
finished, and where is it" for a whole array of runs:

```json
{"outcome": "running", "target_production_ns": 500.0,
 "final_production_time_ns": 128.0, "final_production_step": 64000000,
 "early_stop_enabled": false}
```

`outcome` is `running`, `target_reached` or `detached`. With early stop the
record also carries `ligand_id`, `component_id` and
`consecutive_detached_count`; without it those are left out rather than left
empty, so the file never implies a target that was never watched. A run that
was watched and is resumed without early stop keeps what it found and says
`early_stop_enabled: false`.

### Repeating a run

`seed` is given to the initial velocities, to the integrator and to the
barostat, which draws its volume moves from a generator of its own. With the
same seed and the same input, a run repeats itself exactly — but only when
the forces are summed the same way each time, which on the CPU platform means
one thread (`OPENMM_CPU_THREADS=1`). With several threads the summation order
varies, and a trajectory diverges from its twin within a picosecond however
it is seeded.

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
| `status.json` | how far the run got, and how it ended |
| `pocket.json`, `monitor.csv` | with early stop |

Each run prints the system (particles, waters, ions, constraints, forces,
box, nonbonded method) and the platform with its properties before it
integrates.

## Installing

OpenMM is a dependency of boonza. For GAFF2 ligands, AmberTools comes from
conda-forge; `install.sh` makes a conda environment with both
(`environment.yml`) and installs boonza into it:

```bash
bash install.sh --cuda 12.6                # for GPU nodes whose driver supports CUDA 12.6
bash install.sh --cuda none --openmm 8.2   # without CUDA: run with --platform OpenCL
conda activate boonza
```

Give `--cuda` the release the GPU nodes' driver supports (`CUDA Version` in
`nvidia-smi` on one of them). Without it, conda picks one from the machine
running the installer, often a login node, and older GPU nodes then fail
with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. conda-forge builds OpenMM for CUDA
before 12.6 against NumPy 1 only, which boonza cannot use, so for an older
driver install with `--cuda none` and run on the OpenCL platform, which the
driver compiles for itself.

numpy stays below 2.4, whose builds need x86-64-v2 CPUs that older HPC nodes
lack. `pip check` still reports that four Python tools bundled with AmberTools
(proprep, ndfes, fetkutils, edgembar) ask for numpy below 2; boonza does not
use them, and antechamber, sqm, parmchk2 and tleap do not use numpy.
