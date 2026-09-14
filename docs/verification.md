# Verification and benchmarks

Every feature that reimplements an established tool is tested against that
tool, running the real program where possible. The test suite has 463 tests.

| Feature | Reference | How it is compared |
|---|---|---|
| DMS, MAE read/write; clone, append | msys | full canonical dump of structure and every table |
| selection language | msys | msys's own selection test suite plus extra selections |
| PDB, SDF read/write, bond guessing | msys | canonical dump |
| mmCIF | gemmi | atom records, models, cell |
| GRO | MDAnalysis | atoms and coordinates |
| DCD, XTC, TRR, Amber NetCDF | MDAnalysis | frames, boxes, times; MDAnalysis and SciPy read the files boonza writes |
| Amber prmtop, inpcrd/rst7 | msys `LoadPrmTop`; OpenMM `AmberPrmtopFile` | every table, positions, velocities and cell on msys's test system and five more prmtops (CMAP included); energies |
| PSF | MDAnalysis `PSFParser`; OpenMM `CharmmPsfFile` | atoms, types, charges, masses, residues, segments, bonds on nine PSF variants |
| GROMACS topology | OpenMM `GromacsTopFile`; MDAnalysis `ITPParser` | energies of an Amber-style topology (and against its Amber original); GROMOS structure |
| Desmond DTR, STK | msys molfile | frames, boxes, times; DTRs written by msys (single and double precision, several frames per file), later runs replacing earlier ones in an STK |
| PDB SSBOND/LINK and mmCIF `_struct_conn` bonds | gemmi | bond lists on 4 structures; PDB and mmCIF files of each give the same bonds |
| periodic distances | MDAnalysis | distance matrices, capped pairs |
| Glue / wrapping | msys `Wrapper` | wrapped positions |
| DSSP, phi/psi/omega | mdtraj | codes and angles on 10 PDB files |
| rings (SSSR) | msys | rings and atom order |
| knots | msys `dms-find-knot` | knot lists |
| identical molecules | msys `FindDistinctFragments` | groups |
| OpenMM translation | OpenMM | energies per force group (Amber prmtop, CHARMM36, CHARMM-GUI PSF, GROMACS-style forces, CMAP, impropers, restraints) |
| matchmaker, `chimerax_ss` | UCSF ChimeraX (headless) | alignment, pairs, pruning, RMSDs, transform on 5 structure pairs |
| cealign | Biopython `CEAligner` | RMSD and transform |
| sequence alignment | Biopython `PairwiseAligner` | optimal scores, global, local and free end gaps |
| hydrogen bonds | MDAnalysis `HydrogenBondAnalysis`; mdtraj `baker_hubbard`, `wernet_nilsson` | bonds per frame |
| RMSD, RMSF, Rg, RDF | MDAnalysis | values per frame and bin |
| contacts, SASA, Rg | mdtraj | distances, areas |

Where a reference computes in float32, bonds or points sitting exactly on a
cutoff may differ, and the tests check that each such case is on the cutoff.

One case where boonza and a reference disagree: on a triclinic box,
MDAnalysis's grid search misses a few hydrogen bonds per frame that its own
distance and angle functions confirm. boonza finds them.

## Running the tests

```bash
pytest
```

Reference programs are optional. Tests that need a missing one are skipped.
Their locations are set with environment variables:

| Variable | Default | Used for |
|---|---|---|
| `BOONZA_MSYS_PYTHON` | `/opt/homebrew/bin/python3.10` | Python that can import msys |
| `BOONZA_MSYS_BUILD`, `BOONZA_MSYS_FILES` | `~/msys/build`, `~/msys/tests/files` | msys build and test files |
| `BOONZA_MDTRAJ_PYTHON` | `~/miniforge3/envs/ommflow/bin/python` | mdtraj |
| `BOONZA_BIOPYTHON` | same | Biopython |
| `BOONZA_CHIMERAX` | `/Applications/ChimeraX-1.11.app/Contents/bin/ChimeraX` | ChimeraX |
| `BOONZA_CHARMM_GUI` | two folders in `~/Downloads` | CHARMM-GUI PSF systems |

## Benchmarks

Apple Silicon, single process (numba kernels use all cores):

| Task | boonza | Reference |
|---|---:|---:|
| DMS load, 1.05 M atoms, 4 M terms | 0.96 s | msys 1.05 s |
| DMS save, same | 3.8 s | msys 2.1 s |
| find molecules, 1.05 M atoms | 0.004 s | msys 0.84 s |
| group identical molecules, 333k molecules | 0.31 s | msys 1.85 s |
| triclinic distance matrix 5000 × 5000 | 0.115 s | MDAnalysis 1.0 s |
| periodic pairs within 6 Å, 26k atoms | 0.049 s | MDAnalysis 0.185 s |
| DCD read | 0.06 s | MDAnalysis 0.08 s |

Text formats, 1.05 M atoms (msys's 3.dms tiled 40 times; every program loads
the same files; boonza and msys guess bonds from PDB, gemmi and MDAnalysis do
not):

| Step | boonza | msys | gemmi | MDAnalysis |
|---|---:|---:|---:|---:|
| load PDB | 2.64 s | 1.52 s | 0.28 s | 6.78 s |
| save PDB | 3.47 s | 1.02 s | 0.57 s | 5.61 s |
| load mmCIF | 4.47 s | - | 1.58 s | - |
| save mmCIF | 3.34 s | - | 0.54 s | - |
| load MAE (with force field) | 12.36 s | 5.86 s | - | - |
| save MAE (with force field) | 11.74 s | 15.55 s | - | - |

The PDB and mmCIF writers format lines one at a time in Python, and MAE
reading spends its time in a regex tokenizer. These are the known places to
speed up.

```bash
python benchmarks/bench_dms.py --copies 40
python benchmarks/bench_fragments.py --copies 40
python benchmarks/bench_io.py --copies 40
```
