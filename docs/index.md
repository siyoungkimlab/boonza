# boonza

boonza reads, edits and writes molecular systems so that people and AI models
can reason about the spatial relationships between atoms and molecules. For
AI in particular it is a bridge between three views of the same molecule:

- **1-D text**: file formats, sequences, the atom selection language, SMILES
  through RDKit;
- **2-D graph**: atoms and bonds, residues, chains and molecules, rings,
  symmetry-equivalent atoms;
- **3-D positions that obey the rules of physics**: coordinates, periodic
  boxes, force-field parameters and energies.

Moving between these views, and keeping them consistent, is what the library
is for.

*The name: **B**iomolecular **O**bjects & **O**perations, **N**umba-accelerated,
**Z**ero-copy **A**rrays.*

That covers structure and full force fields, trajectories and the analyses
people usually reach for msys, MDAnalysis, mdtraj, ChimeraX or PyMOL to do. It follows the msys data
model but needs neither msys nor a C++ build: everything is NumPy plus numba.

Most features reproduce an established tool exactly, and the test suite
checks them against that tool (see [Verification](verification.md)).

## Install

With conda, which also brings AmberTools for GAFF2 ligands:

```bash
conda env create -f environment.yml   # Python 3.12, NumPy, numba, RDKit, OpenMM, AmberTools, ...
conda activate boonza
pip install -e . --no-deps            # boonza itself; conda has the dependencies
pip install -e ".[dev]"               # + pytest, ruff and MDAnalysis (a reference in the tests)
```

With pip alone (`pip install -e .`) you get NumPy, numba, RDKit, OpenMM,
pandas and networkx, but not AmberTools: it is not on PyPI. Add it to any
environment with `conda install -c conda-forge ambertools`; boonza finds it
through `$AMBERHOME` or the `antechamber` on your `PATH`.

AmberTools is only needed where something has to be parameterized with
GAFF2 ([Ligands](guide/ligands.md)):

| | AmberTools |
|---|---|
| protein, water, ions, and modified residues the force fields match | not needed |
| `boonza swim` with peptide ligands, or with a library of parameterized DMS files | not needed |
| small-molecule ligands and covalent adducts (`boonza md`, `boonza parameterize --gaff2`, `boonza swim`) | needed where they are parameterized |

`boonza md` parameterizes when a run starts, so the node that starts it needs
AmberTools; `boonza swim` does it while preparing, so its simulations need
only OpenMM.

Python 3.11 or newer. The `boonza` command is installed with the package.

## Quick start

```python
import boonza

s = boonza.load("system.dms")  # also .mae .cms .pdb .cif .gro .sdf .mol
print(s)  # <System ...: 26000 atoms, ... tables>

prot = s.select("protein and name CA")  # msys/VMD selection language
s.atoms["name"][:5]  # every attribute is a NumPy column
atom = s.atom(0)
atom.residue.chain.name, atom.fragment  # hierarchy and molecule

lig = s.clone("resname LIG")  # a self-contained subsystem, force field included
boonza.save(lig, "ligand.sdf")

print(s.describe("resname LIG and name C1"))  # charges, LJ, pairs, bonded terms
problems = boonza.validate(s, strict=True)  # sanity checks

traj = boonza.open_trajectory("run.xtc", s)
rmsd = boonza.rmsd_trajectory(s, traj.read(), "protein and name CA")
```

## Conventions

| Quantity | Unit |
|---|---|
| length, positions, cell | Å |
| energy | kcal/mol |
| angles in parameters and results | degrees (internal geometry functions in `boonza.pbc` return radians) |
| charge | e |
| mass | amu |
| time | ps |

- **Indices are always `0..N-1`.** Deleting or reordering atoms renumbers
  them and returns an old→new map. Handles made before such an edit raise
  `StaleHandleError` instead of silently pointing at another atom.
- **A molecule is a fragment**: a set of atoms connected by bonds.
  `s.fragids` gives each atom's molecule.
- **Cells** are 3×3 arrays whose rows are the box vectors a, b, c. All zeros
  means "not periodic". Triclinic cells are supported everywhere.
- **Frames**: analysis functions take the system's own coordinates by
  default. Or you can pass one `(natoms, 3)` array, an `(nframes, natoms, 3)`
  array, or a `Frames` block read from a trajectory.

## Examples

Start with the [examples](examples.md): 21 runnable scripts in `examples/`, each
shown with its real output. They cover loading and selecting, building and
editing, parameterizing a protein with OpenMM, editing force fields,
superposition, sequence alignment, running and analyzing a short
trajectory, periodic boxes, RDKit, validation and the command line.

## Guide

1. [Systems: structure, columns, editing](guide/systems.md)
2. [Selections](guide/selections.md)
3. [Files and trajectories](guide/io.md)
4. [Force fields](guide/forcefield.md)
5. [viparr force fields](guide/viparr.md)
6. [OpenMM force fields, and comparing force fields](guide/openmm_forcefields.md)
7. [Ligands and covalent adducts (GAFF2)](guide/ligands.md)
8. [Running simulations (`boonza md`)](guide/md.md)
9. [Ligands swimming around a protein (`boonza swim`)](guide/swim.md)
10. [Checking systems: validate, knots, diff](guide/checking.md)
11. [Geometry and periodic boundaries](guide/geometry.md)
12. [Alignment: superposition and sequences](guide/alignment.md)
13. [Analysis](guide/analysis.md)
14. [RDKit and OpenMM](guide/bridges.md)
15. [Command line](guide/cli.md)

[API reference](reference.md) · [Verification and benchmarks](verification.md)

```{toctree}
:hidden:
:caption: Contents

examples
guide/systems
guide/selections
guide/io
guide/forcefield
guide/viparr
guide/openmm_forcefields
guide/ligands
guide/md
guide/swim
guide/building
guide/checking
guide/geometry
guide/alignment
guide/analysis
guide/summaries
guide/bridges
guide/cli
reference
verification
```
