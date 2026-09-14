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

It follows the msys data model (structure plus full force-field tables)
without requiring msys, and adds trajectory analysis, superposition,
sequence alignment and RDKit/OpenMM bridges on top. Everything is NumPy plus
numba.

## Documentation

- [Getting started and guide](docs/index.md)
- [Examples](docs/examples.md): 15 runnable scripts in [`examples/`](examples),
  shown with their real output
- [API reference](docs/reference.md), generated from the code
- [Verification and benchmarks](docs/verification.md)

Regenerate the generated pages with `python docs/gen_reference.py` and
`python docs/gen_examples.py`.

## Status

Implemented:

- Columnar core: atoms, bonds, residues, chains, cts, with user-defined columns
- Force-field tables in the msys model: term tables, shared param tables,
  nonbonded overrides, exclusions, auxiliary tables, `nonbonded_info`
- Editing: add atoms/residues/chains/bonds (singly or vectorized), delete,
  reorder, clone subsets, append systems, copy-on-write param edits
- Fragments (molecules) from the bond graph; atom → residue → chain → ct / fragment
- DMS read/write (plain, `.gz`, `.bz2`), checked against msys
- MAE/CMS read/write with the full `ffio_ff` force field (plain or compressed),
  checked against msys; alchemical files are not supported yet
- msys/VMD atom selection language (`s.select("protein and within 5 of resname LIG")`),
  checked against msys's own selection test suite; residue typing (protein,
  nucleic, water, lipid, backbone, sidechain) as in msys

- PDB read/write (multi-model, TER-aware, hybrid serials), matching msys,
  including its geometric bond guessing (`s.guess_bonds(periodic=...)`), plus
  SSBOND, LINK and CONECT records (with bond orders; mmCIF `_struct_conn` too);
  written as one model, with
  CONECT records where re-guessing would get bonds wrong, so bonds round-trip
- SDF/MOL V2000 read/write with data fields, charges, isotopes and stereo
  flags, matching msys
- GRO read/write (first frame; nm → Å), matching MDAnalysis
- Topologies: Amber prmtop (full force field as msys converts it, with
  inpcrd/rst7 coordinates), CHARMM/NAMD PSF (structure, types, charges,
  bonds) and GROMACS .top (preprocessed, molecules repeated, force field
  for the common function types), checked against msys, MDAnalysis and
  OpenMM
- PDBx/mmCIF read/write (`_atom_site`, models, cell, space group), matching
  gemmi; chain = `auth_asym_id`, segid = `label_asym_id`
- Cell-list neighbor search: `boonza.spatial.pairs_within(pos, r, cell)`
- Trajectories: `boonza.open_trajectory(path, system)` gives lazy, random-access
  frames (`traj[i]`, `traj[::10]`, `traj.chunks(500, atoms="protein")`).
  DCD is read natively via memory mapping (CHARMM/NAMD/X-PLOR, fixed atoms,
  both byte orders); XTC and TRR use MDAnalysis's compiled XDR library;
  Amber NetCDF and Desmond DTR/STK are read natively.  Writers (DCD, XTC,
  TRR, Amber NetCDF): `boonza.open_writer(path, natoms)`.  Checked against
  MDAnalysis (DCD, XTC, TRR, NetCDF) and msys (DTR, STK).

- Periodic geometry (`boonza.pbc`): exact minimum image for orthorhombic and
  triclinic cells; parallel distance matrices, condensed self-distances,
  capped (cutoff) pair searches, bond/angle/dihedral values.  Checked against
  MDAnalysis.
- Glue (`boonza.Glue`): make molecules whole, keep glued groups on their
  jointly closest images, wrap around a center, fit onto a reference, per frame
  or over a trajectory.  A triclinic-capable port of msys pfx and viswizard's
  glue, checked against msys.
- Superposition:
  - `boonza.matchmaker(mobile, reference)`: a port of UCSF ChimeraX
    matchmaker. It uses BLOSUM-62 plus secondary structure from a port of
    ChimeraX's DSSP (`boonza.chimerax_ss`), gap penalties that depend on
    secondary structure, ChimeraX's pruning, and best-scoring chain pairing.
    It reproduces ChimeraX's alignments, pairs, RMSDs and transforms on
    close and remote homologs.
  - `boonza.cealign(mobile, reference)`: structure-only CE alignment
    (PyMOL cealign / Biopython), for homologs with little sequence identity.
    It matches Biopython's CEAligner.
  - `boonza.superpose`: plain Kabsch fits pairing atoms by order (or a simple
    identity-based sequence match), with pruning.

- Secondary structure (`boonza.dssp`): a port of mdtraj's DSSP 2.2, matching
  mdtraj code for code (full and simplified alphabets); backbone phi/psi/omega
  (`boonza.backbone_dihedrals`, per residue and frame, NaN at chain breaks),
  matching mdtraj; SDF V3000 read/write.
- RDKit bridge (`boonza.to_rdkit`, `from_rdkit`, `fragments_to_rdkit`,
  `sel.to_rdkit()`): atom indices, names and residue info are carried both ways;
  `assign_bond_orders` perceives bond orders and charges for ligands read
  from PDB/GRO/CIF; `smarts` selections work through RDKit.
- OpenMM bridge (`boonza.to_openmm`, `from_openmm`, `openmm_energies`):
  harmonic bonds/angles, trig dihedrals, impropers, 1-4 pairs, exclusions,
  Lorentz-Berthelot or geometric combining with NBFIX overrides, constraints
  (including rigid water), virtual sites, position restraints and CMAP.  On an
  Amber system converted by msys, energies match OpenMM's own prmtop reader.
  `from_openmm` also reads the custom forces CHARMM and GROMACS setups use
  (impropers, tabulated NBFIX Lennard-Jones, 1-4 Lennard-Jones bonds,
  Ryckaert-Bellemans torsions, geometric Lennard-Jones, restraints) and CHARMM
  lone pairs (as `virtual_fdat3`), identified by evaluating them rather than
  parsing their text; it also accepts a Topology alone.  Checked on OpenMM's
  `charmm36.xml` and on CHARMM-GUI PSF systems (CHARMM36m protein with NBFIX,
  CGenFF ligand with lone pairs).
- Force-field reports (`boonza.describe(s, "resname LIG")`, `sel.describe()`):
  charges and LJ types per atom; per pair the combined sigma/epsilon, NBFIX,
  exclusions and 1-4 scale factors; every bonded term with atom labels.

- Analysis (`boonza.analysis`):
  - `rmsd_trajectory` and `rmsf` match MDAnalysis RMSD and RMSF.
  - `radius_of_gyration` is mass-weighted like MDAnalysis, or unweighted like
    mdtraj with `weights=None`.
  - `rdf` matches MDAnalysis InterRDF, including normalization and exclusion
    blocks.
  - `residue_contacts` matches mdtraj compute_contacts.
  - `sasa` is mdtraj's Shrake-Rupley with the same sphere points and radii.
  - GRO files now get element masses the way MDAnalysis guesses them.
- Hydrogen bonds, per frame and periodic (triclinic too):
  - `boonza.hbonds(s, frames, between=[...])` uses MDAnalysis
    HydrogenBondAnalysis criteria and matches it frame by frame.
  - `boonza.baker_hubbard` and `boonza.wernet_nilsson` use mdtraj's criteria
    and match mdtraj.
- Exclusions (`boonza.update_exclusions(s, separation=3, pair_scales=None)`):
  exclusions regenerated from the bonds, virtual sites excluded like their
  host atom, and optionally scaled 1-4 pair terms; regenerating the Amber
  test system's tables reproduces msys's prmtop conversion.
- Identical molecules (`boonza.distinct_fragments`, `s.distinct_fragments()`):
  groups of isomorphic molecules exactly as msys FindDistinctFragments.
  Grouping is vectorized and exact, with an isomorphism fallback. On 1.05 M
  atoms (333k molecules):

  | step | boonza | msys |
  |------|-------:|-----:|
  | finding molecules (`fragids`) | 0.004 s | 0.84 s |
  | grouping identical molecules | 0.31 s | 1.85 s |

  Reproduce with `python benchmarks/bench_fragments.py`.
- Sequence alignment (`boonza.align_sequences(a, b)`, `boonza.sequence(s, chain)`,
  `boonza.identity_matrix(seqs)`): global, free-end-gap or local alignment
  with BLOSUM-62 and affine gaps. It reports EMBOSS-style identity,
  similarity and gap fractions. Scores match Biopython's PairwiseAligner.
- Symmetry-corrected ligand RMSD (`boonza.ligand_rmsd`, `boonza.symmetry_rmsd`,
  `boonza rmsd`): fit the proteins, move the ligand with them, then take the
  smallest RMSD over element- and bond-preserving atom mappings (exact
  branch and bound). Matches networkx enumeration and RDKit CalcRMS/GetBestRMS.
  Works per frame over trajectories (each frame fitted on its own).
  `boonza.drmsd` / `boonza drmsd`: pocket-ligand distance RMSD with the same
  symmetry matching, no fitting; matches the MDAnalysis `distance_array` recipe.
- Rings (`boonza.sssr`, `ring_systems`): msys's smallest set of smallest
  rings, or all relevant rings, with ring and atom order matching msys.
- Validation (`boonza.validate(s, strict=True)`, `boonza.find_knots`):
  - msys dms-validate checks: nonbonded coverage, bonds threaded through
    small rings (across periodic boundaries too), cell volume, massless
    atoms, virtual sites.
  - Force-field consistency: every bond has a stretch or constraint term, and
    every molecule has a whole-number net charge (virtual sites count with
    their host atom).
  - Strict checks: constrained hydrogens, sub-1 Å contacts, consistent masses,
    exclusions within three bonds and none beyond, bonded terms connecting
    bonded atoms (Urey-Bradley and Amber-style impropers are allowed), one
    water per residue.

- Diff (`boonza.diff(a, b, atom_map=None)`): atoms, residues, bonds, cell and
  every force-field table compared as sets of terms keyed by canonical atom
  order (as msys dms-diff-ff), so term order and parameter sharing do not
  matter.
- Command line: `boonza info | convert | select | validate | knots | diff |
  describe | dssp | phipsi | rmsd | drmsd` (see `boonza --help`).

Planned: alchemical DMS.

## Install

```bash
pip install -e .[dev]
```

## Example

```python
import boonza

s = boonza.load("system.dms")
atom = s.atom(0)
atom.residue.chain.name, atom.fragment.ids        # hierarchy and molecule
s.atoms["name"][:10]                               # columns are numpy arrays
s.atoms["buried"] = s.atoms["charge"] < 0          # new column

stretch = s.table("stretch_harm")
term = stretch[0]
term["r0"], term["fc"], term.atoms

amap = s.delete_atoms(s.select(s.atoms["anum"] == 1))   # old -> new indices
ligand = s.clone(s.fragment_atoms(s.fragids[-1]))
boonza.save(ligand, "ligand.dms")
```

```python
print(s.describe("resname LIG and name C1 C2"))      # charges, LJ, pairs, terms
mol = s.select("resname LIG").to_rdkit()
topology, omm_system, positions = boonza.to_openmm(s, nonbonded_method="PME")
```

Indices are always `0..N-1`.  Deleting or reordering atoms renumbers them;
handles created earlier raise `StaleHandleError` rather than pointing at
the wrong atom.

## Tests

```bash
pytest
```

Tests compare against msys when it is available.  By default they run
`/opt/homebrew/bin/python3.10` with the msys build in `~/msys/build` and read
msys test files from `~/msys/tests/files`; override with
`BOONZA_MSYS_PYTHON`, `BOONZA_MSYS_BUILD` and `BOONZA_MSYS_FILES`.  Without
msys those tests are skipped.

## Benchmarks

```bash
python benchmarks/bench_dms.py --copies 40
```

Tiles a parameterized DMS into a large system and times boonza against msys.
On `3.dms` × 40 (1.05 M atoms, 4.0 M force-field terms, Apple Silicon):

| step | boonza | msys |
|------|-------:|-----:|
| load | 0.96 s | 1.05 s |
| save | 3.8 s  | 2.1 s  |

Large tables are decoded straight from the SQLite file bytes by a numba
scanner (`boonza/io/_sqlite_scan.py`); small tables, and anything the
scanner does not handle, go through Python's `sqlite3`.  Saving still uses
`sqlite3`.
