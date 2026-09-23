# Files and trajectories

## Structures

```python
s = boonza.load("system.dms")
s = boonza.load("complex.cif", guess_bonds=False)
boonza.save(s, "out.mae")
boonza.save(s, "out.dat", format=".pdb")  # explicit format
```

The format comes from the file extension. Keywords are passed to that
format's reader or writer.

| Format | Read | Write | Notes |
|---|:-:|:-:|---|
| DMS (`.dms`, `.dms.gz`, `.dms.bz2`) | ✓ | ✓ | full force field, extra columns; matches msys; large tables read by a numba SQLite scanner |
| MAE / CMS (`.mae`, `.cms`, compressed) | ✓ | ✓ | full `ffio_ff` force field; matches msys; no alchemical files yet |
| PDB (`.pdb`, compressed) | ✓ | ✓ | models, TER, hybrid-36 serials, CRYST1; msys bond guessing plus SSBOND/CONECT records |
| PDBx/mmCIF (`.cif`, `.mmcif`, `.pdbx`) | ✓ | ✓ | `_atom_site`, models, cell; chain = `auth_asym_id`, segid = `label_asym_id`; matches gemmi |
| GRO (`.gro`) | ✓ | ✓ | first frame; nm → Å; masses as MDAnalysis guesses them |
| Amber prmtop (`.prmtop`, `.parm7`, `.top`) | ✓ | | full force field as msys's `LoadPrmTop` builds it, CMAP included; coordinates from inpcrd/rst7 or NetCDF restarts |
| CHARMM/NAMD/X-PLOR PSF (`.psf`) | ✓ | | atoms, CHARMM types, charges, masses, bonds; segments as chains; standard, EXT and NAMD layouts |
| GROMACS topology (`.top`) | ✓ | | `#include`/`#define`/`#ifdef`; force field for harmonic, Urey-Bradley, periodic, RB and improper terms, LJ combination rules 1-3, pairs, settles, constraints; Martini's cosine and restricted-bending angles and `virtual_sitesn` |
| SDF / MOL (`.sdf`, `.mol`, compressed) | ✓ | ✓ | V2000 and V3000, data fields, charges, isotopes, stereo flags; matches msys |

Reader options:

| Reader | Options |
|---|---|
| DMS | `structure_only` (skip force field), `without_tables` |
| MAE | `structure_only`, `without_tables`, `ignore_unrecognized` |
| PDB, GRO, CIF | `guess_bonds=True` (msys rules, same ct only) |
| prmtop | `coordinates` (inpcrd/rst7 or NetCDF restart), `structure_only` |
| PSF | `coordinates` (a file with the same atoms, or an array) |
| GROMACS top | `coordinates` (.gro, ...), `defines={"FLEXIBLE": ""}`, `include_dirs`, `structure_only` |
| CIF | `struct_conn=True`: add `_struct_conn` bonds (disulfides, covalent links, metal coordination; not hydrogen bonds or symmetry copies) |
| PDB | `conect=True`, `ssbond=True`, `link=True`: apply the file's CONECT, SSBOND and LINK records (msys ignores them) |

Writer options:

| Writer | Options |
|---|---|
| DMS | `structure_only` |
| MAE | `structure_only`, `allow_reorder_atoms`, `append` |
| PDB | `append`, `reorder` (group atoms by chain and residue), `models`, `conect` |
| SDF | `append`, `v3000` (automatic above 999 atoms or bonds) |
| GRO | `precision` |

Multi-entry files (SDF records, PDB models, MAE blocks) load into one
System with one ct each. Use `s.ct_atoms(i)` to take entry `i`.

### PDB bonds

Reading: bonds are guessed from distances (msys rules), then the file's
records are applied.

- `SSBOND` bonds the two cysteine SG atoms whatever their distance. Old
  structures often have S-S distances too long to guess; 1LYZ has 3 of 4.
  Bonds to a crystal symmetry copy (operator other than 1555) are skipped.
- `LINK` bonds the two named atoms (covalent links, metal coordination),
  also skipping symmetry copies. With alternate locations and no altloc in
  the record, every copy is linked.
- `CONECT` adds the listed bonds. An entry repeated two or three times sets
  the bond order, as PyMOL and Open Babel write it. Between atoms that have
  their own CONECT record, the records are authoritative: a guessed bond
  that CONECT does not list is removed (for example a bad heme geometry in
  1MBN). Bonds to atoms without records stay as guessed.

Pass `conect=False, ssbond=False, link=False` for exactly msys's behavior.

mmCIF files get the same bonds from `_struct_conn` (`struct_conn=False` to
skip them), so a structure read as PDB or as mmCIF has the same bonds.

Writing: one model. A system with several cts (after `append`, say) is
written as one model, so every program reads all of it. Only an ensemble
(several cts with the same atoms, as read from an NMR file) keeps one MODEL
per ct, and `models=True` forces msys's one MODEL per ct. With the default
`conect="auto"` the writer runs the reader's bond guessing on the written
coordinates and writes CONECT records only for the atoms where the guess
would differ (a covalent ligand link, a long S-S bond, a missing bond), with
bond orders as repeated entries. Reading the file back gives the same bonds.
`conect=True` writes records for every bonded atom, `conect=False` none.

### Topology files

```python
s = boonza.load("complex.prmtop", coordinates="complex.rst7")  # Amber
s = boonza.load("step5_input.psf", coordinates="step5_input.pdb")  # CHARMM
s = boonza.load("topol.top", coordinates="conf.gro", include_dirs=["/path/to/gromacs/top"])
```

- **Amber prmtop** follows msys's own converter table for table: merged
  Fourier dihedrals, SCEE/SCNB-scaled 1-4 pairs, Lorentz-Berthelot van der
  Waals, exclusions and CMAP. Charges are divided by 18.2223 and elements
  are guessed from masses as msys does (a hydrogen repartitioned to 3 u reads
  as helium). A `.top` file that starts with `%VERSION` is read as a prmtop.
- **PSF** files carry no parameters, so only the structure, charges, masses,
  CHARMM types (a `type` column) and bonds are read. For a CHARMM force field,
  build the OpenMM system from the PSF and parameter files and use
  `boonza.from_openmm`.
- **GROMACS topologies** are preprocessed as GROMACS does, and each molecule
  type is repeated as `[ molecules ]` says. Martini topologies from
  martinize2 read completely: G96 cosine angles (type 2), restricted bending
  (type 10), `[ constraints ]` and `virtual_sitesn` (centre of geometry, mass
  or weights, as `virtual_lc{n}`). Terms with no table (GROMOS quartic bonds,
  tabulated terms, CMAP, `virtual_sites2/3/4`) raise an error; `structure_only=True` reads atoms, residues and bonds. The
  default `#ifdef` choices apply (for example rigid water); pass `defines` to
  change them.

## Trajectories

```python
traj = boonza.open_trajectory("run.xtc", s)  # DCD, XTC, TRR, Amber NetCDF, Desmond DTR/STK
len(traj), traj.natoms
frame = traj[10]  # Frame: positions (natoms, 3), box (3, 3), time, step
view = traj[::10]  # lazy view; also index arrays and masks
block = traj.read()  # Frames: positions (nframes, natoms, 3), boxes, times, steps
block = traj.read(slice(0, 100), atoms="protein")
for chunk in traj.chunks(500, atoms="protein"):  # memory-bounded blocks
    ...
```

- Positions are float32 in Å. Boxes are float64 3×3 matrices, all zeros
  when the file has no cell.
- DCD is memory-mapped: CHARMM, NAMD and X-PLOR variants, fixed atoms,
  either byte order.
- XTC and TRR are read and written natively. The XTC compression is a numba
  port of GROMACS's xdrfile library: frames decode bit for bit and files are
  written byte-identical to GROMACS's. TRR frames that carry only
  velocities or forces have NaN positions.
- Amber NetCDF (`.nc`, `.ncdf`; restart files `.ncrst` as one frame) is
  read natively, from NetCDF-3 classic or 64-bit-offset files. NetCDF-4
  (HDF5) files are not supported.
- Desmond DTR directories and STK lists of them are read natively, in
  single or double precision. In an STK, a later run replaces the frames
  of an earlier one from its first time on, as in msys. DTRs that hold
  only energies or forces (no positions) are refused with an error.

Writing:

```python
with boonza.open_writer("out.dcd", s.natoms) as w:  # also .xtc, .trr, .nc
    for f in traj:
        w.write(f.positions, f.box)
```

Pass `positions=` (an array or a `Frames` block) to any analysis function
to run it over frames. See [Analysis](analysis.md).
