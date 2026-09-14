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
| SDF / MOL (`.sdf`, `.mol`, compressed) | ✓ | ✓ | V2000 and V3000, data fields, charges, isotopes, stereo flags; matches msys |

Reader options:

| Reader | Options |
|---|---|
| DMS | `structure_only` (skip force field), `without_tables` |
| MAE | `structure_only`, `without_tables`, `ignore_unrecognized` |
| PDB, GRO, CIF | `guess_bonds=True` (msys rules, same ct only) |
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

## Trajectories

```python
traj = boonza.open_trajectory("run.xtc", s)  # DCD or XTC
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
- XTC uses the compiled XDR reader that ships with MDAnalysis.

Writing:

```python
with boonza.open_writer("out.dcd", s.natoms) as w:
    for f in traj:
        w.write(f.positions, f.box)
```

Pass `positions=` (an array or a `Frames` block) to any analysis function
to run it over frames. See [Analysis](analysis.md).
