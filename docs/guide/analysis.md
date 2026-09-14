# Analysis

Analysis functions take a System plus, optionally, frames. By default they
use the system's own positions and cell. You can also pass one `(natoms, 3)`
array, an `(nframes, natoms, 3)` array, or a `Frames` block from
`traj.read()` or `traj.chunks()`. Each follows a reference tool exactly.

```python
traj = boonza.open_trajectory("run.xtc", s)
frames = traj.read()
```

## RMSD, RMSF, radius of gyration

```python
boonza.rmsd_trajectory(s, frames, "protein and name CA")  # per frame, fitted
boonza.rmsd_trajectory(s, frames, "backbone", reference=ref_xyz, weights="mass")
boonza.rmsf(s, frames, "protein and name CA")  # per atom
boonza.radius_of_gyration(s, frames, "protein")  # mass weighted
boonza.radius_of_gyration(s, frames, "protein", weights=None)  # mdtraj's default
```

These match MDAnalysis's `RMSD` and `RMSF` and its `radius_of_gyration`
(mdtraj's `compute_rg` with `weights=None`). As in MDAnalysis, `rmsf` uses
frames as given. Superpose them first, for example with
`boonza.Glue(s, fit=...)`.

## Radial distribution function

```python
r, g, edges, counts = boonza.rdf(
    s, "name OW", "name OW", frames, nbins=75, range=(0.0, 15.0), exclusion_block=(1, 1)
)
```

This matches MDAnalysis's `InterRDF`:

- `norm`: `"rdf"`, `"density"` or `"none"`;
- `exclusion_block`: skip pairs within blocks of consecutive atoms;
- `exclude_same`: skip pairs in the same `"residue"`, `"chain"` or
  `"fragment"`.

## Residue contacts

```python
dist, pairs = boonza.residue_contacts(s, frames, scheme="closest-heavy")  # Å, (nframes, npairs)
dist, pairs = boonza.residue_contacts(s, contacts=[[0, 10], [5, 20]], scheme="ca")
```

This matches mdtraj's `compute_contacts`: pairs of residues in the same chain
at least three apart, periodic distances.

## Solvent accessible surface area

```python
area = boonza.sasa(s)  # (nframes, natoms), Å²
res_area = boonza.sasa(s, mode="residue")
boonza.sasa(s, probe_radius=1.4, n_sphere_points=960, radii={"Fe": 2.0})
```

This is mdtraj's Shrake-Rupley, with the same float32 sphere points and
radii. Totals agree with mdtraj to 0.01%.

## Hydrogen bonds

```python
hb = boonza.hbonds(s, frames)  # MDAnalysis criteria
hb = boonza.hbonds(s, frames, between=["protein", "water"])
hb.donor, hb.hydrogen, hb.acceptor, hb.distance, hb.angle, hb.frame
hb.per_frame()  # count per frame
hb.frequency()  # {(donor, hydrogen, acceptor): fraction of frames}

boonza.baker_hubbard(s, frames)  # mdtraj: H···A < 2.5 Å, D-H···A > 120°, present > 10% of frames
boonza.wernet_nilsson(s, frames)  # mdtraj's distance-angle cone, per frame
```

`hbonds` follows MDAnalysis's `HydrogenBondAnalysis`:

- donor–hydrogen pairs come from bonds;
- the D–A distance must be in (1, 3.0] Å;
- the D–H···A angle must be over 150°.

Distances use the true minimum image in any cell. Defaults are N and O donors
and acceptors, and `donors`, `hydrogens` and `acceptors` take selections to
change them. There is no msys-based H-bond finder.

## Secondary structure and backbone angles

```python
codes = boonza.dssp(s, frames)  # (nframes, nresidues): H B E G I T S ' ' / NA
codes = boonza.dssp(s, simplified=True)  # H, E, C
phi, psi, omega = boonza.backbone_dihedrals(s, frames)  # degrees, NaN at chain ends and breaks
donor, acceptor, energy = boonza.backbone_hbonds(s)  # Kabsch-Sander backbone H-bonds
ss = boonza.chimerax_ss(s)  # 'H', 'S', 'O' as ChimeraX assigns (used by matchmaker)
```

- `dssp` is mdtraj's DSSP 2.2, with the same float32 arithmetic.
- `backbone_dihedrals` matches mdtraj's `compute_phi/psi/omega`.
- `chimerax_ss` reproduces ChimeraX's own assignment, which differs slightly
  from DSSP 2.2.

## Rings

```python
rings = boonza.sssr(s)  # msys GetSSSR: rings in ring order
rings = boonza.sssr(s, all_relevant=True)
boonza.ring_systems(s, rings)  # fused ring systems
```
