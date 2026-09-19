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

Chains are also cut where the C-N distance between consecutive residues is
over 2.5 Å, or where a residue lacks backbone atoms, as the DSSP program
does. mdtraj uses the topology's chains only; `breaks=False` reproduces it
exactly. With `box=`, or with Frames and trajectories that carry boxes,
distances use the minimum image, so a protein split across the box needs no
glueing. A PDB file's CRYST1 cell is not used unless you pass it, because it
would add contacts to symmetry mates.

```python
codes = boonza.dssp(s, traj)  # boxes from the trajectory
codes = boonza.dssp(s, frames, box=cell, breaks=True)
```

## Native contacts

```python
q = boonza.native_contacts(s, "segid A", "segid B", positions=frames)  # hard_cut
q = boonza.native_contacts(
    s, sel1, sel2, positions=frames, reference=crystal, method="soft_cut", radius=4.5
)
```

This is the fraction of native contacts Q per frame, as MDAnalysis
`Contacts` computes it. Native contacts are the pairs within `radius` in the
reference: the system's own coordinates, or another System. `method`:

- `"hard_cut"`: counts r ≤ r0;
- `"radius_cut"`: counts r ≤ radius;
- `"soft_cut"`: Best-Hummer-Eaton's 1 / (1 + exp(β (r − λ r0))).

Distances use each frame's box.

## Contact frequencies

```python
rows, cols, freq = boonza.contact_frequency(s, "protein", "resname LIG", positions=frames)
rows, cols, freq = boonza.contact_frequency(s, "protein", positions=frames, level="atom")
```

This gives the fraction of frames in which each pair is in contact. A
residue pair counts when any of its atoms are within `cutoff`. `rows` and
`cols` are residue (or atom) indices. Without a second selection, pairs are
within the first one, excluding a residue with itself.

## Principal components

```python
p = boonza.pca(s, frames, sel="name CA")  # every frame fitted on the first, as MDAnalysis
p.variance, p.cumulated_variance  # Å², largest first
p.components  # (ncomponents, 3 natoms), unit vectors
p.projections  # the frames along each component
p.transform(s, other_frames)  # project other frames the same way
```

The variances and components match MDAnalysis `PCA`. They come from a
singular value decomposition of the centered frames, so large selections do
not build a 3N x 3N matrix.

## Block averaging

```python
b = boonza.block_average(rmsd_per_frame)
b.mean, b.block_sizes, b.sem, b.sem_error
b.estimate  # the plateau: the standard error of the mean
b.statistical_inefficiency  # frames per independent sample
```

This is Flyvbjerg-Petersen blocking. The standard error estimated from
blocks grows with the block size until blocks are longer than the
correlation time, then levels off at the true error.

```python
start = boonza.settled(rmsd_per_frame)  # where to start: the drift before it is noise
```

`settled` scores every start by how many independent samples it leaves --
remaining frames over the statistical inefficiency -- and returns the best,
rather than discarding a guessed percentage. It assumes the series relaxes
to one stationary distribution.

## Representative poses

```python
p = boonza.poses(s, frames, ligand="chain L")  # most populated first
p[0].center  # the medoid: a frame that was simulated, never an average
p[0].population  # its share of the frames
p[0].spread  # how tightly its members sit around it (Å)
p[0].frames  # the members, in time order
p.labels  # the pose of every frame, -1 below min_population
print(p.summary())
```

Frames are compared by the distances between pocket atoms and ligand atoms —
the measure `boonza.drmsd` uses — so nothing is superposed and a protein
that breathes or tumbles does not look like a ligand that moved. Ligand
symmetry is taken out once per frame, so a rotated ring is not a second
pose.

The frames are grouped by average linkage, which builds the whole merge tree
in one pass. `cutoff` only says where to cut it, and the other cutoffs come
off the same tree for nothing:

```python
cutoffs, share, count = p.sweep()  # the largest pose's share, and how many poses
```

A pose whose share holds while the cutoff doubles is a real one, so the
choice of cutoff can be shown rather than trusted.

From the command line, with the structures written out:

```
boonza poses solvated.dms --traj trajectory.dcd -o poses/
```

```
pocket taken from frame 15, where 34 atoms are within 5 A of the ligand
2 poses of 500 frames, cut at 1.5 A dRMSD
pose   frame   share  spread  frames
   0     109   60.8%   0.25 A     304
   1     388   32.0%   0.17 A     160
```

The pocket comes from a *typical* bound frame, not from frame 0 and not
from the most contacting one: one pass counts how often the ligand touches
the protein, and the median of the frames that do is used. The most
contacting frame is an outlier by construction, and frame 0 may have the
ligand somewhere else entirely. `--reference crystal.pdb` fixes the choice,
and `--pocketsel` skips it — the atoms you name are then the pocket, whatever
any frame says. Each pose is written out as a
structure, next to a `poses.json` of the populations, the members and the
sweep.

If the ligand visits several separate patches of the protein, it says so
rather than grouping everything against one of them — `boonza sites` is what
separates them.

```python
counts, share = boonza.pocket_contacts(s, traj)  # per frame, and per protein atom
boonza.bound_frame(counts)  # a median bound frame
```

`--settle` drops the drift at the start of a run, by asking where the series
becomes stationary (`boonza.settled`). It suits one ligand settling into one
pose. It is off by default because a run that genuinely *changes* pose looks
non-stationary too, and everything before the change would be thrown away --
including, often, the most populated pose.

## Binding sites across runs

```python
runs = [boonza.open_trajectory(p, s) for p in paths]
found = boonza.sites(s, runs, ligand="resname LIG")

# or, when the runs hold different ligands: each brings its own system
runs = [(boonza.load(f"{d}/solvated.dms"), boonza.open_trajectory(f"{d}/trajectory.dcd"))
        for d in workdirs]  # fmt: skip
found = boonza.sites(runs[0][0], runs, ligand="chain LIG")
found[0].occupancy  # share of all pooled frames
found[0].runs  # how many independent runs visited it
found[0].arrivals  # separate visits, not frames
found.labels  # site of every pooled frame, -1 for bulk
print(found.summary())
```

Where `boonza.poses` asks *how* a ligand sits in one pocket, `sites` asks
*where* it goes at all. Every frame of every ligand copy of every run
contributes one point — the ligand's heavy-atom centroid, with the protein
superposed on a common reference so runs can be compared. The points are
counted onto a grid, and a site is a connected region visited at least
`enrichment` times more often than bulk solvent would explain. Everything
else is labelled -1 rather than forced into a site.

The threshold is an enrichment over bulk, not a number of frames, so it
means the same thing whatever the box size, run length or copy count. Bulk
is worked out from the box the frames actually had, averaged over them, not
from the cell the structure file carries — under a barostat those differ,
and `found.volume` is the one that was used.

The two levels compose: a site says which frames to look at, and those go to
`poses` for the pose within it. Alignment is used only here, where pockets
are many ångströms apart; the pose measure needs none.

```python
rows = found.frames(0, run=0)  # (run, copy, frame) of site 0 in run 0
copy = rows[0, 1]
frames = rows[rows[:, 1] == copy][:, 2]
p = boonza.poses(s, runs[0][frames], ligand=f"fragid {copy} and noh")
```

Ligand copies usually share a residue name and number, so `fragid` is what
tells them apart.

A site's pocket is worth taking from the site rather than from any one
frame:

```python
pocket = boonza.site_pocket(s, runs, found, 0, protein="protein and noh")
p = boonza.poses(s, runs[0][frames], ligand=f"fragid {copy} and noh", pocket=pocket)
```

`site_pocket` keeps the atoms the ligand touches in at least `share` of that
site's frames, counted across runs and copies. Given `pocket=`, the pose
level uses those atoms and nothing else — no protein selection, no cutoff,
no reference frame.

A site is made of ligand *centres*, so only the protein has to be shared:
the ligands themselves may be different molecules, in different numbers and
of different sizes, from one run to the next. That is what lets a screen of
several fragment sets be pooled into one answer — and a site occupied by two
molecules out of a hundred is a different finding from one that fifty visit.

From the command line, either one system with several trajectories, or a
work directory per run:

```
boonza sites solvated.dms --traj run*.dcd -o sites/
boonza sites --workdir swim/sim_*/md -o sites/      # each with its own solvated.dms
```

```
3 runs, 1350 pooled frames; 47.3% in bulk
site  occupied  runs  copies  arrivals  spread  centre
   0     26.4%     3       2         6    0.7 A     -2.0     7.0     1.0
   1     26.3%     3       2         7    0.7 A      6.0    -0.0     0.0
```

### The density map

`sites` counts the centroids onto a grid on its way to finding sites, and
keeps it:

```python
grid = found.density
grid.enrichment  # (nx, ny, nz), 1 = as often as bulk explains
grid.write_dx("density.dx")  # ChimeraX, VMD and PyMOL read this
```

It is worth having because it finds the sites *without clustering*: the
peaks of the map and the centres of the clusters are two different
calculations, so when they agree you can believe both. `boonza sites -o`
writes it beside `sites.json`.

## Kinetics of a site

```python
found = boonza.sites(s, runs)
rate = boonza.kinetics(s, found, 0, interval_ns=0.1)
rate.residence_ns, rate.dG, rate.dG_interval, rate.events
print(rate.summary())
```

```
site 0: dG -1.65 kcal/mol [-1.75, -1.54] from 153 departures and 136 arrivals
  residence 19.54 ns, KD 68.5 mM, [L] 25.9 mM, bound for 2989 of 10000 ns
  occupancy 0.299 by frames, 0.275 by rates
```

A site's frames are a two-state series per copy, and the dwells between
transitions give the rates: `k_off` from completed departures over bound
time, `k_on` from completed arrivals over unbound time weighted by the free
ligand concentration, which is counted per frame and falls as copies bind. That concentration uses the mean box of the frames, because
`K_D` goes as 1/V: a box 10% away from the one assumed shifts `dG` by
`RT ln(1.1)`, about 0.06 kcal/mol — small, but a bias rather than noise.
Pass `volume_A3` to override it.

Three things decide whether those numbers mean anything.

**Hysteresis.** A copy arrives within `r_in` of the site and leaves only past
`hysteresis` times that. With a single boundary the ligand recrosses it
constantly and every crossing is counted as a departure. On a test process
that really departs 138 times, one boundary finds 2949 and reports a
residence time of 0.97 ns instead of 20 ns; two boundaries find 132.

Worth knowing which way this bites: `dG` moved by 0.06 kcal/mol across that
whole range, because `k_on` and `k_off` inflate together and the ratio
survives. **The boundary hardly matters for thermodynamics and decides
everything for kinetics.**

**Censoring.** A run ends because the wall clock ran out or because
`boonza md --early-stop` saw the ligand leave — not because the dwell ended.
Such a stretch counts its time but not an ending, which is the
maximum-likelihood treatment and the reason an early-stopped run biases
nothing.

**Evidence.** Every number carries the count it rests on. `dG_interval` comes
from resampling whole runs with replacement, so it carries run-to-run
disagreement that a Poisson count cannot see, and `consistent` checks the
occupancy the rates imply against the occupancy the frames show — two routes
to one number, so disagreement means the states are wrong.

From the command line, add `--interval-ns` to `boonza sites`:

```
boonza sites solvated.dms --traj run*.dcd --interval-ns 0.1 -o sites/
```

## Rings

```python
rings = boonza.sssr(s)  # msys GetSSSR: rings in ring order
rings = boonza.sssr(s, all_relevant=True)
boonza.ring_systems(s, rings)  # fused ring systems
```
