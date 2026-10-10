# How the dynamic200 presets were made

`dynamic200` (`boonza/data/sitemap/presets/dynamic200.json`) is
[static200](sitemap_static200.md) fine-tuned on frames of coarse-grained MD
runs of the same 200 training complexes, simulated without their ligand. It
is `boonza sitemap traj`'s default, because its frames are what `traj` reads.
`boonza sitemap structure` keeps static200, which does better on crystal
structures (below). Like static200, it is kept unchanged so that later
presets can be compared with it.

## The presets

Changes from static200 are in bold. The probe (valine's side-chain bead) and
the 2.0 A spacing were held fixed.

| | Martini 2 | Martini 3 | SIRAH |
|---|---|---|---|
| probe | AC2 | SC3 | Y4Cv |
| grid spacing (A) | 2.0 | 2.0 | 2.0 |
| `outside` | **1.77** (2.40) | 2.74 | **1.60** (1.48) |
| `ray_length` (A) | **9.41** (8.0) | **8.05** (8.0) | **9.06** (8.55) |
| `enclosure` | **0.52** (0.57) | **0.52** (0.50) | **0.59** (0.56) |
| `contact_form` | capped | full | full |
| `contact` (kcal/mol) | **-5.72** (-6.02) | 23.98 | **202.21** (138.32) |
| `neighbour_radius` (grid steps) | **1.65** (1.83) | **1.74** (1.92) | 1.43 |
| `neighbours` | **4** (2) | 3 | 5 |
| `min_points` | **2** (3) | **2** (3) | **6** (3) |
| `merge_gap` (A) | **8.20** (9.16) | 6.22 | **5.52** (4.73) |
| `merge_cap` (A^3) | **59.49** (76.11) | **117.22** (0) | **430.2** (276.93) |
| `min_volume` (A^3) | **57.49** (0.12) | **22.12** (34.8) | **8.44** (16.75) |
| SiteScore: intercept | -7.380 | -4.529 | -3.025 |
| SiteScore: sqrt(n) | 0.205 | 0.284 | 0.224 |
| SiteScore: enclosure | 8.014 | 2.691 | -0.520 |
| SiteScore: philic | -0.160 | -0.143 | -0.015 |
| search log, trial | `md_train200_martini2_halfNone_martini2_ft`, 92 | `md_train200_martini3_halfNone_martini3_ft`, 90 | `md_train200_sirah_halfNone_sirah_ft`, 78 |

The most telling changes:

- **Martini 2** looks for sites further from the beads' surface (`outside`
  2.40 to 1.77), but needs more neighbours per point and drops any site
  beyond the first that is under 57 A^3. Its SiteScore leans on enclosure
  far more (3.3 to 8.0).
- **Martini 3** merges neighbouring groups again (`merge_cap` 0 to 117 A^3),
  which static200 had switched off.
- **SIRAH** needs at least 6 points for a site instead of 3. That removes the
  tiny buried voids between SIRAH's beads that made static200's enclosure
  weight negative ([static200](sitemap_static200.md), "Why enclosure differs
  between models"). Enclosure's weight is now close to zero (-2.04 to -0.52).

Martini 3's `contact` (23.98) is static200's and lies above the top of the
range drawn on MD frames (19.30), so its contact filter is off.

## Training data: MD frames of the 200 complexes

**The runs.** `boonza swim` and `boonza md` were run on each prepared holo
complex (`fpocketSet_263/holo/<pdb>.mae`; see [static200](sitemap_static200.md))
in each model:

- the ligand left out (`--cg-selection "not resname LIG"`);
- 18 single-amino-acid probe types at 500 mM;
- 2 nm padding;
- 100 ns of production, a frame every 0.1 ns (1000 frames).

The runs are in `fpocketSet_263/run/<model>/<pdb>/SingleAminoAcid18/sim_000/md_solute`,
made by (`run/boonza_md.py`, one array task per complex and model):

```bash
boonza swim holo/<pdb>.mae --workdir <model>/<pdb>/SingleAminoAcid18/ --model <model> \
    --types 18 --ligands SingleAminoAcid18 --cg-selection "not resname LIG" --cofactors \
    --conc-mM 500.0 --padding-nm 2.0 --production-ns 100 \
    --production-report-interval-ns 0.1 --precision single
boonza md --config <model>/<pdb>/SingleAminoAcid18/sim_000/md.toml
```

Every finished trajectory was checked with `boonza info`: 1000 frames, and
the same atom count as its `solvated.dms`. 191 of the 200 runs finished in
each model. The ones that did not:

| | runs |
|---|---|
| all three models | 1b42, 1bx6, 1dy3, 1ha3, 1j21, 1jbw, 1tdf, 1ydt |
| Martini 2 only | 1b14 |
| Martini 3 only | 1qhg |
| SIRAH only | 1e0v |

**The frames** (benchmark `md_train_prepare.py`). From each run, 5 frames:
the last of each fifth (MD frames 199, 399, 599, 799 and 999, so 20 to
100 ns). That makes 955 frames per model.

- **Protein:** each frame made whole and fitted on the backbone, as
  `boonza sitemap traj` does. The probes are not part of the protein.
- **Grids:** computed as for static200, at 2.0 A, with the residue probes.
- **Truth:** the ligand's heavy atoms, carried onto each frame on its own by
  superposing the whole holo protein (`boonza sitemap traj --holo`'s fit).
  Median fit RMSD is about 1.2 A, and at most 1.43 A.

The ligand is absent from the run, so its site can narrow or close. On these
frames, static200's sites include one within 4 A of the ligand in only
68-79% of frames, against 86-91% on the crystal structures.

## How the search ran

The trial scoring and the local moves are static200's ([how a trial is
judged](sitemap_static200.md), [how the search
ran](sitemap_static200.md)). The 5-fold cross-validation holds a complex's 5
frames out together, so no protein is in both the fitting and the scoring
folds. Per model (benchmark `md_finetune.sh`, `tune.py --set
md_train200_<model> --probe <Val> --spacing 2.0 --tag _ft --local 100`):

- **Probe and spacing held:** Val, 2.0 A.
- **Contact ranges:** redrawn from the MD frames (2nd-80th percentile at
  enclosed points). Martini 2 AC2: full -8.23 to 33.65, capped -9.91 to -4.99.
  Martini 3 SC3: full -5.53 to 19.30, capped -6.49 to -3.05. SIRAH Y4Cv: full
  -5.96 to 262.94, capped -9.10 to -4.33.
- **Starts (9):** static200 as it is, and at both contact forms x the cutoff
  at 10/30/50/70% of its range.
- **Local (100):** 1-3 parameters of the best trial so far changed at a
  time, Gaussian steps of SD 20% of the range for the first 50 and 8% for
  the rest. 91 / 96 / 99 distinct trials (Martini 2 / 3 / SIRAH).
- **Best:** trials 92, 90 and 78.

## Result on the training frames

Cross-validated over the 955 frames of the 191 complexes (5 folds by
complex), PPc Top-1 / 3 / 5 / 10:

| model | static200 | dynamic200 | fill (static200 to dynamic200) |
|---|---|---|---|
| Martini 2 | 0.44 / 0.63 / 0.67 / 0.68 | 0.50 / 0.65 / 0.67 / 0.69 | 0.17 to 0.22 |
| Martini 3 | 0.58 / 0.74 / 0.78 / 0.79 | 0.60 / 0.76 / 0.79 / 0.82 | 0.27 to 0.27 |
| SIRAH | 0.51 / 0.66 / 0.73 / 0.76 | 0.55 / 0.69 / 0.72 / 0.74 | 0.25 to 0.27 |

Objective (½ x (½ x (Top-1 + Top-3) + fill)): Martini 2 0.351 to 0.397,
Martini 3 0.465 to 0.476, SIRAH 0.416 to 0.446.

## Validation

**MD frames of the Schrödinger set's apo runs** (hard set). These are 40
cases from 38 runs, 10 frames each (every 100th of 1000), never used for
tuning. Each case's ligand is carried onto its run by superposition. 400
frames in all, though frames of one run are not independent: the set is
closer to 40 cases than to 400. PPc Top-1 / 3 / 5 / 10, then MOc Top-1 /
3 / 5 / 10:

| model | preset | PPc | MOc | fill |
|---|---|---|---|---|
| Martini 2 | static200 | 0.32 / 0.46 / 0.52 / 0.54 | 0.14 / 0.17 / 0.18 / 0.18 | 0.11 |
| | dynamic200 | 0.36 / 0.51 / 0.55 / 0.56 | 0.19 / 0.23 / 0.24 / 0.25 | 0.14 |
| Martini 3 | static200 | 0.48 / 0.58 / 0.62 / 0.65 | 0.37 / 0.40 / 0.41 / 0.41 | 0.17 |
| | dynamic200 | 0.49 / 0.61 / 0.65 / 0.69 | 0.35 / 0.39 / 0.39 / 0.40 | 0.17 |
| SIRAH | static200 | 0.31 / 0.44 / 0.52 / 0.59 | 0.19 / 0.26 / 0.27 / 0.29 | 0.13 |
| | dynamic200 | 0.39 / 0.47 / 0.55 / 0.59 | 0.29 / 0.32 / 0.34 / 0.35 | 0.16 |

**Crystal apo structures**, PPc Top-1 / 3 / 5 / 10:

| model | preset | 40 apo/holo pairs (easy) | Schrödinger's 41 apo (hard) |
|---|---|---|---|
| Martini 2 | static200 | 0.68 / 0.85 / 0.85 / 0.88 | 0.49 / 0.56 / 0.63 / 0.66 |
| | dynamic200 | 0.60 / 0.85 / 0.88 / 0.90 | 0.41 / 0.56 / 0.59 / 0.59 |
| Martini 3 | static200 | 0.78 / 0.90 / 0.90 / 0.90 | 0.49 / 0.61 / 0.63 / 0.63 |
| | dynamic200 | 0.75 / 0.90 / 0.93 / 0.93 | 0.39 / 0.54 / 0.56 / 0.56 |
| SIRAH | static200 | 0.70 / 0.82 / 0.88 / 0.88 | 0.54 / 0.59 / 0.61 / 0.61 |
| | dynamic200 | 0.65 / 0.75 / 0.78 / 0.80 | 0.56 / 0.63 / 0.63 / 0.71 |

dynamic200 is better on MD frames and mostly worse on crystal structures,
hence one preset per command.

Per-frame site finding cannot recover a site that is closed in a frame. In
MD, Top-1 is well below the crystal structures' for both presets. Grouping
sites across frames into pockets (`boonza sitemap traj`'s consensus) is where
a run's open frames can count, and its parameters are not tuned yet.

## Reproducing

In `~/sitemap/benchmark`:

```bash
./md_finetune.sh       # md_train_prepare.py --frames 5, then tune.py per model
python md_add_probes.py --model martini2 martini3 sirah
for m in martini2 martini3 sirah; do
    python validate.py md_$m --kind md --model $m --presets presets/dynamic200.json
done
```
