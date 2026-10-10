# How the static200 presets were made

`static200` (`boonza/data/sitemap/presets/static200.json`, `--preset
static200`) is the set of `boonza sitemap` presets tuned on 200 static
protein-ligand crystal structures alone, before any fine-tuning on MD frames.
It is kept unchanged so that later presets can be compared with it. This page
records the training data, what was tuned, how the search ran and how the
result did on the validation sets.

The search code is not part of boonza: it is the cgsitemap benchmark
(`~/sitemap/benchmark`: `prepare_mae.py`, `tune.py`, `validate.py`), which
runs boonza's own site finding (`boonza.sitemap`) on precomputed grids. Its
trial logs are named below.

## The presets

| | Martini 2 | Martini 3 | SIRAH |
|---|---|---|---|
| probe (valine side chain) | AC2 | SC3 | Y4Cv |
| grid spacing (A) | 2.0 | 2.0 | 2.0 |
| `outside` | 2.40 | 2.74 | 1.48 |
| `ray_length` (A) | 8.0 | 8.0 | 8.55 |
| `enclosure` | 0.57 | 0.50 | 0.56 |
| `contact_form` | capped | full | full |
| `contact` (kcal/mol) | -6.02 | 23.98 | 138.32 |
| `neighbour_radius` (grid steps) | 1.83 | 1.92 | 1.43 |
| `neighbours` | 2 | 3 | 5 |
| `min_points` | 3 | 3 | 3 |
| `merge_gap` (A) | 9.16 | 6.22 | 4.73 |
| `merge_cap` (A^3) | 76.11 | 0.0 | 276.93 |
| `min_volume` (A^3) | 0.12 | 34.8 | 16.75 |
| SiteScore: intercept | -5.756 | -6.478 | -1.983 |
| SiteScore: sqrt(n) | 0.373 | 0.290 | 0.322 |
| SiteScore: enclosure | 3.286 | 5.777 | -2.040 |
| SiteScore: philic | 0.055 | -0.286 | 0.008 |
| search log, trial | `holo200b_halfNone_martini2_val`, 183 | `holo200b_halfNone_martini3_val`, 31 | `holo200b_halfNone_sirah_val`, 121 |

A site's probability of being the ligand's site is the logistic of its
SiteScore: intercept + w1 sqrt(n) + w2 enclosure + w3 philic, with n the
site's volume (A^3), enclosure its share of rays that enter the protein within
10 A, and philic the probe's mean Lennard-Jones energy with the polar beads
(sign flipped).

Two values sit at an edge of what the search allowed, which matters when
reading them:

- Martini 3's `contact` (23.98) is the top of its range: the contact filter is
  as loose as it can be and removes only points where the probe clashes
  badly. SIRAH's (138.32, in a range up to 308) is loose as well. Only
  Martini 2's capped cutoff (-6.02) is a real requirement of contact.
- Martini 3's `merge_cap` is 0: neighbouring groups are never merged.

## Training data: 200 holo complexes

The structures are `~/Dropbox/PocketFinding/fpocketSet/fpocketSet_263/holo/<pdb>.mae`
(its README has the details). They come from fpocket's training set of 263
protein-ligand complexes (fpocket-data-1.0,
<https://sourceforge.net/projects/fpocket/files/fpocket-1.0/fpocket-src-1.0/fpocket-data-1.0.tgz/download>):

1. **No protein shared with validation.** Every chain of every structure of the
   263, of pp48 (apo and holo) and of the Schrödinger set (apo and holo) was
   aligned with every other. A complex was dropped when any chain is at least
   90% identical to a validation structure: 31 complexes, leaving 232. Eight
   homologues at 30-90% identity were kept.
2. **Incomplete or unusable sites set aside.** The original RCSB entries were
   compared with the files. 32 were set aside: 28 whose site is partly made by
   an assembly partner or symmetry copy the file lacks (`partner_missing`,
   `symmetry_copy_missing`) or whose ligand is split
   (`ligand_split`), and 4 whose ligand is a crystallisation additive. Four
   with drug-like ligands were then taken back (1a9p, 1aog, 1k0b, 1lp6),
   making 200. 1a9p is 87% identical to pp48's 1ulb, so the 1ula-1ulb pair
   was dropped from the validation set instead.
3. **Prepared.** Each complex was prepared with Schrödinger's prepwizard and
   hydrogens added. Each file keeps one protein chain (two where the site needs
   both) and one ligand, renamed `LIG`. Structural metals are kept, with no
   bonds to them. Water, cosolvents, glycans and exposed cofactors are removed,
   and N-termini are charged. Three structures with truncated side chains
   (1e0v, 1ftk, 1qhg) and one with an unnamed terminal oxygen (1c1b) were
   fixed by hand before the final search.

For each complex (`prepare_mae.py --name holo200b --ligand "resname LIG"
--spacing 2.0 1.5`):

- **Protein.** `protein and not (resname LIG)` is mapped to beads by boonza
  (Martini 2 and 3 by `martinize`, SIRAH by its own map, each bead typed as
  the parameterized system would be). The ligand, ions and metals are not part
  of the wall.
- **Grids.** Every grid point's numbers are computed once at 2.0 and 1.5 A
  (`boonza.sitemap.grid`): how far outside the beads it is, 110 rays' first
  hits, and the Lennard-Jones energy of each probe (full and capped) from the
  force field's pair table (Martini 2.2, Martini 3.0.0, SIRAH 2.2). Only the
  three residue-defined probes were computed: Val, Leu/Ile and Met side-chain
  beads (Martini 2 AC2/AC1/C5, Martini 3 SC3/SC2/C6, SIRAH Y4Cv/Y1C/Y3Sm).
- **Truth.** The ligand's heavy atoms, and the protein residues within 5 A of
  them.

All 200 are listed at the end of this page.

## What was tuned

Each trial is one set of site-finding parameters. The parameters, the ranges
they were drawn from, and SiteMap's own value for reference:

| parameter | what it does | range searched | SiteMap |
|---|---|---|---|
| spacing | grid spacing, A | 2.0 or 1.5 | 1.0 |
| probe | probe bead | the three residue beads; held at Val in the final search | -- |
| `outside` | min d^2/R^2 over beads for a point to count as outside the protein | 1.2-4.0 | 2.5 |
| `ray_length` | how far a ray looks for the protein, A | 4-10 | 8 |
| `enclosure` | share of the 110 rays that must hit within `ray_length` | 0.3-0.8 | 0.5 |
| `contact_form` | probe energy with every pair (full) or each pair capped at 0 (capped) | full, capped | -- |
| `contact` | most the probe's energy may be, kcal/mol | per probe and form: 2nd-80th percentile of enclosed points' energies (below) | -1.1 (atoms) |
| `neighbour_radius` | radius for counting neighbouring candidate points, in grid steps | 1.0-2.3 | 1.76 on a 1 A grid |
| `neighbours` | candidates a point needs within that radius | 0-8 | 3 |
| `min_points` | smallest group kept, in points | 1-8 | 3 |
| `merge_gap` | groups whose nearest points are this close may merge, A | 2-10 | 6.5 |
| `merge_cap` | ...as long as one of them is at most this large, A^3 | 0-600 | 100 points |
| `min_volume` | sites after the first need at least this volume, A^3 | 0-200 | 15 points |

Not searched:

- `max_sites` (10);
- the 110 rays, 1 A ray steps and the 12 A Lennard-Jones cut-off;
- the property definitions (Halgren 2009).

The SiteScore weights are not searched either: they are refitted for every
trial (next section).

The `contact` ranges are set per probe and form because energies differ by
model. They are the 2nd and 80th percentiles of the energy at enclosed,
outside points (`outside` >= 2.5, half the rays hitting within 8 A) of 24
evenly picked training grids. For the Val probe:

| model | full | capped |
|---|---|---|
| Martini 2 (AC2) | -8.07 to 34.26 | -10.44 to -5.15 |
| Martini 3 (SC3) | -4.94 to 23.98 | -6.31 to -3.13 |
| SIRAH (Y4Cv) | -5.94 to 308.45 | -9.72 to -4.61 |

## How a trial is judged

For one parameter set (`tune.py`, `judge`):

1. The sites of all 200 complexes are found. A site counts as **right** if
   its centre (centroid of its points) lies within 4 A of a ligand heavy atom
   (PPc, fpocket's criterion).
2. SiteScore's three terms are fitted by logistic regression: right sites
   against the rest, features standardised, L2 with C = 1, balanced class
   weights.
3. **Cross-validation.** 5-fold GroupKFold, each complex held out whole. The
   weights fitted on four folds rank the fifth fold's sites. For each complex
   this gives the rank of its first right site, and the overlap of its top site
   with the ligand.
4. **Fill** = sqrt(LVC x PVN) of the cross-validated top site if it is right,
   else 0, averaged over the 200. On a common voxel lattice, LVC is the share
   of the ligand's volume inside the site, and PVN the share of the site within
   2 A of the ligand. Fill rewards a site that holds the ligand without
   reaching far beyond it. PPc alone favours small sites centred on the ligand
   (with PPc alone, right top sites held only about 30% of the ligand's
   volume).
5. **Objective** = ½ x (½ x (CV Top-1 + CV Top-3) + fill).

The preset's SiteScore weights are the fit on all 200 complexes for the best
trial.

## How the search ran

Each trial costs only thresholding and grouping on the precomputed grids.

**Random trial** (`sample`):
- spacing, probe and form are drawn uniformly from their choices;
- every numeric parameter is drawn uniformly over its range (integers
  inclusive);
- `contact` is drawn uniformly over its probe and form's range.

**Local trial** (`perturb`): a copy of the best trial so far, with 1-3
randomly chosen parameters changed.
- A continuous parameter takes a Gaussian step of SD 20% of its range, clipped
  to the range. The step is SD 8% (scale 0.4) for the second half of the local
  trials.
- An integer parameter steps ±1.
- spacing is redrawn.
- A change of probe or form redraws `contact` within the new range.

Each trial's sites and score are logged (one JSON line each), so a search
resumes where it stopped. A parameter set already tried is not run again.

The presets came out of three stages, the last two each starting from the one
before.

### Stage 1: a fresh search with the probe free

Run on the first version of the 200 prepared structures (`holo200`; 197
usable: the three with truncated side chains failed). The probe was free among
the three residue beads.

- One trial at SiteMap's own settings, then 100 random trials, then local
  trials (300 requested; 272-291 distinct).
- Best objective: Martini 2 0.623 (AC2), Martini 3 0.652 (C6), SIRAH 0.594
  (Y1C).
- Random sampling alone stayed below SiteMap's settings for Martini 2 and 3
  (best random 0.538 and 0.569, against 0.586 and 0.582). The local search
  did the work.
- Logs: `runs/tune/holo200_halfNone_<model>.jsonl`.

### Stage 2: the same effort for each probe

The probes were compared on equal effort so that one residue could be chosen
for all models.

- Every model x probe started from that model's stage-1 best, with the probe
  held fixed.
- 8 start trials: both contact forms x the cutoff at 10/30/50/70% of its range.
- Then 150 local trials.
- Logs: `runs/tune/holo200_halfNone_<model>_fixed<probe>.jsonl`.

| model | Val | Leu/Ile | Met |
|---|---|---|---|
| Martini 2 (AC2 / AC1 / C5) | 0.614 | 0.613 | 0.610 |
| Martini 3 (SC3 / SC2 / C6) | 0.650 | 0.650 | 0.650 |
| SIRAH (Y4Cv / Y1C / Y3Sm) | 0.595 | 0.595 | 0.595 |
| mean | **0.620** | 0.619 | 0.618 |

The three residues tie. SIRAH's three are identical because its best contact
cut-off filters almost nothing, so the probe hardly matters. Val was chosen
for all three models.

### Stage 3: the final search (Val), on the cleaned-up structures

The four structures with defects were fixed (`holo200b`; all 200 usable) and
the grids recomputed. Per model, with the probe held at Val:

- **Starts (27).** Three starting points, each run as it was and at both
  forms x 4 cutoffs (as in stage 2):
  - the best stage-1 trial with its probe set to Val;
  - the best stage-2 Val trial;
  - the presets installed before, with their probe set to Val.
- **Local (250 requested).** 230 / 215 / 232 distinct trials for Martini 2,
  Martini 3 and SIRAH.
- **Best.** Trials 183 (Martini 2), 31 (Martini 3) and 121 (SIRAH).
  Martini 3's best was the 5th local trial, and the remaining ~210 did not
  beat it.
- Logs: `runs/tune/holo200b_halfNone_<model>_val.jsonl` (`holo200b_search.sh`,
  `holo200b_resume.sh`).

All three best trials use 2.0 A spacing; the 1.5 A grids did not win for any
model.

The presets installed before (N0 / SC3 / Y1C) came from an earlier search
with the same objective. That search used PDBFixer-rebuilt, heavy-atom PDB
files of these complexes, with each model's own probe beads.

## Result on the training set

Cross-validated over the 200 complexes (5 folds by complex):

| model | Top-1 | Top-3 | Top-5 | Top-10 | fill | a right site at all |
|---|---|---|---|---|---|---|
| Martini 2 | 0.785 | 0.875 | 0.895 | 0.905 | 0.43 | 0.905 |
| Martini 3 | 0.800 | 0.880 | 0.890 | 0.895 | 0.46 | 0.895 |
| SIRAH | 0.735 | 0.845 | 0.855 | 0.860 | 0.42 | 0.860 |

"A right site at all" caps Top-k: for the remaining complexes, no site has
its centre within 4 A of the ligand. Median fill over the right top sites is
0.56-0.58 (LVC 0.40-0.51, PVN 0.70-0.81). Martini 2 finds 5.9 sites per
complex on average, Martini 3 4.9 and SIRAH 5.3.

## Validation

Neither validation set was used to tune anything. Both are crystal apo
structures, the ligand carried onto each from its holo partner by superposing
the whole protein. PPc is the criterion of the search; MOc is fpocket's
mutual overlap (more than half of the ligand's atoms within 3 A of a site
point, and more than a fifth of the site points within 3 A of the ligand).
Before = the presets installed before static200 (N0 / SC3 / Y1C).

**Easy: 40 apo/holo pairs** (`~/Dropbox/PocketFinding/fpocketSet/LigSite48`,
from fpocket's pp48):

| model | presets | PPc Top-1/3/5/10 | MOc Top-1/3/5/10 | fill |
|---|---|---|---|---|
| Martini 2 | before | 0.62 / 0.85 / 0.85 / 0.88 | 0.57 / 0.80 / 0.80 / 0.80 | 0.36 |
| | static200 | 0.68 / 0.85 / 0.85 / 0.88 | 0.60 / 0.78 / 0.78 / 0.80 | 0.38 |
| Martini 3 | before | 0.85 / 0.88 / 0.93 / 0.93 | 0.75 / 0.78 / 0.82 / 0.82 | 0.49 |
| | static200 | 0.78 / 0.90 / 0.90 / 0.90 | 0.70 / 0.82 / 0.82 / 0.82 | 0.45 |
| SIRAH | before | 0.65 / 0.88 / 0.90 / 0.90 | 0.62 / 0.82 / 0.82 / 0.82 | 0.31 |
| | static200 | 0.70 / 0.82 / 0.88 / 0.88 | 0.62 / 0.72 / 0.78 / 0.78 | 0.41 |

**Hard: the Schrödinger set's 41 apo structures**
(`~/Dropbox/PocketFinding/SchrodingerSet`):

| model | presets | PPc Top-1/3/5/10 | MOc Top-1/3/5/10 | fill |
|---|---|---|---|---|
| Martini 2 | before | 0.54 / 0.59 / 0.59 / 0.59 | 0.41 / 0.46 / 0.46 / 0.46 | 0.25 |
| | static200 | 0.49 / 0.56 / 0.63 / 0.66 | 0.37 / 0.39 / 0.41 / 0.41 | 0.21 |
| Martini 3 | before | 0.44 / 0.56 / 0.56 / 0.59 | 0.41 / 0.49 / 0.49 / 0.49 | 0.19 |
| | static200 | 0.49 / 0.61 / 0.63 / 0.63 | 0.39 / 0.49 / 0.49 / 0.49 | 0.19 |
| SIRAH | before | 0.44 / 0.56 / 0.59 / 0.63 | 0.34 / 0.39 / 0.39 / 0.39 | 0.17 |
| | static200 | 0.54 / 0.59 / 0.61 / 0.61 | 0.37 / 0.41 / 0.41 / 0.41 | 0.23 |

One case is 0.025 (easy) or 0.024 (hard). Most differences from the presets
before are 1-4 cases, so accuracy is about unchanged. The gain is a probe
that is the same residue in every model. The one consistent loss is Martini
3's Top-1 on the easy set (0.85 to 0.78, 3 cases).

Validation cases (`validate.py <set> --kind apo --presets <file> --cases
.../LigSite48/pairs.txt`):

- easy (apo-holo): 1qif-1acj 3app-1apu 1hsi-1ida 1psn-1pso 1l3f-2tmn 3tms-1bid
  8adh-1cdo 1hxf-1dwd 1gcg-1gca 1hel-1hew 1npc-1hyt 1esa-1inc 1brq-1rbp
  8rat-1rob 1swb-1stp 1ifb-2ifb 3ptn-3ptb 1ypi-2ypi 5dfr-4dfr 2ctv-5cna
  5cpa-7cpa 1a6u-1a6w 1djb-1blh 1bya-1byb 1cge-1hfc 1a4j-1igj 1ime-1imb
  1nna-1ivd 1ahc-1mrg 2tga-1mtw 4ca2-1okm 1pdy-1pdz 1bbs-1rne 1stn-1snc
  1pts-1srf 2ctb-2ctc 2cba-2h4n 1krn-2pk4 2sil-2sim 7rat-6rsa
- hard (apo_holo): 1FVR_2OO8 1FXX_3HL8 1JWP_1PZP 1MPU_8EA5 1NEP_2HKA
  1P4O_1JQH 1S2O_1U2S 1URP_2DRI 1VR2_3VHK 1W50_3IXJ 2CEY_6H76 2CM2_1T4J
  2CM2_2H4K 2HQ8_2HPS 2LAO_1LAH 2OHG_2OHV 2SHP_7JVM 2SHP_7JVN 2ZKU_3VQS
  3CJ0_3FQK 3FVJ_2B03 3KJE_3KJG 3LCK_2OFV 3LNZ_4QOC 3NX1_3NX2 3P53_6I11
  3PPN_3PPR 3QXW_3QXV 3V55_6H4A 4I92_4I94 4IC4_4INQ 4P0I_5OTA 4R72_4R74
  4V38_4V3B 5G1M_5G3R 5H9A_6E5L 5ZA4_5YYB 6E5D_6E5F 6HB0_6HBD 6YPK_5OSZ
  7S8Z_2FWZ

## Reproducing

In `~/sitemap/benchmark`, with boonza installed:

```bash
python prepare_mae.py ~/Dropbox/PocketFinding/fpocketSet/fpocketSet_263/holo \
    --name holo200b --ligand "resname LIG" --spacing 2.0 1.5 -j 6
for mp in martini2:AC2 martini3:SC3 sirah:Y4Cv; do
    m=${mp%%:*}; p=${mp##*:}
    python tune.py --model $m --set holo200b --probe $p --tag _val \
        --from-log runs/tune/holo200_halfNone_$m.jsonl \
                   runs/tune/holo200_halfNone_${m}_fixed$p.jsonl \
        --also-preset --local 250 -j 3
done
P=$(python -c "from boonza.sitemap import presets; print(presets.path('static200'))")
python validate.py pp48 --kind apo --presets $P \
    --cases ~/Dropbox/PocketFinding/fpocketSet/LigSite48/pairs.txt
python validate.py schrodinger --kind apo --presets $P
```

The search depends on the starting logs and on its random seed (`--seed 0`,
offset by the number of trials already logged), so a rerun from scratch
reaches similar, not identical, presets. The logs are the record.

## The 200 training complexes

pdb: PDB entry (lower case, as the files are named). ligand: its residue name
in the entry, renamed `LIG` in the prepared file. heavy: the ligand's heavy
atoms in the prepared file. site: protein residues within 5 A of them.

| pdb | ligand | heavy | site | pdb | ligand | heavy | site | pdb | ligand | heavy | site | pdb | ligand | heavy | site |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 184l | I4B | 10 | 15 | 1cel | IBZ | 20 | 18 | 1fk0 | DKA | 12 | 18 | 1khb | GCP | 32 | 28 |
| 1a0t | SUC | 46 | 19 | 1cpu | LIG | 55 | 28 | 1fki | SB1 | 31 | 14 | 1kic | NOS | 19 | 16 |
| 1a27 | NAP | 68 | 48 | 1cte | PYS | 7 | 11 | 1flz | URA | 8 | 12 | 1klt | PMS | 10 | 15 |
| 1a4g | ZMR | 23 | 19 | 1cvz | C48 | 30 | 20 | 1fm6 | BRL | 25 | 25 | 1ko5 | ATP | 31 | 18 |
| 1a4h | GMY | 40 | 20 | 1czs | PHG | 7 | 9 | 1fm9 | 570 | 41 | 30 | 1l9n | BGL | 20 | 13 |
| 1a4r | GDP | 28 | 17 | 1d0x | MNQ | 21 | 20 | 1fo2 | DMJ | 11 | 14 | 1lbt | T80 | 36 | 23 |
| 1a50 | PLP | 18 | 24 | 1d1g | MTX | 81 | 43 | 1fr8 | GDU | 25 | 13 | 1lc3 | NAD | 44 | 21 |
| 1a80 | NAP | 48 | 30 | 1d3v | ABH | 13 | 16 | 1frw | GTP | 32 | 22 | 1lcp | PLU | 10 | 14 |
| 1a82 | ATP | 44 | 31 | 1d6a | GUN | 11 | 13 | 1fs5 | TLA | 10 | 12 | 1ld8 | U49 | 57 | 23 |
| 1a8i | GLS | 17 | 18 | 1daw | ANP | 31 | 24 | 1ft7 | PLU | 10 | 16 | 1lh0 | ORO | 11 | 10 |
| 1a9p | 9DI | 19 | 21 | 1dc4 | G3P | 10 | 13 | 1ftk | KAI | 15 | 17 | 1ll2 | UPG | 36 | 24 |
| 1a9u | SB2 | 27 | 18 | 1ded | ACR | 44 | 23 | 1fwk | ADP | 27 | 27 | 1llo | AMI | 43 | 23 |
| 1ad4 | HH2 | 22 | 21 | 1dgb | NDP | 48 | 23 | 1g27 | BB1 | 23 | 21 | 1lp6 | C5P | 21 | 15 |
| 1adl | ACD | 22 | 22 | 1did | DIG | 11 | 17 | 1g97 | UD1 | 39 | 31 | 1lpm | MPA | 19 | 23 |
| 1ado | 13P | 10 | 13 | 1djw | CIP | 15 | 11 | 1g9h | DAF | 33 | 21 | 1lri | CLR | 28 | 24 |
| 1agw | SU2 | 25 | 17 | 1dkp | IHP | 36 | 19 | 1gjw | MAL | 23 | 16 | 1lsp | BUL | 35 | 24 |
| 1ai5 | MNP | 13 | 14 | 1dlt | CAQ | 8 | 15 | 1gn8 | ATP | 31 | 23 | 1m6p | M6P | 16 | 14 |
| 1alw | ISA | 13 | 12 | 1dmw | HBI | 17 | 16 | 1goy | 3GP | 24 | 15 | 1m7o | 3PG | 11 | 16 |
| 1aog | FAD | 53 | 48 | 1dog | NOJ | 11 | 13 | 1gqg | DCD | 8 | 16 | 1mmy | G6D | 11 | 13 |
| 1aq1 | STU | 35 | 21 | 1dqy | DEP | 8 | 12 | 1gqt | ACP | 31 | 26 | 1n75 | ATP | 31 | 18 |
| 1aur | PMS | 10 | 11 | 1dth | BAT | 32 | 25 | 1gt6 | OLA | 20 | 20 | 1nfs | DED | 14 | 19 |
| 1av5 | AP2 | 27 | 18 | 1dy3 | 87Y | 23 | 17 | 1guz | NAD | 44 | 33 | 1nlm | UD2 | 39 | 26 |
| 1axz | GLA | 12 | 9 | 1e0v | FFC | 22 | 14 | 1gzf | NIR | 44 | 25 | 1nvq | UCN | 36 | 20 |
| 1b02 | UFP | 54 | 30 | 1e2s | CSN | 15 | 18 | 1h5x | IM2 | 20 | 14 | 1oh0 | EQU | 20 | 16 |
| 1b0o | PLM | 18 | 21 | 1e4n | HBO | 15 | 13 | 1h7f | C5P | 21 | 14 | 1opb | RET | 21 | 23 |
| 1b12 | 1PN | 20 | 15 | 1e5h | SIN | 8 | 13 | 1ha3 | MAU | 58 | 30 | 1opm | IYG | 22 | 17 |
| 1b14 | NAD | 44 | 35 | 1e5j | SGC | 46 | 18 | 1hi4 | A3P | 27 | 20 | 1oss | BEN | 9 | 16 |
| 1b42 | M1A | 11 | 8 | 1e6r | AMI | 43 | 24 | 1hlk | 113 | 23 | 13 | 1oyo | 3ID | 11 | 12 |
| 1b8u | NAD | 44 | 28 | 1e7u | KWT | 31 | 21 | 1hnd | COA | 48 | 28 | 1pbo | SES | 13 | 16 |
| 1b8y | IN7 | 26 | 22 | 1e7y | NDP | 27 | 15 | 1hqd | INK | 15 | 17 | 1pfk | ADP | 27 | 20 |
| 1bd4 | URA | 8 | 9 | 1e8u | SLB | 21 | 16 | 1i05 | LTL | 10 | 14 | 1pjc | NAD | 44 | 32 |
| 1be6 | TCA | 10 | 15 | 1efi | GAT | 19 | 9 | 1i0r | FMN | 31 | 26 | 1pyy | OSU | 32 | 17 |
| 1bgg | GCO | 13 | 17 | 1efz | PRF | 13 | 13 | 1ig1 | ANP | 31 | 24 | 1qbb | CBS | 29 | 19 |
| 1bio | SOA | 9 | 13 | 1eg9 | IND | 9 | 13 | 1iih | 3PG | 11 | 18 | 1qhg | ATP | 31 | 20 |
| 1bir | 2GP | 24 | 15 | 1eh5 | PLM | 17 | 22 | 1ikg | REX | 26 | 27 | 1ses | AHX | 30 | 23 |
| 1bk0 | ACV | 24 | 25 | 1eir | BPY | 14 | 15 | 1it7 | GUN | 11 | 15 | 1sli | DAN | 20 | 21 |
| 1bk9 | PBP | 10 | 14 | 1els | PAH | 9 | 15 | 1iwe | IMP | 23 | 20 | 1tdf | NAP | 48 | 24 |
| 1bq4 | BHC | 24 | 13 | 1epb | REA | 22 | 22 | 1j21 | CIR | 12 | 16 | 1try | ISP | 7 | 15 |
| 1bq6 | COA | 48 | 22 | 1eqc | CTS | 13 | 15 | 1jay | NAP | 101 | 38 | 1uca | U2P | 21 | 11 |
| 1br5 | NEO | 18 | 14 | 1ese | DEP | 8 | 15 | 1jbw | TMF | 24 | 13 | 1ueh | OXN | 54 | 32 |
| 1bsv | NDP | 48 | 29 | 1esw | ACR | 44 | 17 | 1jdj | CFP | 11 | 7 | 1xzk | DFP | 10 | 13 |
| 1bul | AP3 | 17 | 18 | 1eus | DAN | 20 | 16 | 1jq3 | AAT | 29 | 37 | 1ydt | IQB | 27 | 25 |
| 1bvv | DFX | 18 | 17 | 1evl | TSB | 30 | 30 | 1jvs | NDP | 48 | 26 | 2bif | ANP | 31 | 25 |
| 1bx6 | BA1 | 40 | 31 | 1ewl | R99 | 26 | 20 | 1jyl | CDC | 31 | 22 | 2ngr | GDP | 28 | 20 |
| 1c0i | FAD | 73 | 53 | 1eyn | 2AN | 21 | 13 | 1jzs | MRC | 35 | 25 | 3jdw | ORN | 9 | 16 |
| 1c1b | GCA | 24 | 17 | 1eyq | NAR | 20 | 16 | 1k0b | GTT | 20 | 17 | 3lkf | PC1 | 11 | 10 |
| 1c22 | MF3 | 12 | 15 | 1f74 | NAY | 20 | 20 | 1k6x | NAD | 44 | 24 | 4thi | PYD | 9 | 12 |
| 1c2t | GAR | 53 | 34 | 1fdk | GLE | 26 | 20 | 1k70 | HPY | 8 | 15 | 5eau | FFF | 27 | 23 |
| 1c3b | BZB | 12 | 14 | 1ffq | AMI | 43 | 29 | 1k74 | 544 | 38 | 31 | 5yas | FAC | 11 | 16 |
| 1c3j | UDP | 25 | 25 | 1fj4 | TLM | 14 | 19 | 1ka0 | AMP | 23 | 19 | 7taa | ABC | 64 | 29 |
