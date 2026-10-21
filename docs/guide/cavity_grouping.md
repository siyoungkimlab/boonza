# How traj's grouping was chosen

`boonza cavity traj` finds sites in every frame of a run and groups them into
pockets. This page is how the default grouping (`--grouping centroid`,
`--cutoff 8`), the point that locates a pocket (its centre) and the ranking
(`--rank p_q90_open`) were chosen, on the Schrödinger set's apo MD runs. The presets
were not changed: every result here is [dynamic200](cavity_dynamic200.md)'s
sites, regrouped.

## The data

**The runs.** The Schrödinger set's apo structures (Shi et al., JCIM 2026,
66, 9236), run with `boonza swim` and `boonza md` as the
[dynamic200](cavity_dynamic200.md) training runs were: 18 single-amino-acid
probe types, 100 ns, 1000 frames, in each of Martini 2, Martini 3 and SIRAH.
40 apo/holo pairs on 38 apo runs (`SchrodingerSet/20261005_apo/<model>/<apo>/SingleAminoAcid18/sim_000/md_solute`);
1ula/1ulb was left out, as a homologue of the training complex 1a9p. Each
pair's ligand is carried onto its run's first fitted frame by superposition
(`--holo`).

**The sites.** `boonza cavity traj` with dynamic200 on every frame, kept in
each run's `sites.pkl`. Grouping, the representative point and the ranking
were then varied on those saved sites, so no site was found again.

**The measure.** PPc: a pocket is right if its representative point is within
4 A of a ligand heavy atom (DCC, as the paper scores MxMD + SiteMap). Top-k is
over the 40 pairs; a pair with no right pocket is a miss. Coverage of the
first right pocket's site: LVC (share of the ligand's volume inside it), PVN
(share of the site within 2 A of the ligand) and fill = √(LVC·PVN), as
`boonza.pockets.overlap` computes them.

**The reference.** MxMD + SiteMap's sites for the same pairs, from the
paper's results, scored the same way (PPc of each site's centre, ranked by
SiteScore): 0.50 / 0.675 / 0.75 / 0.925 (Top-1 / 3 / 5 / 10); its hits'
LVC 0.19, PVN 0.70, fill 0.35. The paper's own ranks give 0.50 / 0.725 /
0.825 / 0.95.

## The candidates

**Groupings:**

- `consensus` (the previous default): sites grouped by the residues lining
  them (Jaccard similarity 0.3 to a pocket's core, rarely co-open pockets
  merged, small or rare pockets absorbed).
- `greedy`: MxMD + SiteMap's consolidation. Best SiteScore first, a site is
  dropped into a kept site whose residues it shares by Tanimoto > 0.4.
- `centroid`: fpocket's, as `boonza pockets` groups fpocket's pockets. Best
  SiteScore first, a site joins the pocket whose mean centre is within the
  cutoff, else starts one; then each site moves to the nearest centre within
  the cutoff until none moves. Cutoffs 6 to 10 A.
- `split`: consensus, then each pocket split by centroid.
- `union`: the dynamic200 and static200 sites of every frame pooled.

**The point that stands for a pocket:** its best-scoring site (`best`), the
per-frame best site most like its core (`typical`), the mean of its per-frame
best sites' centres (`mean`, the **centre**), or the per-frame best site
nearest that centre (`medoid`).

**Rankings:** `p_max`, `p_mean_open`, `p_mean_all`, `occ_x_pmax`, `p_q90`,
`occupancy` (as `--rank`).

## Cross-validation

5 folds by pair. In each, the grouping, point and ranking with the best
objective on the other 32 pairs is scored on the held-out 8:

    objective = (Top-5 + Top-10) / 2 + 0.01 (Top-1 + Top-3) / 2

Top-5 and Top-10 lead because a protein has more than one real pocket, and a
Top-1 miss is often another of them. With dynamic200 and every frame,
cross-validated Top-1 / 3 / 5 / 10:

| model | consensus, best site (previous) | cross-validated, per model | cross-validated, shared |
|---|---|---|---|
| Martini 2 | 0.45 / 0.575 / 0.625 / 0.70 | 0.475 / 0.60 / 0.65 / 0.825 | 0.30 / 0.55 / 0.65 / 0.775 |
| Martini 3 | 0.475 / 0.55 / 0.625 / 0.725 | 0.475 / 0.65 / 0.725 / 0.825 | 0.525 / 0.70 / 0.75 / 0.90 |
| SIRAH | 0.475 / 0.575 / 0.60 / 0.725 | 0.45 / 0.65 / 0.65 / 0.80 | 0.45 / 0.525 / 0.70 / 0.775 |

The folds' picks: centroid in 10 of the 15 per-model picks (consensus in 3,
all SIRAH; greedy and split once each in Martini 3), and the centre (`mean`)
or `typical` as the point in all 15. The ranking was `p_mean_open` in 11,
`p_max` in 3. Forty pairs are too few to choose a cutoff or a ranking finely:
the picks change with which 8 pairs are held out, and ties between choices
move the cross-validated numbers by a pair. So the default is the choice the
folds agree on, centroid grouping and the centre, at the middle cutoff, with
the ranking left as it was (below).

## The default: centroid 8 A, centre, p_max

Every frame, PPc Top-1 / 3 / 5 / 10 and the coverage of the hits:

| model | previous default | LVC / PVN / fill | centroid 8 A, centre, p_max | LVC / PVN / fill |
|---|---|---|---|---|
| Martini 2 | 0.45 / 0.575 / 0.625 / 0.70 | 0.22 / 0.35 / 0.29 | **0.50 / 0.675 / 0.75 / 0.875** | 0.18 / 0.51 / 0.30 |
| Martini 3 | 0.475 / 0.55 / 0.625 / 0.725 | 0.24 / 0.46 / 0.31 | **0.50 / 0.675 / 0.725 / 0.85** | 0.22 / 0.56 / 0.34 |
| SIRAH | 0.475 / 0.575 / 0.60 / 0.725 | 0.31 / 0.43 / 0.35 | **0.475 / 0.625 / 0.70 / 0.80** | 0.24 / 0.68 / 0.38 |
| MxMD + SiteMap | | | 0.50 / 0.675 / 0.75 / 0.925 | 0.19 / 0.70 / 0.35 |

Coverage is of the Top-10 hits: median LVC and PVN, mean fill.

Martini 2 matches MxMD + SiteMap at Top-1 to Top-5, and Martini 3 is within
a pair; at Top-10 they are 2 and 3 pairs short. SIRAH is 1 to 5 pairs short.
The hits' fill is MxMD + SiteMap's (0.30 to 0.38 against 0.35), with less of
the ligand covered by Martini 2's and a lower PVN. The previous default's hits
had a PVN of 0.35 to 0.46: its pockets' sites were larger.

**Why consensus failed.** On a cryptic site, residue consensus let the frames
in which the site was open join a persistent neighbouring pocket that shares
its lining residues. The pocket was then ranked by, and drawn at, the
neighbour's best site, beside the ligand. Grouping by place keeps them apart.

**The cutoff**, with the centre and `p_max`:

| model | 6 A | 7 A | 8 A | 9 A | 10 A |
|---|---|---|---|---|---|
| Martini 2 | 0.45 / 0.60 / 0.65 / 0.80 | 0.45 / 0.675 / 0.725 / 0.825 | 0.50 / 0.675 / 0.75 / 0.875 | 0.425 / 0.65 / 0.725 / 0.80 | 0.45 / 0.625 / 0.70 / 0.80 |
| Martini 3 | 0.525 / 0.675 / 0.75 / 0.875 | 0.50 / 0.65 / 0.725 / 0.875 | 0.50 / 0.675 / 0.725 / 0.85 | 0.45 / 0.65 / 0.80 / 0.875 | 0.475 / 0.625 / 0.725 / 0.825 |
| SIRAH | 0.525 / 0.60 / 0.625 / 0.775 | 0.475 / 0.625 / 0.65 / 0.80 | 0.475 / 0.625 / 0.70 / 0.80 | 0.475 / 0.625 / 0.725 / 0.825 | 0.45 / 0.65 / 0.725 / 0.825 |

8 A is the best or within a pair of it in each model. A cutoff per model was
tried and fitted noise: each model's best cutoff on 32 pairs did no better on
the other 8. `greedy` (Tanimoto 0.4, best site) gave 0.45 / 0.625 / 0.725 /
0.825, 0.475 / 0.65 / 0.70 / 0.825 and 0.475 / 0.625 / 0.675 / 0.775.

**The point**, at 8 A with `p_max`:

| model | best site | typical | medoid | centre |
|---|---|---|---|---|
| Martini 2 | 0.45 / 0.65 / 0.675 / 0.775 | 0.45 / 0.625 / 0.65 / 0.775 | 0.45 / 0.625 / 0.725 / 0.85 | 0.50 / 0.675 / 0.75 / 0.875 |
| Martini 3 | 0.475 / 0.675 / 0.725 / 0.80 | 0.50 / 0.675 / 0.725 / 0.85 | 0.50 / 0.675 / 0.725 / 0.85 | 0.50 / 0.675 / 0.725 / 0.85 |
| SIRAH | 0.475 / 0.60 / 0.625 / 0.70 | 0.45 / 0.60 / 0.65 / 0.75 | 0.475 / 0.625 / 0.70 / 0.80 | 0.475 / 0.625 / 0.70 / 0.80 |

The best-scoring site of a run can lie beside the pocket's usual place, in a
frame where it is larger. So `traj` locates a pocket by its centre and draws
and measures it by its medoid, a real site of a real frame (`md_frame`).

**The ranking**, at 8 A with the centre:

| model | p_q90_open | p_max | p_mean_open | p_q90 | p_mean_all | occ_x_pmax | occupancy |
|---|---|---|---|---|---|---|---|
| Martini 2 | 0.50 / 0.675 / 0.775 / 0.875 | 0.50 / 0.675 / 0.75 / 0.875 | 0.40 / 0.65 / 0.775 / 0.875 | 0.425 / 0.65 / 0.75 / 0.825 | 0.40 / 0.475 / 0.60 / 0.80 | 0.425 / 0.475 / 0.60 / 0.75 | 0.325 / 0.425 / 0.575 / 0.675 |
| Martini 3 | 0.50 / 0.70 / 0.75 / 0.875 | 0.50 / 0.675 / 0.725 / 0.85 | 0.525 / 0.725 / 0.75 / 0.85 | 0.475 / 0.65 / 0.80 / 0.85 | 0.525 / 0.625 / 0.75 / 0.825 | 0.425 / 0.60 / 0.80 / 0.80 | 0.30 / 0.525 / 0.75 / 0.80 |
| SIRAH | 0.425 / 0.60 / 0.65 / 0.80 | 0.475 / 0.625 / 0.70 / 0.80 | 0.425 / 0.55 / 0.675 / 0.80 | 0.425 / 0.525 / 0.625 / 0.75 | 0.375 / 0.55 / 0.575 / 0.75 | 0.325 / 0.50 / 0.575 / 0.775 | 0.225 / 0.475 / 0.55 / 0.70 |

The default is `p_q90_open`: the 90th percentile of a pocket's p over the
frames it is open in, as `boonza pockets` ranks fpocket's pockets (its
quality). A pocket must score well in a tenth of its open frames, not in one,
so a site that is large for a moment wins less often. It gains a pair at Top-5
in Martini 2 and a pair at Top-3, Top-5 and Top-10 in Martini 3, and costs
SIRAH two pairs at Top-1 and Top-5 and one at Top-3; Top-10 is unchanged in
Martini 2 and SIRAH. Other percentiles over the open frames: the 95th as the
90th, the 50th and 75th worse. `p_q90`, the same with closed frames counted as
0, is worse: it penalises the cryptic sites the set is made of, open in few
frames. The other tables on this page are with `p_max`, the default the
grouping and the centre were chosen under.

## An independent check: the 200 training runs

The choice was made on the Schrödinger runs, so it was checked on runs it
never saw: the 200 [dynamic200](cavity_dynamic200.md) training complexes'
MD runs (every frame, the ligand carried on by superposition, as `--holo`).
PPc Top-1 / 3 / 5 / 10:

| model | consensus, best site | LVC / PVN / fill | centroid 8 A, centre, p_max | LVC / PVN / fill |
|---|---|---|---|---|
| Martini 2 | 0.445 / 0.60 / 0.655 / 0.69 | 0.41 / 0.46 / 0.43 | 0.475 / 0.695 / 0.78 / 0.855 | 0.32 / 0.63 / 0.45 |
| Martini 3 | 0.53 / 0.70 / 0.74 / 0.77 | 0.42 / 0.49 / 0.45 | 0.595 / 0.80 / 0.87 / 0.96 | 0.33 / 0.66 / 0.44 |
| SIRAH | 0.46 / 0.595 / 0.645 / 0.69 | 0.46 / 0.46 / 0.46 | 0.525 / 0.73 / 0.81 / 0.865 | 0.40 / 0.64 / 0.48 |

The gain holds, and is larger: Top-10 rises by 0.165 to 0.19 in every model.
Coverage is of the Top-10 hits (median LVC and PVN, mean fill), the medoid's
site for the centre and the best site for consensus: the medoid covers less
of the ligand and lies more on it, and fill is the same.

## How many frames

Centroid 8 A, centre, `p_max`, on the Schrödinger runs (frames 0.1 ns apart):

| frames | Martini 2 | Martini 3 | SIRAH |
|---|---|---|---|
| first 10 ns (100) | 0.40 / 0.575 / 0.675 / 0.725 | 0.60 / 0.70 / 0.75 / 0.85 | 0.45 / 0.625 / 0.725 / 0.85 |
| first 50 ns (500) | 0.45 / 0.60 / 0.725 / 0.825 | 0.50 / 0.75 / 0.775 / 0.85 | 0.525 / 0.65 / 0.725 / 0.825 |
| every 50th (20) | 0.40 / 0.675 / 0.70 / 0.725 | 0.525 / 0.675 / 0.775 / 0.85 | 0.40 / 0.625 / 0.625 / 0.725 |
| every 10th (100) | 0.375 / 0.625 / 0.675 / 0.825 | 0.525 / 0.65 / 0.80 / 0.85 | 0.40 / 0.65 / 0.70 / 0.80 |
| every 2nd (500) | 0.45 / 0.675 / 0.75 / 0.825 | 0.475 / 0.675 / 0.80 / 0.85 | 0.45 / 0.60 / 0.675 / 0.80 |
| every frame (1000) | 0.50 / 0.675 / 0.75 / 0.875 | 0.50 / 0.675 / 0.725 / 0.85 | 0.475 / 0.625 / 0.70 / 0.80 |

Results are fairly flat. The first 10 ns does as well as the whole run in
Martini 3 and SIRAH; Martini 2 gains from every frame, and thinning to 20
frames loses Top-10 in Martini 2 and SIRAH. On the 200 training runs it is
flatter still (Top-10 0.81 to 0.86, 0.92 to 0.96 and 0.84 to 0.87 over the
same windows), and Top-1 falls as frames are added (Martini 3: 0.685 at 20
frames, 0.595 at 1000), likely because `p_max` gives a large, briefly open
site more chances to win.

## What did not help

- **Both presets' sites pooled** (`union`) and **consensus then centroid**
  (`split`): no better than centroid alone.
- **A cutoff per model:** fitted noise (above).
- **SiteMap's cap on the size term.** SiteMap's SiteScore caps the site size
  at 100 points; ours does not. Rescoring with the cap cut Top-1 badly. A
  second try retuned static200 and dynamic200 from scratch with the cap
  (`n` ≤ 100 A^3): on the 200 static structures, cross-validated Top-1 fell
  0.04 to 0.05 with Top-5 and Top-10 within 0.02; on the training MD frames
  Top-10 was 0.69 / 0.76 / 0.69 against 0.69 / 0.82 / 0.74, and Top-1 lower in
  Martini 2 and SIRAH; on the Schrödinger runs (every 10th frame, centroid 8 A) it was worse
  or equal in every model and at every k.
- **A little MD in the ranking.** Ranking by the best frame's SiteScore plus a
  weight times occupancy, how steadily the same grid points belong to the
  pocket, the time-averaged volume, the mean SiteScore while open, or the
  probes' binding near the pocket (`boonza sites`' probe sites' dG and
  occupancy, and the feature hotspots' enrichment, after SILCS), with the
  weight fitted on the 200 training runs: training Top-1 rose up to 0.06 for
  occupancy, Top-5 and Top-10 did not, and on the Schrödinger runs each was
  within a pair of `p_max` alone. Within a run's top 10 these separate the
  right pocket from the wrong ones by AUC 0.55 to 0.70, against 0.70 to 0.78
  for SiteScore, and the best of them grows with size, as SiteScore does.
- **Another point for a pocket:** a SiteScore-weighted centre, the best
  frame's site, the centroid of the grid points most often in the pocket, or
  the single most often: none better than the centre.

## What is left

Of the pairs whose ligand site is not in the top 10, most have a pocket at the
site, often ranked 2 to 8, whose centre lies 4.2 to 6 A from the ligand: its
sites are larger than the ligand's site and extend past it, steadily, so no
single point of it is within 4 A. Two pairs (one in Martini 2, one in SIRAH)
have no site at the ligand in any frame. MxMD + SiteMap reports every frame's
site separately (74 to 150 per pair), so a large pocket has many centres.
Closing the gap needs smaller sites at the site-finding stage, which is the
presets' job, not the grouping's.

## Reproducing

In `~/sitemap/benchmark`:

```bash
python schr_prepare.py                      # data/schr/<preset>/<model>/<pair>.pkl from sites.pkl
./schr_search.sh; ./schr_centroid.sh; ./schr_more.sh   # schr_tune.py: every candidate
python schr_cv.py --every 1                 # cross-validation, coverage, MxMD + SiteMap
python frames_test.py --set schr            # how many frames
python occ_blend.py; python probe_test.py   # MD in the ranking
python centre_test.py; python miss_diag.py  # the point, and what is left
python train_coverage.py                    # coverage on the 200 training runs
```
