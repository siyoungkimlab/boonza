# The volume200 presets

`volume200` (`boonza/data/cavity/presets/volume200.json`) is `boonza cavity`'s preset:
SiteMap's own thresholds, with only the two that depend on the beads tuned, and sites ranked
by their volume. It replaced two presets tuned with a contact filter and fitted SiteScore
weights (static200 and dynamic200, since removed), which it matched with two thresholds.

```bash
boonza cavity traj --workdir run/md_solute --model martini3 -o cavity/ --preset volume200
```

## What it is

- **A point is a site point** if it is outside the protein (d^2/R^2 >= `outside` for every
  bead) and enclosed: at least `enclosure` of 110 rays enter a bead within 10 A, the rays
  sampled every 2 A (the grid spacing). No contact filter: boonza computes no probe
  energies.
- **Everything else is SiteMap's:** 3 neighbours within 1.76 grid steps, groups of at least 3
  points, merging across gaps of up to 6.5 A while one group is at most 100 A^3, sites after
  the first of at least 15 A^3, ten sites at most.
- **Sites are ranked by volume.** The score is 0.1 sqrt(V): only its order matters, the 0.1
  keeping p = 1 / (1 + exp(-score)) away from 1. Pockets in a run are ranked by its 90th
  percentile over their open frames, so by the 90th percentile of their volume.

| | outside | enclosure (of 110 rays, within 10 A) |
|---|---|---|
| Martini 2 | 1.75 | 0.70 |
| Martini 3 | 2.75 | 0.60 |
| SIRAH | 3.00 | 0.60 |

## How the two thresholds were chosen

On MD frames of the 200 training complexes' runs (5 frames of each 100-ns run, 1000 per
model, the ligand left out and placed on each frame by superposition), with sites ranked
by volume, `outside` 1.75-3.5 and `enclosure` 0.55-0.70 were scanned per model and the pair
with the best 1/2 (1/2 (Top-3 + Top-7) + fill) kept. Top-1 is not tuned for: a protein has
several real pockets. Nothing else is fitted.

On those frames (Top-1 / 3 / 5 / 10, fill):

| model | volume200 |
|---|---|
| Martini 2 | 0.49 / 0.64 / 0.68 / 0.71, 0.21 |
| Martini 3 | 0.63 / 0.77 / 0.80 / 0.82, 0.30 |
| SIRAH | 0.51 / 0.67 / 0.72 / 0.74, 0.22 |

## Validation: the Schrodinger set's apo runs

40 pairs, 18 probes, EN 500 kJ/mol/nm^2 at 0.9 nm, every frame, as `traj` groups and ranks
(PPc Top-1 / 3 / 5 / 10):

| model | volume200 |
|---|---|
| Martini 2 | 0.43 / 0.63 / 0.75 / 0.85 |
| Martini 3 | 0.55 / 0.68 / 0.80 / 0.90 |
| SIRAH | 0.55 / 0.60 / 0.63 / 0.75 |

MxMD + SiteMap on the same pairs, scored the same way: 0.50 / 0.68 / 0.75 / 0.93.

Martini 3 matched or beat the presets it replaced with two thresholds; Martini 2 and SIRAH
lost a pair or two at Top-10.

## The default: volume200, pockets ranked by p_max

`boonza cavity traj` uses volume200 and ranks pockets by `p_max` unless told otherwise
(`--preset`, `--rank`). With volume200, `p_max` is the pocket's largest site over the
frames it is open in. On the Schrödinger set's 43 final apo runs (EN 700 kJ/mol/nm^2 at
0.9 nm, every frame; both 5UXA ligands included), PPc Top-1 / 3 / 5 / 10:

| model | with probes | without probes |
|---|---|---|
| Martini 2 | 0.44 / 0.63 / 0.72 / 0.86 | 0.42 / 0.70 / 0.77 / 0.86 |
| Martini 3 | 0.51 / 0.77 / 0.81 / 0.91 | 0.58 / 0.79 / 0.81 / 0.86 |
| SIRAH | 0.47 / 0.60 / 0.67 / 0.74 | 0.51 / 0.65 / 0.74 / 0.84 |

MxMD + SiteMap on the same 43 pairs, scored the same way: 0.47 / 0.65 / 0.74 / 0.93. Ranked
by `p_q90_open` the same pockets reach Top-10 0.77 / 0.88 / 0.72 with probes: `p_max` gains
a pair or two at Top-10 and `p_q90_open` a pair at Top-1.
