# The volume200 presets

`volume200` (`boonza/data/sitemap/presets/volume200.json`, `--preset volume200`) is a
simpler alternative to [dynamic200](sitemap_dynamic200.md): SiteMap's own thresholds, with
only the two that depend on the beads tuned, and sites ranked by their volume.

```bash
boonza sitemap traj --workdir run/md_solute --model martini3 -o sitemap/ --preset volume200
```

## What it is

- **A point is a site point** if it is outside the protein (d^2/R^2 >= `outside` for every
  bead) and enclosed: at least `enclosure` of 110 rays enter a bead within 10 A, the rays
  sampled every 2 A (the grid spacing). No contact filter: the probe's Lennard-Jones
  energy plays no part.
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

| model | dynamic200 | volume200 |
|---|---|---|
| Martini 2 | 0.50 / 0.65 / 0.67 / 0.69, 0.22 | 0.49 / 0.64 / 0.68 / 0.71, 0.21 |
| Martini 3 | 0.60 / 0.76 / 0.79 / 0.82, 0.27 | 0.63 / 0.77 / 0.80 / 0.82, 0.30 |
| SIRAH | 0.55 / 0.69 / 0.73 / 0.74, 0.27 | 0.51 / 0.67 / 0.72 / 0.74, 0.22 |

## Validation: the Schrodinger set's apo runs

40 pairs, 18 probes, EN 500 kJ/mol/nm^2 at 0.9 nm, every frame, as `traj` groups and ranks
(PPc Top-1 / 3 / 5 / 10):

| model | static200 | dynamic200 | volume200 |
|---|---|---|---|
| Martini 2 | 0.43 / 0.63 / 0.75 / 0.90 | 0.50 / 0.68 / 0.78 / 0.88 | 0.43 / 0.63 / 0.75 / 0.85 |
| Martini 3 | 0.55 / 0.73 / 0.75 / 0.88 | 0.50 / 0.70 / 0.75 / 0.88 | 0.55 / 0.68 / 0.80 / 0.90 |
| SIRAH | 0.55 / 0.65 / 0.78 / 0.83 | 0.43 / 0.60 / 0.65 / 0.80 | 0.55 / 0.60 / 0.63 / 0.75 |

MxMD + SiteMap on the same pairs, scored the same way: 0.50 / 0.68 / 0.75 / 0.93.

Martini 3 matches or beats the tuned presets with two thresholds; Martini 2 and SIRAH lose
a pair or two at Top-10. dynamic200 stays `traj`'s default until volume200 is checked on
more runs.
