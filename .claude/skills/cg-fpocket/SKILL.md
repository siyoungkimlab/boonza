---
name: cg-fpocket
description: Find, rank or track binding pockets in coarse-grained proteins (Martini 2, Martini 3, SIRAH) with `boonza pockets` -- fpocket and mdpocket with the flags and pocket score tuned for beads. Use this whenever the user wants pockets, cavities or cryptic sites in a Martini or SIRAH structure or CG MD trajectory (a swim run directory, .gro/.xtc/.dcd with a topology, solvated.dms + trajectory.dcd), wants to coarse-grain an all-atom structure and look for pockets in it, asks about "CG pocket detection", "mdpocket on my Martini run", "consensus pockets", "enclosed cores" or the "tuned fpocket presets", wants predicted pockets checked against the ligand of a holo structure (PPc, MOc), is changing src/boonza/pockets/, or is about to run plain fpocket on beads -- fpocket's all-atom defaults rank pockets on beads almost at random.
---

# fpocket on coarse-grained proteins (`boonza pockets`)

fpocket was built for atoms. On beads its defaults find pockets but rank the
real binding site first only 2-12% of the time. `boonza pockets` gives each
bead an element carrying its polarity and passes a per-model preset of
detection flags plus a refitted score. The presets were tuned the way fpocket
itself was, on its 263-complex training set (PPc: pocket centre < 4 A from
the ligand), and checked on held-out apo/holo sets.

The code is `src/boonza/pockets/`. The guide is `docs/guide/pockets.md`.

## Before anything

```bash
boonza pockets flags --model martini3   # the preset, from data/pockets/presets.json
```

- **fpocket is not part of boonza.** The presets pass `--score_coefficients`,
  which only the build at https://github.com/siyoungkimlab/fpocket reads.
  `find_fpocket` looks in `--fpocket`, then `$FPOCKET_HOME/bin`, then `PATH`,
  and refuses a stock fpocket. On this machine the build is `~/fpocket`
  (`export FPOCKET_HOME=~/fpocket`).
- Read the presets from `presets.json` (or `boonza pockets flags`), not from
  memory: they get retuned. The tuning lives in that fpocket repository
  (`cg/benchmark/`), which writes the presets file here.
- `aa` is fpocket's own defaults, deliberately unchanged.
- The tests that run fpocket skip themselves where it is not installed.

## Rules that are easy to break

- **Chain `LIG` holds probes, not protein.** It holds ligand copies placed by
  `boonza swim`, often amino acids themselves. `protein_ids` always subtracts
  it. Do the same (`not chain "LIG"`) in any selection you write yourself.
- **Never hand fpocket a bead PDB that `write_fpocket_pdb` did not write.**
  fpocket reads polarity from the element column. A raw bead PDB's elements
  are guesses from bead names (BB becomes boron, SC1 sulfur).
- Use the model the structure or simulation is in. The presets are not
  interchangeable.
- Report PPc and MOc, never residue overlap. Place a holo ligand by
  whole-protein superposition (`boonza.sites.known_ligand`), never by
  binding-site residue lists.
- Do not use fpocket's `-p` above 0: a precedence bug in its `refine.c` then
  drops every pocket that is not entirely apolar.
- Keep `-C s` (single linkage). The other linkages are about 10x slower on
  beads.

## One structure

```bash
boonza pockets run protein.mae --model martini3 -o out/      # all-atom: mapped to beads
boonza pockets run cg.gro --top topol.top --model martini2 -o out/ --apo apo.mae
boonza pockets run apo.mae --model sirah -o out/ --holo holo.mae --holo-ligand "resname LIG"
```

Extra fpocket flags after `--` override the preset: `-- -i 20`.

Outputs in `out/`:

- `pockets.csv` and `pockets.json`: the same rows. `pockets.json` also holds
  the settings.
- `view.pml`: the all-atom apo and each pocket as an object `pocket_<rank>`.
  It runs from any directory.
- With `--holo`, each row gets `center_to_nearest_ligand_atom`, `PPc`, `MOc`,
  `ligand_atoms_within_3A` and `spheres_within_3A`. Check the printed
  superposition. On a protein whose domains moved, the ligand follows the
  rigid core and can sit a few A off.

## A trajectory

```bash
boonza pockets traj --workdir run/md --model martini3 -o out/ \
    --apo apo.mae --holo holo.mae --holo-ligand "resname LIG" --every-ns 0.2
boonza pockets traj cg.gro --top topol.top --traj md.xtc --model sirah -o out/
```

The protein beads are made whole and fitted on BB or GC, then written to
`md.pdb` and `md.dcd`. fpocket runs on every frame read (about 0.1-0.5 s
each, so use `--every-ns` on long runs). Pockets are merged into consensus
pockets: greedy, best first, centres within `--consensus-cutoff` (6 A) of a
running centroid, each frame counted once with its best member.

They are ranked three ways:

- **persistence**: mean p over all frames, where `p = sigmoid(score)`.
- **quality**: the 90th percentile of p over the frames the pocket is open
  in. Pockets open in fewer than 5% of frames rank last.
- **quality x burial**: quality times the mean burial
  (`boonza.sites.burial`). This was best for Martini 3.

With `--apo`, a pocket is flagged `cryptic` when it is absent from the apo
crystal structure. Each consensus pocket also gets an **enclosed core** in its
best frame (`enclosed_core`): SiteMap's site-point rules on the beads,
applied on a 2 A grid with each bead's own radius. The cores are SiteMap-sized
(about 150-400 A^3) and do not change the ranking. Radii come from
`boonza.sites.particle_radii` when there is a topology, otherwise from
`data/cg_radii/`.

Outputs:

- `pockets.csv`/`.json` with the columns `rank_quality`, `rank_persistence`,
  `rank_quality_burial`, `quality`, `persistence`, `occupancy`,
  `pocket_burial`, `cryptic`, `best_frame`, `fpocket_score`, `fpocket_p`,
  `fpocket_volume`, `core_volume`, `core_center_*`. With `--holo` it adds
  `PPc`, `MOc`, `share_of_open_frames_PPc`, `PPc_core`,
  `core_ligand_volume_covered` and `core_DVO`.
- `frame_pockets.npz`: every frame's pocket, to redo the merge or the ranking
  without fpocket.
- `pockets.pqr` and `cores.pqr`.
- `view.pml`: objects `pocket_<rank>` and `core_<rank>`.
- With `--mdpocket`, `pocket_frequency.dx` and `pocket_density.dx`. Read the
  density at the preset's `mdpocket_density_iso` (about 5 for Martini), not
  mdpocket's all-atom 8.

## From Python

```python
from boonza.pockets import consensus_pockets, frame_pockets, ppc, preset, write_fpocket_pdb
```

`ppc`, `moc` and `volume_overlap` take `(pocket, ligand)`, as
`boonza.sites.dca`, `dcc` and `coverage` do.

## What to expect

Held-out PPc Top-1 / 3 / 5 / 10 for single structures (`presets.json` has the
current numbers under `validation`):

| | fpocket 48 apo | Schrodinger 41 apo |
|---|---|---|
| all-atom, fpocket defaults | 0.31 / 0.56 / 0.67 / 0.79 | 0.20 / 0.39 / 0.49 / 0.66 |
| Martini 2 preset | 0.60 / 0.81 / 0.88 / 0.94 | 0.37 / 0.49 / 0.56 / 0.63 |
| Martini 3 preset | 0.62 / 0.79 / 0.90 / 0.90 | 0.37 / 0.51 / 0.59 / 0.61 |
| SIRAH preset | 0.65 / 0.77 / 0.81 / 0.88 | 0.41 / 0.51 / 0.56 / 0.68 |

On 40 apo trajectories, consensus pockets by quality x burial match
MxMD + SiteMap from Top-3 on. They cover less of the ligand's volume, because
ligand atoms sit inside the room a bead takes. A site sealed at bead
resolution is sealed for fpocket too. With about 40 structures per column,
differences under about 0.1 are noise.
