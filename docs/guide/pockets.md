# Pockets of coarse-grained proteins (`boonza pockets`)

`boonza pockets` runs fpocket on a protein that is coarse-grained (Martini 2,
Martini 3 or SIRAH) or all-atom, on one structure or on every frame of a
trajectory. A pocket here is the shape of the protein, found without any
ligand. [`boonza sites`](swim.md) answers a different question: where the
ligands of a swim actually went.

```bash
boonza pockets run protein.mae --model martini3 -o pockets/
boonza pockets traj --workdir swim/sim_000/md --model martini3 -o pockets/ \
    --apo apo.mae --holo holo.mae --every-ns 0.2
boonza pockets flags --model sirah
```

## fpocket, and the build it needs

fpocket was written for atoms. Its alpha-sphere sizes, clustering distances
and pocket score all assume atoms 3 to 4 A apart, and a bead has no element
for it to read. So boonza writes each bead with an element that carries its
polarity (`C` apolar, `O` polar):

- Martini beads go by the class letter of their type. C and X are apolar;
  P, Q and D are polar. The N class is set per model by the preset.
- SIRAH backbone beads go by name: GN and GO are polar, GC is apolar. A
  SIRAH side-chain bead is polar when its charge is at least 0.1.

Each model has a preset of fpocket flags and refitted score coefficients.
The presets were tuned the way fpocket's own defaults were, on its
263-complex training set, and checked on its 48 apo/holo pairs. They live in
`boonza/data/pockets/presets.json`; `boonza pockets flags` prints a model's
preset.

The presets pass `--score_coefficients`, which only the fpocket build at
<https://github.com/siyoungkimlab/fpocket> reads. boonza does not ship
fpocket. Build it, then point boonza at it:

```bash
git clone https://github.com/siyoungkimlab/fpocket && cd fpocket
make ARCH=MACOSXARM64        # Apple silicon; plain "make" on Linux
export FPOCKET_HOME=$PWD     # or put $PWD/bin on PATH, or pass --fpocket
```

The tuning scripts live in that repository too (`cg/benchmark/`), and they
write the presets file.

## One structure

```bash
boonza pockets run protein.mae --model sirah -o out/ --holo holo.mae
```

An all-atom structure is mapped onto the model's beads first, by boonza's own
mappers. A coarse-grained structure is taken as it is; give `--top` for a
`.gro` so the bead types are known (without it, Martini types come from the
force field's residue blocks). Probes of a swim, in chain `LIG`, are never
part of the protein.

`out/` gets the following:

- `pockets.csv` and `pockets.json`: the same rows, one per pocket.
  `pockets.json` also records every setting the numbers depend on.
- `view.pml`: run it from anywhere, e.g. `pymol out/view.pml`. It shows each
  pocket on the all-atom apo structure.
- fpocket's own output, in `<name>_out/`.

## A trajectory: consensus pockets

```bash
boonza pockets traj --workdir run/md --model martini3 -o out/ \
    --apo apo.mae --holo holo.mae --every-ns 0.2
```

The protein beads are made whole and fitted on their backbone beads (BB, or
SIRAH's GC). They are written to `md.pdb` and `md.dcd`, and fpocket runs on
every frame read, about a second per frame.

Each pocket gets a probability, `p = 1 / (1 + exp(-score))`. Pockets of
different frames become one consensus pocket when their centres lie within
`--consensus-cutoff` (6 A) of its centroid. On 80 apo trajectories, 6 A
halved how often one binding site split into several consensus pockets,
compared with 4 A. From 7 to 10 A, neighbouring sites start to merge.

In each frame a consensus pocket keeps only its best member. Consensus
pockets are then ranked three ways:

| ranking | what it is | favours |
|---|---|---|
| persistence | mean p over all frames, 0 where absent | pockets that are always there |
| quality | 90th percentile of p over the frames it is open in | pockets that open rarely but well |
| quality x burial | quality times its mean burial | enclosed pockets over grooves |

Pockets open in fewer than 5% of frames rank after the others by quality.
Burial is `boonza.sites.burial`: the share of 26 directions out of the pocket
that meet protein. With `--apo`, a pocket that has no counterpart in the apo
crystal structure is flagged `cryptic`.

Each consensus pocket is drawn and measured in its best frame. There it also
gets an enclosed core, `enclosed_core`: the grid points that SiteMap's
site-point rules keep, applied to the beads with each bead's own radius. The
core is ligand-sized and is reported beside the pocket; it does not change
the rankings.

With `--holo`, the whole holo protein is superposed on the apo: alpha carbons
of every chain, paired by sequence and pruned at 2 A. The ligand moves with
it. This is how the presets were benchmarked. `--holo-fit chain` instead fits
only the chain the ligand sits in or touches (`boonza.sites.known_ligand`).

On a symmetric oligomer the two fits can put the ligand in different,
symmetry-equivalent copies of its site. On the dimer 1MPU/8EA5, for example,
holo chain A pairs with apo chain B by sequence, and the ligand lands across
the twofold axis.

Each pocket is then scored by PPc and MOc, fpocket's own criteria:

- PPc: the pocket's centre is within 4 A of a ligand atom.
- MOc: more than half the ligand atoms are within 3 A of an alpha sphere, and
  more than a fifth of the alpha spheres are within 3 A of the ligand.

The core also gets PPc and volume overlaps with the ligand.

`out/` gets the following:

- `pockets.csv` and `pockets.json`: one row per consensus pocket. The
  columns are `rank_quality`, `rank_persistence`, `rank_quality_burial`,
  `quality`, `persistence`, `occupancy`, `pocket_burial`, `cryptic`,
  `best_frame`, `fpocket_score`, `fpocket_p`, `fpocket_volume`, `core_volume`,
  `PPc`, `MOc`, `PPc_core`, `core_DVO` and others.
- `frame_pockets.npz`: every frame's pocket, so the merge or the ranking can
  be redone without running fpocket again.
- `pockets.pqr` and `cores.pqr`: the alpha spheres and the cores.
- `view.pml`.
- With `--mdpocket`, mdpocket's frequency and density maps.

## From Python

```python
from boonza.pockets import ppc, preset, run_fpocket, write_fpocket_pdb

write_fpocket_pdb(beads, "cg.pdb", model="martini3")
pockets = run_fpocket("cg.pdb", preset("martini3")["flags"])
hits = [ppc(p.centres, ligand) for p in pockets]
```

`ppc`, `moc` and the volume overlap take `(pocket, ligand)` as
`boonza.sites.dca` and `coverage` do, so all of them can be applied to the
same pocket.

## What bead resolution allows

A ligand's atoms sit about 3 A from bead centres, which is inside the room a
bead takes. So a coarse-grained pocket finds the site about as often as an
all-atom search does: Top-3 by PPc on 40 apo trajectories is on par with
MxMD + SiteMap. It covers less of the ligand's volume, because a site that
is sealed at bead resolution is sealed for fpocket too.
