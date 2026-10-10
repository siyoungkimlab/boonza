# SiteMap-style sites of coarse-grained proteins (`boonza sitemap`)

`boonza sitemap` finds binding sites the way Schrödinger's SiteMap does
(Halgren 2007, 2009), on proteins as Martini 2, Martini 3 or SIRAH beads: on
one structure, or on every frame of a coarse-grained run, where the sites are
grouped into pockets that move and change shape with the protein. Like
[`boonza pockets`](pockets.md) it needs no ligand to find a site; a ligand,
when given, is only measured against.

```bash
boonza sitemap structure apo.pdb --model martini3 -o sitemap/ --holo holo.pdb
boonza sitemap traj --workdir swim/sim_000/md_solute --model martini3 -o sitemap/ \
    --apo apo.mae --every 5
```

## How a site is found

SiteMap puts a grid around the protein and keeps the points that are outside
it, enclosed by it and in good contact with a probe; neighbouring points are
grouped, and groups merged across short exposed gaps. A site is ranked by
SiteScore, which grows with its size and enclosure and falls with its
hydrophilicity. Here the same on beads:

- **outside**: at least `outside` bead radii (squared) from every bead centre,
  each bead with its force field's own size;
- **enclosed**: at least `enclosure` of 110 rays enter a bead within
  `ray_length`;
- **contact**: the Lennard-Jones energy of a probe bead at the point, from the
  force field's own pair table (Martini's per-pair terms, SIRAH's combining rule
  and overrides), within a cutoff, full or with each pair capped at zero;
- grouping and merging as SiteMap's, with sizes in A^3 so that they do not
  depend on the grid spacing.

Each point's numbers are computed once per structure; a set of thresholds is
then only compared and grouped, which is what makes tuning them affordable.

## Presets

Each model has a preset: the grid spacing, the probe, the site-finding
thresholds and SiteScore's weights (`boonza/data/sitemap/presets.json`). The
probe is the same residue in every model, valine's side-chain bead (Martini 2
AC2, Martini 3 SC3, SIRAH Y4Cv), so a site means the same thing whichever model
found it; leucine's and methionine's beads did as well in training.

The presets were tuned the way fpocket's and the `boonza pockets` presets
were, on prepared holo complexes from fpocket's training set, the score
refitted for every trial and each trial judged by the cross-validated mean of
Top-1 and Top-3 (the site's centre within 4 A of the ligand) and how much of
the ligand the top site holds. The training set is fpocket's 263 complexes
less every protein that is also in a validation set and those whose site the
file leaves incomplete (a partner chain or symmetry copy the file lacks): 200
complexes.

On the validation sets, which share no protein with the training set (Top-1 /
Top-3 / Top-10, crystal apo structures):

| model | 40 apo/holo pairs from fpocket's pp48 (easy) | Schrödinger's 41 apo (hard) |
|---|---|---|
| Martini 2 | 0.68 / 0.85 / 0.88 | 0.49 / 0.56 / 0.66 |
| Martini 3 | 0.78 / 0.90 / 0.90 | 0.49 / 0.61 / 0.63 |
| SIRAH | 0.70 / 0.82 / 0.88 | 0.54 / 0.59 / 0.61 |

Earlier presets are kept under a name, unchanged, so that a run can use them
again or compare them with the current ones: `--preset static200` (in `traj`
and `structure`) uses the presets tuned on the static training set alone, before
any fine-tuning on MD frames ([how they were made](sitemap_static200.md)).
`--preset` also takes a presets file of one's own.

## One structure

```bash
boonza sitemap structure protein.pdb --model sirah -o out/
boonza sitemap structure apo.pdb --model martini3 -o out/ --holo holo.pdb --holo-ligand "resname THA"
boonza sitemap structure holo.pdb --model martini2 -o out/ --ligand "resname I4B"
```

An all-atom structure is mapped to beads as `boonza pockets` maps it; a
coarse-grained one that carries its bead types (a run's `.dms`) is used as it
is. `--holo` carries a holo structure's ligand onto the structure exactly as
`boonza pockets` does (`--holo-ligand`, `--holo-top`, `--holo-fit`);
`--ligand` takes a ligand from the structure itself, leaving it out of the
protein. Either way each site is measured against it: PPc, MOc, the share of
the ligand's volume in the site (LVC) and the share of the site at it (PVN).

`out/pockets.csv` has a row per site, best first: rank, SiteScore and p (the
score as the probability its logistic fit gives that the site is a ligand's),
volume, enclosure, exposure and the residues lining it. `out/view.pml` shows
the structure, the beads (off) and each site as `pocket_<rank>`.

## A trajectory

```bash
boonza sitemap traj --workdir run/md_solute --model martini3 -o sitemap/ --apo apo.mae --every 5
boonza sitemap traj --workdir run/md_solute --model martini3 -o sitemap/ --every 5 \
    --holo holo.mae --holo-ligand "resname LIG"
```

With `--holo` (and `--holo-ligand`, `--holo-top`, `--holo-fit`, as in
`boonza pockets traj`) a holo structure's ligand is carried onto the run's
first fitted frame, which every frame is fitted on, and each pocket is scored
against it: PPc, MOc, LVC and PVN of its site in its best frame, and the share
of its frames whose site is PPc-right (`frames_PPc`). The view shows the holo
structure (off at the start) and the ligand.

Every analysed frame is made whole and fitted on the backbone (the probe chain
`LIG` is never part of the protein), and its sites are found under the preset,
each with the residues lining it. A pocket moves and changes shape through a
run, so it is named by those residues, not by a place: each frame's site joins
the pocket whose core (the residues lining at least half its sites) it
resembles most, by Jaccard similarity of at least `--similarity`; sites are
reassigned until none moves; two pockets that are rarely open together
(`--cooccur`) are one pocket in different states and are merged; and a pocket
open in few frames or very small joins its nearest neighbour (`--no-absorb`
keeps them apart).

Pockets are ordered by `--rank`: the best frame's p (`p_max`, the default),
its mean while open, its mean over every frame (occupancy x the mean while
open), occupancy x the best, the 90th percentile, or occupancy;
`pockets.csv` has every one of them and each pocket's place under each, with
the pocket's occupancy, volumes, best MD frame, lining residues, the probes at
it in that frame and the probes' pharmacophore hotspots in it.

The hotspots are `boonza.pharmacophore`'s: the probe beads typed into
families, each family's density over every frame against bulk, the regions at
least 20 times bulk kept when at least `--min-probes` distinct probe molecules
make them, each given to the pocket at it most often. They are shown only: no
site is found or ranked with the probes.

`view.pml` has a group per pocket: its site in its best frame, the sites of
every frame (`sites_<rank>`), the protein in the best frame with its bonds but
no elastic network (`frame_<rank>`), the probes there (`probes_<rank>`) and the
hotspots (`pharm_<rank>`), each sized by its extent and labelled with its
family, enrichment and probe count (`show labels, pharm_<rank>`). Scene buttons
return to the opening view (`overview`) or show one pocket alone; the commands
`overview` and `only <rank>` do the same.

Every frame's sites are kept in `sitemap/sites.pkl`, so regrouping or
reranking (`--similarity`, `--cooccur`, `--rank`, `--top`) takes seconds; they
are found again when the frames or the presets change. A frame takes about a
second for a few hundred beads; `--every` analyses fewer frames, and
`-j` sets the workers (every core but two by default).
