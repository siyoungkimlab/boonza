# Cavities of coarse-grained proteins, SiteMap-style (`boonza cavity`)

`boonza cavity` was called `boonza sitemap` until it was renamed, since it is not
Schrödinger's SiteMap but finds sites the way it does; the old name still runs.

`boonza cavity` finds binding sites the way Schrödinger's SiteMap does
(Halgren 2007, 2009), on proteins as Martini 2, Martini 3 or SIRAH beads: on
one structure, or on every frame of a coarse-grained run, where the sites are
grouped into pockets that move and change shape with the protein. Like
[`boonza pockets`](pockets.md) it needs no ligand to find a site; a ligand,
when given, is only measured against.

```bash
boonza cavity structure apo.pdb --model martini3 -o cavity/ --holo holo.pdb
boonza cavity traj --workdir swim/sim_000/md_solute --model martini3 -o cavity/ \
    --apo apo.mae --every 5
```

## How a site is found

SiteMap puts a grid around the protein and keeps the points that are outside
it, enclosed by it and in good contact with a probe; neighbouring points are
grouped, and groups merged across short exposed gaps. Here the same on beads,
without the contact test, and sites ranked by their volume:

- **outside**: at least `outside` bead radii (squared) from every bead centre,
  each bead with its force field's own size (half the Lennard-Jones sigma of its
  type with itself);
- **enclosed**: at least `enclosure` of 110 rays enter a bead within
  `ray_length`;
- grouping and merging as SiteMap's, with sizes in A^3 so that they do not
  depend on the grid spacing.

Each point's numbers are computed once per frame; a set of thresholds is then
only compared and grouped, which is what makes tuning them affordable.

## Presets

Each model has a preset, `volume200` (`boonza/data/cavity/presets/volume200.json`),
the default of both commands: SiteMap's thresholds with only `outside` and
`enclosure` tuned on MD frames, and sites ranked by volume alone
([what it is and how it compares](cavity_volume200.md)). `--preset` (in `traj`
and `structure`) takes a presets file of one's own instead.

## One structure

```bash
boonza cavity structure protein.pdb --model sirah -o out/
boonza cavity structure apo.pdb --model martini3 -o out/ --holo holo.pdb --holo-ligand "resname THA"
boonza cavity structure holo.pdb --model martini2 -o out/ --ligand "resname I4B"
```

An all-atom structure is mapped to beads as `boonza pockets` maps it; a
coarse-grained one that carries its bead types (a run's `.dms`) is used as it
is. `--holo` carries a holo structure's ligand onto the structure exactly as
`boonza pockets` does (`--holo-ligand`, `--holo-top`, `--holo-fit`);
`--ligand` takes a ligand from the structure itself, leaving it out of the
protein. Either way each site is measured against it: PPc, MOc, the share of
the ligand's volume in the site (LVC) and the share of the site at it (PVN).

`out/pockets.csv` has a row per site, best first: rank, score and p
(1 / (1 + exp(-score))), volume, enclosure (the share of its points' rays that meet
the protein within 10 A) and the residues lining it. `out/view.pml` shows
the structure, the beads (off) and each site as `pocket_<rank>`.

## A trajectory

```bash
boonza cavity traj --workdir run/md_solute --model martini3 -o cavity/ --apo apo.mae --every 5
boonza cavity traj --workdir run/md_solute --model martini3 -o cavity/ --every 5 \
    --holo holo.mae --holo-ligand "resname LIG"
```

With `--holo` (and `--holo-ligand`, `--holo-top`, `--holo-fit`, as in
`boonza pockets traj`) a holo structure's ligand is carried onto the run's
first fitted frame, which every frame is fitted on, and each pocket is scored
against it: PPc at the pocket's centre, MOc, LVC and PVN of its medoid site,
and the share of its frames whose site is PPc-right (`frames_PPc`). The view shows the holo
structure (off at the start) and the ligand.

Every analysed frame is made whole and fitted on the backbone (the probe chain
`LIG` is never part of the protein), and its sites are found under the preset,
each with the residues lining it. The sites of every frame are then grouped
into pockets by place, as `boonza pockets` groups fpocket's (`--grouping
centroid`, the default): largest first, each site joins the pocket whose
mean centre is within `--cutoff` (8 A) of its own, or starts one, and sites are
reassigned to the nearest centre until none moves.

A pocket is located by its **centre**, the mean centre of all its sites (a
frame can give it two, and both count), and drawn and measured by its
**medoid**, the site nearest that centre, of any frame. Its median volume is of
all its sites too. Its single best-scoring site is not used for either: in a
run it can lie beside where the pocket usually is. On the three CRUMB benchmark
sets (1180 runs), taking all sites rather than each frame's best left
Top-1/3/5/10 unchanged in every run and model, and fill within 0.01.

`--grouping consensus` groups the sites instead by the residues lining them:
each site joins the pocket whose core (the residues lining at least half its
sites) it resembles most, by Jaccard similarity of at least `--similarity`;
sites are reassigned until none moves; two pockets that are rarely open
together (`--cooccur`) are merged; and a pocket open in few frames or very
small joins its nearest neighbour (`--no-absorb` keeps them apart). On cryptic
sites it let the ligand's site merge with a persistent neighbouring pocket
([how the grouping was chosen](cavity_grouping.md)).

Pockets are ordered by `--rank`: the best frame's p (`p_max`, the default; with
volume200, the pocket's largest site), the 90th percentile of p over the frames a
pocket is open in (`p_q90_open`, as `boonza pockets` ranks), its mean while open, its mean over every frame
(occupancy x the mean while open), occupancy x the best, the 90th percentile
over every frame (closed frames as 0), or occupancy;
`pockets.csv` has every one of them and each pocket's place under each, with
the pocket's occupancy, volumes, centre (`centre_x/y/z`), the MD frame it is
drawn in (`md_frame`, its medoid's) and its best-scoring frame
(`best_md_frame`), lining residues, the probes at it in that frame and the
probes' pharmacophore hotspots in it.

The hotspots are `boonza.pharmacophore`'s: the probe beads typed into
families, each family's density over every frame against bulk, the regions at
least 20 times bulk kept when at least `--min-probes` distinct probe molecules
make them, each given to the pocket at it most often. They are shown only: no
site is found or ranked with the probes.

`view.pml` has a group per pocket: its medoid site, the sites of
every frame (`sites_<rank>`), the protein in the medoid's frame with its bonds but
no elastic network (`frame_<rank>`), the probes there (`probes_<rank>`) and the
hotspots (`pharm_<rank>`), each sized by its extent and labelled with its
family, enrichment and probe count (`show labels, pharm_<rank>`). Scene buttons
return to the opening view (`overview`) or show one pocket alone; the commands
`overview` and `only <rank>` do the same.

Every frame's sites are kept in `cavity/sites.pkl`, so regrouping or
reranking (`--grouping`, `--cutoff`, `--similarity`, `--cooccur`, `--rank`,
`--top`) takes seconds; they
are found again when the frames or the presets change. A frame takes about half a
second for a few hundred beads and a few seconds for several thousand; `--every` analyses fewer frames, and
`-j` sets the workers (every core but two by default). `cavity/timing.txt` (and `timing.json`)
says where a run's time went: loading and fitting, site finding (split into building
each frame's grid and finding its sites), grouping, ranking, measuring against a holo
ligand and writing the report.
