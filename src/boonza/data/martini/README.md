# Martini data

These files are vermouth's (https://github.com/marrink-lab/vermouth-martinize,
commit a9b62ee, Apache License 2.0, see `LICENSE`), unchanged:

- `martini3001/*.ff` and `martini22/*.ff`: the protein residues, links and
  modifications of each version (`vermouth/data/force_fields/<version>`);
- `mappings/<version>/*`: atom-to-bead mappings of those residues and of their
  modifications.  Which atoms make which bead is the version's own -- Martini
  2.2 cuts a tryptophan's rings differently from Martini 3, and reads a
  phenylalanine's ring in another order -- so each version has its own
  directory, from `vermouth/data/mappings/martini3001` and from
  `vermouth/data/mappings` (vermouth's default directory is Martini 2's).  The
  Martini 2 set has no mapping for a protonated aspartate or glutamate, a
  neutral lysine, or the amber-named histidines, because vermouth carries none;
  `martini22/modifications.charmm36.mapping` is its `modifications.mapping`.
  The lipid mappings are left out of both: boonza builds a bilayer from
  templates rather than mapping one down;
- `charmm/aminoacids.rtp` and `charmm/modifications.ff` (renamed from
  `00-modifications.ff`): the reference residues whose hydrogens the mappings
  name (`vermouth/data/force_fields/charmm`).

`water.gro` is boonza's own: 1000 Martini 3 water beads equilibrated in
OpenMM at 300 K and 1 bar (`tests/data/martini/make_water_box.py` remakes
it).
