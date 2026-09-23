# Martini 3 data

These files are vermouth's (https://github.com/marrink-lab/vermouth-martinize,
commit a9b62ee, Apache License 2.0, see `LICENSE`), unchanged:

- `martini3001/*.ff`: Martini 3 protein residues, links and modifications
  (`vermouth/data/force_fields/martini3001`);
- `mappings/*`: atom-to-bead mappings of those residues and of their
  modifications (`vermouth/data/mappings/martini3001`);
- `charmm/aminoacids.rtp` and `charmm/modifications.ff` (renamed from
  `00-modifications.ff`): the reference residues whose hydrogens the mappings
  name (`vermouth/data/force_fields/charmm`).

`water.gro` is boonza's own: 1000 Martini 3 water beads equilibrated in
OpenMM at 300 K and 1 bar (`tests/data/martini/make_water_box.py` remakes
it).
