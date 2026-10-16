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
  neutral lysine, or the amber-named histidines, because vermouth carries none
  -- boonza reads the amber histidines as the CHARMM ones, which it has, and
  builds a neutral acid or lysine from the charged residue with the side
  chain's charge set to zero, saying so;
  `martini22/modifications.charmm36.mapping` is its `modifications.mapping`.
  The lipid mappings are left out of both: boonza builds a bilayer from
  templates rather than mapping one down;
- `charmm/aminoacids.rtp` and `charmm/modifications.ff` (renamed from
  `00-modifications.ff`): the reference residues whose hydrogens the mappings
  name (`vermouth/data/force_fields/charmm`).

`rcsu/protein_map.json` is vermouth's too, extracted from
`vermouth/rcsu/contact_map.py` rather than copied as a file: the van der Waals
radius and chemical character of every atom of the twenty residues
(`PROTEIN_MAP`), and the table that reads a pair of characters as a bond type
(`BOND_TYPE`).  They are the numbers of the RCSU contact-map program
(doi:10.5281/zenodo.3817447), which `boonza.martini.contact_map` needs to
compute the same map.  Four names that file lists with no atoms at all -- CYM,
CYX, HIE and HIP -- are left out, and `boonza.martini.rcsu.ALIASES` reads them
as the residues they are.

`water.gro` is boonza's own: 1000 Martini 3 water beads equilibrated in
OpenMM at 300 K and 1 bar (`tests/data/martini/make_water_box.py` remakes
it).
