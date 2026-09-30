# SIRAH

`sirah_x2.2.zip` is the SIRAH force field, release of July 2020
(`sirah_x2.2_20-07.ff`), from <https://www.sirahff.com>: its parameter files
(`ffnonbonded.itp`, `ffbonded.itp`, `lipid_ffbonded.itp`, `forcefield.itp`),
its residue libraries (`aminoacids.rtp`, `dna.rtp`, `lipid.rtp`, the `.tdb`
termini and `specbond.dat`), its water and ions (`wt4.itp`, `sirah_ions.itp`,
`solv.itp`, and the equilibrated boxes `wt416.gro` and `wlsbox.gro`), and its
mapping files (`tools/CGCONV/maps/*.map`, of which boonza reads
`sirah_prot.map`, `sirah_dna.map` and `sirah_ions.map` to place beads).

The tutorials, the example PDBs and the tools are not here; the tools are
GPL v2, and boonza maps and builds topologies itself.

A run directory gets only what its topology includes -- `forcefield.itp` and
`solv.itp` with what they include in turn -- since that is what a run needs;
everything else is read from this archive where it is needed (the residue
libraries to build a topology, the maps to place beads, `wt416.gro` to fill a
box).  `boonza.sirah.unpack(directory, everything=True)` writes the whole
release instead, for running SIRAH's own tools beside it.

Copyright (c) 2014, Montevideo, Uruguay. Please cite SIRAH when you use it:

- M. R. Machado, E. E. Barrera, F. Klein, M. Sóñora, S. Silva and S. Pantano,
  *J. Chem. Theory Comput.* **15**, 2719-2733 (2019).
- L. Darré, M. R. Machado, A. F. Brandner, H. C. González, S. Ferreira and
  S. Pantano, *J. Chem. Theory Comput.* **11**, 723-739 (2015).
