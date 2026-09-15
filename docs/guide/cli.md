# Command line

```bash
boonza info system.dms                        # atoms, bonds, molecules, cell, tables
boonza convert system.mae system.dms           # any readable format to any writable one
boonza convert system.dms protein.pdb -s protein
boonza convert big.dms small.pdb --structure-only
boonza select system.dms "within 5 of resname LIG"     # prints atom indices
boonza validate system.dms [--strict] [--max-ring 10]
boonza knots system.dms [--max-cycle 10] [-s SEL] [--ignore-excluded]
boonza diff a.dms b.dms [--rtol 1e-6] [--no-positions]
boonza describe system.dms "resname LIG and name C1" [--terms all] [--pairs]
boonza dssp protein.pdb [--simplified]        # one line of codes per chain
boonza phipsi protein.pdb                     # phi/psi/omega table
boonza rmsd docked.pdb crystal.pdb [--ligandsel SEL] [--mobsel SEL] [--refsel SEL] [--align order|sequence|none]
boonza rmsd system.pdb crystal.pdb --traj md.xtc [--ligandsel SEL]   # one ligand RMSD per frame
boonza drmsd system.pdb --traj md.xtc [--reference crystal.pdb] [--cutoff 5]   # pocket-ligand dRMSD
boonza build --smiles SMILES -o out.sdf [--conformers 10]   # a 3D molecule from SMILES
boonza summarize complex.pdb [--focus SEL] [--json]   # a text summary for people and AI
boonza build --sequence ACDEF [--conformation helix] -o out.pdb   # a peptide
boonza parameterize in.dms out.dms -f aa.charmm.c36m -f water.tip3p_charmm   # viparr force fields
boonza parameterize in.dms out.dms -f aa.amber.ff99SB -m aa.amber.ff99SB-ILDN   # with a patch
boonza parameterize in.dms out.dms -x amber19-all.xml -x amber19/opc.xml   # OpenMM XML force fields
boonza parameterize in.dms out.dms -f aa.amber.ff19SB -f water.tip3p --gaff2   # + GAFF2 ligands
boonza md protein.pdb --workdir run   # prepare and run MD; the same command resumes it
boonza diff viparr.dms openmm.dms --canonical --no-positions   # compare two force fields
```

`parameterize` takes force fields in priority order (`-f`, names in
`$VIPARR_FFPATH` or directories); see [viparr force fields](viparr.md).

`validate`, `knots` and `diff` exit with status 1 when they find something,
so they work in scripts and CI.
