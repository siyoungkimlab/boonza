# scripts

Working scripts that live around boonza rather than inside it: structure
preparation with tools boonza does not wrap, and the analysis of a batch of
`boonza md` runs.  They are kept here so they are versioned and shared, not
because they are part of the package -- nothing imports them, and `boonza`
itself never runs them.

| | |
|---|---|
| `prep_moe.sh` | MOE StructurePreparation over many structure files, in parallel: `foo.pdb` -> `foo.prepped.pdb` (or `.mae`, `.cif`, `.dms`).  Fetches PDB codes from RCSB, caps free termini, keeps or drops ligands.  Needs `moebatch` and an SVL script (`MOEBATCH`, `PREP_SVL`). |
| `prepwizard.sh` | the same job with Schrodinger's Protein Preparation Wizard (`PREPWIZARD`, or `prepwizard` on the PATH). |
| `boonza_md_analysis.py` | one CSV row per `boonza md` run, or a per-frame dRMSD for one run: how far the ligand's pocket-distance matrix moved from the start of production, symmetry-corrected (`boonza.drmsd`), with the settings, the monitor's verdict and the boltz confidence of each sample beside it. |
| `boonza_md_convert.py` | the runs of `analysis.csv` in dRMSD order, as `md_final/000_<drmsd>_<folder>.mae`, water and ions dropped.  A qmap script (its header lines pick the queue and the environment), not plain Python: the `!` lines are shell. |

## The pocket selection, in any model

`boonza_md_analysis.py` measures the ligand against pocket atoms, one to a
residue, and its default takes them from whichever model the run used:

```
--pocketsel "(protein and name CA) or name BB GC"
```

`CA` all-atom, `BB` under Martini, `GC` under SIRAH -- whose map puts `GC` on
the alpha carbon itself, so it is the same atom by another name.  Two things
make the `protein and` half necessary and the rest of it not: a calcium ion is
named `CA` too, and `protein` matches no coarse-grained bead at all, since
boonza reads a residue as protein from an N/CA/C backbone that beads do not
have.  `name GN GC GO` takes SIRAH's whole backbone instead, as `boonza sites`
does, at three atoms a residue.

## Style

These follow their own conventions, not the package's: `ruff` is not run over
this directory (`boonza_md_convert.py` is not even Python to it), so keep them
as their authors wrote them.
