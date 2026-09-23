# Martini 3 parameters

The Martini 3 force field, version 3.0.0, files dated 2021-03-29, from
https://cgmartini.nl, included unchanged so that `boonza martinize` and
`boonza bilayer` work without a separate download.

**Please cite** the work these parameters come from:

> PCT Souza et al., "Martini 3: a general purpose force field for
> coarse-grained molecular dynamics", *Nature Methods* 18, 382-388 (2021).
> DOI: 10.1038/s41592-021-01098-3

## Martini 2

The Martini 2 force field, from https://cgmartini.nl, for systems and lipids
that have not been ported to Martini 3.

**Please cite:**

> SJ Marrink et al., "The MARTINI force field: coarse grained model for
> biomolecular simulations", *J. Phys. Chem. B* 111, 7812-7824 (2007).
> DOI: 10.1021/jp071097f

> L Monticelli et al., "The MARTINI coarse-grained force field: extension to
> proteins", *J. Chem. Theory Comput.* 4, 819-834 (2008).
> DOI: 10.1021/ct700324x

- `martini_v2.2.itp` - particle definitions, nonbonded parameters and water
- `martini_v2.0_ions.itp` - ions
- `martini_v2.0_lipids_all_201506.itp` - the 2015-06 lipid collection

Martini 2 and Martini 3 parameters must not be mixed in one system.

Every file here is a released version as published, byte for byte. Modified
or re-tuned variants are deliberately not carried: point `--martini-itp`,
`--lipid-itp` or `martini_itp=` at your own copy to use one.
