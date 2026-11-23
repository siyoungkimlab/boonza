# Fitted models and substructures for the ADME report

Three data files `boonza.viewer.admet` loads when it needs them.  None of them
is boonza's own work: each is a published dataset, or a model fitted from one,
redistributed here under a licence that permits it.  Keep the notices below
with the files.

## `pka_forest.json.gz` — site-level pKa

A 200-tree random forest, 102 200 nodes, that predicts the pKa of an ionisable
atom the SMARTS table in `admet.py` has already located.  It improves the
number, not the coverage.

Fitted from the pKa collection of **Baltruschat & Czodrowski**,
[github.com/czodrowskilab/Machine-learning-meets-pKa](https://github.com/czodrowskilab/Machine-learning-meets-pKa),
doi:10.5281/zenodo.7884512, released under **CC BY 4.0** — which permits
redistribution with attribution.

Scaffold-split accuracy on that collection: MAE 0.844, RMSE 1.183.  The SMARTS
table alone scores MAE 1.318 on the same rows, and Marvin — commercial — 0.803.

## `logs_forest.json.gz` — aqueous solubility

A 120-tree random forest, 11 400 nodes, for log S in mol/L.

Fitted from **AqSolDB** (Sorkun, Khetan & Er, *Sci Data* 2019, **6**, 143),
doi:10.7910/DVN/OVHAW8, released under **CC0 1.0** — a public-domain
dedication, so it carries no redistribution condition at all.

Scaffold-split accuracy: RMSE 1.31, MAE 0.93, R² 0.686.  Delaney's ESOL on the
same split: RMSE 1.683, MAE 1.16, R² 0.482.

## `dimorphite_sites.json` — ionisable substructures

66 SMARTS patterns, used only to find ionisable atoms boonza's own table
misses.  The pKa then comes from the forest above, never from the means stored
here.

From **Dimorphite-DL**, `dimorphite_dl/smarts/site_substructures.smarts`,
copyright 2018 Jacob D. Durrant, University of Pittsburgh,
[github.com/durrantlab/dimorphite_dl](https://github.com/durrantlab/dimorphite_dl).
Licensed under the **Apache License 2.0**; the licence ships beside it as
`LICENSE.dimorphite-dl`.  Cite: Ropp, Kaminsky, Yablonski & Durrant,
"Dimorphite-DL: an open-source program for enumerating the ionization states of
drug-like small molecules", *J Cheminform* 2019, **11**, 14,
doi:10.1186/s13321-019-0336-9

Changes from the original, as the Apache licence section 4(b) asks:

- transcribed from the tab-separated `.smarts` file into JSON;
- entries whose pKa mean is the `-1000.0` sentinel are dropped, since only
  sites carrying a real value are used;
- the mean and standard deviation are kept, the ordering is unchanged, and no
  SMARTS pattern has been altered.

## Format

Each forest is gzipped JSON: `{"features": [...], "info": {...}, "trees":
[[feature, threshold, left, right, value], ...]}`.  A tree is five flat arrays,
walked with plain comparisons — `-2` in `feature` marks a leaf.  No sklearn, no
pickle, nothing to unpack: the whole walk is six lines in `admet.py`.

Delete any of these files and the report falls back to what it did before:
the SMARTS table's pKa, no extra sites, and Delaney's ESOL.
