"""boonza.kinase against Table S7 of Modi & Dunbrack (PNAS 2019, 116:6818, SI Appendix).

    python benchmarks/kinase_table_s7.py CACHE_DIR [-v]

The table lists, for 26 FDA-approved inhibitors, the cluster of each kinase chain they
are bound to in the PDB: 130 chains of 96 entries and 42 kinases
(``kinase_table_s7.csv``, typed from the table).  Each entry is downloaded from the RCSB
into CACHE_DIR once, and every listed chain is classified.

A chain agrees when its spatial group and cluster are the paper's; a chain the paper
calls noise agrees when the spatial group is the paper's and no cluster is assigned.
The rule here is Kincore's (fixed D1/D2 thresholds, nearest centroid within 0.45); the
paper's own labels came from clustering, so a few borderline chains may differ.
"""

import csv
import sys
import urllib.request
from pathlib import Path

import boonza
from boonza.kinase import CENTROIDS, kinase_states

GROUP = {name: group for group, clusters in CENTROIDS.items() for name in clusters}
#: residues the motifs cannot find: ERBB3 has a His for the αC Glu, 3SOA is the K42M mutant
GIVEN = {"4OTW": {"lys": 723, "glu": 740}, "3SOA": {"lys": 42, "glu": 60}}


def main(cache, verbose=False):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(Path(__file__).with_suffix(".csv"), encoding="utf-8")))
    systems, agree, differ = {}, 0, []
    for r in rows:
        pdb = r["pdb"]
        if pdb not in systems:
            path = cache / f"{pdb.lower()}.cif"
            if not path.exists():
                url = f"https://files.rcsb.org/download/{pdb.lower()}.cif"
                urllib.request.urlretrieve(url, path)  # noqa: S310
            systems[pdb] = boonza.load(path)
        noise = r["cluster"].endswith("-noise")
        want = (r["cluster"].split("-")[0], None) if noise else (GROUP[r["cluster"]], r["cluster"])
        found = kinase_states(systems[pdb], chain=r["chain"], **GIVEN.get(pdb, {}))
        st = next((x for x in found if (x.spatial, x.cluster) == want), found[0] if found else None)
        got = "not found" if st is None else st.label
        if st is not None and (st.spatial, st.cluster) == want:
            agree += 1
        else:
            differ.append((r, got, st))
        if verbose:
            print(f"{pdb} {r['chain']} {r['kinase']:8} {r['inhibitor']:12} {r['cluster']:15} {got}")
    print(f"{agree} of {len(rows)} chains agree with Table S7")
    for r, got, st in differ:
        detail = "" if st is None else f"  D1 {st.d1:.1f} D2 {st.d2:.1f} Lys {st.lys} Glu {st.glu}"
        print(f"  {r['pdb']} {r['chain']} {r['kinase']} ({r['inhibitor']}): paper {r['cluster']}, "
              f"here {got}{detail}")  # fmt: skip
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], "-v" in sys.argv[2:]))
