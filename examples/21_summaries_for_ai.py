"""From 3D back to text: a summary for people and language models, a table, and a view.

``boonza.summarize`` states in words and numbers what a structure holds, so
a language model can reason about it without the coordinates.  The same
facts come as JSON-ready data.  ``to_pandas`` gives the atoms as a table,
and ``view`` draws the structure in a notebook (or a standalone page).

    python examples/21_summaries_for_ai.py
"""

import importlib.util
import json

from _common import DATA, OUT

import boonza

hb = boonza.load(DATA / "1HHO.pdb")
summary = boonza.summarize(hb, focus="resname HEM and chain A", max_items=5)
lines = str(summary).splitlines()
print("\n".join(lines[:40]))
if len(lines) > 40:
    print(f"... ({len(lines) - 40} more lines)")

data = summary.to_dict()
(OUT / "1HHO_summary.json").write_text(json.dumps(data, indent=1))
heme = data["focus"]
print("\nas data: the heme's closest residue is", heme["neighbors"][0]["residue"],
      f"at {heme['neighbors'][0]['distance']} Å; buried {heme['buried_fraction']:.0%}")  # fmt: skip

if importlib.util.find_spec("pandas"):
    table = hb.to_pandas("resname HEM and chain A and element Fe N")
    print("\n" + table[["atom", "name", "element", "resname", "chain", "resid",
                        "x", "y", "z"]].to_string(index=False))  # fmt: skip

hb.view("protein or resname HEM").save(OUT / "1HHO_view.html")
print("\nwrote", OUT / "1HHO_summary.json", "and", OUT / "1HHO_view.html")
