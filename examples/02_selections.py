"""A tour of the selection language (msys/VMD syntax).

Each line prints how many atoms a selection picks in hemoglobin (1HHO).
Selections can be combined with and/or/not and parentheses, use distances
(within, nearest), bonds (withinbonds), whole units (same residue as),
comparisons and regular expressions.

    python examples/02_selections.py
"""

from _common import DATA

import boonza

s = boonza.load(DATA / "1HHO.pdb")

selections = [
    "all",
    "protein",
    "water",
    "protein and name CA",
    "backbone and chain A",
    "sidechain and resname HIS",
    "chain B and resid 1 to 20",
    "resname HEM",
    "element Fe",
    "withinbonds 1 of element Fe",  # atoms bonded to the iron (and the iron)
    "protein and within 5 of element Fe",
    "same residue as (protein and within 5 of element Fe)",
    "water and nearest 5 to resname HEM",  # the five waters closest to any heme
    "exwithin 4 of resname HEM",  # near a heme, heme itself excluded
    'name "C[AB]"',  # double quotes: a regular expression (CA and CB)
    "name 'CA'",  # single quotes: a literal name
    "atomicnumber > 8",
    "z > 20 and protein",
    "hydrophobic and name CA",
    "charged and name CA",
    "not (protein or water)",
]
for text in selections:
    print(f"{len(s.select(text)):6d}  {text}")

# A selection gives indices, positions, and whole residues or molecules.
pocket = s.select("protein and within 4 of resname HEM")
residues = pocket.expand("residue")
names = sorted({f"{s.atom(i).residue.name}{s.atom(i).residue.resid}" for i in residues.ids})
print(f"\nheme pocket: {len(residues)} atoms in {len(names)} residues, e.g. {names[:6]}")

# Selections also accept boolean masks and index lists.
heavy = s.select(s.atoms["anum"] > 1)
print("heavy atoms:", len(heavy), "| first five indices:", heavy.ids[:5].tolist())
