"""Molecules (bonded fragments), identical-molecule groups, and rings.

A "molecule" in boonza is a fragment: atoms connected through bonds.
``distinct_fragments`` groups molecules with identical bond graphs (the
msys FindDistinctFragments rule), and ``sssr`` finds rings.

    python examples/04_molecules_and_rings.py
"""

from collections import Counter

import numpy as np
from _common import DATA

import boonza

s = boonza.load(DATA / "1HHO.pdb")
print(s)
print("molecules:", s.nfragments)

# Bond guessing (msys rules) links each heme iron to a histidine, which makes the hemes
# part of the protein molecules.  Cut the iron-protein bonds to make hemes separate.
resname = s.residues["name"][s.atoms["residue"]]
bi, bj = s.bonds["i"], s.bonds["j"]
iron = s.atoms["anum"] == 26
cut = np.flatnonzero((iron[bi] & (resname[bj] != "HEM")) | (iron[bj] & (resname[bi] != "HEM")))
for b in cut:
    other = s.atom(int(bj[b] if iron[bi[b]] else bi[b]))
    print(f"  cutting Fe - {other.residue.name}{other.residue.resid}:{other.name}")
s.delete_bonds(cut)
print("molecules after cutting:", s.nfragments)

# What are the molecules?  Name each by the residue names it contains.
kinds = Counter()
for frag in s.fragments():
    names = sorted(set(s.residues["name"][s.atoms["residue"][frag.ids]].tolist()))
    kinds["+".join(names) if len(names) < 4 else f"protein chain ({len(frag)} atoms)"] += 1
for kind, count in kinds.most_common(6):
    print(f"  {count:4d} x {kind}")

# Group identical molecules: all waters share one group, the two hemes another.
groups = s.distinct_fragments()
print("\ndistinct molecule types:", len(groups))
for rep, members in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:5]:
    atoms = s.fragment_atoms(rep)
    resname = s.residues["name"][s.atoms["residue"][atoms[0]]]
    print(f"  {len(members):4d} copies of a {len(atoms)}-atom molecule starting with {resname}")

# Rings: the smallest set of smallest rings, and fused ring systems.
heme = s.select("resname HEM and chain A")
rings = boonza.sssr(s, heme)
print("\nrings in one heme:", sorted(len(r) for r in rings))
systems = boonza.ring_systems(s, rings)
print("fused ring systems:", [len(x) for x in systems], "(the porphyrin is one system)")

aromatic = boonza.sssr(s, "protein and chain A and resname PHE TYR TRP HIS")
print("rings in PHE/TYR/TRP/HIS side chains of chain A:", len(aromatic),
      Counter(len(r) for r in aromatic))  # fmt: skip
