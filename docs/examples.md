# Examples

Runnable scripts in [`examples/`](../examples). Each one teaches a task step by
step and prints what it finds. The output below is what the scripts
printed when this page was generated with `python docs/gen_examples.py`.
Run any of them with `python examples/NN_name.py`.

Examples marked *needs openmm* or *needs rdkit* print a note and stop when
that package is missing. The structures in `examples/data` come from the
RCSB PDB.

- [01_load_and_inspect.py](#01-load-and-inspect) — Load a structure and look around: hierarchy, columns, selections, sequences
- [02_selections.py](#02-selections) — A tour of the selection language (msys/VMD syntax)
- [03_build_and_edit.py](#03-build-and-edit) — Build a system from scratch, edit it, and write it in several formats
- [04_molecules_and_rings.py](#04-molecules-and-rings) — Molecules (bonded fragments), identical-molecule groups, and rings
- [05_parameterize_with_openmm.py](#05-parameterize-with-openmm) — Parameterize a protein with OpenMM's Amber14 force field and save it as DMS
- [06_edit_forcefield.py](#06-edit-forcefield) — Edit force-field parameters: copy-on-write terms, NBFIX overrides, exclusions
- [07_superposition.py](#07-superposition) — Superpose proteins: ChimeraX-style matchmaker, PyMOL-style cealign, plain fits
- [08_sequence_alignment.py](#08-sequence-alignment) — Pairwise sequence alignment, identity and similarity
- [09_trajectory_analysis.py](#09-trajectory-analysis) — Make a short trajectory, then analyze it
- [10_periodic_boxes.py](#10-periodic-boxes) — Periodic boundaries: minimum-image distances, making molecules whole, wrapping
- [11_rdkit_bridge.py](#11-rdkit-bridge) — Small molecules through RDKit: SMILES to 3D, SMARTS selections, bond orders
- [12_validate_and_diff.py](#12-validate-and-diff) — Catch problems before simulating: validate, find_knots, diff
- [13_structure_analysis.py](#13-structure-analysis) — Analyze one structure: surface area, contacts, secondary structure, angles
- [14_command_line.sh](#14-command-line) — The boonza command line on the example data
- [15_ligand_rmsd.py](#15-ligand-rmsd) — Symmetry-corrected ligand RMSD: equivalent atoms should not count as errors

<a id="01-load-and-inspect"></a>

## 01_load_and_inspect.py: Load a structure and look around: hierarchy, columns, selections, sequences

Hemoglobin (PDB 1HHO) has two protein chains, heme groups and crystal
waters.  Everything boonza knows about it is stored as NumPy columns, and
handles let you walk from an atom to its residue, chain and molecule.

    python examples/01_load_and_inspect.py

```python
import numpy as np
from _common import DATA

import boonza

s = boonza.load(DATA / "1HHO.pdb")
print(s)
print("chains:", s.chains["name"].tolist(), "(protein chains A and B, then their waters)")
print("residues:", s.nresidues, "| molecules:", s.nfragments)

# Every attribute is a NumPy column: slice, mask and compute on it directly.
print("first atom names:", s.atoms["name"][:6].tolist())
print("atomic numbers present:", sorted(set(s.atoms["anum"].tolist())))
print("residue names (first 8):", s.residues["name"][:8].tolist())

# Walk the hierarchy from one atom with handles.
atom = s.atom(100)
res = atom.residue
print(f"atom 100: {atom.name} ({atom.element}) in {res.name}{res.resid}, chain {res.chain.name}")
print(f"  bonded to {[s.atom(i).name for i in atom.bonded_atoms]}; its molecule has "
      f"{len(atom.fragment)} atoms")  # fmt: skip

# Selections use the msys/VMD language and return AtomSel objects.
ca = s.select("protein and name CA")
heme = s.select("resname HEM")
print("C-alpha atoms:", len(ca))
print("heme atoms:", len(heme), "in", len(heme.residues), "heme residues")
# Bond guessing (msys rules) links each heme iron to a histidine, so the hemes
# are part of the protein molecules here; example 04 shows how to separate them.
iron = s.select("element Fe").ids
partners = [s.atom(int(j)) for i in iron for j in s.atom(int(i)).bonded_atoms]
print("iron bonded to:", sorted({f"{a.residue.name}{a.residue.resid}:{a.name}" for a in partners
                                 if a.residue.name != "HEM"}))  # fmt: skip
print("waters within 3.5 A of a heme:", len(s.select("water and within 3.5 of resname HEM")))

# Sequences of the protein chains.
for chain in ("A", "B"):
    seq = boonza.sequence(s, chain)
    print(f"chain {chain}: {len(seq)} residues  {seq[:30]}...")

# Geometry is plain NumPy on the positions (Angstrom).
center = s.positions[ca.ids].mean(axis=0)
extent = np.ptp(s.positions[ca.ids], axis=0)
print("protein center:", np.round(center, 1), " extent:", np.round(extent, 1))
```

Output:

```text
<System 'examples/data/1HHO.pdb': 2396 atoms, 2360 bonds, 401 residues, 4 chains, 0 tables>
chains: ['A', 'B', 'A', 'B'] (protein chains A and B, then their waters)
residues: 401 | molecules: 112
first atom names: ['N', 'CA', 'C', 'O', 'CB', 'CG1']
atomic numbers present: [6, 7, 8, 15, 16, 26]
residue names (first 8): ['VAL', 'LEU', 'SER', 'PRO', 'ALA', 'ASP', 'LYS', 'THR']
atom 100: CE2 (C) in TRP14, chain A
  bonded to ['CD2', 'NE1', 'CZ2']; its molecule has 1114 atoms
C-alpha atoms: 287
heme atoms: 86 in 2 heme residues
iron bonded to: ['HIS87:NE2', 'HIS92:NE2', 'OXY150:O1']
waters within 3.5 A of a heme: 3
chain A: 141 residues  VLSPADKTNVKAAWGKVGAHAGEYGAEALE...
chain B: 146 residues  VHLTPEEKSAVTALWGKVNVDEVGGEALGR...
protein center: [23.6 34.1 11.6]  extent: [46.8 40.  42.7]
```

<a id="02-selections"></a>

## 02_selections.py: A tour of the selection language (msys/VMD syntax)

Each line prints how many atoms a selection picks in hemoglobin (1HHO).
Selections can be combined with and/or/not and parentheses, use distances
(within, nearest), bonds (withinbonds), whole units (same residue as),
comparisons and regular expressions.

    python examples/02_selections.py

```python
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
```

Output:

```text
  2396  all
  2192  protein
   109  water
   287  protein and name CA
   564  backbone and chain A
   114  sidechain and resname HIS
   153  chain B and resid 1 to 20
    86  resname HEM
     2  element Fe
    14  withinbonds 1 of element Fe
    15  protein and within 5 of element Fe
    54  same residue as (protein and within 5 of element Fe)
     5  water and nearest 5 to resname HEM
    84  exwithin 4 of resname HEM
   554  name "C[AB]"
   287  name 'CA'
     9  atomicnumber > 8
   481  z > 20 and protein
   138  hydrophobic and name CA
    74  charged and name CA
    95  not (protein or water)

heme pocket: 312 atoms in 35 residues, e.g. ['ALA65', 'ALA70', 'ASN102', 'ASN97', 'HIS45', 'HIS58']
heavy atoms: 2396 | first five indices: [0, 1, 2, 3, 4]
```

<a id="03-build-and-edit"></a>

## 03_build_and_edit.py: Build a system from scratch, edit it, and write it in several formats

Covers: adding residues, atoms and bonds (one at a time and in bulk),
custom columns, deleting atoms (and the old-to-new index map), stale
handles, cloning a subset, appending systems, and file round trips.

    python examples/03_build_and_edit.py

```python
import numpy as np
from _common import OUT, water_box

import boonza

# --- one molecule, atom by atom --------------------------------------------
m = boonza.System("methanol")
res = m.add_residue(name="MOH", resid=1)  # creates a ct and a chain as needed
c = res.add_atom(name="C", anum=6, mass=12.011, pos=(0.0, 0.0, 0.0))
o = res.add_atom(name="O", anum=8, mass=15.999, pos=(1.43, 0.0, 0.0))
h = res.add_atom(name="H", anum=1, mass=1.008, pos=(1.75, 0.9, 0.0))
m.add_bond(c, o)
m.add_bond(o, h)
print(m, "| bonds:", m.bonds["i"].tolist(), m.bonds["j"].tolist())

# --- many molecules at once: a water box (see _common.water_box) ------------
box = water_box(n=5)  # 125 waters, built with add_atoms/add_bonds in bulk
print(box, "| molecules:", box.nfragments, "| cell:", np.diag(box.cell))

# Custom columns live next to the built-in ones.
box.atoms["depth"] = box.positions[:, 2]
print("custom atom columns:", box.atoms.props)

# --- deleting atoms renumbers them ----------------------------------------------
keep_me = box.atom(100)
old_to_new = box.delete_atoms("same residue as x < 2")  # whole waters: first layer
print("after deleting the first layer of waters:", box.natoms, "atoms,", box.nfragments,
      "molecules")  # fmt: skip
print("old atom 100 is now atom", old_to_new[100], "| deleted atoms map to -1:",
      int((old_to_new == -1).sum()))  # fmt: skip
try:
    print(keep_me.name)  # the handle was made before the deletion
except boonza.StaleHandleError as e:
    print("stale handle caught:", e)
fresh = box.atom(int(old_to_new[100]))
print("fresh handle:", fresh.name, fresh.residue.resid)

# --- cloning, appending ------------------------------------------------------
first10 = box.clone("resid 26 to 35")  # a self-contained copy of part of the system
print("clone of 10 waters:", first10.natoms, "atoms")
mixture = first10.copy()
m.positions = m.positions + [0.0, 0.0, 20.0]  # place it clear of the waters
new_ids = mixture.append(m)  # add the methanol; returns its new atom indices
print("water + methanol:", mixture.natoms, "atoms; methanol atoms are", new_ids.tolist())

# --- writing and reading back ----------------------------------------------------
for suffix in (".dms", ".pdb", ".gro", ".cif", ".mae"):
    path = OUT / f"mixture{suffix}"
    boonza.save(mixture, path)
    back = boonza.load(path)
    print(f"{suffix:5s} round trip: {back.natoms} atoms, {back.nbonds} bonds")

# DMS keeps custom columns exactly; PDB bonds are re-guessed from geometry.
back = boonza.load(OUT / "mixture.dms")
print("depth column survived in DMS:", "depth" in back.atoms.props)
```

Output:

```text
<System 'methanol': 3 atoms, 2 bonds, 1 residues, 1 chains, 0 tables> | bonds: [0, 1] [1, 2]
<System 'water box': 375 atoms, 250 bonds, 125 residues, 1 chains, 0 tables> | molecules: 125 | cell: [15.5 15.5 15.5]
custom atom columns: ['depth']
after deleting the first layer of waters: 300 atoms, 100 molecules
old atom 100 is now atom 25 | deleted atoms map to -1: 75
stale handle caught: Atom 100 was renumbered by a deletion or reorder; get a fresh handle from the system
fresh handle: H1 34
clone of 10 waters: 30 atoms
water + methanol: 33 atoms; methanol atoms are [30, 31, 32]
.dms  round trip: 33 atoms, 22 bonds
.pdb  round trip: 33 atoms, 22 bonds
.gro  round trip: 33 atoms, 22 bonds
.cif  round trip: 33 atoms, 22 bonds
.mae  round trip: 33 atoms, 22 bonds
depth column survived in DMS: True
```

<a id="04-molecules-and-rings"></a>

## 04_molecules_and_rings.py: Molecules (bonded fragments), identical-molecule groups, and rings

A "molecule" in boonza is a fragment: atoms connected through bonds.
``distinct_fragments`` groups molecules with identical bond graphs (the
msys FindDistinctFragments rule), and ``sssr`` finds rings.

    python examples/04_molecules_and_rings.py

```python
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
```

Output:

```text
<System 'examples/data/1HHO.pdb': 2396 atoms, 2360 bonds, 401 residues, 4 chains, 0 tables>
molecules: 112
  cutting Fe - HIS87:NE2
  cutting Fe - HIS92:NE2
  cutting Fe - OXY150:O1
  cutting Fe - OXY150:O1
molecules after cutting: 116
   109 x HOH
     2 x HEM
     2 x OXY
     1 x protein chain (1069 atoms)
     1 x protein chain (1123 atoms)
     1 x PO4

distinct molecule types: 6
   109 copies of a 1-atom molecule starting with HOH
     2 copies of a 43-atom molecule starting with HEM
     2 copies of a 2-atom molecule starting with OXY
     1 copies of a 1069-atom molecule starting with VAL
     1 copies of a 1123-atom molecule starting with VAL

rings in one heme: [5, 5, 5, 5, 16]
fused ring systems: [5] (the porphyrin is one system)
rings in PHE/TYR/TRP/HIS side chains of chain A: 22 Counter({5: 11, 6: 11})
```

<a id="05-parameterize-with-openmm"></a>

## 05_parameterize_with_openmm.py: Parameterize a protein with OpenMM's Amber14 force field and save it as DMS

OpenMM builds the force field; ``boonza.from_openmm`` turns it into DMS
tables (stretches, angles, dihedrals, 1-4 pairs, Lennard-Jones, charges).
Then we inspect the tables, report the parameters of a few atoms, check
the system, compute energies, and make sure a DMS round trip is lossless.

    python examples/05_parameterize_with_openmm.py        (needs openmm)

```python
from _common import OUT, amber_system, require

require("openmm")

import boonza  # noqa: E402

s = amber_system("1TEN.pdb")  # waters removed, hydrogens added, Amber14 parameters
print(s)
print("nonbonded:", s.nonbonded_info.vdw_funct, s.nonbonded_info.vdw_rule)
for name, table in sorted(s.tables.items()):
    print(f"  {name:14s} {len(table):6d} terms  {len(table.params):5d} parameter sets")

# Parameters of the backbone of residue 810, with atom labels and energy formulas.
report = s.describe("resid 810 and name N CA")
text = str(report).splitlines()
print("\n".join(text[:14]))
print("  ...")

# Basic checks (example 12 shows what they catch).
print("validate:", boonza.validate(s) or "no problems")

# Energies of each table through OpenMM, in kcal/mol.
for name, value in boonza.openmm_energies(s).items():
    print(f"  E[{name}] = {value:10.2f}")

# Save as DMS and confirm nothing changes on reading it back.
path = OUT / "1TEN_amber14_copy.dms"
boonza.save(s, path)
print("DMS round trip differences:", boonza.diff(s, boonza.load(path)))
```

Output:

```text
<System 'examples/output/1TEN_amber14.dms': 1375 atoms, 1386 bonds, 89 residues, 1 chains, 6 tables>
nonbonded: vdw_12_6 arithmetic/geometric
  angle_harm       2514 terms     41 parameter sets
  dihedral_trig    4848 terms    176 parameter sets
  exclusion        7547 terms      0 parameter sets
  nonbonded        1375 terms     13 parameter sets
  pair_12_6_es     3647 terms    775 parameter sets
  stretch_harm     1386 terms     26 parameter sets
units: Å, degrees, kcal/mol, e.  nonbonded vdw_12_6 (4 epsilon ((sigma/r)^12 - (sigma/r)^6)), combining rule arithmetic/geometric

atoms
  index        label  element   mass   charge  param    sigma  epsilon
    105   A:GLU810:N        N  14.01  -0.4157      0     3.25     0.17
    106  A:GLU810:CA        C  12.01   0.0145      2  3.39967   0.1094

pairs
    i    j     label_i      label_j           qq  excluded    sigma   epsilon  nbfix
  105  106  A:GLU810:N  A:GLU810:CA  -0.00602765       yes  3.32483  0.136374     no

angle_harm (16 terms)  E = fc (theta - theta0)^2
        atoms                                labels  param  theta0  fc  constrained
    87,89,105     A:ILE809:CA A:ILE809:C A:GLU810:N      4   116.6  70            0
  ...
validate: no problems
  E[stretch_harm] =     117.40
  E[angle_harm] =     237.58
  E[dihedral_trig] =    1071.84
  E[nonbonded] =   -2074.08
  E[total] =    -647.26
DMS round trip differences: []
```

<a id="06-edit-forcefield"></a>

## 06_edit_forcefield.py: Edit force-field parameters: copy-on-write terms, NBFIX overrides, exclusions

Many terms share one parameter row.  Changing a parameter through a term
copies the row first, so only that term changes.  Pair-specific Lennard-
Jones overrides (NBFIX) change one type pair.  ``diff`` shows exactly what
changed, and ``update_exclusions`` rebuilds exclusions and 1-4 pairs from
the bonds.

    python examples/06_edit_forcefield.py        (needs openmm)

```python
from _common import amber_system, require

require("openmm")

import numpy as np  # noqa: E402

import boonza  # noqa: E402

original = amber_system("1TEN.pdb")
s = original.copy()

# --- copy-on-write: change one bond, not every bond of that type ----------
stretch = s.table("stretch_harm")
same_type = np.flatnonzero(stretch.param_ids == stretch.param_ids[0])
print(f"stretch term 0 shares its parameter row with {len(same_type) - 1} other terms")
print("fc before:", stretch[0]["fc"])
stretch[0]["fc"] = 500.0
fc = stretch.values("fc")
print(f"fc after:  term 0 = {fc[0]}, term {same_type[1]} (same type) = {fc[same_type[1]]}")

# --- NBFIX: a special Lennard-Jones pair between two atom types ------------
nb = s.table("nonbonded")
o_type = int(nb.param_ids[s.select("name O").ids[0]])  # backbone carbonyl O
n_type = int(nb.param_ids[s.select("name N").ids[0]])  # backbone amide N
print(f"\nLJ type of O: {nb.params.row(o_type)}")
print(f"LJ type of N: {nb.params.row(n_type)}")
nb.overrides.set(o_type, n_type, sigma=2.9, epsilon=0.25)
print("override stored:", nb.overrides.get(o_type, n_type))

pairs = s.describe("resid 810 and name N O", pairs=True).pairs
for p in pairs:
    print(f"  {p['label_i']} - {p['label_j']}: sigma {p['sigma']:.3f}, "
          f"epsilon {p['epsilon']:.4f}, nbfix {p['nbfix']}")  # fmt: skip

# --- what changed?  diff compares every table as a set of terms -------------
for d in boonza.diff(original, s):
    print("diff:", str(d)[:100])
e0, e1 = boonza.openmm_energies(original), boonza.openmm_energies(s)
print(f"total energy: {e0['total']:.2f} -> {e1['total']:.2f} kcal/mol")

# --- exclusions and 1-4 pairs rebuilt from bonds ----------------------------
rebuilt = original.copy()
rebuilt.del_table("exclusion")
rebuilt.del_table("pair_12_6_es")
boonza.update_exclusions(rebuilt, pair_scales=(0.5, 1 / 1.2))  # Amber 1-4 scaling
changes = boonza.diff(original, rebuilt, rtol=1e-5, tables=["exclusion", "pair_12_6_es"])
print("\nrebuilt exclusions and 1-4 pairs vs original:", changes or "identical")
```

Output:

```text
stretch term 0 shares its parameter row with 126 other terms
fc before: 434.0
fc after:  term 0 = 500.0, term 1 (same type) = 434.0

LJ type of O: {'sigma': 2.959921901149463, 'epsilon': 0.21}
LJ type of N: {'sigma': 3.249998523775958, 'epsilon': 0.17}
override stored: {'sigma': 2.9, 'epsilon': 0.25}
  A:GLU810:N - A:GLU810:O: sigma 2.900, epsilon 0.2500, nbfix True
diff: overrides: nonbonded: 1 pair overrides are in only one system
diff: terms: stretch_harm: 1 of 1386 terms differ (max deviation fc=66): (0, 1) fc 434!=500
total energy: -647.26 -> -689.57 kcal/mol

rebuilt exclusions and 1-4 pairs vs original: identical
```

<a id="07-superposition"></a>

## 07_superposition.py: Superpose proteins: ChimeraX-style matchmaker, PyMOL-style cealign, plain fits

* myoglobin (1MBN) onto hemoglobin alpha (1HHO chain A): ~25% sequence
  identity, so a sequence alignment guided by secondary structure is needed;
* fibronectin FN3 domains 1FNA onto 1TEN: remote homologs, where the
  structure-only CE method shines.

    python examples/07_superposition.py

```python
from _common import DATA, OUT

import boonza

hemo = boonza.load(DATA / "1HHO.pdb")
myo = boonza.load(DATA / "1MBN.pdb")

# matchmaker moves `myo` onto `hemo` and reports what it paired.
r = boonza.matchmaker(myo, hemo, reference_chain="A")
print("matchmaker, myoglobin -> hemoglobin alpha")
print(f"  {len(r.kept)} of {len(r.mobile_atoms)} aligned residue pairs kept, RMSD {r.rmsd:.3f} A "
      f"(all pairs {r.full_rmsd:.3f} A)")  # fmt: skip
print("  reference:", r.aligned_reference[:60])
print("  mobile:   ", r.aligned_mobile[:60])
boonza.save(myo, OUT / "1MBN_on_1HHO.pdb")

# Without choosing chains, every chain pair is tried and the best one is used.
best = boonza.matchmaker(boonza.load(DATA / "1MBN.pdb"), hemo, apply=False)
print("  best-scoring hemoglobin chain:", hemo.chains["name"][best.reference_chain])

# A naive identity-only alignment pairs far fewer residues.
naive = boonza.superpose(boonza.load(DATA / "1MBN.pdb"), hemo, ref_sel="chain A and name CA",
                         apply=False)  # fmt: skip
print(f"  (identity-only alignment: {naive.n_used} of {naive.n_matched} pairs, "
      f"RMSD {naive.rmsd:.3f} A)")  # fmt: skip

# cealign needs no sequence similarity at all.
ten, fna = boonza.load(DATA / "1TEN.pdb"), boonza.load(DATA / "1FNA.pdb")
ce = boonza.cealign(fna, ten)
print("\ncealign, 1FNA -> 1TEN")
print(f"  {len(ce.mobile_atoms)} C-alpha pairs, RMSD {ce.rmsd:.3f} A, z-score {ce.z_score:.2f}")
seq_id = boonza.align_sequences(boonza.sequence(ten), boonza.sequence(fna)).identity
print(f"  sequence identity only {100 * seq_id:.0f}%")
boonza.save(fna, OUT / "1FNA_on_1TEN.pdb")

# The same number by hand: a Kabsch fit of the pairs matchmaker kept, then the RMSD.
mob_xyz = boonza.load(DATA / "1MBN.pdb").positions[r.mobile_atoms[r.kept]]
ref_xyz = hemo.positions[r.reference_atoms[r.kept]]
R, t = boonza.kabsch(mob_xyz, ref_xyz)  # mob_xyz @ R.T + t best matches ref_xyz
print(f"\nby hand: RMSD {boonza.rmsd(mob_xyz, ref_xyz):.1f} A before fitting, "
      f"{boonza.rmsd(mob_xyz @ R.T + t, ref_xyz):.3f} A after the Kabsch fit")  # fmt: skip
```

Output:

```text
matchmaker, myoglobin -> hemoglobin alpha
  105 of 135 aligned residue pairs kept, RMSD 1.052 A (all pairs 2.193 A)
  reference: VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFD....LSH.....G
  mobile:    VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPET...LEKFDRFKHLKTEAEMKA
  best-scoring hemoglobin chain: A
  (identity-only alignment: 24 of 45 pairs, RMSD 1.188 A)

cealign, 1FNA -> 1TEN
  80 C-alpha pairs, RMSD 1.515 A, z-score 5.19
  sequence identity only 27%

by hand: RMSD 35.0 A before fitting, 1.052 A after the Kabsch fit
```

<a id="08-sequence-alignment"></a>

## 08_sequence_alignment.py: Pairwise sequence alignment, identity and similarity

Global (Needleman-Wunsch) and local (Smith-Waterman) alignment with
BLOSUM-62 and affine gaps, EMBOSS-style statistics, and an identity matrix
over several chains.

    python examples/08_sequence_alignment.py

```python
import numpy as np
from _common import DATA

import boonza

hen = boonza.load(DATA / "1LYZ.pdb")
human = boonza.load(DATA / "1LZ1.pdb")

# Systems can be passed directly; their chain sequence is used.
aln = boonza.align_sequences(hen, human)
print(aln)
print(f"\nidentity {aln.identity:.1%}, similarity {aln.similarity:.1%}, "
      f"gaps {aln.gap_fraction:.1%}")  # fmt: skip

# Local alignment finds the best-matching stretch only.
motif = "CAKKIVSDGNGMNAWVAWR"
local = boonza.align_sequences(boonza.sequence(hen), motif, mode="local")
print(f"\nlocal alignment of a motif: score {local.score:g}, starts at residue "
      f"{local.pairs[0, 0] + 1} of hen lysozyme")  # fmt: skip
print(local.aligned_a)
print(local.aligned_b)

# Free end gaps (EMBOSS needle's default) for a fragment against a full chain.
frag = boonza.sequence(hen)[20:60]
semi = boonza.align_sequences(boonza.sequence(hen), frag, end_gaps=False)
print(f"\nfragment 21-60 with free end gaps: score {semi.score:g}, "
      f"identity over the fragment {semi.identical / len(frag):.0%}")  # fmt: skip

# Identity matrix over several chains.
hemo = boonza.load(DATA / "1HHO.pdb")
chains = {
    "Hb alpha": boonza.sequence(hemo, "A"),
    "Hb beta": boonza.sequence(hemo, "B"),
    "myoglobin": boonza.sequence(boonza.load(DATA / "1MBN.pdb")),
    "lysozyme": boonza.sequence(hen),
}
m = boonza.identity_matrix(list(chains.values()))
print("\npairwise identity (%)")
print(" " * 11 + "".join(f"{k:>11s}" for k in chains))
for k, row in zip(chains, m, strict=True):
    print(f"{k:>11s}" + "".join(f"{100 * v:11.0f}" for v in row))
names = list(chains)
i, j = max(((i, j) for i in range(len(names)) for j in range(i + 1, len(names))),
           key=lambda ij: m[ij])  # fmt: skip
print(f"closest pair: {names[i]} and {names[j]} ({m[i, j]:.0%} identical)")
assert np.allclose(np.diag(m), 1)
```

Output:

```text
global alignment, score 447, length 130: identity 78/130 (60.0%), similarity 100/130 (76.9%), gaps 1/130 (0.8%)

KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNT-DGSTDYGILQIN
|||.|||||..:||.|:|.|||.||.||:|.||:||.:||:|||.|. |.||||||.|||
KVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQIN

SRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTD
||:|||||:|||:.|.|::.|||||..:|..:|.|||::|.|..|:.|||||||||:..|
SRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRD

VQAWIRGCRL
|:.:::||.:
VRQYVQGCGV

identity 60.0%, similarity 76.9%, gaps 0.8%

local alignment of a motif: score 109, starts at residue 94 of hen lysozyme
CAKKIVSDGNGMNAWVAWR
CAKKIVSDGNGMNAWVAWR

fragment 21-60 with free end gaps: score 216, identity over the fragment 100%

pairwise identity (%)
              Hb alpha    Hb beta  myoglobin   lysozyme
   Hb alpha        100         43         25         13
    Hb beta         43        100         24         16
  myoglobin         25         24        100         11
   lysozyme         13         16         11        100
closest pair: Hb alpha and Hb beta (43% identical)
```

<a id="09-trajectory-analysis"></a>

## 09_trajectory_analysis.py: Make a short trajectory, then analyze it

With OpenMM installed this runs 2 ps of implicit-solvent molecular dynamics
of the FN3 domain 1TEN and writes a DCD file with boonza's writer.  Without
OpenMM it makes a trajectory by jiggling the crystal structure.  Then:
RMSD, RMSF, radius of gyration, secondary structure, backbone angles and
hydrogen bonds over the frames.

    python examples/09_trajectory_analysis.py

```python
import importlib.util

import numpy as np
from _common import DATA, OUT, remove_incomplete_residues

import boonza

dcd = OUT / "1TEN_md.dcd"
if importlib.util.find_spec("openmm"):
    import openmm
    from openmm import app, unit

    pdb = app.PDBFile(str(DATA / "1TEN.pdb"))
    model = app.Modeller(pdb.topology, pdb.positions)
    model.deleteWater()
    remove_incomplete_residues(model)
    ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    model.addHydrogens(ff)
    system = ff.createSystem(model.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    sim = app.Simulation(model.topology, system,
                         openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                                         0.002 * unit.picoseconds))  # fmt: skip
    sim.context.setPositions(model.positions)
    sim.minimizeEnergy(maxIterations=200)
    s = boonza.from_openmm(model.topology, positions=model.positions)  # structure only
    with boonza.open_writer(dcd, s.natoms) as writer:
        for _ in range(20):
            sim.step(50)  # 0.1 ps between frames
            xyz = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
            writer.write(xyz.value_in_unit(unit.angstrom))
    print("ran 2 ps of MD with OpenMM")
else:
    s = boonza.load(DATA / "1TEN.pdb")
    rng = np.random.default_rng(0)
    with boonza.open_writer(dcd, s.natoms) as writer:
        for k in range(20):
            writer.write(s.positions + rng.normal(0, 0.1 + 0.02 * k, s.positions.shape))
    print("OpenMM not installed: made a jiggled trajectory instead")
boonza.save(s, OUT / "1TEN_md_topology.pdb")

# --- read it back ------------------------------------------------------------
traj = boonza.open_trajectory(dcd, s)
frames = traj.read()
print(f"{len(traj)} frames of {traj.natoms} atoms")

ca = "protein and name CA"
rmsd = boonza.rmsd_trajectory(s, frames, ca)
print("C-alpha RMSD to frame 0 (A):", np.round(rmsd[::4], 2))

rg = boonza.radius_of_gyration(s, frames, "protein")
print(f"radius of gyration: {rg.mean():.2f} +- {rg.std():.2f} A")

# RMSF needs frames superposed first: Glue fits every frame onto the first.
fit = boonza.Glue(s, fit=ca, whole=None, wrap=False)
fitted = np.array([fit(p)[0] for p in frames.positions])
rmsf = boonza.rmsf(s, fitted, ca)
resids = s.residues["resid"][s.atoms["residue"][s.select(ca).ids]]
top = np.argsort(rmsf)[::-1][:5]
print("most flexible residues:", [(int(resids[k]), round(float(rmsf[k]), 2)) for k in top])

# Secondary structure per frame, as strand/helix fractions.
codes = boonza.dssp(s, frames, simplified=True)
protein = codes[0] != "NA"
strand = (codes[:, protein] == "E").mean(axis=1)
print(f"strand fraction: first frame {strand[0]:.0%}, last frame {strand[-1]:.0%}")

phi, psi, omega = boonza.backbone_dihedrals(s, frames)
k = int(np.flatnonzero(s.residues["resid"] == 830)[0])
print(f"residue 830 phi/psi over time: {np.round(phi[::5, k])} / {np.round(psi[::5, k])}")

# Hydrogen bonds in every frame, and the most persistent ones.
hb = boonza.hbonds(s, frames)
print("hydrogen bonds per frame:", hb.per_frame()[::4].tolist())


def label(i):
    a = s.atom(int(i))
    return f"{a.residue.name}{a.residue.resid}:{a.name}"


for (d, _h, a), f in list(hb.frequency().items())[:3]:
    print(f"  {label(d):>12s} -> {label(a):<12s} present in {f:.0%} of frames")
```

Output:

```text
removing incomplete residues: ARG802
ran 2 ps of MD with OpenMM
20 frames of 1375 atoms
C-alpha RMSD to frame 0 (A): [0.   0.3  0.47 0.55 0.65]
radius of gyration: 13.20 +- 0.04 A
most flexible residues: [(879, 0.82), (862, 0.7), (861, 0.65), (878, 0.64), (844, 0.52)]
strand fraction: first frame 54%, last frame 54%
residue 830 phi/psi over time: [-146. -138. -137. -141.] / [159. 144. 161. 163.]
hydrogen bonds per frame: [53, 46, 45, 36, 26]
    THR852:OG1 -> ASP854:OD1   present in 100% of frames
      PHE824:N -> SER807:O     present in 95% of frames
      GLU834:N -> ILE874:O     present in 95% of frames
```

<a id="10-periodic-boxes"></a>

## 10_periodic_boxes.py: Periodic boundaries: minimum-image distances, making molecules whole, wrapping

Uses a small water box.  We break molecules across the box on purpose,
then repair them, and compare orthorhombic and triclinic cells.

    python examples/10_periodic_boxes.py

```python
import numpy as np
from _common import water_box

import boonza
from boonza import pbc

s = water_box(n=4)  # 64 waters in a 12.4 A cube
L = s.cell[0, 0]
print(s, "| cell edge", L, "A")

# Minimum-image distances: two atoms near opposite faces are actually close.
a, b = np.array([[0.5, 5.0, 5.0]]), np.array([[L - 0.5, 5.0, 5.0]])
print("plain distance:", round(float(np.linalg.norm(a - b)), 2),
      "| periodic distance:", round(float(pbc.distances(a, b, s.cell)[0, 0]), 2))  # fmt: skip

# The same helpers accept (a, b, c, alpha, beta, gamma) and triclinic cells.
tric = (L, L, L, 60.0, 60.0, 90.0)
print("periodic distance in a 60/60/90 triclinic cell:",
      round(float(pbc.distances(a, b, tric)[0, 0]), 2))  # fmt: skip

# All O-O pairs within 3.2 A (cell list, periodic).
oxygens = s.positions[s.select("name O").ids]
i, j, d = pbc.capped_distances(oxygens, oxygens, 3.2, s.cell)
nonself = i != j
print("O-O pairs within 3.2 A:", int(nonself.sum() // 2), "| shortest", d[nonself].min().round(2))

# Break molecules: push the last layer of waters so their H atoms cross the x face,
# then wrap every atom into the box on its own.
shifted = s.positions + [2.6, 0.0, 0.0]
wrapped = shifted - np.floor(shifted / L) * L  # atom-wise wrapping splits molecules
oh = pbc.paired_distances(wrapped[s.bonds["i"]], wrapped[s.bonds["j"]])  # no box: raw lengths
print("longest O-H bond after atom-wise wrapping:", oh.max().round(2), "A (broken molecules)")

whole = boonza.make_whole(s, wrapped, s.cell)
oh = np.linalg.norm(whole[s.bonds["i"]] - whole[s.bonds["j"]], axis=1)
print("after make_whole:", oh.max().round(3), "A")

# Glue: make whole and wrap every molecule into the box around a center.
fix = boonza.Glue(s, center="resid 1")
fixed, box = fix(wrapped, s.cell)
center = fixed[s.select("resid 1").ids].mean(axis=0)
offsets = fixed.reshape(-1, 3, 3).mean(axis=1) - center
print("after Glue, every water is within half a box of residue 1:",
      bool((np.abs(offsets) <= L / 2 + 1e-6).all()))  # fmt: skip

# Periodic selections use the cell too.
print("waters within 3.5 A of residue 1, plain vs periodic:",
      len(s.select("same residue as within 3.5 of resid 1", pos=wrapped).ids) // 3,
      len(s.select("same residue as pbwithin 3.5 of resid 1", pos=wrapped).ids) // 3)  # fmt: skip
```

Output:

```text
<System 'water box': 192 atoms, 128 bonds, 64 residues, 1 chains, 0 tables> | cell edge 12.4 A
plain distance: 11.4 | periodic distance: 1.0
periodic distance in a 60/60/90 triclinic cell: 1.0
O-O pairs within 3.2 A: 192 | shortest 3.1
longest O-H bond after atom-wise wrapping: 11.44 A (broken molecules)
after make_whole: 0.958 A
after Glue, every water is within half a box of residue 1: True
waters within 3.5 A of residue 1, plain vs periodic: 6 9
```

<a id="11-rdkit-bridge"></a>

## 11_rdkit_bridge.py: Small molecules through RDKit: SMILES to 3D, SMARTS selections, bond orders

Aspirin goes from a SMILES string to a 3D boonza System and back.  SMARTS
patterns select atoms, and bond orders are recovered from geometry alone,
as needed for ligands read from PDB, GRO or mmCIF files.

    python examples/11_rdkit_bridge.py        (needs rdkit)

```python
from _common import OUT, require

require("rdkit")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

import boonza  # noqa: E402

# SMILES -> RDKit 3D -> boonza (aspirin)
mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
AllChem.EmbedMolecule(mol, randomSeed=1)
s = boonza.from_rdkit(mol, name="aspirin")
print(s)
print("bond orders:", sorted(set(s.bonds["order"].tolist())), "| formal charges:",
      int(abs(s.atoms["formal_charge"]).sum()))  # fmt: skip

# SMARTS patterns work inside selections.
for label, smarts in (("aromatic carbons", "c"), ("carboxylic acid", "C(=O)[OH]"),
                      ("ester", "[CX3](=O)[OX2][#6]"), ("ring atoms", "[R]")):  # fmt: skip
    print(f"  {label:17s} {len(s.select(f'smarts {smarts!r}')):3d} atoms")

# Rings found by boonza itself (no RDKit needed for this part).
print("rings:", [len(r) for r in boonza.sssr(s)])

# boonza -> RDKit keeps atom order, names and a back-reference to boonza indices.
back = s.to_rdkit()
print("round-trip SMILES:", Chem.MolToSmiles(Chem.RemoveHs(back)))
print("boonza index of RDKit atom 5:", back.GetAtomWithIdx(5).GetIntProp("boonza_index"))

# Structures from PDB/GRO have no bond orders; perceive them from geometry.
plain = s.copy()
plain.bonds["order"] = [1] * plain.nbonds
plain.atoms["formal_charge"] = [0] * plain.natoms
boonza.assign_bond_orders(plain)
same = Chem.MolToSmiles(Chem.RemoveHs(plain.to_rdkit())) == Chem.MolToSmiles(Chem.RemoveHs(back))
print("bond orders recovered from geometry:", same)

# Write an SDF with a data field, and read it back.
s.ct(0)["source"] = "boonza example"
boonza.save(s, OUT / "aspirin.sdf")
print("SDF data field:", boonza.load(OUT / "aspirin.sdf").ct(0)["source"])
```

Output:

```text
<System 'aspirin': 21 atoms, 21 bonds, 1 residues, 1 chains, 0 tables>
bond orders: [1, 2] | formal charges: 0
  aromatic carbons    6 atoms
  carboxylic acid     3 atoms
  ester               4 atoms
  ring atoms          6 atoms
rings: [6]
round-trip SMILES: CC(=O)Oc1ccccc1C(=O)O
boonza index of RDKit atom 5: 5
bond orders recovered from geometry: True
SDF data field: boonza example
```

<a id="12-validate-and-diff"></a>

## 12_validate_and_diff.py: Catch problems before simulating: validate, find_knots, diff

We take a clean parameterized protein, break it in several typical ways,
and let ``validate`` find each problem.  Then a hand-made "knot": a bond
threaded through a benzene ring.

    python examples/12_validate_and_diff.py        (needs openmm)

```python
from _common import amber_system, require

require("openmm")

import numpy as np  # noqa: E402

import boonza  # noqa: E402

clean = amber_system("1TEN.pdb")
clean.cell = np.eye(3) * 60.0  # give it a box so the box check passes
print("clean system, basic checks:", boonza.validate(clean) or "no problems")

broken = clean.copy()
broken.table("stretch_harm").delete_terms([0])  # a bond loses its stretch term
q = broken.atoms["charge"].copy()
q[10] += 0.25  # a charge typo
broken.atoms["charge"] = q
ca = broken.select("name CA").ids
broken.table("exclusion").add_terms([[int(ca[0]), int(ca[-1])]])  # a stray exclusion
pos = broken.positions
pos[ca[5]] = pos[ca[6]] + 0.3  # two atoms on top of each other
broken.positions = pos

print("\nbroken system, strict checks:")
for p in boonza.validate(broken, strict=True):
    if p.check in ("constraints", "constrained_hydrogens"):
        continue  # this example system has no constraints by design
    print(f"  [{p.check}] {p.message[:100]}")

print("\nwhat differs between the clean and broken systems:")
for d in boonza.diff(clean, broken):
    print("  " + str(d)[:110])

# A knot: a C-C bond passing through the middle of a benzene ring.
s = boonza.System("knot")
angles = np.radians(np.arange(6) * 60.0)
ring = np.column_stack([1.39 * np.cos(angles), 1.39 * np.sin(angles), np.zeros(6)])
s.add_atoms(8, name=[f"C{k}" for k in range(8)], anum=6, mass=12.011,
            pos=np.vstack([ring, [[0.0, 0.0, -0.77], [0.0, 0.0, 0.77]]]))  # fmt: skip
s.add_bonds([[k, (k + 1) % 6] for k in range(6)] + [[6, 7]])
for ring_atoms, bond, _ in boonza.find_knots(s):
    print(f"\nknot: bond {bond} passes through ring {list(ring_atoms)}")
```

Output:

```text
clean system, basic checks: no problems

broken system, strict checks:
  [stretch] 1 bonds have no stretch or constraint term: 0-1
  [charge] 1 molecules have a non-integer net charge: fragment 0 (-7.7500)
  [contacts] 1 unbonded atom pairs are within 1 Å: 70-87 (0.520)
  [extra_exclusions] 1 excluded pairs are more than three bonds apart: 3-1361

what differs between the clean and broken systems:
  atoms: charge differs for 1 atoms: 10: 0.3421 != 0.5921
  atoms: pos differs for 1 atoms: max deviation 2.525; atoms 70
  terms: exclusion: 1 terms only in the second system: (3, 1361)
  terms: stretch_harm: 1 terms only in the first system: (0, 1)

knot: bond (7, 6) passes through ring [5, 4, 3, 2, 1, 0]
```

<a id="13-structure-analysis"></a>

## 13_structure_analysis.py: Analyze one structure: surface area, contacts, secondary structure, angles

Hen egg-white lysozyme (1LYZ).

    python examples/13_structure_analysis.py

```python
import numpy as np
from _common import DATA

import boonza

s = boonza.load(DATA / "1LYZ.pdb")
protein = s.clone("protein")  # crystal waters would bury the surface
print(protein)

# Solvent accessible surface area (Shrake-Rupley, 1.4 A probe).
area = boonza.sasa(protein, mode="residue")[0]
names = [f"{n}{r}" for n, r in zip(protein.residues["name"], protein.residues["resid"],
                                   strict=True)]  # fmt: skip
print(f"total SASA {area.sum():.0f} A^2")
exposed = np.argsort(area)[::-1][:5]
print("most exposed residues:", [(names[k], round(float(area[k]))) for k in exposed])
print("buried residues (< 5 A^2):", int((area < 5).sum()))

# Residue contacts: closest heavy-atom distance for every residue pair.
dist, pairs = boonza.residue_contacts(protein)
close = pairs[dist[0] < 4.0]
print(f"\n{len(close)} residue pairs in contact (< 4 A heavy-atom distance)")
long_range = close[np.abs(close[:, 1] - close[:, 0]) > 20]
print("long-range contacts (> 20 residues apart), first five:",
      [(names[a], names[b]) for a, b in long_range[:5]])  # fmt: skip

# Secondary structure: DSSP and the ChimeraX assignment.
dssp = boonza.dssp(protein)[0]
print("\nDSSP:        ", "".join(c if c.strip() else "-" for c in dssp))
print("ChimeraX H/S:", "".join(boonza.chimerax_ss(protein)))
counts = {c: int((dssp == c).sum()) for c in "HGIEBTS"}
print("DSSP counts:", {k: v for k, v in counts.items() if v})

# Backbone dihedrals: a Ramachandran summary.
phi, psi, omega = (x[0] for x in boonza.backbone_dihedrals(protein))
helical = (phi > -100) & (phi < -30) & (psi > -80) & (psi < -10)
print(f"residues in the helical phi/psi region: {int(helical.sum())}; "
      f"cis peptide bonds: {int((np.abs(omega) < 30).sum())}")  # fmt: skip

# Backbone hydrogen bonds (Kabsch-Sander energies) and disulfides.
donor, acceptor, energy = boonza.backbone_hbonds(protein)
print(f"backbone H-bonds: {len(donor)}, strongest {energy.min():.2f} kcal/mol")
# Disulfides: PDB bonds are guessed from distance with msys's rule (S-S under 2.16 A).
# 1LYZ is an old 2 A structure with stretched S-S geometry, so compare with a looser cut.
sg = protein.select("name SG").ids
d = boonza.pbc.distances(protein.positions[sg], protein.positions[sg])
for a, b in zip(*[x.tolist() for x in np.nonzero(np.triu(d < 2.5, 1))], strict=True):
    i, j = int(sg[a]), int(sg[b])
    print(f"  CYS{protein.atom(i).residue.resid}-CYS{protein.atom(j).residue.resid}: "
          f"S-S {d[a, b]:.2f} A, bonded: {protein.find_bond(i, j) is not None}")  # fmt: skip
print(f"radius of gyration: {boonza.radius_of_gyration(protein)[0]:.2f} A")
```

Output:

```text
<System 'examples/data/1LYZ.pdb': 976 atoms, 1028 bonds, 125 residues, 1 chains, 0 tables>
total SASA 6675 A^2
most exposed residues: [('ARG128', 227), ('ARG14', 192), ('ARG45', 150), ('ARG21', 148), ('THR47', 144)]
buried residues (< 5 A^2): 22

237 residue pairs in contact (< 4 A heavy-atom distance)
long-range contacts (> 20 residues apart), first five: [('LYS1', 'PHE38'), ('LYS1', 'ASN39'), ('LYS1', 'THR40'), ('LYS1', 'SER86'), ('VAL2', 'PHE38')]

DSSP:         -B--GGGGHHHHTT-TTBTTB-THHHHHGGGTTTTBSS-EEE-TTS-EEETTTTEETTTS-BSS-TT---TT-SBGGGGGSS--HHHHHHHHHHTTSSSGGGGSHHHHHHTTTS-GGGGSTT---
ChimeraX H/S: OOOOOOHHHHHHOOOOOHHHOOHHHOOOHHHOOOOOOOOSSSOOOOOSSSHHHOOOOOOOOOOOOOOOOOOOOOOHHHHHOOOOHHHHHHHHHHOOOOOHHHHOHHHHHHOOOOOHHHHOOOOOO
DSSP counts: {'H': 25, 'G': 20, 'E': 8, 'B': 6, 'T': 31, 'S': 15}
residues in the helical phi/psi region: 44; cis peptide bonds: 0
backbone H-bonds: 78, strongest -6.42 kcal/mol
  CYS6-CYS127: S-S 2.06 A, bonded: True
  CYS30-CYS115: S-S 2.42 A, bonded: False
  CYS64-CYS80: S-S 2.34 A, bonded: False
  CYS76-CYS94: S-S 2.22 A, bonded: False
radius of gyration: 13.95 A
```

<a id="14-command-line"></a>

## 14_command_line.sh: The boonza command line on the example data

    bash examples/14_command_line.sh

After `pip install -e .` the command is simply `boonza`; this script calls
it through the Python that runs it so it also works uninstalled.

```bash
#!/usr/bin/env bash
# The boonza command line on the example data.
#
#     bash examples/14_command_line.sh
#
# After `pip install -e .` the command is simply `boonza`; this script calls
# it through the Python that runs it so it also works uninstalled.
set -e
cd "$(dirname "$0")"
boonza() { "${PYTHON:-python}" -m boonza.cli "$@"; }
mkdir -p output

echo "\$ boonza info data/1LYZ.pdb"
boonza info data/1LYZ.pdb

echo; echo "\$ boonza select data/1LYZ.pdb \"resname TRP and name CA\""
boonza select data/1LYZ.pdb "resname TRP and name CA"

echo; echo "\$ boonza convert data/1LYZ.pdb output/lysozyme_protein.cif -s protein"
boonza convert data/1LYZ.pdb output/lysozyme_protein.cif -s protein

echo; echo "\$ boonza dssp data/1LYZ.pdb --simplified"
boonza dssp data/1LYZ.pdb --simplified

echo; echo "\$ boonza phipsi data/1LYZ.pdb | head -6"
boonza phipsi data/1LYZ.pdb | head -6

echo; echo "\$ boonza diff data/1LYZ.pdb output/lysozyme_protein.cif --no-positions | head -3"
boonza diff data/1LYZ.pdb output/lysozyme_protein.cif --no-positions | head -3 || true
```

Output:

```text
$ boonza info data/1LYZ.pdb
data/1LYZ.pdb: 1102 atoms, 1066 bonds, 230 residues, 2 chains, 1 cts, 100 molecules
cell: [79.1 0 0] [0 79.1 0] [0 0 37.9]
extra atom columns: occupancy, bfactor

$ boonza select data/1LYZ.pdb "resname TRP and name CA"
212 480 494 814 840 939

$ boonza convert data/1LYZ.pdb output/lysozyme_protein.cif -s protein
wrote 976 atoms to output/lysozyme_protein.cif

$ boonza dssp data/1LYZ.pdb --simplified
A: CECCHHHHHHHHHHCCCCCECCECCHHHHHHHHHHHCCECCCEEECCCCCEEECCCCEECCCCCECCCCCCCCCCCCCEHHHHHCCCCHHHHHHHHHHCCCCCHHHHCHHHHHHCCCCCHHHHCCCCCC

$ boonza phipsi data/1LYZ.pdb | head -6
chain  resid  res      phi      psi    omega
    A      1  LYS        -   122.48        -
    A      2  VAL   -98.72   126.85   168.46
    A      3  PHE   -72.96   166.20   164.63
    A      4  GLY  -111.07   158.97  -176.44
    A      5  ARG   -56.42   -77.02  -162.90

$ boonza diff data/1LYZ.pdb output/lysozyme_protein.cif --no-positions | head -3
atoms: 1102 atoms != 976 atoms
residues: 230 residues != 125 residues
chains: 2 chains != 1 chains
```

<a id="15-ligand-rmsd"></a>

## 15_ligand_rmsd.py: Symmetry-corrected ligand RMSD: equivalent atoms should not count as errors

A plain RMSD pairs atoms by name or order.  When a ligand has equivalent
atoms (the two oxygens of a carboxylate, a flipped phenyl ring, the methyls
of a t-butyl group) the same pose can get a large RMSD just because atoms
were named the other way round.  ``symmetry_rmsd`` takes the smallest RMSD
over every mapping that preserves elements and bonds.  ``ligand_rmsd`` is
the docking workflow: superpose the proteins, then compare the ligands.
``drmsd`` needs no fitting: it compares pocket-ligand distances.

Here: the two hemes of hemoglobin, in the alpha and the beta subunit, after
superposing the subunits.

    python examples/15_ligand_rmsd.py

```python
import numpy as np
from _common import DATA

import boonza

hb = boonza.load(DATA / "1HHO.pdb")

# Superpose the beta subunit onto the alpha subunit (different sequences: matchmaker).
beta_on_alpha = hb.copy()
fit = boonza.matchmaker(beta_on_alpha, hb, mobile_chain="B", reference_chain="A")
print(f"beta onto alpha: {len(fit.kept)} C-alpha pairs, RMSD {fit.rmsd:.2f} A")

# Heme of beta (moved) vs heme of alpha, in place: a pose comparison.
r = boonza.symmetry_rmsd(beta_on_alpha, hb, atoms="resname HEM and chain B",
                         reference_atoms="resname HEM and chain A")  # fmt: skip
print(f"heme RMSD, atoms paired by order:  {r.plain_rmsd:.3f} A")
print(f"heme RMSD, symmetry-corrected:     {r.rmsd:.3f} A")

swapped = [(hb.atom(int(a)).name, beta_on_alpha.atom(int(b)).name)
           for a, b in zip(r.reference_atoms, r.mapping, strict=True)
           if hb.atom(int(a)).name != beta_on_alpha.atom(int(b)).name]  # fmt: skip
print("atoms matched to a differently named partner:", swapped)

# Two files can name equivalent atoms the other way round.  Swap the names of the
# carboxylate oxygens of both propionates (O1A <-> O2A, O1D <-> O2D) in the beta heme.
renamed = beta_on_alpha.copy()
pos = renamed.positions
for a, b in (("O1A", "O2A"), ("O1D", "O2D")):
    i = renamed.select(f"resname HEM and chain B and name {a}").ids
    j = renamed.select(f"resname HEM and chain B and name {b}").ids
    pos[i], pos[j] = pos[j].copy(), pos[i].copy()
renamed.positions = pos
swap = boonza.symmetry_rmsd(renamed, hb, atoms="resname HEM and chain B",
                            reference_atoms="resname HEM and chain A")  # fmt: skip
print(f"\noxygen names swapped: plain {swap.plain_rmsd:.3f} A, "
      f"symmetry-corrected {swap.rmsd:.3f} A (unchanged)")  # fmt: skip

# The answer does not depend on atom order: shuffle the beta heme's atoms.
shuffled = beta_on_alpha.copy()
heme_b = shuffled.select("resname HEM and chain B").ids
order = np.arange(shuffled.natoms)
order[heme_b] = np.random.default_rng(0).permutation(heme_b)
shuffled.reorder_atoms(order)
again = boonza.symmetry_rmsd(shuffled, hb, atoms="resname HEM and chain B",
                             reference_atoms="resname HEM and chain A")  # fmt: skip
print(f"after shuffling the atom order: symmetry-corrected {again.rmsd:.3f} A "
      f"(plain RMSD no longer meaningful: {again.plain_rmsd})")  # fmt: skip

# Shape only (conformer comparison): superpose the two hemes themselves.
shape = boonza.symmetry_rmsd(beta_on_alpha, hb, atoms="resname HEM and chain B",
                             reference_atoms="resname HEM and chain A", superpose=True)  # fmt: skip
print(f"heme shape difference (fitted): {shape.rmsd:.3f} A over {shape.isomorphisms} "
      "equivalent atom mappings")  # fmt: skip


def random_rotation(rng):
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# The docking workflow in one call.  Here the "docked" system is the crystal
# structure moved somewhere else, so the ligand RMSD should be zero.
rng = np.random.default_rng(1)
docked = hb.copy()
docked.positions = docked.positions @ random_rotation(rng).T + [40.0, -10.0, 5.0]
res = boonza.ligand_rmsd(docked, hb, ligand="resname HEM and chain A")
print(f"\nligand_rmsd: protein fit {res.fit_rmsd:.2e} A, heme RMSD {res.rmsd:.2e} A")

# Trajectories: pass frames (an array, a Frames block, or an open trajectory,
# e.g. boonza.open_trajectory("md.xtc", system)).  Every frame is fitted on the
# protein on its own, the heme moves with it, and the RMSDs come back per frame.
frames = np.stack([hb.positions @ random_rotation(rng).T + rng.normal(0, 10, 3)
                   + rng.normal(0, 0.1 * k, hb.positions.shape) for k in range(6)])  # fmt: skip
per = boonza.ligand_rmsd(hb, hb, ligand="resname HEM and chain A", positions=frames)
print("frames of a tumbling, increasingly jiggled hemoglobin:")
print("  protein fit RMSD (A):", np.round(per.fit_rmsd, 2))
print("  heme RMSD (A):       ", np.round(per.rmsd, 2))

# dRMSD: the RMS change of every pocket-atom-to-ligand-atom distance, no fitting.
# The pocket is the C-alpha atoms within 5 A of the ligand in the reference (the
# first frame by default).  Equivalent ligand atoms are matched in each frame.
d = boonza.drmsd(hb, ligand="resname HEM and chain A", positions=frames)
print(f"\ndRMSD over {len(d.pocket)} pocket C-alphas and {len(d.reference_ligand)} heme atoms:")
print("  symmetry-corrected (A):", np.round(d.drmsd, 2))
print("  atoms in order (A):    ", np.round(d.plain_drmsd, 2))
```

Output:

```text
beta onto alpha: 119 C-alpha pairs, RMSD 0.95 A
heme RMSD, atoms paired by order:  0.938 A
heme RMSD, symmetry-corrected:     0.938 A
atoms matched to a differently named partner: []

oxygen names swapped: plain 1.137 A, symmetry-corrected 0.938 A (unchanged)
after shuffling the atom order: symmetry-corrected 0.938 A (plain RMSD no longer meaningful: None)
heme shape difference (fitted): 0.697 A over 4 equivalent atom mappings

ligand_rmsd: protein fit 4.55e-14 A, heme RMSD 4.35e-14 A
frames of a tumbling, increasingly jiggled hemoglobin:
  protein fit RMSD (A): [0.   0.17 0.34 0.49 0.71 0.86]
  heme RMSD (A):        [0.   0.15 0.32 0.53 0.64 0.96]

dRMSD over 7 pocket C-alphas and 43 heme atoms:
  symmetry-corrected (A): [0.   0.13 0.27 0.42 0.53 0.86]
  atoms in order (A):     [0.   0.13 0.27 0.42 0.53 0.86]
```
