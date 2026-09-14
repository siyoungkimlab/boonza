# Checking systems: validate, knots, diff

## `validate`

```python
for p in boonza.validate(s):  # basic checks
    print(p)  # "stretch: 1 bonds have no stretch or constraint term: 12-13"
problems = boonza.validate(s, strict=True)
p.check, p.message, p.atoms  # machine-readable
```

An empty list means the system passed.

**Basic checks.** msys `dms-validate`, plus force-field consistency:

| Check | Problem reported |
|---|---|
| `nonbonded` | a particle without nonbonded parameters (or tables but no nonbonded table) |
| `stretch` | a bond between real atoms with no stretch or constraint term |
| `charge` | a molecule whose net charge is not a whole number (virtual sites count with their host) |
| `knot` | a bond passing through a ring of at most `max_ring` (10) atoms, including across periodic boundaries |
| `box` | a cell without positive volume |
| `mass` | a real atom with zero mass |
| `virtual` | a virtual site listed in several virtual tables |
| `modified_interaction` | `interaction_grp` names disagreeing with the `modified_interaction` table |

**Strict checks** (`strict=True`) add simulation-readiness checks:

| Check | Problem reported |
|---|---|
| `constraints`, `constrained_hydrogens` | no constraint terms, or hydrogens left unconstrained |
| `cell` | a non-diagonal cell |
| `contacts` | unbonded atoms closer than 1 Å (periodic) |
| `masses` | atoms of one element with different masses |
| `exclusions` | a pair within three bonds that is not excluded |
| `extra_exclusions` | an excluded pair more than three bonds apart |
| `term_topology` | a bonded term whose atoms are not bonded as it requires (Urey-Bradley 1-3 stretches and Amber-style impropers in `dihedral_trig` are allowed) |
| `waters` | a residue holding more than one water molecule |

## Knots

```python
for ring, bond, idx in boonza.find_knots(s, max_cycle_size=10):
    print(f"bond {bond} passes through ring {ring}")
```

A port of msys `dms-find-knot`, with the same results. It finds bonds
threaded through rings, a common artifact of building membranes or
solvating. `ignore_excluded_knots=True` skips knots whose atoms are all
excluded from each other.

## Diff

```python
for d in boonza.diff(a, b):  # empty list when they match
    print(d)  # "terms: stretch_harm: 1 of 1805 terms differ (...)"
boonza.diff(a, b, atom_map=amap, positions=False, tables=["nonbonded"])
```

`diff` compares atoms, residues, chains, bonds, cell and every force-field
table. Terms are compared as sets keyed by their atoms in a canonical order,
as msys `dms-diff-ff` does:

- a term equals its reverse;
- constraint partners are unordered;
- impropers list their center atom first.

Term order, parameter sharing and annotation columns therefore don't count
as differences. `atom_map[i]` gives the atom of `b` matching atom `i` of `a`.
Use it when the files order atoms differently.
