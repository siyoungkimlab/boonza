# Force fields

boonza stores a force field exactly as msys and DMS files do. That means any
DMS or MAE force field round-trips losslessly, and OpenMM, CHARMM, Amber and
GROMACS systems can be brought in through the [OpenMM bridge](bridges.md).

## The model

- A **term table** holds interactions of one kind with a fixed number of
  atoms (`stretch_harm`: 2, `angle_harm`: 3, `dihedral_trig`: 4, `nonbonded`:
  1, ...).
- Each **term** lists its atoms and points at one row of a **parameter table**.
  Many terms usually share a row.
- `nonbonded_info` names the van der Waals form (`vdw_12_6`) and combining
  rule (`arithmetic/geometric` or `geometric`).
- The nonbonded table's **override table** holds pair-specific parameters
  (NBFIX).
- **Auxiliary tables** hold extra data, such as CMAP grids.

```python
s.tables  # {name: TermTable}
t = s.table("stretch_harm")
len(t), t.natoms, t.category  # category: bond, constraint, virtual, nonbonded, exclusion, polar
t.atoms  # (nterms, 2) atom indices
t.values("fc"), t.values("r0")  # per-term parameter values
t.params.props  # ['r0', 'fc', ...]
term = t[0]
term["r0"], term.atoms, term.param
term["fc"] = 350.0  # copy-on-write: other terms sharing the row are untouched

t.find_with_any([10]), t.find_with_all([10, 11]), t.find_exact([10, 11])
s.nonbonded_info.vdw_rule
nb = s.table("nonbonded")
nb.overrides.items()
s.aux_tables  # e.g. {"cmap1": ParamTable}
```

## Building tables

```python
t = s.add_table_from_schema("stretch_harm")  # schema gives param columns
p = t.params.add_param(r0=1.53, fc=310.0)
t.add_term([0, 1], p)
t.add_terms(pairs, params=p)  # vectorized

nb = s.add_nonbonded_from_schema("vdw_12_6", "arithmetic/geometric")
types = nb.params.add_params(2, sigma=[3.4, 2.5], epsilon=[0.1, 0.03])
nb.add_terms(np.arange(s.natoms)[:, None], types[type_of_atom])
nb.overrides.set(types[0], types[1], sigma=3.0, epsilon=0.2)  # NBFIX
```

`boonza.TERM_SCHEMAS` and `boonza.NONBONDED_SCHEMAS` list every DMS table
schema. Functional forms follow the DMS specification. For example,
`stretch_harm` is fc (r − r0)², with no ½.

## Exclusions and 1-4 pairs

```python
boonza.update_exclusions(s)  # exclude pairs up to 3 bonds apart
boonza.update_exclusions(s, pair_scales=(0.5, 1 / 1.2))  # + Amber-style scaled 1-4 pairs
```

Virtual sites are excluded like their host atom. Regenerating the tables of
an Amber system reproduces msys's own prmtop conversion.

## Reporting parameters: `describe`

```python
report = s.describe("resname LIG and name C1 C2")  # or boonza.describe(s, sel)
print(report)
report.atoms  # per atom: charge, LJ type, sigma, epsilon
report.pairs  # per pair: combined sigma/epsilon, NBFIX, excluded, 1-4 scale factors
report.terms  # {table: rows of every bonded term touching the atoms}
report.to_dict()
```

`terms="all"` keeps only terms whose atoms are all selected. Pairs are listed
automatically for up to 30 atoms, and with `pairs=True` for larger
selections. Output for two atoms of a WW domain
(`s.describe("resid 5 and name CA C")`), shortened:

```
units: Å, degrees, kcal/mol, e.  nonbonded vdw_12_6 (4 epsilon ((sigma/r)^12 - (sigma/r)^6)), combining rule geometric

atoms
  index    label  element     mass  charge  param  sigma  epsilon  typekey  comment  ff
     60  SER5:CA        C  12.0112    0.14      1    3.5    0.066                     1
     61   SER5:C        C  12.0112     0.5      2   3.75    0.105                     1

pairs
   i   j  label_i  label_j    qq  excluded    sigma    epsilon  nbfix
  60  61  SER5:CA   SER5:C  0.07       yes  3.62284  0.0832466     no

angle_harm (16 terms)  E = fc (theta - theta0)^2
     atoms                    labels  param  theta0    fc  typekey  comment  ff  constrained
  45,59,60     VAL4:C SER5:N SER5:CA      4   121.9    50                     1            0
  59,60,61     SER5:N SER5:CA SER5:C      8   110.1    63                     1            0
  ...
```

Pair rows also give each pair's topological distance (`bonds`: the number
of bonds along the shortest path; None when not connected), its distance
`r`, and its bare pair energy at the current positions (`e_vdw`, `e_es`,
`energy`, in kcal/mol). This is the full Lennard-Jones and Coulomb energy
unless the pair is excluded, plus any scaled 1-4 term, with no cutoff or
periodic images. Summed over all the pairs of a molecule, it equals the
nonbonded energy OpenMM computes with NoCutoff (the tests check this).

```python
report = boonza.describe(s, "resname LIG", pairs=True)
sum(p["energy"] for p in report.pairs)
boonza.topological_distances(s, "resname LIG")  # bonds apart; -1 when not connected
frames = report.to_pandas()  # {"atoms", "pairs", "stretch_harm", ...} as DataFrames
```

## Energies

With OpenMM installed, `boonza.openmm_energies(s)` evaluates the force field
and returns kcal/mol per table plus the total. See [RDKit and OpenMM](bridges.md).
