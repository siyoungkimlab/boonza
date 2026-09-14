# Selections

`s.select("...")` implements the msys/VMD atom selection language: the same
grammar, keywords, macros and evaluation order as msys. It is tested against
msys's own selection test suite.

```python
s.select("protein and name CA")
s.select("resname LIG or (water and within 3.5 of resname LIG)")
s.select("same residue as within 5 of resid 42")
s.select("pbwithin 6 of protein")  # periodic, uses the system cell
s.select("water and nearest 10 to protein")
s.select("x > 10 and charge < -0.5")
s.select('name "C.*"')  # double quotes: regular expression; single quotes are literal
s.select("smarts 'c1ccccc1'")  # needs RDKit
s.select("resid 10 to 20 and backbone")
```

`s.select(string, pos=xyz, cell=box)` evaluates geometric clauses on other
coordinates, for example a trajectory frame.

## Keywords

| Kind | Keywords |
|---|---|
| strings | `name`, `resname`, `chain`, `segid`/`segname`, `element`, `insertion` |
| integers | `index`, `resid`, `residue`, `fragment`/`fragid`, `ct`/`ctnumber`, `atomicnumber`, `numbonds`/`degree` |
| numbers | `x`, `y`, `z`, `vx`, `vy`, `vz`, `mass`, `charge` |
| structure | `all`, `none`, `protein`, `nucleic`, `polymer` (protein or nucleic; a boonza addition), `water`, `lipid`, `backbone`, `sidechain`, `hydrogen`, `noh` |
| functions | `smarts '...'`, `sequence '...'` (one-letter residue sequence), `paramtype TABLE TYPE` |

Values can be lists (`name CA CB`), ranges (`resid 10 to 20`) or
comparisons (`mass > 12`, `sqr(x) + sqr(y) < 100`). Arithmetic supports
`+ - * / % ^` and `abs`, `sqr`, `sqrt`.

## Geometry

| Clause | Meaning |
|---|---|
| `within D of SEL` | atoms within D Å of SEL, SEL included |
| `exwithin D of SEL` | the same, excluding SEL |
| `pbwithin D of SEL` | periodic `within`, using the cell |
| `nearest K to SEL` | the K atoms closest to SEL |
| `pbnearest K to SEL` | periodic `nearest` |
| `withinbonds N of SEL` | atoms at most N bonds from SEL |
| `same residue as SEL` | also `same chain`, `same fragment`, `same resid`, ... |

## Macros

`acidic acyclic aliphatic alpha amino aromatic at basic bonded buried carbon
cg charged cyclic heme hetero hydrophobic ion ions large legacy_ion lipids
medium neutral nitrogen polar purine pyrimidine small solvent sugar sulfur
surface`

## Evaluation order

As in msys, clauses narrow the atoms chosen so far, from left to right. So
`water and nearest 5 to protein` picks the five closest *waters*. `within`,
`nearest` and `same ... as` take everything to their right as their argument:
`within 3 of A and B` means `within 3 of (A and B)`. Use parentheses when
you mean something else.

Residue classes (`protein`, `water`, `backbone`, ...) follow msys's
`Analyze()`: residues are typed by their atoms and bonds, not only by name.
