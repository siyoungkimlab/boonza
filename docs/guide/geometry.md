# Geometry and periodic boundaries

## `boonza.pbc`

```python
from boonza import pbc

d = pbc.distances(a, b, box)  # (n, m) matrix, minimum image
d = pbc.self_distances(a, box)  # condensed i < j (scipy pdist order)
d = pbc.paired_distances(a, b, box)  # a[k]-b[k], e.g. bond lengths
i, j, d = pbc.capped_distances(a, b, 5.0, box)  # pairs within a cutoff (cell list)
v = pbc.minimum_image(vectors, box)
theta = pbc.angles(a, b, c, box)  # radians
phi = pbc.dihedrals(a, b, c, d, box)  # radians, IUPAC sign
```

Boxes are 3×3 arrays of cell vectors. `(a, b, c, alpha, beta, gamma)`
tuples are accepted too, and `None` or zeros mean no periodicity.

- **Orthorhombic cells** use the direct minimum-image rule.
- **Triclinic cells** are reduced in fractional coordinates, then the
  neighboring images are checked. That gives the true minimum image for any
  cell shape, not an approximation.

Kernels are numba and run in parallel. For example, a 5000 × 5000 triclinic
distance matrix takes 0.12 s, and all pairs within 6 Å in a 26k-atom box
take 0.05 s.

## Making molecules whole, wrapping, fitting: `Glue`

`Glue` fixes periodic frames. It is a triclinic-capable port of msys pfx and
viswizard's glue, and gives the same results as msys's `Wrapper`.

```python
fix = boonza.Glue(s, glue="protein or resname LIG", center="protein", fit="protein and name CA")
pos, box = fix(frame.positions, frame.box)  # one frame
for block in fix.frames(traj):  # a whole trajectory, chunk by chunk
    ...

whole = boonza.make_whole(s)  # just make every molecule whole
```

Each frame goes through four steps:

1. Molecules are made whole along their bonds.
2. The molecules in each `glue` group move to the periodic images that bring
   them jointly closest.
3. Everything else is wrapped into the box around `center`, or into the
   primary cell.
4. The `fit` atoms are superposed onto `reference`, the first frame by
   default. The box rotates with them, and `fix.rmsd` holds the fit's RMSD.

## Neighbor search

Selections such as `within` and `nearest` run on float32 cell lists, so
atoms right at a cutoff are classified the same way msys classifies them.
`boonza.spatial.pairs_within(pos, r, cell)` returns all pairs within `r`.
