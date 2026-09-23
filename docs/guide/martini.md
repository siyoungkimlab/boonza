# Martini 3 coarse-grained proteins

`boonza.martinize` turns an all-atom protein into Martini 3 beads and their
topology, as martinize2 does, and the result runs in OpenMM with the energies
GROMACS gives it. GROMACS isn't needed at any step.

```python
import boonza
from boonza.martini import OPENMM_OPTIONS, solvate

s = boonza.load("protein.pdb")
m = boonza.martinize(s, elastic=True)  # beads, topology, bead positions (Å)
m = solvate(m, salt=0.15)  # Martini water and NaCl around it
m.save("cg")  # topol.top, molecule_N.itp, solvent.itp, cg.gro
cg = m.system("martini_v3.0.0.itp")  # a boonza System with Martini parameters
topology, system, positions = boonza.to_openmm(cg, **OPENMM_OPTIONS)
```

```bash
boonza martinize protein.pdb cg --elastic --solvate   # the same, from the command line
```

The bead types' nonbonded parameters come from the Martini release file
`martini_v3.0.0.itp` (from cgmartini.nl). boonza doesn't ship that file, and
martinize2 doesn't either, so `m.system` asks for its path. The `topol.top`
that `save` writes only `#include`s it.

## What martinize does

This is martinize2's method, run on vermouth's own data files: its Martini 3
residue blocks, links, modifications and mappings (Apache-2.0, in
`boonza/data/martini`).

1. **Molecules.** Residues are joined by the structure's bonds between
   residues (CONECT and SSBOND records, for instance) and by vermouth's
   distance rule. Chains connected by a disulfide become one molecule, and a
   chain break splits a chain into two.
2. **Beads.** Each bead is the mass-weighted centre of the atoms its
   residue's mapping file lists. Hydrogens are identified by the atom they
   are bonded to, not by their names.
3. **Protonation and termini.** The hydrogens in the structure decide
   protonation, with the same modifications martinize2 uses:
   - Asp and Glu with a carboxyl hydrogen are neutral.
   - Lys with two amine hydrogens is neutral; with three, or none, it is charged.
   - His with hydrogens on both ring nitrogens is charged (`HIS-HP`), and
     with one on ND1 only it is the neutral δ tautomer (`HIS-HD`).

   Residue names such as HSD, HIE, ASH or LYN choose their own blocks.
   Termini are charged unless you pass `neutral_termini=True`. A terminus is
   a residue bonded to exactly one other residue.
4. **Secondary structure.** By default boonza's DSSP assigns it; pass `ss=`
   to set it yourself, one DSSP code per residue. The codes are converted to
   Martini's, where helices get start, end and short-helix codes. The
   backbone bonds, angles and dihedrals follow from them.
5. **Links.** Each link from the force field is applied wherever its pattern
   of beads matches, in file order: backbone terms by secondary structure,
   side-chain corrections (`scfix`), disulfides, and dihedrals for extended
   regions if you ask for them (`extdih`). Some parameters are measured from
   the structure, such as the phases of the side-chain dihedrals.
6. **Elastic network** (`elastic=True`). Backbone beads `elastic_lower` to
   `elastic_upper` Å apart get harmonic bonds, except those within
   `res_min_dist` residues of each other along the bonds (2 by default, as in
   martinize2). The force constant is `elastic_fc` kJ/mol/nm², optionally
   with martinize2's distance decay.

Options map to martinize2's flags. Distances are in Å here and in nm there:

| boonza | martinize2 |
|---|---|
| `ss="..."` / default | `-ss` / `-dssp` |
| `elastic`, `elastic_fc`, `elastic_lower`, `elastic_upper` | `-elastic`, `-ef`, `-el`, `-eu` |
| `elastic_decay`, `elastic_power`, `elastic_min_fc`, `res_min_dist` | `-ea`, `-ep`, `-em`, `-ermd` |
| `cys="auto"`, `"none"`, or a distance | `-cys` |
| `neutral_termini` | `-nt` |
| `scfix=False`, `extdih=True` | `-noscfix`, `-ed` |

## Where boonza differs from martinize2

These cases are rare, and each difference is deliberate:

- **A residue with missing side-chain atoms.** A bead with no atoms has no
  position, so boonza raises an error listing every such residue.
  martinize2 writes NaN coordinates. Rebuild the missing atoms first (with
  PDBFixer or Modeller, for instance).
- **A histidine with a hydrogen on ND1 only.** boonza makes it the neutral
  δ tautomer. martinize2 first adds the HE2 of its reference histidine,
  which makes every HD1-carrying histidine charged.
- **Hydrogen names.** Any names work in boonza, because it reads hydrogens
  by their bonds. martinize2 needs names its reference residues know, and
  fails on others (boonza's own peptides name them H1, H2, ...).
- **Proteins only.** `martinize` maps proteins; water and ions come from
  `solvate`, and other Martini molecules from their own topologies (see
  below). Leave everything else out of `atoms` (the default is
  `"protein"`); boonza names any residue it can't map.

## Water and ions

`solvate` puts the proteins at the centre of a box and tiles Martini 3 water
around them: W beads, each standing for four waters, from a box boonza
equilibrated at 300 K and 1 bar (density 987 kg/m³, as GROMACS gives it).

- **Box.** By default a cube of the proteins' largest extent plus `padding`
  (10 Å) on each side; `box=` sets the edges.
- **Clashes.** Water beads within 4.2 Å of a protein bead are removed. This
  is twice the 0.21 nm radius Martini users give `gmx solvate`. Where tiles
  meet the box's faces, one of each pair closer than 3.5 Å is removed.
- **Ions.** Na⁺ and Cl⁻ (Martini 3's `TQ5` beads) first cancel the proteins'
  charge, then ion pairs bring NaCl to `salt` mol/L. They replace water beads
  at least 5 Å from the proteins.

`save` writes the water and ion molecule types to `solvent.itp`, as
`martini_v3.0.0_solvents_v1.itp` and `martini_v3.0.0_ions_v1.itp` define
them, so only `martini_v3.0.0.itp` is needed. The box shrinks by about 10%
in volume in the first 100 ps of NPT, as the gaps left around the protein
close.

## Other Martini molecules

Molecules from Martini's own topology files load with `boonza.load` (or
`boonza.io.load_top`), the same way. That includes lipids, sterols and small
molecules, with their constraints and virtual sites. Martini 3's cholesterol
(`CHOL`, in `martini_v3.0_sterols_v1.0.itp`) builds five of its nine beads
as out-of-plane virtual sites on a triangle of constraints, and runs in
OpenMM as in GROMACS (see [Verification](../verification.md)).

## Running it

`to_openmm` needs Martini's nonbonded settings: a reaction field with
ε_r = 15 and ε_rf = ∞ (0), Lennard-Jones shifted to zero at the cutoff,
an 11 Å cutoff and no dispersion correction. `boonza.martini.OPENMM_OPTIONS`
holds all of these (see [RDKit and OpenMM](bridges.md#martini)). Martini
runs with a 20 fs time step, but a 20 fs step straight from an energy
minimum can blow up. OpenMM's minimizer stops higher than GROMACS's, and the
first large steps meet the strain it leaves. `equilibrate` minimizes, then
steps up through 2, 5 and 10 fs before 20:

```python
import openmm as mm
import openmm.unit as u
from openmm import app
from boonza.martini import equilibrate

system.addForce(mm.MonteCarloBarostat(1 * u.bar, 310 * u.kelvin))
integrator = mm.LangevinMiddleIntegrator(310 * u.kelvin, 1 / u.picosecond, 0.020 * u.picosecond)
simulation = app.Simulation(topology, system, integrator)
simulation.context.setPositions(positions)
equilibrate(simulation, temperature=310)  # minimize, then 2, 5, 10 and 20 fs
simulation.step(50_000)  # 1 ns
```

OpenMM holds Martini's constraints, the rigid rings of Trp, Tyr, Phe and
His and cholesterol's core included, to about 10⁻⁶ of their lengths at
20 fs. It solves coupled constraints with CCMA, and its default tolerance is
10⁻⁵.

Membranes and Gō models are not yet in boonza.
