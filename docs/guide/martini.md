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
cg = m.system()  # a boonza System with Martini parameters
topology, system, positions = boonza.to_openmm(cg, **OPENMM_OPTIONS)
```

```bash
boonza martinize protein.pdb cg --elastic --solvate   # the same, from the command line
```

The bead types' nonbonded parameters come from the Martini release file
`martini_v3.0.0.itp`, and the lipids and sterols from theirs. boonza carries
the Martini 3.0.0 release (cgmartini.nl, 2021-03-29) in
`boonza/data/martini/params`, so nothing needs downloading and `topol.top`
`#include`s them by a path GROMACS can resolve as written:

```python
from boonza.martini import parameters, NONBONDED, LIPIDS

parameters()  # every file carried
parameters(NONBONDED)  # martini_v3.0.0.itp
parameters(*LIPIDS)  # phospholipids and sterols
```

Give a path of your own wherever one is taken — `m.system("…/martini_v3.0.0.itp")`,
`--martini-itp`, `--lipid-itp` — to use another version, or a lipid boonza does
not carry. **Please cite** the parameters: PCT Souza et al., *Nature Methods*
18, 382-388 (2021), DOI 10.1038/s41592-021-01098-3.

## Martini 2

Martini 2 comes too — `martini_v2.2.itp`, `martini_v2.0_ions.itp` and the
2015-06 lipid collection `martini_v2.0_lipids_all_201506.itp` — for the lipids
and systems that were never ported to Martini 3. `bilayer` and `solvate` build
either version; `martini=2` picks the water and ions written beside the lipids,
and the parameter file the topology includes:

```python
from boonza.martini import LIPIDS_FOR, bilayer, parameters

itps = [str(p) for p in parameters(*LIPIDS_FOR[2])]
m = bilayer(itps, {"DPPC": 1}, size=60.0, martini=2, salt=0.15)
m.save("cg")  # topol.top, cg.gro: GROMACS can run them
```

```bash
boonza md --model martini2 --solvate membrane --upper "DPPC:1" --size-nm 6 \
    --barostat membrane --workdir run
boonza md --model martini2 --solvate membrane --upper "DPPC:1" --box-nm 6 \
    --barostat membrane           # built and run in one command
```

Martini 2 defines its water and ions upstream, so boonza includes them instead
of writing a `solvent.itp` whose moleculetypes would clash — Martini 3's
parameter file holds no moleculetype at all, so there boonza writes one. Such a
bilayer matches GROMACS 2024.6 term by term, to a potential of -67283.37
against -67283.4 kJ/mol, and a plain 128-DPPC topology read from disk to
-28760.11 against -28760.1.

`martinize` builds Martini 3 only, so a Martini 2 *protein* has to come from a
topology you already have (martinize2's, say), which `boonza md cg/topol.top
--model martini2` will run. The two versions must never be mixed in one system:
their bead types share names and mean different things, and boonza refuses a
Martini 3 protein in a Martini 2 membrane rather than building it.

**Please cite** Martini 2: SJ Marrink et al., *J. Phys. Chem. B* 111,
7812-7824 (2007), and L Monticelli et al., *J. Chem. Theory Comput.* 4,
819-834 (2008).

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

## Membranes

`boonza.martini.bilayer` builds a lipid bilayer in water, optionally around
proteins, as insane does:

```python
from boonza.martini import bilayer

lipids = ["martini_v3.0.0_phospholipids_v1.itp", "martini_v3.0_sterols_v1.0.itp"]
m = bilayer(lipids, {"POPC": 7, "CHOL": 3}, size=100.0)  # a 100 Å square membrane
m = bilayer(lipids, {"POPC": 7, "CHOL": 3}, {"POPC": 5, "POPS": 2, "CHOL": 3})  # asymmetric
protein = boonza.martinize(boonza.load("receptor_opm.pdb"), elastic=True)
m = bilayer(lipids, {"POPC": 7, "CHOL": 3}, protein=protein, protein_origin=True)
m.save("membrane")  # topol.top (including the lipid files), .itp files, cg.gro
```

From the command line the same membrane is built and run by `boonza md`
([MD](md.md#martini-runs)); there is no separate build command:

```bash
boonza md receptor_opm.pdb --model martini3 --solvate membrane \
    --upper POPC:7,CHOL:3 --opm --elastic --barostat membrane --workdir run
```

- **Lipids.** Any molecule type in the lipid files works: phospholipids,
  sterols, or your own. boonza builds a straight template from the
  topology. Bead levels are 3.3 Å apart down from the head, as insane stacks
  them, and chains zig-zag so that bonded beads stay a bond apart.
  Constraints are set to their lengths and virtual sites placed from their
  parents (cholesterol's rigid core, for instance).
- **Leaflets.** Each leaflet is a lattice at `area_per_lipid` (60 Å²),
  filled in the ratios given, with each lipid turned about the membrane
  normal the way that best clears its neighbours. Water fills `water` Å
  beyond the lipids on each side, then Na⁺ and Cl⁻ neutralize the system and
  add `salt`.
- **Proteins** come from `martinize`, with the membrane normal along z.
  With `protein_origin` (`--opm`), the protein's z = 0 is the midplane, as
  OPM orients structures; otherwise its centre is. Lipids within 4.5 Å of a
  protein bead are left out, and the box is tall enough for the whole
  protein.

Run membranes with semi-isotropic pressure: OpenMM's
`MonteCarloMembraneBarostat` with `XYIsotropic` and `ZFree`, as
`Pcoupltype = semiisotropic` in GROMACS.

## Probes

`boonza swim --model martini3` maps a protein's surface with dipeptide
probes, which need no parameterization of their own; see
[swim](swim.md#coarse-grained-probes---model-martini3).
`boonza.martini.probes` builds them:

```python
from boonza.martini.probes import probe, probe_sequences

probe_sequences()  # 105 dipeptides of the 14 probe residues
probe("EK")  # a martinized Glu-Lys probe, ends neutral, free to bend
```

Pharmacophore features of beads come from what each bead stands for, in
`boonza.martini.features`, which is what `boonza sites --features` uses on a
coarse-grained run.

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
