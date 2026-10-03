# Martini coarse-grained proteins

`boonza.martinize` turns an all-atom protein into Martini beads and their
topology — Martini 3 by default, Martini 2.2 with `forcefield="martini22"` — as
martinize2 does, and the result runs in OpenMM with the energies GROMACS gives
it. GROMACS isn't needed at any step.

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

`martinize` builds either version — `boonza martinize in.pdb cg --model
martini2`, or `martinize(s, forcefield="martini22")` — from vermouth's files for
that version, and `--model martini2` on `boonza md` or `boonza swim` maps a
protein the same way before running it. A topology you already have
(martinize2's, say) still runs as it did: `boonza md cg/topol.top --model
martini2`.

Martini 2.2 is not Martini 3 with other numbers. It leans on secondary
structure, where Martini 3 does not: its backbone bead changes type and its
terms change with the fold, so the DSSP string is part of the topology. Its
residues are CHARMM's — `HSD`, `HSE`, `HSP`, and no `HIS` — and boonza renames
them, because martinize2 given a residue called `HIS` builds the `HSD` block but
leaves the name, which `martini22`'s `protein_resnames` macro does not list, so
every link skips that residue and the backbone comes out severed there. It also
cuts the residues differently: a tryptophan's two rings are split along another
line and a phenylalanine's is read in another order, so each version has its own
mappings (`boonza/data/martini/mappings/<version>`). Nothing holds a bead placed
by the wrong version's rules in a ring its constraints can satisfy, and a run
started from one dies with a NaN a few steps in. Martini 2 has no mapping for a
protonated aspartate or glutamate or a neutral lysine, and boonza says so and
leaves the residue charged.

The two versions must never be mixed in one system: their bead types share names
and mean different things, and boonza refuses a Martini 3 protein in a Martini 2
membrane rather than building it.

**Please cite** Martini 2: SJ Marrink et al., *J. Phys. Chem. B* 111,
7812-7824 (2007), and L Monticelli et al., *J. Chem. Theory Comput.* 4,
819-834 (2008).

## Cofactors it has no residue for

A structural zinc, an ADP, a heme: Martini has no block for any of them, so
`martinize` refuses the structure rather than quietly dropping them.

```bash
boonza martinize protein.pdb cg --cofactors        # hold them as inert beads
```

`--cofactors` (or `cofactors=True`) maps each one as **one uncharged apolar bead
per heavy atom** — an alanine's side chain in Martini 3, the apolar bead Martini
2 builds one from, since Martini 2's alanine is a single backbone bead with no
side chain at all. Each bead carries its atom's own mass, every pair inside the
cofactor is excluded, its shape is held by bands between the beads, and each bead
is tethered to the backbone beads within `cofactor_reach` of it. Any coordination
the structure shows — anything within 2.6 Å, which a metal is well inside and a
passing contact is not — is bonded too, and that bond is made before the
molecules are split, so a zinc holding two loops together becomes a crosslink
rather than landing in a moleculetype of its own where nothing could bond it.

**The bands are dense, and that was measured rather than reasoned.** Three points
fix a rigid body, so a few bands ought to do — and they do not, because the bands
are soft. On a heme over 5 ps, banded to two neighbours each and tethered twice,
it strays 12 Å; and its own shape drifts just as far, which is the tell. A body
of *n* beads wants about 3*n*−6 independent bands to be rigid — 123 for a heme's
43 — where two per bead gives 56, many of them redundant around a ring. Tethers
cannot hold a shape that will not hold itself.

How far the tethers reach is then the lever:

| `cofactor_reach` | bands | the heme strays | its shape drifts | backbone rmsf near it |
|---|---|---|---|---|
| 9 Å | 688 | 2.27 Å | 2.12 Å | 0.30 Å |
| **12 Å** (default) | 1718 | **1.20 Å** | **0.97 Å** | 0.24 Å |
| 15 Å | 3028 | 0.88 Å | 0.57 Å | 0.22 Å |
| 20 Å | 4751 | 0.59 Å | 0.47 Å | 0.20 Å |

Twelve is where the returns fall off: the cofactor is held inside a bead's own
radius and the protein around it is stiffened least. Stiffening it *is* the cost —
you are trading some local flexibility for a cofactor that stays where it was
put. And the bands have to stay soft: at 700 kJ/mol/nm² (`cofactor_fc`, the
elastic network's own) this is stable, while dense *and* stiff blew up on the
first step.

This is deliberately a statement about **volume and nothing else**: probes cannot
enter the room the cofactor takes, and no charge or chemistry is invented for it.
It also cannot distort the fold, which the bands and the elastic network hold.
Measured on a phospholipase with two structural calciums, through the ramp to
20 fs and 5 ps of production, the cofactor holds to 1.7–2.1 Å of where the
structure put it while the protein moves 1.1 Å rms — less than a bead's radius.

Under SIRAH the metals need none of this: it maps a zinc, a calcium and a
magnesium to ions of its own, with real parameters. What it does not give them is
any bonded term, an ion being one bead of its own moleculetype — so a structural
zinc came out as a bead that simply diffuses away, and a zinc finger had nothing
holding its loops together. boonza now bonds a *coordinated* ion to whatever holds
it, keeping SIRAH's own type and charge; an ion in solvent has nothing within
2.6 Å and stays the free ion it is.

Two things it will not do:

- **It will not treat a residue of the chain as a cofactor.** A D-amino acid or a
  modified residue has a backbone, and mapping it inert would throw away a side
  chain in the middle of a protein, so it is refused by name instead.
- **It warns when the cofactor is not buried.** Buried is the case this is honest
  for: nothing can reach the cofactor, so nothing reads its missing chemistry. One
  sitting in solvent can be reached, and a probe will settle on an apolar bead
  where a phosphate or a charge belongs. Fewer than eight protein heavy atoms
  within 5 Å is the test; a coordinated metal has twenty or more.

A cofactor with parameters of its own deserves them instead: include its `.itp`
and leave it out of the selection. boonza carries Martini 3's nucleobases and a
library of rings and heterocycles to build one from, and both versions have a
divalent cation (`CA`) for a calcium that really is a free ion in a site.

## What martinize does

This is martinize2's method, run on vermouth's own data files: the residue
blocks, links, modifications and mappings of the version asked for (Apache-2.0,
in `boonza/data/martini`).

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
   with martinize2's distance decay. A band is a bond of one molecule, so it
   never joins two of them: a peptide bound to a receptor keeps its own
   topology and can leave. `elastic_selection` narrows the network to some of
   the residues -- `"chain A"` holds the receptor and leaves that peptide free,
   which matters when the peptide is folded enough to pick up bands of its
   own and would otherwise be frozen in the shape it arrived in.

   Residues whose heavy atoms are too close to be unbonded are taken as
   bonded, as martinize2 does, which is how a peptide bond or a disulfide is
   found. Across two chains a disulfide is ordinary (insulin, an antibody),
   but anything else is almost always a clash, and it would quietly make the
   two one molecule under one network; boonza refuses that and names the
   atoms. A link that is real belongs in the structure's own connectivity,
   which is taken as given.

Options map to martinize2's flags. Distances are in Å here and in nm there:

| boonza | martinize2 |
|---|---|
| `ss="..."` / default | `-ss` / `-dssp` |
| `elastic`, `elastic_fc`, `elastic_lower`, `elastic_upper` | `-elastic`, `-ef`, `-el`, `-eu` |
| `elastic_selection` | none: martinize2 holds every molecule |
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
    --upper POPC:7,CHOL:3 --opm --barostat membrane --workdir run
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
[swim](swim.md#coarse-grained-probes---model-martini3---model-sirah).
`boonza.martini.probes` builds them:

```python
from boonza.martini.probes import probe, probe_sequences

probe_sequences()  # 105 dipeptides of the 14 probe residues
probe("EK")  # a martinized Glu-Lys probe, ends neutral, free to bend
```

`Martinized.for_viewing()` gives the same system without its elastic network,
which `boonza md` writes as `view.dms` and `view.mae` for every run, network
or not: the bands are bonds like any other, and a viewer draws every one of
them. Everything else is kept,
atom for atom and in order, so a trajectory still lines up.

The beads keep their names. `backbone_as_ca=True` renames the backbone bead
`CA`, which is what a viewer traces a chain through, but it is **off by
default** and was on once: a viewer that knows amino acids reads a residue of
`GLU: CA SC1` as a broken one and draws its own bonds over it, which is a worse
hairball than the rubber bands were (`solvated.dms`, with `BB`, opens fine in
the same viewer). Turn it on for a viewer that wants a CA trace and perceives
no bonds of its own, and remember the bead is not an alpha carbon: it stands for
the whole backbone, and consecutive ones sit about 0.35 nm apart where alpha
carbons sit 0.38 apart.

The view's cts are named `boonza view: no elastic network`, which is how the
file says what has been taken out of it -- a `.dms` and a `.mae` both keep a ct
name -- and how `boonza md` knows to refuse to run it.

`Martinized.system()` reads the topology back for its parameters but keeps
the bead positions it holds, so nothing is rounded to a `.gro`'s three
decimals, and it gives the residues the chains their beads came from, which
no GROMACS coordinate file has a column for. `boonza md` therefore keeps
`chain A` and `chain B` through a coarse-grained run, for selections and for
`--monitor-chain`.

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

Gō models are not yet in boonza.
