# MM/GBSA binding energies (`boonza gbsa`)

A pose that holds for 50 ns has proved it is stable, not that it is right. A
ligand parked in a wrong but greasy groove holds as well as the native pose
does, and nothing measured on the trajectory alone — drift, contacts, how many
runs agree — asks whether the pose is *favourable*. `boonza gbsa` does: it
takes each frame's energy three times on the same coordinates, as the complex,
as the receptor alone and as the ligand alone, with the water replaced by a
Generalized Born model, and the difference is what binding is worth.

```
dG = E(complex) − E(receptor) − E(ligand)
   = dE_vdw + dE_elec + dG_GB + dG_SA
```

This is single-trajectory MM/GBSA (Kollman et al. 2000; Genheden & Ryde 2015).
The bonded terms cancel exactly, since the pieces have the coordinates the
complex has, and what is left is the interaction energy screened by the
solvent model. There is no entropy term: normal modes cost more than the
dynamics did, and for ranking poses of one ligand in one protein — the same
atoms in every pose, so the same systematic errors too — they are left out as
a rule.

```bash
boonza gbsa md/runs/*/ -o gbsa --ligandsel "chain L"     # one row per run, every frame
boonza gbsa md/runs/*/ -o gbsa --ligandsel "chain L" --stride 5 --salt 0.15
boonza gbsa pose.mae --ligandsel "resname LIG"           # one structure, as it is
boonza gbsa complex.dms --traj md.dcd --ligandsel "resname LIG"
```

**What the number is good for is ranking poses of one ligand against one
protein**, where everything but the placement is the same — the question
`boonza triage` leaves open. Across different ligands the usual MM/GBSA
caveats apply, and an absolute binding free energy it is not.

## What is scored

The frames are production as it was written, every `--stride`-th of them, each
made whole and the ligand imaged beside the receptor first (a trajectory is
wrapped, and a ligand can sit a box-length from its pocket). So what is scored
is the pose the dynamics relaxed into, not the one the run was started from: a
run that left a bad start and settled is scored where it settled, and the best
frame of a run can be better than any starting pose was.

A run directory is read as `boonza md` left it: `solvated.dms`,
`trajectory.dcd` and `status.json`. A run whose ligand left and stopped it
(`detached`) is listed with that note and not scored, unless `--keep-detached`.
A structure file is scored as one frame.

The solvent is dropped and everything else kept, as `boonza solute` does;
`--keep` says what counts as solute instead, to keep a structural ion. The
ligand is `--ligandsel` and the receptor is the rest, or `--receptorsel`. A
ligand bonded to the receptor is refused: a covalent complex cannot be split
into the two, and its bonded terms would not cancel.

## What is written

`summary.csv` has one row per run; `frames/<run>.csv` has one row per frame
scored. All energies are kcal/mol.

| column | |
|---|---|
| `dg` | mean over the frames |
| `sem`, `std` | its standard error (frames taken as independent, which a nanosecond apart they nearly are) and standard deviation |
| `best`, `best_frame` | the lowest frame, and which one |
| `late` | the mean of the second half, where a pose that settled late shows what it settled into |
| `vdw`, `elec` | Lennard-Jones and Coulomb between the ligand and the receptor |
| `gb` | the polar solvation: what screening the pair costs, usually positive and most of the way back from `elec` |
| `sa` | the nonpolar solvation: surface tension times the solvent-accessible area buried, favourable |
| `bonded` | the terms inside each molecule, complex minus pieces: zero to the precision of the arithmetic, and kept as the check that it is |

Rank by `dg`, or by `late` for a set where runs settle late, or by `best` when
one well-placed frame is what will be opened next.

## The model

The GB model is OpenMM's implementation of Amber's, by Amber's names:
`--model GBn2` (igb=8, the default and Amber's recommendation), `OBC2` (igb=5),
`OBC1`, `HCT` and `GBn`. The radii are the ones Amber assigns — mbondi3 for
GBn2, mbondi2 for the rest — read off the elements and bonds through OpenMM's
own tables, so a GAFF2 ligand is treated exactly as the protein is, by what
its atoms are bonded to and not by residue name. `--salt` adds Debye screening
for that much salt (mol/L), with Amber's conversion and its 0.73 for the ions
the solute excludes.

The nonpolar term follows Amber's MM/PBSA convention rather than OpenMM's
analytical ACE term: 0.0072 kcal/mol/Å² (`surften`, no offset) times the
solvent-accessible area the complex buries, on a Shrake–Rupley surface with a
1.4 Å probe. Hydrogens are part of that surface, as they are in mdtraj; the
area buried differs a little from LCPO's, and the term is small either way.

The full potential goes into every piece, constrained bonds as the flexible
terms they stand for, so the bonded energy of each piece is the molecule's own
and the cancellation is a real check rather than an arithmetic identity. A
position restraint the run carried (`posre_harm`) is the run's and not the
molecule's, and is dropped.

Each frame costs three single-point energies on the solute: about half a
second on OpenMM's CPU platform for a 150-residue protein, or six seconds on
the Reference platform (double precision, one thread), which is the fallback
where there is no CPU platform; `--platform` picks one. A 50-frame run is
under half a minute on the former and five minutes on the latter, and a
thousand runs are a job array either way. CUDA works too, but an OpenMM
context is made for each of the three pieces, and a GPU held in
`exclusive_process` mode allows a process only one.

## From Python

```python
import boonza

s = boonza.load("md/runs/run_041/solvated.dms")
traj = boonza.open_trajectory("md/runs/run_041/trajectory.dcd", s)
out = boonza.gbsa(s, traj, ligand="chain L", model="GBn2", salt=0.15, stride=5)
out.total  # dG per frame scored
out.terms["gb"]  # one array per term; out.terms["bonded"] is the check
out.summary()  # dg, sem, best, best_frame, late, and each term's mean
out.pieces["ligand"]  # the energies the differences were taken from
```

`boonza.gb_force(system, model="GBn2", salt=0.15)` builds the GB force on its
own, to add to any OpenMM System `boonza.to_openmm` made of the same system.
