# Ligands swimming around a protein (`boonza swim`)

`boonza swim` sets up many simulations in which copies of ligands swim in the
water around a protein, from a ligand library too large for one simulation:

```bash
boonza swim protein.pdb --ligands library.sdf --types 5 --copies 3 \
    --workdir swim --dihedral-restraint ss --repulsion --production-ns 100
```

With N ligands and `--types` k, there are N // k simulations of k ligand
types each; the ligands left over join the last simulations, one each (500
ligands and k = 5: 100 simulations of 5; 501: the last has 6; 502: the last
two have 6). Each simulation holds the protein and `--copies` copies of each
of its ligand types.

Every simulation is a [`boonza md`](md.md) run, so all its options apply to
all of them (force fields, HMR, lengths, intervals, platform, ...), and each
one resumes like any `boonza md` run. The defaults are `boonza md`'s too,
so nothing is restrained and nothing repels unless you say so. What swim adds
is the right selection when you do ask: `--dihedral-restraint ss` holds the
protein's helices and sheets but sets
`dihedral_restraint_selection = "not chain LIG"`, so peptide ligands swim
free, and `--repulsion` keeps the ligand copies apart (see below).

The ligands are chain `LIG`. A ligand of one residue takes its own name as
the residue name — an SDF record's title line, its `_Name`, so a fragment
library keeps its catalogue IDs (`Z359510198`) — and each copy has its own
residue number; a record without a name falls back to `LIG`, and peptide
ligands keep their residue names (ACE, ALA, ...). `sim_NNN/ligands.csv` says
which residue numbers are which ligand and copy.

Names of any length survive in `input.dms` and in MAE files, but a PDB
written later (`solvated.pdb`, `final.pdb`) cuts a residue name to four
characters and a chain name to one, so `Z359510198` shows there as `Z359`
and fragments sharing a prefix become indistinguishable. `ligands.csv` and
the DMS files remain authoritative.
`boonza swim` writes:

| File | |
|---|---|
| `sim_000/`, `sim_001/`, ... | `input.dms` (the protein and the placed ligands), `ligands.csv` (which residues are which ligand) and `md.toml` (the settings); the run itself goes to `md/` |
| `simulations.txt` | one `boonza md --config .../md.toml` command per simulation |
| `assignment.csv` | the ligands of each simulation, with each one's highest similarity to another in the same simulation |
| `ligands/L000/`, ... | each ligand as placed, and its force-field patch |

Run the simulations as a job array, one line of `simulations.txt` each
(`sed -n "${SLURM_ARRAY_TASK_ID}p" swim/simulations.txt | bash` with
`--array=1-100`), or all here, one after another, with `--run`.

## Which ligands share a simulation

Similar ligands are kept apart, and the split is deterministic (the same
library always gives the same simulations):

1. Morgan fingerprints (radius 2, 2048 bits) of the ligands are clustered
   (Butina, Tanimoto distance 0.65), and the ligands are dealt round-robin in
   cluster order, so the members of a cluster go to different simulations.
2. Ligands are then swapped between simulations while a swap lowers the total
   Tanimoto similarity within simulations (for libraries of up to 5000
   ligands). On four homologous series of four (alcohols, amines, acids,
   alkylbenzenes) into four simulations, each simulation gets one of each.
3. Ligands with the same bond graph (stereoisomers, repeated entries) never
   share a simulation: their templates could not be told apart.

## Force fields of the ligands

Each ligand is parameterized once, and simulations share the result:

- A ligand from a DMS file that has a force field keeps it: its parameters
  become a template with a type per atom. It must use 12-6 Lennard-Jones with
  arithmetic/geometric combining and the 1-4 scaling of the protein force
  field (checked); CMAP, Urey-Bradley terms and virtual sites are refused.
- Otherwise the ligand is first matched against the protein force field. A
  peptide made of standard residues (with ACE/NME caps or charged termini)
  takes it; from an SDF file, which has no residues, the molecule is cut at
  its peptide bonds and taken as a peptide only if every piece is a standard
  residue, so drug-like molecules with amides stay whole.
- What the force fields cannot match gets GAFF2 with AM1-BCC charges
  ([Ligands](ligands.md)), several ligands at once with `--jobs`. Each
  ligand's GAFF2 types and templates carry its code (`c3~L017`), so ligand
  patches never collide.

Results are kept in `ligands/`; running `boonza swim` again reuses them and
leaves simulations that have started alone.

## Keeping the ligands apart

Ligand copies at these concentrations tend to stick together. `--repulsion`
keeps the ligands of each simulation apart: they get their own chain, `LIG`
(`LIG2`, ... if the protein uses `LIG`; PDB files, with one-letter chains,
shorten it), and each simulation's settings then say
`repulsion_selection = "chain LIG"`, a `boonza md` setting. Heavy atoms of
different ligand molecules (copies of one ligand included) then feel a
flat-bottom wall, E = k (d0 - r)^2 for r < d0, with d0 =
`repulsion_distance_nm` (0.5 nm; touching carbons sit at about 0.37 nm) and
k = `repulsion_kJ` (500 kJ/mol/nm^2, about 8 kJ/mol per heavy-atom pair at
contact). Nothing changes within a molecule or between ligands and the
protein or water. It is a force of the OpenMM system, so restarts keep it.
Without `--repulsion` nothing repels, as in `boonza md`.

## Placing the ligands

The box is a cube of the protein's largest extent plus twice `padding_nm`.
Each copy gets a random orientation and a random position in the box, at
least `--clearance` Å (3) from the protein's and the other ligands' heavy
atoms. Placement is seeded by `seed` and the simulation's number, so it is
reproducible. The box is then filled with water and ions as `boonza md` does.
