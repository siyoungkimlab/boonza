# Alignment: superposition and sequences

Which to use:

| Task | Function |
|---|---|
| superpose homologous proteins (any sequence identity above ~20%) | `boonza.matchmaker` |
| superpose remote homologs or structures with unrelated sequences | `boonza.cealign` |
| superpose two conformations of the same molecule | `boonza.superpose(..., match="order")` or `kabsch` |
| align sequences and get identity or similarity | `boonza.align_sequences` |
| ligand pose RMSD with equivalent atoms (docking, MD) | `boonza.ligand_rmsd`, `boonza.symmetry_rmsd` |

## `matchmaker`: ChimeraX matchmaker

```python
result = boonza.matchmaker(mobile, reference)  # moves `mobile`
result = boonza.matchmaker(mobile, reference, mobile_chain="B", reference_chain="A")
result.rmsd  # over pairs kept after pruning
result.full_rmsd  # over all aligned pairs
len(result.kept), len(result.mobile_atoms)
result.aligned_reference, result.aligned_mobile  # gapped sequences
result.rotation, result.translation, result.apply(xyz)
```

A port of UCSF ChimeraX matchmaker. It reproduces ChimeraX's alignment,
residue pairs, pruning, RMSDs and transform, and the tests run ChimeraX
itself to check this. Each step works as follows:

- **Sequence alignment:** Needleman-Wunsch with BLOSUM-62 mixed 70/30 with a
  secondary-structure score.
- **Secondary structure:** from `boonza.chimerax_ss`, a port of ChimeraX's
  own DSSP.
- **Gaps:** opening costs 18 inside helices and strands, 6 elsewhere and 12
  at chain ends; extension costs 1.
- **Pairing:** aligned residues pair by CA, or C4′ for nucleotides.
- **Pruning:** fitting repeats while discarding the worst pairs, at most 10%
  per round and at most half of those beyond 2 Å, until every remaining pair
  is within `cutoff`.
- **Chains:** without chain arguments, every chain pair is tried and the
  best-scoring one wins.

Myoglobin vs hemoglobin α (about 25% identical):

| Method | Pairs kept | RMSD |
|---|---|---|
| ChimeraX matchmaker | 105 / 135 | 1.052 Å |
| `boonza.matchmaker` | 105 / 135 | 1.052 Å |
| identity-only alignment (`superpose`) | 24 / 45 | 1.19 Å |

Options: `cutoff=None` disables pruning, `ss=False` aligns by sequence only,
and extra keywords (`gap_open`, `ss_fraction`, ...) go to
`boonza.needleman_wunsch`.

## `cealign`: structure-only alignment (PyMOL cealign)

```python
r = boonza.cealign(mobile, reference)  # moves `mobile`
r = boonza.cealign(mobile, reference, sel="chain A", ref_sel="chain B")
r.rmsd, r.z_score, len(r.mobile_atoms)
```

Combinatorial Extension (Shindyalov & Bourne 1998), ported from Biopython's
version of PyMOL's cealign, with the same results. It uses only CA (or C4′)
geometry, so it works when sequences have diverged too far for matchmaker.
`boonza.ce_align(ref_xyz, mobile_xyz)` runs it on raw coordinates.

## `superpose`, `kabsch`, `rmsd`

```python
R, t = boonza.kabsch(mobile_xyz, target_xyz, weights=None)  # mobile @ R.T + t ≈ target
boonza.rmsd(a, b, superpose=True)
res = boonza.superpose(mobile, reference, sel="protein and name CA", match="order")
```

`superpose` pairs atoms by order, or by a simple identity-based sequence
match, and prunes outliers.

## Sequence alignment and similarity

```python
aln = boonza.align_sequences("MKTAYIAKQRQISFVK", "MKTAHIAKQRQLSFVK")
aln = boonza.align_sequences(system_a, system_b, chain_a="A", chain_b="A")
aln = boonza.align_sequences(a, b, mode="local")  # Smith-Waterman
aln = boonza.align_sequences(a, b, end_gaps=False)  # free end gaps (EMBOSS needle)

aln.score
aln.identity  # identical pairs / alignment columns (EMBOSS "Identity")
aln.similarity  # positively scoring pairs / columns (EMBOSS "Similarity")
aln.gap_fraction, aln.identity_shorter
aln.aligned_a, aln.aligned_b, aln.pairs
print(aln)

boonza.sequence(system, "A")  # one-letter sequence of chain A
boonza.identity_matrix([seq1, seq2, seq3])  # pairwise identity (or metric="similarity")
```

Scoring is BLOSUM-62 by default, or any `{(x, y): score}` dictionary, with
affine gaps. A gap of length L costs `gap_open + (L − 1) × gap_extend`, with
defaults 10 and 0.5. This follows Biopython's PairwiseAligner and EMBOSS,
and the scores match Biopython. When several alignments tie for the best
score, boonza may report a different one than Biopython: the score is the
same, but identity can differ slightly. Printing an alignment gives
60-column blocks:

```
global alignment, score 70, length 16: identity 14/16 (87.5%), similarity 16/16 (100.0%), gaps 0/16 (0.0%)

MKTAYIAKQRQISFVK
||||:||||||:||||
MKTAHIAKQRQLSFVK
```

`|` marks identical residues, `:` a positive BLOSUM-62 score (Y/H and I/L
here), and `.` other aligned pairs.

## Symmetry-corrected ligand RMSD

A plain RMSD pairs atoms by name or order. Ligands with equivalent atoms
(carboxylate oxygens, flipped phenyl rings, t-butyl methyls) then get a large
RMSD for a correct pose just because atoms were named the other way round.
`symmetry_rmsd` takes the smallest RMSD over every mapping of reference atoms
onto mobile atoms that preserves elements and bonds (graph isomorphisms,
connectivity only by default). This is the quantity spyrmsd and RDKit's
`CalcRMS` report. The two molecules may list their atoms in different orders.

```python
# docking: fit the proteins (C-alpha pairs in order), move the ligand with them,
# then the symmetry-corrected in-place RMSD of the ligand heavy atoms
r = boonza.ligand_rmsd(docked, crystal)  # default selections
r = boonza.ligand_rmsd(docked, crystal, ligand="resname LIG", align="sequence")
r.rmsd, r.plain_rmsd, r.fit_rmsd

# any two copies of a molecule
r = boonza.symmetry_rmsd(pose, crystal, "resname LIG")  # in place
r = boonza.symmetry_rmsd(conf_a, conf_b, superpose=True)  # shape only
r.rmsd, r.plain_rmsd, r.mapping  # mapping: mobile atom per reference atom
per_frame = boonza.symmetry_rmsd(s, crystal, "resname LIG", positions=frames).rmsd

# a trajectory: every frame is fitted on the protein on its own, the ligand
# moves with it, and the values come back per frame (read chunk by chunk)
traj = boonza.open_trajectory("md.xtc", system)
r = boonza.ligand_rmsd(system, crystal, ligand="resname LIG", positions=traj)
r.rmsd, r.fit_rmsd, r.plain_rmsd  # arrays, one value per frame
```

### dRMSD: pocket-ligand distances, no fitting

`drmsd` compares the distances between pocket atoms and ligand atoms with
those in a reference: `sqrt(mean((d - d_ref)**2))` over the (pocket x
ligand) matrix. The pocket is the `protein` atoms (C-alpha by default)
within `cutoff` Å of the ligand in the reference. By default the reference is
the first frame, as in a "drift from the starting pose" analysis. Nothing is
superposed. Distances use each frame's periodic box.

As with `symmetry_rmsd`, ligand atoms are matched in every frame by the
element- and bond-preserving mapping that gives the smallest dRMSD, so
swapped carboxylate oxygens or a flipped ring are not counted as motion. The
cost splits into a sum over ligand atoms, so the same exact branch-and-bound
search applies. `plain_drmsd` pairs atoms in listed order and reproduces the
usual MDAnalysis script (`distance_array` of pocket vs ligand against frame 0).

```python
traj = boonza.open_trajectory("md.xtc", system)
d = boonza.drmsd(system, ligand="resname LIG", positions=traj)  # vs frame 0
d = boonza.drmsd(system, crystal, ligand="resname LIG", positions=traj, cutoff=6.0)
d.drmsd, d.plain_drmsd, d.pocket, d.reference_distances
```

```bash
boonza drmsd system.pdb --traj md.xtc --ligandsel "resname LIG" [--reference crystal.pdb]
```

With a separate reference, protein atoms are paired in selection order
(`reference_protein` if the selections differ), so the two proteins need the
same atoms. Pocket atoms are not symmetry-matched; choose atoms without
equivalents (C-alpha, backbone) when using a finer selection.

The defaults follow a typical docking evaluation:

| Option | Default |
|---|---|
| protein atoms fitted | `"protein and name CA and not resname NMA NME ACE"` |
| ligand | `"not (polymer or water or ions) and noh"` |
| hydrogens | ignored |

`align="sequence"` fits different proteins with `matchmaker`, and
`align=None` skips the fit when the structures already share a frame.

The best mapping is found by an exact branch-and-bound search: mappings grow
along the bonds, and any partial mapping that cannot beat the best RMSD so far
is dropped. Molecules with many equivalent groups therefore stay fast.
Enumerating every isomorphism, as a networkx loop does, grows factorially
with symmetry. The results match that enumeration and RDKit's `CalcRMS` and
`GetBestRMS`.

From the command line:

```bash
boonza rmsd docked.pdb crystal.pdb --ligandsel "resname LIG" [--align sequence]
boonza rmsd system.pdb crystal.pdb --traj md.xtc --ligandsel "resname LIG"   # per frame
```

