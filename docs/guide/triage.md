# Triaging many runs of one system (`boonza triage`, `boonza modes`)

A thousand simulations of one protein and one ligand, started from a thousand
poses, ask a different question from one simulation. Not *how good is this run*
but **how many different things happened, and how many runs found each**.

Three kinds of run come out of such a set:

* the ligand leaves and the run stops (`boonza md --early-stop`). Nothing has to
  look at those: their `status.json` says `detached`, and `boonza triage` skips
  them unless asked not to.
* the ligand sits where it started.
* the ligand starts badly, leaves, and settles somewhere else. **This is the
  interesting kind and the easiest to lose**, because every measure taken
  against the *starting* pose scores it badly — it was right to move.

```bash
boonza triage md/*/ -o triage          # one row and one fingerprint per run
boonza modes triage -o review          # group them, and open one of each
pymol review/session.pml               # or: vmd -e review/session.vmd
```

Every run has to be the same system: the fingerprints are compared column by
column and the medoids are fitted on matching backbone atoms.

## What each run is measured by

`boonza triage` writes `summary.csv`, one row per run. The dRMSD columns are
symmetry-corrected, against three references:

| column | against |
|---|---|
| `from_start`, `drifted`, `from_start_max` | the first production frame; `drifted` is its second half |
| `from_end` | the last frame |
| `from_pose`, `population`, `n_poses` | the run's own most populated pose |
| `late_population`, `late_from_pose` | the same, over the second half alone |
| `moved` | how far the contacts travelled, first half's mean against the second's |
| `settled` | it left where it started **and** then held one pose |

`settled` is the third class above. `drifted` being large is not a complaint
about such a run — it is the point.

Each reference is asked for on its own. A ligand that has wandered off has no
pocket within reach of the last frame and dRMSD against *that* frame cannot be
defined, but the rest of the run still can be, and dropping the row over one
reference loses exactly the runs worth looking at.

## What the runs are grouped by

`analysis/fingerprint.npy` — written as `fingerprints.npz` — is a
**pharmacophore fingerprint**: for each of RDKit's six feature families, how
near that kind of group comes to each residue. One contact fingerprint per
family, over that family's atoms alone, laid end to end.

A plain contact fingerprint cannot tell a flipped ligand from the pose it came
from: the same residues are touched either way. A typed one can, because the
flip puts a donor where an acceptor was. Turning a haem over inside its own
pocket moves a plain fingerprint by a cosine distance of 0.07 and a typed one
by 0.49.

It is also what makes grouping a thousand runs affordable. A full pairwise
dRMSD is tens of minutes at 9 ms a pair; these vectors are a cosine apart, and
the whole matrix is under a second.

Coarse-grained beads are typed too — Martini's and SIRAH's own tables give the
same six families for amino-acid beads. An **inert cofactor** is the exception
and deliberately so: `--cofactors` builds one uncharged apolar bead per heavy
atom, so there is no chemistry left to read, and `triage` says so in `note` and
keeps every other number.

## Reading the modes

```
3 pose(s) -> 2 mode(s) at cosine 0.25

mode  runs  share  widest  medoid
   0     2    67%    0.20  run_041
   1     1    33%    0.00  run_118
```

**How many independent starting poses found a mode is a stronger signal than
any score on one run.** `widest` is the largest cosine distance inside a mode:
a large one says `--cutoff` has merged two things, and is how that cutoff —
a guess, not a measured constant — gets checked on your own set.
