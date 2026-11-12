# Ranking docked poses by reversible work (`boonza ctmd`)

A docking score is good at telling a library's plausible ligands from its
implausible ones and poor at ranking the plausible ones against each other. A
converged binding free energy ranks them well and costs too much to ask of
hundreds. CTMD sits between the two: push the ligand off its pose with
well-tempered metadynamics, and score it by the reversible work the bias had to do.
Nothing has to converge. What is compared is how hard each pose was to break.

The method is Adury, Tiwary, Gu and Shekhar, *J. Chem. Inf. Model.* **2026**, 66,
11929, and the estimator it rests on is Tiwary and Parrinello, *J. Phys. Chem. B*
**2015**, 119, 736. Their code and data are at
<https://github.com/mrinal-shekhar/CTMD>.

## What c(t) is

A well-tempered run's bias converges to the free energy only up to an offset that
moves as the run goes on. The 2015 paper showed what that offset is:

    c(t) = 1/beta * log( int ds exp(beta*gamma/(gamma - 1) * V(s,t))
                       / int ds exp(beta/(gamma - 1) * V(s,t)) )

It is the reversible work the bias has done. A ligand that will not leave its pose
forces the bias to keep building, so c(t) climbs; one that drifts out on its own
needs almost no work, so it stays low. PLUMED prints this as `metad.rct`; OpenMM
has no equivalent, so `boonza.ctmd.reversible_work` computes it from the bias grid
`openmm.app.Metadynamics` already keeps.

## Running it

The input is a prepared, equilibrated complex — what `boonza md` leaves as
`equilibrated.dms`. One directory per pose, named after the pose:

```bash
boonza ctmd run equilibrated.dms -r docked.mae -w out/ZINC000130612584
boonza ctmd score out/* --hits hits.csv --fraction 0.3
```

`run` does the unbiased filter first: 5 ns with no bias, and a ligand that leaves
the site there is scored 0 and never biased, since there is nothing to measure. The
rest get `--replicas` biased trajectories, each writing a `colvar_K.dat` in PLUMED's
format (so the authors' analysis scripts read boonza's files, and `boonza ctmd
score` reads theirs).

`-r/--reference` is the structure the pose is measured against, and it should be the
**docked** pose rather than the end of the filter run. That is deliberate in the
published protocol: measuring from the docked pose makes drift during the unbiased
run count against the ligand.

## How a pose is scored

One replica's score is c(t) at the moment the ligand leaves, or at the end of the
run when it never does. A pose's score is **the lowest of its replicas'** — the
easiest way out is the one a ligand takes. Ligands are ranked by that score, highest
first, and a filtered ligand's 0 puts it last.

With more replicas than the three the protocol asks for, the score is bootstrapped
as the published analysis does it: draw three, take their lowest, average those.
The spread of the draws is the error. The paper's results converge by five
replicas.

```python
from boonza.ctmd import score_pose, rank, enrichment_factor, bedroc

poses = [score_pose(d) for d in dirs]
for p in rank(poses):
    print(p)
print(enrichment_factor(hits, [p.score for p in rank(poses)], 0.3))
```

## The coordinate: drmsd, not aligned RMSD

This is the one deliberate difference from the paper, and it matters.

The published protocol biases the RMSD of the ligand after superposing the protein,
which PLUMED expresses with per-atom weights in a reference PDB. OpenMM's
`RMSDForce` superposes and measures on the same atoms, so it cannot say "align on
the protein, measure the ligand". The pocket-ligand **drmsd** says the same thing
without a superposition: it compares pocket-ligand distances with the reference's,
so moving or turning the whole system changes nothing and only the ligand's motion
relative to its pocket is seen. That is `boonza.drmsd`, and the pocket is the one it
uses — alpha carbons within 5 Å of the ligand, backbone rather than every atom, so
that a side chain swinging about does not move the coordinate on its own.

Two consequences:

- **The published width and range are for their coordinate, not this one.** drmsd
  over N pocket-ligand pairs has a different scale from an aligned RMSD and grows
  faster as the ligand leaves, since every pair distance grows at once. `--sigma`,
  `--grid` and `--cutoff` are worth looking at on a system whose answer you know
  before trusting a ranking. The paper reports the method to be robust to these,
  which helps, but that was tested with their coordinate.
- **Keep the pair rule fixed across a library.** `--pocket`, `--pocket-cutoff` and
  the ligand selection all change the coordinate's scale, so a ranking is only
  meaningful if every ligand was measured the same way.

The symmetry-corrected drmsd that `boonza drmsd` reports by default is **not** used
here, and cannot be: its correction is a discrete minimum over atom mappings, which
is not differentiable and so cannot sit inside a biasing force. The coordinate uses
a fixed mapping. Symmetry correction stays an analysis tool.

## The published parameters

From `amber_MD_template/plumed_template.dat` and `analysis/default_params.py` of the
authors' repository, which are boonza's defaults too:

| | value | note |
|---|---|---|
| hill height | 1.75 kJ/mol | `--height` |
| hill width | 0.015 nm | `--sigma`; the paper's text says kJ/mol, which is a slip |
| pace | 1 ps | `--pace`, 500 steps at 2 fs |
| bias factor | 10 | `--bias-factor` |
| grid | 0 to 1.5 nm, 200 bins | `--grid`, `--bins` |
| escape | 0.6 nm for 200 ps | `--cutoff`, `--commit` |
| replicas | 3, converged by 5 | `--replicas` |
| per replica | 5 ns | `--ns` |
| filter | 5 ns unbiased | `--filter-ns` |
| temperature | 303 K | `--temperature`; their PLUMED file says 300 |

Two of their own files disagree about the escape cutoff: `default_params.py` says
0.6 nm held for 1000 frames (200 ps, the paper's figure), while the function
defaults in `common_analysis.py` say 0.3–0.35 nm held for 200 frames (40 ps). The
published numbers are the ones used here.

boonza runs a Langevin thermostat where the paper used stochastic velocity
rescaling, and a Monte Carlo barostat where it used Parrinello-Rahman.

## What it is for, and what it is not

The paper reports EF30 of about 1.7–2.5 across four targets — JAK2, AmpC, α2AR and
CB1R — against 1.0 for chance, beating docking scores and more consistent than
Boltz-2. It is a triage step for the few hundred candidates a virtual screen has
already narrowed to, not a primary screen: with 28–41 ligands per target the paper
could not measure EF1 or EF5 at all.

It also inherits the docking pose. That is what makes it immune to the memorization
the same paper demonstrates in Boltz-2 — which keeps enriching even when the active
site is mutated to all-alanine, all-phenylalanine or charge-flipped, because docking
cannot produce a pose in an abrogated site at all. It also means a wrong pose gives
a meaningless score.

Cost scales with the system. The paper's 41,000–75,000 atom systems took about
2.5 h per replica on an A100, so roughly 7.5 GPU-hours a ligand plus the filter. A
few hundred thousand atoms is several times that, which is a handful of ligands
rather than a library.

## Checking it against the published data

The authors' repository carries every COLVAR file behind the paper — four systems,
10 replicas a ligand, 28 ligands for AmpC — so the scoring half can be checked
without running anything:

```bash
boonza ctmd score CTMD/publication_data/AmpC/ZINC*/ \
    --hits CTMD/publication_data/AmpC/hit_data_mol2.spc
```

`boonza ctmd score` reads their `COLVAR_K` files as readily as its own, and `--hits`
reads their `hit_data_mol2.spc` as it stands. On their AmpC data that gives

```
4 binders of 26 ligands (15.4% by chance)
EF30 = 2.44  (1.0 is chance)
BEDROC(alpha=20) = 0.591
```

against the "enrichment factor ≈2.5 at 30%" the paper reports for AmpC, and a
BEDROC above the 0.3–0.5 it describes elsewhere, AmpC being the system it calls out
as the exception. Two of their 28 ligands wrote no data at all and are left out of
the ranking rather than scored.

### What that check does and does not show

It runs no simulation. Their COLVAR files already hold a finished c(t) series —
their last column is PLUMED's `metad.rct` — so the only code path it exercises is
reading those files, scoring each replica where the ligand left, taking the lowest,
ranking, and the enrichment metrics. That path is the published one, and this is
real evidence for it.

Three things it leaves untested:

- **`reversible_work` is not exercised at all**, because the c(t) it would compute
  is already in their file. Its only checks are the identities in the tests: a flat
  bias returns its own value, no bias returns 0, and the defining ratio comes out
  right. It has never been compared with PLUMED's `rct` on the same trajectory.
- **The coordinate** is theirs, not boonza's: an aligned RMSD rather than the drmsd.
- **The enrichment this implementation gives** is therefore unmeasured. Reproducing
  the paper's EF30 from their numbers is not the same as reproducing it from runs of
  our own.

Closing that gap means running the protocol: for AmpC, 26 ligands by a 5 ns filter
and three 5 ns biased replicas each, about 520 ns of MD, which on the paper's own
timings is some 200 GPU-hours. Until that is done, treat the biasing half as
implemented and smoke-tested, not validated. Running one system both ways and
comparing the rankings is the check that matters.
