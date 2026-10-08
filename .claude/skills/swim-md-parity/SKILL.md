---
name: swim-md-parity
description: Keep `boonza swim` and `boonza md` in step. Use this whenever changing, fixing or adding anything to either command -- a new flag or setting, a build or run behaviour, a force-field or model change, a bug fix in src/boonza/md/cgswim.py, prepare.py, run.py or config.py. Both commands build and run the same kinds of coarse-grained and all-atom system through two separate call sites, so a change made in one is usually needed in the other, and has repeatedly been forgotten.
---

# Keep swim and md in step

`boonza swim` and `boonza md` share a settings parser and almost nothing else. They
build the same kinds of system through **two separate call sites**:

| | builds | runs |
|---|---|---|
| `boonza swim` | `src/boonza/md/cgswim.py` (`prepare`) | writes `md.toml`, then `boonza md` runs it |
| `boonza md` | `src/boonza/md/prepare.py` (`build_martini_system`, the SIRAH and all-atom paths) | `src/boonza/md/run.py` |

So a change in one is a change half-made. **Every time you touch either command, check
the other before you call it done.**

## What this has already cost

* Martini's elastic parameters (`elastic_fc`, `elastic_upper`, `elastic_lower`) were
  reachable from `boonza martinize` but from neither `swim` nor `md`, so every Martini
  run was silently pinned at 700 kJ/mol/nm² over 0-0.9 nm.
* `boonza swim --model martini2` worked; `boonza md --model martini2` refused to
  coarse-grain at all, because `prepare.py` never passed `forcefield=` to `martinize`
  and a guard had been written around the symptom.
* SIRAH's elastic network was given a selection, a stiffness and a reach, but its lower
  bound stayed hard-coded inside `add_elastic_network`, reachable from nowhere.

Each was found by a user hitting it, not by a test.

## Checklist

1. **Find the twin.** Grep both `cgswim.py` and `prepare.py` for the function you are
   changing -- `martinize(`, `sirahize(`, `solvate(`, `build_sirah(` all appear in both,
   with different arguments. Compare the two argument lists line by line; a difference is
   a bug until you can say why it is not.
2. **Arguments are defined once and consumed twice.** `config.py`'s `build_parser` serves
   both commands, so a new flag appears in `boonza md --help` and `boonza swim --help` for
   free -- which proves nothing. Follow the setting to where it is *read*, and confirm
   both paths read it. A flag that parses and is never used is worse than no flag.
3. **When you add an argument to one, ask whether it belongs to the other.** Usually it
   does: both commands coarse-grain, solvate, restrain and run. If it genuinely belongs to
   only one -- probe concentration is a swim's, a membrane's leaflets are a build's -- say
   so in the help text, and make the other command refuse it rather than ignore it (see
   `ALL_ATOM_ONLY`, `MARTINI_ONLY`, `SIRAH_ONLY`, `CG_ONLY` in `config.py`).
4. **Settings `swim` writes into `md.toml` are build-time or run-time, not both.** A
   build-time setting (`elastic`, `elastic_selection`, `cg_selection`) is answered while
   the swim builds and then popped from `md.toml`; a run-time one (`elastic_network_*`,
   `repulsion_*`, restraints) is carried through and applied by `run.py`. Putting one in
   the wrong class is silent: the run ignores it.
5. **Test both paths.** `tests/test_cgswim.py` drives both -- `boonza.md.swim.main` for the
   swim and `boonza.md.prepare.build_martini_system` for the md path. A test that exercises
   only one has covered half the change. Where a behaviour is observable in the built
   system (bead types, bond force constants, charges), assert on the **built topology**,
   not on the settings: settings can be right while nothing reads them.
6. **The models too.** A change to one coarse-grained model usually has to reach
   the other two -- that is the sibling skill, `model-parity`. Both axes apply to
   most changes: two commands times three models.
7. **Defaults.** Do not change an analysis default without being asked. If a model default
   changes, remember `md.toml` records what each run used, so finished runs stay
   reproducible -- say that when reporting.
