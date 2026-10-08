---
name: model-parity
description: Keep Martini 2, Martini 3 and SIRAH in step. Use this whenever changing, fixing or adding anything to one coarse-grained model -- src/boonza/martini/build.py, src/boonza/sirah/build.py, probes.py, the cofactor or elastic-band code, or anything that reads a built CG system. The three models solve the same problems in separate code, so a bug in one is usually a bug in another, and a fix made in one has repeatedly been left out of the others.
---

# Keep the three models in step

Martini 2.2, Martini 3 and SIRAH are **three separate implementations of the same
job**: map a structure to beads, bond them, hold them, write a topology, read it
back. They share `boonza.cofactors`, `boonza.io.gromacs` and little else.

| | source | force fields |
|---|---|---|
| Martini | `src/boonza/martini/build.py` | `martini22`, `martini3001` (`FORCEFIELD_FOR`) |
| SIRAH | `src/boonza/sirah/build.py` | one |

Martini 2 is **not** Martini 3 with different numbers; `INSTEAD_OF` and the `.ff`
files differ in residue names, bead counts and bond types. Treat `martini22`,
`martini3001` and `sirah` as three, not two.

**Every time you fix or extend one, check the other two before you call it done.**

## What this has already cost

* A GROMACS topology has no column for an insertion code and renumbers each
  molecule's residues, so `60, 60A, 60B` came back as one residue with three
  backbone beads. **Both** builders had it, in their own `_name_chains`; a
  thrombin went from 296 residues to 319 when both were fixed.
* `for_viewing` left the rubber bands out in Martini and **nothing at all** out in
  SIRAH, whose docstring said there was nothing to leave out -- while a SIRAH
  benzamidine carried 184 bands and came out as a star.
* A cofactor that bonds to nothing is a molecule by itself, so the tethers had no
  protein beads to reach. SIRAH already folded it into the nearest molecule;
  **Martini never did**, and a benzamidine with 36 backbone beads inside 12 Å got
  zero bands and walked 8.8 Å out of its pocket in 0.5 ns.
* Checking the *other* direction on that same fix found the mirror bug: SIRAH
  asked `all(b.cofactor for b in m.beads)`, and it has beads for Na, Ca, Mg and
  Zn -- so one ion sharing a chain with a ligand cost that ligand all 75 of its
  bands. Martini was immune, because its ions are not `ff.blocks` entries.

Each was found by a user looking at a picture or a trajectory, not by a test.

## Checklist

1. **Name the other two.** Before finishing, say explicitly what the fix means for
   `martini22`, for `martini3001` and for `sirah`. "Martini only" is an answer, but
   it has to be an answer, not an omission.
2. **Look for the twin function.** The two builders use different names for the
   same job -- `_hold_cofactors` / `_attach_cofactors`, `_name_chains` in both,
   `restraint_bonds` / `for_viewing` in both, `_write_itp` / `itp`. Grep for the
   *behaviour*, not the name.
3. **Check both directions.** If one model lacks what the other has, port it. Then
   ask the opposite question: does the model you copied *from* have a bug the other
   avoided by accident? That question has paid twice.
4. **Classification is where they diverge.** The two models decide "is this a
   protein / an ion / a cofactor" by different tests -- Martini by whether the
   force field has a block, SIRAH by a per-bead `cofactor` flag and `ION_BEADS`.
   A fix phrased in one model's terms is usually wrong in the other's. Phrase it as
   the property you mean ("holds no protein"), then express that twice.
5. **Assert on the built system, in a parametrised test.** `@pytest.mark.parametrize
   ("model", ["martini22", "martini3001", "sirah"])` over `.system()` -- bond counts,
   band counts, residue keys, bead types -- not on the arguments passed. Three
   models that agree on a number are three models that agree. `tests/test_cgswim.py`
   and `tests/test_martinize.py` both have this shape already.
6. **Keep the real differences, and say which they are.** Not every asymmetry is a
   bug, and flattening one is its own mistake:
   * Martini 2.2 has no neutral ASP/GLU/LYS block and no second histidine; those
     residues keep the charged form (`INSTEAD_OF`, `NEUTRAL_INSTEAD`). Martini 3
     and SIRAH have them.
   * Rubber bands are GROMACS type 6 under `martini22` (a potential, no exclusion,
     invisible to a viewer) and type 1 under `martini3001` (a real bond, which a
     viewer draws). `elastic_network_bond_type` says which.
   * Martini has ion beads for Na, Cl and Ca; SIRAH also has K, Mg and Zn.
   * SIRAH puts beads on named hydrogens, so its path needs a protonated
     structure and Martini's does not.
7. **The commands too.** A model change usually has to reach `boonza swim` and
   `boonza md` alike -- that is the sibling skill, `swim-md-parity`. Both axes
   apply to most changes: three models times two commands.
