# Ligands and covalent adducts (GAFF2)

Protein force fields have templates for the standard residues only. For
everything else (drug-like ligands, non-standard amino acids and residues
carrying a covalently bound ligand), boonza makes new templates with GAFF2
atom types and AM1-BCC charges, and adds them to the protein force field as a
viparr patch. The protein keeps the protein force field's parameters as far
as its chemistry allows.

It needs [AmberTools](https://ambermd.org/AmberTools.php) (antechamber, sqm,
parmchk2, tleap), which is installed with conda; OpenFF is not needed (it is an
option for free ligands, [below](#openff-instead-of-gaff2), as is MMFF94,
[which needs RDKit alone](#mmff94-instead-of-gaff2)):

```bash
conda install -c conda-forge ambertools
```

boonza finds it through `$AMBERHOME`, or the `antechamber` on your `PATH`
(`amberhome=` overrides both).

```python
import boonza

s = boonza.load("complex.dms")  # all hydrogens; bond orders and charges, see below
ffs = ["aa.amber.ff19SB", "water.tip3p", "ions.amber1jc.tip3p"]
patch = boonza.gaff2_patch(s, ffs)
ff = boonza.merge_forcefields("aa.amber.ff19SB", patch)
p = boonza.parameterize(s, [ff, "water.tip3p", "ions.amber1jc.tip3p"])

boonza.write_forcefield(patch, "my-ligands")  # keep it and reuse it
```

```bash
boonza parameterize complex.dms out.dms -f aa.amber.ff19SB -f water.tip3p --gaff2 \
    --save-patch my-ligands
boonza parameterize complex.dms out.dms -f aa.amber.ff19SB -m my-ligands -f water.tip3p
```

`boonza.find_unmatched(s, ffs)` lists the residue groups that would get
templates, without running anything.

## What happens

1. **Groups.** Residues that no force field in the list matches are
   collected, with the residues bonded to them by anything other than a
   peptide bond or a disulfide. So a Cys bound to an acrylamide warhead is
   parameterized together with the warhead, even though its graph alone
   matches CYX. Bonded residues form one group; a free ligand is a group of
   its own, and identical copies share one template.
2. **Caps.** Where a group is bonded to the protein through a peptide bond,
   the neighbouring residue's atoms across the bond (C, O and CA, or N, H and
   CA) are kept, and the CA is closed with hydrogens into an ACE or NME
   methyl group.
3. **GAFF2 and AM1-BCC.** antechamber types the capped group with GAFF2 and
   computes AM1-BCC charges with sqm; parmchk2 writes every parameter the
   group uses from GAFF 2.11 (`gaff="2.2"` takes AmberTools' newer
   `gaff2.dat`), and tleap builds the topology, as openmmforcefields does
   for its `gaff-2.11` generator. GAFF 2.11 is `gaff211.dat` in newer
   AmberTools and `gaff2.dat` in older ones; boonza tells them apart by the
   version on the file's first line.
4. **Protein atoms.** An amino acid (a residue in a chain, or one whose
   parent is named) keeps the protein force field's types and charges from
   its parent residue. The parent is, in order:
   - the one you give (`parents={"MSE": "MET"}`, `--parent MSE=MET`, or
     `parents = { MSE = "MET" }` in `boonza md`'s settings);
   - the file's (PDB `MODRES`, mmCIF `_pdbx_struct_mod_residue`), kept as
     the residue property `parent`;
   - the PDB chemical component dictionary's for common modified residues
     (`boonza.gaff.KNOWN_PARENTS`: MSE, SEP, TPO, PTR, CSO, HYP, MLY, ...);
   - else a guess: the amino acid whose template matches the most atoms.
     A guess must be unambiguous and cover the backbone and CB; otherwise
     boonza stops and asks for the parent rather than risk a wrong type.

   Among the parent's templates (termini, protonation states), the one
   matching the most atoms is used. The backbone is found by atom names or,
   without them, by structure (N-CA-C(=O) with its peptide bonds), and the
   templates are matched by bond graph, so a residue whose atoms are named
   C1, C2, ... works too. Each choice is logged, with how many heavy atoms
   kept protein types (and listed in `components.json` by `boonza md`, and
   in the patch's `parents`). An atom keeps the parent's type and charge
   when it and all its bonded neighbours match the parent atom's
   neighbours, and a hydrogen only with its heavy atom. So backbone and CB terms, the protein
   impropers and the residue's CMAP come from the protein force field,
   including the mirrored CMAP of a D residue.
5. **Everything else** takes tleap's values: each bond, angle, dihedral and
   improper that touches a GAFF2 atom becomes a parameter row for exactly
   those atom types. GAFF2 types get a suffix per group (`c3~1`), so two
   groups never share rows and a GAFF2 type never meets a protein row.
6. **Charges.** Each residue is brought to its formal charge by shifting its
   GAFF2 atoms evenly. A shift above 0.1 e per atom is reported with a
   `ViparrWarning`. For example, a Cys bound to a ligand keeps the Cys charges
   on all atoms but SG, so SG alone takes the correction.

The templates also record the element of the atom across each external bond
and, across bonds other than peptide bonds, the formula of that atom's
residue (`external_elements` and `external_residues` in the template file,
which viparr ignores). When two templates match a residue, the one with more
of these checks wins. So the Cys above takes its own template rather than
CYX, while real disulfides still take CYX, even when the ligand is bound
through a sulfur (a mixed disulfide).

## How far protein types reach

`protein_extent` (`--protein-extent`) chooses how much of an amino acid
carrying GAFF2 atoms keeps the protein force field:

- `"matched"` (the default): every atom that matches its parent residue
  with all its bonded neighbours, as described above. On a Lys acetylated at
  NZ, CG, CD and CE keep ff19SB types and only NZ is GAFF2.
- `"cb"`: only the backbone (N, H, CA, HA, C, O, and OXT or H1-H3 at the
  termini), CB and the hydrogens on CB, and only those that match. On the
  same Lys, CG, CD, CE and NZ are GAFF2. The residue's charge correction is
  then spread over more atoms.

## Pseudopeptides

A pseudopeptide is a ligand that is part peptide: a peptidomimetic, a capped
or cyclic peptide with unnatural pieces, drawn or read as one molecule and so
**one residue**. Nothing above applies to it, because there are no residues to
cut at and no peptide bonds to a protein, and by default it is GAFF2 from end
to end. `pseudopeptide` (`--pseudopeptide`) gives its peptide part the protein
force field instead:

1. **GAFF2 for the whole molecule**, as for any ligand, with AM1-BCC charges
   from one AM1 calculation at the reference conformer (no geometry
   optimization, whatever `charge_geometry` says). This takes about a second
   at 90 atoms and three at 150; the optimization is what takes hours on a
   molecule this size.
2. **Residue matches.** Every N-CA-C(=O) in the molecule is compared with the
   amino-acid templates of the protein force field itself. Where a whole
   template is there atom for atom, hydrogens included -- ALA, LYS, a
   protonation state such as ASH, a charged terminus such as NALA or CLYS,
   CYX on each side of a disulfide -- those atoms are that residue. N must be
   an amide nitrogen and C an amide carbon, as in a chain. An acetyl, N-methyl
   amide or primary amide on a matched residue is ACE, NME or NHE.
3. **Backbone matches.** Where the side chain is not a standard one, the
   backbone alone is matched, as far as each atom is bonded like the
   template's: on alanine's types when there is a side chain, glycine's when
   there is none, proline's when a ring closes on N. The side chain, CB
   included, is GAFF2. An atom that is not bonded like the template's stays
   GAFF2 too: the nitrogen of an amine or an N-alkylated residue, the carbon
   of an ester.
4. **Parameters.** Matched atoms take the protein's atom types, so every
   bond, angle, torsion and improper among them is the protein force field's,
   and each N-CA-C whose three atoms matched gets its CMAP (mirrored for a D
   residue). The atoms on either side of it may be GAFF2. A term that touches
   a GAFF2 atom is GAFF2's, as for a covalent adduct.
5. **Charges**, one of two ways:
   - `"protein"` copies the template charges onto the matched atoms. They no
     longer sum to the molecule's charge with the AM1-BCC charges of the
     rest, so the difference is spread evenly over the GAFF2 atoms. A shift
     above 0.1 e per atom is reported with a `ViparrWarning`: it means few
     GAFF2 atoms are carrying a large correction.
   - `"types"` keeps the AM1-BCC charges on every atom, under the protein's
     types and bonded terms. Nothing has to be spread.

A molecule in which everything matches, such as Ac-Ala-Lys-Pro-Gly-Asp-NHMe
given as one residue, comes out with exactly the energies ff19SB gives it
residue by residue.

```python
lig = boonza.load("ligand.sdf")
patch = boonza.gaff2_patch(lig, ["aa.amber.ff19SB"], pseudopeptide="protein", pdf="ligand.pdf")
for u in patch.units:  # what was found
    print(u["template"], u["match"], u["atoms"])
```

```bash
boonza parameterize ligand.sdf out.dms -f aa.amber.ff19SB --gaff2 \
    --pseudopeptide --pdf ligand.pdf
```

`--pdf` (`pdf=`) writes the ligand in 2D as a vector PDF with a transparent
page, the atoms that carry protein parameters on yellow -- look at it before
running anything. `boonza md` and `boonza fep prep` write it into their work
directory as `pseudopeptide_<residue>.pdf`, and `boonza swim` into each
ligand's directory.

`pseudopeptide` is off unless asked for, in `boonza md`, `boonza swim` and
`boonza fep` alike (`--pseudopeptide`, or `pseudopeptide = "protein"` in the
settings file), so ligands that happen to contain an amino acid keep the
parameters they had. A swim's kept ligand templates are made again when the
setting changes. In a relative free-energy calculation every ligand of the
series is typed the same way, and atoms that keep the protein's types in two
ligands have identical parameters in both. The hybrid carries their CMAP
terms as it carries torsions: one among mapped atoms moves from the first
ligand's map to the second's (and does not move when the two are the same);
one that touches an atom only one ligand has stays on throughout.

Residues of a chain are not affected: a non-standard residue in a protein or
in a peptide given residue by residue is handled as described above.

## OpenFF instead of GAFF2

A free ligand can take an OpenFF force field such as Sage in place of GAFF2,
through [openmmforcefields](https://github.com/openmm/openmmforcefields):

```python
patch = boonza.openff_patch(s, ffs, ligandff="openff-2.2.1")
ff = boonza.merge_forcefields("aa.amber.ff19SB", patch)
p = boonza.parameterize(s, [ff, "water.tip3p", "ions.amber1jc.tip3p"])
```

```bash
boonza parameterize in.dms out.dms -f aa.amber.ff19SB -f water.tip3p --openff
boonza md complex.dms --ligandff openff-2.2.1
```

GAFF2 stays the default. `ligandff` (`--ligandff`) names the force field:
`gaff-2.11`, or one of Sage's releases, `openff-2.0.0` to `openff-2.3.0`, as
far as the installed openmmforcefields has it.

It needs openmmforcefields, which brings the OpenFF toolkit, and for NAGL
charges openff-nagl. They are optional: GAFF2 needs none of them, and they
bring PyTorch (about 60 packages, 150 MB).

```bash
bash install.sh --openff          # with boonza's environment
conda install -c conda-forge openmmforcefields openff-forcefields openff-nagl openff-nagl-models
```

`environment.yml` lists them commented out. Without them
`ligandff = "openff-..."` stops with the command to install them.

- **What is OpenFF's.** openmmforcefields' `SMIRNOFFTemplateGenerator` types
  the molecule, and its bonds, angles, torsions, impropers and Lennard-Jones
  become the patch as they are: the energies of those terms equal
  openmmforcefields' own. Sage has three impropers to a planar centre, one
  for each order of the outer atoms, and all three are kept.
- **Charges** are the kind the force field specifies, unless `charge_model`
  (`--charge-model`) names one:
  - `openff-2.3.0` specifies **NAGL** charges: a graph network trained on
    AM1-BCC, from openff-nagl. There is no quantum calculation, so a ligand
    of 500 atoms is charged in under a second, the charges do not depend on
    the pose (`charge_geometry` does not apply), and AmberTools is not
    needed. They are the charges openmmforcefields assigns on its own.
  - `openff-2.2.1` and earlier specify **AM1-BCC**, computed by AmberTools
    as for GAFF2: `charge_geometry` means the same, and a molecule has the
    same charges under GAFF2 and under these.
  - `charge_model = "nagl"` gives an earlier release NAGL charges (what to
    reach for when the ligand is peptide-sized), and `"am1bcc"` gives 2.3.0
    AmberTools'. The two differ by more than rounding: 0.28 e on the sulfur
    of a sulfonamide. GAFF2 has AM1-BCC alone.
- **1-4 electrostatics** take the scale of the Amber force field the patch
  joins, 0.8333 in viparr's files where Sage says 0.833333: hundredths of a
  kcal/mol in a drug-sized molecule, and the one place the result is not
  openmmforcefields' to the last digit.
- **Free ligands of one residue.** A covalent adduct and a non-standard
  residue mix protein types with the ligand's, which is built on GAFF2: in
  `boonza md`, `swim` and `fep` they get GAFF2 templates beside the OpenFF
  ones, and the log says which (`boonza.openff.ligand_patch` does both;
  `openff_patch` alone stops at them). A pseudopeptide is refused: it is a
  free ligand, and its protein types are GAFF2's to mix.
- **`boonza md`, `boonza swim` and `boonza fep`** all take `--ligandff`.
  A swim's kept ligand templates are made again when the force field
  changes. A `boonza fep prep --extend` must name the force field the first
  prep used: one network has one ligand force field.

## MMFF94 instead of GAFF2

A free ligand can also take MMFF94, with the parameters RDKit has: no
AmberTools, no OpenFF, no quantum calculation, and a few milliseconds a
molecule.

```python
patch = boonza.mmff_patch(s, ffs)  # or ligandff="mmff94s"
ff = boonza.merge_forcefields("aa.amber.ff19SB", patch)
p = boonza.parameterize(s, [ff, "water.tip3p", "ions.amber1jc.tip3p"])
```

```bash
boonza parameterize in.dms out.dms -f aa.amber.ff19SB -f water.tip3p --mmff
boonza md complex.dms --ligandff mmff94
boonza swim receptor.dms --ligand ligand.sdf --ligandff mmff94
```

`ligandff` is `mmff94`, or `mmff94s`, the variant that keeps delocalized
trigonal nitrogens flat (made for minimized geometries; MMFF94 is the one
made for dynamics).

- **Inside the ligand the energy is MMFF94's, every term.** MMFF94 is not an
  Amber-style force field, and nothing is approximated to make it one: bond
  stretches are quartic, angle bends cubic (a cosine where the angle is
  linear at rest), a stretch-bend term couples the two, out-of-plane bending
  is a Wilson angle, torsions have three terms, and atoms three or more
  bonds apart meet by the buffered 14-7 potential and a buffered Coulomb term
  (1-4 pairs at 0.75) instead of Lennard-Jones 12-6. The energy equals
  RDKit's term by term, to 0.0002 kcal/mol, at the minimum and bent far out
  of shape. The harmonic part of each bond and angle is an ordinary
  `stretch_harm` or `angle_harm` term, so bonds to hydrogen are constrained
  as everywhere else; the rest are six tables of boonza's own
  (`stretch_anharm`, `angle_anharm`, `angle_linear`, `stretchbend_harm`,
  `wilson_harm`, `pair_buf14_7_es`), which a `.dms` keeps and OpenMM runs as
  custom forces. A `.mae`, GROMACS or Amber file cannot carry them.
- **Charges** are MMFF94's own bond-charge increments, from the formal
  charges and bond orders of the input: instant, and the same in every pose.
  `charge_model` and `charge_geometry` do not apply, and a charge model other
  than `auto` is refused (the energy is fitted with these charges).
- **Between the ligand and everything else there is no MMFF94.** The
  protein, water and ions are Amber's, with no 14-7 parameters, and MMFF94
  has no rule for meeting them. boonza's rule is its own, and **not a
  published or validated force field**:
  - plain Coulomb with the MMFF94 charges;
  - 12-6 Lennard-Jones with each ligand atom's own MMFF94 minimum distance
    and well depth, combined with the other atom's as Amber combines;
  - no Lennard-Jones on a hydrogen that MMFF94 counts a hydrogen-bond donor.
    MMFF94 shrinks a donor-acceptor contact to 0.8 of its size, and Amber
    gives a water or hydroxyl hydrogen none; a 12-6 wall at the unshrunk size
    holds hydrogen bonds 0.6 Å too far apart (the last column below).

  Hydrogen bonds to a TIP3P water, minimized in vacuum (H···acceptor
  distance, and for the force fields that mix with Amber the binding
  energy in kcal/mol):

  | ligand, its role | MMFF94, both molecules | boonza's rule | GAFF2 | 12-6 on donor H too |
  |---|---|---|---|---|
  | N-methylacetamide, donor | 1.82 Å | 1.81 Å, -7.3 | 1.92 Å, -6.3 | 2.43 Å, -4.6 |
  | N-methylacetamide, acceptor | 1.73 Å | 1.71 Å, -8.6 | 1.74 Å, -8.8 | 1.71 Å, -8.6 |
  | methanol, donor | 1.76 Å | 1.71 Å, -6.9 | 1.78 Å, -6.5 | 2.44 Å, -3.6 |
  | phenol, donor | 1.72 Å | 1.68 Å, -9.4 | 1.77 Å, -8.1 | 2.39 Å, -5.1 |
  | pyrrole, donor | 1.84 Å | 1.85 Å, -6.4 | 1.92 Å, -5.7 | 2.42 Å, -4.0 |
  | methylammonium, donor | 1.57 Å | 1.63 Å, -20.7 | 1.75 Å, -20.1 | 2.33 Å, +0.7 |
  | acetate, acceptor | 1.87 Å | 1.95 Å, -20.8 | 1.95 Å, -20.8 | 1.95 Å, -20.8 |

  So the geometry is MMFF94's, and a donor's hydrogen bond is 0.4 to 1.3
  kcal/mol stronger than GAFF2's. Seven dimers are a sanity check, not a
  validation: the ligand's conformational energies are MMFF94's, and its
  binding energies are no one's tested model. Two copies of a ligand meet by
  the same rule.
- **Free ligands of one residue**, as with OpenFF: a covalent adduct and a
  non-standard residue get GAFF2 templates beside the MMFF94 ones in
  `boonza md` and `boonza swim`, and a pseudopeptide is refused.
- **What MMFF94 has no types for** (metals, boron, most inorganic chemistry)
  stops with a message; GAFF2 or OpenFF may take it.
- **`boonza md` and `boonza swim`** take it. **`boonza fep` refuses it**: its
  hybrid topology morphs harmonic, torsion, Lennard-Jones and Coulomb terms,
  not MMFF94's own.
- **Size.** Every pair of atoms in a ligand is a term of its own, so the
  number of terms grows with the square of its size: 5,000 for 100 atoms,
  140,000 for 530. Drug-sized ligands cost nothing; a peptide-sized one is
  better served by GAFF2 or OpenFF.

## Drawings of covalent adducts

When a ligand is bound to an amino acid other than by a peptide bond,
`gaff2_patch(..., draw=directory)` (`boonza parameterize --gaff2 --draw
DIR`; `boonza md` does it in its work directory) writes
`covalent_<residues>.png`: a 2D drawing of the ligand and the residue it is
bound to. Heavy atoms that keep protein types are blue and GAFF2 atoms
orange; the residue's atoms and the ligand atom bound to it are labelled
with their names, and `*` marks where the chain goes on. Free ligands and
non-standard residues without a ligand are not drawn.

## What the input needs

- All hydrogens, and bonds.
- Formal charges. SDF, MAE and DMS files carry them with bond orders. When the
  bonds of a group are all single (PDB, GRO, mmCIF), boonza perceives bond
  orders and charges with RDKit (`boonza.assign_bond_orders`) for a total
  charge taken from `charges={"LIG": -1}` (`--charge LIG=-1`), or 0.
- An Amber protein force field in the list (12-6 Lennard-Jones,
  Lorentz-Berthelot, 1-4 scales 1/1.2 and 1/2, like GAFF2): the patch is for
  the first force field with amino-acid templates, wherever it is listed. With no force
  fields, `gaff2_patch` returns a complete GAFF2 force field.

## How it is checked

- A free ligand parameterized from the patch reproduces tleap's prmtop term
  for term: bonds, angles, dihedrals, impropers, 1-4 pairs, exclusions,
  Lennard-Jones, charges and masses. The same holds after the patch is
  written to disk and read back.
- Against openmmforcefields' `GAFFTemplateGenerator` (`gaff-2.11`, AM1-BCC
  through AmberTools), aspirin gets the same charges and terms. The only
  difference is one improper whose equivalent atoms are listed in another
  order.
- A Cys-acrylamide adduct in a peptide keeps ff14SB types and charges on the
  Cys backbone and CB. Every term that touches a GAFF2 atom equals tleap's,
  every residue has an integer charge, and with ff19SB the Cys keeps its
  residue-specific CMAP. The system minimizes and runs stably in OpenMM.
- A covalent drug on a protein cysteine (a PDB entry; hydrogens from PDBFixer and
  RDKit) under ff19SB. The cysteine and the ligand form the only group. It keeps
  its ff19SB types, including the Cys-specific CA type, and its CMAP; SG is
  the one GAFF2 atom in the cysteine and takes a +0.16 e correction. The cysteine and the
  ligand each come to 0, and the protein minimizes and runs 1 ps in OpenMM.
  AM1-BCC on the capped group (about 100 atoms) took 4 minutes, which is
  most of the time; identical copies of a ligand are charged once.

## Limits

- Groups are capped only across peptide bonds; a group bonded to the rest by
  another kind of bond is refused. So is a residue in a chain whose backbone
  N or C does not match its parent: its terms would reach past the cap.
- Only Amber hosts, and GAFF2 only; for CHARMM, CGenFF is not included.
