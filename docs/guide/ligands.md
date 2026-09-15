# Ligands and covalent adducts (GAFF2)

Protein force fields have templates for the standard residues only. For
everything else (drug-like ligands, non-standard amino acids and residues
carrying a covalently bound ligand), boonza makes new templates with GAFF2
atom types and AM1-BCC charges, and adds them to the protein force field as a
viparr patch. The protein keeps the protein force field's parameters as far
as its chemistry allows.

It needs [AmberTools](https://ambermd.org/AmberTools.php) (antechamber, sqm,
parmchk2, tleap), which is installed with conda; OpenFF is not needed:

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
- KRAS G12C with covalent sotorasib (PDB 6OIM; hydrogens from PDBFixer and
  RDKit) under ff19SB. Cys12 and the ligand form the only group. Cys12 keeps
  its ff19SB types, including the Cys-specific CA type, and its CMAP; SG is
  the one GAFF2 atom in Cys12 and takes a +0.16 e correction. Cys12 and the
  ligand each come to 0, and the protein minimizes and runs 1 ps in OpenMM.
  AM1-BCC on the capped group (about 100 atoms) took 4 minutes, which is
  most of the time; identical copies of a ligand are charged once.

## Limits

- Groups are capped only across peptide bonds; a group bonded to the rest by
  another kind of bond is refused. So is a residue in a chain whose backbone
  N or C does not match its parent: its terms would reach past the cap.
- Only Amber hosts, and GAFF2 only; for CHARMM, CGenFF is not included.
