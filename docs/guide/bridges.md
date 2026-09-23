# RDKit and OpenMM

## RDKit

```python
mol = s.select("resname LIG").to_rdkit()  # or boonza.to_rdkit(s, atoms)
mols = boonza.fragments_to_rdkit(s, "not water")  # one molecule per fragment
s2 = boonza.from_rdkit(mol)  # hydrogens must be explicit...
s2 = boonza.from_rdkit(Chem.MolFromSmiles("CCO"), add_hydrogens=True)
boonza.assign_bond_orders(s, "resname LIG")  # perceive orders and charges (PDB/GRO/CIF ligands)
s.select("smarts 'c1ccccc1'")  # SMARTS inside selections
```

- Every RDKit atom carries its boonza index as the int property
  `boonza_index` and its name as `_Name`. Residue data travels as PDB
  residue info, and ct properties become molecule properties.
- Bond orders come from the system. Bonds to pseudo particles become
  zero-order.
- As in RDKit's MOL reader, open valences get implicit hydrogens. Pass
  `implicit_hydrogens=False` to keep radicals.
- Stereochemistry is assigned from 3D coordinates.
- `assign_bond_orders` uses RDKit's xyz2mol on each molecule and chooses the
  resonance form with the fewest charges.

## OpenMM

```python
topology, system, positions = boonza.to_openmm(s, nonbonded_method="PME", cutoff=9.0)
energies = boonza.openmm_energies(s)  # kcal/mol per table + "total"

s = boonza.from_openmm(topology, system, positions)  # parameterized
s = boonza.from_openmm(topology, positions=positions)  # structure only
boonza.save(s, "system.dms")
```

`to_openmm` translates:

- harmonic bonds and angles, trig dihedrals, harmonic impropers;
- `pair_12_6_es` 1-4 pairs and exclusions;
- Lorentz-Berthelot or geometric combining, NBFIX overrides;
- constraints, including rigid water;
- virtual sites (`lc2`-`lc7`, `out3`, `fdat3`);
- position restraints and CMAP.

Constrained stretch and angle terms are left out when `constraints=True`.

### Martini

Martini runs with GROMACS's reaction field and potential-shifted
Lennard-Jones, which `to_openmm` reproduces:

```python
s = boonza.load("topol.top", coordinates="cg.gro")  # e.g. from martinize2
topology, system, positions = boonza.to_openmm(
    s,
    nonbonded_method="CutoffPeriodic",
    cutoff=11.0,
    dispersion_correction=False,
    epsilon_r=15,
    epsilon_rf=0,
    lj_shift=True,
)
```

- `epsilon_r` screens every charge interaction, 1-4 pairs included.
- `epsilon_rf=0` means an infinite reaction-field dielectric, as in GROMACS.
  The reaction field is not a plain cutoff: the force goes to zero at the
  cutoff, and GROMACS adds a term for each excluded pair within it and one
  for each charge. boonza adds both (forces `nonbonded_rf_exclusions` and
  `nonbonded_rf_self`), so energies equal GROMACS's. Only
  `epsilon_rf == epsilon_r` is a plain cutoff of the screened Coulomb.
- `lj_shift` shifts Lennard-Jones to zero at the cutoff (GROMACS's
  `Potential-shift-verlet`).
- Martini's cosine (`angle_cosine_harm`) and restricted-bending
  (`angle_restricted`) angles become custom angle forces, and
  `virtual_lc4`-`virtual_lc7` become centre-of-weight sites.
- GROMACS's `-rerun` takes virtual sites as they are in the file, so
  compare energies with sites placed by the same construction.

`from_openmm` reads what OpenMM builds from Amber, CHARMM and GROMACS inputs
and from `ForceField` XML:

- standard forces;
- Urey-Bradley terms;
- Ryckaert-Bellemans torsions (as exact Fourier series);
- CMAP;
- CHARMM lone pairs (as `virtual_fdat3`);
- the custom forces those setups create: harmonic impropers, tabulated NBFIX
  Lennard-Jones, 1-4 Lennard-Jones bonds, geometric Lennard-Jones and
  position restraints.

Custom forces are recognized by evaluating them on small test geometries,
not by parsing their expressions, so equivalent formulas are all accepted.
Forces with no DMS equivalent raise an error, or are skipped with
`ignore_unknown=True`.

For example, to turn a CHARMM-GUI system into DMS:

```python
from openmm import app

params = app.CharmmParameterSet(*toppar_files)
psf = app.CharmmPsfFile("step5_assembly.psf")
crd = app.CharmmCrdFile("step5_assembly.crd")
omm = psf.createSystem(params, nonbondedMethod=app.NoCutoff, constraints=None)
s = boonza.from_openmm(psf.topology, omm, crd.positions)
boonza.save(s, "system.dms")
```

Energies after the round trip match OpenMM's own evaluation, category by
category, for Amber (prmtop), CHARMM36 (`charmm36.xml` and CHARMM-GUI PSF
with NBFIX and CGenFF lone pairs) and GROMACS-style forces.
