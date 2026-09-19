# API reference

Generated from the code by `docs/gen_reference.py`; see the [guide](index.md) for explanations and examples.

## Files

### `boonza.load(path, format: 'str | None' = None, **kwargs)`

Read a system from ``path``; the format comes from the extension unless given.

### `boonza.save(system, path, format: 'str | None' = None, **kwargs) -> 'None'`

Write ``system`` to ``path``; the format comes from the extension unless given.

### `boonza.open_trajectory(path, system=None, format: 'str | None' = None) -> 'Trajectory'`

Open a trajectory: DCD, XTC, TRR, Amber NetCDF, or Desmond DTR/STK.

With ``system`` the atom counts must agree.

### `boonza.open_writer(path, natoms: 'int', format: 'str | None' = None, **kwargs)`

A DCD, XTC, TRR or Amber NetCDF writer:
``with open_writer(p, n) as w: w.write(positions, box)``.

### `class boonza.Trajectory(path, system=None)`

Base class; formats implement ``_read(indices) -> Frames`` for all atoms.

### `class boonza.Frames(indices: 'np.ndarray', positions: 'np.ndarray', boxes: 'np.ndarray', times: 'np.ndarray', steps: 'np.ndarray') -> None`

A block of frames: positions (nframes, natoms, 3), boxes (nframes, 3, 3), ...

### `class boonza.Frame(index: 'int', positions: 'np.ndarray', box: 'np.ndarray', time: 'float', step: 'int') -> None`

One frame: positions (natoms, 3), box (3, 3), time (ps) and MD step.

### `class boonza.DMSError(...)`

Malformed or unsupported DMS content.

## System and handles

### `class boonza.System(name: 'str' = '')`

A molecular system: atoms, bonds, residues, chains, cts and force-field tables.

### `class boonza.AtomSel(system, ids)`

An ordered set of atoms from one system.

### `class boonza.Atom(system, id)`

### `class boonza.Bond(system, id)`

### `class boonza.Residue(system, id)`

### `class boonza.Chain(system, id)`

### `class boonza.Ct(system, id)`

A namespace of chains, plus free-form key/value properties.

### `class boonza.StaleHandleError(...)`

A handle was used after the indices it refers to were renumbered.

### `class boonza.NonbondedInfo(vdw_funct: 'str' = '', vdw_rule: 'str' = '', es_funct: 'str' = '') -> None`

NonbondedInfo(vdw_funct: 'str' = '', vdw_rule: 'str' = '', es_funct: 'str' = '')

## Force-field tables

### `class boonza.TermTable(name: 'str', natoms: 'int', category: 'str' = 'bond', params=None, system=None)`

Interactions of fixed arity, each pointing at one param row.

### `class boonza.Term(table: 'TermTable', id: 'int')`

One interaction in a TermTable.

### `class boonza.ParamTable()`

Rows of parameters; columns are named properties.

### `class boonza.Param(table: 'ParamTable', id: 'int')`

One row of a ParamTable.

### `class boonza.OverrideTable()`

Pairwise replacement parameters, keyed by a pair of param ids (NBFIX-style).

### `boonza.TERM_SCHEMAS`

72 entries: `alchemical_angle_harm`, `alchemical_angle_harm_soft`, `alchemical_dihedral_trig`, `alchemical_dihedral_trig_soft`, `alchemical_improper_harm`, `alchemical_improper_harm_soft`, `alchemical_pair_12_6_es`, `alchemical_pair_exp_6_es`, `alchemical_softstretch_harm`, `alchemical_stretch_harm`, `alchemical_stretch_morse`, `alchemical_torsiontorsion_cmap`, `angle_fbhw`, `angle_harm`, `constraint_ah1`, `constraint_ah1R`, `constraint_ah2`, `constraint_ah2R`, `constraint_ah3`, `constraint_ah3R`, `constraint_ah4`, `constraint_ah4R`, `constraint_ah5`, `constraint_ah6`, `constraint_ah7`, `constraint_ah8`, `constraint_hoh`, `dihedral6_trig`, `dihedral_fourier`, `dihedral_trig`, `exclusion`, `improper_anharm`, `improper_fbhw`, `improper_harm`, `inplanewag_harm`, `pair_12_6_es`, `pair_exp_6_es`, `pair_softcore_es`, `posre_fbhw`, `posre_harm`, `pseudopol_fermi`, `rigid_explicit2`, `rigid_explicit3`, `rigid_explicit4`, `rigid_explicit5`, `rigid_explicit6`, `rigid_explicit7`, `rigid_explicit8`, `rigid_explicit9`, `softened_stretch_harm`, `softstretch_harm`, `stretch_harm`, `stretch_morse`, `torsiontorsion_cmap`, `virtual_fdat3`, `virtual_lc1`, `virtual_lc2`, `virtual_lc2n`, `virtual_lc3`, `virtual_lc3n`, `virtual_lc4`, `virtual_lc4n`, `virtual_lc5`, `virtual_lc5n`, `virtual_lc6`, `virtual_lc6n`, `virtual_lc7`, `virtual_lc7n`, `virtual_midpoint`, `virtual_out3`, `virtual_out3n`, `virtual_sp3`

### `boonza.NONBONDED_SCHEMAS`

4 entries: `polynomial_cij`, `vdw_12_6`, `vdw_exp_6`, `vdw_exp_6s`

### `boonza.update_exclusions(system, separation: 'int' = 3, pair_scales=None) -> 'None'`

Replace the exclusion table (and, with ``pair_scales``, the 1-4 pairs) from the bonds.

## Reports and checks

### `boonza.describe(system, atoms=None, terms: 'str' = 'any', pairs: 'bool | None' = None, tables=None) -> 'ForceFieldReport'`

Force-field report for ``atoms`` (indices, AtomSel or selection string).

``terms="any"`` lists every term touching one of the atoms; ``"all"``
only terms whose atoms are all chosen.  ``pairs`` lists every pair of
chosen atoms (default: when at most 30 atoms are chosen).  ``tables``
limits the term tables reported.

### `class boonza.ForceFieldReport(nonbonded_info: 'dict', atoms: 'list[dict]', pairs: 'list[dict]' = <factory>, terms: 'dict[str, list[dict]]' = <factory>) -> None`

Rows describing the force field of a set of atoms (see ``describe``).

### `boonza.validate(system, strict: 'bool' = False, max_ring: 'int' = 10) -> 'list[Problem]'`

Problems found in ``system`` (empty when it passes); see the module docstring.

### `class boonza.Problem(check: 'str', message: 'str', atoms: 'tuple[int, ...]' = ()) -> None`

One failed check: its name, a message and the atoms involved.

### `boonza.find_knots(system, max_cycle_size=None, selection='all', ignore_excluded_knots=False, positions=None, cell=None) -> 'list[tuple]'`

Bonds passing through rings, as msys dms-find-knot.

Returns ``(ring, bond, idx)`` tuples: ``bond`` (i, j) crosses the
triangle ring[0], ring[idx], ring[idx + 1].  Rings come from ``sssr``
over the selected atoms, limited to ``max_cycle_size`` atoms; bonds
within 10 Å of a ring are tested.  The search is repeated with the
system shifted by half a box to catch knots across periodic boundaries.
``ignore_excluded_knots`` skips bonds whose atoms are excluded from every
ring atom (needs an exclusion table).

### `boonza.diff(a, b, atom_map=None, rtol: 'float' = 1e-06, atol: 'float' = 1e-09, positions: 'bool' = True, tables=None, canonical: 'bool' = False) -> 'list[Difference]'`

Differences between systems ``a`` and ``b`` (empty when they match).

``positions=False`` ignores coordinates, velocities and the cell;
``tables`` limits the force-field tables compared.  ``canonical``
compares force fields made by different programs: see :func:`canonical_forcefield`.

### `class boonza.Difference(kind: 'str', message: 'str') -> None`

One difference: what kind of data differs and how.

## Molecules and rings

### `boonza.distinct_fragments(system) -> 'dict[int, list[int]]'`

{representative fragment id: fragment ids of all identical molecules}.

### `boonza.sssr(system, atoms=None, all_relevant: 'bool' = False) -> 'list[list[int]]'`

Smallest set of smallest rings among ``atoms`` (default all), as msys GetSSSR.

The SSSR is not unique; ``all_relevant`` returns the union of all such
sets (every relevant cycle).  Each ring lists its atoms in ring order.

### `boonza.ring_systems(system, rings) -> 'list[list[int]]'`

Group ``rings`` into fused systems sharing bonds (msys RingSystems); ring indices.

## Geometry

### `class boonza.Glue(system, glue=None, center=None, fit=None, reference=None, whole='all', wrap: 'bool' = True, weights=None)`

Per-frame periodic fixing; see the module docstring.

``glue``: selection or list of selections (strings or atom indices); the
molecules touching each one are kept together.  ``center``: atoms whose
center the box is wrapped around.  ``fit``: atoms superposed onto
``reference`` (positions of all atoms or of the fit atoms; the first
processed frame when None).  ``whole``: atoms to make whole ("all", a
selection, or None to skip).  ``weights``: "mass" or None for fitting.

### `boonza.make_whole(system, positions=None, box=None) -> 'np.ndarray'`

Positions with every bonded molecule unbroken across the box (a copy).

## Superposition

### `boonza.matchmaker(mobile, reference, mobile_chain=None, reference_chain=None, cutoff=2.0, ss=True, apply: 'bool' = True, **nw_options) -> 'MatchResult'`

Superpose ``mobile`` onto ``reference`` (Systems) by sequence alignment, ChimeraX style.

``mobile_chain``/``reference_chain``: chain index or chain name; by
default every pair of chains with sequence is tried and the best
alignment score wins.  ``ss``: use secondary structure from
``chimerax_ss`` (True), none (False), or a dict {system id: per-residue
H/S/O array}.  ``cutoff=None`` fits all pairs without pruning.  Extra
keywords go to ``needleman_wunsch`` (``gap_open``, ``ss_fraction``, ...).
``apply`` moves ``mobile`` (positions and cell).

### `class boonza.MatchResult(rotation: 'np.ndarray', translation: 'np.ndarray', rmsd: 'float', full_rmsd: 'float', score: 'float', mobile_atoms: 'np.ndarray', reference_atoms: 'np.ndarray', kept: 'np.ndarray', mobile_chain: 'int', reference_chain: 'int', aligned_mobile: 'str' = '', aligned_reference: 'str' = '', extra: 'dict' = <factory>) -> None`

Outcome of ``matchmaker``: the transform and the residue and atom pairing.

### `boonza.needleman_wunsch(seq1, seq2, ss1=None, ss2=None, matrix=None, gap_open=12.0, gap_extend=1.0, ss_fraction=0.3, ss_scores=None, gap_open_helix=18.0, gap_open_strand=18.0, gap_open_other=6.0)`

ChimeraX Needleman-Wunsch: (score, matched positions in seq1, in seq2).

``ss1``/``ss2`` are strings of H (helix), S (strand), O (other) or blank
(unknown), one per residue; without them only the similarity matrix and
the plain gap penalties are used.  Penalties are given as positive numbers.

### `boonza.chimerax_ss(system, positions=None, energy_cutoff: 'float' = -0.5, min_helix_len: 'int' = 3, min_strand_len: 'int' = 3) -> 'np.ndarray'`

Per-residue secondary structure 'H' (helix), 'S' (strand) or 'O', as ChimeraX assigns it.

A port of ChimeraX's own Kabsch-Sander implementation (atomstruct
CompSS.cpp, Copyright Regents of the University of California, LGPL 2.1),
which matchmaker uses to guide sequence alignment.  It differs from
``dssp`` (the mdtraj DSSP 2.2 port): hydrogens named H are used when
present, residues are consecutive in residue order (bonded neighbors get
rebuilt imide hydrogens), helices of any kind (3-10, alpha, pi) count,
beta bulges merge ladders, ladders shorter than ``min_strand_len`` are
dropped, and a helix overrides a strand.

### `boonza.cealign(mobile, reference, sel=None, ref_sel=None, window: 'int' = 8, max_gap: 'int' = 30, apply: 'bool' = True) -> 'CEResult'`

Superpose ``mobile`` onto ``reference`` (Systems) by CE structural alignment.

Like PyMOL's cealign, no sequence similarity is needed.  ``sel`` and
``ref_sel`` restrict the atoms considered (e.g. "chain A"); guide atoms
are CA (and C4' for nucleotides).  ``apply`` moves ``mobile``.

### `class boonza.CEResult(rotation: 'np.ndarray', translation: 'np.ndarray', rmsd: 'float', z_score: 'float', mobile_atoms: 'np.ndarray', reference_atoms: 'np.ndarray') -> None`

Outcome of ``cealign``: transform, fit and the paired guide atoms.

### `boonza.ce_align(reference, mobile, window: 'int' = 8, max_gap: 'int' = 30, final_optimization: 'bool' = True)`

Align ``mobile`` onto ``reference`` (C-alpha coordinates, (n, 3) arrays) by CE.

Returns (reference indices, mobile indices, RMSD, rotation, translation,
z-score); ``mobile @ rotation.T + translation`` superposes the mobile
coordinates.  As in Biopython, the longest paths are ranked by RMSD and
a significant alignment (z >= 3.5) is refined by shifting each aligned
position by up to half a window.

### `boonza.superpose(mobile, reference, sel: 'str' = 'protein and name CA', ref_sel: 'str | None' = None, match: 'str' = 'sequence', cutoff: 'float | None' = 2.0, iterations: 'int' = 50, apply: 'bool' = True) -> 'Superposition'`

Superpose ``mobile`` onto ``reference`` (Systems) using the selected atoms.

``match="sequence"`` pairs one atom per residue by aligning residue
sequences; ``match="order"`` pairs the selections in order.  With
``cutoff``, the pairs farthest apart after a fit are pruned, at most 10%
per cycle and only those beyond ``cutoff`` (like ChimeraX matchmaker),
and the fit is repeated until none are beyond it.  ``apply`` moves
``mobile`` (positions and cell).

### `boonza.kabsch(mobile, target, weights=None) -> 'tuple[np.ndarray, np.ndarray]'`

Rotation R and translation t minimizing |mobile @ R.T + t - target| (weighted).

### `boonza.rmsd(a, b, weights=None, superpose: 'bool' = False) -> 'float'`

RMSD between two sets of positions, optionally after the best rigid fit.

## Sequences

### `boonza.align_sequences(a, b, mode: 'str' = 'global', matrix=None, gap_open: 'float' = 10.0, gap_extend: 'float' = 0.5, end_gaps: 'bool' = True, chain_a=None, chain_b=None) -> 'SequenceAlignment'`

Align two sequences (strings, or Systems with ``chain_a``/``chain_b``).

``matrix``: {(x, y): score} (default BLOSUM-62).  ``gap_open`` and
``gap_extend`` are positive penalties.  See the module docstring for the
scoring and the statistics.

### `class boonza.SequenceAlignment(seq_a: 'str', seq_b: 'str', aligned_a: 'str', aligned_b: 'str', score: 'float', mode: 'str', pairs: 'np.ndarray', positive: 'int') -> None`

A pairwise alignment: gapped strings, score, paired positions and statistics.

### `boonza.sequence(system, chain=None) -> 'str'`

One-letter sequence of a chain (index or name) of amino acids and nucleotides.

### `boonza.identity_matrix(sequences, metric: 'str' = 'identity', **kwargs) -> 'np.ndarray'`

Pairwise ``metric`` ("identity", "similarity" or "identity_shorter") of sequences.

## Analysis

### `boonza.rmsd_trajectory(system, positions=None, sel='all', reference=None, weights=None, fit: 'bool' = True) -> 'np.ndarray'`

RMSD (Å) of the ``sel`` atoms in each frame to ``reference`` (MDAnalysis RMSD).

``reference``: positions of all atoms or of the selected atoms (default:
the first frame).  With ``fit`` each frame is optimally superposed first.
``weights``: None, "mass" or per-atom weights.

### `boonza.rmsf(system, positions=None, sel='all') -> 'np.ndarray'`

Root mean square fluctuation (Å) of each ``sel`` atom about its mean position.

As MDAnalysis RMSF, frames are used as given: superpose them first
(e.g. with ``boonza.Glue(fit=...)``) to remove overall motion.

### `boonza.radius_of_gyration(system, positions=None, sel='all', weights='mass') -> 'np.ndarray'`

Radius of gyration (Å) per frame.

``weights="mass"``: about the center of mass, as MDAnalysis;
``weights=None``: equal weights about the centroid, as mdtraj's default.
Systems without masses (e.g. from PDB files) use element masses as
MDAnalysis guesses them.

### `boonza.rdf(system, g1, g2, positions=None, nbins: 'int' = 75, range=(0.0, 15.0), norm: 'str' = 'rdf', exclusion_block=None, exclude_same=None)`

Radial distribution function between groups ``g1`` and ``g2`` (MDAnalysis InterRDF).

Returns (bin centers, rdf, edges, counts).  ``norm``: "rdf" (g(r), using
the average box volume), "density" (single-particle density) or "none"
(counts).  ``exclusion_block=(nA, nB)`` drops pairs within consecutive
blocks of nA and nB atoms (e.g. the same molecule); ``exclude_same`` drops
pairs in the same "residue", "chain" or "fragment".

### `boonza.residue_contacts(system, positions=None, contacts='all', scheme: 'str' = 'closest-heavy', ignore_nonprotein: 'bool' = True, periodic: 'bool' = True)`

Residue-residue distances (Å) per frame, as mdtraj compute_contacts.

``contacts="all"``: pairs of residues in the same chain at least three
apart (only residues with a CA atom when ``ignore_nonprotein``), or an
(n, 2) array of residue indices.  ``scheme``: "ca" (CA-CA), "closest"
(closest atoms) or "closest-heavy" (closest non-hydrogen atoms).
Returns (distances of shape (nframes, npairs), residue pairs).

### `boonza.sasa(system, positions=None, probe_radius: 'float' = 1.4, n_sphere_points: 'int' = 960, mode: 'str' = 'atom', atoms=None, radii=None) -> 'np.ndarray'`

Solvent accessible surface area (Å^2) per atom or residue, per frame (mdtraj).

``radii``: {element symbol: radius in Å} overriding mdtraj's defaults.
``atoms``: compute only these atoms (others report -1, as mdtraj);
they are still occluded by every atom.  Frames are not made whole.

### `boonza.hbonds(system, positions=None, box=None, donors=None, hydrogens=None, acceptors=None, d_a_cutoff: 'float' = 3.0, angle_cutoff: 'float' = 150.0, between=None) -> 'HBonds'`

Hydrogen bonds by MDAnalysis HydrogenBondAnalysis criteria.

``positions``: one frame (natoms, 3), several (nframes, natoms, 3), or a
Frames/Trajectory block; the system's own positions and cell by default.
``box``: 3x3 cell (or one per frame) when ``positions`` has none.
``donors``/``hydrogens``/``acceptors``: selections or indices; defaults are
N and O donors bonded to hydrogens and N and O acceptors.  ``between``:
a pair of selections, or a list of pairs, to keep only bonds across them.

### `class boonza.HBonds(frame: 'np.ndarray', donor: 'np.ndarray', hydrogen: 'np.ndarray', acceptor: 'np.ndarray', distance: 'np.ndarray', angle: 'np.ndarray', nframes: 'int') -> None`

Hydrogen bonds found over frames; one entry per bond per frame.

### `boonza.baker_hubbard(system, positions=None, box=None, freq: 'float' = 0.1, exclude_water=True, distance_cutoff: 'float' = 2.5, angle_cutoff: 'float' = 120.0) -> 'np.ndarray'`

(donor, hydrogen, acceptor) triplets bonded in more than ``freq`` of frames (mdtraj).

Criterion: hydrogen-acceptor distance under ``distance_cutoff`` Å and
donor-hydrogen-acceptor angle above ``angle_cutoff`` degrees.

### `boonza.wernet_nilsson(system, positions=None, box=None, exclude_water=True) -> 'list[np.ndarray]'`

Per frame, (donor, hydrogen, acceptor) triplets by the Wernet-Nilsson cone (mdtraj).

Criterion: donor-acceptor distance under 3.3 Å - 0.00044 Å/deg^2 * theta^2,
with theta the hydrogen-donor-acceptor angle under 45 degrees.

### `boonza.dssp(system, positions=None, simplified: 'bool' = False, box=None, breaks: 'bool' = True) -> 'np.ndarray'`

DSSP codes, shape (nframes, nresidues).

``positions``: one frame, a (nframes, natoms, 3) array, a Frames block or
a Trajectory (read chunk by chunk, backbone atoms only).  ``box``: a (3, 3)
box or one per frame (Å) for minimum-image distances; Frames and
trajectories use their own boxes unless ``box`` is given (``box=False``
ignores them).  ``breaks``: cut chains at C-N gaps over 2.5 Å as the
DSSP program does (False: topology chains only, as mdtraj).

### `boonza.backbone_dihedrals(system, positions=None, box=None) -> 'BackboneDihedrals'`

Backbone phi, psi and omega per residue in degrees, each (nframes, nresidues).

phi(i) = C(i-1)-N(i)-CA(i)-C(i), psi(i) = N(i)-CA(i)-C(i)-N(i+1) and
omega(i) = CA(i-1)-C(i-1)-N(i)-CA(i), the peptide bond before residue i,
as mdtraj and the PDB define them.  Residues are linked when consecutive
in one chain with C-N under 2.5 Å in the first frame (the DSSP break
rule); angles lacking a linked neighbor or a backbone atom are NaN.
``box`` (3x3, Å) applies the minimum image to each bond vector.

### `boonza.backbone_hbonds(system, positions=None, box=None, breaks: 'bool' = True)`

Kabsch-Sander backbone H-bonds of one frame: (donor residue, acceptor residue, kcal/mol).

``box`` and ``breaks`` as in ``dssp``.

## RDKit

### `boonza.to_rdkit(system, atoms=None, sanitize: 'bool' = True, implicit_hydrogens: 'bool' = True, conformer: 'bool' = True, stereo: 'bool' = True, residue_info: 'bool' = True)`

RDKit molecule of ``atoms`` (indices, AtomSel, selection string; default all).

Bonds use the system's orders; bonds to pseudo particles are zero-order,
and bonds with a true ``aromatic`` property stay aromatic.  Like RDKit's
MOL reader, open valences are filled with implicit hydrogens; pass
``implicit_hydrogens=False`` to keep them as radicals.  With a conformer,
stereochemistry is assigned from 3D.  ct properties become molecule
properties.

### `boonza.from_rdkit(mol, conf_id: 'int' = -1, add_hydrogens: 'bool' = False, name: 'str | None' = None) -> 'System'`

System from an RDKit molecule; bonds are kekulized.

Implicit hydrogens are an error unless ``add_hydrogens`` (then they are
added, with coordinates when the molecule has a conformer).

### `boonza.fragments_to_rdkit(system, atoms=None, **kwargs) -> 'list'`

One RDKit molecule per molecule (fragment) touching ``atoms``.

### `boonza.assign_bond_orders(system, atoms=None, charge: 'int | None' = None) -> 'None'`

Perceive bond orders and formal charges from connectivity and geometry, in place.

Uses RDKit's DetermineBondOrders (xyz2mol) on each molecule touching
``atoms``; hydrogens must be present.  The total charge of each molecule
is its current formal charge sum unless ``charge`` is given (only for a
single molecule).  Best suited to ligands and other small molecules.

## OpenMM

### `boonza.to_openmm(system, nonbonded_method: 'str' = 'NoCutoff', cutoff: 'float' = 9.0, constraints: 'bool' = True, dispersion_correction: 'bool' = True, ewald_tolerance: 'float' = 0.0005)`

(Topology, System, positions) for OpenMM; cutoff in Å.

``nonbonded_method`` is an OpenMM NonbondedForce method name (NoCutoff,
CutoffNonPeriodic, CutoffPeriodic, Ewald, PME, LJPME).  With
``constraints``, the constraint tables become OpenMM constraints and
stretch/angle terms marked ``constrained`` are left out.

### `boonza.from_openmm(topology, omm_system=None, positions=None, ignore_unknown: 'bool' = False) -> 'System'`

System from an OpenMM Topology and, optionally, System (positions in nm).

Without ``omm_system`` only the structure is converted, with masses from
the elements.  Forces become DMS tables: harmonic bonds (Urey-Bradley
terms included) and angles, periodic and Ryckaert-Bellemans torsions,
CMAP, Coulomb and Lennard-Jones with exceptions, and the custom forms
CHARMM, GROMACS and OpenMM's ForceField use: harmonic impropers,
position restraints, tabulated Lennard-Jones (NBFIX), per-particle
geometric Lennard-Jones and 12-6/Coulomb pair bonds.  Custom forces are
recognized by evaluating them on small test geometries, not by their
text.  ``ignore_unknown`` skips forces without a DMS equivalent instead
of raising.

### `boonza.openmm_energies(system, positions=None, platform: 'str' = 'Reference', **kwargs) -> 'dict'`

Energy of each translated force in kcal/mol (and the total).

## Per-format readers and writers (`boonza.io`)

### `boonza.io.load_cif(path, guess_bonds: 'bool' = True, struct_conn: 'bool' = True) -> 'System'`

Read the first data block with an ``_atom_site`` loop.

Bonds are guessed from distances (``guess_bonds``), then the
``_struct_conn`` records (disulfides, covalent links, metal coordination;
not hydrogen bonds or bonds to symmetry copies) are added with
``struct_conn``, as SSBOND/LINK are for PDB files.

### `boonza.io.load_dms(path, structure_only: 'bool' = False, without_tables: 'bool' = False) -> 'System'`

Read a DMS file.

``without_tables`` skips the force field; ``structure_only`` also drops
pseudo particles (atomic number 0).

### `boonza.io.load_gro(path, guess_bonds: 'bool' = True) -> 'System'`

### `boonza.io.load_mae(path, structure_only: 'bool' = False, without_tables: 'bool' = False, ignore_unrecognized: 'bool' = False) -> 'System'`

Read an MAE/CMS file (optionally gzip/bzip2 compressed); all cts go into one System.

### `boonza.io.load_pdb(path, guess_bonds: 'bool' = True, conect: 'bool' = True, ssbond: 'bool' = True, link: 'bool' = True) -> 'System'`

Read a PDB file (optionally gzip/bzip2 compressed).

``guess_bonds``: bond atoms by distance (msys rules).  ``conect``,
``ssbond`` and ``link``: apply the file's CONECT, SSBOND and LINK records
on top (see the module notes); all False for msys behavior.

### `boonza.io.load_prmtop(path, coordinates=None, structure_only: 'bool' = False, without_tables: 'bool' = False) -> 'System'`

Read an Amber prmtop (or parm7) file, optionally with coordinates.

``coordinates``: an inpcrd/rst7 file (ASCII) or an Amber NetCDF restart
(positions, velocities and the cell).  ``structure_only`` or
``without_tables`` skip the force-field tables (bonds are still made).

### `boonza.io.load_psf(path, coordinates=None) -> 'System'`

Read a PSF file; ``coordinates``: a file boonza can load with the same
atoms (PDB, CRD-like formats, ...) or an (natoms, 3) array.

### `boonza.io.load_sdf(path) -> 'System'`

Read every entry of an SDF file (optionally gzip/bzip2 compressed).

### `boonza.io.load_top(path, coordinates=None, defines=None, include_dirs=None, structure_only: 'bool' = False) -> 'System'`

Read a GROMACS topology; ``coordinates``: a .gro/.pdb (anything boonza loads).

### `boonza.io.save_cif(system: 'System', path) -> 'None'`

Write an mmCIF file with ``_cell``, ``_symmetry`` and an ``_atom_site`` loop.

### `boonza.io.save_dms(system: 'System', path, structure_only: 'bool' = False) -> 'None'`

Write a DMS file; ``.gz`` / ``.bz2`` suffixes are compressed.

### `boonza.io.save_gro(system: 'System', path, precision: 'int' = 3) -> 'None'`

Write the system as .gro (nm); velocities are written when any is nonzero.

### `boonza.io.save_mae(system, path, structure_only: 'bool' = False, allow_reorder_atoms: 'bool' = False, append: 'bool' = False) -> 'None'`

Write ``system`` as MAE (``.gz``/``.bz2`` suffixes are compressed); one block per ct.

### `boonza.io.save_pdb(system: 'System', path, append: 'bool' = False, reorder: 'bool' = False, models='auto', conect='auto') -> 'None'`

Write a PDB file; ``reorder`` groups atoms by chain and residue first.

``models``: "auto" writes one model, or one MODEL per ct when the cts are
an ensemble (same atoms in each); True always one MODEL per ct (msys);
False always one model.  ``conect``: "auto" writes CONECT records for the
atoms whose bonds re-guessing on reading would get wrong; True for every
bonded atom; False never.

### `boonza.io.save_sdf(system: 'System', path, append: 'bool' = False, v3000: 'bool | None' = None) -> 'None'`

Write each ct as one SDF entry; V3000 when forced or above 999 atoms or bonds.

## Periodic geometry (`boonza.pbc`)

Periodic-boundary geometry and fast distances.

Boxes are (3, 3) arrays with the cell vectors as rows (Å); ``None`` or all
zeros means no periodicity, and ``(a, b, c, alpha, beta, gamma)`` is also
accepted.  Orthorhombic boxes use the direct minimum-image rule.  Triclinic
boxes are reduced in fractional coordinates and then the neighboring images
are checked, which gives the true minimum image for any cell shape.
Kernels are numba, parallel over the first set of positions.

    d = distances(a, b, box)                 # (n, m) matrix
    i, j, d = capped_distances(a, b, 5.0, box)
    theta = angles(p0, p1, p2, box)          # radians

### `boonza.pbc.angles(a, b, c, box=None) -> 'np.ndarray'`

Angle a-b-c in radians for every triple.

### `boonza.pbc.as_box(box) -> 'np.ndarray | None'`

(3, 3) cell vectors from a box or ``(a, b, c, alpha, beta, gamma)``; None if not periodic.

### `boonza.pbc.capped_distances(a, b, cutoff: 'float', box=None)`

Pairs (i of a, j of b) within ``cutoff``: returns (i, j, distance) sorted by (i, j).

Periodic searches consider the 27 nearest images, exact while the cutoff
is below half the smallest box width.

### `boonza.pbc.dihedrals(a, b, c, d, box=None) -> 'np.ndarray'`

Dihedral a-b-c-d in radians, in (-pi, pi], IUPAC sign convention.

### `boonza.pbc.distances(a, b, box=None) -> 'np.ndarray'`

All distances between two sets of positions: an (n, m) matrix.

### `boonza.pbc.minimum_image(vectors, box=None) -> 'np.ndarray'`

Shortest periodic images of displacement vectors, shape (n, 3).

### `boonza.pbc.paired_distances(a, b, box=None) -> 'np.ndarray'`

Distance between a[k] and b[k] for every k (bond lengths).

### `boonza.pbc.periodic_pairs(a, b, r: 'float', box)`

Pairs within ``r`` under periodic boundaries, on a cell grid aligned with the box.

``b=None`` gives pairs i < j within ``a``.  Returns (i, j, squared
distance) sorted by (i, j), or None when the box is too small for three
cells of height ``r`` along each cell vector.

### `boonza.pbc.self_distances(a, box=None) -> 'np.ndarray'`

Distances between all pairs i < j of one set, in condensed (scipy pdist) order.

## Other

### `class boonza.BlockAverage(mean: 'float', block_sizes: 'np.ndarray', sem: 'np.ndarray', sem_error: 'np.ndarray') -> None`

Standard error of the mean from blocks of increasing size (see ``block_average``).

### `class boonza.DRMSD(drmsd: 'float | np.ndarray', plain_drmsd: 'float | np.ndarray | None', pocket: 'np.ndarray', reference_pocket: 'np.ndarray', mapping: 'np.ndarray', mobile_ligand: 'np.ndarray', reference_ligand: 'np.ndarray', reference_distances: 'np.ndarray') -> None`

Outcome of ``drmsd``: pocket-ligand distance RMSD (per frame with ``positions``).

### `class boonza.Density(origin: 'np.ndarray', spacing: 'float', counts: 'np.ndarray', expected: 'float') -> None`

How often the ligand's centroid was in each cell of a grid, and what bulk would give.

``enrichment`` is the map worth looking at: one means as often as
wandering through the box uniformly would explain, and a site is where it
is large.  It is a check on the sites that does not come from clustering
at all.

### `class boonza.Dwell(run: 'int', copy: 'int', first: 'int', frames: 'int', bound: 'bool', censored: 'bool') -> None`

One stretch a copy spent in a state, and whether we saw it end.

<<<<<<< HEAD
### `class boonza.Hotspot(family: 'str', center: 'np.ndarray', enrichment: 'float', volume: 'float', points: 'int', ligands: 'int') -> None`

One place a kind of atom gathers, and what says so.
=======
### `class boonza.Interactions(values: 'np.ndarray', residues: 'np.ndarray', where: 'np.ndarray', center: 'float', width: 'float') -> None`

A fingerprint per frame and copy, and which residues its columns are.
>>>>>>> interaction-fingerprints

### `class boonza.LigandRMSD(rmsd: 'float | np.ndarray', plain_rmsd: 'float | np.ndarray | None', fit_rmsd: 'float | np.ndarray', rotation: 'np.ndarray', translation: 'np.ndarray', mapping: 'np.ndarray', mobile_ligand: 'np.ndarray', reference_ligand: 'np.ndarray') -> None`

Outcome of ``ligand_rmsd``: the protein fit and the ligand RMSD in that frame.

### `class boonza.MoleculeView(pdb: 'str', nframes: 'int', styles: 'list', width: 'int', height: 'int', background: 'str', interval: 'int')`

One 3Dmol.js viewer: shown by Jupyter, or written as a page with ``save``.

### `class boonza.OpenMMForcefield()`

OpenMM XML force field files, read into boonza (see :func:`load_openmm_forcefield`).

### `class boonza.PCA(atoms: 'np.ndarray', mean: 'np.ndarray', variance: 'np.ndarray', components: 'np.ndarray', projections: 'np.ndarray', reference: 'np.ndarray | None') -> None`

Principal components of atomic fluctuations (see ``pca``).

### `class boonza.Pose(center: 'int', frames: 'np.ndarray', population: 'float', spread: 'float') -> None`

One group of frames: its medoid, its share of the frames, its spread.

### `class boonza.PoseSet(poses: 'list[Pose]', labels: 'np.ndarray', distances: 'np.ndarray', merges: 'np.ndarray', cutoff: 'float', min_population: 'float') -> None`

The poses of a trajectory, most populated first.

### `class boonza.Rates(site: 'int', k_off: 'float', k_on: 'float', KD: 'float', dG: 'float', residence_ns: 'float', events: 'int', arrivals: 'int', bound_ns: 'float', unbound_ns: 'float', concentration: 'float', occupancy: 'float', occupancy_from_rates: 'float', dG_interval: 'tuple[float, float]', censored: 'int', dwells: 'list[Dwell]' = <factory>) -> None`

The kinetics of one site, with what they rest on.

### `class boonza.Site(center: 'np.ndarray', points: 'np.ndarray', occupancy: 'float', runs: 'int', copies: 'int', arrivals: 'int', spread: 'float') -> None`

One place the ligand is found, and the evidence for it.

### `class boonza.SiteSet(sites: 'list[Site]', labels: 'np.ndarray', where: 'np.ndarray', centroids: 'np.ndarray', spacing: 'float', enrichment: 'float', systems: 'list' = <factory>, volume: 'float' = 0.0, density: 'Density | None' = None) -> None`

The sites of a set of runs, most occupied first.

### `class boonza.Summary(title: 'str', sections: 'list[tuple[str, list[str]]]' = <factory>, data: 'dict' = <factory>) -> None`

A structure summary: ``str()`` for text (Markdown), ``to_dict()`` for data.

### `class boonza.SymmetryRMSD(rmsd: 'float | np.ndarray', plain_rmsd: 'float | np.ndarray | None', mapping: 'np.ndarray', mobile_atoms: 'np.ndarray', reference_atoms: 'np.ndarray', isomorphisms: 'int | None' = None, truncated: 'bool' = False) -> None`

Outcome of ``symmetry_rmsd``.

### `class boonza.ViparrForcefield(name: 'str', rules: 'Rules', templates, params: 'dict', cmaps=())`

A viparr force field: rules, templates, parameter tables and CMAP grids.

``params[table]`` lists the :class:`ParamRow` of each parameter file in
file order; ``cmaps[k - 1]`` is the grid that ``cmapid`` ``cmapk`` names,
an array of (phi, psi, energy) rows.

### `boonza.block_average(values, min_blocks: 'int' = 4) -> 'BlockAverage'`

Flyvbjerg-Petersen blocking of a time series (for example an RMSD or Q per frame).

For blocks of 1, 2, 4, ... frames (at least ``min_blocks`` blocks; a
remainder is dropped), the standard error of the mean is estimated from
the spread of the block means.  It grows with the block size until blocks
are longer than the correlation time, then levels off at the true error
(``estimate``).

### `boonza.bound_frame(counts) -> 'int'`

A typical bound frame: the median of the frames that touch the protein.

The *most* contacting frame is by construction an outlier -- the one where
a loop happened to close in -- and letting it decide the pocket lets one
frame speak for the run.

### `boonza.build_constraints(system: 'System', atoms=None, keep: 'bool' = False, exclude=()) -> 'None'`

Add viparr's constraints to a parameterized system, in place.

A heavy atom and the hydrogens bonded to it become one ``constraint_ahN``
term (heavy atom first, lengths from ``stretch_harm``); a water oxygen
with two hydrogens becomes ``constraint_hoh`` (with the H-O-H angle from
``angle_harm``). The ``stretch_harm`` and ``angle_harm`` terms they
replace get ``constrained = 1``, so :func:`boonza.to_openmm` turns them
into OpenMM constraints (``keep=True`` leaves them unconstrained).
``atoms`` limits the atoms considered; ``exclude`` skips kinds such as
``"hoh"`` or ``"ah1"``. Existing constraints of those atoms are replaced.

### `boonza.canonical_forcefield(system)`

A copy of ``system`` whose force field has one form for each interaction.

Programs write the same force field differently: one periodic torsion
term per periodicity or several per row, k with phase 0 or -k with phase
180, 1-4 electrostatics and Lennard-Jones in one term or two, a sigma for
atoms without Lennard-Jones or not.  Here every ``dihedral_trig`` and
``improper_trig`` term of an atom tuple (read in either direction) is
summed into the Fourier coefficients of table ``dihedral_fourier``
(E = c0 + sum_n a_n cos(n phi) + b_n sin(n phi)), ``pair_12_6_es`` terms
are summed per atom pair, sigma is 0 where epsilon is 0, and terms whose
energy is zero are dropped, so :func:`diff` sees only differences that
change the energy.

### `boonza.contact_frequency(system, sel1, sel2=None, positions=None, cutoff: 'float' = 4.5, level: 'str' = 'residue', periodic: 'bool' = True)`

How often each pair is in contact: (rows, cols, fraction of frames).

A residue pair (``level="residue"``) is in contact in a frame when any of
their atoms are within ``cutoff`` Å; ``level="atom"`` uses atom pairs.
``rows`` and ``cols`` are residue (or atom) indices of ``sel1`` and
``sel2``, and the matrix has one fraction per (row, col).  Without
``sel2`` the pairs are within ``sel1``, excluding a residue (atom) with
itself.  With ``periodic``, distances use each frame's box.

### `boonza.drmsd(system, reference=None, ligand: 'str' = 'not (polymer or water or ions) and noh', protein: 'str' = 'protein and name CA', cutoff: 'float' = 5.0, positions=None, reference_ligand=None, reference_protein=None, symmetry: 'bool' = True, heavy_only: 'bool' = True, bond_orders: 'bool' = False, periodic: 'bool' = True) -> 'DRMSD'`

Distance RMSD (Å) between pocket atoms and ligand atoms, against a reference.

The pocket is the ``protein`` atoms within ``cutoff`` Å of the ligand in
the reference.  For every frame, the (pocket x ligand) distance matrix
``d`` is compared with the reference matrix: ``sqrt(mean((d - d_ref)**2))``.
No superposition is needed, and the ligand's internal distances are not
included.

``reference``: a System (a crystal structure, say), or None for the first
frame of ``positions`` (the system's own coordinates without it), as in a
"drift from frame 0" analysis.  Protein atoms are paired between the two
systems in selection order (``reference_protein`` defaults to
``protein``; the counts must agree).  Ligand atoms are paired by symmetry:
with ``symmetry=True`` each frame uses the element- and bond-preserving
atom mapping with the smallest dRMSD (exact, like ``symmetry_rmsd``), so a
flipped ring or swapped carboxylate oxygens do not count as motion;
``plain_drmsd`` pairs the ligand atoms in listed order.

``positions``: one frame, an (nframes, natoms, 3) array, a Frames block or
a Trajectory (read chunk by chunk, only pocket and ligand atoms).  With
``periodic=True`` distances use the minimum image of each frame's box (the
system's cell for plain arrays).

### `boonza.dwells(found, site: 'int', hysteresis: 'float' = 2.0, quantile: 'float' = 0.9) -> 'list[Dwell]'`

The stretches each copy spent in ``site`` and out of it.

A copy enters when its centroid comes within ``r_in`` -- the distance
holding ``quantile`` of the site's own frames -- and leaves only past
``hysteresis`` times that.  The first and last stretch of every copy are
censored: they were cut by the trajectory, not by the ligand.

### `boonza.feature_maps(system, runs=None, reference=None, ligand: 'str' = 'not (polymer or water or ions) and noh', align: 'str' = 'protein and name CA', families=('Donor', 'Acceptor', 'Aromatic', 'Hydrophobe', 'PosIonizable', 'NegIonizable'), spacing: 'float' = 1.0, periodic: 'bool' = True) -> 'dict[str, Density]'`

A map per feature family: how much more often than bulk each kind is found where.

Every map shares one grid, so they can be read against each other -- a
place that wants an acceptor and not a donor is the interesting kind.
Bulk is worked out per family, from that family's own count, so a ligand
set rich in one kind does not make its map look hot everywhere.

### `boonza.feature_points(system, positions=None, reference=None, ligand: 'str' = 'not (polymer or water or ions) and noh', align: 'str' = 'protein and name CA', families=('Donor', 'Acceptor', 'Aromatic', 'Hydrophobe', 'PosIonizable', 'NegIonizable'), periodic: 'bool' = True) -> 'tuple[dict, np.ndarray]'`

``({family: (n, 3) positions}, {family: (n,) which copy})`` in the reference's frame.

### `boonza.find_unmatched(system: 'System', forcefields, path=None) -> 'list[list[int]]'`

Residue groups that ``forcefields`` cannot parameterize.

A group is a set of bonded residues without a matching template, plus the
residues bonded to them other than through a peptide bond or disulfide
(the residue a covalent ligand is bound to).  Returns residue indices.

### `boonza.from_smiles(smiles: 'str', name: 'str' = 'LIG', seed: 'int' = 42, optimize: 'bool' = True, conformers: 'int' = 1) -> 'System'`

A 3D molecule from a SMILES string (through RDKit).

Hydrogens are added, ``conformers`` conformers are embedded (RDKit ETKDG,
reproducible with ``seed``) and, with ``optimize``, minimized with MMFF94
(UFF when MMFF lacks parameters); the lowest-energy one is kept.  Bond
orders and formal charges come from the SMILES.  The molecule is one
residue named ``name`` with atoms named C1, C2, ..., H1, ...

### `boonza.gaff`

### `boonza.gaff2_patch(system: 'System', forcefields=(), *, charges=None, parents=None, gaff: 'str' = '2.11', charge_method: 'str' = 'bcc', amberhome=None, workdir=None, path=None, protein_extent: 'str' = 'matched', draw=None, tag=None) -> 'ViparrForcefield'`

A viparr patch with GAFF2 templates for what ``forcefields`` cannot parameterize.

``forcefields`` is the list to be given to :func:`boonza.parameterize`;
the patch is for the Amber protein force field among them (the first with
amino-acid templates, see :func:`host_index`) and is merged onto it::

    patch = boonza.gaff2_patch(system, ["aa.amber.ff14SB", "water.tip3p"])
    ff = boonza.merge_forcefields("aa.amber.ff14SB", patch)
    out = boonza.parameterize(system, [ff, "water.tip3p"])

With no force fields, the patch is a complete GAFF2 force field.

Each group from :func:`find_unmatched`, capped with ACE/NME where it is
bonded to a protein, is typed and charged by :func:`run_gaff2`.  Atoms of
an amino acid keep the types and charges of its parent residue's template
as far as each atom and its bonded neighbours match it. The parent is the
one ``parents`` gives (residue name to standard residue, as
``{"MSE": "MET"}``), else the file's (PDB ``MODRES``, mmCIF
``_pdbx_struct_mod_residue``), else the PDB dictionary's
(:data:`KNOWN_PARENTS`), else the template that matches best; a guess
that is ambiguous or misses the backbone and CB raises. Backbones are
found by atom names or by structure and templates matched by bond graph,
so names are not needed; only residues in a chain, or with a parent
named, count as amino acids. The patch's ``parents`` attribute lists
each choice.
Each residue is brought to its formal charge (``charges`` maps residue
names to charges where the input has none) by shifting its GAFF2 atoms
evenly. GAFF2 types are suffixed per group (``c3~1``) so groups never
share parameters; templates pin the elements (and, across bonds other than
peptide bonds, the residue formulas) of their external atoms, so a Cys
bound to a ligand takes its own template rather than CYX, even when the
ligand is bound through a sulfur.
``workdir`` keeps the AmberTools files, one directory per template.

``protein_extent``: "matched" keeps protein types on every amino-acid
atom that matches its parent with all its neighbours; "cb" only on the
backbone (N, H, CA, HA, C, O, and OXT or H1-H3 at the termini), CB and
the hydrogens on CB, so everything past CB is GAFF2. ``draw``: a
directory for ``covalent_<residues>.png``, a 2D drawing of each covalent
adduct (a ligand bound to an amino acid other than by a peptide bond)
with heavy atoms colored by where their types come from. ``tag`` makes
the GAFF2 types (``c3~<tag>``) and template names of this patch unique,
so patches made separately (one per ligand) can join one force field.

<<<<<<< HEAD
### `boonza.hotspots(maps, enrichment: 'float' = 20.0, min_volume: 'float' = 3.0) -> 'list[Hotspot]'`

The peaks of the maps: what to put where, most enriched first.

A hotspot is a connected region a family visits at least ``enrichment``
times more often than bulk would explain, of at least ``min_volume`` A^3
so that a single lucky frame is not one.  ``ligands`` counts the distinct
molecules that put a feature there, which is the part worth trusting: a
place five unlike molecules choose is a better bet than one a single
molecule sat in for a long time.
=======
### `boonza.interaction_fingerprints(system, positions=None, ligand: 'str' = 'not (polymer or water or ions) and noh', protein: 'str' = 'protein and noh', center: 'float' = 4.0, width: 'float' = 1.0, residues=None, periodic: 'bool' = True) -> 'Interactions'`

How near the ligand comes to each residue, per frame, softened to 0-1.

Every copy of the ligand -- one per molecule of the selection -- gives one
fingerprint per frame.  ``residues`` fixes the columns, which is what lets
fingerprints from different ligands, or different systems with the same
protein, be compared: pass the ``residues`` of an earlier result.
>>>>>>> interaction-fingerprints

### `boonza.kinetics(system, found, site: 'int', interval_ns: 'float', temperature: 'float' = 310.0, hysteresis: 'float' = 2.0, quantile: 'float' = 0.9, bootstrap: 'int' = 400, seed: 'int' = 0, volume_A3: 'float | None' = None) -> 'Rates'`

Rates, residence time and dG of one site, with an interval from resampling runs.

``interval_ns`` is the time between frames.  The free-ligand
concentration is counted per frame, from the copies not in the site and
the box volume, so it falls as copies bind.  The volume is the mean of
the boxes the frames actually had, which under a barostat is not the box
the structure file carries; ``volume_A3`` overrides it.  The interval comes from
resampling whole runs with replacement, which carries run-to-run
disagreement that a Poisson count cannot see; with one run it is the
dwells that are resampled instead.

Rates rest on completed events.  A dwell the trajectory cut short counts
its time and not its ending, so a run stopped early -- by the wall clock
or by ``--early-stop`` -- biases nothing.

### `boonza.ligand_centroids(system, positions=None, reference=None, ligand: 'str' = 'not (polymer or water or ions) and noh', align: 'str' = 'protein and name CA', periodic: 'bool' = True) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]'`

``(centroids (nframes ncopies, 3), (frame, copy) of each, the box volume of each)``.

The volumes come from the frames themselves, not from the system's stored
cell: under a barostat the box is not what the structure file says, and
the concentration a rate is measured against depends on it.

Each copy of the ligand -- one per molecule of the selection -- gives one
centroid per frame, taken in the copy's own periodic image and then moved
to the image nearest the protein.  The frame's ``align`` atoms are
superposed on the reference's, and the same transform is applied to the
centroid, so points from different runs live in one frame of reference.

### `boonza.ligand_features(system, ligand: 'str' = 'not (polymer or water or ions) and noh', families=('Donor', 'Acceptor', 'Aromatic', 'Hydrophobe', 'PosIonizable', 'NegIonizable')) -> 'list[list[tuple]]'`

Per ligand copy, ``[(family, atom indices), ...]`` as RDKit types them.

A feature is placed at the centre of its atoms, so an aromatic ring counts
once at the middle of the ring rather than six times around it.
``ZnBinder`` and ``LumpedHydrophobe`` are left out: the first is a special
case and the second repeats what ``Hydrophobe`` already says.

### `boonza.ligand_rmsd(mobile, reference, ligand: 'str' = 'not (polymer or water or ions) and noh', reference_ligand=None, fit: 'str' = 'protein and name CA and not resname NMA NME ACE', reference_fit=None, align: 'str | None' = 'order', positions=None, heavy_only: 'bool' = True, bond_orders: 'bool' = False, apply: 'bool' = False) -> 'LigandRMSD'`

Superpose ``mobile`` onto ``reference`` by protein atoms, then the ligand RMSD.

The docking-pose RMSD: ``fit`` atoms (C-alpha by default) are superposed,
the transform moves the whole mobile system (ligand included), and the
ligand RMSD is the symmetry-corrected, in-place RMSD between the ligand
atoms (``ligand``/``reference_ligand`` selections; by default everything
that is not protein, nucleic acid, water or ions, without hydrogens).
``align``: "order" pairs the fit atoms in order (same protein);
"sequence" uses ``boonza.matchmaker`` (different sequences or numbering);
None when the two are already in one frame.

``positions``: mobile coordinates instead of ``mobile.positions``: one
frame, an (nframes, natoms, 3) array, a Frames block or a Trajectory.
Every frame is fitted on its own and the results are arrays with one value
per frame (rotations (nframes, 3, 3)); trajectories are read chunk by
chunk, only the fit and ligand atoms.  With ``align="sequence"`` the
residue pairing comes from the first frame.  ``apply`` moves ``mobile``
(single structures only).

### `boonza.load_forcefield(name, path=None, require_rules: 'bool' = True) -> 'ViparrForcefield'`

Read a viparr force field: a directory, a name in ``path`` (default
``$VIPARR_FFPATH``), or a name in the viparr-ffpublic copy bundled with
boonza (see :func:`bundled_version`).

``require_rules=False`` reads a patch that has no ``rules`` file (for
:func:`merge_forcefields`).

### `boonza.load_openmm_forcefield(*files) -> 'OpenMMForcefield'`

Read OpenMM force field XML files, as ``openmm.app.ForceField(*files)`` does.

A name that is not a path is looked up in the OpenMM XML files bundled
with boonza (OpenMM 8.6.1, :func:`bundled_version`), then in the
installed OpenMM's data directories, so
``load_openmm_forcefield("amber19-all.xml", "amber19/opc.xml")`` reads the
same files whatever OpenMM is installed. ``<Include>`` files are followed.
``files`` of the result says where each came from (``bundled:`` names).

### `boonza.merge_forcefields(base, patch, append_only: 'bool' = False, path=None) -> 'ViparrForcefield'`

``base`` with ``patch`` merged in, as viparr's ``-m`` (or ``-a`` with
``append_only``) options do.

Templates of the patch replace templates of the same name; parameter rows
of the patch replace the rows of ``base`` with the same types, and rows
with new types go first; CMAP grids are replaced. Rules merge (plugins
and info are joined; functional forms and scale factors must agree).
``append_only`` refuses to replace anything. ``base`` and ``patch`` may
be force fields or names in ``path``; neither is modified.

### `boonza.native_contacts(system, sel1, sel2, positions=None, reference=None, radius: 'float' = 4.5, method: 'str' = 'hard_cut', beta: 'float' = 5.0, lambda_constant: 'float' = 1.8, periodic: 'bool' = True) -> 'np.ndarray'`

Fraction of native contacts Q in each frame (MDAnalysis Contacts).

Native contacts are the ``sel1`` x ``sel2`` atom pairs within ``radius``
Å in ``reference`` (a System with the same selections; default the
system's own coordinates), with reference distances r0.  ``method``:
"hard_cut" counts r <= r0, "radius_cut" counts r <= radius, "soft_cut"
is Best, Hummer and Eaton's 1 / (1 + exp(beta (r - lambda_constant r0)))
with beta in 1/Å.  With ``periodic``, distances use each frame's box (and
the reference cell).

### `boonza.neutralize(system: 'System', cation='Na', anion='Cl', charge='formal_charge', chain: 'str' = 'ION', chain2: 'str' = 'ION2', solute_pad: 'float' = 5.0, ion_pad: 'float' = 3.0, water_pad: 'float' = 0.0, concentration: 'float' = 0.0, keep: 'str' = 'none', random_seed: 'int' = 0) -> 'System'`

Replace water molecules with ions (msys ``dms-neutralize``).

Enough counterions are added to cancel the solute's charge (the sum of
``charge``: "formal_charge", "charge", or a number), then ion pairs up to
``concentration`` (mol/L, counted against the waters as msys does).
Waters within ``solute_pad`` of anything that is not water, or matching
``keep``, are never replaced; they are picked in a random order fixed by
``random_seed``.  As in msys, candidate waters closer than
``ion_pad`` squared (Å) to an earlier pick are skipped.  Ions sit at the
replaced water's mass-weighted center, in a new ct: counterions in chain
``chain`` and the others in ``chain2``, numbered from 1.  Unlike msys,
ions get their element mass.

### `boonza.parameterize(system: 'System', forcefields, *, rename_atoms: 'bool' = False, rename_residues: 'bool' = False, fix_masses: 'bool' = True, fatal: 'bool' = True, cmap_chirality: 'bool' = True, reorder_ids: 'bool' = True, constraints: 'bool' = True, path=None) -> 'System'`

A copy of ``system`` with a force field from viparr force fields.

``forcefields`` is a list of :class:`ViparrForcefield` objects, names in
``path`` (default ``$VIPARR_FFPATH``) or directories. Each molecule takes
its parameters from the first force field whose templates match all its
residues, so list the force field you trust most first; combine
force fields into one with :func:`merge_forcefields`.

The input's force field and pseudo particles are dropped; virtual sites
come from the templates. ``rename_atoms``/``rename_residues`` copy names
from the matched templates. ``fix_masses`` gives all atoms of an element
the median of their masses, as viparr does by default. ``fatal=False``
turns missing parameters into warnings. ``cmap_chirality=False`` applies
L CMAP grids to D residues as viparr does. Pseudo particles go
right after their parent atoms, so every residue's atoms stay together as
OpenMM requires (viparr's ``--reorder-ids``); ``reorder_ids=False``
appends them after all real atoms, as viparr does by default.
``constraints`` adds viparr's constraints (see :func:`build_constraints`),
as viparr does by default.

### `boonza.parameterize_openmm(system: 'System', forcefield, *, constraints=None, rigid_water=None, hydrogen_mass=None, residue_templates=None, ignore_external_bonds: 'bool' = False) -> 'System'`

A copy of ``system`` with the force field OpenMM's ``createSystem`` would build.

``forcefield`` is an :class:`OpenMMForcefield` or XML file names (as for
``openmm.app.ForceField``).  ``constraints`` is None, ``"hbonds"``,
``"allbonds"`` or ``"hangles"``; ``rigid_water`` (default: the templates'
choice, rigid) makes residues named HOH rigid; ``hydrogen_mass`` (amu)
repartitions mass onto hydrogens.  Constrained bonds and angles stay in
their tables, marked ``constrained``, and the constraints go to
``constraint_ahN``/``constraint_hoh`` tables.  ``residue_templates`` maps
residue indices to template names.

### `boonza.pca(system, positions=None, sel='name CA', align: 'bool' = True, n_components=None) -> 'PCA'`

Principal component analysis of the ``sel`` atoms over frames (MDAnalysis PCA).

With ``align`` every frame is first superposed on the first frame (as
MDAnalysis does); the covariance of the fitted coordinates is divided by
nframes - 1.  Components come from a singular value decomposition of the
centered frames, so at most min(nframes, 3 natoms) are returned (the rest
have zero variance), each with its largest element positive.

### `boonza.peptide(sequence: 'str', conformation='helix', seed: 'int' = 0, optimize: 'bool' = True) -> 'System'`

A peptide built from a one-letter sequence (through RDKit).

``conformation``: "helix", "sheet", "extended", "polyproline", one
(phi, psi) pair, or one pair per residue (degrees).  The chain is built
by RDKit with PDB atom and residue names, every peptide bond is set
trans, and phi/psi are set residue by residue (proline's phi is fixed by
its ring).  With ``optimize`` the structure is minimized with MMFF94
while phi/psi are held, so side chains relax without losing the
backbone (omega included).  Termini are free amine and acid, as RDKit
builds them.

### `boonza.plot_interactions(fingerprints, system, path, share: 'float' = 0.2, labels=None, order: 'bool' = True) -> 'bool'`

Draw what each ligand touches as a heatmap; False without matplotlib.

Rows are the fingerprints, columns the residues any of them comes near.
The colour is one hue from light to dark because the value is a magnitude:
a rainbow would invent boundaries where the data has none.  With ``order``
the rows are arranged so that ligands which agree sit together, which is
what makes two ways of binding one site visible as two blocks.

### `boonza.pocket_contacts(system, positions=None, ligand: 'str' = 'not (polymer or water or ions) and noh', protein: 'str' = 'protein and name CA', cutoff: 'float' = 5.0, periodic: 'bool' = True) -> 'tuple[np.ndarray, np.ndarray]'`

``(protein atoms in contact per frame, share of frames each is in contact)``.

One chunked pass, reading only the ligand and the candidate protein
atoms.  The per-frame count says whether the ligand is bound at all,
without caring which pose it is in; the per-atom share says which atoms a
pocket is really made of, over the whole run rather than in one frame.

### `boonza.pose_distances(system, positions=None, reference=None, ligand: 'str' = 'not (polymer or water or ions) and noh', protein: 'str' = 'protein and name CA', pocket_cutoff: 'float' = 5.0, pocket=None, symmetry: 'bool' = True, heavy_only: 'bool' = True, bond_orders: 'bool' = False, periodic: 'bool' = True) -> 'np.ndarray'`

The (nframes, nframes) dRMSD matrix of a trajectory, in A.

Every frame becomes the matrix of distances between the pocket atoms --
the ``protein`` atoms within ``pocket_cutoff`` A of the ligand in the
reference -- and the ligand atoms, and two frames are compared by the
RMS difference of those distances.

``reference``: a System (a crystal structure, or a frame where the ligand
is bound), or None for the first frame.  It decides which atoms the
pocket is made of, so a run that starts with the ligand elsewhere -- out
in bulk, or not yet settled -- wants one rather than its own first frame.
:func:`pocket_contacts` and :func:`bound_frame` find a fair one.

``pocket``: the pocket atoms themselves, as a selection or atom indices,
when you would rather say than have it worked out -- from the atoms a
site contacts over many runs, say.  ``protein``, ``pocket_cutoff`` and
the reference then do not enter into which atoms are used.

With ``symmetry`` each frame's ligand
atoms are first matched to the first frame's, so equivalent atoms do not
count as motion.  That mapping is chosen once per frame rather than once
per pair, which is what keeps this quadratic in frames but linear in
symmetry searches; when two frames would rather be compared through
different mappings the distance between them is an upper bound.

### `boonza.poses(system, positions=None, reference=None, cutoff: 'float' = 1.5, min_population: 'float' = 0.02, ligand: 'str' = 'not (polymer or water or ions) and noh', protein: 'str' = 'protein and name CA', pocket_cutoff: 'float' = 5.0, pocket=None, symmetry: 'bool' = True, heavy_only: 'bool' = True, bond_orders: 'bool' = False, periodic: 'bool' = True) -> 'PoseSet'`

The poses a trajectory holds, most populated first.

Frames within ``cutoff`` A dRMSD of one another (average linkage) are one
pose; a pose holding less than ``min_population`` of the frames, or fewer
than two of them, is left out of the list and its frames are labelled -1.
Each pose is reported by its medoid -- the member with the smallest mean
distance to the others -- so what comes back is always a frame that was
simulated.

### `boonza.repartition_hydrogen_masses(system: 'System', selection: 'str' = 'not water', mass: 'float' = 3.024, repartition: 'bool' = True) -> 'System'`

Set the mass of the hydrogens in ``selection`` (msys ``dms-hmr``).

With ``repartition`` the added mass is taken from the heavy atom each
hydrogen is bonded to, so the total mass is unchanged (hydrogen mass
repartitioning, for 4 fs time steps); without it only the hydrogens
change (for example deuterium, 2.014).

### `boonza.settled(values, min_frames: 'int' = 20) -> 'int'`

The frame a time series settles at: where to start to keep the most signal.

Every start is scored by how many independent samples it leaves --
remaining frames over :attr:`BlockAverage.statistical_inefficiency` -- and
the best is returned.  Discarding too little leaves the drift in; too
much throws away sampling, and this trades the two off rather than
guessing a percentage (Chodera 2016).  A series shorter than
``min_frames``, or one with no drift to speak of, settles at 0.

### `boonza.similarity(a, b) -> 'float'`

How alike two fingerprints are: 1 the same residues to the same degree, 0 none.

The cosine of the two, which asks about the pattern rather than how
deeply either sits -- a small fragment in a pocket and a large one
reaching further into it are alike here if they touch the same residues.

### `boonza.similarity_matrix(values) -> 'np.ndarray'`

Every fingerprint against every other, as cosines.

### `boonza.site_interactions(system, runs, found, site: 'int', ligand: 'str' = 'not (polymer or water or ions) and noh', protein: 'str' = 'protein and noh', center: 'float' = 4.0, width: 'float' = 1.0, periodic: 'bool' = True) -> 'Interactions'`

One fingerprint per ligand that visits ``site``: what each of them touches there.

Rows are ``(run, copy)`` rather than frames, each the average over that
copy's frames in the site.  The columns are the same residues for every
row, so two rows can be compared however unlike the two molecules are --
which is the question a pose cannot answer.

### `boonza.site_pocket(system, runs, found: 'SiteSet', k: 'int', protein: 'str' = 'protein and name CA', ligand: 'str' = 'not (polymer or water or ions) and noh', cutoff: 'float' = 5.0, share: 'float' = 0.5, periodic: 'bool' = True) -> 'np.ndarray'`

The atoms a site is made of: those the ligand touches in ``share`` of its frames.

A pocket from one frame is one frame's opinion.  This counts over every
frame assigned to the site, across runs and copies, so an atom earns its
place by being there for the ligand rather than by happening to be close
when the reference was taken.  Hand the result to :func:`boonza.poses` as
``pocket=`` and the pose level stops depending on a reference at all.

### `boonza.sites(system, runs=None, reference=None, ligand: 'str' = 'not (polymer or water or ions) and noh', align: 'str' = 'protein and name CA', spacing: 'float' = 1.0, enrichment: 'float' = 20.0, min_occupancy: 'float' = 0.005, periodic: 'bool' = True) -> 'SiteSet'`

Where the ligand is found across ``runs``, most occupied first.

``runs`` is one trajectory (or array of frames) or a list of them; each is
treated as independent evidence, and a site visited by several runs is a
claim several simulations agree on.  A site is a connected group of grid
cells the ligand visits at least ``enrichment`` times more often than
bulk solvent would explain; everything else is bulk, and is labelled -1
rather than forced into a site.  Sites below ``min_occupancy`` of the
pooled frames are left out.

### `boonza.solvate(solute: 'System', solvent=None, box=None, thickness: 'float' = 5.0, min_solute_dist: 'float' = 2.4, min_solvent_dist: 'float' = 1.0, solvent_selection: 'str' = 'oxygen', center_selection: 'str' = 'all', remove_buried: 'bool' = False) -> 'System'`

Tile a solvent box around ``solute`` and remove overlaps (msys ``dms-solvate``).

``remove_buried`` also drops solvent that ends up in a hydrophobic void
(between lipid tails, say), by :func:`remove_buried_water`.

``box``: the box edge lengths (one value or three, Å); by default a cube
of the solute's largest extent plus ``thickness`` on each side.  The
solute is first centered on ``center_selection`` ("none" to skip).
``solvent``: a System or file with a periodic cell (default: the
bundled TIP3P box).  Solvent molecules are removed when a
``solvent_selection`` atom is within ``min_solute_dist`` of the solute
(periodically), when their center lies outside the box, or when any of
their atoms is within ``min_solvent_dist`` of a periodic image of the
solvent.  Water chains are named W1, W2, ... with residues numbered from 1.

### `boonza.summarize(system, focus=None, cutoff: 'float' = 4.0, max_items: 'int' = 20, title: 'str | None' = None) -> 'Summary'`

A text summary of ``system`` (see the module docstring).

``focus``: a selection (string, indices or AtomSel) to describe in
detail: its surroundings within ``cutoff`` Å, bonds, polar contacts and
burial.  ``max_items`` caps every list; longer lists say how many more.

### `boonza.symmetry_rmsd(mobile, reference, atoms=None, reference_atoms=None, positions=None, superpose: 'bool' = False, heavy_only: 'bool' = True, bond_orders: 'bool' = False, max_isomorphisms: 'int' = 100000) -> 'SymmetryRMSD'`

Smallest RMSD (Å) over atom mappings that preserve elements and bonds.

``atoms`` selects the molecule in ``mobile`` (default: every atom) and
``reference_atoms`` in ``reference`` (default: the same selection).
``positions``: mobile coordinates to use instead of ``mobile.positions``:
one frame, an (nframes, natoms, 3) array, a Frames block or a Trajectory
(one RMSD per frame; trajectories are read chunk by chunk).
``superpose=False`` (default) compares coordinates as they are, as for a
docked pose in the protein frame; ``superpose=True`` fits each mapping
first (the RMSD between conformers).  ``heavy_only`` drops hydrogens and
pseudo particles.  ``bond_orders=True`` also requires bond orders to match.
Connectivity alone is the default, as in most docking benchmarks: with
Kekulé bond orders a flipped phenyl ring or a carboxylate's two oxygens
would no longer count as equivalent.

### `boonza.topological_distances(system, atoms=None, targets=None, max_distance=None) -> 'np.ndarray'`

Bonds along the shortest bond path, shape (len(atoms), len(targets)).

``atoms`` and ``targets`` are indices, AtomSels or selection strings
(``targets`` defaults to ``atoms``, ``atoms`` to every atom).  -1 marks
pairs that are not connected, or farther apart than ``max_distance``.

### `boonza.view(system, atoms=None, positions=None, style: 'str' = 'auto', width: 'int' = 640, height: 'int' = 480, water: 'bool' = False, background: 'str' = 'white', interval: 'int' = 100) -> 'MoleculeView'`

A notebook view of ``atoms`` (default all), displayed with 3Dmol.js.

``style="auto"`` draws polymers as cartoons, other molecules as sticks,
ions as spheres and hides water (``water=True`` shows it as lines);
"cartoon", "sticks", "lines" and "spheres" apply to everything.
``positions``: one frame or several ((nframes, natoms, 3) or Frames),
animated ``interval`` ms apart.

### `boonza.viparr`

### `boonza.wanted(spots, center, within: 'float' = 6.0) -> 'list[Hotspot]'`

The hotspots near a place, most agreed-upon first: what that pocket asks for.

### `boonza.write_forcefield(ff: 'ViparrForcefield', directory) -> 'Path'`

Write ``ff`` as a viparr force-field directory: ``rules`` (left out
for a patch without rules), ``templates`` and one file per parameter
table, which :func:`load_forcefield` and viparr read back.

### `boonza.write_hotspots(path, spots) -> 'None'`

Write the hotspots as pseudo-atoms: one per peak, named for what it wants.

``DON``, ``ACC``, ``ARO``, ``HYD``, ``CAT``, ``ANI``, with the enrichment
in the B-factor column and the tolerance radius as the occupancy, so a
viewer can size and colour them without being told anything else.
