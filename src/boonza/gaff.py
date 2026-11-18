"""GAFF2 templates for ligands, non-standard residues and covalent adducts.

AmberTools types each molecule or residue group that the force fields cannot
parameterize with GAFF2 and charges it with AM1-BCC (antechamber, sqm,
parmchk2 and tleap; no OpenFF).  :func:`gaff2_patch` turns the result into a
viparr patch force field, merged onto an Amber protein force field with
:func:`boonza.merge_forcefields` so that :func:`boonza.parameterize` treats
the new residues like any other template.

Residues bonded to a protein are cut at their peptide bonds and capped with
ACE/NME built from the neighbouring atoms.  Atoms of an amino acid keep the
protein force field's types and charges as far as each atom and its
neighbours match the parent residue, so backbone and CB terms, protein
impropers and CMAP come from the protein force field; every term that
touches a GAFF2 atom takes the value tleap gave it.  Each residue's charge
is brought to its formal charge by shifting its GAFF2 atoms evenly.

A pseudopeptide -- one ligand residue that is part peptide -- has no residues
to cut at.  With ``pseudopeptide`` on, the amino acids inside it are found by
their bonds (:meth:`_Builder._units`) and keep the protein force field in the
same way, CMAP included, while the molecule stays one residue and one template.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np

from .system import System
from .viparr import (
    ParamRow,
    Rules,
    Template,
    ViparrError,
    ViparrForcefield,
    ViparrWarning,
    _Parameterizer,
    load_forcefield,
)

__all__ = [
    "AmberToolsError",
    "describe_units",
    "find_amber_tools",
    "find_unmatched",
    "gaff2_patch",
    "run_gaff2",
]

#: GAFF2 releases and the AmberTools parameter files that may hold them, in
#: order: newer AmberTools ship 2.11 as gaff211.dat and a later GAFF2 as
#: gaff2.dat; older ones have only gaff2.dat, which is 2.11.
GAFF_FILES = {"2.11": ("gaff211.dat", "gaff2.dat"), "2.2": ("gaff2.dat",)}
_TOOLS = ("antechamber", "sqm", "parmchk2", "tleap")
_PLUGINS = ["exclusions", "mass", "bonds", "angles", "vdw1", "propers", "impropers"]
_VALENCE = {5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
_CH = 1.09  # C-H length of the hydrogens that close cap methyl groups (A)
_ZERO = {"phi0": 0.0, **{f"fc{k}": 0.0 for k in range(7)}}


class AmberToolsError(RuntimeError):
    """AmberTools is missing or one of its programs failed."""


def find_amber_tools(amberhome=None) -> Path:
    """The AmberTools installation: ``amberhome``, else ``$AMBERHOME``, else
    the one whose ``antechamber`` is on ``PATH``."""
    if amberhome is not None:
        home = Path(amberhome)
    elif os.environ.get("AMBERHOME"):
        home = Path(os.environ["AMBERHOME"])
    else:
        exe = shutil.which("antechamber")
        home = Path(exe).parent.parent if exe else None
    if home is not None and all((home / "bin" / t).exists() for t in _TOOLS):
        return home
    raise AmberToolsError(
        "AmberTools (antechamber, sqm, parmchk2, tleap) was not found: install it with "
        "'conda install -c conda-forge ambertools' and activate that environment, or set AMBERHOME"
    )


def _gaff_version(path: Path) -> str:
    """The version on a GAFF parameter file's title line, e.g. "2.11"."""
    with path.open(errors="replace", encoding="utf-8") as fh:
        title = fh.readline()
    words = title.split("Version", 1)[1].split() if "Version" in title else []
    return words[0].rstrip(",)") if words else "unknown"


def gaff_parameters(home, gaff: str = "2.11") -> Path:
    """The parameter file of the ``gaff`` release in the AmberTools at
    ``home``, recognized by the version on its title line."""
    if gaff not in GAFF_FILES:
        raise ValueError(f"gaff must be one of {sorted(GAFF_FILES)}")
    parm = Path(home) / "dat" / "leap" / "parm"
    found = []
    for name in GAFF_FILES[gaff]:
        path = parm / name
        if path.is_file():
            version = _gaff_version(path)
            if version == gaff or version.startswith(gaff + "."):
                return path
            found.append(f"{name} is version {version}")
    have = "; ".join(found) if found else f"no {' or '.join(GAFF_FILES[gaff])}"
    raise AmberToolsError(
        f"GAFF {gaff} parameters are not in {parm} ({have}): use an AmberTools that has "
        "them ('conda install -c conda-forge ambertools')"
    )


def _run(cmd: list[str], cwd: Path, env: dict, output: str) -> str:
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8")
    log = proc.stdout + proc.stderr
    if proc.returncode != 0 or not (cwd / output).exists():
        tail = "\n".join(log.strip().splitlines()[-20:])
        raise AmberToolsError(f"{Path(cmd[0]).name} failed in {cwd}:\n{tail}")
    return log


#: where AM1-BCC charges are computed: "optimized", the molecule as given then optimized
#: by sqm (AmberTools' default, minutes for a drug-sized one); "reference", one AM1
#: calculation at a conformer built from the bonds alone (:func:`reference_conformer`):
#: under a second, and a molecule has the same charges in every pose
CHARGE_GEOMETRIES = ("optimized", "reference")
#: sqm without its geometry optimization: the AM1 calculation alone
SINGLE_POINT = "qm_theory='AM1', maxcyc=0, scfconv=1.d-8"
#: run_gaff2's results at reference geometries, by molecule and settings
_REFERENCE_RUNS: dict = {}


def reference_conformer(s: System) -> System:
    """``s`` at a conformer made from its bonds alone (RDKit ETKDG with a fixed
    seed, then MMFF94 where it has parameters), atoms in order: the same for every
    pose of the molecule.  Stereochemistry is read from ``s``'s coordinates.  When
    RDKit cannot embed it, an error: falling back on ``s``'s own geometry would charge
    whatever pose it is in (a backmapped one, at worst)."""
    from rdkit.Chem import AllChem

    from .chem import to_rdkit

    mol = to_rdkit(s, sanitize=True, implicit_hydrogens=False)  # stereo from s's 3D
    params = AllChem.ETKDGv3()
    params.randomSeed = 20251027
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True  # large or strained molecules
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise AmberToolsError(
                f"RDKit could not build a reference conformer of {s.natoms} atoms for "
                "its AM1-BCC charges; charge_geometry='optimized' charges it as it is"
            )
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    out = s.clone()
    out.positions = mol.GetConformer().GetPositions()
    return out


def _graph_key(s: System, *settings) -> tuple:
    """A molecule as its atoms (element, formal charge) and bonds, in order."""
    bonds = sorted(zip(s.bonds["i"].tolist(), s.bonds["j"].tolist(), s.bonds["order"].tolist(),
                       strict=True))  # fmt: skip
    return (tuple(s.atoms["anum"].tolist()), tuple(s.atoms["formal_charge"].tolist()),
            tuple(bonds), *settings)  # fmt: skip


def run_gaff2(fragment: System, net_charge: int, *, gaff: str = "2.11",
              charge_method: str = "bcc", charge_geometry: str = "optimized",
              amberhome=None, workdir=None) -> System:  # fmt: skip
    """GAFF2 parameters for one molecule with all its hydrogens, from AmberTools.

    antechamber types the atoms and computes charges (``charge_method``,
    AM1-BCC by default, for a molecule of ``net_charge``, at the geometry
    ``charge_geometry`` says: :data:`CHARGE_GEOMETRIES`), parmchk2 writes
    every parameter the molecule uses from the ``gaff`` release, and tleap
    builds the topology.  Returns the tleap prmtop read as a system, atoms
    in the order of ``fragment``.  ``workdir`` keeps the AmberTools files.
    At reference geometries a molecule is run once (kept by its atoms and bonds).
    """
    if charge_geometry not in CHARGE_GEOMETRIES:
        raise ValueError(f"charge_geometry must be one of {CHARGE_GEOMETRIES}")
    if charge_geometry == "reference":
        key = _graph_key(fragment, int(net_charge), gaff, charge_method)
        if key not in _REFERENCE_RUNS or workdir is not None:
            at = reference_conformer(fragment)
            _REFERENCE_RUNS[key] = _run_gaff2(at, net_charge, gaff, charge_method, SINGLE_POINT,
                                              amberhome, workdir)  # fmt: skip
        return _REFERENCE_RUNS[key].copy()
    return _run_gaff2(fragment, net_charge, gaff, charge_method, None, amberhome, workdir)


def _run_gaff2(fragment, net_charge, gaff, charge_method, sqm, amberhome, workdir) -> System:
    from .io.amber import load_prmtop
    from .io.sdf import save_sdf

    if gaff not in GAFF_FILES:
        raise ValueError(f"gaff must be one of {sorted(GAFF_FILES)}")
    home = find_amber_tools(amberhome)
    parm = gaff_parameters(home, gaff)
    env = {**os.environ, "AMBERHOME": str(home)}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(workdir) if workdir is not None else Path(tmp)
        d.mkdir(parents=True, exist_ok=True)
        save_sdf(fragment, d / "in.sdf")
        bin_ = home / "bin"
        _run([str(bin_ / "antechamber"), "-i", "in.sdf", "-fi", "sdf", "-o", "out.mol2",
              "-fo", "mol2", "-at", "gaff2", "-c", charge_method, "-nc", str(int(net_charge)),
              "-rn", "MOL", "-pf", "y", "-dr", "no", *(["-ek", sqm] if sqm else [])],
             d, env, "out.mol2")  # fmt: skip
        shutil.copy(parm, d / "gaff.dat")
        _run([str(bin_ / "parmchk2"), "-i", "out.mol2", "-f", "mol2", "-p", "gaff.dat",
              "-s", "2", "-a", "Y", "-o", "out.frcmod"], d, env, "out.frcmod")  # fmt: skip
        (d / "leap.in").write_text(
            "source leaprc.gaff2\nloadamberparams gaff.dat\nm = loadmol2 out.mol2\n"
            "loadamberparams out.frcmod\nsaveamberparm m out.prmtop out.inpcrd\nquit\n",
            encoding="utf-8",
        )
        log = _run([str(bin_ / "tleap"), "-f", "leap.in"], d, env, "out.prmtop")
        if "Errors = 0" not in log:
            raise AmberToolsError(f"tleap reported errors in {d}:\n{log[-2000:]}")
        p = load_prmtop(d / "out.prmtop")
    if p.natoms != fragment.natoms or (p.atoms["anum"] != fragment.atoms["anum"]).any():
        raise AmberToolsError("antechamber changed the atoms of the molecule")
    return p


class GroupError(ViparrError):
    """A group of residues that could not be given GAFF2 parameters: ``residues`` and
    ``atoms`` are its indices in the system, for a caller to show which it was."""

    def __init__(self, message: str, residues, atoms):
        super().__init__(message)
        self.residues = [int(r) for r in residues]
        self.atoms = [int(a) for a in atoms]


def find_unmatched(system: System, forcefields, path=None) -> list[list[int]]:
    """Residue groups that ``forcefields`` cannot parameterize.

    A group is a set of bonded residues without a matching template, plus the
    residues bonded to them other than through a peptide bond or disulfide
    (the residue a covalent ligand is bound to).  Returns residue indices.
    """
    return _Builder(system, _load(forcefields, path)).groups()


def gaff2_patch(system: System, forcefields=(), *, charges=None, parents=None,
                gaff: str = "2.11", charge_method: str = "bcc",
                charge_geometry: str = "optimized", amberhome=None,
                workdir=None, path=None, protein_extent: str = "matched",
                draw=None, tag=None, pseudopeptide: str = "off",
                pdf=None) -> ViparrForcefield:  # fmt: skip
    """A viparr patch with GAFF2 templates for what ``forcefields`` cannot parameterize.

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

    ``pseudopeptide`` (:data:`PSEUDOPEPTIDE`) looks for amino acids inside a
    residue that is not one itself: a ligand that is part peptide, drawn or
    read as one molecule.  Each N-CA-C(=O) in it is compared with the protein
    force field's own templates.  A *residue match* is a whole template found
    atom for atom, hydrogens and all (ALA, LYS, a charged N-terminal NALA...),
    and with it the ACE, NME or NHE cap bonded to it; a *backbone match* is
    what is left where the side chain is not a standard one: the backbone on
    alanine's types, glycine's (no side chain) or proline's (a ring back to
    N), the side chain GAFF2.  Matched atoms take the protein types, so every
    bond, angle, torsion and improper among them and the CMAP of each matched
    N-CA-C are the protein force field's; terms that touch a GAFF2 atom are
    GAFF2's.  "protein" copies the template charges as well and brings the
    residue back to its formal charge by shifting the GAFF2 atoms evenly;
    "types" keeps AM1-BCC charges on every atom.  Such a molecule is charged
    at its reference conformer whatever ``charge_geometry`` says.  The
    patch's ``units`` attribute lists what was found, and ``pdf`` (a file) or
    ``draw`` (a directory, ``pseudopeptide_<residue>.pdf``) gets a 2D drawing
    of the molecule on a transparent page, the atoms with protein parameters
    on yellow.
    """
    if protein_extent not in PROTEIN_EXTENTS:
        raise ValueError(f"protein_extent must be one of {PROTEIN_EXTENTS}")
    if pseudopeptide not in PSEUDOPEPTIDE:
        raise ValueError(f"pseudopeptide must be one of {PSEUDOPEPTIDE}")
    ffs = _load(forcefields, path)
    b = _Builder(system, ffs, charges or {}, parents or {},
                 dict(gaff=gaff, charge_method=charge_method, charge_geometry=charge_geometry,
                      amberhome=amberhome),
                 None if workdir is None else Path(workdir), protein_extent,
                 None if draw is None else Path(draw))  # fmt: skip
    b.tag = None if tag is None else str(tag)
    b.pseudopeptide = pseudopeptide
    b.pdf = None if pdf is None else Path(pdf)
    return b.patch()


def only_gaff2(args, command: str) -> None:
    """Refuse a ligand force field other than GAFF2 where ``command`` builds its
    ligands with GAFF2 alone, rather than build them with GAFF2 and say nothing."""
    ligandff = str(getattr(args, "ligandff", None) or "gaff-2.11")
    if not ligandff.startswith("gaff"):
        raise ValueError(f"{command} builds its ligands with GAFF2: ligandff = '{ligandff}' "
                         "is boonza md's and boonza parameterize's so far")  # fmt: skip


def describe_units(patch) -> list[str]:
    """What a patch's pseudopeptides matched, a line per ligand residue and one per
    drawing: for a log."""
    by: dict[tuple, list[dict]] = {}
    for u in getattr(patch, "units", ()):
        by.setdefault((u["residue"], u["chain"]), []).append(u)
    out = []
    for (residue, chain), units in by.items():
        kinds = {"residue": [], "backbone": [], "cap": []}
        for u in units:
            kinds[u["match"]].append(u["template"])
        parts = [f"{len(v)} {k} match{'es' if len(v) > 1 else ''} ({' '.join(v)})"
                 for k, v in kinds.items() if v]  # fmt: skip
        heavy = sum(len(u["atoms"]) for u in units)
        where = f" ({chain.strip()})" if chain.strip() else ""
        out.append(f"{residue}{where} pseudopeptide: {', '.join(parts)}; {heavy} heavy atoms "
                   "take the protein force field (AM1-BCC at the reference conformer)")  # fmt: skip
    for d in getattr(patch, "drawings", ()):
        if str(d).endswith(".pdf"):
            out.append(f"Pseudopeptide drawn in {Path(d).name} (yellow: protein parameters)")
    return out


#: Standard parents of common modified residues, from the PDB's chemical
#: component dictionary (``mon_nstd_parent_comp_id``).
KNOWN_PARENTS = {
    "MSE": "MET", "SEP": "SER", "TPO": "THR", "PTR": "TYR", "CSO": "CYS", "CSD": "CYS",
    "OCS": "CYS", "CME": "CYS", "SNC": "CYS", "CSS": "CYS", "HYP": "PRO", "MLY": "LYS",
    "M3L": "LYS", "MLZ": "LYS", "ALY": "LYS", "KCX": "LYS", "LLP": "LYS", "NLE": "LEU",
    "ABA": "ALA", "AIB": "ALA", "CGU": "GLU", "HIC": "HIS", "MHS": "HIS",
    # boonza view writes SGA (a sulfono-gamma-AA unit, two residues become one):
    # it stands on glycine's backbone, and the rest of it gets GAFF2
    "SGA": "GLY",
}  # fmt: skip
_VARIANTS = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
             "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS"}  # fmt: skip
_STANDARD = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
             "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HYP", *_VARIANTS}  # fmt: skip


def _family(name: str) -> str:
    """The residue a template or residue name stands for: ALA for NALA and
    CALA, HIS for HID, HIE and HIP, and so on."""
    n = str(name).upper()
    if len(n) == 4 and n[0] in "NC" and n[1:] in _STANDARD:
        n = n[1:]
    return _VARIANTS.get(n, n)


#: How far amino-acid atoms keep protein types (see :func:`gaff2_patch`).
PROTEIN_EXTENTS = ("matched", "cb")
_UP_TO_CB = {"N", "H", "H1", "H2", "H3", "HN", "CA", "HA", "HA2", "HA3", "C", "O", "OXT", "CB"}
#: Amino acids inside a ligand residue (see :func:`gaff2_patch`): "off"; "protein",
#: their protein types and charges; "types", their protein types under AM1-BCC charges.
PSEUDOPEPTIDE = ("off", "protein", "types")
_BACKBONES = ("ALA", "GLY", "PRO")  # the backbones a non-standard side chain stands on
_BACKBONE = _UP_TO_CB - {"CB"}
_CAPS = ("ACE", "NME", "NHE")
_REACH = 9  # bonds from CA within which a unit's atoms are looked for (ARG: 7)
#: the yellow under atoms with protein parameters in a pseudopeptide's drawing
YELLOW = (1.0, 0.9, 0.2)


class _Conflict(ViparrError):
    """Two terms with the same types and different GAFF2 parameters: ``atoms`` and
    ``first`` are the fragment atoms of the term refused and of the one kept."""

    def __init__(self, message, atoms, first):
        super().__init__(message)
        self.atoms, self.first = atoms, first


def _up_to_cb(t: Template, ti: int) -> bool:
    """Whether template atom ``ti`` is a backbone atom, CB, or a hydrogen on CB."""
    if t.names[ti] in _UP_TO_CB:
        return True
    return t.anum[ti] == 1 and any(t.names[j] == "CB" for j in t._nbrs[ti])


def host_index(forcefields) -> int:
    """The index of the force field GAFF2 patches join: the first one with
    amino-acid templates (N, CA and C), else the first."""
    for k, ff in enumerate(forcefields):
        if any({"N", "CA", "C"} <= set(t.names) for t in ff.templates):
            return k
    return 0


def _load(forcefields, path) -> list[ViparrForcefield]:
    if isinstance(forcefields, str | os.PathLike | ViparrForcefield):
        forcefields = [forcefields]
    return [ff if isinstance(ff, ViparrForcefield) else load_forcefield(ff, path)
            for ff in forcefields]  # fmt: skip


def _check_host(ff: ViparrForcefield) -> None:
    r = ff.rules
    ok = (r.vdw_func == "lj12_6_sig_epsilon" and r.vdw_comb_rule == "arithmetic/geometric"
          and len(r.es_scale) == 3 and r.es_scale[:2] == [0.0, 0.0] == r.lj_scale[:2]
          and abs(r.es_scale[2] - 1 / 1.2) < 1e-3 and r.lj_scale[2] == 0.5)  # fmt: skip
    if not ok:
        raise ViparrError(
            f"{Path(ff.name).name} is not an Amber force field (12-6 LJ, Lorentz-Berthelot, "
            "1-4 scales 1/1.2 and 1/2); GAFF2 cannot join it"
        )


class _Builder:
    def __init__(self, system, ffs, charges=None, parents=None, run=None, workdir=None,
                 extent="matched", draw=None):  # fmt: skip
        self.extent, self.draw = extent, draw
        self.tag: str | None = None
        self.pseudopeptide = "off"
        self.pdf: Path | None = None
        self.units: list[dict] = []  # the amino acids found inside ligand residues
        self.row_atoms: dict[tuple, tuple] = {}
        self.drawings: list[Path] = []
        self.P = P = _Parameterizer(system, ffs, False, False, True)
        self.ffs = ffs
        self.host = ffs[host_index(ffs)] if ffs else None
        if self.host is not None:
            _check_host(self.host)
        self.charges, self.parents = charges or {}, parents or {}
        self.run_opts, self.workdir = run or {}, workdir
        # parents the file names (PDB MODRES, mmCIF _pdbx_struct_mod_residue)
        res = P.s.residues
        self.file_parents = ([str(x) for x in res["parent"].tolist()] if "parent" in res.props
                             else [""] * P.s.nresidues)  # fmt: skip
        self._bbs: dict[int, dict | None] = {}
        self.chosen: list[dict] = []  # the parent each amino acid got, and why
        # residues pulled into a group only to move a cut away from a GAFF2 atom: they
        # keep their own force field's types, charges and template
        self.grown: set[int] = set()
        s = P.s
        self.anum, self.nbrs, self.names = P.anum, P.nbrs, P.names
        self.residue, self.resnames = P.residue, P.resnames
        self.resid = s.residues["resid"].tolist()
        self.pos = P.pos
        self.formal = s.atoms["formal_charge"].tolist()
        self.res_atoms: list[list[int]] = [[] for _ in range(s.nresidues)]
        for a, r in enumerate(self.residue):
            if self.anum[a] > 0:
                self.res_atoms[r].append(a)
        self.typed: dict[int, dict[int, tuple[str, str]]] = {}
        self.templates: list[Template] = []
        self.rows: dict[str, dict[str, list[dict]]] = {}
        self.memo: dict[str, str] = {}
        # per group: (the system atom of each fragment atom, the tleap prmtop)
        self.runs: list[tuple[list[int], System]] = []

    # -- which residues need templates ---------------------------------------------
    def _label(self, a: int) -> str:
        r = self.residue[a]
        return f"{self.resnames[r]}{self.resid[r]}:{self.names[a]}"

    def _peptide(self, i: int, j: int) -> bool:
        """A peptide bond: the backbone C of one residue to the backbone N of
        another, found by name or by structure."""
        if {self.anum[i], self.anum[j]} != {6, 7}:
            return False
        n, c = (i, j) if self.anum[i] == 7 else (j, i)
        return self._backbone_atom(n, "N") and self._backbone_atom(c, "C")

    def _link(self, i: int, j: int) -> bool:
        """A peptide bond or a disulfide: the bonds protein templates expect."""
        return self._peptide(i, j) or (
            self.anum[i] == self.anum[j] == 16 and self.names[i] == self.names[j] == "SG"
        )

    def _keeps_types(self, r: int) -> set[int] | None:
        """Atoms of residue ``r`` that keep their parent template's protein types, or
        None where the parent cannot be told (:meth:`_group` raises about that)."""
        try:
            t, m, ok, _ = self._parent(r)
        except ViparrError:
            return None
        if self.extent == "cb" and t is not None:
            ok = {a for a in ok if _up_to_cb(t, m[a])}
        return ok

    def _matched(self, r: int) -> dict[int, tuple[str, str]] | None:
        """Types of residue ``r``'s atoms from the first force field matching it."""
        if r not in self.typed:
            self.typed[r] = None
            for k, ff in enumerate(self.ffs):
                tpl, pairs = self.P._find(ff, k, self.res_atoms[r])
                if tpl is not None:
                    self.typed[r] = {a: (tpl.btype[ti], tpl.nbtype[ti]) for ti, a in pairs}
                    break
        return self.typed[r]

    def groups(self) -> list[list[int]]:
        todo = {r for r, atoms in enumerate(self.res_atoms) if atoms and self._matched(r) is None}
        changed = True
        while changed:  # residues bound to a new one by anything but a protein link join it
            changed = False
            for r in list(todo):
                for a in self.res_atoms[r]:
                    for b in self.nbrs[a]:
                        rb = self.residue[b]
                        if self.anum[b] > 0 and rb not in todo and not self._link(a, b):
                            todo.add(rb)
                            changed = True
        changed = True
        while changed:
            # A protein link is cut and capped, and the terms across the cut come from
            # the host force field, which has them only for its own types.  Where the
            # atom at the cut takes a GAFF2 type instead -- an N-alkylated backbone
            # nitrogen, say, which no amino-acid template matches -- the group grows
            # across the link, so the cut lands on a residue that does match its parent
            # and the capped bond keeps protein types on both sides.
            changed = False
            for r in list(todo):
                keep = self._keeps_types(r)
                if keep is None:
                    continue
                for a in self.res_atoms[r]:
                    if a in keep:
                        continue
                    for b in self.nbrs[a]:
                        rb = self.residue[b]
                        if self.anum[b] > 0 and rb not in todo and self._link(a, b):
                            todo.add(rb)
                            self.grown.add(rb)
                            changed = True
            # A matched residue linked to the group through two of its own atoms (one
            # amino acid between an N-terminal acyl cap and the rest of a macrocycle,
            # say) would be capped from both sides at once, its atoms closed with
            # hydrogens toward bonds that are kept: it joins the group instead.
            for rb in {self.residue[b] for r in todo for a in self.res_atoms[r]
                       for b in self.nbrs[a]} - todo:  # fmt: skip
                ends = {b for b in self.res_atoms[rb] if self.anum[b] > 0
                        and any(self.residue[a] in todo for a in self.nbrs[b])}  # fmt: skip
                if len(ends) > 1:
                    todo.add(rb)
                    self.grown.add(rb)
                    changed = True
        parent = {r: r for r in todo}

        def root(r):
            while parent[r] != r:
                parent[r] = parent[parent[r]]
                r = parent[r]
            return r

        for r in todo:
            for a in self.res_atoms[r]:
                for b in self.nbrs[a]:
                    if self.residue[b] in todo:
                        parent[root(self.residue[b])] = root(r)
        out: dict[int, list[int]] = {}
        for r in sorted(todo):
            out.setdefault(root(r), []).append(r)
        return list(out.values())

    # -- parent residue templates ------------------------------------------------
    def _bb(self, r: int) -> dict | None:
        """The backbone atoms {"N", "CA", "C", "O"} of residue ``r``: by name,
        else by structure (N-CA-C(=O) with a peptide bond to a neighbour)."""
        if r in self._bbs:
            return self._bbs[r]
        atoms = self.res_atoms[r]
        inside = set(atoms)
        anum, nbrs = self.anum, self.nbrs
        by: dict[str, int] = {}
        for a in atoms:
            by.setdefault(str(self.names[a]), a)
        n, ca, c = by.get("N"), by.get("CA"), by.get("C")
        found = None
        if (n is not None and ca is not None and c is not None
                and (anum[n], anum[ca], anum[c]) == (7, 6, 6)
                and ca in nbrs[n] and c in nbrs[ca]):  # fmt: skip
            found = (n, ca, c)
        else:
            options = []
            for n in atoms:
                if anum[n] != 7:
                    continue
                for ca in nbrs[n]:
                    if ca not in inside or anum[ca] != 6:
                        continue
                    for c in nbrs[ca]:
                        if (c == n or c not in inside or anum[c] != 6
                                or self._carbonyl_o(c, inside) is None):  # fmt: skip
                            continue
                        links = sum(1 for x in nbrs[n] if x not in inside and anum[x] == 6)
                        links += sum(1 for x in nbrs[c] if x not in inside and anum[x] == 7)
                        if links:
                            options.append((links, n, ca, c))
            most = max((o[0] for o in options), default=0)
            top = [o for o in options if o[0] == most]
            if len(top) == 1:  # several: not an amino acid we can place
                found = top[0][1:]
        out = None
        if found is not None:
            n, ca, c = found
            o = by.get("O")
            if o is None or anum[o] != 8 or o not in nbrs[c]:
                o = self._carbonyl_o(c, inside)
            out = {"N": n, "CA": ca, "C": c, "O": o}
        self._bbs[r] = out
        return out

    def _carbonyl_o(self, c: int, inside) -> int | None:
        return next((o for o in self.nbrs[c] if o in inside and self.anum[o] == 8
                     and sum(1 for x in self.nbrs[o] if self.anum[x] > 1) == 1), None)  # fmt: skip

    def _backbone_atom(self, a: int, key: str) -> bool:
        """Whether ``a`` is the backbone N or C (``key``) of its residue; caps
        such as ACE and NME count by name."""
        if self.names[a] == key:
            return True
        bb = self._bb(self.residue[a])
        return bb is not None and bb[key] == a

    def _in_chain(self, r: int) -> bool:
        """Whether residue ``r`` has a peptide bond to a neighbour."""
        bb = self._bb(r)
        if bb is None:
            return False
        inside = set(self.res_atoms[r])
        return any(self._peptide(a, b) for a in (bb["N"], bb["C"])
                   for b in self.nbrs[a] if b not in inside and self.anum[b] > 0)  # fmt: skip

    def _parent_source(self, r: int) -> tuple[str, str | None]:
        """Where the parent of residue ``r`` comes from, and its name: given,
        from the file (MODRES), known (the PDB's dictionary), or guessed."""
        name = str(self.resnames[r])
        for source, value in (("given", self.parents.get(name)), ("file", self.file_parents[r]),
                              ("known", KNOWN_PARENTS.get(name))):  # fmt: skip
            if value:
                return source, str(value)
        return "guessed", None

    def _amino(self, r: int) -> bool:
        """An amino acid: a backbone, and a place in a chain or a parent named for it."""
        return self._bb(r) is not None and (
            self._in_chain(r) or self._parent_source(r)[0] != "guessed"
        )

    def _cb(self, r: int, bb: dict) -> int | None:
        inside = set(self.res_atoms[r])
        side = [a for a in self.nbrs[bb["CA"]]
                if a in inside and self.anum[a] > 1 and a not in (bb["N"], bb["C"])]  # fmt: skip
        return side[0] if len(side) == 1 else None

    def _parent(self, r: int):
        """(template, {atom: template atom}, atoms that keep its types, source).

        The parent is the one given, else the file's, else the PDB
        dictionary's, else its own name when that is a standard amino acid;
        among its templates (termini, protonation states) the one matching
        most atoms wins. With none of those, every amino-acid template is
        tried, and the guess must be unambiguous and cover the backbone and
        CB, or this raises.
        """
        if self.host is None or not self._amino(r):
            return None, {}, set(), None
        bb = self._bb(r)
        name = str(self.resnames[r])
        label = f"{name}{self.resid[r]}"
        source, value = self._parent_source(r)
        if value is None and _family(name) in _STANDARD:
            # a residue called ALA names its own parent, as surely as MODRES or the PDB
            # dictionary would: the atoms that still match alanine keep their protein
            # types and the rest gets GAFF2.  Guessing from every template instead, and
            # then demanding the guess cover the backbone and CB, refused a residue
            # whose own name said what it was
            source, value = "its name", name
        amino = [t for t in self.host.templates if {"N", "CA", "C"} <= set(t.names)]
        if value is not None:
            candidates = [t for t in amino if _family(t.name) == _family(value)]
            if not candidates:
                raise ViparrError(f"{label}: its parent {value!r} ({source}) is not an amino acid "
                                  f"of {Path(self.host.name.split('__')[0]).name}")  # fmt: skip
        else:
            candidates = amino
        atoms = self.res_atoms[r]
        scored = []
        for t in candidates:
            m = self._map(r, t, bb)
            ok = self._fits(atoms, t, m)
            heavy = sum(1 for a in ok if self.anum[a] > 1)
            agree = sum(1 for a, i in m.items() if self.names[a] == t.names[i])
            key = (heavy, len(ok), t.name == name, _family(t.name) == name, agree, -t.natoms)
            scored.append((key, t, m, ok))
        scored.sort(key=lambda x: x[0], reverse=True)
        key, t, m, ok = scored[0]
        if source == "guessed":
            hint = (
                f'give it: parents = {{ {name} = "..." }} in the settings, '
                f"--parent {name}=... on the command line, or parents={{{name!r}: ...}}"
            )
            rivals = sorted({_family(x[1].name) for x in scored if x[0][0] == key[0]})
            if len(rivals) > 1 and name not in rivals:
                raise ViparrError(f"cannot tell the parent residue of {label}: {', '.join(rivals)} "
                                  f"match it equally well; {hint}")  # fmt: skip
            core = [bb["N"], bb["CA"], bb["C"], bb["O"], self._cb(r, bb)]
            if any(a is not None and a not in ok for a in core):
                raise ViparrError(
                    f"cannot tell the parent residue of {label}: the best match, "
                    f"{t.name}, does not cover its backbone and CB; {hint}"
                )
        if not ok:
            return None, {}, set(), None
        return t, m, ok, source

    def _map(self, r: int, t: Template, bb: dict, atoms=None) -> dict[int, int]:
        """Atoms of residue ``r`` (or ``atoms``, some of it) to atoms of ``t``:
        the backbone, then the most heavy atoms bonded as in ``t`` (equal names
        preferred; names are not needed), and each hydrogen after its atom."""
        atoms = self.res_atoms[r] if atoms is None else atoms
        inside = set(atoms)
        anum, nbrs, names = self.anum, self.nbrs, self.names
        where = {nm: i for i, nm in enumerate(t.names) if t.anum[i] > 0}
        m: dict[int, int] = {}
        for key in ("N", "CA", "C", "O"):
            a, i = bb.get(key), where.get(key)
            if a is not None and i is not None and t.anum[i] == anum[a]:
                m[a] = i
        if len(m) < 3:
            return {}
        tadj = [set(x) for x in t._nbrs]
        order, seen, queue = [], set(m), list(m)
        while queue:  # heavy atoms outward from the backbone
            a = queue.pop(0)
            for b in nbrs[a]:
                if b in inside and anum[b] > 1 and b not in seen:
                    seen.add(b)
                    order.append(b)
                    queue.append(b)
        used = set(m.values())
        best = [dict(m), sum(names[a] == t.names[i] for a, i in m.items())]
        budget = [20000]

        def extend(k: int, agree: int) -> None:
            budget[0] -= 1
            if budget[0] < 0 or len(m) + len(order) - k < len(best[0]):
                return
            if k == len(order):
                if (len(m), agree) > (len(best[0]), best[1]):
                    best[0], best[1] = dict(m), agree
                return
            a = order[k]
            mapped = [b for b in nbrs[a] if b in m]
            if mapped:
                bonded = set(mapped)
                for i in sorted(tadj[m[mapped[0]]], key=lambda i: t.names[i] != names[a]):
                    if i in used or t.anum[i] != anum[a]:
                        continue
                    if not all(m[b] in tadj[i] for b in mapped):
                        continue
                    if any(i in tadj[j] for b, j in m.items() if b not in bonded):
                        continue
                    m[a] = i
                    used.add(i)
                    extend(k + 1, agree + (t.names[i] == names[a]))
                    del m[a]
                    used.discard(i)
            extend(k + 1, agree)  # or leave ``a`` out: a modification

        extend(0, best[1])
        m = best[0]
        used = set(m.values())
        for a, i in list(m.items()):
            hs = sorted(b for b in nbrs[a] if b in inside and anum[b] == 1)
            ths = [j for j in t._nbrs[i] if t.anum[j] == 1 and j not in used]
            for b in list(hs):
                j = next((j for j in ths if t.names[j] == names[b]), None)
                if j is not None:
                    m[b] = j
                    used.add(j)
                    hs.remove(b)
                    ths.remove(j)
            for b, j in zip(hs, ths, strict=False):
                m[b] = j
                used.add(j)
        return m

    def _fits(self, atoms, t: Template, m: dict[int, int]) -> set[int]:
        """Mapped atoms whose bonded neighbours are exactly the template atom's;
        a hydrogen keeps its protein type only with its atom."""
        inside = set(atoms)
        ok = set()
        for a, i in m.items():
            t_in = [j for j in t._nbrs[i] if t.anum[j] > 0]
            t_out = [j for j in t._nbrs[i] if t.anum[j] == -1]
            mine_in = [b for b in self.nbrs[a] if b in inside and self.anum[b] > 0]
            mine_out = [b for b in self.nbrs[a] if b not in inside and self.anum[b] > 0]
            if len(mine_in) != len(t_in) or len(mine_out) != len(t_out):
                continue
            if any(m.get(b) not in t_in for b in mine_in):
                continue
            if any(not self._link(a, b) for b in mine_out):
                continue
            ok.add(a)
        with_atom = {a for a in ok if any(b in ok for b in self.nbrs[a] if b in inside)}
        return {a for a in ok if self.anum[a] > 1 or a in with_atom}

    # -- amino acids inside a ligand residue ---------------------------------------
    def _real(self, a: int) -> list[int]:
        return [b for b in self.nbrs[a] if self.anum[b] > 0]

    def _carbonyl(self, c: int) -> bool:
        """A carbon with three neighbours, one of them an oxygen or sulfur on it alone."""
        return (self.anum[c] == 6 and len(self._real(c)) == 3
                and any(self.anum[o] in (8, 16) and len(self._real(o)) == 1
                        for o in self.nbrs[c]))  # fmt: skip

    def _unit_link(self, name: str, b: int) -> bool:
        """Whether ``b`` may stand where the template atom ``name`` has its external
        bond: an amide carbon on N, an amide nitrogen on C, a sulfur on SG."""
        if name == "N":
            return self._carbonyl(b)
        if name == "C":
            return self.anum[b] == 7 and len(self._real(b)) == 3
        if name == "SG":
            return self.anum[b] == 16 and len(self._real(b)) == 2
        return False

    def _unit_fits(self, t: Template, m: dict[int, int]) -> set[int]:
        """Mapped atoms bonded exactly as their template atoms are: the same atoms
        of the template around them, and where it has an external bond, one atom
        more that could be the next residue's.  An atom with a single neighbour (a
        hydrogen, a carbonyl oxygen) keeps its protein type only with that neighbour,
        and that neighbour only with it."""
        ok = set()
        for a, i in m.items():
            t_in = {j for j in t._nbrs[i] if t.anum[j] > 0}
            n_out = sum(1 for j in t._nbrs[i] if t.anum[j] == -1)
            mine = self._real(a)
            out = [b for b in mine if m.get(b) not in t_in]
            if len(mine) - len(out) != len(t_in) or len(out) != n_out:
                continue
            if all(self._unit_link(t.names[i], b) for b in out):
                ok.add(a)
        # an atom is the template's only with the template's end atoms on it: the
        # carbon of an ester has a carboxylate's neighbours by element, and is not one
        ends = {a for a, i in m.items() if len(t._nbrs[i]) == 1}
        ok = {a for a in ok if all(b in ok for b in self.nbrs[a] if b in ends)}
        return {a for a in ok if len(self._real(a)) > 1 or self._real(a)[0] in ok}

    def _backbones(self, r: int) -> list[dict]:
        """Every N-CA-C(=O) of residue ``r``: a nitrogen, a carbon with four
        neighbours on it, and a carbonyl carbon on that."""
        inside = set(self.res_atoms[r])
        out = []
        for n in self.res_atoms[r]:
            if self.anum[n] != 7:
                continue
            for ca in self.nbrs[n]:
                if ca not in inside or self.anum[ca] != 6 or len(self._real(ca)) != 4:
                    continue
                for c in self.nbrs[ca]:
                    if c == n or c not in inside or not self._carbonyl(c):
                        continue
                    os_ = [o for o in self.nbrs[c] if self.anum[o] == 8 and o in inside
                           and sum(1 for x in self.nbrs[o] if self.anum[x] > 1) == 1]  # fmt: skip
                    if os_:  # the carbonyl oxygen before a hydroxyl one
                        o = min(os_, key=lambda o: len(self._real(o)))
                        out.append({"N": n, "CA": ca, "C": c, "O": o})
        return out

    def _near(self, r: int, ca: int, without=()) -> list[int]:
        """Atoms of residue ``r`` within :data:`_REACH` bonds of ``ca``."""
        inside = set(self.res_atoms[r]) - set(without)
        seen, shell = {ca}, [ca]
        for _ in range(_REACH):
            shell = [b for a in shell for b in self.nbrs[a] if b in inside and b not in seen
                     and not seen.add(b)]  # fmt: skip
        return [a for a in self.res_atoms[r] if a in seen]

    def _units(self, r: int) -> list[dict]:
        """The amino acids inside residue ``r``, a ligand: dicts with the template
        ``t``, the atom map ``m``, the atoms ``ok`` that keep its types and the
        ``match``, "residue" (the whole template, atom for atom), "backbone" (the
        backbone of ALA, GLY or PRO under another side chain) or "cap"."""
        if self.host is None or self.pseudopeptide == "off":
            return []
        amino = [t for t in self.host.templates if {"N", "CA", "C"} <= set(t.names)]
        backbones = self._backbones(r)
        whole = []
        for bb in backbones:
            near = self._near(r, bb["CA"])
            best = None
            for t in amino:
                if t.natoms > len(near):
                    continue
                m = self._map(r, t, bb, near)
                if len(m) != t.natoms:
                    continue
                ok = self._unit_fits(t, m)
                if len(ok) == t.natoms:
                    agree = sum(1 for a, i in m.items() if self.names[a] == t.names[i])
                    key = (t.natoms, agree, _family(t.name) == t.name)
                    if best is None or key > best[0]:
                        best = (key, {"t": t, "m": m, "ok": ok, "match": "residue", "bb": bb})
            if best is not None:
                whole.append(best)
        units, taken = [], set()
        for _, u in sorted(whole, key=lambda x: x[0], reverse=True):
            if not taken & set(u["m"]):
                units.append(u)
                taken |= set(u["m"])
        frames = [t for t in amino if _family(t.name) in _BACKBONES]
        for bb in backbones:
            if bb["CA"] in taken:
                continue
            near = self._near(r, bb["CA"], taken - set(bb.values()))
            best = None
            for t in frames:
                m = self._map(r, t, bb, near)
                ok = {a for a in self._unit_fits(t, m) if t.names[m[a]] in _BACKBONE} - taken
                if bb["CA"] not in ok:  # no alpha carbon of the protein's: not a unit
                    continue
                if _family(t.name) == "PRO" and bb["N"] not in ok:
                    continue  # proline's backbone is its ring closing on N
                key = (sum(1 for a in ok if self.anum[a] > 1), len(ok), -t.natoms)
                if best is None or key > best[0]:
                    best = (key, {"t": t, "m": {a: m[a] for a in ok}, "ok": ok,
                                  "match": "backbone", "bb": bb})  # fmt: skip
            if best is not None:
                units.append(best[1])
                taken |= best[1]["ok"]
        units.sort(key=lambda u: u["bb"]["CA"])
        return units + self._caps(r, units, taken)

    def _caps(self, r: int, units, taken) -> list[dict]:
        """ACE, NME and NHE where they are bonded to a unit's backbone N or C that
        keeps its protein type: the caps of a peptide, not any acetyl or amide."""
        ends = {"N": set(), "C": set()}
        for u in units:
            for key in ends:
                a = u["bb"][key]
                if a in u["ok"]:
                    ends[key].add(a)
        inside = set(self.res_atoms[r])
        out = []
        for t in self.host.templates:
            if t.name not in _CAPS:
                continue
            roots = [i for i in range(len(t.names)) if t.anum[i] > 0
                     and any(t.anum[j] == -1 for j in t._nbrs[i])]  # fmt: skip
            if len(roots) != 1:
                continue
            root = roots[0]
            partner = ends["N" if t.anum[root] == 6 else "C"]  # ACE's C is on a unit's N
            for a in sorted({b for p in partner for b in self.nbrs[p]} & inside - taken):
                if self.anum[a] != t.anum[root]:
                    continue
                m = {a: root}
                self._grow(t, a, root, m, inside - taken)
                if len(m) == t.natoms and len(self._unit_fits(t, m)) == t.natoms:
                    out.append({"t": t, "m": m, "ok": set(m), "match": "cap", "bb": None})
                    taken |= set(m)
        return out

    def _grow(self, t: Template, a: int, i: int, m: dict[int, int], free) -> None:
        """Map the template atoms around ``i`` onto the atoms around ``a``, outward:
        each onto a free atom of its element with as many neighbours (a cap is a small
        tree, where the first that fits is as good as any)."""
        used = set(m.values())
        for j in t._nbrs[i]:
            if t.anum[j] <= 0 or j in used:
                continue
            b = next((b for b in self.nbrs[a] if b in free and b not in m
                      and self.anum[b] == t.anum[j]
                      and len(self._real(b)) == len(t._nbrs[j])), None)  # fmt: skip
            if b is not None:
                m[b] = j
                used.add(j)
                self._grow(t, b, j, m, free)

    # -- one group ---------------------------------------------------------------
    def patch(self) -> ViparrForcefield:
        groups = self.groups()
        for k, residues in enumerate(groups, start=1):
            if self._known(residues):
                continue
            try:
                self._group(residues, k)
            except Exception as e:
                # the error alone ("Unable to determine valence of atom 0 ...") names
                # an atom of a fragment nobody sees: say which residues it was
                chains = self.P.s.chains["name"]
                rchain = self.P.s.residues["chain"]
                where = " + ".join(
                    f"{self.resnames[r]}{self.resid[r]}"
                    + (f" (chain {str(chains[rchain[r]]).strip()})"
                       if str(chains[rchain[r]]).strip() else "")
                    for r in residues)  # fmt: skip
                more = len(groups) - 1
                raise GroupError(
                    f"no GAFF2 parameters for {where}: {e}"
                    + (f" ({more} more group{'s' if more > 1 else ''} of residues without a "
                       "force-field template in this structure)" if more else ""),
                    residues, [a for r in residues for a in self.res_atoms[r]]) from e  # fmt: skip
        params = {}
        for table, rows in self.rows.items():
            params[table] = [ParamRow(key, p, self.memo[key]) for key, plist in rows.items()
                             for p in plist]  # fmt: skip
        if self.host is None:
            rules = Rules(info=[f"GAFF2 {self.run_opts.get('gaff', '2.11')} (AmberTools)"],
                          vdw_func="lj12_6_sig_epsilon", vdw_comb_rule="arithmetic/geometric",
                          es_scale=[0.0, 0.0, 1 / 1.2], lj_scale=[0.0, 0.0, 0.5],
                          plugins=list(_PLUGINS))  # fmt: skip
        else:
            rules = Rules(es_scale=[], lj_scale=[])
        out = ViparrForcefield("gaff2-patch", rules, self.templates, params)
        out.parents = self.chosen  # the parent each amino acid got, and why
        out.units = self.units  # the amino acids found inside ligand residues
        out.drawings = self.drawings
        return out

    def _known(self, residues) -> bool:
        """Whether templates made for an earlier group match every residue."""
        if not self.templates:
            return False
        ff = ViparrForcefield("made", Rules(), self.templates, {})
        return all(self.P._find_uncached(ff, self.res_atoms[r])[0] is not None for r in residues)

    def _group(self, residues: list[int], k: int) -> None:
        atoms = [a for r in residues for a in self.res_atoms[r]]
        inside = set(atoms)
        protein: dict[int, tuple[Template, int]] = {}
        parents = {}
        for r in residues:
            t, m, ok, source = self._parent(r)
            if self.extent == "cb" and t is not None:
                ok = {a for a in ok if _up_to_cb(t, m[a])}
            parents[r] = (t, m, ok)
            if t is not None:
                heavy = [a for a in self.res_atoms[r] if self.anum[a] > 1]
                chain = self.P.s.chains["name"][self.P.s.residues["chain"][r]]
                self.chosen.append({"residue": f"{self.resnames[r]}{self.resid[r]}",
                                    "chain": str(chain), "parent": t.name, "source": source,
                                    "protein_heavy_atoms": sum(a in ok for a in heavy),
                                    "heavy_atoms": len(heavy)})  # fmt: skip
            for a in ok:
                protein[a] = (t, m[a])
        # a ligand residue that is part peptide: the amino acids inside it
        found: dict[int, list[dict]] = {}
        am1: set[int] = set()  # protein-typed atoms that keep their AM1-BCC charge
        for r in residues:
            if parents[r][0] is not None or r in self.grown:
                continue
            found[r] = self._units(r)
            for u in found[r]:
                for a in u["ok"]:
                    protein[a] = (u["t"], u["m"][a])
                if self.pseudopeptide == "types":
                    am1 |= u["ok"]
            self._report(r, found[r])
        frag, f = self._capped(atoms, residues)
        index = {a: i for i, a in enumerate(frag)}
        name = self._template_name(residues)
        workdir = None if self.workdir is None else self.workdir / name
        opts = dict(self.run_opts)
        if any(found.values()):
            # sqm would optimize a molecule this size for hours, and charges fitted to
            # one pose have little to say next to charges copied from a force field
            opts["charge_geometry"] = "reference"
        p = run_gaff2(f, int(f.atoms["formal_charge"].sum()), workdir=workdir, **opts)
        self.runs.append((frag, p))

        nb = p.table("nonbonded")
        pid = nb.param_ids[np.argsort(nb.atoms[:, 0])]
        gtype, sigma, epsilon = (
            nb.params["type"][pid],
            nb.params["sigma"][pid],
            nb.params["epsilon"][pid],
        )
        gcharge, gmass = p.atoms["charge"], p.atoms["mass"]
        tag = f"~{k}" if self.tag is None else f"~{self.tag}" + (f".{k}" if k > 1 else "")
        memo = f"GAFF2 {self.run_opts.get('gaff', '2.11')} with AmberTools ({name})"
        # Two terms may have the same types and different GAFF2 parameters: the
        # protein types are coarser than GAFF2's where they meet (an acyl carbon on
        # proline's N and one on alanine's are both "c - N", and GAFF2 has n and ns
        # there).  A GAFF2 atom of one of the two then takes a type of its own.
        split: dict[int, int] = {}
        kept = ({t: dict(rows) for t, rows in self.rows.items()}, dict(self.memo))
        while True:
            self.row_atoms = {}
            types: list[tuple[str, str] | None] = [None] * f.natoms
            gaff_atoms = set()
            for i, a in enumerate(frag):
                if a in inside:
                    if a in protein:
                        t, ti = protein[a]
                        types[i] = (t.btype[ti], t.nbtype[ti])
                    else:
                        own = f":{split[i]}" if i in split else ""
                        types[i] = (gtype[i] + tag + own,) * 2
                        gaff_atoms.add(i)
                        self._add(
                            "vdw1",
                            (types[i][1],),
                            [{"sigma": float(sigma[i]), "epsilon": float(epsilon[i])}],
                            memo,
                        )
                        self._add("mass", (types[i][0],), [{"amu": float(gmass[i])}], memo)
                else:
                    matched = self._matched(self.residue[a])
                    types[i] = matched[a]
            try:
                impropers = self._terms(p, f, types, gaff_atoms, len(frag), memo)
                break
            except _Conflict as e:
                mine, first = set(e.atoms) & gaff_atoms, set(e.first) & gaff_atoms
                free = sorted((mine ^ first) - set(split))
                if not free:
                    raise
                split[free[0]] = len(split) + 1
                self.rows = {t: dict(rows) for t, rows in kept[0].items()}
                self.memo = dict(kept[1])
        if self.draw is not None and self._covalent(residues):
            self.drawings.append(self._draw(residues, frag, f, gaff_atoms))
        for r, units in found.items():
            if units and (self.draw is not None or self.pdf is not None):
                self.drawings.append(self._draw_units(r, units))

        parts = [self.resnames[r] or "LIG" for r in residues]
        for i, r in enumerate(residues):
            if r in self.grown:  # the host force field parameterizes it, as before
                continue
            t, m, _ = parents[r]
            units = [(t, m)] if t is not None else [(u["t"], u["m"]) for u in found.get(r, [])]
            part = parts[i] if parts.count(parts[i]) == 1 else f"{parts[i]}{i + 1}"
            tname = name if len(residues) == 1 else f"{name}_{part}"
            self.templates.append(self._template(r, tname, types, index, protein, gcharge, f,
                                                 units, impropers, inside, am1))  # fmt: skip

    def _report(self, r: int, units) -> None:
        chain = str(self.P.s.chains["name"][self.P.s.residues["chain"][r]])
        for u in units:
            heavy = sorted(a for a in u["ok"] if self.anum[a] > 1)
            label = f"{self.resnames[r] or 'LIG'}{self.resid[r]}"
            self.units.append({"residue": label, "chain": chain,
                               "template": u["t"].name, "match": u["match"],
                               "atoms": [str(self.names[a]) for a in heavy],
                               "indices": heavy})  # fmt: skip

    def _draw_units(self, r: int, units) -> Path:
        """A 2D drawing of ligand residue ``r`` as a PDF with a transparent page: the
        heavy atoms that carry protein parameters, and the bonds among them, on yellow."""
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D

        from .chem import to_rdkit
        from .draw import svg_to_pdf

        atoms = self.res_atoms[r]
        protein = {a for u in units for a in u["ok"]}
        try:
            mol = to_rdkit(self.P.s, atoms, sanitize=True, implicit_hydrogens=False,
                           residue_info=False)  # fmt: skip
        except Exception:  # as it is, without stereo: a drawing is not worth an error
            mol = to_rdkit(self.P.s, atoms, sanitize=False, implicit_hydrogens=False,
                           stereo=False, residue_info=False, conformer=False)  # fmt: skip
        for atom in mol.GetAtoms():
            atom.SetIntProp("at", atoms[atom.GetIdx()])
        mol = Chem.RemoveHs(mol, sanitize=False)
        marked = [a.GetIdx() for a in mol.GetAtoms() if a.GetIntProp("at") in protein]
        on = set(marked)
        bonds = [b.GetIdx() for b in mol.GetBonds()
                 if b.GetBeginAtomIdx() in on and b.GetEndAtomIdx() in on]  # fmt: skip
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(mol)
        side = int(min(2400, max(500, 60 * mol.GetNumAtoms() ** 0.5 * 4)))
        drawer = rdMolDraw2D.MolDraw2DSVG(side, int(side * 0.75))
        o = drawer.drawOptions()
        o.clearBackground = False  # a transparent page
        o.fillHighlights = True
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol, highlightAtoms=marked, highlightAtomColors=dict.fromkeys(marked, YELLOW),
            highlightBonds=bonds, highlightBondColors=dict.fromkeys(bonds, YELLOW))  # fmt: skip
        drawer.FinishDrawing()
        label = re.sub(r"[^A-Za-z0-9+_-]", "_", f"{self.resnames[r] or 'LIG'}{self.resid[r]}")
        if self.pdf is not None:
            n = sum(1 for d in self.drawings if d.suffix == ".pdf")
            out = self.pdf if n == 0 else self.pdf.with_name(
                f"{self.pdf.stem}-{n + 1}{self.pdf.suffix}")  # fmt: skip
        else:
            out = self.draw / f"pseudopeptide_{label}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(svg_to_pdf(drawer.GetDrawingText()))
        return out

    def _amino_acids(self, residues) -> set[int]:
        return {r for r in residues if self._amino(r)}

    def _covalent(self, residues) -> bool:
        """A ligand bound to an amino acid: a group with both kinds of residue."""
        return 0 < len(self._amino_acids(residues)) < len(residues)

    def _draw(self, residues, frag, f, gaff_atoms) -> Path:
        """A 2D drawing of a covalent adduct: heavy atoms with protein types in
        blue, GAFF2 atoms in orange, ``*`` where the chain goes on."""
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D

        from .chem import to_rdkit

        wanted = set(residues)
        group = {i for i, a in enumerate(frag) if self.residue[a] in wanted}
        bonds = list(zip(f.bonds["i"].tolist(), f.bonds["j"].tolist(), strict=True))
        adj: dict[int, list[int]] = {}
        for i, j in bonds:
            adj.setdefault(i, []).append(j)
            adj.setdefault(j, []).append(i)
        stubs = {j for i in group for j in adj.get(i, []) if j not in group}
        aa = self._amino_acids(residues)
        mol = Chem.RWMol(to_rdkit(f, sanitize=False, implicit_hydrogens=False, stereo=False,
                                  residue_info=False, conformer=False))  # fmt: skip
        for atom in mol.GetAtoms():
            i = atom.GetIdx()
            atom.SetIntProp("frag", i)
            atom.SetNoImplicit(True)
            if i in stubs:  # the neighbouring residue, as an attachment point
                r = self.residue[frag[i]]
                atom.SetAtomicNum(0)
                atom.SetFormalCharge(0)
                atom.SetProp("atomNote", f"{self.resnames[r]}{self.resid[r]}")
            elif i in group and atom.GetAtomicNum() > 1:
                r = self.residue[frag[i]]
                bound = any(j in group and self.residue[frag[j]] in aa - {r} for j in adj[i])
                if r in aa or bound:
                    atom.SetProp("atomNote", str(self.names[frag[i]]))
        for i in sorted(set(range(mol.GetNumAtoms())) - group - stubs, reverse=True):
            mol.RemoveAtom(i)
        mol = Chem.RemoveHs(mol.GetMol(), sanitize=False)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL
                         ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS)  # fmt: skip
        blue, orange = (0.55, 0.75, 1.0), (1.0, 0.68, 0.35)
        atom_colors = {}
        for atom in mol.GetAtoms():
            i = atom.GetIntProp("frag")
            if i in group and atom.GetAtomicNum() > 1:
                atom_colors[atom.GetIdx()] = orange if i in gaff_atoms else blue
        bond_colors = {}
        for b in mol.GetBonds():
            c = atom_colors.get(b.GetBeginAtomIdx())
            if c is not None and c == atom_colors.get(b.GetEndAtomIdx()):
                bond_colors[b.GetIdx()] = c
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(mol)
        label = "+".join(f"{self.resnames[r]}{self.resid[r]}" for r in residues)
        host = Path(self.host.name.split("__")[0]).name if self.host else "protein"
        reach = "backbone and CB" if self.extent == "cb" else "as far as they match"
        drawer = rdMolDraw2D.MolDraw2DCairo(1000, 800)
        drawer.drawOptions().legendFontSize = 22
        drawer.drawOptions().annotationFontScale = 0.7
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol, legend=f"{label}   blue: {host} types ({reach})   orange: GAFF2",
            highlightAtoms=list(atom_colors), highlightAtomColors=atom_colors,
            highlightBonds=list(bond_colors), highlightBondColors=bond_colors)  # fmt: skip
        drawer.FinishDrawing()
        out = self.draw / ("covalent_" + re.sub(r"[^A-Za-z0-9+_-]", "_", label) + ".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(drawer.GetDrawingText())
        return out

    def _template_name(self, residues) -> str:
        base = "+".join(self.resnames[r] or "LIG" for r in residues) + "_gaff2"
        if self.tag is not None:
            base = f"{self.tag}_{base}"
        taken = {t.name for t in self.templates} | (
            {t.name for t in self.host.templates} if self.host else set())  # fmt: skip
        name, n = base, 1
        while name in taken or any(x.startswith(name + "_") for x in taken):
            n += 1
            name = f"{base}{n}"
        return name

    def _capped(self, atoms, residues):
        """The group with ACE/NME caps: the neighbouring residue's atoms across
        each peptide bond, carbons among them closed with hydrogens."""
        inside = set(atoms)
        frag = list(atoms)
        index = {a: i for i, a in enumerate(frag)}
        new_h: list[tuple[int, np.ndarray]] = []
        for a in atoms:
            for b in self.nbrs[a]:
                if b in inside or self.anum[b] == 0:
                    continue
                if not self._peptide(a, b):
                    raise ViparrError(f"cannot cap the bond {self._label(a)}-{self._label(b)}: "
                                      "only peptide bonds are capped")  # fmt: skip
                if self._matched(self.residue[b]) is None:
                    raise ViparrError(f"{self._label(b)} has no template to cap with")
                if b in index:
                    continue
                index[b] = len(frag)
                frag.append(b)
                for y in self.nbrs[b]:
                    if y in index or self.anum[y] == 0:
                        continue
                    index[y] = len(frag)
                    frag.append(y)
                    heavy = [o for o in self.nbrs[y] if self.anum[o] > 0]
                    if self.anum[y] == 1 or len(heavy) == 1:
                        continue
                    if self.anum[y] != 6:
                        raise ViparrError(f"cannot cap at {self._label(y)}: only carbons "
                                          "become methyl groups")  # fmt: skip
                    for z in heavy:
                        if z != b and z not in index:
                            v = self.pos[z] - self.pos[y]
                            new_h.append((index[y], self.pos[y] + _CH * v / np.linalg.norm(v)))
        n = len(frag)
        pos = np.vstack([self.pos[frag], np.array([q for _, q in new_h]).reshape(-1, 3)])
        anum = np.array([self.anum[a] for a in frag] + [1] * len(new_h), np.int64)
        f = System.from_arrays(pos, anum=anum, resnames="MOL")
        f.atoms["formal_charge"] = np.array([self.formal[a] for a in frag] + [0] * len(new_h))
        pairs, orders = [], []
        for i, a in enumerate(frag):
            for b in self.nbrs[a]:
                j = index.get(b)
                if j is not None and i < j:
                    pairs.append((i, j))
                    orders.append(self.P.order[(min(a, b), max(a, b))])
        pairs += [(i, n + h) for h, (i, _) in enumerate(new_h)]
        orders += [1] * len(new_h)
        f.add_bonds(np.array(pairs, np.int64), order=np.array(orders, np.int64))
        if _needs_orders(f):
            from .chem import assign_bond_orders

            total = 0
            for r in residues:
                q = self.charges.get(self.resnames[r])
                total += int(q) if q is not None else sum(self.formal[a] for a in self.res_atoms[r])
            assign_bond_orders(f, charge=total)
        return frag, f

    def _add(self, table: str, types: tuple, params: list[dict], memo: str, atoms=()) -> None:
        rows = self.rows.setdefault(table, {})
        key, rkey = " ".join(types), " ".join(reversed(types))
        for kk in (key, rkey):
            if kk in rows:
                if _canon(rows[kk]) != _canon(params):
                    raise _Conflict(f"conflicting GAFF2 parameters for {table} {kk}",
                                    atoms, self.row_atoms.get((table, kk), ()))  # fmt: skip
                return
        rows[key] = params
        self.memo[key] = memo
        self.row_atoms[(table, key)] = tuple(atoms)

    def _terms(self, p, f, types, gaff_atoms, ncap, memo) -> list[tuple[int, ...]]:
        """Rows for every term that touches a GAFF2 atom; returns the impropers."""
        bonded = set(zip(f.bonds["i"].tolist(), f.bonds["j"].tolist(), strict=True))
        bonded |= {(j, i) for i, j in bonded}

        def wanted(t) -> bool:
            if not any(i in gaff_atoms for i in t):
                return False
            if any(i >= ncap for i in t):
                raise ViparrError(
                    "a GAFF2 atom is next to a cap: the backbone N and C of a residue in "
                    "a protein must match its parent residue"
                )
            return True

        def rows_of(table, cols):
            tab = p.table(table)
            prm = {c: tab.params[c][tab.param_ids] for c in cols}
            for n, t in enumerate(tab.atoms.tolist()):
                yield tuple(t), {c: float(prm[c][n]) for c in cols}

        for table, cols in (("stretch_harm", ("r0", "fc")), ("angle_harm", ("theta0", "fc"))):
            for t, row in rows_of(table, cols):
                if wanted(t):
                    self._add(table, tuple(types[i][0] for i in t), [row], memo, t)
        cols = ("phi0", *(f"fc{k}" for k in range(7)))
        propers: dict[tuple, list[dict]] = {}
        impropers: dict[tuple, list[dict]] = {}
        for t, row in rows_of("dihedral_trig", cols):
            proper = (t[0], t[1]) in bonded and (t[1], t[2]) in bonded and (t[2], t[3]) in bonded
            (propers if proper else impropers).setdefault(t, []).append(row)
        seen = {min(t, t[::-1]) for t in propers}
        nbrs = [[] for _ in range(f.natoms)]
        for i, j in bonded:
            nbrs[i].append(j)
        for j in range(f.natoms):  # proper dihedrals tleap left out get zero rows
            for k in nbrs[j]:
                if k < j:
                    continue
                for i in nbrs[j]:
                    for m in nbrs[k]:
                        if (
                            i != k
                            and m != j
                            and m != i
                            and min((i, j, k, m), (m, k, j, i)) not in seen
                        ):
                            propers[(i, j, k, m)] = [dict(_ZERO)]
                            seen.add(min((i, j, k, m), (m, k, j, i)))
        for t, plist in propers.items():
            if wanted(t):
                self._add("dihedral_trig", tuple(types[i][0] for i in t), plist, memo, t)
        out = []
        for t, plist in impropers.items():
            if wanted(t):
                if len(plist) != 1:
                    raise ViparrError("an improper with several terms is not supported")
                self._add("improper_trig", tuple(types[i][0] for i in t), plist, memo, t)
                out.append(t)
        return out

    def _template(self, r, name, types, index, protein, gcharge, f, units, impropers,
                  group, am1=()):  # fmt: skip
        """The template of residue ``r``.  ``units`` are the (template, atom map)
        pairs its protein-typed atoms come from: its parent, or the amino acids
        found inside a ligand; ``group`` the atoms typed with it, ``am1`` those of
        them that keep an AM1-BCC charge under a protein type."""
        ratoms = self.res_atoms[r]
        local = {a: i for i, a in enumerate(ratoms)}
        anum = [self.anum[a] for a in ratoms]
        names = _unique_names([self.names[a] for a in ratoms], anum)
        btype = [types[index[a]][0] for a in ratoms]
        nbtype = [types[index[a]][1] for a in ratoms]
        charge = [protein[a][0].charge[protein[a][1]] if a in protein and a not in am1
                  else float(gcharge[index[a]]) for a in ratoms]  # fmt: skip
        bonds, ext, pinned, formulas = [], {}, {}, {}
        for a in ratoms:
            for b in self.nbrs[a]:
                if self.anum[b] == 0:
                    continue
                if b in local:
                    if local[a] < local[b]:
                        bonds.append((local[a], local[b]))
                    continue
                if b not in ext:
                    ext[b] = len(names)
                    names.append(f"${len(ext)}")
                    anum.append(-1)
                    charge.append(0.0)
                    btype.append("")
                    nbtype.append("")
                    pinned[ext[b]] = self.anum[b]
                    if not self._link(a, b):  # the partner's residue too: a ligand's S is no SG
                        formulas[ext[b]] = self.P._formula_of(b)
                bonds.append((local[a], ext[b]))
        where = {**local, **ext}

        # formal charge by shifting the GAFF2 atoms evenly
        gaff = [i for i, a in enumerate(ratoms) if a not in protein or a in am1]
        q = self.charges.get(self.resnames[r])
        want = int(q) if q is not None else int(round(sum(
            int(f.atoms["formal_charge"][index[a]]) for a in ratoms)))  # fmt: skip
        have = sum(charge[: len(ratoms)])
        if gaff:
            shift = (want - have) / len(gaff)
            for i in gaff:
                charge[i] += shift
            if abs(shift) > 0.1:
                warnings.warn(f"{self.resnames[r]}{self.resid[r]}: GAFF2 charges shifted by "
                              f"{shift:+.3f} e on each of {len(gaff)} atoms to reach {want:+d}",
                              ViparrWarning, stacklevel=4)  # fmt: skip
        elif abs(want - have) > 1e-3:
            warnings.warn(f"{self.resnames[r]}{self.resid[r]}: charge {have:+.4f}, not {want:+d}",
                          ViparrWarning, stacklevel=4)  # fmt: skip

        # the parent's impropers and CMAP where all their atoms keep its types
        tuples = {"impropers": [], "cmap": []}
        for parent, pmap in units:
            inv = {i: a for a, i in pmap.items()}

            def system_atom(ti, parent=parent, pmap=pmap, inv=inv):
                if parent.anum[ti] > 0:
                    a = inv.get(ti)
                    return a if a in protein else None
                inner = inv.get(parent._nbrs[ti][0])
                if inner is None or inner not in protein:
                    return None
                out = [b for b in self.nbrs[inner] if b not in pmap and self.anum[b] > 0]
                return out[0] if len(out) == 1 else None

            for key, src in (("impropers", parent.impropers), ("cmap", parent.cmaps)):
                for tup in src:
                    mapped = [system_atom(ti) for ti in tup]
                    if any(a is None for a in mapped):
                        continue
                    # an improper over a GAFF2 atom of this group is tleap's (below);
                    # a CMAP only needs its N, CA and C to be the protein's
                    if key == "impropers" and any(a in group and a not in protein
                                                  for a in mapped):  # fmt: skip
                        continue
                    tuples[key].append(tuple(where[a] for a in mapped))
        # GAFF2 impropers centred on this residue's atoms
        frag_of = {i: a for a, i in index.items()}
        for t in impropers:
            atoms = [frag_of[i] for i in t]
            if atoms[2] in local and all(a in where for a in atoms):
                tuples["impropers"].append(tuple(where[a] for a in atoms))
        return Template(name, names, anum, charge, btype, nbtype, [""] * len(names), bonds,
                        tuples["impropers"], tuples["cmap"], [], [], pinned, formulas)  # fmt: skip


def _unique_names(names: list[str], anum: list[int]) -> list[str]:
    """``names``, or element + count (C1, C2, ..., H1, ...) when they repeat."""
    if all(names) and len(set(names)) == len(names):
        return names
    from .elements import symbol

    counts: dict[str, int] = {}
    out = []
    for z in anum:
        sym = symbol(z) or "X"
        counts[sym] = counts.get(sym, 0) + 1
        out.append(f"{sym}{counts[sym]}")
    return out


def _canon(params: list[dict]) -> list:
    return sorted(tuple(sorted((k, round(v, 6)) for k, v in p.items())) for p in params)


def _needs_orders(f: System) -> bool:
    """Whether bond orders are missing: an atom lacks valence (a structure file's
    single bonds, possibly next to a ligand that has its orders)."""
    order = f.bonds["order"]
    val = np.zeros(f.natoms, np.int64)
    np.add.at(val, f.bonds["i"], order)
    np.add.at(val, f.bonds["j"], order)
    for z, fc, v in zip(f.atoms["anum"].tolist(), f.atoms["formal_charge"].tolist(),
                        val.tolist(), strict=True):  # fmt: skip
        want = _VALENCE.get(z)
        if want is not None and v < want + (fc if z in (7, 8, 15, 16) else -abs(fc)):
            return True
    return False
