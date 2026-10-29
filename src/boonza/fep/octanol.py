"""1-octanol boxes for a water-to-octanol transfer leg (boonza fep prep --phase).

A relative transfer free energy is one edge in two legs, in water and in
octanol: ΔΔG_transfer = ΔG(octanol) - ΔG(water), and the change of log P
between the two ligands is -ΔΔG_transfer / (RT ln 10).

The box: N_OCTANOL octanols with GAFF2 types and AM1-BCC charges (the ligands'
force field) and, for wet octanol, as many TIP3P waters as water-saturated
octanol holds -- the phase a shake-flask log P is measured in.  packmol
(AmberTools) packs them periodically below liquid density, and a run at
constant pressure brings them to it.  It is built once per work directory, in
WORKDIR/octanol/box.dms, and tiled around each edge's ligands as the TIP3P box
is around a water leg, in whole copies so the octanol has no seam.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..system import System

OCTANOL_SMILES = "CCCCCCCCO"
#: the octanol residue, so the GAFF2 patch holds one template for every molecule
RESNAME = "OCT"
#: octanols in the box: 3.9 nm across, wider than two 0.9 nm cutoffs and than a
#: small ligand with 1.5 nm of padding
N_OCTANOL = 200
#: water's mole fraction in water-saturated 1-octanol at 25 C (2.3 mol/L)
WET_WATER_FRACTION = 0.27
#: packmol packs at this density (g/cm^3), below the liquid's, so it finds room
PACK_DENSITY = 0.6
#: 1-octanol at 25 C (g/cm^3); a box far from it is warned about
DENSITY = 0.824
#: constant-pressure run that brings the packed box to liquid density
EQUILIBRATION_NS = 2.0
#: octanol heavy atoms nearer the ligands than this (Å) are cleared, with their molecule
OCTANOL_RADIUS = 2.4

_AVOGADRO = 6.02214076e23


def n_waters(n_octanol: int, wet: bool) -> int:
    """Waters beside ``n_octanol`` octanols: none dry, WET_WATER_FRACTION wet."""
    if not wet:
        return 0
    return int(round(n_octanol * WET_WATER_FRACTION / (1.0 - WET_WATER_FRACTION)))


def molecule(seed: int = 0) -> System:
    """One 1-octanol, with hydrogens and a conformer, as residue RESNAME."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from ..chem import from_rdkit

    mol = Chem.AddHs(Chem.MolFromSmiles(OCTANOL_SMILES))
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise RuntimeError("RDKit could not embed 1-octanol")
    AllChem.MMFFOptimizeMolecule(mol)
    s = from_rdkit(mol, name="octanol")
    s.residues["name"] = [RESNAME] * s.nresidues
    return s


def _water() -> System:
    """One TIP3P water, from the box boonza.solvate tiles."""
    from ..build import WATER_BOX
    from ..io import load

    box = load(WATER_BOX)
    return box.clone(np.flatnonzero(np.asarray(box.fragids) == 0))


def _mass(s: System) -> float:
    from ..elements import MSYS_MASSES

    return float(sum(MSYS_MASSES[int(z)] for z in s.atoms["anum"]))


def _xyz(s: System, path: Path) -> None:
    from ..elements import msys_symbol

    lines = [str(s.natoms), s.name or "molecule"]
    lines += [f"{msys_symbol(int(z))} {x:.4f} {y:.4f} {w:.4f}"
              for z, (x, y, w) in zip(s.atoms["anum"], np.asarray(s.positions), strict=True)]
    path.write_text("\n".join(lines) + "\n")


def pack(octanol: System, water: System, n_octanol: int, n_water: int, seed: int = 0,
         density: float = PACK_DENSITY) -> System:  # fmt: skip
    """``n_octanol`` octanols and ``n_water`` waters packed by packmol into a
    periodic cube at ``density`` (g/cm^3); molecules whole, octanols first."""
    exe = shutil.which("packmol")
    if exe is None:
        raise RuntimeError("packmol was not found: it comes with AmberTools "
                           "('conda install -c conda-forge ambertools')")  # fmt: skip
    grams = (n_octanol * _mass(octanol) + n_water * _mass(water)) / _AVOGADRO
    edge = (grams / density) ** (1.0 / 3.0) * 1e8  # Å
    with tempfile.TemporaryDirectory(prefix="boonza_packmol_") as tmp:
        d = Path(tmp)
        _xyz(octanol, d / "octanol.xyz")
        _xyz(water, d / "water.xyz")
        blocks = [("octanol.xyz", n_octanol)] + ([("water.xyz", n_water)] if n_water else [])
        script = ["tolerance 2.0", "filetype xyz", "output packed.xyz", f"seed {seed + 1}",
                  f"pbc 0. 0. 0. {edge:.3f} {edge:.3f} {edge:.3f}"]  # fmt: skip
        for name, n in blocks:
            script += [f"structure {name}", f"  number {n}", "end structure"]
        (d / "pack.inp").write_text("\n".join(script) + "\n")
        with (d / "pack.inp").open() as fh:
            done = subprocess.run([exe], stdin=fh, cwd=d, capture_output=True, text=True)
        out = d / "packed.xyz"
        if done.returncode != 0 or not out.exists():
            raise RuntimeError(f"packmol failed:\n{done.stdout[-2000:]}")
        rows = out.read_text().splitlines()[2:]
    xyz = np.array([[float(v) for v in r.split()[1:4]] for r in rows if r.strip()])
    box = None
    for template, n in ((octanol, n_octanol), (water, n_water)):
        for _ in range(n):
            copy = template.copy()
            k = 0 if box is None else box.natoms
            copy.positions = xyz[k:k + copy.natoms]
            if box is None:
                box = copy
            else:
                box.append(copy)
    if box.natoms != len(xyz):
        raise RuntimeError(f"packmol wrote {len(xyz)} atoms for {box.natoms}")
    box.residues["resid"] = np.arange(1, box.nresidues + 1)
    box.cell = np.diag([edge] * 3)
    return box


def _wrap(s: System) -> None:
    """Every molecule back in the cell by the center of its heavy atoms (what
    boonza.solvate keeps a molecule by), the molecule kept whole."""
    pos = np.asarray(s.positions, float)
    edge = np.diag(s.cell)
    frag = np.asarray(s.fragids)
    heavy = (np.asarray(s.atoms["anum"]) > 1).astype(float)
    n = np.bincount(frag, weights=heavy)
    center = np.stack([np.bincount(frag, weights=heavy * pos[:, d]) / n for d in range(3)],
                      axis=1)  # fmt: skip
    s.positions = pos - (np.floor(center / edge) * edge)[frag]


def equilibrate(box: System, ff, args, ns: float = EQUILIBRATION_NS,
                log=print) -> tuple[System, float]:  # fmt: skip
    """``box`` minimized and run for ``ns`` at ``args``' temperature and
    pressure, with the legs' cutoff, timestep and hydrogen masses: its last
    positions and cell, and its density (g/cm^3) over the run's second half."""
    import openmm

    from ..omm import to_openmm
    from .prep import _by_tag, _parameterize

    box = box.copy()
    box.atoms["fep_index"] = np.arange(1, box.natoms + 1)
    param = _parameterize(box, ff, args.hmr)
    where = _by_tag(param, box.natoms)
    _, system, _ = to_openmm(param, nonbonded_method="PME", cutoff=10.0 * args.cutoff_nm,
                             constraints=True, dispersion_correction=False)  # fmt: skip
    system.addForce(openmm.MonteCarloBarostat(args.pressure, args.temperature, 25))
    dt = args.integration_fs / 1000.0
    integrator = openmm.LangevinMiddleIntegrator(args.temperature, 1.0, dt)
    extra = [openmm.Platform.getPlatformByName(args.platform)] if args.platform else []
    context = openmm.Context(system, integrator, *extra)
    u = openmm.unit
    context.setPeriodicBoxVectors(*(np.asarray(param.cell) / 10.0 * u.nanometer))
    context.setPositions(np.asarray(param.positions) / 10.0 * u.nanometer)
    openmm.LocalEnergyMinimizer.minimize(context, 10.0, 5000)
    context.setVelocitiesToTemperature(args.temperature, 1)
    grams = _mass(box) / _AVOGADRO
    chunks = 20
    steps = max(int(round(ns * 1000.0 / dt / chunks)), 1)
    densities = []
    for k in range(chunks):
        integrator.step(steps)
        v = context.getState().getPeriodicBoxVolume().value_in_unit(u.nanometer**3)
        densities.append(grams / (v * 1e-21))
        if k % 5 == 4:
            log(f"  octanol box: {(k + 1) * steps * dt / 1000:g} ns, {densities[-1]:.3f} g/cm^3")
    state = context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(u.nanometer) * 10.0
    out = box.copy()
    out.positions = pos[where]
    cell = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer)
    out.cell = np.asarray(cell) * 10.0
    _wrap(out)
    return out, float(np.mean(densities[chunks // 2:]))


def box(work: Path, ff, args, wet: bool, path=None, log=print) -> System:
    """The octanol box of a work directory: ``path`` if given, else
    WORKDIR/octanol/box.dms if built before (and of the same kind), else built
    (packed and equilibrated) and written there."""
    from ..io import load, save

    if path is not None:
        log(f"octanol box from {path}")
        return load(path)
    d = Path(work) / "octanol"
    dms, meta_path = d / "box.dms", d / "box.json"
    if dms.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("wet") == wet:
            log(f"octanol box from {dms} ({meta['octanols']} octanols, {meta['waters']} waters)")
            return load(dms)
        raise ValueError(f"{dms} is {'wet' if meta.get('wet') else 'dry'} octanol: "
                         "use a fresh --workdir for the other kind")  # fmt: skip
    nw = n_waters(N_OCTANOL, wet)
    log(f"{'wet' if wet else 'dry'} octanol box: {N_OCTANOL} octanols, {nw} waters "
        f"(packmol, then {EQUILIBRATION_NS:g} ns at constant pressure)")  # fmt: skip
    packed = pack(molecule(), _water(), N_OCTANOL, nw)
    done, density = equilibrate(packed, ff, args, EQUILIBRATION_NS, log)
    if abs(density - DENSITY) > 0.05 * DENSITY:
        log(f"  warning: the octanol box is at {density:.3f} g/cm^3, octanol's is "
            f"{DENSITY} at 25 C")  # fmt: skip
    d.mkdir(parents=True, exist_ok=True)
    save(done, dms)
    meta_path.write_text(json.dumps({"wet": wet, "octanols": N_OCTANOL, "waters": nw,
                                     "density": round(density, 4),
                                     "edge_nm": round(float(done.cell[0, 0]) / 10.0, 4),
                                     "equilibration_ns": EQUILIBRATION_NS}, indent=1))  # fmt: skip
    return done
