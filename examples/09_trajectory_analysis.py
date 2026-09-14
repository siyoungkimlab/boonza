"""Make a short trajectory, then analyze it.

With OpenMM installed this runs 2 ps of implicit-solvent molecular dynamics
of the FN3 domain 1TEN and writes a DCD file with boonza's writer.  Without
OpenMM it makes a trajectory by jiggling the crystal structure.  Then:
RMSD, RMSF, radius of gyration, secondary structure, backbone angles and
hydrogen bonds over the frames.

    python examples/09_trajectory_analysis.py
"""

import importlib.util

import numpy as np
from _common import DATA, OUT, remove_incomplete_residues

import boonza

dcd = OUT / "1TEN_md.dcd"
if importlib.util.find_spec("openmm"):
    import openmm
    from openmm import app, unit

    pdb = app.PDBFile(str(DATA / "1TEN.pdb"))
    model = app.Modeller(pdb.topology, pdb.positions)
    model.deleteWater()
    remove_incomplete_residues(model)
    ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    model.addHydrogens(ff)
    system = ff.createSystem(model.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    sim = app.Simulation(model.topology, system,
                         openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                                         0.002 * unit.picoseconds))  # fmt: skip
    sim.context.setPositions(model.positions)
    sim.minimizeEnergy(maxIterations=200)
    s = boonza.from_openmm(model.topology, positions=model.positions)  # structure only
    with boonza.open_writer(dcd, s.natoms) as writer:
        for _ in range(20):
            sim.step(50)  # 0.1 ps between frames
            xyz = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
            writer.write(xyz.value_in_unit(unit.angstrom))
    print("ran 2 ps of MD with OpenMM")
else:
    s = boonza.load(DATA / "1TEN.pdb")
    rng = np.random.default_rng(0)
    with boonza.open_writer(dcd, s.natoms) as writer:
        for k in range(20):
            writer.write(s.positions + rng.normal(0, 0.1 + 0.02 * k, s.positions.shape))
    print("OpenMM not installed: made a jiggled trajectory instead")
boonza.save(s, OUT / "1TEN_md_topology.pdb")

# --- read it back ------------------------------------------------------------
traj = boonza.open_trajectory(dcd, s)
frames = traj.read()
print(f"{len(traj)} frames of {traj.natoms} atoms")

ca = "protein and name CA"
rmsd = boonza.rmsd_trajectory(s, frames, ca)
print("C-alpha RMSD to frame 0 (A):", np.round(rmsd[::4], 2))

rg = boonza.radius_of_gyration(s, frames, "protein")
print(f"radius of gyration: {rg.mean():.2f} +- {rg.std():.2f} A")

# RMSF needs frames superposed first: Glue fits every frame onto the first.
fit = boonza.Glue(s, fit=ca, whole=None, wrap=False)
fitted = np.array([fit(p)[0] for p in frames.positions])
rmsf = boonza.rmsf(s, fitted, ca)
resids = s.residues["resid"][s.atoms["residue"][s.select(ca).ids]]
top = np.argsort(rmsf)[::-1][:5]
print("most flexible residues:", [(int(resids[k]), round(float(rmsf[k]), 2)) for k in top])

# Secondary structure per frame, as strand/helix fractions.
codes = boonza.dssp(s, frames, simplified=True)
protein = codes[0] != "NA"
strand = (codes[:, protein] == "E").mean(axis=1)
print(f"strand fraction: first frame {strand[0]:.0%}, last frame {strand[-1]:.0%}")

phi, psi, omega = boonza.backbone_dihedrals(s, frames)
k = int(np.flatnonzero(s.residues["resid"] == 830)[0])
print(f"residue 830 phi/psi over time: {np.round(phi[::5, k])} / {np.round(psi[::5, k])}")

# Hydrogen bonds in every frame, and the most persistent ones.
hb = boonza.hbonds(s, frames)
print("hydrogen bonds per frame:", hb.per_frame()[::4].tolist())


def label(i):
    a = s.atom(int(i))
    return f"{a.residue.name}{a.residue.resid}:{a.name}"


for (d, _h, a), f in list(hb.frequency().items())[:3]:
    print(f"  {label(d):>12s} -> {label(a):<12s} present in {f:.0%} of frames")
