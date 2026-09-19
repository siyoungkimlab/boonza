"""Find the pockets, the pose in one, its kinetics, and what the pocket asks for.

A screen makes three trajectories of a small protein with three unlike
fragments swimming around it.  Then, in the order the questions come:

    where does a ligand go?          boonza.sites
    what is the pocket made of?      boonza.site_pocket
    how does it sit there?           boonza.poses
    how long does it stay?           boonza.kinetics
    do different molecules agree?    boonza.interaction_fingerprints
    what should go there?            boonza.feature_maps, boonza.hotspots

    python examples/24_binding_analysis.py
"""

import numpy as np
from _common import OUT

import boonza

# --- a screen: one protein, three unlike fragments, three runs ----------------
protein = boonza.peptide("AAAAAAAAAAAA", conformation="helix")
protein.positions = protein.positions - protein.positions.mean(0)
s = protein.clone()
for smiles in ("c1ccccc1O", "c1ccccc1C(=O)O", "c1ccccc1C=O"):
    s.append(boonza.from_smiles(smiles))
s.cell = np.diag([34.0, 34.0, 34.0])

lig = s.select("not polymer and noh").ids
frag = np.unique(np.asarray(s.fragids)[lig])
copies = [s.select(f"fragid {int(f)}").ids for f in frag]
base = s.positions.copy()
for c in copies:
    base[c] -= base[c].mean(0)
POCKET = np.array([6.0, 0.0, 2.0])  # where the fragments will gather


def run(seed, nframes=150):
    """A run in which each fragment binds and unbinds, with the protein tumbling."""
    rng = np.random.default_rng(seed)
    bound = [False] * len(copies)
    out = []
    for _ in range(nframes):
        x = base.copy()
        for i, c in enumerate(copies):
            if bound[i] and rng.random() < 0.05:
                bound[i] = False
            elif not bound[i] and rng.random() < 0.08:
                bound[i] = True
            x[c] = base[c] + (POCKET + rng.normal(scale=0.5, size=3) if bound[i]
                              else rng.uniform(-14, 14, size=3))  # fmt: skip
        angle = rng.uniform(0, 2 * np.pi)
        ca, sa = np.cos(angle), np.sin(angle)
        out.append(x @ np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]]).T)
    return np.array(out)


runs = [run(seed) for seed in (1, 2, 3)]

# --- where does a ligand go? --------------------------------------------------
found = boonza.sites(s, runs, ligand="not polymer and noh")
print(found.summary())

# --- what is the pocket made of? ----------------------------------------------
pocket = boonza.site_pocket(s, runs, found, 0, protein="protein and noh", share=0.3)
residues = sorted({int(s.residues["resid"][s.atoms["residue"][a]]) for a in pocket})
print(f"\npocket of site 0: {len(pocket)} atoms in residues {residues}")

# --- how does one fragment sit there? -----------------------------------------
rows = found.frames(0, run=0)
copy = int(np.bincount(rows[:, 1]).argmax())
frames = rows[rows[:, 1] == copy][:, 2]
p = boonza.poses(s, runs[0][frames], ligand=f"fragid {int(frag[copy])} and noh", pocket=pocket)
print(f"\nfragment {copy} in site 0, over {len(frames)} frames:")
print(p.summary())

# --- how long does it stay? ---------------------------------------------------
rate = boonza.kinetics(s, found, 0, interval_ns=0.1, temperature=298.0, bootstrap=100)
print()
print(rate.summary())

# --- do different molecules agree? --------------------------------------------
f = boonza.site_interactions(s, runs, found, 0, ligand="not polymer and noh")
same = boonza.similarity_matrix(f.values)
print(f"\n{len(f)} fragments visited site 0; they touch")
for row, (r, c) in zip(f.values, f.where.tolist(), strict=True):
    keep = np.flatnonzero(row >= 0.5)
    names = [f.names(s)[i] for i in keep]
    print(f"  run {r} fragment {c}: {' '.join(names) if names else '(nothing above half)'}")
print(f"agreement between them: {same[np.triu_indices(len(f), 1)].mean():.2f}")

# --- what should go there? ----------------------------------------------------
maps = boonza.feature_maps(s, runs, ligand="not polymer")
spots = boonza.hotspots(maps, enrichment=20.0)
print(f"\n{len(spots)} hotspots; site 0 asks for")
for h in boonza.wanted(spots, found[0].center)[:4]:
    print(f"  {h.family:12s} {h.ligands} fragments, {h.enrichment:.0f}x bulk, "
          f"radius {h.radius:.1f} A")  # fmt: skip
boonza.write_hotspots(OUT / "hotspots.pdb", spots)
maps["Acceptor"].write_dx(OUT / "acceptor.dx")
f.write_structure(s, OUT / "touched.pdb")
print(f"\nwrote hotspots.pdb, acceptor.dx and touched.pdb to {OUT.name}/")
