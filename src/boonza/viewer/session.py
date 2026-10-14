"""One model of the viewer: a system, optionally a trajectory, and what to draw.

boonza does the reading, bonds, selections, secondary structure and periodic
fixing (Glue); coarse-grained systems (Martini, SIRAH, any beads) are told
apart by their spacing and drawn with bead radii.

What a model shows when it opens is the protein as a cartoon, the pocket
(whole residues within 5 A of the ligand, hydrogens only where polar) and
the ligand, as boonza modes' PyMOL session shows a pose.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

CACHE = 64  # glued frames kept on the server

#: the ligand when none is named: what is not polymer, solvent, ions or membrane
LIGAND = "not (protein or nucleic or water or ions or lipid)"
#: atoms drawn as sticks: heavy atoms, and hydrogens only on N, O and S
POLAR = "(not hydrogen or withinbonds 1 of (element N O S))"
POCKET = 5.0  # A: residues this near the ligand are the pocket
POLAR_CONTACT = 3.5  # A: N/O pairs between ligand and the rest drawn as contacts

# Coarse-grained models: the selection keywords (protein, water, ions) are
# atomistic, so residues are picked by name.  The cartoon traces the bead at the
# alpha carbon (Martini BB, SIRAH GC) or the phosphate (SIRAH PX).
CG_TRACE = {"BB": 1, "GC": 1, "PX": 2}
CG_BACKBONE = "BB GN GC GO PX"
CG_IONS = 'ION NA CL "NA+" "CL-" K "K+" CA "CA+" "CA2+" MG ZN'
CG_SOLVENT = "W PW WF SOL HOH WAT TIP3 " + CG_IONS
CG_POLYMER = f"(same residue as name {' '.join(CG_TRACE)})"
CG_LIGAND = f"not {CG_POLYMER} and not resname {CG_SOLVENT}"
# beads touch at about sigma (4-5 A), so a bead's pocket reaches further than an atom's
CG_POCKET = 6.0
# a bead with no size from its force field or type: Martini's regular bead,
# sigma 4.7 A
CG_RADIUS = 2.35


def is_coarse_grained(system) -> bool:
    """Whether particles sit farther apart than bonded atoms do.

    Every atom of an atomistic structure has a neighbor within 1.65 A (C-H 1.1,
    C-C 1.54, water O-H 0.96), except lone ions.  Beads are 2.5-5 A apart (SIRAH
    keeps N-CA-O spacing on its backbone, so not every bead), whatever model or
    names they use.
    """
    from ..spatial import pairs_within

    n = system.natoms
    if n < 2:
        return False
    i, j, _ = pairs_within(np.asarray(system.positions, np.float64), 1.65)
    near = np.zeros(n, bool)
    near[i] = near[j] = True
    # leave out ions and lone beads, unless nothing is bonded (e.g. a .gro)
    many = np.bincount(system.fragids)[system.fragids] > 1 if system.nbonds else np.ones(n, bool)
    return bool(many.any() and near[many].mean() < 0.8)


#: P-SEA's reference geometry for alpha carbons: (d2, d3, d4, dihedral, angle) of a helix
#: and of a strand.  Martini's BB bead sits at the backbone's centroid (N, CA, C, O),
#: not on the alpha carbon; its references are P-SEA's shifted by the difference
#: between BB and CA in an ideal helix and strand (boonza.peptide, "helix" and
#: "sheet": helix d2 4.81 vs 5.66 A, d3 4.98 vs 5.23, d4 6.66 vs 6.81, dihedral 60.2
#: vs 47.9, angle 97.3 vs 93.0; a strand barely differs).  The tolerances are P-SEA's.
PSEA = {
    "CA": {"H": (5.5, 5.3, 6.4, 50.0, 89.0), "E": (6.7, 9.9, 12.4, -170.0, 124.0)},
    "BB": {"H": (4.65, 5.05, 6.25, 62.3, 93.3), "E": (6.72, 10.0, 12.47, -170.0, 124.0)},
}
PSEA_TOLERANCE = (0.5, 0.5, 0.6, 20.0, 12.0)  # helix
PSEA_TOLERANCE_E = (0.6, 0.9, 1.1, 45.0, 14.0)  # strand


def trace_secondary_structure(pos, chain, kind, bead: str = "CA") -> list[str]:
    """Helix (H), strand (E) or coil (C) per residue from its traced bead alone (the
    alpha carbon, Martini BB, SIRAH GC), as P-SEA assigns it: the distances to the
    beads two, three and four along and the virtual bond angle and dihedral.  Helices
    run at least 5 residues, strands 3, and a strand needs a partner strand (a bead of
    another strand within 5.5 A), so a stretched coil is not called one.  ``pos``:
    one position per residue (NaN where none), ``chain``: per residue, ``kind``: 1 for
    amino acids.  ``bead``: "CA" (alpha carbons, SIRAH GC) or "BB" (Martini)."""
    n = len(pos)
    ss = np.array(["C" if k == 1 else "" for k in kind], dtype=object)
    ok = np.isfinite(pos).all(1) & (np.asarray(kind) == 1)

    def linked(i, j):  # i..j in one chain, nothing missing, no gap
        if j >= n or not ok[i : j + 1].all() or len(set(chain[i : j + 1])) > 1:
            return False
        return bool((np.linalg.norm(np.diff(pos[i : j + 1], axis=0), axis=1) < 4.6).all())

    def dist(i, j):
        return float(np.linalg.norm(pos[i] - pos[j]))

    def dihedral(a, b, c, d):
        b0, b1, b2 = pos[a] - pos[b], pos[c] - pos[b], pos[d] - pos[c]
        b1 = b1 / np.linalg.norm(b1)
        v, w = b0 - b0 @ b1 * b1, b2 - b2 @ b1 * b1
        return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))

    def angle(a, b, c):
        u, v = pos[a] - pos[b], pos[c] - pos[b]
        return float(
            np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
        )

    ref = PSEA[bead]
    hd2, hd3, hd4, ha, ht = ref["H"]
    ed2, ed3, ed4, ea, et = ref["E"]
    th, te = PSEA_TOLERANCE, PSEA_TOLERANCE_E
    helix = np.zeros(n, bool)
    strand = np.zeros(n, bool)
    for i in range(n):
        if linked(i, i + 4):
            d2, d3, d4 = dist(i, i + 2), dist(i, i + 3), dist(i, i + 4)
            helix[i] |= abs(d2 - hd2) < th[0] and abs(d3 - hd3) < th[1] and abs(d4 - hd4) < th[2]
            strand[i] |= abs(d2 - ed2) < te[0] and abs(d3 - ed3) < te[1] and abs(d4 - ed4) < te[2]
        if i >= 1 and linked(i - 1, i + 2):
            a, t = dihedral(i - 1, i, i + 1, i + 2), angle(i - 1, i, i + 1)
            helix[i] |= abs(a - ha) < th[3] and abs(t - ht) < th[4]
            strand[i] |= abs(((a - ea + 180) % 360) - 180) < te[3] and abs(t - et) < te[4]

    def runs(flags, shortest):
        out = np.zeros(n, bool)
        i = 0
        while i < n:
            j = i
            while j < n and flags[j] and (j == i or linked(j - 1, j)):
                j += 1
            if j - i >= shortest:
                out[i:j] = True
            i = max(j, i + 1)
        return out

    h = runs(helix, 5)
    e = runs(strand & ~h, 3)
    # a strand pairs with another strand
    idx = np.flatnonzero(e)
    for i in idx:
        partners = [j for j in idx if abs(j - i) > 3 and dist(i, j) < 5.5]
        if not partners:
            e[i] = False
    e = runs(e, 3)
    ss[h] = "H"
    ss[e & ~h] = "E"
    return [str(x) for x in ss]


def bead_radii(system) -> np.ndarray:
    """Radius of each bead: sigma/2 from its force field, else from its type's table."""
    from ..pockets.core import bead_radii as table_radii
    from ..sites import particle_radii

    ids = np.arange(system.natoms)
    if "nonbonded" in system.table_names:
        r = particle_radii(system, ids)
    elif "type" in system.atoms.props:
        types = system.atoms["type"]
        r = max((table_radii(types, m) for m in ("martini3", "martini2", "sirah")),
                key=lambda r: np.isfinite(r).sum())  # fmt: skip
    else:
        r = np.full(system.natoms, np.nan)
    return np.where(np.isfinite(r) & (r > 0.5), r, CG_RADIUS)


def default_reps(system, cg: bool, ligand: str | None = None, carbon: str = "green") -> list:
    """Cartoon, pocket and ligand (and ions), each where its selection has atoms."""
    if cg:
        lig = ligand or CG_LIGAND
        reps = [
            (CG_POLYMER, "cartoon", "chain"),
            (f"(same residue as ({CG_POLYMER} and within {CG_POCKET:g} of ({lig}))) "
             f"and not ({lig})", "ballstick", "restype"),
            (f"({lig}) and not resname {CG_SOLVENT}", "ballstick", carbon),
            (f"resname {CG_IONS}", "spheres", "resname"),
        ]  # fmt: skip
    else:
        lig = ligand or LIGAND
        reps = [
            ("protein or nucleic", "cartoon", "chain"),
            # closed off: 'same residue as' takes everything after it
            (f"(same residue as ((protein or nucleic) and within {POCKET:g} of ({lig}))) "
             f"and not ({lig}) and {POLAR}", "sticks", "element"),
            (f"(same residue as ({lig})) and {POLAR}", "sticks", f"element/{carbon}"),
            ("ions and not (" + lig + ")", "spheres", "element"),
        ]  # fmt: skip
    out = [{"sel": sel, "style": style, "color": color}
           for sel, style, color in reps if len(system.select(sel))]  # fmt: skip
    if any(r["color"] == carbon or r["color"].endswith("/" + carbon) for r in out):
        # with a ligand, a neutral cartoon: its carbons are the colour that stands out
        out[0]["color"] = "white" if out[0]["style"] == "cartoon" else out[0]["color"]
    if not any(r["style"] != "spheres" for r in out):  # no polymer, no ligand: just atoms
        out = [{"sel": "all", "style": "ballstick" if cg else "sticks", "color": "element"}]
    return out


class MemoryTrajectory:
    """Frames held in memory (the structures of a multi-structure file), read as a
    trajectory is: positions (nframes, natoms, 3), boxes (nframes, 3, 3) or none."""

    def __init__(self, positions, boxes=None):
        self.positions = np.asarray(positions, np.float32)
        n = len(self.positions)
        self.boxes = np.zeros((n, 3, 3)) if boxes is None else np.asarray(boxes, float)
        self.natoms = self.positions.shape[1]

    def __len__(self) -> int:
        return len(self.positions)

    def read(self, indices, atoms=None):
        from ..trajectory import Frames

        idx = np.asarray(indices, np.int64)
        pos = self.positions[idx] if atoms is None else self.positions[idx][:, atoms]
        zeros = np.zeros(len(idx))
        return Frames(idx, pos, self.boxes[idx], zeros, zeros.astype(np.int64))


class Session:
    """A system, optionally a trajectory, and the Glue applied to its frames.

    ``atoms``: a selection kept for viewing (default all); the trajectory is
    read for those atoms only.  ``whole``, ``center`` and ``fit`` set up a
    :class:`boonza.Glue` on the viewed atoms (``fit`` onto the first frame);
    ``whole`` defaults to on with a trajectory.  ``ligand``: the selection the
    pocket and contacts are about, ``carbon`` its carbons' colour.  ``reps``
    replace the default representations, ``radii`` the atoms' radii (spheres
    from a file of points), ``cg`` the coarse-grained guess.  ``sizes``: radii
    for some atoms only, by selection (a site's points kept with its protein).
    """

    def __init__(self, system, trajectory=None, atoms=None, whole: bool | None = None,
                 center: str | None = None, fit: str | None = None, title: str = "",
                 ligand: str | None = None, carbon: str = "green", reps=None, radii=None,
                 cg: bool | None = None, sizes: dict[str, float] | None = None):  # fmt: skip
        self.sizes = dict(sizes or {})
        if atoms is None:
            self.ids = None
            self.system = system
        else:
            ids = system.select(atoms).ids if isinstance(atoms, str) else np.asarray(atoms)
            if len(ids) == 0:
                raise ValueError(f"no atoms in {atoms!r}")
            self.ids = np.asarray(ids, dtype=np.int64)
            self.system = system.clone(self.ids)
        self.trajectory = trajectory
        if trajectory is not None and trajectory.natoms != system.natoms:
            raise ValueError(f"the trajectory has {trajectory.natoms} atoms, "
                             f"the structure {system.natoms}")  # fmt: skip
        self.title = title
        self.cg = is_coarse_grained(self.system) if cg is None else bool(cg)
        # a structure without bond orders (PDB, mmCIF) gets them from names and geometry
        self.guessed_orders = 0
        if not self.cg and radii is None and self.system.nbonds:
            from ..bondorders import guess_bond_orders

            self.guessed_orders = guess_bond_orders(self.system)
        self.ligand = ligand
        self.carbon = carbon
        self.reps = reps
        self.radii = None if radii is None else np.asarray(radii, float)
        self.version = 0  # bumped by every edit, so the page knows to reload
        self._lock = threading.RLock()
        self._cache: OrderedDict = OrderedDict()
        self._ss: dict = {}
        self.set_glue(trajectory is not None if whole is None else whole, center, fit)

    @property
    def nframes(self) -> int:
        return len(self.trajectory) if self.trajectory is not None else 1

    def set_glue(self, whole: bool = True, center: str | None = None,
                 fit: str | None = None) -> None:  # fmt: skip
        """Change the periodic fixing and fitting; frames are recomputed."""
        from ..glue import Glue

        s = self.system
        for sel in (center, fit):
            if sel and len(s.select(sel)) == 0:
                raise ValueError(f"no atoms in {sel!r}")
        with self._lock:
            self.whole, self.center, self.fit = bool(whole), center or None, fit or None
            if self.whole or self.center or self.fit:
                # fit onto the first frame, whatever frame is asked for first
                reference = self._raw(0)[0] if self.fit else None
                self.glue = Glue(s, center=self.center, fit=self.fit, reference=reference,
                                 whole="all" if self.whole else None,
                                 wrap=self.center is not None)  # fmt: skip
            else:
                self.glue = None
            self._cache.clear()
            self._ss.clear()

    def _raw(self, i: int):
        if self.trajectory is None:
            return self.system.positions, self.system.cell
        f = self.trajectory.read([i], atoms=self.ids)
        return f.positions[0], f.boxes[0]

    def frame(self, i: int):
        """Positions (natoms, 3) float32 and box (3, 3) of frame ``i``, glued."""
        if not 0 <= i < self.nframes:
            raise IndexError(f"frame {i} of {self.nframes}")
        with self._lock:
            if i in self._cache:
                self._cache.move_to_end(i)
                return self._cache[i]
            pos, box = self._raw(i)
            if self.glue is not None:
                pos, box = self.glue(pos, box)
            out = (np.ascontiguousarray(pos, dtype="<f4"), np.asarray(box, dtype=np.float64))
            self._cache[i] = out
            if len(self._cache) > CACHE:
                self._cache.popitem(last=False)
            return out

    def select(self, text: str, frame: int = 0) -> np.ndarray:
        pos, box = self.frame(frame)
        return self.system.select(text, pos=pos.astype(np.float64), cell=box).ids

    def ss(self, frame: int = 0) -> list[str]:
        """Simplified DSSP per residue ("H", "E", "C", "" for non-protein)."""
        from ..secondary import dssp

        with self._lock:
            if frame not in self._ss:
                if self.cg and self.radii is None:  # beads: from the trace alone
                    codes = self._trace_ss(frame)
                elif self.cg or len(self.system.select("protein and name CA")) == 0:
                    codes = [""] * self.system.nresidues
                else:
                    pos, box = self.frame(frame)
                    row = dssp(self.system, pos.astype(np.float64), simplified=True,
                               box=box if box.any() else None)[0]  # fmt: skip
                    codes = ["" if c == "NA" else str(c) for c in row]
                if len(self._ss) > CACHE:
                    self._ss.clear()
                self._ss[frame] = codes
            return self._ss[frame]

    def _trace_ss(self, frame: int) -> list[str]:
        s = self.system
        names = [str(x) for x in s.atoms["name"]]
        atom_res = np.asarray(s.atoms["residue"])
        pos = np.full((s.nresidues, 3), np.nan)
        kind = np.zeros(s.nresidues, np.int64)
        frame_pos = self.frame(frame)[0]
        for a, name in enumerate(names):
            if CG_TRACE.get(name) == 1:
                pos[atom_res[a]] = frame_pos[a]
                kind[atom_res[a]] = 1
        chain = np.asarray(s.residues["chain"])
        bead = "BB" if "BB" in names else "CA"
        return trace_secondary_structure(pos, chain, kind, bead)

    def contacts(self, frame: int = 0) -> list:
        """Polar contacts of the ligand: N/O pairs between it and the rest (not water)
        within POLAR_CONTACT, as ``[i, j, distance]``.  A distance, not an H-bond
        test, so structures without hydrogens show them too."""
        if self.cg:
            return []
        lig = f"same residue as ({self.ligand or LIGAND})"
        a = self.select(f"({lig}) and element N O", frame)
        if not len(a):
            return []
        b = self.select(f"(not ({lig})) and not water and element N O and "
                        f"within {POLAR_CONTACT:g} of ({lig})", frame)  # fmt: skip
        if not len(b):
            return []
        pos = self.frame(frame)[0].astype(np.float64)
        d = np.linalg.norm(pos[a][:, None] - pos[b][None], axis=2)
        i, j = np.nonzero(d <= POLAR_CONTACT)
        return [[int(a[x]), int(b[y]), round(float(d[x, y]), 2)] for x, y in zip(i, j, strict=True)]

    def topology(self) -> dict:
        """Everything the page needs once: atoms, residues, chains, bonds, defaults."""
        from ..elements import radii, symbol
        from .properties import charges, hydrophobicity

        s = self.system
        A, R, C = s._atoms, s._residues, s._chains
        anum = np.asarray(A.column("anum"), dtype=np.int64)
        atom_res = np.asarray(A.column("residue"), dtype=np.int64)
        res_chain = np.asarray(R.column("chain"), dtype=np.int64)

        # per residue: the atom traced by the cartoon and the one that orients it
        trace = np.full(s.nresidues, -1, np.int64)
        guide = np.full(s.nresidues, -1, np.int64)
        kind = np.zeros(s.nresidues, np.int64)  # 1 protein, 2 nucleic
        if self.cg:
            names = np.array([str(x) for x in A.column("name")])
            for name, k in CG_TRACE.items():
                ids = np.flatnonzero(names == name)
                trace[atom_res[ids]] = ids
                kind[atom_res[ids]] = k
        elif self.radii is None:
            for sel, k, which in (("protein and name CA", 1, trace),
                                  ("protein and name O", 1, guide),
                                  ("nucleic and name P", 2, trace),
                                  ("nucleic and name \"C1'\"", 2, guide)):  # fmt: skip
                ids = s.select(sel).ids
                which[atom_res[ids]] = ids
                if which is trace:
                    kind[atom_res[ids]] = k

        if self.radii is not None:
            radius = self.radii
        else:
            radius = bead_radii(s) if self.cg else radii(anum)
        radius = np.array(radius, float)
        for sel, r in self.sizes.items():
            radius[s.select(sel).ids] = r
        carbon = self.carbon
        reps = self.reps or default_reps(s, self.cg, self.ligand, carbon)
        # MOE's classes, for the R, S and L picking buttons: 0 receptor, 1 ligand, 2 solvent
        klass = np.ones(s.natoms, np.int64)
        if self.radii is None:
            poly = CG_POLYMER if self.cg else "protein or nucleic"
            solvent = f"resname {CG_SOLVENT}" if self.cg else "water or ions"
            klass[s.select(poly).ids] = 0
            klass[s.select(solvent).ids] = 2
        bonds = s.bonds
        if self.radii is None:
            q = charges(s, self.cg)
        else:
            q = np.zeros(s.natoms)
        return {
            "title": self.title,
            "path": str(self.path) if getattr(self, "path", None) else "",
            "editable": self.trajectory is None and self.radii is None,
            "guessed_orders": self.guessed_orders,
            "version": self.version,
            "natoms": s.natoms,
            "nframes": self.nframes,
            "cg": self.cg,
            "ligand": self.ligand or (CG_LIGAND if self.cg else LIGAND),
            "carbon": self.carbon,  # the ligand's carbons' colour
            "solvent": f"resname {CG_SOLVENT}" if self.cg else "water or ions",
            "atoms": {
                "name": [str(x) for x in A.column("name")],
                "element": [symbol(int(z)) if z > 0 else "X" for z in anum.tolist()],
                "radius": np.round(radius, 3).tolist(),
                "residue": atom_res.tolist(),
                "bfactor": (
                    np.round(np.asarray(s.atoms["bfactor"], float), 2).tolist()
                    if "bfactor" in s.atoms.props
                    else [0.0] * s.natoms
                ),
                "charge": np.asarray(A.column("formal_charge"), dtype=np.int64).tolist(),
                # what a surface is coloured by (properties.py)
                "q": np.round(q, 3).tolist(),
                "lipo": np.round(hydrophobicity(s, q, self.cg), 2).tolist(),
                "class": klass.tolist(),
                "fragment": np.asarray(s.fragids, dtype=np.int64).tolist(),
            },  # fmt: skip
            "residues": {
                "name": [str(x) for x in R.column("name")],
                "resid": [int(x) for x in R.column("resid")],
                "insertion": [str(x) for x in R.column("insertion")],
                "chain": res_chain.tolist(),
                "trace": trace.tolist(),
                "guide": guide.tolist(),
                "kind": kind.tolist(),
            },
            "chains": {
                "name": [str(x) for x in C.column("name")],
                "segid": [str(x) for x in C.column("segid")],
            },
            "bonds": {
                "i": np.asarray(bonds["i"], dtype=np.int64).tolist(),
                "j": np.asarray(bonds["j"], dtype=np.int64).tolist(),
                "order": np.asarray(bonds["order"], dtype=np.int64).tolist(),
            },
            "ss": self.ss(0),
            "glue": {"whole": self.whole, "center": self.center, "fit": self.fit},
            "reps": reps,
        }

    # --- editing -----------------------------------------------------------------

    UNDO = 30  # edits that can be undone

    def _editable(self) -> None:
        if self.trajectory is not None:
            raise ValueError("editing works on a structure, not a trajectory: "
                             "save a frame and open that")  # fmt: skip
        if self.radii is not None:
            raise ValueError("these are spheres from a file, not a molecule")

    def _remember(self) -> None:
        self._undo = getattr(self, "_undo", [])
        self._undo.append((self.system.copy(), getattr(self, "topo_version", 0)))
        del self._undo[: -self.UNDO]
        self._redo = []  # a new edit ends what could be redone

    def _changed(self, topology: bool) -> dict:
        """After an edit: caches dropped, versions bumped (the page reloads)."""
        self.version += 1
        if topology:
            self.topo_version = getattr(self, "topo_version", 0) + 1
        self.set_glue(self.whole, None, None)
        return {"version": self.version, "topology": topology, "natoms": self.system.natoms,
                "undo": len(getattr(self, "_undo", []))}  # fmt: skip

    def edit(self, op: str, atoms=(), value=None, hydrogens: bool = True) -> dict:
        """One edit (see boonza.viewer.edit): ``op`` is element, charge, bond, delete or
        hydrogens; ``hydrogens`` adjusts the touched atoms' hydrogens after it.  The
        result's ``picked`` is where ``atoms`` went (-1 for deleted ones)."""
        from . import edit

        self._editable()
        s = self.system
        atoms = [int(a) for a in atoms]
        if any(not 0 <= a < s.natoms for a in atoms):
            raise IndexError("atom index out of range")
        if not atoms and op not in ("molecule", "cap"):
            raise ValueError("pick atoms first")
        if op == "molecule":  # a new molecule from SMILES, at the view's centre
            from .edit import add_molecule

            centre, smiles = value["centre"], value["smiles"]
            with self._lock:
                self._remember()
                try:
                    add_molecule(self.system, smiles, np.asarray(centre, float))
                except Exception:
                    self.system = self._undo.pop()[0]
                    raise
                return {**self._changed(topology=True), "picked": []}
        if op == "cap" and not atoms:  # every chain
            atoms = list(range(s.natoms))
        with self._lock:
            self._remember()
            s.atoms["view_edit"] = np.arange(1, s.natoms + 1, dtype=np.int64)  # follows atoms
            touched = list(atoms)
            try:
                if op == "element":
                    edit.set_element(s, atoms, value)
                elif op == "charge":
                    edit.set_charge(s, atoms, int(value))
                elif op == "bond":
                    if len(atoms) != 2:
                        raise ValueError("pick the two atoms of the bond")
                    edit.set_bond(s, atoms[0], atoms[1], int(value))
                elif op == "delete":
                    near = {b for a in atoms for b, _ in edit.neighbors(s, a)} - set(atoms)
                    index = edit.delete(s, atoms)
                    touched = [int(index[b]) for b in near if index[b] >= 0]
                elif op == "hydrogens":
                    hydrogens = True
                elif op == "attach":  # a functional group in place of a hydrogen
                    a = atoms[0]
                    if s.atoms["anum"][a] == 1:
                        anchor, leaving = edit.neighbors(s, a)[0][0], a
                    else:
                        hs = [b for b, _ in edit.neighbors(s, a) if s.atoms["anum"][b] == 1]
                        if not hs:
                            raise ValueError("select a hydrogen, or an atom that has one")
                        anchor, leaving = a, hs[0]
                    smiles = edit.GROUPS.get(str(value), str(value))
                    edit.attach(s, anchor, leaving, smiles)
                    hydrogens = False
                elif op == "mutate":
                    for r in sorted({int(s.atoms["residue"][a]) for a in atoms}):
                        edit.mutate(s, r, str(value))
                    hydrogens = False
                elif op == "cap":
                    chains = sorted(
                        {int(s.residues["chain"][s.atoms["residue"][a]]) for a in atoms}
                    )
                    for c in chains:
                        for end in {"N": "N", "C": "C", "both": "NC"}[str(value)]:
                            edit.cap(s, c, end)
                    hydrogens = False
                elif op == "name":
                    if len(atoms) != 1 or not str(value).strip():
                        raise ValueError("pick one atom and give it a name")
                    names = np.asarray(s.atoms["name"]).astype(object)
                    names[atoms[0]] = str(value).strip()[:8]
                    s.atoms["name"] = names.astype(str)
                    hydrogens = False
                else:
                    raise ValueError(f"unknown edit {op!r}")
                if hydrogens:
                    edit.adjust_hydrogens(s, [a for a in touched if a < s.natoms])
            except Exception:
                self.system = self._undo.pop()[0]
                raise
            # where the picked atoms are now (-1: deleted); added hydrogens carry 0
            where = {int(k) - 1: i for i, k in enumerate(s.atoms["view_edit"].tolist()) if k}
            s.atoms.del_prop("view_edit")
            return {**self._changed(topology=True), "picked": [where.get(a, -1) for a in atoms]}

    def undo(self) -> dict:
        if not getattr(self, "_undo", None):
            raise ValueError("nothing to undo")
        with self._lock:
            self._redo = getattr(self, "_redo", [])
            self._redo.append((self.system, getattr(self, "topo_version", 0)))
            self.system, topo = self._undo.pop()
            self.topo_version = topo
            return self._changed(topology=True)

    def redo(self) -> dict:
        if not getattr(self, "_redo", None):
            raise ValueError("nothing to redo")
        with self._lock:
            self._undo.append((self.system, getattr(self, "topo_version", 0)))
            self.system, topo = self._redo.pop()
            self.topo_version = topo
            return self._changed(topology=True)

    def _geometry_edit(self, change) -> dict:
        """A change of positions only (``change(system)`` returns a report)."""
        self._editable()
        with self._lock:
            self._remember()
            try:
                report = change(self.system)
            except Exception:
                self.system = self._undo.pop()[0]
                raise
            return {**self._changed(topology=False), **(report or {})}

    def set_positions(self, atoms, xyz) -> dict:
        """Put ``atoms`` at ``xyz`` (A): atoms moved with the mouse."""
        atoms = np.asarray(atoms, np.int64)
        xyz = np.asarray(xyz, float).reshape(-1, 3)
        if len(atoms) != len(xyz):
            raise ValueError("one position per atom")

        def change(s):
            pos = np.array(s.positions, float)
            pos[atoms] = xyz
            s.positions = pos

        return self._geometry_edit(change)

    def set_distance(self, atoms, length: float) -> dict:
        from .edit import set_distance

        a, b = (int(x) for x in atoms)
        return self._geometry_edit(lambda s: {"value": round(set_distance(s, a, b, length), 4)})

    def set_angle(self, atoms, degrees: float) -> dict:
        from .edit import set_angle

        a, b, c = (int(x) for x in atoms)
        return self._geometry_edit(lambda s: {"value": round(set_angle(s, a, b, c, degrees), 3)})

    def align(self, reference, method: str = "matchmaker", sel: str | None = None) -> dict:
        """Superpose this model onto ``reference`` (a Session) with boonza: matchmaker
        (sequence alignment and pruning, as ChimeraX), cealign (structure only) or
        superpose (``sel``, default alpha carbons, paired by sequence)."""
        from ..align import superpose
        from ..ce import cealign
        from ..matchmaker import matchmaker

        if method not in ("matchmaker", "cealign", "superpose"):
            raise ValueError("method: matchmaker, cealign or superpose")
        ref = reference.system

        def change(s):
            if method == "matchmaker":
                r = matchmaker(s, ref)
                a, b = r.aligned_mobile, r.aligned_reference
                pairs = [(x, y) for x, y in zip(a, b, strict=True) if x != "-" and y != "-"]
                same = sum(1 for x, y in pairs if x == y)
                return {"rmsd": round(float(r.rmsd), 3), "full_rmsd": round(float(r.full_rmsd), 3),
                        "pairs": int(len(r.kept)), "paired": int(len(r.mobile_atoms)),
                        "identity": round(same / len(pairs), 3) if pairs else None}  # fmt: skip
            if method == "cealign":
                r = cealign(s, ref)
                return {"rmsd": round(float(r.rmsd), 3), "pairs": int(len(r.mobile_atoms)),
                        "z_score": round(float(r.z_score), 2)}  # fmt: skip
            r = superpose(s, ref, sel=sel or "protein and name CA")
            return {"rmsd": round(float(r.rmsd), 3), "pairs": int(r.n_used)}

        out = self._geometry_edit(change)
        return {**out, "method": method}

    def atom_info(self, a: int, assign: bool = False) -> dict:
        """What an atom is, and its force field: the partial charge and Lennard-Jones
        sigma (A) and epsilon (kcal/mol) from the file's force field, else from the one
        Minimize uses once it is assigned (``assign`` assigns it now)."""
        from .edit import neighbors

        s = self.system
        a = int(a)
        if not 0 <= a < s.natoms:
            raise IndexError(f"atom {a} of {s.natoms}")
        r = int(s.atoms["residue"][a])
        out = {"index": a, "name": str(s.atoms["name"][a]), "anum": int(s.atoms["anum"][a]),
               "formal_charge": int(s.atoms["formal_charge"][a]),
               "mass": round(float(s.atoms["mass"][a]), 3),
               "residue": f"{s.residues['name'][r]}{int(s.residues['resid'][r])}",
               "bonds": [[int(b), int(o)] for b, o in neighbors(s, a)]}  # fmt: skip
        source, sys_, k = None, None, a
        if "nonbonded" in s.table_names:
            source, sys_ = "file", s
        elif self.radii is None and (assign or getattr(self, "_param", None) is not None):
            sys_ = self._parameterized()
            order = np.asarray(sys_.atoms["view_index"])
            k = int(np.flatnonzero(order == a)[0])
            source = "assigned"
        if sys_ is not None:
            out["charge"] = round(float(sys_.atoms["charge"][k]), 4)
            nb = sys_.table("nonbonded")
            rows = np.flatnonzero(np.asarray(nb.atoms)[:, 0] == k)
            if len(rows):
                pid = int(np.asarray(nb.param_ids)[rows[0]])
                for key in ("sigma", "epsilon", "type"):
                    if key in nb.params.props:
                        v = nb.params[key][pid]
                        out[key] = str(v) if key == "type" else round(float(v), 4)
            out["force_field"] = source
        return out

    def set_ligand(self, atoms) -> dict:
        """These atoms (whole residues) are the ligand: pocket, contacts, minimize."""
        atoms = [int(a) for a in atoms]
        if not atoms:
            raise ValueError("select the ligand's atoms first")
        self.ligand = "same residue as index " + " ".join(map(str, atoms))
        return {"ligand": self.ligand}

    def save(self, path) -> str:
        from ..io import save

        save(self.system, path)
        return str(path)

    def _parameterized(self, log=print):
        """The force-field copy for energies, kept until the topology changes."""
        from .edit import parameterize

        key = getattr(self, "topo_version", 0)
        cached = getattr(self, "_param", None)
        if cached is None or cached[0] != key:
            self._param = (key, parameterize(self.system, log=log))
        p = self._param[1]
        p.positions = np.asarray(self.system.positions)[np.asarray(p.atoms["view_index"])]
        return p

    def dihedral(self, atoms) -> dict:
        """The a-b-c-d angle and the atoms that turn with it.  Two bonded atoms are
        the bond: its outer atoms are chosen (boonza.viewer.edit.dihedral_from_bond)."""
        from .edit import dihedral_angle, dihedral_from_bond, dihedral_name, rotating_side

        self._editable()
        atoms = [int(x) for x in atoms]
        if len(atoms) == 2:
            atoms = dihedral_from_bond(self.system, *atoms)
        elif len(atoms) != 4:
            raise ValueError("pick a bond (two atoms) or four atoms")
        a, b, c, d = atoms
        moving, sign = rotating_side(self.system, b, c)
        return {"angle": round(dihedral_angle(self.system.positions, a, b, c, d), 2),
                "atoms": atoms, "name": dihedral_name(self.system, atoms),
                "moving": moving.tolist(), "sign": sign}  # fmt: skip

    def set_dihedral(self, atoms, degrees: float) -> dict:
        from .edit import set_dihedral

        self._editable()
        with self._lock:
            self._remember()
            angle = set_dihedral(self.system, *(int(x) for x in atoms), float(degrees))
            return {**self._changed(topology=False), "angle": round(angle, 2)}

    def dihedral_scan(self, atoms, step: float = 10.0) -> dict:
        from .edit import dihedral_scan

        self._editable()
        with self._lock:
            return dihedral_scan(self.system, *(int(x) for x in atoms), self._parameterized(),
                                 step=step, alone=self._alone(int(atoms[1])))  # fmt: skip

    def _alone(self, a: int) -> list[int]:
        """What "isolated" means for atom a: its residue in a polymer, else its molecule."""
        s = self.system
        if self.radii is None and len(s.select(f"index {a} and (protein or nucleic)")):
            return [int(x) for x in s.residue_atoms(int(s.atoms["residue"][a]))]
        frag = s.fragids
        return np.flatnonzero(frag == frag[a]).tolist()

    def ligand2d(self, atoms=(), dark: bool = True, width: int = 360, height: int = 260) -> dict:
        """A 2D drawing (SVG, RDKit) of the ligand: the molecule of the selected atoms,
        else the first of the ligand selection; heavy atoms, hydrogens implicit.
        ``atoms`` (selected) are highlighted; ``coords`` say where each atom is drawn."""
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D

        from ..chem import to_rdkit

        s = self.system
        frag = s.fragids
        lig = [
            a for a in atoms if len(s.select(f"index {a} and not (protein or nucleic or water)"))
        ]
        if not lig:
            ids = s.select(f"same residue as ({self.ligand or LIGAND})").ids
            if not len(ids):
                raise ValueError("no ligand: choose one (Ligand | Choose Ligand)")
            lig = [int(ids[0])]
        mol_atoms = np.flatnonzero(np.isin(frag, frag[lig]))
        polymer = set(s.select("protein or nucleic").ids.tolist())
        if polymer & set(mol_atoms.tolist()):  # bonded to the protein (heme, a covalent
            # ligand): the ligand's own residues
            res = set(np.asarray(s.atoms["residue"])[lig].tolist())
            mol_atoms = np.flatnonzero(np.isin(np.asarray(s.atoms["residue"]), list(res)))
        heavy = [int(a) for a in mol_atoms if s.atoms["anum"][a] > 1]
        if len(heavy) > 300:
            raise ValueError("that molecule is too large for a 2D drawing")
        try:
            mol = to_rdkit(s, heavy, sanitize=True, implicit_hydrogens=True)
        except Exception:  # a metal RDKit's valence rules reject (heme's iron)
            from rdkit import Chem

            mol = to_rdkit(s, heavy, sanitize=False, implicit_hydrogens=False)
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)
        AllChem.Compute2DCoords(mol)
        draw = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = draw.drawOptions()
        if dark:
            rdMolDraw2D.SetDarkMode(opts)
        opts.setBackgroundColour((0, 0, 0, 0))
        opts.clearBackground = False
        local = {a: k for k, a in enumerate(heavy)}
        hi = [local[a] for a in atoms if a in local]
        draw.DrawMolecule(mol, highlightAtoms=hi,
                          highlightAtomColors={k: (1.0, 0.45, 0.85) for k in hi})  # fmt: skip
        draw.FinishDrawing()
        coords = [
            [round(p.x, 1), round(p.y, 1)]
            for p in (draw.GetDrawCoords(k) for k in range(len(heavy)))
        ]
        r = int(s.atoms["residue"][heavy[0]])
        name = f"{s.residues['name'][r]}{int(s.residues['resid'][r]) or ''}".strip() or "ligand"
        return {"svg": draw.GetDrawingText(), "atoms": heavy, "coords": coords, "name": name,
                "width": width, "height": height}  # fmt: skip

    def minimize(self, mobile: str | None = None, restraint: float = 10.0, fixed=None,
                 tethered=None) -> dict:  # fmt: skip
        """Relax ``mobile`` (default the ligand and its pocket), the other heavy atoms
        restrained; or, given ``fixed`` and ``tethered`` atoms (MOE's Constrain), those
        held and the rest free.  Hydrogens are never restrained."""
        from .edit import minimize, pocket_selection

        self._editable()
        constrained = bool(fixed) or bool(tethered)
        if not constrained:
            mobile = mobile or pocket_selection(self.ligand or LIGAND)
        with self._lock:
            p = self._parameterized()
            self._remember()
            try:
                report = minimize(self.system, None if constrained else mobile,
                                  restraint=restraint, parameterized=p,
                                  fixed=fixed if constrained else None,
                                  tethered=tethered if constrained else None)  # fmt: skip
            except Exception:
                self.system = self._undo.pop()[0]
                raise
            return {**self._changed(topology=False), **report,
                    "mobile": "constraints" if constrained else mobile}  # fmt: skip
