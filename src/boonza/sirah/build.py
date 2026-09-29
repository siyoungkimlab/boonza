"""Map a structure onto SIRAH beads, as SIRAH's cgconv.pl does.

A SIRAH bead sits on one atom of the structure rather than at the centre of
several: the map says which, and gives alternatives for the atoms that go by
several names (a carboxyl oxygen is ``O``, ``OC2``, ``OT1`` or ``O1``).  The
map is SIRAH's own file, carried with the force field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import read, unpack


@dataclass
class Bead:
    """One bead: where it sits, what it is called, and where it came from."""

    name: str  # GN, GC, BCG, ...
    residue: str  # the coarse-grained residue, sA, sR, ...
    resid: int
    chain: str
    insertion: str
    position: np.ndarray  # Å
    source: str  # the atom it was placed on


@dataclass
class Mapping:
    """A coarse-grained residue of the map: its name, and its beads' atoms."""

    residue: str
    beads: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)


def read_map(text: str | None = None) -> dict[str, Mapping]:
    """SIRAH's mapping, as ``{all-atom residue: Mapping}``.

    Several all-atom names can share one coarse-grained residue -- SIRAH 2.2
    maps HIS, HIE, HSE and the protonated HIP and HSP alike, to the neutral
    epsilon tautomer, since it has no charged histidine.
    """
    out: dict[str, Mapping] = {}
    current: Mapping | None = None
    names: list[str] = []
    for line in (text if text is not None else read("sirah_prot.map")).splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        if line.startswith(">"):
            current, names = None, []
            continue
        key, _, rest = line.partition(" ")
        rest = rest.strip()
        if key == "CGNAME":
            current = Mapping(rest)
            for name in names:  # ALLNAME may come first or second
                out[name] = current
        elif key == "ALLNAME":
            names = rest.split()
            if current is not None:
                for name in names:
                    out[name] = current
        elif key == "MAP" and current is not None:
            atoms, _, bead = rest.partition("=>")
            current.beads.append((bead.strip(), tuple(atoms.split())))
    return out


#: All-atom residues whose protonation the map does not keep, and what SIRAH
#: makes of them: worth saying out loud rather than changing in silence.
RETYPED = {
    "HIP": "a neutral histidine (SIRAH 2.2 has no charged one)",
    "HSP": "a neutral histidine (SIRAH 2.2 has no charged one)",
    "HID": "the delta tautomer",
    "HSD": "the delta tautomer",
    "LYN": "a charged lysine",
    "ASH": "a neutral aspartate",
    "GLH": "a neutral glutamate",
}


def _in_library_order(entry: Mapping) -> list[tuple[str, tuple[str, ...]]]:
    """The residue's beads in the order its topology lists them.

    The map names them in its own order (GC, GO, GN, then the side chain),
    while the library -- and so the topology, and cgconv's own output -- runs
    GN, GC, side chain, GO.  A bead the library does not hold, or one it holds
    and the map does not, is a disagreement worth raising.
    """
    library, _ = read_residues()
    known = library.get(entry.residue)
    if known is None:
        return entry.beads
    order = {name: k for k, (name, *_rest) in enumerate(known.atoms)}
    mapped = {bead for bead, _ in entry.beads}
    if mapped != set(order):
        raise ValueError(f"SIRAH's map and its library disagree about {entry.residue}: "
                         f"{sorted(mapped ^ set(order))}")  # fmt: skip
    return sorted(entry.beads, key=lambda b: order[b[0]])


def map_structure(system, atoms: str = "protein", log=None) -> list[Bead]:
    """The beads of ``atoms``, each on the atom SIRAH's map names for it.

    A residue the map does not know, or one missing the atom a bead sits on,
    raises rather than coming out with a bead short.
    """
    mapping = read_map()
    ids = system.select(atoms).ids
    if not len(ids):
        raise ValueError(f"no atoms in {atoms!r}")
    residue = np.asarray(system.atoms["residue"])[ids]
    names = np.asarray(system.atoms["name"])[ids]
    xyz = np.asarray(system.positions)[ids]
    res = system.residues
    chains = np.asarray(system.chains["name"])
    unknown, missing, changed = set(), [], set()
    out: list[Bead] = []
    for r in dict.fromkeys(residue.tolist()):
        here = residue == r
        resname = str(res["name"][r]).strip().upper()
        entry = mapping.get(resname)
        if entry is None:
            unknown.add(resname)
            continue
        if resname in RETYPED:
            changed.add(f"{resname} as {RETYPED[resname]}")
        where = {str(n).strip(): x for n, x in zip(names[here], xyz[here], strict=True)}
        for bead, wanted in _in_library_order(entry):
            found = next((a for a in wanted if a in where), None)
            if found is None:
                missing.append(f"{resname}{int(res['resid'][r])} has no {' or '.join(wanted)}"
                               f" for bead {bead}")  # fmt: skip
                continue
            out.append(Bead(bead, entry.residue, int(res["resid"][r]),
                            str(chains[res["chain"][r]]).strip(), str(res["insertion"][r]).strip(),
                            np.asarray(where[found], float), found))  # fmt: skip
    if unknown:
        raise ValueError(f"SIRAH's map has no {', '.join(sorted(unknown))}; leave them out of "
                         f"atoms={atoms!r}")  # fmt: skip
    if missing:
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise ValueError("the structure is missing atoms SIRAH maps beads onto: "
                         + "; ".join(missing[:5]) + more)  # fmt: skip
    if changed and log is not None:
        log(f"Note: SIRAH maps {', '.join(sorted(changed))}")
    return out


@dataclass
class Residue:
    """A coarse-grained residue of SIRAH's library."""

    name: str
    atoms: list[tuple[str, str, float]] = field(default_factory=list)  # bead, type, charge
    bonds: list[tuple[str, str]] = field(default_factory=list)  # names, '+X' is the next residue
    angles: list[tuple[str, ...]] = field(default_factory=list)
    dihedrals: list[tuple[str, ...]] = field(default_factory=list)
    impropers: list[tuple[str, ...]] = field(default_factory=list)


#: What the library's ``[ bondedtypes ]`` line says, as GROMACS reads it.
@dataclass
class Bonded:
    bonds: int = 1
    angles: int = 1
    dihedrals: int = 9
    impropers: int = 4
    all_dihedrals: bool = True
    nrexcl: int = 3
    hh14: bool = True
    remove_dihedrals: bool = False


def read_residues(text: str | None = None) -> tuple[dict[str, Residue], Bonded]:
    """SIRAH's residue library: what each coarse-grained residue is made of.

    Angles and dihedrals are usually left empty there, as they are in any
    GROMACS ``.rtp``: they follow from the bonds, and pdb2gmx generates them.
    """
    residues: dict[str, Residue] = {}
    bonded = Bonded()
    current: Residue | None = None
    section = ""
    for raw in (text if text is not None else read("aminoacids.rtp")).splitlines():
        line = raw.split(";")[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            if name == "bondedtypes":
                current, section = None, "bondedtypes"
            elif name in ("atoms", "bonds", "angles", "dihedrals", "impropers", "exclusions",
                          "cmap"):  # fmt: skip
                section = name
            else:
                current = residues.setdefault(name, Residue(name))
                section = ""
            continue
        parts = line.split()
        if section == "bondedtypes":
            v = [int(x) for x in parts]
            bonded = Bonded(v[0], v[1], v[2], v[3], bool(v[4]), v[5], bool(v[6]),
                            bool(v[7]) if len(v) > 7 else False)  # fmt: skip
            section = ""
        elif current is None:
            continue
        elif section == "atoms" and len(parts) >= 3:
            current.atoms.append((parts[0], parts[1], float(parts[2])))
        elif section == "bonds" and len(parts) >= 2:
            current.bonds.append((parts[0], parts[1]))
        elif section == "angles" and len(parts) >= 3:
            current.angles.append(tuple(parts[:3]))
        elif section == "dihedrals" and len(parts) >= 4:
            current.dihedrals.append(tuple(parts[:4]))
        elif section == "impropers" and len(parts) >= 4:
            current.impropers.append(tuple(parts[:4]))
    return residues, bonded


@dataclass
class Molecule:
    """One chain of beads and everything bonded within it."""

    name: str
    beads: list[Bead] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    charges: list[float] = field(default_factory=list)
    masses: list[float] = field(default_factory=list)
    bonds: list[tuple[int, int]] = field(default_factory=list)
    pairs: list[tuple[int, int]] = field(default_factory=list)
    angles: list[tuple[int, int, int]] = field(default_factory=list)
    dihedrals: list[tuple[int, int, int, int]] = field(default_factory=list)
    impropers: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def natoms(self) -> int:
        return len(self.beads)


def _chains_of(beads: list[Bead]) -> list[list[int]]:
    """Beads grouped into molecules: a chain runs while the residues do."""
    out: list[list[int]] = []
    last: tuple | None = None
    for k, b in enumerate(beads):
        here = (b.chain, b.resid, b.insertion)
        if last is not None and (b.chain != last[0] or b.resid not in (last[1], last[1] + 1)):
            out.append([])
        elif last is None:
            out.append([])
        out[-1].append(k)
        last = here
    return out


def _angles_from(bonds: list[tuple[int, int]], natoms: int) -> list[tuple[int, int, int]]:
    """Every i-j-k of the bond graph, as pdb2gmx generates them."""
    neighbours: list[set[int]] = [set() for _ in range(natoms)]
    for i, j in bonds:
        neighbours[i].add(j)
        neighbours[j].add(i)
    out = []
    for j in range(natoms):
        near = sorted(neighbours[j])
        for a in range(len(near)):
            for b in range(a + 1, len(near)):
                out.append((near[a], j, near[b]))
    return out


def _dihedrals_from(bonds: list[tuple[int, int]], natoms: int) -> list[tuple[int, int, int, int]]:
    """Every i-j-k-l over every bond j-k: SIRAH's library asks for all of them."""
    neighbours: list[set[int]] = [set() for _ in range(natoms)]
    for i, j in bonds:
        neighbours[i].add(j)
        neighbours[j].add(i)
    out = []
    for j, k in bonds:
        for i in sorted(neighbours[j] - {k}):
            for m in sorted(neighbours[k] - {j, i}):
                out.append((i, j, k, m))
    return out


def _pairs_from(dihedrals, bonds, angles) -> list[tuple[int, int]]:
    """The 1-4 pairs of the dihedrals, less anything already 1-2 or 1-3."""
    near = {tuple(sorted(b)) for b in bonds} | {tuple(sorted((a[0], a[2]))) for a in angles}
    out, seen = [], set()
    for i, _j, _k, m in dihedrals:
        key = tuple(sorted((i, m)))
        if key in near or key in seen or key[0] == key[1]:
            continue
        seen.add(key)
        out.append(key)
    return out


def read_masses(text: str | None = None) -> dict[str, float]:
    """The mass of each bead type, from ``atomtypes.atp``.

    SIRAH's ``[ atomtypes ]`` carries 0 for every mass -- the masses are in
    this file, which is why pdb2gmx writes them into the topology it builds.
    """
    out = {}
    for raw in (text if text is not None else read("atomtypes.atp")).splitlines():
        line = raw.split(";")[0].split()
        if len(line) >= 2:
            out[line[0]] = float(line[1])
    return out


def read_termini(text: str) -> dict[str, dict[str, tuple[str, float, float]]]:
    """A ``.tdb`` file: which beads each terminus replaces, and with what."""
    out: dict[str, dict[str, tuple[str, float, float]]] = {}
    current, section = None, ""
    for raw in text.splitlines():
        line = raw.split(";")[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            if name in ("replace", "add", "delete", "impropers", "bonds"):
                section = name
            else:
                current = out.setdefault(name, {})
                section = ""
            continue
        parts = line.split()
        if current is not None and section == "replace" and len(parts) >= 4:
            current[parts[0]] = (parts[1], float(parts[2]), float(parts[3]))
    return out


#: How far apart two BSG beads are taken to be a disulfide (Å), as
#: ``specbond.dat`` gives it.
DISULFIDE = 2.0


def sirahize(system, atoms: str = "protein", *, termini: str = "Charged",
             disulfides: bool = True, log=None) -> Sirahized:  # fmt: skip
    """Map ``system`` onto SIRAH beads and build the topology of each chain.

    The beads come from SIRAH's map, their topology from its residue library,
    and the angles, dihedrals and 1-4 pairs follow from the bonds, the way
    pdb2gmx generates them.  ``termini`` is ``"Charged"`` or ``"Neutral"``,
    named as the library's ``.tdb`` files are.
    """
    beads = map_structure(system, atoms, log)
    library, bonded = read_residues()
    masses = read_masses()
    n_ter = read_termini(read("aminoacids.n.tdb"))
    c_ter = read_termini(read("aminoacids.c.tdb"))
    if termini == "None":  # the chain ends keep the charges of their residues
        n_ter, c_ter = {termini: {}}, {termini: {}}
    for which, where in (("N", n_ter), ("C", c_ter)):
        if termini not in where:
            raise ValueError(f"the {which} terminus has no {termini!r}; the library holds "
                             f"{', '.join(sorted(where))}, or 'None' to leave them be")  # fmt: skip
    molecules = []
    for m, members in enumerate(_chains_of(beads)):
        mine = [beads[k] for k in members]
        mol = Molecule(f"molecule_{m}", mine)
        index = {(b.resid, b.name): k for k, b in enumerate(mine)}
        first, last = mine[0].resid, mine[-1].resid
        for k, b in enumerate(mine):
            entry = library[b.residue]
            by_name = {name: (kind, charge) for name, kind, charge in entry.atoms}
            kind, charge = by_name[b.name]
            if b.resid == first and b.name in n_ter[termini]:
                kind, _mass, charge = n_ter[termini][b.name]
            if b.resid == last and b.name in c_ter[termini]:
                kind, _mass, charge = c_ter[termini][b.name]
            mol.types.append(kind)
            mol.charges.append(charge)
            mol.masses.append(masses.get(kind, 0.0))
            del k
        for b in {(x.resid, x.residue) for x in mine}:
            resid, resname = b
            for one, two in library[resname].bonds:
                a = index.get((resid, one.lstrip("+-")))
                other = index.get((resid + (1 if two.startswith("+") else 0), two.lstrip("+-")))
                if a is not None and other is not None:
                    mol.bonds.append((min(a, other), max(a, other)))
            for imp in library[resname].impropers:
                got = [index.get((resid + (1 if n.startswith("+") else 0), n.lstrip("+-")))
                       for n in imp]  # fmt: skip
                if all(g is not None for g in got):
                    mol.impropers.append(tuple(got))
        mol.bonds = sorted(set(mol.bonds))
        molecules.append(mol)
    if disulfides:
        _add_disulfides(molecules, log)
    for mol in molecules:
        mol.angles = _angles_from(mol.bonds, mol.natoms)
        mol.dihedrals = _dihedrals_from(mol.bonds, mol.natoms)
        mol.pairs = _pairs_from(mol.dihedrals, mol.bonds, mol.angles)
        mol.impropers = sorted(set(mol.impropers))
    positions = np.array([b.position for b in beads], float)
    return Sirahized(molecules, positions, np.asarray(system.cell, float), nrexcl=bonded.nrexcl)


def _add_disulfides(molecules, log=None) -> None:
    """Bond the BSG beads of cysteines close enough to be bridged."""
    found = 0
    for mol in molecules:
        sg = [k for k, b in enumerate(mol.beads) if b.name == "BSG"]
        for a in range(len(sg)):
            for b in range(a + 1, len(sg)):
                i, j = sg[a], sg[b]
                d = float(np.linalg.norm(mol.beads[i].position - mol.beads[j].position))
                if d <= DISULFIDE:
                    mol.bonds.append((min(i, j), max(i, j)))
                    found += 1
        mol.bonds = sorted(set(mol.bonds))
    if found and log is not None:
        log(f"Disulfides: {found} bridge(s) between BSG beads within {DISULFIDE:g} A")


@dataclass
class Sirahized:
    """A coarse-grained system: its molecules, its beads' positions, its box."""

    molecules: list[Molecule]
    positions: np.ndarray  # Å
    cell: np.ndarray | None = None
    nrexcl: int = 3
    solvent: list[tuple[str, int]] = field(default_factory=list)  # WT4, NaW, ClW counts

    @property
    def nbeads(self) -> int:
        return len(self.positions)

    def itp(self, k: int) -> str:
        """One molecule's topology, with its parameters left to the force field."""
        mol = self.molecules[k]
        out = [f"[ moleculetype ]\n{mol.name} {self.nrexcl}\n", "[ atoms ]"]
        for n, (bead, kind, charge, mass) in enumerate(
            zip(mol.beads, mol.types, mol.charges, mol.masses, strict=True), start=1
        ):
            out.append(f"{n:6d} {kind:<6s} {bead.resid:5d} {bead.residue:<5s} {bead.name:<5s}"
                       f" {n:5d} {charge:8.3f} {mass:8.3f}")  # fmt: skip
        for title, rows, funct in (("bonds", mol.bonds, 1), ("pairs", mol.pairs, 1),
                                   ("angles", mol.angles, 1), ("dihedrals", mol.dihedrals, 9),
                                   ("dihedrals", mol.impropers, 4)):  # fmt: skip
            if not rows:
                continue
            out.append(f"\n[ {title} ]")
            out += [" ".join(f"{a + 1:5d}" for a in row) + f" {funct:5d}" for row in rows]
        return "\n".join(out) + "\n"

    def top(self, forcefield: str = "./sirah.ff") -> str:
        """The system's topology, including the force field beside it."""
        lines = [f'#include "{forcefield}/forcefield.itp"']
        lines += [f'#include "{m.name}.itp"' for m in self.molecules]
        if any(c for _, c in self.solvent):
            lines.append(f'#include "{forcefield}/solv.itp"')
        lines += ["", "[ system ]", "SIRAH system", "", "[ molecules ]"]
        lines += [f"{m.name} 1" for m in self.molecules]
        lines += [f"{n} {c}" for n, c in self.solvent if c]
        return "\n".join(lines) + "\n"

    def save(self, directory, forcefield: bool = True) -> Path:
        """Write the topology, its molecules and the force field it needs.

        With ``forcefield`` the carried files are written to ``sirah.ff``
        beside the topology, so the directory runs anywhere, in GROMACS as
        well as here.
        """
        from ..io import save as save_structure

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        if forcefield:
            unpack(out)
        for k, mol in enumerate(self.molecules):
            (out / f"{mol.name}.itp").write_text(self.itp(k))
        (out / "topol.top").write_text(self.top())
        built = self.system()
        save_structure(built, out / "cg.dms")  # what boonza reads back, unrounded
        save_structure(built, out / "cg.gro")  # what GROMACS needs, rounded to 0.001 nm
        return out / "topol.top"

    def system(self):
        """The beads as a boonza System, with SIRAH's parameters on them."""
        import tempfile

        from ..io.gromacs import load_top

        with tempfile.TemporaryDirectory() as tmp:
            ff = unpack(tmp)
            for k, mol in enumerate(self.molecules):
                (Path(tmp) / f"{mol.name}.itp").write_text(self.itp(k))
            (Path(tmp) / "topol.top").write_text(self.top(str(ff)))
            s = load_top(Path(tmp) / "topol.top", include_dirs=[tmp, str(ff)])
        s.positions = np.asarray(self.positions, float)
        if self.cell is not None and np.any(self.cell):
            s.cell = np.asarray(self.cell, float)
        _name_chains(s, self.molecules)
        return s


def _name_chains(s, molecules) -> None:
    """Give the beads the chains they were mapped from, which no .gro holds."""
    of_bead = [b.chain for mol in molecules for b in mol.beads]
    if not any(of_bead):
        return
    of_bead += [""] * (s.natoms - len(of_bead))
    residue = np.asarray(s.atoms["residue"])
    first = np.zeros(s.nresidues, np.int64)
    first[residue[::-1]] = np.arange(s.natoms)[::-1]
    want = [of_bead[int(a)] for a in first]
    chains = {str(s.chains["name"][c]): s.chain(c) for c in range(s.nchains)}
    for name in dict.fromkeys(want):
        if name not in chains:
            chains[name] = s.add_chain(name=name)
    for r, name in enumerate(want):
        s.residue(r).chain = chains[name]
    s._prune_hierarchy()


#: SIRAH's water box, as the force field ships it: an equilibrated cube that
#: tiles to fill any box.
WATER_BOX = "wt416.gro"
#: A water whose bead comes this close to the solute is left out, as SIRAH's
#: own tutorial removes them (Å).
WATER_CLASH = 3.0
#: An ion goes no closer than this to the solute (Å).
ION_DISTANCE = 5.0
#: One ion pair for this many waters is about 0.15 M, which is how SIRAH's
#: appendix counts its electrolytes.
WATERS_PER_PAIR = 34.0
SALT_OF_THAT = 0.15
#: The water box's own density (WT4 per nm^3).  WT4 settles a little below it,
#: at about 2.93 per nm^3 (0.97 g/mL) at 300 K and 1 bar -- boonza and GROMACS
#: agree on that to three figures on SIRAH's own box -- so a solvated box that
#: starts a few per cent thin equilibrates rather than collapses.
TILE_DENSITY = 16 / 1.72**3
EQUILIBRIUM_DENSITY = 2.93


def _water_tile() -> tuple[np.ndarray, float]:
    """The box's beads (Å, four to a molecule) and its edge."""
    lines = read(WATER_BOX).splitlines()
    n = int(lines[1])
    xyz = np.array([[float(line[20 + 8 * k : 28 + 8 * k]) for k in range(3)]
                    for line in lines[2 : 2 + n]], float) * 10.0  # fmt: skip
    edge = float(lines[2 + n].split()[0]) * 10.0
    return xyz, edge


#: Two waters of the equilibrated box come no closer than 3.1 A, so anything
#: closer than this is two tiles meeting, not water as it packs.
SEAM = 2.5


def _without_overlaps(water: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Drop one of each pair of waters that only meet because tiles do."""
    from ..spatial import pairs_within

    if not len(water):
        return water
    flat = water.reshape(-1, 3)
    i, j, _d2 = pairs_within(flat, SEAM, cell=np.diag(box))
    mine, theirs = i // 4, j // 4
    across = mine != theirs
    drop: set[int] = set()
    for a, b in zip(mine[across].tolist(), theirs[across].tolist(), strict=True):
        if a not in drop and b not in drop:
            drop.add(max(a, b))
    return water[[k for k in range(len(water)) if k not in drop]]


def solvate(m: Sirahized, padding: float = 10.0, box=None, salt: float = 0.15,
            neutralize: bool = True, clash: float = WATER_CLASH,
            ion_distance: float = ION_DISTANCE, seed: int = 0, fit: bool = True,
            log=None) -> Sirahized:  # fmt: skip
    """``m`` in a box of SIRAH's WT4 water, with NaW and ClW ions.

    The water is the force field's own equilibrated box, tiled to fill the
    cell and cut where it meets the solute: a molecule with any bead within
    ``clash`` Å of one of the solute's is left out, as SIRAH's tutorial
    removes them.  Ions replace whole waters at least ``ion_distance`` Å from
    the solute -- enough to cancel the solute's charge, then pairs until the
    salt reaches ``salt`` mol/L, counted as SIRAH counts it: one pair for
    every 34 waters is about 0.15 M.

    ``fit`` grows the box to whole tiles of the water box (1.72 nm), so the
    water meets itself as it was equilibrated and only the solute displaces
    any; cutting mid-tile costs a slab of water at every face.
    """
    from ..spatial import min_dist2

    rng = np.random.default_rng(seed)
    solute = np.asarray(m.positions, float)
    tile, side = _water_tile()
    if box is None:
        edge = float(np.ptp(solute, axis=0).max()) + 2 * padding
        box = np.full(3, edge)
    box = np.asarray(box, float)
    if box.ndim == 2:
        box = np.diag(box).astype(float)
    if fit:
        # whole tiles fill without a seam, so only the solute displaces water;
        # a box cut mid-tile loses a slab of it at every face
        box = np.ceil(box / side - 1e-9) * side
    solute = solute - solute.mean(0) + box / 2

    counts = np.ceil(box / side).astype(int)
    molecules = tile.reshape(-1, 4, 3)
    waters = []
    for i in range(counts[0]):
        for j in range(counts[1]):
            for k in range(counts[2]):
                moved = molecules + np.array([i, j, k]) * side
                # a molecule belongs where its first bead falls, so the tiles
                # meet without losing the ones that straddle a seam
                waters.append(moved[np.all((moved[:, 0] >= 0) & (moved[:, 0] < box), axis=1)])
    water = np.concatenate(waters)
    near = min_dist2(water.reshape(-1, 3), solute, clash, cell=np.diag(box))
    water = water[~(near.reshape(-1, 4) <= clash**2).any(axis=1)]
    water = _without_overlaps(water, box)

    charge = sum(c * count for mol, count in zip(m.molecules, [1] * len(m.molecules), strict=True)
                 for c in mol.charges)  # fmt: skip
    net = round(charge)
    counter = abs(net) if neutralize else 0
    pairs = int(round(len(water) * salt / (WATERS_PER_PAIR * SALT_OF_THAT)))
    wanted = counter + 2 * pairs
    far = np.flatnonzero(
        min_dist2(water[:, 0], solute, ion_distance, cell=np.diag(box)) > ion_distance**2
    )
    if wanted > len(far):
        raise ValueError(f"no room for {wanted} ions: only {len(far)} waters sit more than "
                         f"{ion_distance:g} A from the solute")  # fmt: skip
    chosen = rng.choice(far, size=wanted, replace=False) if wanted else np.array([], int)
    na = pairs + (counter if net < 0 else 0)
    cl = pairs + (counter if net > 0 else 0)
    ions = water[chosen][:, 0]  # an ion is one bead, where the water's first was
    keep = np.setdiff1d(np.arange(len(water)), chosen)
    out = Sirahized(m.molecules, np.concatenate([solute, water[keep].reshape(-1, 3), ions]),
                    np.diag(box), m.nrexcl,
                    [("WT4", int(len(keep))), ("NaW", int(na)), ("ClW", int(cl))])  # fmt: skip
    if log is not None:
        volume = float(np.prod(box / 10))
        log(f"Solvated: {len(keep)} WT4, {na} NaW and {cl} ClW in {volume:.1f} nm^3 "
            f"({len(keep) / volume:.2f} WT4/nm^3, where WT4 settles at about "
            f"{EQUILIBRIUM_DENSITY:.2f} under NPT)")  # fmt: skip
    return out
