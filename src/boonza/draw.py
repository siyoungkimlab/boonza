"""Pictures of molecules from SMILES: one, or a grid of many.

RDKit draws them, so what this adds is the part that is tedious by hand:
reading a list of SMILES, saying which one failed to parse rather than dying
on the first, laying a grid out over as many pages as it takes, and finding
the scaffold a series shares so that every molecule is drawn the same way
up with its common core marked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Suffixes that can be written, and whether RDKit draws them as pixels.
FORMATS = {".png": True, ".svg": False}
SIZE = (300, 250)  # per molecule, as RDKit's grids are sized
COLUMNS = 4
MCS_TIMEOUT = 20  # seconds; a big series can otherwise search a long time


@dataclass
class Drawing:
    """What a call to :func:`draw` produced.

    ``files`` are the pictures written (more than one when ``rows`` caps the
    grid), ``failures`` the SMILES RDKit could not read, ``core`` the SMARTS
    that was marked, and ``aligned`` how many molecules were laid out on it.
    ``core`` is None when ``--mcs`` found nothing the molecules share, which
    is worth saying: the picture is then simply unmarked.
    """

    files: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    core: str | None = None
    aligned: int = 0
    drawn: int = 0


def read_smiles(text: str) -> list[tuple[str, str]]:
    """``(smiles, name)`` per non-empty line, as a ``.smi`` file holds them.

    A line is a SMILES and, after whitespace, an optional name; ``#`` starts
    a comment.  The name is what a grid labels the molecule with, so a file
    that carries names needs nothing else to make a legible picture.
    """
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        smiles, _, name = line.partition(" ")
        out.append((smiles.strip(), name.strip()))
    return out


def molecules(smiles, names=None) -> tuple[list, list[str], list[str]]:
    """Parse ``smiles``; returns the molecules, their labels and the failures.

    A SMILES RDKit cannot read is collected rather than raised, so one bad
    line in a list of fifty does not cost the other forty-nine.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")  # the failures are returned, not printed
    names = list(names) if names is not None else [""] * len(smiles)
    if len(names) != len(smiles):
        raise ValueError(f"{len(smiles)} SMILES and {len(names)} names")
    mols, labels, bad = [], [], []
    for text, name in zip(smiles, names, strict=True):
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            bad.append(text)
            continue
        mols.append(mol)
        labels.append(name or text)
    return mols, labels, bad


def common_core(mols, timeout: int = MCS_TIMEOUT):
    """The maximum common substructure of ``mols``, or None if there is none.

    Bonds match on order and atoms on element, which is what makes the result
    a scaffold a chemist would recognize rather than a shape that happens to
    overlap.  Needs two molecules; one molecule is its own core.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    if len(mols) < 2:
        return Chem.Mol(mols[0]) if mols else None
    found = rdFMCS.FindMCS(
        mols,
        timeout=timeout,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
    )
    if found.canceled and not found.smartsString:
        return None
    return Chem.MolFromSmarts(found.smartsString) if found.smartsString else None


def orient(mols, core) -> int:
    """Lay every molecule out so that ``core`` sits the same way up in each.

    Returns how many were matched.  The rest keep their own depiction: a
    molecule without the core has nothing to be aligned to, and forcing one
    would draw a worse picture than leaving it alone.
    """
    from rdkit.Chem import AllChem, rdDepictor

    if core is None:
        return 0
    rdDepictor.Compute2DCoords(core)
    done = 0
    for mol in mols:
        if not mol.HasSubstructMatch(core):
            AllChem.Compute2DCoords(mol)
            continue
        try:
            rdDepictor.GenerateDepictionMatching2DStructure(mol, core)
            done += 1
        except (ValueError, RuntimeError):  # a match the depiction cannot honour
            AllChem.Compute2DCoords(mol)
    return done


def matches(mols, query) -> list[list[int]]:
    """The atoms of each molecule that ``query`` (a mol) hits, for highlighting.

    RDKit's grid takes lists, not tuples, and a molecule the query misses
    gets an empty one rather than being dropped.
    """
    if query is None:
        return [[] for _ in mols]
    return [sorted({a for hit in mol.GetSubstructMatches(query) for a in hit}) for mol in mols]


def _pages(n: int, columns: int, rows: int | None) -> int:
    """Molecules per page: everything, unless ``rows`` caps the grid."""
    return n if rows is None else max(1, columns * rows)


def _named(path: Path, page: int, pages: int) -> Path:
    """``out.png`` for one page, ``out-1.png`` ... when there are several."""
    return path if pages == 1 else path.with_name(f"{path.stem}-{page + 1}{path.suffix}")


def draw(smiles, path, names=None, size=SIZE, columns: int = COLUMNS, rows: int | None = None,
         highlight: str | None = None, mcs: bool = False, align: bool = False,
         labels: bool = True) -> Drawing:  # fmt: skip
    """Draw ``smiles`` into ``path`` (.png or .svg); returns a :class:`Drawing`.

    One molecule fills the picture; several are laid out ``columns`` wide,
    each ``size`` pixels, labelled with their names (or their SMILES, when
    the list carries none) unless ``labels`` is off.  ``rows`` caps the grid,
    and anything beyond one page goes to ``out-2.png``, ``out-3.png`` and so
    on, so a library does not become one unreadable image.

    ``highlight`` is a SMARTS marked wherever it is found.  ``mcs`` finds the
    largest scaffold the molecules share and marks that instead, which is the
    version to reach for when you do not already know what they have in
    common.  ``align`` lays them out so that the marked core sits the same way
    up in every picture, which is what makes a series comparable by eye.
    """
    from rdkit import Chem
    from rdkit.Chem import Draw

    path = Path(path)
    if path.suffix.lower() not in FORMATS:
        raise ValueError(f"cannot write {path.suffix or path.name}: "
                         f"draw writes {', '.join(sorted(FORMATS))}")  # fmt: skip
    if columns < 1 or (rows is not None and rows < 1):
        raise ValueError("'columns' and 'rows' must be at least 1")
    smiles = [smiles] if isinstance(smiles, str) else list(smiles)
    if not smiles:
        raise ValueError("give at least one SMILES to draw")
    mols, text, bad = molecules(smiles, names)
    if not mols:
        raise ValueError(f"none of the {len(bad)} SMILES could be read: {', '.join(bad[:3])}")

    if highlight and mcs:
        raise ValueError("give --highlight or --mcs, not both: they mark different cores")
    core = None
    if mcs:
        core = common_core(mols)
    elif highlight:
        core = Chem.MolFromSmarts(highlight)
        if core is None:
            raise ValueError(f"cannot read {highlight!r} as SMARTS")
    aligned = orient(mols, core if core is not None else common_core(mols)) if align else 0
    hits = matches(mols, core)

    per_page = _pages(len(mols), columns, rows)
    pages = -(-len(mols) // per_page)
    svg = not FORMATS[path.suffix.lower()]
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for page in range(pages):
        lo, hi = page * per_page, min((page + 1) * per_page, len(mols))
        here = mols[lo:hi]
        # returnPNG asks for the bytes; without it RDKit hands back a PIL image
        kind = {"useSVG": True} if svg else {"returnPNG": True}
        data = Draw.MolsToGridImage(
            here,
            molsPerRow=max(1, min(columns, len(here))),
            subImgSize=tuple(size),
            legends=list(text[lo:hi]) if labels else [""] * len(here),
            highlightAtomLists=list(hits[lo:hi]) if core is not None else None,
            **kind,
        )
        out = _named(path, page, pages)
        out.write_bytes(data.encode() if isinstance(data, str) else bytes(data))
        written.append(out)
    smarts = Chem.MolToSmarts(core) if core is not None else None
    return Drawing(written, bad, smarts, aligned, len(mols))
