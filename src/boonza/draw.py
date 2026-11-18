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

    A line is a SMILES and, after a space or a tab, an optional name, which
    may hold spaces and may be quoted; a line that starts with ``#`` is a
    comment (a ``#`` further along is a triple bond, or part of the name).
    The name is what a grid labels the molecule with, so a file that carries
    names needs nothing else to make a legible picture.
    """
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        smiles, *rest = line.split(None, 1)
        name = rest[0].strip() if rest else ""
        if len(name) >= 2 and name[0] == name[-1] and name[0] in "\"'":
            name = name[1:-1].strip()
        out.append((smiles, name))
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
    from rdkit.Geometry import Point3D

    if core is None:
        return 0
    # The core takes its coordinates from the first molecule that has it, laid
    # out whole.  Laid out alone, the core leaves no room for what hangs off
    # it, and a substituent ends up drawn on top of a core atom.
    first = next((m for m in mols if m.HasSubstructMatch(core)), None)
    if first is None:
        for mol in mols:
            AllChem.Compute2DCoords(mol)
        return 0
    AllChem.Compute2DCoords(first)
    rdDepictor.Compute2DCoords(core)
    placed, template = first.GetConformer(), core.GetConformer()
    for k, a in enumerate(first.GetSubstructMatch(core)):
        p = placed.GetAtomPosition(a)
        template.SetAtomPosition(k, Point3D(p.x, p.y, 0.0))
    done = 0
    for mol in mols:
        if mol is first:
            done += 1
            continue
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


# ---- PDF ---------------------------------------------------------------------------


def _rgb(text: str) -> tuple[float, float, float] | None:
    text = text.strip()
    if not text.startswith("#") or len(text) != 7:
        return None  # "none", or a paint this does not know: not drawn
    return tuple(int(text[k : k + 2], 16) / 255 for k in (1, 3, 5))


def _path_ops(d: str) -> list[str]:
    """PDF path operators for an SVG path of absolute M, L, Q, C and Z."""
    import re

    out, x, y = [], 0.0, 0.0
    for cmd, rest in re.findall(r"([MLQCZ])([^MLQCZ]*)", d):
        v = [float(n) for n in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", rest)]
        if cmd == "Z":
            out.append("h")
        elif cmd in "ML":
            for k in range(0, len(v) - 1, 2):
                x, y = v[k], v[k + 1]
                out.append(f"{x:.2f} {y:.2f} {'m' if cmd == 'M' and k == 0 else 'l'}")
        elif cmd == "Q":  # a quadratic curve as the cubic PDF has
            for k in range(0, len(v) - 3, 4):
                qx, qy, ex, ey = v[k : k + 4]
                c = (x + 2 * (qx - x) / 3, y + 2 * (qy - y) / 3,
                     ex + 2 * (qx - ex) / 3, ey + 2 * (qy - ey) / 3, ex, ey)  # fmt: skip
                out.append(" ".join(f"{n:.2f}" for n in c) + " c")
                x, y = ex, ey
        else:
            for k in range(0, len(v) - 5, 6):
                out.append(" ".join(f"{n:.2f}" for n in v[k : k + 6]) + " c")
                x, y = v[k + 4], v[k + 5]
    return out


def svg_to_pdf(svg: str) -> bytes:
    """One of RDKit's SVG drawings as a one-page vector PDF, the page as large as the
    picture.  Nothing is painted that the SVG does not paint, so a drawing made
    without a background (``clearBackground = False``) has a transparent page.
    It reads what RDKit writes -- paths, ellipses, rectangles and plain text -- and
    is not a general SVG converter."""
    import xml.etree.ElementTree as ET
    import zlib

    root = ET.fromstring(svg.encode("utf-8") if isinstance(svg, str) else svg)
    box = [float(n) for n in (root.get("viewBox") or "0 0 300 300").split()]
    width, height = box[2], box[3]
    ops = [f"1 0 0 -1 {-box[0]:.2f} {height + box[1]:.2f} cm"]  # SVG's y runs down
    text = False

    def paint(el, body: list[str]) -> None:
        style = dict(item.split(":", 1) for item in (el.get("style") or "").split(";")
                     if ":" in item)  # fmt: skip
        fill = _rgb(el.get("fill") or style.get("fill", "#000000" if el.tag.endswith("text")
                                                else "none"))  # fmt: skip
        stroke = _rgb(style.get("stroke", el.get("stroke") or "none"))
        line = float((style.get("stroke-width") or el.get("stroke-width") or "1").rstrip("px"))
        if line <= 0:
            stroke = None
        if fill is None and stroke is None:
            return
        ops.append("q")
        if fill is not None:
            ops.append("{:.3f} {:.3f} {:.3f} rg".format(*fill))
        if stroke is not None:
            ops.append("{:.3f} {:.3f} {:.3f} RG {:.2f} w".format(*stroke, line))
            ops.append("1 J 1 j" if style.get("stroke-linecap") == "round" else "0 J 0 j")
        ops.extend(body)
        even = "*" if style.get("fill-rule") == "evenodd" else ""
        ops.append(f"B{even}" if fill and stroke else f"f{even}" if fill else "S")
        ops.append("Q")

    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "path" and el.get("d"):
            paint(el, _path_ops(el.get("d")))
        elif tag == "rect":
            x, y, w, h = (float(el.get(k) or 0) for k in ("x", "y", "width", "height"))
            paint(el, [f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re"])
        elif tag == "ellipse":
            cx, cy, rx, ry = (float(el.get(k) or 0) for k in ("cx", "cy", "rx", "ry"))
            k = 0.5523  # four cubic arcs
            pts = [(cx + rx, cy)]
            for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
                if sx * sy > 0:
                    pts += [(cx + sx * rx, cy + sy * k * ry), (cx + sx * k * rx, cy + sy * ry),
                            (cx, cy + sy * ry)]  # fmt: skip
                else:
                    pts += [(cx + sx * k * rx, cy + sy * ry), (cx + sx * rx, cy + sy * k * ry),
                            (cx + sx * rx, cy)]  # fmt: skip
            body = ["{:.2f} {:.2f} m".format(*pts[0])]
            for n in range(1, 13, 3):
                body.append(" ".join(f"{a:.2f} {b:.2f}" for a, b in pts[n : n + 3]) + " c")
            paint(el, [*body, "h"])
        elif tag == "text" and (el.text or "").strip():  # RDKit built without FreeType
            style = dict(item.split(":", 1) for item in (el.get("style") or "").split(";")
                         if ":" in item)  # fmt: skip
            size = float((style.get("font-size") or "12").rstrip("px"))
            words = el.text.strip().encode("latin-1", "replace").decode("latin-1")
            x, y = float(el.get("x") or 0), float(el.get("y") or 0)
            anchor = style.get("text-anchor") or el.get("text-anchor") or "start"
            x -= {"middle": 0.5, "end": 1.0}.get(anchor, 0.0) * 0.56 * size * len(words)
            fill = _rgb(el.get("fill") or style.get("fill", "#000000")) or (0.0, 0.0, 0.0)
            safe = words.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append("q {:.3f} {:.3f} {:.3f} rg BT /F1 {:.2f} Tf 1 0 0 -1 {:.2f} {:.2f} Tm "
                       "({}) Tj ET Q".format(*fill, size, x, y, safe))  # fmt: skip
            text = True

    stream = zlib.compress("\n".join(ops).encode("latin-1"))
    fonts = "/Font << /F1 5 0 R >> " if text else ""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.2f} {height:.2f}] "
         f"/Resources << {fonts}>> /Contents 4 0 R >>").encode(),
        b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(stream) + stream
        + b"\nendstream",
    ]  # fmt: skip
    if text:
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for n, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, xref))  # fmt: skip
    return bytes(out)
