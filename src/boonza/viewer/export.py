"""A viewer as plain files, for a web host with no Python behind it (GitHub Pages).

    >>> export(viewer, "site/")      # then: python -m http.server -d site

The page is the same one ``boonza view`` serves; what the server would answer
(the models, each model's topology, every frame, the secondary structure and
the ligand's contacts of each frame) is written once into ``site/data/``, and
the page reads those files instead.  Nothing on it can change a model: editing,
minimizing, the Python line and opening files need ``boonza view``.

Selections typed on the page need the server too; the few the page makes by
itself (each model's default representations, the ligand and its binding site
for the presets and Focus on Ligand) are worked out here.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .outputs import Viewer
from .session import CG_POCKET

STATIC = Path(__file__).resolve().parent.parent / "data" / "viewer"


def _site_selections(topo: dict) -> list[str]:
    """The selections ui.js builds for its presets and Select | Binding Site, as it writes them."""
    lig = topo["ligand"]
    if topo["cg"]:
        pocket = (f"(same residue as ((same residue as name BB GC PX) and within {CG_POCKET:g} "
                  f"of ({lig}))) and not ({lig})")  # fmt: skip
    else:
        pocket = (f"(same residue as ((protein or nucleic) and within 4.5 of ({lig}))) "
                  f"and not ({lig})")  # fmt: skip
    return [f"same residue as ({lig})", pocket]


def _write_json(path: Path, value) -> int:
    text = json.dumps(
        value,
        separators=(",", ":"),
        default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
    )
    path.write_text(text)
    return len(text)


def export(viewer: Viewer, out, ss_each_frame: bool = True) -> dict:
    """Write ``viewer`` into the directory ``out`` as a static site; returns the
    number of models and frames, and the bytes written."""
    out = Path(out)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    for f in STATIC.iterdir():
        if f.is_file() and f.name != "index.html":
            shutil.copyfile(f, out / f.name)
    page = (STATIC / "index.html").read_text()
    page = page.replace("</head>", '<meta name="boonza-static" content="1"></head>', 1)
    (out / "index.html").write_text(page)

    size = 0
    listing = viewer.listing()
    listing.pop("groups", None)  # the builder's: nothing is built here
    listing["static"] = True
    size += _write_json(data / "models.json", listing)
    nframes = 0
    for k, model in enumerate(viewer.models):
        if model.closed:
            continue
        s = viewer.session(k)
        topo = s.topology()
        for rep in topo["reps"]:
            rep["ids"] = s.select(rep["sel"]).tolist()
        topo["selections"] = {}
        for sel in _site_selections(topo):
            try:
                topo["selections"][sel] = s.select(sel).tolist()
            except Exception:  # no ligand, or a selection this model cannot take
                pass
        size += _write_json(data / f"topology-m{k}.json", topo)
        has_ligand = len(topo["selections"].get(_site_selections(topo)[0], [])) > 0
        for i in range(s.nframes):
            pos, box = s.frame(i)
            blob = np.concatenate([pos.ravel(), np.asarray(box, "<f4").ravel()]).astype("<f4")
            (data / f"frame-m{k}-i{i}.bin").write_bytes(blob.tobytes())
            size += blob.nbytes
            if ss_each_frame and s.nframes > 1:
                size += _write_json(data / f"ss-m{k}-i{i}.json", {"ss": s.ss(i)})
            if has_ligand:
                size += _write_json(data / f"contacts-m{k}-i{i}.json", {"contacts": s.contacts(i)})
        nframes += s.nframes
    return {"models": sum(not m.closed for m in viewer.models), "frames": nframes, "bytes": size}
