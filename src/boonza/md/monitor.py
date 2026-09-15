"""Stopping production when a ligand or binder has left its pocket.

The pocket is the receptor's heavy atoms within ``pocket_cutoff_nm`` of the
target's at the start of production, saved in ``pocket.json`` and never
recomputed. Every ``monitor_interval_ns`` (a checkpoint step) the minimum
periodic distance between target and pocket and the number of atom pairs
closer than ``contact_cutoff_nm`` are recorded in ``monitor.csv``; production
stops after ``confirmation_checks`` checks in a row with no contact and a
distance above ``detach_cutoff_nm``. ``status.json`` records the outcome.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import MONITOR_SELECTORS

FIELDS = (
    "production_time_ns",
    "step",
    "min_ligand_pocket_distance_nm",
    "contact_count",
    "initial_contact_count",
    "contact_fraction",
    "consecutive_detached_count",
    "detached",
)
SETTINGS = (
    "monitor_interval_ns",
    "pocket_cutoff_nm",
    "contact_cutoff_nm",
    "detach_cutoff_nm",
    "confirmation_checks",
)


@dataclass
class Monitor:
    ligand_id: str
    component_id: str
    selector_kind: str
    selector_value: str | None
    ligand_atom_indices: list[int]
    ligand_heavy_atom_indices: list[int]
    pocket_atom_indices: list[int]
    pocket_residues: list[dict]
    initial_contact_count: int
    settings: dict


def select_target(info: dict, args) -> dict:
    """The component chosen by the one selector given, or the only ligand."""
    comps = info["components"]
    if args.monitor_ligand is not None:
        found = [c for c in comps if c["ligand_id"] == args.monitor_ligand]
        kind, value = "ligand", args.monitor_ligand
    elif args.monitor_chain is not None:
        found = [c for c in comps if args.monitor_chain in c["chains"]]
        kind, value = "chain", args.monitor_chain
        if len(found) > 1:
            raise ValueError(
                f"chain {value} is in {len(found)} components: " + ", ".join(c["id"] for c in found)
            )
    elif args.monitor_component is not None:
        found = [c for c in comps if c["id"] == args.monitor_component]
        kind, value = "component", args.monitor_component
    elif getattr(args, "monitor_selection", None) is not None:
        sel = info["selection"]  # the selected atoms themselves are the target
        return {"id": "selection", "ligand_id": None, **sel, "_kind": "selection",
                "_value": args.monitor_selection}  # fmt: skip
    else:
        found = [c for c in comps if c["ligand_id"]]
        kind, value = "automatic", None
        if len(found) != 1:
            raise ValueError(
                f"early stop needs a target: {len(found)} ligands found; choose one "
                "with monitor_ligand, monitor_chain, monitor_component or monitor_selection "
                "(boonza md --list-components shows them)"
            )
    if not found:
        choices = ", ".join(c["ligand_id"] or c["id"] for c in comps)
        raise ValueError(f"no component matches {kind} {value!r}; the components: {choices}")
    return {**found[0], "_kind": kind, "_value": value}


def _protein_atoms(s) -> np.ndarray:
    names, res = s.atoms["name"], s.atoms["residue"]
    backbone: dict[int, set] = {}
    for a in np.flatnonzero(np.isin(names, ["N", "CA", "C"])).tolist():
        backbone.setdefault(int(res[a]), set()).add(str(names[a]))
    aa = [r for r, got in backbone.items() if got == {"N", "CA", "C"}]
    return np.isin(res, aa)


def initialize(s, positions_nm, box_nm, info: dict, args) -> Monitor:
    """The target and its pocket at the start of production."""
    target = select_target(info, args)
    atoms = list(target["production_atom_indices"])
    anum = s.atoms["anum"]
    heavy = [a for a in atoms if anum[a] > 1]
    if not heavy:
        raise ValueError(f"target {target['id']} has no heavy atoms")
    excluded = set(atoms)
    for c in info["components"]:
        if c["ligand_id"]:
            excluded.update(c["production_atom_indices"])
    protein = _protein_atoms(s)
    if protein.any() and set(np.flatnonzero(protein).tolist()) <= set(atoms):
        raise ValueError(f"target {target['id']} holds every protein atom; choose a binder")
    receptor = [a for a in np.flatnonzero(protein & (anum > 1)).tolist() if a not in excluded]
    d = min_distances(positions_nm, receptor, heavy, box_nm)
    pocket = [a for a, x in zip(receptor, d.tolist(), strict=True) if x <= args.pocket_cutoff_nm]
    if not pocket:
        raise ValueError(
            f"target {target['id']} has no receptor heavy atom within "
            f"{args.pocket_cutoff_nm:g} nm: it is not bound at the start"
        )
    _, contacts = measure(heavy, pocket, positions_nm, box_nm, args.contact_cutoff_nm)
    res = s.residues
    by_res: dict[int, list[int]] = {}
    for a in pocket:
        by_res.setdefault(int(s.atoms["residue"][a]), []).append(a)
    residues = [
        {
            "chain_id": str(s.chains["name"][res["chain"][r]]),
            "residue_name": str(res["name"][r]),
            "residue_id": int(res["resid"][r]),
            "atom_indices": v,
        }
        for r, v in by_res.items()
    ]
    return Monitor(
        target["ligand_id"] or target["id"],
        target["id"],
        target["_kind"],
        target["_value"],
        atoms,
        heavy,
        pocket,
        residues,
        contacts,
        {k: getattr(args, k) for k in SETTINGS},
    )


def _minimum_image(d, box):
    if box is None:
        return d
    inv = np.linalg.inv(box)
    f = d @ inv
    f -= np.floor(f + 0.5)
    return f @ box


def _blocks(pos, first, second, box, size=512):
    pos = np.asarray(pos, dtype=np.float64)
    b = pos[np.asarray(second, dtype=np.int64)]
    first = np.asarray(first, dtype=np.int64)
    for k in range(0, len(first), size):
        a = pos[first[k : k + size]]
        yield np.sqrt((_minimum_image(b[None] - a[:, None], box) ** 2).sum(-1))


def min_distances(pos, first, second, box) -> np.ndarray:
    """Each ``first`` atom's shortest periodic distance to ``second``."""
    if not len(first) or not len(second):
        return np.zeros(0)
    return np.concatenate([blk.min(1) for blk in _blocks(pos, first, second, box)])


def measure(target, pocket, pos, box, cutoff: float) -> tuple[float, int]:
    """(minimum periodic distance, number of pairs within ``cutoff``)."""
    low, n = math.inf, 0
    for blk in _blocks(pos, target, pocket, box):
        low = min(low, float(blk.min()))
        n += int((blk <= cutoff).sum())
    return low, n


def detachment_update(
    distance: float, contacts: int, detach_cutoff: float, previous: int, needed: int
) -> tuple[int, bool]:
    """(consecutive detached checks, whether detachment is confirmed)."""
    count = previous + 1 if contacts == 0 and distance > detach_cutoff else 0
    return count, count >= needed


def write_pocket(path, m: Monitor) -> None:
    Path(path).write_text(json.dumps(asdict(m), indent=1) + "\n", encoding="utf-8")


def load_pocket(path) -> Monitor:
    try:
        return Monitor(**json.loads(Path(path).read_text(encoding="utf-8")))
    except FileNotFoundError:
        raise FileNotFoundError(f"early stop cannot resume without {path}") from None


def check_saved(m: Monitor, args, natoms: int) -> None:
    """A restored pocket must be for the same target and settings."""
    for key in MONITOR_SELECTORS:
        value = getattr(args, key)
        if value is not None and (
            m.selector_value != value or m.selector_kind != key.split("_")[1]
        ):
            raise ValueError(
                f"{key} = {value!r} differs from the saved target "
                f"{m.selector_kind} {m.selector_value!r}"
            )
    for key in ("pocket_cutoff_nm", "contact_cutoff_nm", "detach_cutoff_nm"):
        if m.settings.get(key) != getattr(args, key):
            raise ValueError(f"{key} differs from the saved pocket's ({m.settings.get(key)})")
    if max(m.ligand_atom_indices + m.pocket_atom_indices, default=-1) >= natoms:
        raise ValueError("the saved pocket does not match the system")


def last_row(path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return rows[-1] if rows else None


def append_row(path, row: dict) -> bool:
    """Append a check; a step already recorded (after a restart) is skipped."""
    last = last_row(path)
    if last is not None:
        if int(last["step"]) == int(row["step"]):
            return False
        if int(last["step"]) > int(row["step"]):
            raise ValueError(f"{path} already has step {last['step']}, after {row['step']}")
    path = Path(path)
    header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if header:
            w.writeheader()
        w.writerow({k: row[k] for k in FIELDS})
    return True


def restore_count(path, step: int) -> int:
    """The consecutive detached count at the checkpoint's step."""
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8"))) if Path(path).is_file() else []
    for row in reversed(rows):
        if int(row["step"]) <= step:
            return int(row["consecutive_detached_count"])
    return 0


def status(
    outcome: str, m: Monitor, target_ns: float, time_ns: float, step: int, count: int
) -> dict:
    return {
        "outcome": outcome,
        "ligand_id": m.ligand_id,
        "component_id": m.component_id,
        "target_production_ns": target_ns,
        "final_production_time_ns": time_ns,
        "final_production_step": step,
        "consecutive_detached_count": count,
        "early_stop_enabled": True,
    }


def write_status(path, doc: dict) -> None:
    Path(path).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def load_status(path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
