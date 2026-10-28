"""boonza fep result: every edge's ΔΔG and every ligand's ΔG, in one table.

Each repeat of each leg is analyzed on its own (boonza.fep.analysis); a leg's
ΔG is the mean over repeats and its error their standard deviation (openfe's
report; MBAR's own error is shown beside it), and an edge's ΔΔG = ΔG(complex)
- ΔG(solvent), of binding; without a receptor ΔG(solvent) - ΔG(vacuum), of
hydration.  Ligands' values come from the network's least squares
(boonza.fep.network.estimate), relative to their mean.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import network
from .analysis import analyze
from .sampler import Replicas

#: below this, neighbouring states share too few samples to trust MBAR
MIN_OVERLAP = 0.03


def _leg(directory: Path) -> dict | None:
    meta = json.loads((directory / "leg.json").read_text())
    runs = sorted(directory.glob("run_*.npz"))
    out = []
    for p in runs:
        r = Replicas.load(p)
        if len(r.u_kln) < 4:
            continue
        a = analyze(r, temperature=meta["temperature"])
        half = len(a["forward"]) // 2 - 1  # each estimate from half the data
        forward, reverse = a["forward"][:, 0], a["reverse"][:, 0]
        out.append({"dG": a["dG"], "mbar_error": a["dG_error"],
                    "overlap": a["min_neighbour_overlap"], "mixing": a["min_neighbour_mixing"],
                    "iterations": a["iterations"],
                    "forward_reverse": float(abs(forward[half] - reverse[half]))})  # fmt: skip
    if not out:
        return None
    dg = np.array([x["dG"] for x in out])
    planned = int(round(meta["production_ns"] * 1000.0 / meta.get("iteration_ps", 2.5)))
    return {"dG": float(dg.mean()), "sd": float(dg.std(ddof=1)) if len(dg) > 1 else float("nan"),
            "mbar_error": float(np.sqrt(np.mean([x["mbar_error"] ** 2 for x in out]))),
            "repeats": len(out), "planned_repeats": meta["repeats"],
            "overlap": min(x["overlap"] for x in out), "mixing": min(x["mixing"] for x in out),
            "complete": min(x["iterations"] for x in out) / max(planned, 1),
            "forward_reverse": max(x["forward_reverse"] for x in out), "meta": meta}  # fmt: skip


def results(workdir, log=print) -> dict:
    """Analyze every leg under ``workdir``; print one table, write edges and ligands CSVs."""
    work = Path(workdir)
    names = [row["name"] for row in csv.DictReader((work / "ligands.csv").open())]
    rows, fits = [], []
    for edge in sorted((work / "edges").iterdir()):
        legs = {leg: _leg(edge / leg) for leg in ("complex", "solvent", "vacuum")
                if (edge / leg).is_dir()}  # fmt: skip
        if not legs or any(v is None for v in legs.values()):
            rows.append({"edge": edge.name, "status": "not run"})
            continue
        meta = next(iter(legs.values()))["meta"]
        a, b = meta["ligands"]
        solv, comp, vac = legs.get("solvent"), legs.get("complex"), legs.get("vacuum")
        ddg = comp["dG"] - solv["dG"] if comp else solv["dG"] - vac["dG"]
        sd = [x["sd"] for x in legs.values()]
        err = float(np.sqrt(np.nansum(np.square(sd)))) if not np.all(np.isnan(sd)) else float("nan")
        mbar = float(np.sqrt(sum(x["mbar_error"] ** 2 for x in legs.values())))
        worst = min(x["overlap"] for x in legs.values())
        row = {"edge": edge.name, "ligand_a": names[a], "ligand_b": names[b],
               "ddG": ddg, "error": err, "mbar_error": mbar,
               "kind": "binding" if comp else "hydration",
               "dG_complex": comp["dG"] if comp else float("nan"), "dG_solvent": solv["dG"],
               "dG_vacuum": vac["dG"] if vac else float("nan"),
               "repeats": min(x["repeats"] for x in legs.values()),
               "min_overlap": worst, "min_mixing": min(x["mixing"] for x in legs.values()),
               "forward_reverse": max(x["forward_reverse"] for x in legs.values()),
               "complete": min(x["complete"] for x in legs.values()),
               "status": "ok" if worst >= MIN_OVERLAP else "poor overlap"}  # fmt: skip
        rows.append(row)
        fits.append((a, b, ddg, err if np.isfinite(err) and err > 0 else mbar))
    _write(work / "results_edges.csv", rows)
    _print_edges(rows, log)
    out = {"edges": rows}
    connected = len({i for f in fits for i in f[:2]}) == len(names)
    if fits and connected:
        try:
            g, ge = network.estimate(len(names), fits)
        except ValueError:
            g = None
        if g is not None:
            ligs = [{"ligand": n, "dG": float(v), "error": float(e)}
                    for n, v, e in zip(names, g, ge, strict=True)]  # fmt: skip
            _write(work / "results_ligands.csv", ligs)
            log("")
            log(f"{'ligand':<24}{'ΔG (rel. mean)':>16}{'error':>8}")
            for x in sorted(ligs, key=lambda x: x["dG"]):
                log(f"{x['ligand']:<24}{x['dG']:>16.2f}{x['error']:>8.2f}")
            out["ligands"] = ligs
            for cycle, total in network.cycle_closures(len(names), fits):
                if abs(total) > 1.0:
                    log(f"cycle {' -> '.join(names[k] for k in cycle)} does not close: "
                        f"{total:+.2f} kcal/mol")  # fmt: skip
    return out


def _write(path: Path, rows) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def _print_edges(rows, log) -> None:
    head = (f"{'edge':<32}{'ΔΔG':>8}{'±':>6}{'MBAR':>6}{'cplx/vac':>9}{'solvent':>9}"
            f"{'rep':>4}{'overlap':>8}{'mixing':>7}{'fwd/rev':>8}{'done':>6}  status")  # fmt: skip
    log(head)
    for r in rows:
        if "ddG" not in r:
            log(f"{r['edge']:<32}{'':>8}{'':>6}{'':>6}{'':>9}{'':>9}{'':>4}{'':>8}{'':>7}{'':>8}"
                f"{'':>6}  {r['status']}")  # fmt: skip
            continue
        log(f"{r['edge']:<32}{r['ddG']:>8.2f}{r['error']:>6.2f}{r['mbar_error']:>6.2f}"
            f"{r['dG_complex'] if r['kind'] == 'binding' else r['dG_vacuum']:>9.2f}"
            f"{r['dG_solvent']:>9.2f}{r['repeats']:>4d}"
            f"{r['min_overlap']:>8.3f}{r['min_mixing']:>7.2f}{r['forward_reverse']:>8.2f}"
            f"{100 * r['complete']:>5.0f}%  {r['status']}")  # fmt: skip
