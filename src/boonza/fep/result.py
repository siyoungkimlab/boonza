"""boonza fep result: every edge's ΔΔG and every ligand's ΔG, in one table.

Each repeat of each leg is analyzed on its own (boonza.fep.analysis); a leg's
ΔG is the mean over repeats and its error their standard deviation (openfe's
report; MBAR's own error is shown beside it), and an edge's ΔΔG = ΔG(complex)
- ΔG(solvent), of binding; without a receptor ΔG(solvent) - ΔG(vacuum), of
hydration, or ΔG(octanol) - ΔG(solvent), of transfer from water to octanol,
whose change of log P (``dlogP``, B's less A's) is -ΔΔG / (RT ln 10).
Ligands' values come from the network's least squares
(boonza.fep.network.estimate), relative to their mean.

``results_ligands.sdf`` is the ligands' SDF (the one prep kept, or ``ligands``
for a work directory prepared before prep kept one), every field kept, with
FEP_dG and FEP_dG_error (kcal/mol, relative to the ligands' mean) added; where
ligands were measured, FEP_dG_shifted too (on the measured scale, the
correlation plot's), and for binding FEP_KD_nM from it; for transfer,
FEP_logP_rel.  Ligands without a measured value are predictions: shifted with
the rest, in the SDF and the CSV, and dashed lines on the correlation plot.

ΔΔG is as openfe reports it, without a long-range dispersion correction (a
hybrid cannot carry one).  Beside it, ``disp`` is what the correction would add
-- each periodic leg's end states' difference dK (leg.json, from prep) over the
run's mean volume, combined as the legs are -- and ``ddG_disp`` the sum.
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
#: Boltzmann's constant, kcal/mol/K
KB_KCAL = 0.0019872041


def _leg(directory: Path) -> dict | None:
    meta = json.loads((directory / "leg.json").read_text())
    runs = sorted(directory.glob("run_*.npz"))
    out = []
    dk = float(meta.get("dispersion_dK", 0.0))  # kcal/mol Å^3
    for p in runs:
        r = Replicas.load(p)
        if len(r.u_kln) < 4:
            continue
        a = analyze(r, temperature=meta["temperature"])
        half = len(a["forward"]) // 2 - 1  # each estimate from half the data
        forward, reverse = a["forward"][:, 0], a["reverse"][:, 0]
        volumes = np.abs(np.linalg.det(np.asarray(r.boxes, float))) * 1000.0  # nm^3 to Å^3
        disp = dk * float(np.mean(1.0 / volumes)) if dk else 0.0
        out.append({"dG": a["dG"], "dispersion": disp, "mbar_error": a["dG_error"],
                    "overlap": a["min_neighbour_overlap"], "mixing": a["min_neighbour_mixing"],
                    "iterations": a["iterations"],
                    "forward_reverse": float(abs(forward[half] - reverse[half]))})  # fmt: skip
    if not out:
        return None
    dg = np.array([x["dG"] for x in out])
    planned = int(round(meta["production_ns"] * 1000.0 / meta.get("iteration_ps", 2.5)))
    return {"dG": float(dg.mean()), "sd": float(dg.std(ddof=1)) if len(dg) > 1 else float("nan"),
            "dispersion": float(np.mean([x["dispersion"] for x in out])),
            "mbar_error": float(np.sqrt(np.mean([x["mbar_error"] ** 2 for x in out]))),
            "repeats": len(out), "planned_repeats": meta["repeats"],
            "overlap": min(x["overlap"] for x in out), "mixing": min(x["mixing"] for x in out),
            "complete": min(x["iterations"] for x in out) / max(planned, 1),
            "forward_reverse": max(x["forward_reverse"] for x in out), "meta": meta}  # fmt: skip


def results(workdir, log=print, experimental=None, ligands=None) -> dict:
    """Analyze every leg under ``workdir``; print one table, write edges and ligands
    CSVs and the ligands' SDF with the results (the module docstring), and, where
    ligands have measured values -- from ligands.csv (prep), or an ``experimental``
    SDF instead -- the correlation plots (boonza.fep.experiment).  ``ligands``: the
    SDF to add the results to, if prep kept none in ``workdir``."""
    work = Path(workdir)
    table = list(csv.DictReader((work / "ligands.csv").open()))
    names = [row["name"] for row in table]
    rows, fits, temperature = [], [], None
    for edge in sorted((work / "edges").iterdir()):
        legs = {leg: _leg(edge / leg) for leg in ("complex", "solvent", "vacuum", "octanol")
                if (edge / leg).is_dir()}  # fmt: skip
        if not legs or any(v is None for v in legs.values()):
            rows.append({"edge": edge.name, "status": "not run"})
            continue
        meta = next(iter(legs.values()))["meta"]
        temperature = meta["temperature"]
        a, b = meta["ligands"]
        solv, comp, vac = legs.get("solvent"), legs.get("complex"), legs.get("vacuum")
        octanol = legs.get("octanol")
        if comp:
            kind, top, bottom = "binding", comp, solv
        elif octanol:
            kind, top, bottom = "transfer", octanol, solv
        else:
            kind, top, bottom = "hydration", solv, vac
        ddg = top["dG"] - bottom["dG"]
        disp = top["dispersion"] - bottom["dispersion"]
        sd = [x["sd"] for x in legs.values()]
        err = float(np.sqrt(np.nansum(np.square(sd)))) if not np.all(np.isnan(sd)) else float("nan")
        mbar = float(np.sqrt(sum(x["mbar_error"] ** 2 for x in legs.values())))
        worst = min(x["overlap"] for x in legs.values())
        rt_ln10 = KB_KCAL * meta["temperature"] * np.log(10.0)
        row = {"edge": edge.name, "ligand_a": names[a], "ligand_b": names[b],
               "ddG": ddg, "error": err, "mbar_error": mbar, "dispersion": disp,
               "ddG_disp": ddg + disp,
               "kind": kind,
               "dG_complex": comp["dG"] if comp else float("nan"), "dG_solvent": solv["dG"],
               "dG_vacuum": vac["dG"] if vac else float("nan"),
               "dG_octanol": octanol["dG"] if octanol else float("nan"),
               "dlogP": -ddg / rt_ln10 if octanol else float("nan"),
               "dlogP_error": err / rt_ln10 if octanol else float("nan"),
               "repeats": min(x["repeats"] for x in legs.values()),
               "min_overlap": worst, "min_mixing": min(x["mixing"] for x in legs.values()),
               "forward_reverse": max(x["forward_reverse"] for x in legs.values()),
               "complete": min(x["complete"] for x in legs.values()),
               "status": "ok" if worst >= MIN_OVERLAP else "poor overlap"}  # fmt: skip
        rows.append(row)
        fits.append((a, b, ddg, err if np.isfinite(err) and err > 0 else mbar))
    _write(work / "results_edges.csv", rows)
    _print_edges(rows, log)
    for r in rows:
        if r.get("kind") == "transfer":
            log(
                f"{r['ligand_a']} -> {r['ligand_b']}: log P {r['dlogP']:+.2f} "
                f"± {r['dlogP_error']:.2f} (water to octanol)"
            )
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
    measured = _measured(work, table, rows, experimental)
    if measured:
        out["experiment"] = _compare(work, names, rows, out.get("ligands"), measured, log)
    if out.get("ligands"):
        kind = next((r["kind"] for r in rows if "kind" in r), "binding")
        _write_sdf(work, out["ligands"], kind, temperature, ligands, log)
    return out


def _write_sdf(work: Path, ligs, kind: str, temperature: float, source, log) -> Path | None:
    """``results_ligands.sdf``: the ligands' SDF with the results as fields."""
    from rdkit import Chem

    source = Path(source) if source is not None else work / "ligands.sdf"
    if not source.exists():
        log(f"no ligands' SDF to add the results to ({source}; prep keeps one from now on, "
            "or give it with --ligands)")  # fmt: skip
        return None
    by_name = {x["ligand"]: x for x in ligs}
    rt = KB_KCAL * temperature
    path, written = work / "results_ligands.sdf", 0
    with Chem.SDWriter(str(path)) as w:
        for m in Chem.SDMolSupplier(str(source), removeHs=False):
            if m is None:
                continue
            name = m.GetProp("_Name").strip() if m.HasProp("_Name") else ""
            for p in list(m.GetPropNames()):  # a results SDF given again: its old values go
                if p.startswith("FEP_"):
                    m.ClearProp(p)
            x = by_name.get(name)
            if x is not None:
                m.SetProp("FEP_dG", f"{x['dG']:.3f}")
                m.SetProp("FEP_dG_error", f"{x['error']:.3f}")
                if "dG_shifted" in x:
                    m.SetProp("FEP_dG_shifted", f"{x['dG_shifted']:.3f}")
                    if kind == "binding":
                        m.SetProp("FEP_KD_nM", f"{np.exp(x['dG_shifted'] / rt) * 1e9:.4g}")
                if kind == "transfer":
                    m.SetProp("FEP_logP_rel", f"{-x['dG'] / (rt * np.log(10.0)):.3f}")
                written += 1
            w.write(m)
    log(f"{path}: {written} of {len(ligs)} ligands with FEP_ fields")
    return path


def _measured(work: Path, table, rows, experimental) -> dict:
    """{ligand: (ΔG, field)} measured: from ``experimental`` (an SDF) if given,
    else the values prep kept in ligands.csv."""
    from .experiment import from_sdf

    done = [r for r in rows if "ddG" in r]
    if experimental is not None:
        if not done:
            return {}
        edge = work / "edges" / done[0]["edge"]
        leg = next(d for d in sorted(edge.iterdir()) if (d / "leg.json").exists())
        temperature = json.loads((leg / "leg.json").read_text())["temperature"]
        got = from_sdf(experimental, temperature, done[0]["kind"])
        unknown = sorted(set(got) - {r["name"] for r in table})
        if unknown:
            raise ValueError(f"{experimental}: ligands not in this set: {', '.join(unknown)}")
        return got
    return {r["name"]: (float(r["exp_dG"]), r["exp_field"]) for r in table
            if r.get("exp_dG", "") != ""}  # fmt: skip


def _compare(work: Path, names, rows, ligs, measured: dict, log) -> dict:
    """Calculated against measured, per ligand (shifted to the measured mean) and per
    edge: the statistics, printed, written beside the results, and plotted."""
    from .experiment import correlation_plot, describe, statistics

    done = [r for r in rows if "ddG" in r]
    kind = done[0]["kind"] if done else "binding"
    quantity = {"binding": "ΔG", "hydration": "ΔG_hyd", "transfer": "ΔG_transfer"}[kind]
    fields = sorted({f for _, f in measured.values()})
    log("")
    log(
        f"measured: {len(measured)} of {len(names)} ligands ({', '.join(fields)})"
        + ("; pIC50 taken for a dissociation constant" if "pIC50" in fields else "")
    )
    out = {}
    if ligs:
        have = [x for x in ligs if x["ligand"] in measured]
        if len(have) >= 2:
            exp = np.array([measured[x["ligand"]][0] for x in have])
            calc = np.array([x["dG"] for x in have])
            shift = float(exp.mean() - calc.mean())  # relative: one constant is free
            calc = calc + shift
            stats = statistics(calc, exp)
            for x in ligs:  # the predictions too, on the same scale
                x["dG_shifted"] = float(x["dG"] + shift)
            for x, e in zip(have, exp, strict=True):
                x["exp_dG"], x["exp_field"] = float(e), measured[x["ligand"]][1]
            _write(work / "results_ligands.csv", ligs)
            log(f"ligands, {quantity} shifted to the measured mean: " + "; ".join(describe(stats)))
            predicted = [(x["ligand"], x["dG_shifted"]) for x in ligs
                         if x["ligand"] not in measured]  # fmt: skip
            if predicted:
                log(f"  {len(predicted)} without a measured value, predicted: "
                    + ", ".join(f"{n} {v:.2f}" for n, v in predicted))  # fmt: skip
            files = correlation_plot(work / "plots" / "dG", calc, exp,
                                     calc_err=[x["error"] for x in have], quantity=quantity,
                                     stats=stats, log=log, names=[x["ligand"] for x in have],
                                     predicted=predicted)  # fmt: skip
            out["ligands"] = stats
            if files:
                log(f"  {', '.join(str(f) for f in files)}")
    pairs = [r for r in done if r["ligand_a"] in measured and r["ligand_b"] in measured]
    if pairs:
        exp = np.array([measured[r["ligand_b"]][0] - measured[r["ligand_a"]][0] for r in pairs])
        calc = np.array([r["ddG"] for r in pairs])
        for r, e in zip(pairs, exp, strict=True):
            r["exp_ddG"] = float(e)
        _write(work / "results_edges.csv", rows)
        stats = statistics(calc, exp)
        log(f"edges, Δ{quantity}: " + "; ".join(describe(stats)))
        errors = [r["error"] if np.isfinite(r["error"]) else r["mbar_error"] for r in pairs]
        files = correlation_plot(work / "plots" / "ddG", calc, exp, calc_err=errors,
                                 quantity=f"Δ{quantity}", stats=stats, log=log,
                                 names=[r["edge"] for r in pairs])  # fmt: skip
        out["edges"] = stats
        if files:
            log(f"  {', '.join(str(f) for f in files)}")
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


#: the leg each kind of edge sets against its water leg, the table's "other" column
OTHER = {"binding": "dG_complex", "hydration": "dG_vacuum", "transfer": "dG_octanol"}


def _print_edges(rows, log) -> None:
    head = (f"{'edge':<32}{'ΔΔG':>8}{'±':>6}{'MBAR':>6}{'disp':>6}{'+disp':>8}{'other':>9}"
            f"{'solvent':>9}"
            f"{'rep':>4}{'overlap':>8}{'mixing':>7}{'fwd/rev':>8}{'done':>6}  status")  # fmt: skip
    log(head)
    for r in rows:
        if "ddG" not in r:
            log(f"{r['edge']:<32}{'':>8}{'':>6}{'':>6}{'':>6}{'':>8}{'':>9}{'':>9}{'':>4}"
                f"{'':>8}{'':>7}{'':>8}{'':>6}  {r['status']}")  # fmt: skip
            continue
        log(f"{r['edge']:<32}{r['ddG']:>8.2f}{r['error']:>6.2f}{r['mbar_error']:>6.2f}"
            f"{r['dispersion']:>6.2f}{r['ddG_disp']:>8.2f}"
            f"{r[OTHER[r['kind']]]:>9.2f}"
            f"{r['dG_solvent']:>9.2f}{r['repeats']:>4d}"
            f"{r['min_overlap']:>8.3f}{r['min_mixing']:>7.2f}{r['forward_reverse']:>8.2f}"
            f"{100 * r['complete']:>5.0f}%  {r['status']}")  # fmt: skip
