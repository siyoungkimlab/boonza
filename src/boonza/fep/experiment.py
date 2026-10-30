"""Calculated against measured: each ligand's experimental ΔG from its SDF fields,
and the correlation plots and statistics of boonza fep result.

A ligand gives at most one of FIELDS, converted at the run's temperature to a
free energy in kcal/mol: a dissociation constant as RT ln K, a pIC50 as
-RT ln 10 pIC50 (IC50 taken for K -- right only up to a constant, Cheng-Prusoff
left out, which a relative comparison mostly does not see), a ΔG as it is.

Relative free energies fix every ligand's ΔG but for one constant, so the
calculated ΔGs are shifted to the measured ones' mean over the ligands that have
both; an edge's ΔΔG needs no shift, and a ligand without a measured value is
shifted with the rest and drawn as a dashed line at its prediction.  The plots
are drawn by :mod:`boonza.fep.plot_correlation`, which result copies beside
their data so they can be redrawn and tailored without boonza.
"""

from __future__ import annotations

import math

import numpy as np

#: kcal/mol/K
KB_KCAL = 0.0019872041
#: the fields read from a ligand's SDF record, and how each becomes ΔG at T (kcal/mol)
FIELDS = {
    "dG": lambda v, t: v,
    "KD_nM": lambda v, t: KB_KCAL * t * math.log(v * 1e-9),
    "KD_uM": lambda v, t: KB_KCAL * t * math.log(v * 1e-6),
    "pIC50": lambda v, t: -KB_KCAL * t * math.log(10.0) * v,
}
#: the fields that are binding constants, meaningless for a hydration or transfer set
BINDING_ONLY = ("KD_nM", "KD_uM", "pIC50")
#: bootstrap resamples for the statistics' 95% intervals
BOOTSTRAP = 1000


def experimental_dG(props: dict, name: str, temperature: float, kind: str = "binding"):
    """(ΔG in kcal/mol, the field it came from) of one ligand's SDF properties, or
    None without any of FIELDS; more than one is an error, as is a binding
    constant for a hydration or transfer set or a constant that is not positive."""
    found = [f for f in FIELDS if f in props and str(props[f]).strip() != ""]
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(f"ligand {name!r} has {', '.join(found)}: give one experimental value")
    field = found[0]
    if kind != "binding" and field in BINDING_ONLY:
        raise ValueError(f"ligand {name!r}: {field} is a binding constant, and this is a "
                         f"{kind} set; give its dG (kcal/mol)")  # fmt: skip
    try:
        value = float(props[field])
    except (TypeError, ValueError):
        raise ValueError(f"ligand {name!r}: {field} = {props[field]!r} is not a number") from None
    if field.startswith("KD") and value <= 0:
        raise ValueError(f"ligand {name!r}: {field} = {value} is not positive")
    return FIELDS[field](value, temperature), field


def from_sdf(path, temperature: float, kind: str = "binding") -> dict:
    """{ligand name: (ΔG, field)} of every record of ``path`` that has a value."""
    from rdkit import Chem

    out = {}
    for k, m in enumerate(Chem.SDMolSupplier(str(path), removeHs=False)):
        if m is None:
            continue
        name = (m.GetProp("_Name").strip() if m.HasProp("_Name") else "") or f"ligand{k + 1}"
        got = experimental_dG(m.GetPropsAsDict(), name, temperature, kind)
        if got is not None:
            out[name] = got
    return out


def statistics(calc, exp, seed: int = 0) -> dict:
    """RMSE, MUE, R² and Kendall's τ of ``calc`` against ``exp``, each with a 95%
    bootstrap interval (None where too few points for it)."""
    calc, exp = np.asarray(calc, float), np.asarray(exp, float)
    n = len(calc)

    def measures(c, e):
        d = c - e
        out = {"RMSE": float(np.sqrt(np.mean(d**2))), "MUE": float(np.mean(np.abs(d)))}
        if len(c) >= 3 and np.std(c) > 0 and np.std(e) > 0:
            out["R2"] = float(np.corrcoef(c, e)[0, 1] ** 2)
            out["tau"] = _kendall(c, e)
        return out

    point = measures(calc, exp)
    stats = {k: {"value": v, "low": None, "high": None} for k, v in point.items()}
    if n >= 3:
        rng = np.random.default_rng(seed)
        draws = {k: [] for k in point}
        for _ in range(BOOTSTRAP):
            pick = rng.integers(0, n, n)
            for k, v in measures(calc[pick], exp[pick]).items():
                if k in draws and np.isfinite(v):
                    draws[k].append(v)
        for k, vals in draws.items():
            if vals:
                stats[k]["low"], stats[k]["high"] = (float(x) for x in
                                                     np.percentile(vals, [2.5, 97.5]))  # fmt: skip
    stats["n"] = n
    return stats


def _kendall(a, b) -> float:
    """Kendall's τ-b, without scipy."""
    concordant = discordant = ties_a = ties_b = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif da == db:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    return (concordant - discordant) / denom if denom else float("nan")


def describe(stats: dict) -> list[str]:
    """The statistics as lines of text, intervals beside the values."""
    names = {"RMSE": "RMSE", "MUE": "MUE", "R2": "R²", "tau": "Kendall τ"}
    lines = []
    for key, label in names.items():
        if key in stats:
            s = stats[key]
            ci = f" [{s['low']:.2f}, {s['high']:.2f}]" if s["low"] is not None else ""
            lines.append(f"{label} {s['value']:.2f}{ci}")
    lines.append(f"n = {stats['n']}")
    return lines


def log_mw(smiles) -> np.ndarray:
    """ln of each molecule's molecular weight (g/mol), from its SMILES."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    return np.array([math.log(Descriptors.MolWt(Chem.MolFromSmiles(s))) for s in smiles])


def null_fit(x, exp) -> tuple[float, float]:
    """The null model's slope and intercept: ``exp`` ≈ slope·x + intercept by least
    squares, x the ligands' log MW -- bigger binds tighter, the baseline a series'
    FEP has to beat.  Fitted to the measured values, so its statistics are in-sample."""
    slope, intercept = np.polyfit(np.asarray(x, float), np.asarray(exp, float), 1)
    return float(slope), float(intercept)


def null_line(stats: dict) -> str:
    """The null model's statistics in one line, for the plot's corner."""
    labels = (("RMSE", "RMSE"), ("R2", "R²"), ("tau", "τ"))
    parts = [f"{label} {stats[k]['value']:.2f}" for k, label in labels if k in stats]
    return "log MW: " + ", ".join(parts)


# --- plots -------------------------------------------------------------------------------------


def correlation_plot(path_stem, calc, exp, calc_err=None, exp_err=None, quantity="ΔG",
                     stats=None, log=print, names=None, predicted=(), null=None,
                     null_stats=None) -> list:  # fmt: skip
    """Calculated against measured, with y = x and a ±1 kcal/mol band, each of
    ``predicted`` [(name, ΔG)] -- calculated, not measured -- a dashed horizontal
    line, and ``null`` (the null model's value for each point, its statistics
    ``null_stats``) as open circles; written as ``<path_stem>_white.pdf`` and
    ``_black.pdf``, the data beside them (``<path_stem>.csv``, ``.json``) and the
    script that drew them (``plot_correlation.py``, to redraw and tailor); the
    files, or [] without matplotlib."""
    import filecmp
    import shutil
    from pathlib import Path

    from . import plot_correlation

    stem = Path(path_stem)
    names = list(names) if names is not None else [str(k + 1) for k in range(len(calc))]
    lines = describe(stats) if stats else []
    if null_stats:
        lines.append(null_line(null_stats))
    plot_correlation.write_data(stem, names, calc, exp, calc_err, exp_err, quantity, lines,
                                predicted, null)  # fmt: skip
    script, copy = Path(plot_correlation.__file__), stem.parent / "plot_correlation.py"
    if not copy.exists():
        shutil.copyfile(script, copy)
    elif not filecmp.cmp(script, copy, shallow=False):
        log(f"  {copy} differs from boonza's (edited?): left as it is")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return []
    return plot_correlation.plot(stem, calc, exp, calc_err=calc_err, exp_err=exp_err,
                                 quantity=quantity, lines=lines, predicted=predicted,
                                 null=null, log=log)  # fmt: skip
