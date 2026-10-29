"""Calculated against measured: each ligand's experimental ΔG from its SDF fields,
and the correlation plots and statistics of boonza fep result.

A ligand gives at most one of FIELDS, converted at the run's temperature to a
free energy in kcal/mol: a dissociation constant as RT ln K, a pIC50 as
-RT ln 10 pIC50 (IC50 taken for K -- right only up to a constant, Cheng-Prusoff
left out, which a relative comparison mostly does not see), a ΔG as it is.

Relative free energies fix every ligand's ΔG but for one constant, so the
calculated ΔGs are shifted to the measured ones' mean over the ligands that have
both; an edge's ΔΔG needs no shift.  Each plot is written twice, for a white and
a black background (``*_white.pdf`` in black ink, ``*_black.pdf`` in white),
transparent, with its text kept as text.
"""

from __future__ import annotations

import math
import warnings

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


# --- plots -------------------------------------------------------------------------------------

#: the user's figure style: Arial, 14 pt ticks and 16 pt labels, transparent,
#: text kept editable (TrueType); one file per background, ink to match
THEMES = {"white": "black", "black": "white"}
FONT = "Arial"


def _style(ink: str) -> dict:
    return {"font.family": FONT, "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16,
            "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
            "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
            "text.usetex": False, "mathtext.default": "regular",
            "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
            "savefig.edgecolor": "none", "savefig.transparent": True, "legend.framealpha": 0,
            "text.color": ink, "axes.edgecolor": ink, "axes.labelcolor": ink,
            "xtick.color": ink, "ytick.color": ink, "legend.edgecolor": ink,
            "errorbar.capsize": 3, "lines.linewidth": 1.5, "axes.linewidth": 1.0,
            "savefig.bbox": "tight"}  # fmt: skip


def _font_ready(log) -> bool:
    from matplotlib import font_manager

    try:
        font_manager.findfont(FONT, fallback_to_default=False)
        return True
    except ValueError:
        pass
    import os
    import sys
    from pathlib import Path

    for folder in (Path(os.environ.get("CONDA_PREFIX", sys.prefix)) / "fonts",
                   Path.home() / ".fonts", Path.home() / ".local/share/fonts"):  # fmt: skip
        for p in sorted(folder.glob("[Aa]rial*.ttf")):
            font_manager.fontManager.addfont(str(p))
    try:
        font_manager.findfont(FONT, fallback_to_default=False)
        return True
    except ValueError:
        log(f"  warning: {FONT} is not installed, so the plots use matplotlib's own font; "
            "'conda install -c conda-forge mscorefonts' puts it in this env")  # fmt: skip
        return False


def correlation_plot(path_stem, calc, exp, calc_err=None, exp_err=None, quantity="ΔG",
                     stats=None, log=print) -> list:  # fmt: skip
    """Calculated against measured, with y = x and a ±1 kcal/mol band, written as
    ``<path_stem>_white.pdf`` and ``_black.pdf``; the files, or [] without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    from pathlib import Path

    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    calc, exp = np.asarray(calc, float), np.asarray(exp, float)
    lo = float(min(calc.min(), exp.min())) - 1.0
    hi = float(max(calc.max(), exp.max())) + 1.0
    font = _font_ready(log)
    written = []
    for background, ink in THEMES.items():
        style = _style(ink)
        if not font:
            style.pop("font.family")
        with mpl.rc_context(style), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # a missing glyph, named above
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            line = np.array([lo, hi])
            ax.fill_between(line, line - 1.0, line + 1.0, color=ink, alpha=0.12, lw=0)
            ax.plot(line, line, color=ink, lw=1.0)
            ax.errorbar(exp, calc, xerr=exp_err, yerr=calc_err, fmt="o", color="tab:blue",
                        ecolor="tab:blue", ms=6, mew=0)  # fmt: skip
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.set_xlabel(f"experimental {quantity} (kcal/mol)")
            ax.set_ylabel(f"calculated {quantity} (kcal/mol)")
            if stats:
                ax.text(0.04, 0.96, "\n".join(describe(stats)), transform=ax.transAxes,
                        va="top", ha="left", fontsize=12)  # fmt: skip
            path = stem.with_name(f"{stem.name}_{background}.pdf")
            fig.savefig(path, format="pdf", transparent=True)
            plt.close(fig)
            written.append(path)
    return written
