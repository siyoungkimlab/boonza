"""A calculated-against-measured correlation plot, as boonza fep result draws it.

boonza fep result copies this file into WORKDIR/plots/ beside each plot's data
and draws its plots with it, so, from that directory,

    python plot_correlation.py dG      # dG.csv and dG.json -> dG_white.pdf, dG_black.pdf
    python plot_correlation.py ddG

redraws them exactly; edit the copy freely (result leaves an edited copy alone).
It needs numpy and matplotlib only, not boonza.

``<name>.csv``: one row per point, ``name, calc, exp, calc_err, exp_err, null``
(kcal/mol).  A row with ``exp`` empty is a prediction alone -- a ligand without a
measured value -- drawn as a dashed horizontal line at its calculated value,
named at the right; the statistics leave it out.  ``null``, where given, is what
the null model (log MW, fitted to the measured values) predicts for the point,
drawn as open gray circles beside the calculated ones.  ``<name>.json``: the
quantity on the axes and the statistics' lines written in the corner.

Both axes share their range, their scale and their ticks.

Each plot is written twice, for a white and a black background
(``*_white.pdf`` in black ink, ``*_black.pdf`` in white), transparent, Arial,
the text kept as TrueType text so Illustrator and Affinity can edit it.
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

#: one file per background, the ink to match
THEMES = {"white": "black", "black": "white"}
FONT = "Arial"
#: the measured points, and the predictions without a measurement
POINT_COLOR = "tab:blue"
PREDICTED_COLOR = "tab:orange"
NULL_COLOR = "tab:gray"
#: the band around y = x (kcal/mol)
BAND = 1.0


def style(ink: str) -> dict:
    """matplotlib settings: 14 pt ticks, 16 pt labels, transparent, text as text."""
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


def font_ready(log=print) -> bool:
    """Whether Arial is there, looking in the conda env's and the user's font folders
    too; matplotlib's own font, with a warning, if not."""
    import os

    from matplotlib import font_manager

    try:
        font_manager.findfont(FONT, fallback_to_default=False)
        return True
    except ValueError:
        pass
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


def ticks(lo: float, hi: float) -> list[float]:
    """The ticks both axes share: round steps within [lo, hi]."""
    from matplotlib.ticker import MaxNLocator

    at = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    return [float(t) for t in at if lo - 1e-9 <= t <= hi + 1e-9]


def plot(path_stem, calc, exp, calc_err=None, exp_err=None, quantity="ΔG", lines=(),
         predicted=(), null=None, log=print) -> list[Path]:  # fmt: skip
    """``calc`` against ``exp`` with y = x and a ±BAND band, ``lines`` of text in the
    corner, each of ``predicted`` [(name, calc)] a dashed horizontal line, and
    ``null`` (the null model's value for each point) as open circles; both axes
    the same range and ticks; written as ``<path_stem>_white.pdf`` and
    ``_black.pdf``, which it returns."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    calc, exp = np.asarray(calc, float), np.asarray(exp, float)
    null = None if null is None else np.asarray(null, float)
    values = np.concatenate([calc, exp, [v for _, v in predicted],
                             [] if null is None else null[np.isfinite(null)]])  # fmt: skip
    lo, hi = float(values.min()) - 1.0, float(values.max()) + 1.0
    at = ticks(lo, hi)
    font = font_ready(log)
    written = []
    for background, ink in THEMES.items():
        rc = style(ink)
        if not font:
            rc.pop("font.family")
        with mpl.rc_context(rc), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # a missing glyph, named above
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            line = np.array([lo, hi])
            ax.fill_between(line, line - BAND, line + BAND, color=ink, alpha=0.12, lw=0)
            ax.plot(line, line, color=ink, lw=1.0)
            for name, value in predicted:  # no measurement: the prediction alone
                ax.axhline(value, color=PREDICTED_COLOR, ls="--", lw=1.0)
                ax.text(hi - 0.02 * (hi - lo), value, name, color=PREDICTED_COLOR,
                        ha="right", va="bottom", fontsize=12)  # fmt: skip
            if null is not None:
                ax.plot(exp, null, "o", mfc="none", mec=NULL_COLOR, ms=6, mew=1.0,
                        label="log MW")  # fmt: skip
            if len(calc):
                ax.errorbar(exp, calc, xerr=exp_err, yerr=calc_err, fmt="o",
                            color=POINT_COLOR, ecolor=POINT_COLOR, ms=6, mew=0,
                            label="FEP")  # fmt: skip
            if null is not None and len(calc):
                ax.legend(loc="lower right", fontsize=12, handletextpad=0.2, borderaxespad=0.3)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xticks(at)
            ax.set_yticks(at)
            ax.set_aspect("equal")
            ax.set_xlabel(f"experimental {quantity} (kcal/mol)")
            ax.set_ylabel(f"calculated {quantity} (kcal/mol)")
            if lines:
                ax.text(0.04, 0.96, "\n".join(lines), transform=ax.transAxes,
                        va="top", ha="left", fontsize=12)  # fmt: skip
            path = stem.with_name(f"{stem.name}_{background}.pdf")
            fig.savefig(path, format="pdf", transparent=True)
            plt.close(fig)
            written.append(path)
    return written


def write_data(path_stem, names, calc, exp, calc_err=None, exp_err=None, quantity="ΔG",
               lines=(), predicted=(), null=None) -> None:  # fmt: skip
    """``<path_stem>.csv`` and ``.json``: what :func:`redraw` reads back."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    n = len(calc)
    ce = list(calc_err) if calc_err is not None else [None] * n
    ee = list(exp_err) if exp_err is not None else [None] * n
    nu = list(null) if null is not None else [None] * n

    def num(v):
        return "" if v is None or not np.isfinite(v) else f"{float(v):.4f}"

    with stem.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "calc", "exp", "calc_err", "exp_err", "null"])
        for k in range(n):
            w.writerow([names[k], num(calc[k]), num(exp[k]), num(ce[k]), num(ee[k]),
                        num(nu[k])])  # fmt: skip
        for name, value in predicted:
            w.writerow([name, num(value), "", "", "", ""])
    stem.with_suffix(".json").write_text(
        json.dumps({"quantity": quantity, "lines": list(lines)}, ensure_ascii=False, indent=1)
        + "\n"
    )


def redraw(path_stem, log=print) -> list[Path]:
    """The plot of ``<path_stem>.csv`` and ``.json``, drawn again."""
    stem = Path(path_stem)
    rows = list(csv.DictReader(stem.with_suffix(".csv").open()))
    meta = json.loads(stem.with_suffix(".json").read_text())
    measured = [r for r in rows if r["exp"] != ""]

    def column(key):
        got = [float(r[key]) if r.get(key, "") != "" else np.nan for r in measured]
        return None if all(np.isnan(got)) else got

    return plot(stem, [float(r["calc"]) for r in measured], [float(r["exp"]) for r in measured],
                calc_err=column("calc_err"), exp_err=column("exp_err"),
                quantity=meta.get("quantity", "ΔG"), lines=meta.get("lines", ()),
                predicted=[(r["name"], float(r["calc"])) for r in rows if r["exp"] == ""],
                null=column("null"), log=log)  # fmt: skip


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    for name in argv:
        stem = Path(name)
        stem = stem.with_suffix("") if stem.suffix in (".csv", ".json", ".pdf") else stem
        for p in redraw(stem):
            print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
