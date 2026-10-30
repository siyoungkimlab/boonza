"""boonza fep's measured values: SDF fields to ΔG, the statistics, and the plots."""

import math

import numpy as np
import pytest

from boonza.fep import experiment
from boonza.fep.experiment import experimental_dG, statistics

T = 298.15
RT = 0.0019872041 * T


def test_conversions():
    assert experimental_dG({"dG": "-9.5"}, "x", T) == (-9.5, "dG")
    one_nm, field = experimental_dG({"KD_nM": 1.0}, "x", T)
    assert field == "KD_nM" and one_nm == pytest.approx(RT * math.log(1e-9))
    assert one_nm == pytest.approx(-12.28, abs=0.01)
    one_mm = experimental_dG({"KD_uM": 1000.0}, "x", T)[0]
    assert one_mm == pytest.approx(one_nm + RT * math.log(1e6))
    assert experimental_dG({"KD_uM": "0.001"}, "x", T)[0] == pytest.approx(one_nm)
    # an IC50 of 1 nM is taken for a 1 nM dissociation constant
    assert experimental_dG({"pIC50": 9.0}, "x", T)[0] == pytest.approx(one_nm)
    assert experimental_dG({"MW": 300.0, "pIC50": ""}, "x", T) is None  # nothing to read


def test_refused():
    with pytest.raises(ValueError, match="KD_nM, pIC50"):
        experimental_dG({"KD_nM": 5.0, "pIC50": 8.0}, "lig7", T)
    with pytest.raises(ValueError, match="binding constant"):
        experimental_dG({"KD_nM": 5.0}, "x", T, kind="hydration")
    assert experimental_dG({"dG": -3.0}, "x", T, kind="transfer") == (-3.0, "dG")
    with pytest.raises(ValueError, match="not positive"):
        experimental_dG({"KD_uM": 0.0}, "x", T)
    with pytest.raises(ValueError, match="not a number"):
        experimental_dG({"pIC50": ">9"}, "x", T)


def test_statistics():
    exp = np.array([-12.0, -10.5, -9.0, -8.0, -11.0])
    s = statistics(exp, exp)
    assert s["n"] == 5 and s["RMSE"]["value"] == 0.0 and s["MUE"]["value"] == 0.0
    assert s["R2"]["value"] == pytest.approx(1.0) and s["tau"]["value"] == pytest.approx(1.0)
    shifted = statistics(exp + 1.0, exp)
    assert shifted["RMSE"]["value"] == pytest.approx(1.0)
    assert shifted["MUE"]["value"] == pytest.approx(1.0)
    assert statistics(-exp, exp)["tau"]["value"] == pytest.approx(-1.0)
    low, high = shifted["RMSE"]["low"], shifted["RMSE"]["high"]
    assert low == pytest.approx(1.0) and high == pytest.approx(1.0)  # every resample is off by 1
    two = statistics([1.0, 2.0], [1.5, 2.5])  # too few for R², τ or an interval
    assert "R2" not in two and two["RMSE"]["low"] is None


def test_plot(tmp_path):
    pytest.importorskip("matplotlib")
    exp = [-12.0, -10.5, -9.0, -8.0]
    calc = [-11.6, -10.9, -9.4, -7.2]
    stats = statistics(calc, exp)
    files = experiment.correlation_plot(tmp_path / "plots" / "dG", calc, exp,
                                        calc_err=[0.2, 0.3, 0.2, 0.4], stats=stats,
                                        log=lambda *_: None)  # fmt: skip
    assert [f.name for f in files] == ["dG_white.pdf", "dG_black.pdf"]
    for f in files:
        data = f.read_bytes()
        # text kept as text: TrueType (whole, or as a composite CIDFontType2 for Δ, τ and
        # ±), never Type 3, which Illustrator and Affinity cannot edit
        assert data.startswith(b"%PDF") and b"/Type3" not in data
        assert b"/TrueType" in data or b"/CIDFontType2" in data


def test_plot_with_predictions_and_the_script_that_redraws_it(tmp_path):
    import subprocess
    import sys

    pytest.importorskip("matplotlib")
    exp = [-12.0, -10.5, -9.0, -8.0]
    calc = [-11.6, -10.9, -9.4, -7.2]
    plots = tmp_path / "plots"
    files = experiment.correlation_plot(plots / "dG", calc, exp, calc_err=[0.2, 0.3, 0.2, 0.4],
                                        stats=statistics(calc, exp), log=lambda *_: None,
                                        names=["a", "b", "c", "d"],
                                        predicted=[("new1", -10.2), ("new2", -13.5)])  # fmt: skip
    assert [f.name for f in files] == ["dG_white.pdf", "dG_black.pdf"]
    rows = (plots / "dG.csv").read_text().splitlines()
    assert rows[0] == "name,calc,exp,calc_err,exp_err,null" and len(rows) == 1 + 4 + 2
    assert rows[-1] == "new2,-13.5000,,,,"  # a prediction: no measured value
    script = plots / "plot_correlation.py"
    text = script.read_text()
    assert "import boonza" not in text and "from boonza" not in text  # stands alone
    for f in files:
        f.unlink()
    done = subprocess.run([sys.executable, "-I", "plot_correlation.py", "dG"], cwd=plots,
                          capture_output=True, text=True)  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert all(f.exists() and f.read_bytes().startswith(b"%PDF") for f in files)
    script.write_text(text + "\n# tailored\n")  # an edited copy is left alone
    notes = []
    experiment.correlation_plot(plots / "dG", calc, exp, log=notes.append)
    assert script.read_text().endswith("# tailored\n") and any("left as it is" in n for n in notes)


def test_the_null_model_and_axes_alike(tmp_path):
    import json

    pytest.importorskip("matplotlib")
    from boonza.fep import plot_correlation

    lmw = experiment.log_mw(["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1"])
    assert np.all(np.diff(lmw) > 0)
    exp = -3.0 * lmw + 5.0
    slope, intercept = experiment.null_fit(lmw, exp)
    assert slope == pytest.approx(-3.0) and intercept == pytest.approx(5.0)
    calc = exp + np.array([0.5, -0.4, 0.3, -0.2])
    null = slope * lmw + intercept
    plots = tmp_path / "plots"
    experiment.correlation_plot(plots / "dG", calc, exp, stats=statistics(calc, exp),
                                null=null, null_stats=statistics(null, exp),
                                log=lambda *_: None)  # fmt: skip
    rows = (plots / "dG.csv").read_text().splitlines()
    assert all(r.split(",")[-1] for r in rows[1:])  # every measured point has its null
    lines = json.loads((plots / "dG.json").read_text())["lines"]
    assert lines[0].startswith("FEP: RMSE") and "[" not in lines[0]  # no intervals
    assert lines[1].startswith("log MW: RMSE 0.00") and lines[-1] == "n = 4"
    at = plot_correlation.ticks(-12.3, -7.9)  # the ticks both axes are given
    assert at == sorted(at) and -12.3 <= at[0] and at[-1] <= -7.9 and len(at) >= 3
    assert len({round(b - a, 9) for a, b in zip(at, at[1:], strict=False)}) == 1  # evenly spaced
