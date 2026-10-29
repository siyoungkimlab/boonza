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
        assert data.startswith(b"%PDF") and b"TrueType" in data  # text kept as text
