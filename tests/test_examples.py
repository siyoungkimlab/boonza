"""Every script in examples/ must run cleanly (they skip themselves politely when
an optional package is missing)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p for p in (ROOT / "examples").iterdir()
                 if p.suffix in (".py", ".sh") and p.name[:2].isdigit())  # fmt: skip


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_example_runs(script):
    cmd = [sys.executable, str(script)] if script.suffix == ".py" else ["bash", str(script)]
    env = {**os.environ, "PYTHON": sys.executable, "PYTHONWARNINGS": "ignore"}
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, f"{script.name} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    assert r.stdout.strip(), f"{script.name} printed nothing"
