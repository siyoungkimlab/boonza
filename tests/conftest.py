import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

MSYS_PYTHON = os.environ.get("BOONZA_MSYS_PYTHON", "/opt/homebrew/bin/python3.10")
MSYS_BUILD = Path(os.environ.get("BOONZA_MSYS_BUILD", "~/msys/build")).expanduser()
MSYS_FILES = Path(os.environ.get("BOONZA_MSYS_FILES", "~/msys/tests/files")).expanduser()


def _msys_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(MSYS_BUILD / "lib" / "python")
    env["DYLD_LIBRARY_PATH"] = str(MSYS_BUILD / "lib")
    return env


@functools.cache
def msys_available() -> bool:
    try:
        r = subprocess.run(
            [MSYS_PYTHON, "-c", "import msys"],
            env=_msys_env(), cwd=HERE, capture_output=True, timeout=120,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def run_msys(*args):
    """Run tests/msys_oracle.py under the msys Python and return its JSON output."""
    if not msys_available():
        pytest.skip("msys not available; set BOONZA_MSYS_PYTHON and BOONZA_MSYS_BUILD")
    r = subprocess.run(
        [MSYS_PYTHON, str(HERE / "msys_oracle.py"), *map(str, args)],
        env=_msys_env(), cwd=HERE, capture_output=True, text=True,
    )  # fmt: skip
    if r.returncode:
        raise RuntimeError(f"msys oracle failed:\n{r.stderr}")
    return json.loads(r.stdout)


def msys_file(name: str) -> Path:
    path = MSYS_FILES / name
    if not path.exists():
        pytest.skip(f"msys test file {name} not found in {MSYS_FILES}")
    return path
