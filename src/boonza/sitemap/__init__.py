"""The old name of :mod:`boonza.cavity` (boonza sitemap, renamed): kept so scripts
that import it, and the sites.pkl caches pickled under it, still load."""

import importlib
import sys

from ..cavity import *  # noqa: F403

for _name in ("beads", "cli", "forcefield", "grid", "pharm", "presets", "report", "sites",
              "structure", "traj"):  # fmt: skip
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"boonza.cavity.{_name}")
