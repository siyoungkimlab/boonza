"""Sphinx configuration: the Markdown pages (MyST) with the Furo theme."""

import re
from pathlib import Path

project = "boonza"
author = "Siyoung Kim"
copyright = "2026, Siyoung Kim"
_init = (Path(__file__).resolve().parent.parent / "src" / "boonza" / "__init__.py").read_text()
release = version = re.search(r'__version__ = "([^"]+)"', _init).group(1)

extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build"]
myst_heading_anchors = 3

html_theme = "furo"
html_title = "boonza"
