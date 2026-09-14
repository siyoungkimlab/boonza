"""Generate docs/examples.md by running every script in examples/.

    python docs/gen_examples.py

Each example appears with its description, its code and the output it
actually printed, so the page always matches the code.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
MAX_LINES = 60


def _describe(path: Path) -> tuple[str, str, str]:
    """(title, description, code without the module docstring) of an example."""
    text = path.read_text()
    if path.suffix == ".py":
        doc = ast.get_docstring(ast.parse(text)) or path.stem
        body = text.split('"""', 2)[2].lstrip("\n") if text.startswith('"""') else text
    else:  # shell: leading comment block
        lines = []
        for ln in text.splitlines()[1:]:  # the comment block after the shebang
            if not ln.startswith("#"):
                break
            lines.append(ln[2:] if ln.startswith("# ") else "")
        doc = "\n".join(lines)
        body = text
    title, _, rest = doc.partition("\n")
    return title.strip().rstrip("."), rest.strip("\n"), body.rstrip()


def _run(path: Path) -> str:
    cmd = [sys.executable, str(path)] if path.suffix == ".py" else ["bash", str(path)]
    env = {**os.environ, "PYTHON": sys.executable, "PYTHONWARNINGS": "ignore"}
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=900)
    if r.returncode:
        raise RuntimeError(f"{path.name} failed:\n{r.stdout}\n{r.stderr}")
    lines = r.stdout.replace(str(ROOT) + "/", "").rstrip().splitlines()
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES] + [f"... ({len(lines) - MAX_LINES} more lines)"]
    return "\n".join(lines)


def main():
    scripts = sorted(p for p in EXAMPLES.iterdir()
                     if p.suffix in (".py", ".sh") and p.name[:2].isdigit())  # fmt: skip
    out = [
        "# Examples",
        "",
        "Runnable scripts in [`examples/`]"
        "(https://github.com/siyoungkimlab/boonza/tree/main/examples)."
        " Each one teaches a task step by",
        "step and prints what it finds. The output below is what the scripts",
        "printed when this page was generated with `python docs/gen_examples.py`.",
        "Run any of them with `python examples/NN_name.py`.",
        "",
        "Examples marked *needs openmm* or *needs rdkit* print a note and stop when",
        "that package is missing. The structures in `examples/data` come from the",
        "RCSB PDB.",
        "",
    ]
    for p in scripts:
        title, _, _ = _describe(p)
        out.append(f"- [{p.name}](#{p.stem.replace('_', '-')}) — {title}")
    out.append("")
    for p in scripts:
        title, desc, code = _describe(p)
        lang = "python" if p.suffix == ".py" else "bash"
        print(f"running {p.name} ...", flush=True)
        output = _run(p)
        out += [f'<a id="{p.stem.replace("_", "-")}"></a>', "", f"## {p.name}: {title}", ""]
        if desc:
            out += [desc, ""]
        out += [f"```{lang}", code, "```", "", "Output:", "", "```text", output, "```", ""]
    target = ROOT / "docs" / "examples.md"
    target.write_text("\n".join(out).rstrip() + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
