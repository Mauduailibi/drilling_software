"""Gera documentação HTML navegável a partir das docstrings no estilo NumPy.

Usage
-----
    .venv/bin/python scripts/build_docs.py

Em seguida abra ``docs/api/index.html``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "docs" / "api"


def main() -> None:
    """Invoca o pdoc nos dois pacotes de produto, na GUI e no ponto de entrada."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    OUT.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pdoc",
        "--docformat",
        "numpy",
        "-o",
        str(OUT),
        str(SRC / "drilling"),
    ]
    raise SystemExit(subprocess.call(command, env=env, cwd=ROOT))


if __name__ == "__main__":
    main()
