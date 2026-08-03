"""Build .ipynb files from dependency-free percent-format Python sources."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "notebook_sources"
OUTPUT_DIR = ROOT / "notebooks"
MARKER = re.compile(r"^# %%(\s+\[markdown\])?\s*$")
REPOSITORY_URL = "https://github.com/PacktPublishing/LLMs-in-Finance"
REPOSITORY_REF = "main"


def markdown_from_comments(lines: list[str]) -> str:
    cleaned = []
    for line in lines:
        if line == "#":
            cleaned.append("")
        elif line.startswith("# "):
            cleaned.append(line[2:])
        elif line.startswith("#"):
            cleaned.append(line[1:])
        else:
            cleaned.append(line)
    return "\n".join(cleaned).strip() + "\n"


def cell_id(path: Path, index: int, source: str) -> str:
    digest = hashlib.sha1(
        f"{path.name}:{index}:{source}".encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return digest[:12]


def parse_source(path: Path) -> list[dict]:
    cells: list[dict] = []
    kind: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if kind is None:
            buffer = []
            return
        source = (
            markdown_from_comments(buffer)
            if kind == "markdown"
            else "\n".join(buffer).rstrip() + "\n"
        )
        if not source.strip():
            buffer = []
            return
        index = len(cells)
        if kind == "markdown":
            cells.append(
                {
                    "cell_type": "markdown",
                    "id": cell_id(path, index, source),
                    "metadata": {},
                    "source": source.splitlines(keepends=True),
                }
            )
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": cell_id(path, index, source),
                    "metadata": {},
                    "outputs": [],
                    "source": source.splitlines(keepends=True),
                }
            )
        buffer = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        match = MARKER.match(raw)
        if match:
            flush()
            kind = "markdown" if match.group(1) else "code"
        else:
            buffer.append(raw)
    flush()
    if not cells:
        raise ValueError(f"No notebook cells found in {path}")
    return cells


def colab_cells(path: Path) -> list[dict]:
    """Return deterministic GitHub/Colab links and an offline-safe bootstrap."""

    notebook_name = f"{path.stem}.ipynb"
    github_url = f"{REPOSITORY_URL}/blob/{REPOSITORY_REF}/notebooks/{notebook_name}"
    colab_url = (
        "https://colab.research.google.com/github/"
        f"PacktPublishing/LLMs-in-Finance/blob/{REPOSITORY_REF}/notebooks/{notebook_name}"
    )
    markdown = (
        f'[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]'
        f"({colab_url}) · [View source on GitHub]({github_url})\n\n"
        "The next cell is inert outside Google Colab. In Colab it retrieves the "
        "official repository and installs the local package before the lab runs.\n"
    )
    bootstrap = f"""# Google Colab bootstrap — inert during local and CI execution.
import os as _os
from pathlib import Path as _Path
import subprocess as _subprocess
import sys as _sys

try:
    import google.colab as _google_colab  # type: ignore[import-not-found]
except ImportError:
    _in_colab = False
else:
    _in_colab = True

if _in_colab:
    _repo = _Path("/content/LLMs-in-Finance")
    if _repo.exists() and not (_repo / ".git").is_dir():
        raise RuntimeError(f"Refusing to overwrite non-repository path: {{_repo}}")
    if _repo.exists():
        _subprocess.run(
            ["git", "-C", str(_repo), "fetch", "--depth", "1", "origin", "{REPOSITORY_REF}"],
            check=True,
        )
        _subprocess.run(
            ["git", "-C", str(_repo), "checkout", "--detach", "FETCH_HEAD"],
            check=True,
        )
    else:
        _subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "{REPOSITORY_REF}",
                "{REPOSITORY_URL}.git",
                str(_repo),
            ],
            check=True,
        )
    _subprocess.run(
        [_sys.executable, "-m", "pip", "install", "-q", "-e", str(_repo)],
        check=True,
    )
    _os.chdir(_repo)
"""
    return [
        {
            "cell_type": "markdown",
            "id": cell_id(path, -2, markdown),
            "metadata": {"tags": ["launch-links"]},
            "source": markdown.splitlines(keepends=True),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": cell_id(path, -1, bootstrap),
            "metadata": {"tags": ["colab-bootstrap"]},
            "outputs": [],
            "source": bootstrap.splitlines(keepends=True),
        },
    ]


def build(path: Path) -> Path:
    title = path.stem.replace("_", " ")
    source_cells = parse_source(path)
    cells = [source_cells[0], *colab_cells(path), *source_cells[1:]]
    notebook = {
        "cells": cells,
        "metadata": {
            "book": {
                "title": "Large Language Models in Finance",
                "author": "Miquel Noguer i Alonso",
                "source": path.name,
                "data_policy": (
                    "fictional teaching fixtures by default; optional external "
                    "adapters are disabled"
                ),
                "repository": REPOSITORY_URL,
                "repository_ref": REPOSITORY_REF,
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11+",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
            "title": title,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / f"{path.stem}.ipynb"
    destination.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return destination


def main() -> None:
    sources = sorted(p for p in SOURCE_DIR.glob("*.py") if not p.name.startswith("_"))
    if not sources:
        raise SystemExit(f"No sources found in {SOURCE_DIR}")
    built = [build(path) for path in sources]
    print(f"Built {len(built)} notebooks:")
    for path in built:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
