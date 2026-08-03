"""Write a machine-readable manifest for the current executed release."""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "LLM-in-Finance-notebooks-v1.1.0"
REPOSITORY_URL = "https://github.com/PacktPublishing/LLMs-in-Finance"
TRANSIENT_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def notebook_record(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    outputs = [output for cell in code_cells for output in cell.get("outputs", [])]
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "code_cells": len(code_cells),
        "figures": sum("image/png" in output.get("data", {}) for output in outputs),
        "error_outputs": sum(output.get("output_type") == "error" for output in outputs),
        "stderr_streams": sum(
            output.get("output_type") == "stream" and output.get("name") == "stderr"
            for output in outputs
        ),
        "fully_executed": all(cell.get("execution_count") is not None for cell in code_cells),
        "colab_bootstrap_cells": sum(
            "colab-bootstrap" in cell.get("metadata", {}).get("tags", [])
            for cell in code_cells
        ),
    }


def file_record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def data_paths() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "data").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )


def repository_paths() -> list[Path]:
    paths = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(
            part in TRANSIENT_PARTS or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if relative.parts[0] in {"data", "notebooks"}:
            continue
        if relative == Path("RELEASE_MANIFEST.json"):
            continue
        if path.suffix in {".pyc", ".zip"}:
            continue
        paths.append(path)
    return sorted(paths)


def discovered_test_count() -> int:
    added_paths = [str(ROOT), str(ROOT / "src")]
    for path in reversed(added_paths):
        sys.path.insert(0, path)
    try:
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        pending = [suite]
        while pending:
            item = pending.pop()
            if isinstance(item, unittest.TestSuite):
                pending.extend(item)
            elif item.__class__.__name__ == "_FailedTest":
                raise RuntimeError(f"test discovery failed while importing {item}")
        return suite.countTestCases()
    finally:
        for path in added_paths:
            if path in sys.path:
                sys.path.remove(path)


def project_metadata() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]


def main() -> None:
    notebooks = [notebook_record(path) for path in sorted((ROOT / "notebooks").glob("*.ipynb"))]
    data = [file_record(path) for path in data_paths()]
    repository = [file_record(path) for path in repository_paths()]
    manifest = {
        "release": RELEASE,
        "release_date": "2026-07-30",
        "book": {
            "title": "Large Language Models in Finance",
            "author": "Miquel Noguer i Alonso",
            "publisher": "Packt Publishing",
            "isbn": "978-1-83702-453-7",
            "production_id": "B32413",
        },
        "scope": "additive educational notebook layer; does not replace canonical companion tag",
        "repository": {
            "url": REPOSITORY_URL,
            "default_branch": "main",
            "license": "MIT",
            "python_requires": project_metadata()["requires-python"],
        },
        "provenance": {
            "trusted_baseline_archive": "Large_Language_Models_in_Finance_Notebook_Suite_v1.0.0.zip",
            "trusted_baseline_sha256": "f7da74eed510ad959898e402172cb5b5b39911d07c420143b512c91b42aa3522",
            "verified_replacement_bundle": "files - 2026-07-28T102259.194.zip",
            "verified_replacement_sha256": "5ddf57d02a67b1690bac33f7449c30b3fdaac70a85d74d7f5524754663289f96",
            "replacement_file_count": 10,
            "untrusted_concurrent_tree_used": False,
        },
        "notebooks": notebooks,
        "data_files": data,
        "repository_files": repository,
        "verification": {
            "notebook_count": len(notebooks),
            "core_notebook_count": sum(
                not item["path"].endswith("17_optional_real_model_and_sec.ipynb")
                for item in notebooks
            ),
            "optional_extension_count": sum(
                item["path"].endswith("17_optional_real_model_and_sec.ipynb")
                for item in notebooks
            ),
            "code_cells": sum(item["code_cells"] for item in notebooks),
            "figures": sum(item["figures"] for item in notebooks),
            "error_outputs": sum(item["error_outputs"] for item in notebooks),
            "stderr_streams": sum(item["stderr_streams"] for item in notebooks),
            "all_fully_executed": all(item["fully_executed"] for item in notebooks),
            "credential_required_for_default_run": False,
            "external_adapters_enabled_in_default_run": False,
            "live_execution_enabled": False,
            "all_colab_bootstraps_present": all(
                item["colab_bootstrap_cells"] == 1 for item in notebooks
            ),
            "readme_preview_pngs": len(
                list((ROOT / "assets" / "previews").glob("*.png"))
            ),
            "unit_tests_discovered": discovered_test_count(),
        },
    }
    destination = ROOT / "RELEASE_MANIFEST.json"
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {destination.name}")
    print(json.dumps(manifest["verification"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
