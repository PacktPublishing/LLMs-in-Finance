"""Verify notebooks, manifest hashes, fixture contracts, and repository safety."""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from write_release_manifest import (  # noqa: E402
    RELEASE,
    data_paths,
    discovered_test_count,
    notebook_record,
    repository_paths,
)

from finllm_lab.core import file_sha256  # noqa: E402
from finllm_lab.documents import extract_contract_terms  # noqa: E402

REPOSITORY_URL = "https://github.com/PacktPublishing/LLMs-in-Finance"
EXPECTED_PREVIEWS = {
    "attention_mechanics.png",
    "capstone_scorecard.png",
    "governance_dashboard.png",
    "learning_path.png",
    "point_in_time_rag.png",
}


def check_manifest_records(
    failures: list[str],
    manifest: dict,
    key: str,
    paths: list[Path],
) -> None:
    records = {item["path"]: item for item in manifest.get(key, [])}
    actual = {str(path.relative_to(ROOT)): path for path in paths}
    missing = sorted(set(actual) - set(records))
    extra = sorted(set(records) - set(actual))
    if missing:
        failures.append(f"{key}: missing manifest records: {', '.join(missing)}")
    if extra:
        failures.append(f"{key}: manifest references absent files: {', '.join(extra)}")
    for relative, path in actual.items():
        record = records.get(relative)
        if record is None:
            continue
        if record.get("bytes") != path.stat().st_size:
            failures.append(f"{relative}: byte count differs from manifest")
        if record.get("sha256") != file_sha256(path):
            failures.append(f"{relative}: SHA-256 differs from manifest")


def main() -> None:
    failures: list[str] = []
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if len(notebooks) != 18:
        failures.append(f"expected 18 notebooks, found {len(notebooks)}")
    for path in notebooks:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{path.name}: invalid JSON ({exc})")
            continue
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        if not code_cells:
            failures.append(f"{path.name}: no code cells")
        if any(
            output.get("output_type") == "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        ):
            failures.append(f"{path.name}: contains error output")
        if any(cell.get("execution_count") is None for cell in code_cells):
            failures.append(f"{path.name}: contains unexecuted code cells")
        if any(
            output.get("output_type") == "stream" and output.get("name") == "stderr"
            for cell in code_cells
            for output in cell.get("outputs", [])
        ):
            failures.append(f"{path.name}: contains stderr output")
        bootstrap_cells = [
            cell
            for cell in code_cells
            if "colab-bootstrap" in cell.get("metadata", {}).get("tags", [])
        ]
        if len(bootstrap_cells) != 1:
            failures.append(
                f"{path.name}: expected one Colab bootstrap, found {len(bootstrap_cells)}"
            )
        metadata = notebook.get("metadata", {}).get("book", {})
        if metadata.get("repository") != REPOSITORY_URL:
            failures.append(f"{path.name}: incorrect repository metadata")
        if metadata.get("repository_ref") != "main":
            failures.append(f"{path.name}: incorrect repository ref")

    extension_path = ROOT / "notebooks" / "17_optional_real_model_and_sec.ipynb"
    if extension_path.exists():
        extension_notebook = json.loads(extension_path.read_text(encoding="utf-8"))
        extension_outputs = json.dumps(
            [
                output
                for cell in extension_notebook["cells"]
                for output in cell.get("outputs", [])
            ],
            ensure_ascii=False,
        )
        if extension_outputs.count("SKIPPED_BY_DESIGN") < 2:
            failures.append("optional extension did not skip both external adapters")
        if "RAN —" in extension_outputs:
            failures.append("optional extension contains a live external-adapter run")

    calls = pd.read_csv(ROOT / "data" / "earnings_calls.csv")
    if not calls["data_status"].eq("fictional").all():
        failures.append("earnings data are not fully marked fictional")
    if len(calls) != 84 or calls["ticker"].nunique() != 4:
        failures.append("earnings fixture must contain 84 calls across 4 issuers")
    for _, issuer in calls.groupby("ticker"):
        observed = issuer["operating_margin_pct"].diff().iloc[1:]
        declared = issuer["margin_change_bps"].iloc[1:] / 100
        if not bool(((observed - declared).abs() <= 0.002).all()):
            failures.append(f"{issuer.iloc[0]['ticker']}: margin/basis-point units disagree")

    contracts = [
        json.loads(line)
        for line in (ROOT / "data" / "contracts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for contract in contracts:
        if "currency_conflict" in extract_contract_terms(contract["text"]):
            failures.append(f"{contract['contract_id']}: currency declaration conflicts with amount")

    panel = pd.read_csv(
        ROOT / "data" / "market_text_panel.csv",
        parse_dates=["decision_time", "return_start_at", "return_end_at"],
    )
    if not (panel["return_start_at"] > panel["decision_time"]).all():
        failures.append("market fixture has a return that begins before its decision")
    if not (panel["return_end_at"] > panel["return_start_at"]).all():
        failures.append("market fixture has a non-positive return window")

    risky_patterns = [
        r"api[_-]?key\s*=",
        r"sk-[A-Za-z0-9]{20,}",
        r"live" + r"[_-]?broker",
    ]
    text_paths = [
        *ROOT.glob("notebook_sources/*.py"),
        *ROOT.glob("src/**/*.py"),
        *ROOT.glob("scripts/*.py"),
    ]
    for path in text_paths:
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in risky_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                failures.append(f"{path.relative_to(ROOT)}: matched forbidden pattern {pattern}")

    nested_archives = sorted(
        path for path in ROOT.rglob("*.zip") if "dist" not in path.relative_to(ROOT).parts
    )
    if nested_archives:
        failures.append(
            "release tree contains nested ZIP files: "
            + ", ".join(str(path.relative_to(ROOT)) for path in nested_archives)
        )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        relative_target = target.split("#", 1)[0]
        if relative_target and not (ROOT / relative_target).exists():
            failures.append(f"README.md: broken relative link {target}")
    for attribute, target in re.findall(
        r"\b(href|src)=\"([^\"]+)\"",
        readme,
    ):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        relative_target = target.split("#", 1)[0]
        if relative_target and not (ROOT / relative_target).exists():
            failures.append(f"README.md: broken HTML {attribute} {target}")
    expected_colab_prefix = (
        "https://colab.research.google.com/github/"
        "PacktPublishing/LLMs-in-Finance/blob/main/notebooks/"
    )
    for path in notebooks:
        expected_url = expected_colab_prefix + path.name
        if expected_url not in readme:
            failures.append(f"README.md: missing Colab link for {path.name}")

    preview_paths = sorted((ROOT / "assets" / "previews").glob("*.png"))
    if {path.name for path in preview_paths} != EXPECTED_PREVIEWS:
        failures.append("README preview PNG set is missing or contains stale files")
    image_paths = [
        *preview_paths,
        ROOT / "assets" / "finllm_suite_demo.gif",
        ROOT / "assets" / "github_social_preview.png",
    ]
    for path in image_paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            failures.append(f"{path.relative_to(ROOT)}: invalid image ({exc})")

    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    if project.get("version") != "1.1.0":
        failures.append("pyproject version is not 1.1.0")
    if project.get("requires-python") != ">=3.11":
        failures.append("pyproject Python requirement must be >=3.11")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if not license_text.startswith("MIT License\n\nCopyright (c) 2025 Packt"):
        failures.append("LICENSE does not match the official Packt MIT identity")

    manifest_path = ROOT / "RELEASE_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"RELEASE_MANIFEST.json: invalid or missing ({exc})")
        manifest = {}
    if manifest:
        if manifest.get("release") != RELEASE:
            failures.append(
                f"manifest release is {manifest.get('release')!r}, expected {RELEASE!r}"
            )
        check_manifest_records(
            failures,
            manifest,
            "notebooks",
            notebooks,
        )
        check_manifest_records(
            failures,
            manifest,
            "data_files",
            data_paths(),
        )
        check_manifest_records(
            failures,
            manifest,
            "repository_files",
            repository_paths(),
        )
        current_notebooks = [notebook_record(path) for path in notebooks]
        current_verification = {
            "notebook_count": len(current_notebooks),
            "core_notebook_count": sum(
                not item["path"].endswith("17_optional_real_model_and_sec.ipynb")
                for item in current_notebooks
            ),
            "optional_extension_count": sum(
                item["path"].endswith("17_optional_real_model_and_sec.ipynb")
                for item in current_notebooks
            ),
            "code_cells": sum(item["code_cells"] for item in current_notebooks),
            "figures": sum(item["figures"] for item in current_notebooks),
            "error_outputs": sum(item["error_outputs"] for item in current_notebooks),
            "stderr_streams": sum(item["stderr_streams"] for item in current_notebooks),
            "all_fully_executed": all(
                item["fully_executed"] for item in current_notebooks
            ),
            "all_colab_bootstraps_present": all(
                item["colab_bootstrap_cells"] == 1 for item in current_notebooks
            ),
            "readme_preview_pngs": len(preview_paths),
            "unit_tests_discovered": discovered_test_count(),
        }
        recorded_verification = manifest.get("verification", {})
        for key, value in current_verification.items():
            if recorded_verification.get(key) != value:
                failures.append(
                    f"manifest verification {key!r} is stale: "
                    f"{recorded_verification.get(key)!r} != {value!r}"
                )
        provenance = manifest.get("provenance", {})
        if provenance.get("untrusted_concurrent_tree_used") is not False:
            failures.append("manifest provenance does not exclude the untrusted tree")
        repository = manifest.get("repository", {})
        if repository.get("url") != REPOSITORY_URL:
            failures.append("manifest repository URL is incorrect")
        if repository.get("license") != "MIT":
            failures.append("manifest license is not MIT")

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("PASS")
    print(f"  notebooks: {len(notebooks)} (17 core + 1 optional bridge)")
    print(f"  fictional earnings rows: {len(calls)}")
    print("  error outputs: 0")
    print("  stderr streams: 0")
    print("  manifest hashes: current")
    print("  obvious credential/live-broker patterns: 0")


if __name__ == "__main__":
    main()
