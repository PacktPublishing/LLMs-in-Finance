from __future__ import annotations

import json
import re
import tomllib
import unittest
from pathlib import Path

from PIL import Image

import finllm_lab
from scripts.package_release import is_transient
from scripts.write_release_manifest import RELEASE

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_URL = "https://github.com/PacktPublishing/LLMs-in-Finance"


class RepositoryContractTests(unittest.TestCase):
    def test_version_and_repository_identity_are_aligned(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as stream:
            metadata = tomllib.load(stream)
        project = metadata["project"]
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertEqual(project["version"], "1.1.0")
        self.assertEqual(project["requires-python"], ">=3.11")
        self.assertEqual(project["license"], "MIT")
        self.assertNotIn(
            "License :: OSI Approved :: MIT License",
            project.get("classifiers", []),
        )
        self.assertEqual(
            metadata["build-system"]["requires"],
            ["setuptools==82.0.1", "wheel==0.47.0"],
        )
        self.assertEqual(finllm_lab.__version__, "1.1.0")
        self.assertEqual(RELEASE, "LLM-in-Finance-notebooks-v1.1.0")
        self.assertIn("version: 1.1.0", citation)
        self.assertIn(f'repository-code: "{REPOSITORY_URL}"', citation)

    def test_license_matches_official_packt_identity(self) -> None:
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertTrue(license_text.startswith("MIT License\n"))
        self.assertIn("Copyright (c) 2025 Packt", license_text)
        self.assertIn(
            "Permission is hereby granted, free of charge",
            license_text,
        )

    def test_notebooks_have_one_colab_bootstrap(self) -> None:
        notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
        self.assertEqual(len(notebooks), 18)
        for path in notebooks:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            bootstraps = [
                cell
                for cell in notebook["cells"]
                if "colab-bootstrap" in cell.get("metadata", {}).get("tags", [])
            ]
            self.assertEqual(len(bootstraps), 1, path.name)
            self.assertEqual(
                notebook["metadata"]["book"]["repository"],
                REPOSITORY_URL,
            )
            self.assertEqual(notebook["metadata"]["book"]["repository_ref"], "main")

    def test_optional_adapters_are_offline_by_default(self) -> None:
        notebook = json.loads(
            (ROOT / "notebooks" / "17_optional_real_model_and_sec.ipynb").read_text(
                encoding="utf-8"
            )
        )
        source_text = "".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn(
            'RUN_REAL_MODEL = os.getenv("FINLLM_RUN_REAL_MODEL", "0") == "1"',
            source_text,
        )
        self.assertIn(
            'RUN_SEC = os.getenv("FINLLM_RUN_SEC", "0") == "1"',
            source_text,
        )

        outputs = [
            output
            for cell in notebook["cells"]
            for output in cell.get("outputs", [])
        ]
        if outputs:
            output_text = json.dumps(outputs, ensure_ascii=False)
            self.assertGreaterEqual(output_text.count("SKIPPED_BY_DESIGN"), 2)
            self.assertNotIn("RAN —", output_text)

    def test_readme_local_links_and_colab_links_resolve(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        markdown_targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme)
        html_targets = [
            target
            for _, target in re.findall(r'\b(href|src)="([^"]+)"', readme)
        ]
        for target in markdown_targets + html_targets:
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = target.split("#", 1)[0]
            if relative:
                self.assertTrue((ROOT / relative).exists(), target)
        for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            expected = (
                "https://colab.research.google.com/github/"
                "PacktPublishing/LLMs-in-Finance/blob/main/notebooks/"
                f"{path.name}"
            )
            self.assertIn(expected, readme)

    def test_generated_preview_assets_are_valid(self) -> None:
        expected = {
            "attention_mechanics.png",
            "capstone_scorecard.png",
            "governance_dashboard.png",
            "learning_path.png",
            "point_in_time_rag.png",
        }
        previews = sorted((ROOT / "assets" / "previews").glob("*.png"))
        self.assertEqual({path.name for path in previews}, expected)
        for path in [
            *previews,
            ROOT / "assets" / "finllm_suite_demo.gif",
            ROOT / "assets" / "github_social_preview.png",
        ]:
            with Image.open(path) as image:
                image.verify()
        with Image.open(ROOT / "assets" / "github_social_preview.png") as social:
            self.assertEqual(social.size, (1280, 640))

    def test_workflows_use_current_official_actions(self) -> None:
        notebooks = (ROOT / ".github" / "workflows" / "notebooks.yml").read_text()
        codeql = (ROOT / ".github" / "workflows" / "codeql.yml").read_text()
        release = (ROOT / ".github" / "workflows" / "release.yml").read_text()
        for workflow in (notebooks, codeql, release):
            self.assertIn("permissions:", workflow)
            self.assertIn("actions/checkout@v6", workflow)
        self.assertIn("actions/setup-python@v6", notebooks)
        self.assertIn("actions/upload-artifact@v7", notebooks)
        self.assertIn("github/codeql-action/init@v4", codeql)
        self.assertIn("github/codeql-action/analyze@v4", codeql)

    def test_tracked_tree_contains_no_nested_release_archive(self) -> None:
        archives = [
            path
            for path in ROOT.rglob("*.zip")
            if not is_transient(path.relative_to(ROOT))
        ]
        self.assertEqual(archives, [])
        links = [
            path
            for path in ROOT.rglob("*")
            if path.is_symlink() and not is_transient(path.relative_to(ROOT))
        ]
        self.assertEqual(links, [])


if __name__ == "__main__":
    unittest.main()
