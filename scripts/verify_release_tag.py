"""Fail when a GitHub release tag does not match the package version."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def expected_tag() -> str:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    return f"notebook-suite-v{version}"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_release_tag.py TAG")
    actual = sys.argv[1]
    expected = expected_tag()
    if actual != expected:
        raise SystemExit(f"release tag {actual!r} does not match expected {expected!r}")
    print(f"PASS: release tag {actual}")


if __name__ == "__main__":
    main()
