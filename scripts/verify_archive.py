"""Verify archive safety and byte-for-byte clean-build reproducibility."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

IGNORED_PARTS = {
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.suffix == ".pyc":
            continue
        hashes[relative.as_posix()] = sha256(path)
    return hashes


def validate_members(archive: ZipFile) -> str:
    roots = set()
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive member: {member.filename}")
        if not path.parts:
            raise ValueError("archive contains an empty path")
        roots.add(path.parts[0])
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"archive contains a symbolic link: {member.filename}")
        if path.suffix.lower() == ".zip":
            raise ValueError(f"archive contains a nested ZIP: {member.filename}")
    if roots != {"LLMs-in-Finance"}:
        raise ValueError(f"archive must have one LLMs-in-Finance root, found {sorted(roots)}")
    return "LLMs-in-Finance"


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def verify_archive(path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="finllm-release-") as temp:
        destination = Path(temp)
        with ZipFile(path) as archive:
            root_name = validate_members(archive)
            archive.extractall(destination)
        root = destination / root_name
        before = tree_hashes(root)
        env = os.environ.copy()
        env.update(
            {
                "LC_ALL": "C.UTF-8",
                "MPLBACKEND": "Agg",
                "PYTHONHASHSEED": "0",
                "PYTHONPATH": "src",
                "TZ": "UTC",
            }
        )
        run(
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "src",
                "scripts",
                "tests",
                "notebook_sources",
            ],
            root,
            env,
        )
        run(["ruff", "check", "src", "scripts", "tests"], root, env)
        run([sys.executable, "scripts/generate_data.py"], root, env)
        run([sys.executable, "scripts/build_notebooks.py"], root, env)
        run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-v",
            ],
            root,
            env,
        )
        run([sys.executable, "scripts/execute_notebooks.py", "--in-place"], root, env)
        run([sys.executable, "scripts/export_readme_previews.py"], root, env)
        run([sys.executable, "scripts/write_release_manifest.py"], root, env)
        run([sys.executable, "scripts/verify_release.py"], root, env)
        after = tree_hashes(root)
        changed = sorted(
            key
            for key in set(before) | set(after)
            if before.get(key) != after.get(key)
        )
        if changed:
            details = "\n".join(f"  - {item}" for item in changed)
            raise RuntimeError(f"archive is not byte-for-byte reproducible:\n{details}")
        print("PASS: safe archive and byte-for-byte clean-build reproduction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_archive(args.archive.resolve())


if __name__ == "__main__":
    main()
