"""Create a deterministic, flat release archive for GitHub."""

from __future__ import annotations

import argparse
import hashlib
import tomllib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "LLMs-in-Finance"
FIXED_TIMESTAMP = (2026, 7, 30, 0, 0, 0)
TRANSIENT_PARTS = {
    ".git",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}


def is_transient(relative: Path) -> bool:
    return any(
        part in TRANSIENT_PARTS or part.endswith(".egg-info")
        for part in relative.parts
    )


def release_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return str(tomllib.load(stream)["project"]["version"])


def release_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if is_transient(relative):
            continue
        if path.is_symlink():
            raise ValueError(f"release tree contains a symbolic link: {relative}")
        if not path.is_file():
            continue
        if path.suffix in {".pyc", ".zip"} or path.name == ".DS_Store":
            continue
        files.append(path)
    return sorted(files)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_archive(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    version = release_version()
    destination = output_dir / f"LLMs-in-Finance-notebook-suite-v{version}.zip"
    with ZipFile(
        destination,
        mode="w",
        compression=ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in release_files():
            relative = path.relative_to(ROOT).as_posix()
            info = ZipInfo(f"{ARCHIVE_ROOT}/{relative}", FIXED_TIMESTAMP)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, path.read_bytes(), compresslevel=9)

    checksum_path = destination.with_suffix(".zip.sha256")
    checksum_path.write_text(
        f"{sha256(destination)}  {destination.name}\n",
        encoding="utf-8",
    )
    return destination, checksum_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination, checksum_path = write_archive(args.output_dir.resolve())
    print(f"Wrote {destination}")
    print(f"Wrote {checksum_path}")
    print(f"SHA-256: {sha256(destination)}")
    print(f"Bytes: {destination.stat().st_size}")


if __name__ == "__main__":
    main()
