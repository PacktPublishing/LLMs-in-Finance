from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int = 42) -> np.random.Generator:
    """Seed standard generators and return an explicit NumPy generator.

    Note: ``PYTHONHASHSEED`` is deliberately not set here. It only takes effect
    when read at interpreter start-up, so assigning it from inside a running
    process is a no-op and would give a false impression of control. Set it in
    the environment before launching Python if string-hash determinism matters.
    """
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root from either the root or the notebooks directory."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "notebooks").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root containing pyproject.toml")


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    """Serialize an object deterministically for audit hashing."""
    return json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def canonical_hash(value: Any) -> str:
    """Return a SHA-256 hash of canonical JSON."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
