from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .contracts import Document, ensure_aware


class TemporalLeakageError(ValueError):
    """Raised when an artifact was not available at decision time."""


def assert_documents_available(
    documents: Iterable[Document], decision_time: datetime
) -> tuple[Document, ...]:
    decision_time = ensure_aware(decision_time, "decision_time")
    docs = tuple(documents)
    invalid = [d.document_id for d in docs if d.available_at > decision_time]
    if invalid:
        raise TemporalLeakageError(
            "Documents unavailable at decision time: " + ", ".join(sorted(invalid))
        )
    return docs


def filter_available(
    documents: Iterable[Document], decision_time: datetime
) -> tuple[Document, ...]:
    decision_time = ensure_aware(decision_time, "decision_time")
    return tuple(d for d in documents if d.available_at <= decision_time)


@dataclass(frozen=True, eq=False)
class TemporalSplit:
    """Index arrays for a chronological split.

    ``eq=False`` because the fields are NumPy arrays and a generated ``__eq__``
    would raise on an ambiguous array truth value.
    """

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def chronological_split(
    timestamps: Sequence,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> TemporalSplit:
    """Split indices chronologically, never separating rows that share a timestamp.

    The requested fractions locate a provisional cut; the cut is then advanced to
    the next distinct timestamp so one decision time cannot appear in two
    windows. A stable sort keeps ordering reproducible when timestamps tie.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie in (0, 1)")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must lie in [0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")

    ts = pd.to_datetime(pd.Series(timestamps), utc=True)
    if ts.isna().any():
        raise ValueError("timestamps must not contain missing values")
    order = np.argsort(ts.to_numpy(), kind="stable")
    sorted_ts = ts.to_numpy()[order]
    n = len(order)

    def snap(position: int) -> int:
        position = int(np.clip(position, 0, n))
        while 0 < position < n and sorted_ts[position] == sorted_ts[position - 1]:
            position += 1
        return position

    a = min(snap(max(1, int(n * train_fraction))), n)
    b = min(snap(max(a + 1, int(n * (train_fraction + validation_fraction)))), n)
    split = TemporalSplit(order[:a], order[a:b], order[b:])
    empty = [
        name
        for name, part in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
        if len(part) == 0
    ]
    if empty:
        raise ValueError(
            "chronological_split produced empty window(s): "
            + ", ".join(empty)
            + f"; {len(np.unique(sorted_ts))} distinct timestamps cannot support the "
            "requested fractions without splitting a timestamp across windows"
        )
    return split


def availability_audit(
    frame: pd.DataFrame,
    decision_col: str,
    availability_cols: Sequence[str],
) -> pd.DataFrame:
    """Audit artifact availability against decision time, failing closed on nulls.

    A missing availability timestamp is a violation, not a pass: the pipeline
    cannot demonstrate the artifact existed at decision time. Comparing a NaT
    yields False, so a naive audit silently reports PASS on unknown provenance.
    """
    decisions = pd.to_datetime(frame[decision_col], utc=True)
    rows = []
    for col in availability_cols:
        available = pd.to_datetime(frame[col], utc=True)
        missing = available.isna() | decisions.isna()
        late = available > decisions
        violations = missing | late
        rows.append(
            {
                "artifact": col,
                "violations": int(violations.sum()),
                "missing_timestamps": int(missing.sum()),
                "violation_rate": float(violations.mean()),
                "status": "FAIL" if violations.any() else "PASS",
            }
        )
    return pd.DataFrame(rows)
