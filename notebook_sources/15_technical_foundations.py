# %% [markdown]
# # Chapter 15 Lab: Executable Technical Foundations
#
# **Runtime:** 2–3 minutes · **Difficulty:** intermediate · **Credentials:** none
#
# This notebook is a compact acceptance test for the suite's central invariants: stable
# attention, typed evidence, time checks, claim support, calibration, policy gates, hashes,
# and reproducible files.

# %% [markdown]
# ## Learning objectives
#
# - translate mathematical assumptions into executable assertions;
# - verify typed contracts and time awareness;
# - exercise retrieval, faithfulness, calibration, and authorization;
# - fingerprint datasets and run configuration; and
# - connect each book object to a notebook implementation.

# %%
from pathlib import Path
import json
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from finllm_lab import Claim, Document, EvidenceBundle, ToolCall, canonical_hash, seed_everything
from finllm_lab.core import file_sha256
from finllm_lab.governance import GuardedExecutor, PolicyGate, ToolSchema
from finllm_lab.metrics import brier_score, expected_calibration_error
from finllm_lab.rag import BM25Index, faithfulness_score
from finllm_lab.temporal import (
    TemporalLeakageError,
    assert_documents_available,
    availability_audit,
    chronological_split,
)
from finllm_lab.text import multi_head_attention, scaled_dot_product_attention, stable_softmax

rng = seed_everything(1515)

# %% [markdown]
# ## 1. Numerical stability and attention invariants

# %%
softmax = stable_softmax(np.array([1000.0, 1001.0, 1002.0]))
q = rng.normal(size=(7, 5))
k = rng.normal(size=(7, 5))
v = rng.normal(size=(7, 4))
context, weights = scaled_dot_product_attention(q, k, v)
batched_context, batched_weights = scaled_dot_product_attention(
    np.stack([q, q]),
    np.stack([k, k]),
    np.stack([v, v]),
)
inputs = rng.normal(size=(7, 8))
projections = [
    (
        rng.normal(size=(8, 4)),
        rng.normal(size=(8, 4)),
        rng.normal(size=(8, 4)),
    )
    for _ in range(2)
]
multi_context, multi_weights = multi_head_attention(
    inputs,
    projections,
    rng.normal(size=(8, 8)),
)
attention_checks = {
    "softmax_finite": bool(np.isfinite(softmax).all()),
    "softmax_sums_to_one": bool(np.isclose(softmax.sum(), 1.0)),
    "attention_rows_sum_to_one": bool(np.allclose(weights.sum(axis=1), 1.0)),
    "context_shape_correct": context.shape == (7, 4),
    "batched_attention_shape_correct": batched_context.shape == (2, 7, 4),
    "batched_attention_rows_sum_to_one": bool(
        np.allclose(batched_weights.sum(axis=-1), 1.0)
    ),
    "multi_head_output_shape_correct": multi_context.shape == (7, 8),
    "multi_head_weight_shape_correct": multi_weights.shape == (2, 7, 7),
}
attention_checks

# %% [markdown]
# ## 2. Evidence and temporal invariants

# %%
decision_time = datetime(2026, 7, 1, 12, tzinfo=timezone.utc)
valid_doc = Document(
    "doc-valid",
    "Revenue increased by 8 percent in fiscal 2025.",
    "fictional filing",
    datetime(2026, 2, 15, tzinfo=timezone.utc),
    issuer="ACME",
)
future_doc = Document(
    "doc-future",
    "Revenue increased by 12 percent in fiscal 2026.",
    "fictional filing",
    datetime(2026, 8, 15, tzinfo=timezone.utc),
    issuer="ACME",
)
assert_documents_available([valid_doc], decision_time)
try:
    assert_documents_available([future_doc], decision_time)
    future_blocked = False
except TemporalLeakageError:
    future_blocked = True

evidence = tuple(BM25Index([valid_doc, future_doc]).search("How did revenue change?", k=2, decision_time=decision_time))
claim = Claim("claim-15", "Revenue increased by 8 percent in fiscal 2025.")
evidence_checks = {
    "future_document_blocked": future_blocked,
    "retrieval_excludes_future": all(item.document_id != "doc-future" for item in evidence),
    "faithfulness": faithfulness_score([EvidenceBundle(claim, evidence)]),
}

availability_fixture = pd.DataFrame(
    {
        "decision": [decision_time, decision_time],
        "artifact": [decision_time, pd.NaT],
    }
)
availability_result = availability_audit(
    availability_fixture,
    "decision",
    ["artifact"],
).iloc[0]
tie_timestamps = pd.to_datetime(
    [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
        "2026-02-01T00:00:00Z",
        "2026-03-01T00:00:00Z",
        "2026-04-01T00:00:00Z",
    ]
)
tie_split = chronological_split(
    tie_timestamps,
    train_fraction=0.40,
    validation_fraction=0.20,
)
tie_windows = [
    set(tie_timestamps.take(indices))
    for indices in (tie_split.train, tie_split.validation, tie_split.test)
]
temporal_contract_checks = {
    "missing_availability_fails": availability_result["status"] == "FAIL",
    "timestamp_tie_not_split": not (
        tie_windows[0] & tie_windows[1] or tie_windows[1] & tie_windows[2]
    ),
    "all_temporal_windows_nonempty": all(
        len(part) > 0 for part in (tie_split.train, tie_split.validation, tie_split.test)
    ),
}
evidence_checks

# %% [markdown]
# ## 3. Calibration invariants

# %%
probabilities = np.array([0.05, 0.15, 0.30, 0.72, 0.88, 0.96])
labels = np.array([0, 0, 1, 1, 1, 1])
calibration_checks = {
    "brier": brier_score(probabilities, labels),
    "ece": expected_calibration_error(probabilities, labels, bins=3),
    "probabilities_valid": bool(((probabilities >= 0) & (probabilities <= 1)).all()),
}
calibration_checks

# %% [markdown]
# ## 4. Authorization invariant

# %%
tool_counter = {"calls": 0}

def calculate_var(portfolio_id: str) -> dict:
    tool_counter["calls"] += 1
    return {"portfolio_id": portfolio_id, "var_usd": 82_500}

gate = PolicyGate(
    {"risk_agent": {"calculate_var"}},
    schemas={
        "calculate_var": ToolSchema(
            "calculate_var",
            required_arguments=frozenset({"portfolio_id"}),
        )
    },
)
executor = GuardedExecutor(
    {"calculate_var": calculate_var},
    gate,
)
allowed_call = ToolCall(
    "req-15-a",
    "risk_agent",
    "calculate_var",
    {"portfolio_id": "P-001"},
    decision_time,
)
rejected_call = ToolCall(
    "req-15-b",
    "research_agent",
    "calculate_var",
    {"portfolio_id": "P-001"},
    decision_time,
)
allowed_decision, _ = executor.execute(allowed_call)
rejected_decision, _ = executor.execute(rejected_call)
executor.execute(allowed_call)
conflicting_call = ToolCall(
    "req-15-a",
    "risk_agent",
    "calculate_var",
    {"portfolio_id": "P-999"},
    decision_time,
)
conflict_decision, _ = executor.execute(conflicting_call)
published_head = executor.head_hash
chain_report = executor.verify_chain(expected_head=published_head)
authorization_checks = {
    "authorized_call_allowed": allowed_decision.status == "allow",
    "unauthorized_call_rejected": rejected_decision.status == "reject",
    "idempotent_retry_executed_once": tool_counter["calls"] == 1,
    "conflicting_key_reuse_rejected": conflict_decision.status == "reject",
    "audit_anchored_to_genesis": chain_report["anchored_to_genesis"],
    "audit_links_intact": chain_report["links_intact"],
    "published_head_matches": chain_report["head_matches"],
}
authorization_checks

# %% [markdown]
# ## 5. Dataset fingerprints

# %%
data_files = sorted(
    path
    for path in (ROOT / "data").iterdir()
    if path.is_file() and path.suffix in {".csv", ".json", ".jsonl"}
)
hashes = pd.DataFrame(
    {
        "file": [path.name for path in data_files],
        "bytes": [path.stat().st_size for path in data_files],
        "sha256": [file_sha256(path) for path in data_files],
    }
)
hashes

# %% [markdown]
# ## 6. Book-to-notebook crosswalk

# %%
crosswalk = pd.DataFrame(
    [
        ("Typed contracts and audit objects", "00, 05, 06, 14, 16"),
        ("Point-in-time checks", "04, 05, 13, 15, 16"),
        ("Deterministic retrieval", "04, 08, 15, 16"),
        ("Policy-gated tools", "00, 06, 14, 15, 16"),
        ("Calibration and task metrics", "03, 07, 11, 15"),
        ("Leakage experiment", "13"),
        ("Systemic-risk controls", "14"),
    ],
    columns=["book_object", "notebooks"],
)
crosswalk

# %% [markdown]
# ## 7. Acceptance record

# %%
boolean_checks = {
    **attention_checks,
    "future_document_blocked": evidence_checks["future_document_blocked"],
    "retrieval_excludes_future": evidence_checks["retrieval_excludes_future"],
    "faithfulness_is_one": evidence_checks["faithfulness"] == 1.0,
    "probabilities_valid": calibration_checks["probabilities_valid"],
    **temporal_contract_checks,
    **authorization_checks,
}
acceptance = {
    "status": "PASS" if all(boolean_checks.values()) else "FAIL",
    "checks": boolean_checks,
    "python": platform.python_version(),
    "data_hashes": dict(zip(hashes["file"], hashes["sha256"])),
}
acceptance["fingerprint"] = canonical_hash(acceptance)
print("Acceptance:", acceptance["status"])
print("Fingerprint:", acceptance["fingerprint"])
pd.DataFrame({"check": list(boolean_checks), "pass": list(boolean_checks.values())})

# %% [markdown]
# ## Takeaways
#
# - Mathematical assumptions become credible when code and governance enforce them.
# - Tests should fail closed on naive timestamps and future evidence.
# - Task quality, calibration, faithfulness, and policy compliance need separate checks.
# - Reproducibility includes data and configuration hashes, not only a random seed.
#
# **Next:** run the capstone to assemble the full governed earnings-intelligence workflow.
