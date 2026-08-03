# %% [markdown]
# # Start Here: The Governed Financial LLM Lab
#
# **Book:** *Large Language Models in Finance*  
# **Author:** Miquel Noguer i Alonso  
# **Runtime:** about 1 minute · **Difficulty:** introductory · **Credentials:** none
#
# This is the front door to the companion notebook suite. It checks the environment,
# introduces the fictional point-in-time data, and executes the book's central contract:
#
# > time-valid evidence → model proposal → deterministic control → human review
#
# All default examples are CPU-friendly, deterministic, and safe to publish. They never
# contact a live broker, use private data, or require an LLM API.

# %% [markdown]
# ## Learning objectives
#
# By the end of this lab, you will be able to:
#
# 1. navigate the 17-notebook learning path;
# 2. distinguish publication time from decision-time availability;
# 3. link a material claim to identifiable evidence;
# 4. pass a typed tool request through a fail-closed policy gate; and
# 5. produce a reproducibility fingerprint for a run.

# %%
from pathlib import Path
import json
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from finllm_lab import Claim, Document, EvidenceBundle, ToolCall, canonical_hash, seed_everything
from finllm_lab.governance import GuardedExecutor, PolicyGate, ToolSchema
from finllm_lab.plotting import COLORS, finish, set_finance_theme
from finllm_lab.rag import BM25Index, faithfulness_score
from finllm_lab.temporal import assert_documents_available

set_finance_theme()
rng = seed_everything(32413)
pd.set_option("display.max_colwidth", 70)
print("Repository: PacktPublishing/LLMs-in-Finance")
print("Python compatibility: 3.11+ | Default execution: portable CPU")

# %% [markdown]
# ## 1. The learning path
#
# The sequence follows the book chapter by chapter. Notebooks 04–06 form the core
# governed-RAG-and-agents path; notebooks 11–15 form the assurance path. The capstone
# combines both.

# %%
path = pd.DataFrame(
    [
        ("00", "Start here", "foundation", 10),
        ("01", "Earnings-call assistant", "foundation", 25),
        ("02", "Transformer mechanics", "foundation", 30),
        ("03", "Fine-tuning & calibration", "models", 35),
        ("04", "Point-in-time RAG", "systems", 40),
        ("05", "Governed multi-agent research", "systems", 40),
        ("06", "Hardened tool invocation", "systems", 35),
        ("07", "Front / middle / back office", "applications", 35),
        ("08", "Documents & suitability", "applications", 30),
        ("09", "Preference optimization", "models", 35),
        ("10", "Inference & capacity", "infrastructure", 30),
        ("11", "Governance control tower", "assurance", 35),
        ("12", "Evidence matrix", "assurance", 30),
        ("13", "Point-in-time forecasting", "assurance", 40),
        ("14", "Bounded autonomous agents", "frontier", 40),
        ("15", "Technical foundations", "assurance", 30),
        ("16", "Capstone", "capstone", 55),
    ],
    columns=["notebook", "lab", "track", "estimated_minutes"],
)
path

# %%
track_colors = {
    "foundation": COLORS["blue"],
    "models": COLORS["teal"],
    "systems": COLORS["coral"],
    "applications": COLORS["gold"],
    "infrastructure": COLORS["slate"],
    "assurance": COLORS["green"],
    "frontier": COLORS["red"],
    "capstone": COLORS["navy"],
}
fig, ax = plt.subplots(figsize=(9, 6.2))
ordered = path.iloc[::-1]
ax.barh(
    ordered["notebook"] + "  " + ordered["lab"],
    ordered["estimated_minutes"],
    color=[track_colors[t] for t in ordered["track"]],
)
ax.set_title("Notebook learning path")
ax.set_xlabel("Estimated minutes")
finish(ax, "Repository curriculum")
plt.tight_layout()

# %% [markdown]
# ## 2. Inspect the data contract
#
# Financial evidence is not admissible merely because it exists. The relevant timestamp
# is when it became available to the system. The next cell confirms that every call has
# an explicit UTC availability time and shows the latest fictional observations.

# %%
calls = pd.read_csv(ROOT / "data" / "earnings_calls.csv", parse_dates=["published_at", "available_at"])
data_check = pd.DataFrame(
    {
        "rows": [len(calls)],
        "issuers": [calls["ticker"].nunique()],
        "first_available": [calls["available_at"].min()],
        "last_available": [calls["available_at"].max()],
        "missing_availability": [int(calls["available_at"].isna().sum())],
        "fictional_rows": [int(calls["data_status"].eq("fictional").sum())],
    }
)
display(data_check)
calls.sort_values("available_at").tail(4)[
    ["ticker", "period", "available_at", "revenue_growth_pct", "sentiment_label"]
]

# %% [markdown]
# ## 3. Execute the minimum evidence contract
#
# A retriever returns identifiers and passages. A claim is scored against that evidence.
# The example deliberately uses a fixed decision time so the output is reproducible.

# %%
decision_time = datetime(2026, 3, 1, 12, tzinfo=timezone.utc)
corpus = (
    Document(
        document_id="filing-ACME-2025Q4-results",
        text="Acme Cloud Systems reported revenue growth of 11.2% for 2025Q4.",
        source="fictional 10-K",
        available_at=datetime(2026, 2, 10, 14, tzinfo=timezone.utc),
        issuer="ACME",
        section="results",
    ),
    Document(
        document_id="filing-ACME-2025Q4-risk",
        text="Key risks included cloud price competition and data-center capacity.",
        source="fictional 10-K",
        available_at=datetime(2026, 2, 10, 14, 5, tzinfo=timezone.utc),
        issuer="ACME",
        section="risk_factors",
    ),
)
assert_documents_available(corpus, decision_time)
evidence = tuple(BM25Index(corpus).search("How fast did ACME revenue grow?", k=2, decision_time=decision_time))
claim = Claim("claim-001", "ACME revenue growth was 11.2% in 2025Q4.")
score = faithfulness_score([EvidenceBundle(claim, evidence)])
pd.DataFrame(
    {
        "claim": [claim.text],
        "evidence_ids": [[item.evidence_id for item in evidence]],
        "faithfulness": [score],
    }
)

# %% [markdown]
# ## 4. Put a tool call behind policy
#
# The model may propose a call; authorization remains deterministic. Repeating the same
# read-only request returns the cached result, illustrating idempotent execution.

# %%
call_counter = {"count": 0}

def market_snapshot(symbol: str, as_of: str) -> dict:
    call_counter["count"] += 1
    return {"symbol": symbol, "as_of": as_of, "price": 101.25, "status": "fictional"}

gate = PolicyGate(
    {"research_agent": {"market_snapshot"}},
    schemas={
        "market_snapshot": ToolSchema(
            "market_snapshot",
            required_arguments=frozenset({"symbol", "as_of"}),
        )
    },
)
executor = GuardedExecutor({"market_snapshot": market_snapshot}, gate)
request = ToolCall(
    request_id="req-start-001",
    actor="research_agent",
    tool="market_snapshot",
    arguments={"symbol": "ACME", "as_of": "2026-03-01T12:00:00+00:00"},
    requested_at=decision_time,
)
first = executor.execute(request)
second = executor.execute(request)
pd.DataFrame(
    [
        {"run": "first", "decision": first[0].status, "result": first[1]},
        {"run": "repeat", "decision": second[0].status, "result": second[1]},
    ]
), call_counter

# %% [markdown]
# ## 5. Reproducibility fingerprint
#
# A material output should identify the code/data configuration that produced it. This
# compact fingerprint is not a replacement for a full manifest, but it makes silent changes
# immediately visible.

# %%
run_manifest = {
    "suite_version": "1.1.0",
    "book_production_id": "B32413",
    "seed": 32413,
    "decision_time": decision_time,
    "corpus_ids": [d.document_id for d in corpus],
    "claim": claim,
    "policy_tools": sorted(gate.permissions["research_agent"]),
    "faithfulness": score,
}
fingerprint = canonical_hash(run_manifest)
print("Run fingerprint:", fingerprint)
print("Contract status:", "PASS" if score == 1.0 and first[0].status == "allow" else "FAIL")

# %% [markdown]
# ## Takeaways
#
# - Availability time, not file presence, defines admissible evidence.
# - Material claims need evidence identifiers and numerical support.
# - LLM proposals do not authorize their own tools.
# - Deterministic hashes make versions and silent changes observable.
#
# **Next:** open `01_earnings_call_assistant.ipynb`. After completing the core
# sequence, notebook 17 provides the optional real-model and SEC bridge.
