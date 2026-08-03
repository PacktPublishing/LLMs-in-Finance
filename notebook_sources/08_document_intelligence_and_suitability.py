# %% [markdown]
# # Chapter 8 Lab: Financial Documents, Covenants, and Suitability
#
# **Runtime:** 2–3 minutes · **Difficulty:** intermediate · **Credentials:** none
#
# This notebook combines text extraction with deterministic financial controls. It extracts
# loan terms, recomputes covenant headroom, retrieves a time-valid liquidity passage, and
# applies client-suitability rules to hypothetical proposals.

# %% [markdown]
# ## Learning objectives
#
# - preserve exact contract terms and spans;
# - separate extraction from covenant calculation;
# - escalate missing or breached terms;
# - synthesize contract and filing evidence; and
# - keep advisory text behind deterministic suitability controls.

# %%
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from finllm_lab import Document, canonical_hash, seed_everything
from finllm_lab.documents import covenant_checks, extract_contract_terms, suitability_gate
from finllm_lab.plotting import COLORS, finish, set_finance_theme
from finllm_lab.rag import BM25Index

set_finance_theme()
seed_everything(808)
contracts = [json.loads(line) for line in (ROOT / "data" / "contracts.jsonl").read_text().splitlines()]
clients = json.loads((ROOT / "data" / "client_profiles.json").read_text())

# %% [markdown]
# ## 1. Extract terms, then calculate
#
# The text component proposes structured terms. The numerical component accepts only typed
# values and recomputes compliance.

# %%
required_terms = {
    "facility_amount_mn",
    "maturity_years",
    "spread_bps",
    "max_leverage",
    "min_coverage",
    "currency",
}
contract_rows, check_rows = [], []
for contract in contracts:
    terms = extract_contract_terms(contract["text"])
    contract_rows.append(
        {
            "contract_id": contract["contract_id"],
            "ticker": contract["ticker"],
            **terms,
            "extraction_complete": (
                required_terms.issubset(terms) and "currency_conflict" not in terms
            ),
        }
    )
    for check in covenant_checks(
        terms,
        contract["observed_leverage"],
        contract["observed_interest_coverage"],
    ):
        check_rows.append(
            {
                "ticker": contract["ticker"],
                "covenant": check.name,
                "observed": check.observed,
                "threshold": check.threshold,
                "operator": check.operator,
                "headroom": check.headroom,
                "compliant": check.compliant,
            }
        )
terms_table = pd.DataFrame(contract_rows)
checks_table = pd.DataFrame(check_rows)
display(terms_table)
checks_table

# %%
fig, ax = plt.subplots(figsize=(8.5, 4.2))
colors = [COLORS["green"] if ok else COLORS["red"] for ok in checks_table["compliant"]]
labels = checks_table["ticker"] + " · " + checks_table["covenant"]
ax.barh(labels, checks_table["headroom"], color=colors)
ax.axvline(0, color=COLORS["navy"], linewidth=1)
ax.set(title="Covenant headroom", xlabel="Positive = compliant headroom")
finish(ax, "Fictional contracts and observations")
plt.tight_layout()

# %% [markdown]
# ## 2. Fail closed on an incomplete contract

# %%
incomplete_text = (
    "The borrower enters a USD revolving facility of $100 million with a maturity of 3 years. "
    "The leverage ratio shall not exceed 3.25x."
)
incomplete_terms = extract_contract_terms(incomplete_text)
required = {
    "facility_amount_mn",
    "maturity_years",
    "spread_bps",
    "max_leverage",
    "min_coverage",
    "currency",
}
missing = sorted(required - set(incomplete_terms))
incomplete_decision = {
    "status": "escalate" if missing else "continue",
    "missing_fields": missing,
    "extracted_terms": incomplete_terms,
}
incomplete_decision

# %% [markdown]
# ## 3. Join contract logic with filing evidence
#
# A covenant calculation is only as current as the observations behind it. We retrieve the
# latest fictional liquidity section for VELA and attach its evidence identifier.

# %%
filing_rows = [json.loads(line) for line in (ROOT / "data" / "filings.jsonl").read_text().splitlines()]
filing_docs = tuple(
    Document(
        document_id=row["document_id"],
        text=row["text"],
        source=row["source"],
        available_at=pd.Timestamp(row["available_at"]).to_pydatetime(),
        issuer=row["ticker"],
        section=row["section"],
        metadata={"period": row["period"]},
    )
    for row in filing_rows
)
decision_time = pd.Timestamp("2026-02-20T18:00:00Z")
liquidity_evidence = BM25Index(filing_docs).search(
    "VELA net leverage and interest coverage 2025Q4",
    k=3,
    decision_time=decision_time.to_pydatetime(),
    issuer="VELA",
    section="liquidity",
    period="2025Q4",
)
pd.DataFrame(
    [
        {
            "document_id": item.document_id,
            "score": item.score,
            "passage": item.passage,
        }
        for item in liquidity_evidence
    ]
)

# %% [markdown]
# ## 4. Suitability is a deterministic constraint

# %%
proposals = [
    {
        "proposal_id": "short-duration-bond",
        "risk_score": 2,
        "horizon_years": 3,
        "liquidity_days": 2,
        "complex": False,
    },
    {
        "proposal_id": "private-credit-note",
        "risk_score": 4,
        "horizon_years": 10,
        "liquidity_days": 120,
        "complex": True,
    },
]
suitability_rows = []
for profile in clients:
    for proposal in proposals:
        status, reasons = suitability_gate(profile, proposal)
        suitability_rows.append(
            {
                "client": profile["client_id"],
                "proposal": proposal["proposal_id"],
                "status": status,
                "reasons": "; ".join(reasons) or "none",
            }
        )
suitability = pd.DataFrame(suitability_rows)
suitability

# %% [markdown]
# ## 5. Assemble a review packet

# %%
vela_checks = checks_table[checks_table["ticker"] == "VELA"]
review_packet = {
    "case_id": "credit-review-VELA-2025Q4",
    "decision_time": decision_time.isoformat(),
    "contract_id": "facility-VELA-01",
    "extraction_complete": bool(
        terms_table.loc[terms_table["ticker"] == "VELA", "extraction_complete"].iloc[0]
    ),
    "covenant_results": vela_checks.to_dict("records"),
    "evidence_ids": [item.evidence_id for item in liquidity_evidence],
    "status": "escalate" if (~vela_checks["compliant"]).any() else "review",
    "allowed_actions": (
        ["request_updated_financials", "escalate_to_credit_officer"]
        if (~vela_checks["compliant"]).any()
        else ["close_as_clear", "request_updated_financials"]
    ),
}
review_packet["fingerprint"] = canonical_hash(review_packet)
review_packet

# %% [markdown]
# ## Takeaways
#
# - Extracted text is an input to calculation, not the calculation itself.
# - Missing terms and negative headroom are explicit escalation conditions.
# - Multi-document reasoning needs timestamps and evidence IDs across every source.
# - Suitability rules belong outside generative prose.
#
# **Challenge:** add amendment precedence so a later contract amendment overrides only the
# terms it explicitly changes.
