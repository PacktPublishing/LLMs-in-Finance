from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class CovenantResult:
    name: str
    observed: float
    threshold: float
    operator: str
    compliant: bool
    headroom: float


def extract_contract_terms(text: str) -> dict[str, float | str]:
    patterns = {
        "maturity_years": r"maturity of ([0-9.]+)\s*years",
        "spread_bps": r"(?:spread|margin) of ([0-9.]+)\s*basis points",
        "max_leverage": r"leverage ratio (?:shall not exceed|of no more than) ([0-9.]+)x",
        "min_coverage": r"interest coverage ratio (?:shall be at least|of no less than) ([0-9.]+)x",
    }
    out: dict[str, float | str] = {}
    amount = re.search(
        r"facility of\s*(?:(USD|EUR|GBP)|([$€£]))?\s*"
        r"([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*million",
        text,
        flags=re.IGNORECASE,
    )
    amount_currency = None
    if amount:
        out["facility_amount_mn"] = float(amount.group(3).replace(",", ""))
        amount_currency = (
            amount.group(1).upper()
            if amount.group(1)
            else {"$": "USD", "€": "EUR", "£": "GBP"}.get(amount.group(2))
        )
    for name, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            out[name] = float(match.group(1))
    declared_currency = re.search(r"\b(USD|EUR|GBP)\b", text, flags=re.IGNORECASE)
    declared = declared_currency.group(1).upper() if declared_currency else None
    currency = declared or amount_currency
    if currency:
        out["currency"] = currency
    if declared and amount_currency and declared != amount_currency:
        out["currency_conflict"] = (
            f"declared {declared} but amount symbol implies {amount_currency}"
        )
    return out


def covenant_checks(
    terms: Mapping[str, float | str],
    leverage: float,
    interest_coverage: float,
) -> list[CovenantResult]:
    results = []
    if "max_leverage" in terms:
        threshold = float(terms["max_leverage"])
        results.append(
            CovenantResult(
                "leverage",
                leverage,
                threshold,
                "<=",
                leverage <= threshold,
                threshold - leverage,
            )
        )
    if "min_coverage" in terms:
        threshold = float(terms["min_coverage"])
        results.append(
            CovenantResult(
                "interest_coverage",
                interest_coverage,
                threshold,
                ">=",
                interest_coverage >= threshold,
                interest_coverage - threshold,
            )
        )
    return results


def suitability_gate(profile: Mapping, proposal: Mapping) -> tuple[str, list[str]]:
    reasons = []
    if proposal["risk_score"] > profile["risk_tolerance"]:
        reasons.append("proposal risk exceeds documented tolerance")
    if proposal["horizon_years"] > profile["horizon_years"]:
        reasons.append("proposal horizon exceeds client horizon")
    if proposal["liquidity_days"] > profile["max_liquidity_days"]:
        reasons.append("proposal liquidity is incompatible with client needs")
    if proposal.get("complex", False) and not profile.get("complex_products_approved", False):
        reasons.append("complex-product approval is absent")
    return ("reject" if reasons else "allow"), reasons
