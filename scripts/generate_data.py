"""Generate all deterministic, fictional datasets used by the notebook suite."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 32413
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def quarter_end(period: pd.Period) -> pd.Timestamp:
    return period.end_time.normalize()


def build_earnings(rng: np.random.Generator) -> pd.DataFrame:
    issuers = {
        "ACME": ("Acme Cloud Systems", 1480.0, 0.09, 24.0),
        "NIMB": ("Nimbus Payments", 920.0, 0.12, 19.0),
        "ORBT": ("Orbit Industrial Analytics", 730.0, 0.07, 17.0),
        "VELA": ("Vela Consumer Finance", 1110.0, 0.06, 21.0),
    }
    periods = pd.period_range("2021Q1", "2026Q1", freq="Q")
    rows: list[dict] = []
    for ticker_index, (ticker, (issuer, base_revenue, trend, base_margin)) in enumerate(
        issuers.items()
    ):
        revenue = base_revenue
        margin = base_margin
        for t, period in enumerate(periods):
            cycle = 3.4 * np.sin((t + ticker_index) / 2.8)
            revenue_growth = 100 * trend + cycle + rng.normal(0, 3.0)
            revenue *= 1 + revenue_growth / 400
            margin_change = 12 * cycle + rng.normal(0, 55)
            # ``margin`` is measured in percentage points and ``margin_change`` in
            # basis points: 100 bps = 1 percentage point.
            margin = float(np.clip(margin + margin_change / 100, 8, 38))
            guidance_delta = float(np.clip(0.45 * cycle + rng.normal(0, 1.6), -6, 6))
            tone = (
                0.13 * revenue_growth
                + 0.007 * margin_change
                + 0.35 * guidance_delta
                + rng.normal(0, 0.55)
            )
            if tone > 1.0:
                label = "positive"
                phrase = "Management described demand as resilient and execution as strong."
            elif tone < -0.8:
                label = "negative"
                phrase = "Management cited softer demand, execution pressure, and elevated uncertainty."
            else:
                label = "neutral"
                phrase = "Management described conditions as mixed and maintained a balanced outlook."
            q_end = quarter_end(period)
            published = (q_end + pd.Timedelta(days=34 + ticker_index)).tz_localize(
                "UTC"
            ) + pd.Timedelta(hours=13)
            availability = published + pd.Timedelta(minutes=12)
            sign_margin = "+" if margin_change >= 0 else ""
            sign_guidance = "+" if guidance_delta >= 0 else ""
            text = (
                f"{issuer} reported {period} revenue of ${revenue:,.1f} million, "
                f"up {revenue_growth:.1f}% year over year. Operating margin was "
                f"{margin:.1f}%, a change of {sign_margin}{margin_change:.0f} basis points. "
                f"Management changed full-year revenue guidance by "
                f"{sign_guidance}{guidance_delta:.1f}%. {phrase} "
                "The call also discussed customer retention, financing conditions, "
                "competitive intensity, and the timing of planned investment."
            )
            rows.append(
                {
                    "document_id": f"call-{ticker}-{period}",
                    "ticker": ticker,
                    "issuer": issuer,
                    "period": str(period),
                    "published_at": published.isoformat(),
                    "available_at": availability.isoformat(),
                    "revenue_usd_mn": round(revenue, 2),
                    "revenue_growth_pct": round(revenue_growth, 3),
                    "operating_margin_pct": round(margin, 3),
                    "margin_change_bps": round(margin_change, 2),
                    "guidance_delta_pct": round(guidance_delta, 3),
                    "sentiment_score": round(tone, 4),
                    "sentiment_label": label,
                    "text": text,
                    "data_status": "fictional",
                }
            )
    frame = pd.DataFrame(rows).sort_values(["available_at", "ticker"]).reset_index(drop=True)
    frame.to_csv(DATA / "earnings_calls.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return frame


def build_filings(rng: np.random.Generator, earnings: pd.DataFrame) -> None:
    risk_text = {
        "ACME": "Key risks include cloud price competition, data-center capacity, cyber incidents, and customer concentration.",
        "NIMB": "Key risks include chargebacks, fraud losses, network availability, regulation, and partner-bank concentration.",
        "ORBT": "Key risks include industrial cyclicality, component availability, project timing, and export controls.",
        "VELA": "Key risks include credit losses, funding spreads, consumer regulation, and model risk in underwriting.",
    }
    docs: list[dict] = []
    latest = earnings[earnings["period"].between("2023Q1", "2026Q1")].copy()
    for _, row in latest.iterrows():
        available = pd.Timestamp(row["available_at"]) + pd.Timedelta(days=2)
        leverage = float(
            np.clip(
                2.4
                - 0.025 * row["revenue_growth_pct"]
                - 0.002 * row["margin_change_bps"]
                + rng.normal(0, 0.18),
                0.8,
                4.8,
            )
        )
        coverage = float(np.clip(6.5 / leverage + rng.normal(0, 0.3), 1.1, 8.0))
        cash = float(row["revenue_usd_mn"] * (0.16 + rng.normal(0, 0.015)))
        debt = float(cash * leverage * 1.8)
        sections = {
            "results": (
                f"For {row['period']}, {row['issuer']} recorded revenue of "
                f"${row['revenue_usd_mn']:,.1f} million, representing "
                f"{row['revenue_growth_pct']:.1f}% year-over-year growth. "
                f"Operating margin was {row['operating_margin_pct']:.1f}%."
            ),
            "liquidity": (
                f"Quarter-end cash was ${cash:,.1f} million and gross debt was "
                f"${debt:,.1f} million. Net leverage was {leverage:.2f}x and "
                f"interest coverage was {coverage:.2f}x. Management stated that "
                "liquidity was sufficient for the current operating plan."
            ),
            "risk_factors": risk_text[row["ticker"]],
            "guidance": (
                f"Management changed full-year revenue guidance by "
                f"{row['guidance_delta_pct']:+.1f}% and said the range remains "
                "sensitive to demand, pricing, and financing conditions."
            ),
        }
        for section, text in sections.items():
            doc_id = f"filing-{row['ticker']}-{row['period']}-{section}"
            docs.append(
                {
                    "document_id": doc_id,
                    "ticker": row["ticker"],
                    "issuer": row["issuer"],
                    "period": row["period"],
                    "form": "10-Q" if not row["period"].endswith("Q4") else "10-K",
                    "section": section,
                    "published_at": available.isoformat(),
                    "available_at": (available + pd.Timedelta(minutes=8)).isoformat(),
                    "text": text,
                    "source": "fictional regulatory filing",
                    "data_status": "fictional",
                }
            )
    write_jsonl(DATA / "filings.jsonl", docs)

    questions: list[dict] = []
    for ticker in sorted(earnings["ticker"].unique()):
        period = "2025Q4"
        matching = [d for d in docs if d["ticker"] == ticker and d["period"] == period]
        decision = max(pd.Timestamp(d["available_at"]) for d in matching) + pd.Timedelta(hours=1)
        issuer = matching[0]["issuer"]
        for section, query in [
            ("results", f"What revenue and growth did {issuer} report for {period}?"),
            ("liquidity", f"What were {issuer}'s leverage and interest coverage for {period}?"),
            ("risk_factors", f"What principal risks did {issuer} disclose?"),
            ("guidance", f"How did {issuer} change revenue guidance?"),
        ]:
            relevant = f"filing-{ticker}-{period}-{section}"
            questions.append(
                {
                    "question_id": f"q-{ticker}-{section}",
                    "ticker": ticker,
                    "query": query,
                    "decision_time": decision.isoformat(),
                    "relevant_document_ids": [relevant],
                }
            )
    (DATA / "rag_questions.json").write_text(
        json.dumps(questions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_market_panel(rng: np.random.Generator) -> None:
    dates = pd.bdate_range("2021-01-04", "2025-12-31", tz="UTC")
    latent = np.zeros(len(dates))
    shock = rng.normal(0, 1, len(dates))
    for i in range(1, len(dates)):
        latent[i] = 0.82 * latent[i - 1] + 0.58 * shock[i]
    next_return = 0.0017 * np.tanh(latent) + rng.normal(0, 0.0105, len(dates))
    observed = latent + rng.normal(0, 0.45, len(dates))
    revised = observed + 45 * next_return + rng.normal(0, 0.12, len(dates))
    regime = np.where(latent > 0.75, "expansion", np.where(latent < -0.75, "stress", "normal"))
    decision_time = dates + pd.Timedelta(hours=16)
    text_available = dates + pd.Timedelta(hours=15, minutes=58)
    revised_available = dates + pd.Timedelta(days=1, hours=12)
    return_start = decision_time + pd.Timedelta(minutes=1)
    return_end = decision_time[1:].append(
        pd.DatetimeIndex(
            [dates[-1] + pd.offsets.BDay(1) + pd.Timedelta(hours=16)]
        )
    )
    panel = pd.DataFrame(
        {
            "date": dates.date.astype(str),
            "ticker": "ACME",
            "decision_time": decision_time.astype(str),
            "text_available_at": text_available.astype(str),
            "revised_available_at": revised_available.astype(str),
            "return_start_at": return_start.astype(str),
            "return_end_at": return_end.astype(str),
            "text_signal": observed.round(6),
            "revised_text_signal": revised.round(6),
            "latent_regime": regime,
            "next_return": next_return.round(8),
        }
    )
    panel["close"] = (100 * (1 + panel["next_return"]).cumprod()).round(4)
    panel.to_csv(DATA / "market_text_panel.csv", index=False)


def build_transactions(rng: np.random.Generator) -> None:
    n = 1600
    timestamps = pd.date_range("2025-01-02", periods=n, freq="3h", tz="UTC")
    amount = np.exp(rng.normal(8.0, 1.25, n))
    cross_border = rng.binomial(1, 0.22, n)
    new_beneficiary = rng.binomial(1, 0.16, n)
    cash_equivalent = rng.binomial(1, 0.08, n)
    country_risk = rng.choice([0, 1, 2, 3], size=n, p=[0.48, 0.30, 0.17, 0.05])
    velocity = rng.poisson(2.2, n) + cross_border + new_beneficiary
    night = ((timestamps.hour < 6) | (timestamps.hour > 22)).astype(int)
    score = (
        0.000045 * amount
        + 0.95 * cross_border
        + 1.1 * new_beneficiary
        + 1.4 * cash_equivalent
        + 0.55 * country_risk
        + 0.18 * velocity
        + 0.65 * night
        + rng.normal(0, 0.75, n)
    )
    label = (score > 4.45).astype(int)
    frame = pd.DataFrame(
        {
            "transaction_id": [f"txn-{i:05d}" for i in range(n)],
            "timestamp": timestamps.astype(str),
            "customer_hash": [f"customer-{i % 173:03d}" for i in range(n)],
            "amount_usd": amount.round(2),
            "cross_border": cross_border,
            "new_beneficiary": new_beneficiary,
            "cash_equivalent": cash_equivalent,
            "country_risk": country_risk,
            "velocity_24h": velocity,
            "night_transaction": night,
            "investigation_label": label,
            "data_status": "fictional",
        }
    )
    frame.to_csv(DATA / "transactions.csv", index=False)


def build_contracts_and_clients() -> None:
    contracts = [
        {
            "contract_id": "facility-ACME-01",
            "ticker": "ACME",
            "text": (
                "The borrower enters a USD revolving facility of $450 million with a "
                "maturity of 4 years and a margin of 185 basis points. The net leverage "
                "ratio shall not exceed 3.50x and the interest coverage ratio shall be at "
                "least 2.25x. Financial covenants are tested quarterly."
            ),
            "observed_leverage": 3.18,
            "observed_interest_coverage": 2.61,
        },
        {
            "contract_id": "facility-NIMB-01",
            "ticker": "NIMB",
            "text": (
                "The borrower enters a USD term facility of $300 million with a maturity "
                "of 5 years and a spread of 210 basis points. The leverage ratio shall not "
                "exceed 3.00x and the interest coverage ratio shall be at least 2.75x."
            ),
            "observed_leverage": 3.14,
            "observed_interest_coverage": 2.82,
        },
        {
            "contract_id": "facility-ORBT-01",
            "ticker": "ORBT",
            "text": (
                "The borrower enters a EUR revolving facility of €220 million with a "
                "maturity of 3 years and a margin of 165 basis points. The leverage ratio "
                "shall not exceed 2.75x and the interest coverage ratio shall be at least 3.00x."
            ),
            "observed_leverage": 2.41,
            "observed_interest_coverage": 3.34,
        },
        {
            "contract_id": "facility-VELA-01",
            "ticker": "VELA",
            "text": (
                "The borrower enters a USD revolving facility of $600 million with a "
                "maturity of 4 years and a spread of 240 basis points. The leverage ratio "
                "shall not exceed 4.00x and the interest coverage ratio shall be at least 1.80x."
            ),
            "observed_leverage": 3.76,
            "observed_interest_coverage": 1.69,
        },
    ]
    write_jsonl(DATA / "contracts.jsonl", contracts)

    clients = [
        {
            "client_id": "client-conservative",
            "risk_tolerance": 2,
            "horizon_years": 4,
            "max_liquidity_days": 5,
            "complex_products_approved": False,
        },
        {
            "client_id": "client-balanced",
            "risk_tolerance": 3,
            "horizon_years": 8,
            "max_liquidity_days": 30,
            "complex_products_approved": False,
        },
        {
            "client_id": "client-sophisticated",
            "risk_tolerance": 5,
            "horizon_years": 15,
            "max_liquidity_days": 180,
            "complex_products_approved": True,
        },
    ]
    (DATA / "client_profiles.json").write_text(
        json.dumps(clients, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_preferences(rng: np.random.Generator) -> None:
    actions = ["answer", "clarify", "refuse", "escalate"]
    rows = []
    cases = [
        ("supported_summary", "answer", "refuse"),
        ("ambiguous_metric", "clarify", "answer"),
        ("future_document", "refuse", "answer"),
        ("material_trade", "escalate", "answer"),
        ("restricted_client", "refuse", "answer"),
    ]
    for i in range(250):
        case, chosen, rejected = cases[i % len(cases)]
        base = actions.index(chosen) - actions.index(rejected)
        rows.append(
            {
                "pair_id": f"pref-{i:04d}",
                "case_type": case,
                "chosen_action": chosen,
                "rejected_action": rejected,
                "chosen_reward": round(1.2 + rng.normal(0, 0.18), 5),
                "rejected_reward": round(-0.4 + rng.normal(0, 0.22), 5),
                "reference_margin": round(0.08 * base + rng.normal(0, 0.12), 5),
                "data_status": "fictional preference",
            }
        )
    pd.DataFrame(rows).to_csv(DATA / "preference_pairs.csv", index=False)


def build_infrastructure_and_governance(rng: np.random.Generator) -> None:
    rows = []
    for precision, quality, speed, memory in [
        ("fp32", 1.000, 1.0, 1.0),
        ("fp16", 0.998, 1.65, 0.58),
        ("int8", 0.991, 2.25, 0.34),
        ("int4", 0.972, 2.85, 0.21),
    ]:
        for batch in [1, 4, 16, 32]:
            service = 5.2 * speed * (batch**0.62)
            p95 = 1000 * (0.22 + 0.035 * batch) / speed
            cost = 3.4 * memory / (batch**0.35)
            rows.append(
                {
                    "configuration": f"{precision}-b{batch}",
                    "precision": precision,
                    "batch_size": batch,
                    "quality_index": quality,
                    "service_rate_rps": round(service, 4),
                    "p95_latency_ms": round(p95, 3),
                    "cost_per_1k_requests_usd": round(cost, 4),
                    "memory_index": memory,
                    "data_status": "illustrative benchmark",
                }
            )
    pd.DataFrame(rows).to_csv(DATA / "infrastructure_configs.csv", index=False)

    system_names = [
        "Earnings Research",
        "Credit Review",
        "AML Triage",
        "Client Service",
        "Market Surveillance",
        "Portfolio Research",
    ]
    inventory = []
    for i in range(36):
        impact = int(rng.integers(1, 6))
        autonomy = int(rng.integers(0, 5))
        exposure = int(rng.integers(1, 6))
        controls = int(rng.integers(1, 6))
        inherent = impact + autonomy + exposure
        residual = max(1, inherent - controls)
        inventory.append(
            {
                "system_id": f"sys-{i:03d}",
                "system_name": f"{system_names[i % len(system_names)]} {i // 6 + 1}",
                "impact": impact,
                "autonomy": autonomy,
                "external_exposure": exposure,
                "control_strength": controls,
                "inherent_risk": inherent,
                "residual_risk": residual,
                "owner_assigned": bool(i % 7),
                "last_validation_days": int(rng.integers(10, 420)),
                "data_status": "fictional inventory",
            }
        )
    pd.DataFrame(inventory).to_csv(DATA / "governance_inventory.csv", index=False)

    evidence = []
    systems = ["Lexical baseline", "Calibrated classifier", "Hybrid retrieval", "Governed agent"]
    tasks = ["sentiment", "retrieval", "policy compliance"]
    for i in range(28):
        system = systems[i % len(systems)]
        task = tasks[i % len(tasks)]
        n = int(rng.integers(180, 1400))
        quality = float(np.clip(0.66 + 0.045 * systems.index(system) + rng.normal(0, 0.025), 0, 1))
        se = np.sqrt(max(quality * (1 - quality), 1e-5) / n)
        evidence.append(
            {
                "experiment_id": f"exp-{i:03d}",
                "system": system,
                "task": task,
                "sample_size": n,
                "quality": round(quality, 5),
                "ci_low": round(max(0, quality - 1.96 * se), 5),
                "ci_high": round(min(1, quality + 1.96 * se), 5),
                "temporal_split": bool(i % 4),
                "point_in_time": bool(i % 3),
                "independent_review": bool(i % 5 == 0),
                "data_status": "illustrative evidence card",
            }
        )
    pd.DataFrame(evidence).to_csv(DATA / "evidence_matrix.csv", index=False)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    earnings = build_earnings(rng)
    build_filings(rng, earnings)
    build_market_panel(rng)
    build_transactions(rng)
    build_contracts_and_clients()
    build_preferences(rng)
    build_infrastructure_and_governance(rng)
    generated = sorted(
        path.name
        for path in DATA.iterdir()
        if path.is_file() and path.name not in {"README.md"}
    )
    print(f"Generated {len(generated)} deterministic data files with seed {SEED}:")
    for name in generated:
        print(f"  {name}")


if __name__ == "__main__":
    main()
