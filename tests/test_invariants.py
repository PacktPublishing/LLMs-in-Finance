from __future__ import annotations

import json
import math
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from finllm_lab import Claim, Document, Evidence, ToolCall, TradeIntent
from finllm_lab.documents import extract_contract_terms
from finllm_lab.governance import GuardedExecutor, PolicyGate, ToolSchema
from finllm_lab.market import (
    backtest_signal,
    herding_null_baseline,
    spectral_abscissa,
    spectral_radius,
)
from finllm_lab.metrics import (
    annualized_sharpe,
    proportion_interval,
    release_score,
    sharpe_standard_error,
)
from finllm_lab.rag import BM25Index, lexical_support, split_sentences
from finllm_lab.rl import categorical_kl
from finllm_lab.temporal import availability_audit, chronological_split
from finllm_lab.text import (
    NGramLanguageModel,
    multi_head_attention,
    scaled_dot_product_attention,
)

NOW = datetime(2026, 7, 1, 12, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


class LibraryInvariantTests(unittest.TestCase):
    def test_attention_preserves_leading_axes(self) -> None:
        rng = np.random.default_rng(1)
        q = rng.normal(size=(2, 3, 5, 4))
        k = rng.normal(size=(2, 3, 5, 4))
        v = rng.normal(size=(2, 3, 5, 6))
        context, weights = scaled_dot_product_attention(q, k, v)
        self.assertEqual(context.shape, (2, 3, 5, 6))
        self.assertEqual(weights.shape, (2, 3, 5, 5))
        self.assertTrue(np.allclose(weights.sum(axis=-1), 1.0))

    def test_attention_rejects_fully_masked_query(self) -> None:
        q = k = v = np.eye(3)
        mask = np.tril(np.ones((3, 3), dtype=bool))
        mask[1] = False
        with self.assertRaises(ValueError):
            scaled_dot_product_attention(q, k, v, mask)

    def test_multi_head_attention_concatenates_then_projects(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(size=(5, 8))
        projections = [
            (
                rng.normal(size=(8, 4)),
                rng.normal(size=(8, 4)),
                rng.normal(size=(8, 4)),
            )
            for _ in range(2)
        ]
        output, weights = multi_head_attention(x, projections, rng.normal(size=(8, 8)))
        self.assertEqual(output.shape, (5, 8))
        self.assertEqual(weights.shape, (2, 5, 5))

    def test_ngram_scoring_does_not_mutate_model(self) -> None:
        model = NGramLanguageModel().fit(["revenue grew"])
        before = set(model.next_counts)
        model.probability("unknown", ["never-seen"])
        self.assertEqual(before, set(model.next_counts))

    def test_sentence_split_preserves_vs_abbreviation(self) -> None:
        text = "Revenue was $1.5 million vs. $1.2 million previously. Margin rose."
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertIn("$1.2 million", sentences[0])

    def test_numeric_claim_support_normalizes_currency(self) -> None:
        evidence = Evidence(
            "ev-1",
            "doc-1",
            "Revenue was $1.5 million.",
            NOW,
            1.0,
        )
        self.assertTrue(
            lexical_support(Claim("c-1", "Revenue was 1.5 million."), [evidence])
        )

    def test_bm25_statistics_are_point_in_time(self) -> None:
        current = Document("current", "revenue grew", "fixture", NOW)
        future = Document(
            "future",
            "revenue revenue revenue declined",
            "fixture",
            NOW + timedelta(days=1),
        )
        index = BM25Index([current, future], decision_time=NOW)
        self.assertEqual([doc.document_id for doc in index.documents], ["current"])
        self.assertEqual(index.doc_frequency["declined"], 0)

    def test_bm25_metadata_filters(self) -> None:
        docs = [
            Document(
                "a",
                "revenue grew",
                "fixture",
                NOW,
                issuer="ACME",
                section="results",
                metadata={"period": "2025Q4"},
            ),
            Document(
                "b",
                "revenue grew",
                "fixture",
                NOW,
                issuer="NIMB",
                section="results",
                metadata={"period": "2025Q4"},
            ),
        ]
        result = BM25Index(docs, decision_time=NOW).search(
            "revenue",
            issuer="ACME",
            section="results",
            period="2025Q4",
        )
        self.assertEqual([item.document_id for item in result], ["a"])

    def test_availability_audit_fails_closed_on_missing_time(self) -> None:
        frame = pd.DataFrame({"decision": [NOW], "artifact": [pd.NaT]})
        audit = availability_audit(frame, "decision", ["artifact"]).iloc[0]
        self.assertEqual(audit["status"], "FAIL")
        self.assertEqual(audit["missing_timestamps"], 1)

    def test_chronological_split_keeps_timestamp_ties_together(self) -> None:
        timestamps = pd.to_datetime(
            [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
                "2026-02-01T00:00:00Z",
                "2026-03-01T00:00:00Z",
                "2026-04-01T00:00:00Z",
            ]
        )
        split = chronological_split(timestamps, 0.4, 0.2)
        windows = [
            set(timestamps.take(indices))
            for indices in (split.train, split.validation, split.test)
        ]
        self.assertFalse(windows[0] & windows[1])
        self.assertFalse(windows[1] & windows[2])

    def test_chronological_split_rejects_empty_windows(self) -> None:
        with self.assertRaises(ValueError):
            chronological_split([NOW, NOW, NOW], 0.6, 0.2)

    def test_policy_checks_completeness_before_authority(self) -> None:
        gate = PolicyGate(
            {},
            schemas={
                "snapshot": ToolSchema(
                    "snapshot",
                    required_arguments=frozenset({"as_of"}),
                )
            },
        )
        decision = gate.decide(ToolCall("r", "unknown", "snapshot", {}, NOW))
        self.assertEqual(decision.status, "reject")
        self.assertIn("requires argument", decision.reason)

    def test_policy_gate_never_raises(self) -> None:
        gate = PolicyGate({"agent": {"tool"}})
        malformed = ToolCall("r", "agent", "tool", None, NOW)  # type: ignore[arg-type]
        decision = gate.decide(malformed)
        self.assertEqual(decision.status, "reject")

    def test_executor_idempotency_and_conflict_detection(self) -> None:
        counter = {"calls": 0}

        def tool(value: int) -> int:
            counter["calls"] += 1
            return value

        executor = GuardedExecutor({"tool": tool}, PolicyGate({"agent": {"tool"}}))
        call = ToolCall("same", "agent", "tool", {"value": 1}, NOW)
        executor.execute(call)
        executor.execute(call)
        conflict, _ = executor.execute(
            ToolCall("same", "agent", "tool", {"value": 2}, NOW)
        )
        self.assertEqual(counter["calls"], 1)
        self.assertEqual(conflict.status, "reject")

    def test_audit_chain_has_genesis_and_external_head(self) -> None:
        executor = GuardedExecutor(
            {"tool": lambda: "ok"},
            PolicyGate({"agent": {"tool"}}),
        )
        executor.execute(ToolCall("r", "agent", "tool", {}, NOW))
        report = executor.verify_chain(expected_head=executor.head_hash)
        self.assertTrue(report["anchored_to_genesis"])
        self.assertTrue(report["links_intact"])
        self.assertTrue(report["head_matches"])

    def test_constant_return_sharpe_is_nan(self) -> None:
        self.assertTrue(math.isnan(annualized_sharpe(np.ones(30) * 0.01)))

    def test_sharpe_standard_error_is_finite(self) -> None:
        self.assertGreater(sharpe_standard_error(1.0, 252), 0)

    def test_categorical_kl_is_batch_invariant(self) -> None:
        p = np.array([[0.7, 0.3]])
        q = np.array([[0.5, 0.5]])
        one = categorical_kl(p, q)
        repeated = categorical_kl(np.repeat(p, 5, axis=0), np.repeat(q, 5, axis=0))
        self.assertAlmostEqual(one, repeated)

    def test_wilson_interval_stays_in_unit_interval(self) -> None:
        point, low, high = proportion_interval(1, 2)
        self.assertEqual(point, 0.5)
        self.assertGreaterEqual(low, 0)
        self.assertLessEqual(high, 1)

    def test_release_gate_requires_every_metric(self) -> None:
        with self.assertRaises(KeyError):
            release_score({}, {"recall": ("min", 0.9)})

    def test_contract_amount_preserves_currency_unit(self) -> None:
        terms = extract_contract_terms(
            "The borrower enters a EUR facility of €220 million with a maturity "
            "of 3 years and a margin of 165 basis points. The leverage ratio shall "
            "not exceed 2.75x and the interest coverage ratio shall be at least 3.00x."
        )
        self.assertEqual(terms["currency"], "EUR")
        self.assertEqual(terms["facility_amount_mn"], 220.0)
        self.assertNotIn("currency_conflict", terms)

    def test_contract_currency_conflict_is_explicit(self) -> None:
        terms = extract_contract_terms("The borrower enters a EUR facility of $220 million.")
        self.assertIn("currency_conflict", terms)

    def test_backtest_reports_observation_count(self) -> None:
        frame = pd.DataFrame({"signal": [1, -1], "next_return": [0.01, -0.01]})
        result = backtest_signal(frame, "signal", cost_bps=0)
        self.assertEqual(result["observations"], 2)

    def test_continuous_and_discrete_stability_metrics_differ(self) -> None:
        matrix = np.array([[0.5, 0.0], [0.0, -1.2]])
        self.assertAlmostEqual(spectral_abscissa(matrix), 0.5)
        self.assertAlmostEqual(spectral_radius(matrix), 1.2)

    def test_herding_null_baseline_is_positive(self) -> None:
        self.assertGreater(herding_null_baseline(60), 0)

    def test_trade_intent_rejects_nonpositive_limit(self) -> None:
        with self.assertRaises(ValueError):
            TradeIntent("ACME", "BUY", 1, 0.0, "test", ("ev",), 0.8).validate()


class GeneratedFixtureTests(unittest.TestCase):
    def test_earnings_fixture_dimensions_and_margin_units(self) -> None:
        calls = pd.read_csv(ROOT / "data" / "earnings_calls.csv")
        self.assertEqual(len(calls), 84)
        self.assertEqual(calls["ticker"].nunique(), 4)
        for _, issuer in calls.groupby("ticker"):
            observed_change = issuer["operating_margin_pct"].diff().iloc[1:]
            declared_change = issuer["margin_change_bps"].iloc[1:] / 100
            self.assertTrue(np.allclose(observed_change, declared_change, atol=0.002))

    def test_contract_fixture_has_no_currency_conflict(self) -> None:
        contracts = [
            json.loads(line)
            for line in (ROOT / "data" / "contracts.jsonl").read_text().splitlines()
        ]
        self.assertTrue(
            all(
                "currency_conflict" not in extract_contract_terms(item["text"])
                for item in contracts
            )
        )

    def test_market_returns_begin_after_decision(self) -> None:
        panel = pd.read_csv(
            ROOT / "data" / "market_text_panel.csv",
            parse_dates=["decision_time", "return_start_at", "return_end_at"],
        )
        self.assertTrue((panel["return_start_at"] > panel["decision_time"]).all())
        self.assertTrue((panel["return_end_at"] > panel["return_start_at"]).all())


if __name__ == "__main__":
    unittest.main()
