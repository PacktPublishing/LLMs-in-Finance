from __future__ import annotations

import unittest
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from finllm_lab import Claim, Document, EvidenceBundle, ToolCall, TradeIntent, canonical_hash
from finllm_lab.governance import GuardedExecutor, PolicyGate
from finllm_lab.rag import BM25Index, faithfulness_score
from finllm_lab.rl import grpo_advantages
from finllm_lab.temporal import TemporalLeakageError, assert_documents_available
from finllm_lab.text import scaled_dot_product_attention, stable_softmax

NOW = datetime(2026, 7, 1, tzinfo=timezone.utc)


class CoreContractTests(unittest.TestCase):
    def test_stable_softmax_and_attention(self) -> None:
        probs = stable_softmax(np.array([1000.0, 1001.0, 1002.0]))
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue(np.isclose(probs.sum(), 1.0))
        q = np.eye(3)
        context, weights = scaled_dot_product_attention(q, q, q)
        self.assertEqual(context.shape, (3, 3))
        self.assertTrue(np.allclose(weights.sum(axis=1), 1.0))

    def test_temporal_checks_fail_closed(self) -> None:
        valid = Document("d1", "Revenue rose 8 percent.", "fixture", NOW)
        future = Document(
            "d2",
            "Revenue rose 12 percent.",
            "fixture",
            datetime(2026, 8, 1, tzinfo=timezone.utc),
        )
        assert_documents_available([valid], NOW)
        with self.assertRaises(TemporalLeakageError):
            assert_documents_available([future], NOW)

    def test_retrieval_and_faithfulness(self) -> None:
        doc = Document("d1", "Revenue increased by 8 percent in fiscal 2025.", "fixture", NOW)
        evidence = tuple(BM25Index([doc]).search("How did revenue change?", k=1, decision_time=NOW))
        claim = Claim("c1", "Revenue increased by 8 percent in fiscal 2025.")
        self.assertEqual(evidence[0].document_id, "d1")
        self.assertEqual(faithfulness_score([EvidenceBundle(claim, evidence)]), 1.0)

    def test_policy_gate_and_idempotency(self) -> None:
        counter = {"calls": 0}

        def tool(value: int) -> dict:
            counter["calls"] += 1
            return {"value": value}

        executor = GuardedExecutor({"tool": tool}, PolicyGate({"agent": {"tool"}}))
        call = ToolCall("r1", "agent", "tool", {"value": 3}, NOW)
        first = executor.execute(call)
        second = executor.execute(call)
        self.assertEqual(first[0].status, "allow")
        self.assertEqual(second[1], {"value": 3})
        self.assertEqual(counter["calls"], 1)
        self.assertEqual(
            executor.audit[1].parent_hash, canonical_hash(asdict(executor.audit[0]))
        )

    def test_trade_intent_validation(self) -> None:
        valid = TradeIntent("ACME", "BUY", 10, 101.0, "Test", ("ev-1",), 0.8)
        valid.validate()
        with self.assertRaises(ValueError):
            TradeIntent("ACME", "HOLD", 10, None, "Invalid", tuple(), 0.5).validate()

    def test_grpo_constant_group_is_zero(self) -> None:
        rewards = np.array([[1.0, 2.0, 3.0], [0.4, 0.4, 0.4]])
        advantages = grpo_advantages(rewards)
        self.assertTrue(np.allclose(advantages[0].mean(), 0.0))
        self.assertTrue(np.allclose(advantages[1], 0.0))


if __name__ == "__main__":
    unittest.main()
