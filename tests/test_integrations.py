from __future__ import annotations

import io
import json
import unittest
from copy import deepcopy
from datetime import datetime, timezone

from finllm_lab.integrations import (
    ExternalAccessDisabled,
    SecSnapshot,
    company_fact_observations,
    fetch_sec_company_facts,
    generate_with_transformers,
)

NOW = datetime(2026, 7, 30, 12, tzinfo=timezone.utc)


class _Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False

    def geturl(self):
        return "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"


class OptionalIntegrationTests(unittest.TestCase):
    def test_sec_access_is_disabled_by_default(self) -> None:
        with self.assertRaises(ExternalAccessDisabled):
            fetch_sec_company_facts(
                "320193",
                user_agent="AIFI research@example.org",
            )

    def test_sec_requires_declared_contact(self) -> None:
        with self.assertRaises(ValueError):
            fetch_sec_company_facts(
                "320193",
                user_agent="anonymous-script",
                enabled=True,
                opener=lambda *_args, **_kwargs: _Response(b"{}"),
            )

    def test_sec_snapshot_records_hash_and_fixed_host(self) -> None:
        payload = {"cik": 320193, "facts": {}}
        raw = json.dumps(payload).encode()
        seen = {}

        def opener(request, timeout):
            seen["url"] = request.full_url
            seen["agent"] = request.headers["User-agent"]
            seen["timeout"] = timeout
            return _Response(raw)

        snapshot = fetch_sec_company_facts(
            "CIK320193",
            user_agent="AIFI research@example.org",
            enabled=True,
            opener=opener,
            now=lambda: NOW,
        )
        self.assertEqual(snapshot.cik, "0000320193")
        self.assertEqual(
            seen["url"],
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        )
        self.assertEqual(seen["agent"], "AIFI research@example.org")
        self.assertEqual(seen["timeout"], 20.0)
        self.assertEqual(len(snapshot.raw_sha256), 64)
        self.assertEqual(snapshot.retrieved_at, NOW)

    def test_sec_rejects_a_cross_host_redirect(self) -> None:
        class RedirectedResponse(_Response):
            def geturl(self):
                return "https://example.org/untrusted.json"

        with self.assertRaisesRegex(RuntimeError, "redirected outside"):
            fetch_sec_company_facts(
                "320193",
                user_agent="AIFI research@example.org",
                enabled=True,
                opener=lambda *_args, **_kwargs: RedirectedResponse(b'{"facts": {}}'),
            )

    def test_company_facts_are_point_in_time_and_amendment_aware(self) -> None:
        original_payload = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "accn": "0001",
                                    "end": "2025-12-31",
                                    "filed": "2026-01-02",
                                    "form": "10-K",
                                    "val": 100,
                                    "fy": 2025,
                                    "fp": "FY",
                                },
                                {
                                    "accn": "future",
                                    "end": "2026-03-31",
                                    "filed": "2026-05-01",
                                    "form": "10-Q",
                                    "val": 30,
                                    "fy": 2026,
                                    "fp": "Q1",
                                },
                            ]
                        }
                    }
                }
            }
        }
        amended_payload = deepcopy(original_payload)
        amended_payload["facts"]["us-gaap"]["Revenues"]["units"]["USD"].append(
            {
                "accn": "0002",
                "end": "2025-12-31",
                "filed": "2026-01-04",
                "form": "10-K/A",
                "val": 110,
                "fy": 2025,
                "fp": "FY",
            }
        )
        original_snapshot = SecSnapshot(
            cik="0000320193",
            source_url="https://data.sec.gov/example",
            retrieved_at=datetime(2026, 1, 3, 12, tzinfo=timezone.utc),
            raw_sha256="a" * 64,
            payload=original_payload,
        )
        amended_snapshot = SecSnapshot(
            cik="0000320193",
            source_url="https://data.sec.gov/example",
            retrieved_at=datetime(2026, 1, 5, 12, tzinfo=timezone.utc),
            raw_sha256="b" * 64,
            payload=amended_payload,
        )
        before_amendment = company_fact_observations(
            original_snapshot,
            taxonomy="us-gaap",
            concept="Revenues",
            unit="USD",
            decision_time=datetime(2026, 1, 4, tzinfo=timezone.utc),
        )
        after_amendment = company_fact_observations(
            amended_snapshot,
            taxonomy="us-gaap",
            concept="Revenues",
            unit="USD",
            decision_time=datetime(2026, 1, 6, tzinfo=timezone.utc),
        )
        self.assertEqual([item["value"] for item in before_amendment], [100])
        self.assertEqual([item["value"] for item in after_amendment], [110])
        self.assertEqual(
            after_amendment[0]["available_at"],
            datetime(2026, 1, 5, 6, tzinfo=timezone.utc),
        )

    def test_company_facts_reject_a_future_snapshot(self) -> None:
        snapshot = SecSnapshot(
            cik="0000320193",
            source_url="https://data.sec.gov/example",
            retrieved_at=NOW,
            raw_sha256="a" * 64,
            payload={"facts": {}},
        )
        with self.assertRaisesRegex(ValueError, "retrieved after decision_time"):
            company_fact_observations(
                snapshot,
                taxonomy="us-gaap",
                concept="Revenues",
                unit="USD",
                decision_time=datetime(2026, 7, 29, tzinfo=timezone.utc),
            )

    def test_model_access_is_disabled_by_default(self) -> None:
        with self.assertRaises(ExternalAccessDisabled):
            generate_with_transformers("Summarize this evidence.")

    def test_model_adapter_records_resolved_revision(self) -> None:
        class Config:
            _commit_hash = "resolved-commit"

        class Model:
            config = Config()

        class Generator:
            model = Model()

            def __call__(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return [{"generated_text": [{"role": "assistant", "content": "Grounded draft."}]}]

        seen = {}

        def factory(task, **kwargs):
            seen["task"] = task
            seen.update(kwargs)
            return Generator()

        record = generate_with_transformers(
            "Summarize only the supplied evidence.",
            model_id="test/model",
            revision="requested-tag",
            max_new_tokens=32,
            enabled=True,
            pipeline_factory=factory,
        )
        self.assertEqual(seen["task"], "text-generation")
        self.assertFalse(seen["trust_remote_code"])
        self.assertEqual(record.response, "Grounded draft.")
        self.assertEqual(record.resolved_revision, "resolved-commit")
        self.assertTrue(record.deterministic_decoding)
        self.assertEqual(len(record.prompt_sha256), 64)


if __name__ == "__main__":
    unittest.main()
