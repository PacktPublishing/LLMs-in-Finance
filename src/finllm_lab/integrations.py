"""Fail-closed adapters for optional model inference and public SEC data.

Nothing in this module performs network access unless the caller passes
``enabled=True``. The default notebook release remains credential-free,
offline, deterministic, and based on fictional fixtures.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib import metadata
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SEC_COMPANYFACTS_BASE = "https://data.sec.gov/api/xbrl/companyfacts"
_EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


class ExternalAccessDisabled(RuntimeError):
    """Raised when an optional external operation was not explicitly enabled."""


@dataclass(frozen=True)
class GenerationRecord:
    """Auditable metadata and text returned by an optional model backend."""

    backend: str
    backend_version: str
    model_id: str
    requested_revision: str
    resolved_revision: str
    prompt_sha256: str
    response: str
    max_new_tokens: int
    deterministic_decoding: bool


@dataclass(frozen=True)
class SecSnapshot:
    """Raw SEC company-facts response with retrieval provenance."""

    cik: str
    source_url: str
    retrieved_at: datetime
    raw_sha256: str
    payload: Mapping[str, Any]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalized_cik(cik: str | int) -> str:
    text = str(cik).strip()
    if text.upper().startswith("CIK"):
        text = text[3:]
    if not text.isdigit() or len(text) > 10:
        raise ValueError("CIK must contain at most ten decimal digits")
    return text.zfill(10)


def _validate_sec_user_agent(user_agent: str) -> str:
    value = user_agent.strip()
    if len(value) < 8 or not _EMAIL_PATTERN.search(value):
        raise ValueError(
            "SEC_USER_AGENT must identify an organization or person and include "
            "a monitored contact email"
        )
    return value


def fetch_sec_company_facts(
    cik: str | int,
    *,
    user_agent: str,
    enabled: bool = False,
    timeout_seconds: float = 20.0,
    max_response_bytes: int = 25_000_000,
    opener: Callable[..., Any] = urlopen,
    now: Callable[[], datetime] | None = None,
) -> SecSnapshot:
    """Fetch one SEC company-facts payload under an explicit opt-in gate.

    The endpoint is fixed to ``data.sec.gov``. A declared user agent with a
    contact email is mandatory, and oversized or malformed responses fail
    closed.
    """

    if not enabled:
        raise ExternalAccessDisabled(
            "SEC access is disabled; pass enabled=True only for an intentional public-data run"
        )
    normalized_cik = _normalized_cik(cik)
    declared_agent = _validate_sec_user_agent(user_agent)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_response_bytes <= 0:
        raise ValueError("max_response_bytes must be positive")

    url = f"{SEC_COMPANYFACTS_BASE}/CIK{normalized_cik}.json"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": declared_agent,
        },
        method="GET",
    )
    with opener(request, timeout=timeout_seconds) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise RuntimeError(f"SEC request failed with HTTP status {status}")
        final_url = response.geturl() if hasattr(response, "geturl") else url
        parsed_final_url = urlparse(final_url)
        if (
            parsed_final_url.scheme != "https"
            or parsed_final_url.hostname != "data.sec.gov"
        ):
            raise RuntimeError("SEC request redirected outside data.sec.gov")
        raw = response.read(max_response_bytes + 1)
    if len(raw) > max_response_bytes:
        raise ValueError("SEC response exceeds max_response_bytes")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SEC response is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("facts"), dict):
        raise ValueError("SEC response is missing the facts mapping")

    clock = now or (lambda: datetime.now(timezone.utc))
    retrieved_at = clock()
    if retrieved_at.tzinfo is None:
        raise ValueError("retrieved_at must be timezone-aware")
    return SecSnapshot(
        cik=normalized_cik,
        source_url=url,
        retrieved_at=retrieved_at.astimezone(timezone.utc),
        raw_sha256=_sha256_bytes(raw),
        payload=payload,
    )


def company_fact_observations(
    snapshot: SecSnapshot,
    *,
    taxonomy: str,
    concept: str,
    unit: str,
    decision_time: datetime,
    allowed_forms: Sequence[str] = ("10-K", "10-Q", "10-K/A", "10-Q/A"),
) -> list[dict[str, Any]]:
    """Return the latest admissible observation for each reporting period.

    Company Facts exposes filing dates but not an intraday acceptance timestamp
    for every fact. To avoid look-ahead, the adapter assigns a conservative
    availability time of 06:00 UTC on the day after ``filed``. Amendments replace
    earlier values only when the amendment itself is admissible at decision time.
    The raw snapshot must itself have been retrieved no later than the decision
    time; historical evaluation therefore requires archived snapshots.
    """

    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    if snapshot.retrieved_at.tzinfo is None:
        raise ValueError("snapshot retrieved_at must be timezone-aware")
    cutoff = decision_time.astimezone(timezone.utc)
    if snapshot.retrieved_at.astimezone(timezone.utc) > cutoff:
        raise ValueError("SEC snapshot was retrieved after decision_time")
    try:
        entries = snapshot.payload["facts"][taxonomy][concept]["units"][unit]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"Missing SEC fact {taxonomy}.{concept} in unit {unit}") from exc
    if not isinstance(entries, list):
        raise ValueError("SEC fact unit payload must be a list")

    admissible: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("SEC fact observation must be an object")
        required = {"accn", "end", "filed", "form", "val"}
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(f"SEC fact observation is missing: {', '.join(missing)}")
        if item["form"] not in allowed_forms:
            continue
        filed = datetime.strptime(item["filed"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        available_at = filed + timedelta(days=1, hours=6)
        if available_at > cutoff:
            continue
        record = {
            "cik": snapshot.cik,
            "taxonomy": taxonomy,
            "concept": concept,
            "unit": unit,
            "value": item["val"],
            "period_end": item["end"],
            "fiscal_year": str(item.get("fy", "")),
            "fiscal_period": str(item.get("fp", "")),
            "form": item["form"],
            "accession": item["accn"],
            "filed": item["filed"],
            "available_at": available_at,
            "snapshot_sha256": snapshot.raw_sha256,
            "retrieved_at": snapshot.retrieved_at,
            "source_url": snapshot.source_url,
        }
        key = (
            record["period_end"],
            record["fiscal_year"],
            record["fiscal_period"],
        )
        previous = admissible.get(key)
        if previous is None or (
            record["available_at"],
            record["accession"],
        ) > (
            previous["available_at"],
            previous["accession"],
        ):
            admissible[key] = record
    return sorted(
        admissible.values(),
        key=lambda item: (
            item["period_end"],
            item["available_at"],
            item["accession"],
        ),
    )


def _generated_text(output: Any) -> str:
    if not isinstance(output, list) or not output:
        raise ValueError("Transformers pipeline returned an unexpected result")
    item = output[0]
    if not isinstance(item, dict) or "generated_text" not in item:
        raise ValueError("Transformers pipeline result is missing generated_text")
    generated = item["generated_text"]
    if isinstance(generated, str):
        return generated.strip()
    if isinstance(generated, list) and generated:
        last = generated[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str):
            return last["content"].strip()
    raise ValueError("Transformers pipeline generated_text has an unsupported shape")


def generate_with_transformers(
    prompt: str,
    *,
    model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
    revision: str = "main",
    max_new_tokens: int = 128,
    enabled: bool = False,
    pipeline_factory: Callable[..., Any] | None = None,
) -> GenerationRecord:
    """Run deterministic local Transformers inference under an explicit gate."""

    if not enabled:
        raise ExternalAccessDisabled(
            "Model loading is disabled; pass enabled=True only for an intentional model run"
        )
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not model_id.strip() or not revision.strip():
        raise ValueError("model_id and revision must not be empty")
    if not 1 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be between 1 and 512")

    if pipeline_factory is None:
        try:
            from transformers import pipeline as transformers_pipeline
        except ImportError as exc:
            raise RuntimeError(
                'Install the optional model stack with: pip install -e ".[extensions]"'
            ) from exc
        pipeline_factory = transformers_pipeline

    generator = pipeline_factory(
        "text-generation",
        model=model_id,
        revision=revision,
        trust_remote_code=False,
        device=-1,
    )
    output = generator(
        [{"role": "user", "content": prompt}],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    response = _generated_text(output)
    model = getattr(generator, "model", None)
    config = getattr(model, "config", None)
    resolved = getattr(config, "_commit_hash", None) or revision
    try:
        backend_version = metadata.version("transformers")
    except metadata.PackageNotFoundError:
        backend_version = "injected-test-backend"
    return GenerationRecord(
        backend="transformers",
        backend_version=backend_version,
        model_id=model_id,
        requested_revision=revision,
        resolved_revision=str(resolved),
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        response=response,
        max_new_tokens=max_new_tokens,
        deterministic_decoding=True,
    )
