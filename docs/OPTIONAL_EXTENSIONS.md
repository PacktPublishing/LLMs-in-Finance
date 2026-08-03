# Optional model and public-data extensions

The default repository is deliberately offline, fictional, deterministic, and
credential-free. Notebook 17 adds explicit bridges to external systems without
making them part of the release benchmark.

## Local open-weight model

Install the optional stack:

```bash
python -m pip install -r requirements-extensions.txt
python -m pip install --no-deps -e .
```

Then launch Jupyter with the model gate enabled:

```bash
FINLLM_RUN_REAL_MODEL=1 jupyter lab
```

The example model is
[`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).
Its upstream card identifies it as an approximately 0.4-billion-parameter,
Apache-2.0 model and warns that output can be inaccurate or biased.

For a reproducible experiment, set `FINLLM_MODEL_REVISION` to an immutable
model commit. Notebook 17 records the requested revision, the backend-resolved
revision, software version, prompt hash, decoding configuration, and response.
Greedy decoding is used, `trust_remote_code` is disabled, and automatic release
is always false.

The integration card in
[`model_cards/optional_smollm2_360m.md`](../model_cards/optional_smollm2_360m.md)
contains no invented benchmark result.

## SEC EDGAR company facts

The [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
provide public JSON data without an authentication key. Automated access must
follow the SEC's fair-access policy and declare a user agent with a monitored
contact email.

```bash
export FINLLM_RUN_SEC=1
export SEC_USER_AGENT="Your Organization research@example.org"
jupyter lab
```

The adapter:

- fixes the destination host to `data.sec.gov`;
- limits request time and response size;
- retains a SHA-256 hash of the raw response;
- records retrieval time, source URL, accession, form, and filing date;
- rejects any snapshot retrieved after the analytical decision time;
- assigns a conservative 06:00 UTC next-day availability timestamp; and
- applies amendments only after the amendment becomes admissible.

The default release executes the same parsing logic against a synthetic,
SEC-shaped payload. Its amendment example uses separate archived snapshots
before and after the amendment. The release does not cache or ship a
real-company response.

## What enabling an adapter does not do

Enabling a model or data source does not:

- validate financial accuracy;
- prove that the source is point-in-time complete;
- create an investment recommendation;
- authorize a tool call or transaction;
- turn the educational controls into production controls; or
- make a mutable external dependency reproducible.

A publishable extension needs an immutable snapshot, licensing review, a
chronological benchmark, finite-sample uncertainty, adversarial tests, a
completed model/data card, and accountable human approval.
