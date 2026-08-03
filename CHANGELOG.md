# Changelog

All notable changes to the notebook suite are documented here. The book's
canonical companion release remains the immutable `B32413-v1.0.0`; notebook
suite versions are independent additive releases.

## 1.1.0 — 2026-07-30

### GitHub and Colab

- Bound the suite to the official
  `PacktPublishing/LLMs-in-Finance` repository and `main` branch.
- Added CI and CodeQL badges, per-notebook Colab launch/bootstrap cells,
  role-based learning paths, selected-output previews, an animated tour, and a
  GitHub social-preview image.
- Added issue forms, a pull-request checklist, security policy, Dependabot,
  release automation, and an exact publication guide.

### Optional external bridge

- Added notebook 17 with an explicitly gated local Transformers model and SEC
  EDGAR Company Facts adapter.
- Added immutable-run metadata, prompt hashing, deterministic decoding,
  fixed-host and response-size controls, raw SEC hashing, conservative
  availability timestamps, archived-snapshot timing gates, and amendment-aware
  point-in-time filtering.
- Added an honest integration card that reports no unexecuted model benchmark.

### Reproducibility and packaging

- Corrected the supported Python version to 3.11+ to match the exact release
  dependencies.
- Replaced the provisional local license with the official repository's MIT
  license.
- Added linting, Python 3.11/3.12 CI, deterministic preview generation,
  deterministic ZIP creation, safe-member checks, tag/version matching, and a
  byte-for-byte clean-build verification from the packaged archive.

## 1.0.1 — 2026-07-28

### Integrity

- Rebuilt from the verified v1.0.0 notebook-suite archive in an isolated tree.
- Merged the ten independently recovered `finllm_lab` modules and recorded
  the incoming hashes in `audit_logs/REBUILD_PROVENANCE.md`.
- Excluded the reported concurrently rewritten tree and its unsupported model
  card from every source, fixture, notebook, and release artifact.
- Expanded the release manifest to hash the executed notebooks, fixtures, data
  cards, source modules, tests, workflows, and documentation.

### Correctness

- Made BM25 corpus statistics point-in-time, temporal audits fail closed, and
  chronological splits stable and timestamp-boundary safe.
- Hardened tool-schema validation, request-ID idempotency, audit-chain genesis
  anchoring, and external-head verification.
- Corrected multi-head attention, degenerate Sharpe handling, batch-invariant
  categorical KL, sentence splitting, and numeric text normalization.
- Corrected the basis-point fixture conversion, contract currency consistency,
  and explicit post-decision return windows.
- Removed label-revealing evaluation shortcuts and selected operational
  thresholds only on training or validation data.
- Added finite-sample uncertainty intervals and honest `HOLD` outcomes where a
  control does not meet its release criterion.

### Verification

- Expanded the credential-free regression suite to 35 tests.
- Rebuilt and executed all 17 notebooks with deterministic fictional data.
- Added clean-build CI, exact dependency pins, and end-to-end release
  verification.

## 1.0.0 — 2026-07-28

- Initial 17-notebook educational companion suite.
