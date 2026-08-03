# Release checklist

## Automated gate

- [ ] Install Python 3.12 and the exact core/dev requirement files.
- [ ] Run `make release`.
- [ ] Confirm lint, tests, 18 notebook executions, previews, manifest checks,
      safe-archive checks, and fresh-extraction reproduction all pass.
- [ ] Confirm the generated archive contains one `LLMs-in-Finance/` root and no
      nested ZIP, symbolic link, cache, credential, or live-execution endpoint.
- [ ] Record the ZIP SHA-256 from `dist/*.sha256`.

## Content gate

- [ ] Confirm every material claim is supported by fictional or cited evidence.
- [ ] Confirm every temporal path fails closed on missing or future timestamps.
- [ ] Confirm thresholds are selected outside untouched test windows.
- [ ] Confirm uncertainty intervals accompany finite-sample release metrics.
- [ ] Confirm optional model/SEC adapters remain `SKIPPED_BY_DESIGN` in committed
      outputs.
- [ ] Confirm data and model cards match the actual fixtures and executed state.

## GitHub gate

- [ ] Confirm `LICENSE` matches Packt's official MIT license.
- [ ] Confirm all README and per-notebook Colab links use
      `PacktPublishing/LLMs-in-Finance` on `main`.
- [ ] Confirm the `notebooks` and `codeql` workflows are required on `main`.
- [ ] Upload `assets/github_social_preview.png` as the repository social image.
- [ ] Confirm secret scanning, Dependabot, and private vulnerability reporting.
- [ ] Tag `notebook-suite-v1.1.0` only after protected `main` passes.
- [ ] Confirm the release workflow publishes the deterministic ZIP and checksum.

Follow [`docs/PUBLISHING.md`](docs/PUBLISHING.md) for the exact handoff.
