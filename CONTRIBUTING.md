# Contributing

Thank you for improving the companion labs.

## Development contract

1. Support Python 3.11 and 3.12.
2. Keep every default notebook credential-free, offline, and CPU-friendly.
3. Use only synthetic, public, or properly licensed data.
4. Preserve point-in-time availability and post-decision return windows.
5. Separate model proposals from calculations, policy gates, execution, and
   human approval.
6. Report uncertainty and retain honest negative or `HOLD` outcomes.
7. Add tests for every material control, metric, parser, or adapter.
8. Never commit credentials, client data, proprietary datasets, model caches,
   or live-execution endpoints.

Optional network or model integrations must be disabled by default, bounded to
documented hosts or model identifiers, provenance-recorded, and unable to
bypass deterministic controls.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
```

Run the complete tracked build:

```bash
make lint
make reproduce
```

Before proposing a release:

```bash
make release
```

This creates and verifies a deterministic archive in `dist/`; that directory is
ignored and must not be committed.

## Notebook workflow

Notebook source lives in `notebook_sources/`. Generated `.ipynb` files live in
`notebooks/`; do not edit both independently.

1. Edit the percent-format Python source.
2. Run `python scripts/build_notebooks.py`.
3. Run `python scripts/execute_notebooks.py --in-place`.
4. Run `python scripts/export_readme_previews.py`.
5. Update tests, documentation, cards, and changelog as needed.
6. Run `python scripts/write_release_manifest.py` last.
7. Run `python scripts/verify_release.py`.

The builder inserts the standard Colab launch and bootstrap cells. Do not copy
that bootstrap into individual source files.

## Pull requests

Keep pull requests focused and describe:

- the learning or research objective;
- the temporal and data contract;
- changed outputs and failure cases;
- tests and uncertainty estimates;
- any new dependency, external host, or model artifact; and
- the human decision that remains outside model authority.

Use private vulnerability reporting for security defects; see
[`SECURITY.md`](SECURITY.md).
