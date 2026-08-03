# Security policy

## Supported release

Security fixes are applied to the latest notebook-suite release on `main`.
Educational outputs from older tags remain immutable reproducibility records.

## Reporting a vulnerability

Use the repository's **Security → Report a vulnerability** flow:

<https://github.com/PacktPublishing/LLMs-in-Finance/security/advisories/new>

Do not disclose an unpatched vulnerability in a public issue. Include the
affected commit, a minimal reproduction, likely impact, and any suggested
mitigation. Never attach credentials, personal data, proprietary financial data,
or live-broker details.

## Security boundary

The default suite:

- requires no credentials or external network access;
- has no live-broker or order-routing integration;
- executes deterministic local teaching workflows; and
- treats model output as an untrusted proposal.

Optional model and SEC adapters are disabled by default. Enabling them expands
the local threat surface and requires the user to review dependencies, model
artifacts, data licensing, network policy, and institutional controls.
