# Learning paths

The repository has 17 chapter-aligned core labs and one optional external-system
bridge. Use the shortest path that matches your objective; the notebook numbers
remain the canonical full-course order.

## Fifteen-minute tour

No installation is required.

1. Read the architecture and release contract in notebook 00.
2. Inspect the point-in-time retrieval outputs in notebook 04.
3. Jump to the final scorecard in notebook 16.

This path is for reviewers, editors, and readers deciding whether to run the
suite.

## Two-hour fast path

| Time | Notebook | Focus |
|---:|---|---|
| 15 min | 00 · Start here | evidence → proposal → control → review |
| 35 min | 04 · Point-in-time RAG | admissible corpus, retrieval, faithfulness |
| 30 min | 06 · Hardened tools | schemas, authority, idempotency, audit chain |
| 40 min | 16 · Capstone | integrated memo and release scorecard |

Run the existing cells and study the deliberate failure cases. Leave the
extension exercises for a second session.

## One-day practitioner path

Complete notebooks 00, 01, 02, 04, 05, 06, 11, and 16. This path adds model
mechanics, evaluation, multi-agent orchestration, and governance to the
two-hour sequence.

## Full chapter path

Work through notebooks 00–16 in order. Each lab assumes only concepts already
introduced in the book or earlier notebooks. A full pass takes approximately
9–11 hours before optional exercises.

## Role-specific paths

### Quantitative researcher

01 → 02 → 03 → 04 → 09 → 12 → 13 → 15

Focus on evaluation design, temporal splits, uncertainty, preference
optimization, and leakage-aware backtests.

### Risk, validation, or governance

00 → 04 → 05 → 06 → 07 → 08 → 11 → 12 → 15 → 16

Focus on evidence admissibility, deterministic controls, group metrics,
release holds, and ownership.

### ML or platform engineer

02 → 03 → 04 → 05 → 06 → 09 → 10 → 14 → 16 → 17

Focus on attention, calibration, retrieval, orchestration, typed tools,
capacity, stability, and optional external adapters.

### Instructor

Use notebooks 00, 02, 04, 06, 11, 13, and 16 as anchor labs. The remaining
notebooks work well as team assignments because each ends with exercises and
has deterministic expected outputs.

## Optional external bridge

Notebook 17 is intentionally outside the chapter sequence. Complete notebooks
04 and 06 first. The extension introduces a real local Transformers model and
public SEC data, but keeps both disabled in the reproducible default run.

See [OPTIONAL_EXTENSIONS.md](OPTIONAL_EXTENSIONS.md) before enabling either
adapter.
