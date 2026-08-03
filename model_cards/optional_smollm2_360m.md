# Optional SmolLM2 bridge — integration card

## Identity

- **Upstream model:** `HuggingFaceTB/SmolLM2-360M-Instruct`
- **Upstream card:** <https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct>
- **Upstream license:** Apache-2.0
- **Parameter scale:** approximately 0.4 billion parameters
- **Repository role:** opt-in integration example in notebook 17
- **Default release state:** disabled and not downloaded

This card describes the adapter contract, not a validated financial model. The
executed repository contains no output from the upstream model and makes no
claim about its financial-task accuracy.

## Intended use

- Demonstrate recorded, deterministic local text generation.
- Show how external model text enters the same evidence, numerical, policy, and
  human-review controls as any other proposal.
- Provide a lightweight starting point for a reader-run benchmark.

## Prohibited use

- Investment advice, autonomous order creation, or live execution.
- Treating generated prose as evidence.
- Release without task-specific evaluation and accountable human review.
- Silent use of a mutable upstream revision in a claimed reproducible result.

## Data and timing

- **Repository evaluation data:** none; optional model execution is skipped.
- **Training cutoff:** consult the upstream model card.
- **Point-in-time policy:** only supplied, admissible evidence may enter the
  prompt.
- **Contamination status:** not assessed for the repository's fictional
  benchmark.

## Performance

No task metric, calibration result, faithfulness result, latency claim, or cost
claim is reported for this optional model. Readers who enable it must record:

- immutable resolved model revision;
- software and hardware versions;
- complete prompt hash and decoding configuration;
- held-out chronological evaluation window;
- class-wise metrics and finite-sample intervals;
- citation, numerical consistency, and policy-test results; and
- latency and memory measurements on the actual deployment hardware.

## Controls

- Network/model access is disabled unless explicitly enabled.
- `trust_remote_code=False`.
- Decoding is greedy (`do_sample=False`).
- Generated text has no execution authority.
- Automatic release remains disabled even after a successful model call.

## Known limitations

The upstream card warns that output may be factually inaccurate, logically
inconsistent, or biased and that the model primarily handles English. Its small
size makes it suitable for an integration demonstration, not an assumed
financial-performance baseline.
