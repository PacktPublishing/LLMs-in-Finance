# Data card: synthetic point-in-time market/text panel

- **Purpose:** show how revised data and future labels inflate a backtest.
- **Source:** seeded autoregressive latent state plus Gaussian return noise.
- **Status:** fully synthetic and not calibrated to a specific security.
- **Valid signal:** `text_signal`, available before `decision_time`.
- **Contaminated signal:** `revised_text_signal`, available after the decision.
- **Outcome:** decimal `next_return`, earned from `return_start_at` through
  `return_end_at`, strictly after the decision.
- **Limitations:** the controlled data-generating process is pedagogical; headline Sharpe ratios are not investment evidence.
