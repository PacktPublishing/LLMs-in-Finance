# Data

Every dataset in this directory is deterministic and fictional. No row represents
a real issuer, client, transaction, contract, filing, or investment result.

| File | Purpose | Temporal field |
|---|---|---|
| `earnings_calls.csv` | Financial text, extraction, sentiment, calibration | `available_at` |
| `filings.jsonl` | Point-in-time retrieval and claim support | `available_at` |
| `rag_questions.json` | Retrieval benchmark relevance judgments | `decision_time` |
| `market_text_panel.csv` | Clean versus contaminated backtests | decision, artifact availability, and return-window timestamps |
| `transactions.csv` | AML triage and cross-office workflow | `timestamp` |
| `contracts.jsonl` | Covenant extraction and monitoring | static fictional fixtures |
| `client_profiles.json` | Suitability control examples | static fictional fixtures |
| `preference_pairs.csv` | PPO/GRPO/DPO mathematics | synthetic preferences |
| `infrastructure_configs.csv` | Capacity, cost, quality, and latency trade-offs | illustrative benchmark |
| `governance_inventory.csv` | Model inventory and control dashboard | `last_validation_days` |
| `evidence_matrix.csv` | Evidence-quality comparison | illustrative experiment cards |

Regenerate all data with:

```bash
python scripts/generate_data.py
```

The generation seed is `32413`, the Packt production identifier for this edition.

Notebook 17 contains an optional SEC EDGAR adapter, but the committed release
does not download, cache, or ship real-company data. Its default execution uses
a synthetic SEC-shaped payload to test provenance, amendment, and point-in-time
logic. See [`docs/OPTIONAL_EXTENSIONS.md`](../docs/OPTIONAL_EXTENSIONS.md).
