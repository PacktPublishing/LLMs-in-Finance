# Data card: fictional filing corpus

- **Purpose:** time-valid retrieval, metadata filtering, citation, and faithfulness evaluation.
- **Source:** deterministic fictional 10-Q/10-K-style passages.
- **Status:** no text is copied from an actual filing.
- **Unit:** one issuer-period-section record.
- **Availability:** each record has a UTC `available_at` timestamp.
- **Ground truth:** `rag_questions.json` identifies one relevant section for each query.
- **Limitations:** relevance judgments are single-answer and lexical cues are cleaner than production corpora.

