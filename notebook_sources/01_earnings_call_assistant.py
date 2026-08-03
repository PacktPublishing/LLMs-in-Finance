# %% [markdown]
# # Chapter 1 Lab: Build an Earnings-Call Assistant
#
# **Runtime:** 2–3 minutes · **Difficulty:** introductory · **Credentials:** none
#
# This notebook turns the chapter's running case into a small, auditable workflow:
# extraction, language modeling, tone classification, evidence, and a reviewable output.
# It uses fictional calls and transparent baselines so every intermediate result can be
# inspected.

# %% [markdown]
# ## Learning objectives
#
# - tokenize finance-specific expressions without destroying quantities;
# - compare a bigram model with a contextual text classifier;
# - extract exact financial quantities and their character spans;
# - separate a draft summary from its evidence; and
# - identify where a production LLM would add value.

# %%
from pathlib import Path
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from finllm_lab import Claim, Document, EvidenceBundle, seed_everything
from finllm_lab.plotting import COLORS, finish, set_finance_theme
from finllm_lab.rag import BM25Index, faithfulness_score
from finllm_lab.temporal import chronological_split
from finllm_lab.text import NGramLanguageModel, financial_metrics, tokenize_financial

set_finance_theme()
rng = seed_everything(101)
calls = pd.read_csv(ROOT / "data" / "earnings_calls.csv", parse_dates=["published_at", "available_at"])
calls = calls.sort_values("available_at").reset_index(drop=True)
tone_phrases = [
    "Management described demand as resilient and execution as strong.",
    "Management cited softer demand, execution pressure, and elevated uncertainty.",
    "Management described conditions as mixed and maintained a balanced outlook.",
]
calls["model_text"] = calls["text"]
for phrase in tone_phrases:
    calls["model_text"] = calls["model_text"].str.replace(
        phrase,
        "Management discussed execution, demand, and uncertainty.",
        regex=False,
    )
calls[["ticker", "period", "revenue_growth_pct", "margin_change_bps", "guidance_delta_pct", "sentiment_label"]].head()

# %% [markdown]
# ## 1. Financial tokenization
#
# Generic whitespace tokenization separates symbols from quantities inconsistently. The
# book's applications need `$1,234.5`, `15.3%`, `125 bps`, and `10-Q` to remain useful
# units.

# %%
sample = calls.iloc[-1]["text"]
tokens = tokenize_financial(sample)
metrics = pd.DataFrame(financial_metrics(sample))
print("First 35 tokens:")
print(tokens[:35])
metrics

# %% [markdown]
# ## 2. What a statistical language model learns
#
# A smoothed bigram model captures local phrasing but not deep context. We train on calls
# before 2025 and inspect likely continuations and out-of-sample perplexity.

# %%
early = calls[calls["available_at"] < "2025-01-01"]
late = calls[calls["available_at"] >= "2025-01-01"]
bigram = NGramLanguageModel(n=2, alpha=0.5).fit(early["text"])
predictions = pd.DataFrame(
    bigram.predict_next(["revenue"], k=8), columns=["next_token", "probability"]
)
perplexities = late["text"].map(bigram.perplexity)
display(predictions)
pd.Series(perplexities).describe().to_frame("out_of_sample_perplexity")

# %%
fig, ax = plt.subplots()
ax.hist(perplexities, bins=12, color=COLORS["blue"], edgecolor="white")
ax.axvline(perplexities.median(), color=COLORS["coral"], linestyle="--", label="median")
ax.set(title="Bigram perplexity on later calls", xlabel="Perplexity", ylabel="Calls")
ax.legend()
finish(ax)
plt.tight_layout()

# %% [markdown]
# ## 3. A contextual baseline
#
# We now fit a TF–IDF logistic classifier using a strictly chronological split. The
# generated closing sentence names the label almost verbatim, so it is neutralized before
# modeling. We also score all three declared classes even when one class is absent from the
# short test window.

# %%
split = chronological_split(calls["available_at"], train_fraction=0.65, validation_fraction=0.15)
train, test = calls.iloc[split.train], calls.iloc[split.test]
all_labels = sorted(calls["sentiment_label"].unique())
classifier = Pipeline(
    [
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=2500)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=101)),
    ]
)
classifier.fit(train["model_text"], train["sentiment_label"])
pred = classifier.predict(test["model_text"])
macro_f1 = f1_score(
    test["sentiment_label"],
    pred,
    labels=all_labels,
    average="macro",
    zero_division=0,
)
print(f"Chronological test macro-F1: {macro_f1:.3f}")
print("Chronological test support:", test["sentiment_label"].value_counts().to_dict())
print(
    classification_report(
        test["sentiment_label"],
        pred,
        labels=all_labels,
        digits=3,
        zero_division=0,
    )
)

# %%
fig, ax = plt.subplots(figsize=(5.8, 4.8))
ConfusionMatrixDisplay.from_predictions(
    test["sentiment_label"],
    pred,
    labels=all_labels,
    normalize="true",
    cmap="Blues",
    values_format=".2f",
    ax=ax,
)
ax.set_title("Tone classification: chronological holdout")
plt.tight_layout()

# %% [markdown]
# ## 4. Assemble a reviewable assistant output
#
# The assistant extracts quantities, predicts tone, retrieves the source passage, and builds
# material claims. It does **not** turn the tone score into an investment recommendation.

# %%
target = calls[(calls["ticker"] == "ACME") & (calls["period"] == "2025Q4")].iloc[0]
target_doc = Document(
    document_id=target["document_id"],
    text=target["text"],
    source="fictional earnings call",
    available_at=target["available_at"].to_pydatetime(),
    issuer=target["ticker"],
    section="earnings_call",
)
retrieved = tuple(
    BM25Index([target_doc]).search(
        "revenue growth operating margin guidance",
        k=1,
        decision_time=(target["available_at"] + pd.Timedelta(hours=1)).to_pydatetime(),
    )
)
predicted_tone = classifier.predict([target["model_text"]])[0]
claims = [
    Claim("revenue", f"Revenue grew {target['revenue_growth_pct']:.1f}% year over year."),
    Claim("margin", f"Operating margin was {target['operating_margin_pct']:.1f}%."),
    Claim("guidance", f"Guidance changed {target['guidance_delta_pct']:+.1f}%."),
]
bundles = [EvidenceBundle(claim, retrieved) for claim in claims]
assistant_output = {
    "issuer": target["issuer"],
    "period": target["period"],
    "tone": predicted_tone,
    "key_metrics": financial_metrics(target["text"])[:6],
    "claims": [claim.text for claim in claims],
    "evidence_ids": [e.evidence_id for e in retrieved],
    "faithfulness": faithfulness_score(bundles),
    "review_status": "human review required before investment use",
}
assistant_output

# %% [markdown]
# ## 5. Inspect the signal over time
#
# The chart is an analysis aid, not a backtest. It makes drift and changing class balance
# visible before the classifier is used downstream.

# %%
acme = calls[calls["ticker"] == "ACME"].copy()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(acme["period"], acme["sentiment_score"], marker="o", color=COLORS["teal"])
ax.axhline(1.0, color=COLORS["green"], linestyle="--", linewidth=1, label="positive threshold")
ax.axhline(-0.8, color=COLORS["red"], linestyle="--", linewidth=1, label="negative threshold")
ax.axhline(0, color=COLORS["slate"], linewidth=0.8)
ax.set(title="ACME management-tone signal", xlabel="Quarter", ylabel="Synthetic tone score")
ax.tick_params(axis="x", rotation=60)
ax.legend(ncol=2)
finish(ax)
plt.tight_layout()

# %% [markdown]
# ## Takeaways
#
# - Exact quantities require finance-aware tokenization and span-preserving extraction.
# - Label-revealing template language must be removed before evaluating a tone baseline.
# - A transparent chronological baseline is the minimum comparison for a larger model.
# - Classification, extraction, summarization, and recommendation are different tasks.
# - A polished answer is not auditable until claims point to time-valid evidence.
#
# **Challenge:** replace the TF–IDF model with a locally approved encoder, keep the same
# chronological split, and compare macro-F1, calibration, latency, and memory.
