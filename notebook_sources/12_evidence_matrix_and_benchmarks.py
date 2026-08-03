# %% [markdown]
# # Chapter 12 Lab: Evidence Matrices and Honest Benchmarking
#
# **Runtime:** 2–3 minutes · **Difficulty:** intermediate · **Credentials:** none
#
# A headline score is not evidence by itself. This notebook compares fictional experiment
# cards by uncertainty, temporal validity, point-in-time discipline, and independent review.
# It shows how a lower raw score can deserve stronger claims than a higher but weaker study.

# %% [markdown]
# ## Learning objectives
#
# - build a minimum comparison record;
# - visualize sampling uncertainty;
# - separate empirical quality from evidence maturity;
# - shrink noisy small-sample estimates; and
# - license claims according to the strength of the evaluation.

# %%
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from finllm_lab import canonical_hash, seed_everything
from finllm_lab.metrics import bootstrap_interval
from finllm_lab.plotting import COLORS, finish, set_finance_theme

set_finance_theme()
seed_everything(1212)
evidence = pd.read_csv(ROOT / "data" / "evidence_matrix.csv")
evidence.head()

# %% [markdown]
# ## 1. Add evidence maturity

# %%
evidence["maturity_score"] = evidence[
    ["temporal_split", "point_in_time", "independent_review"]
].astype(int).mean(axis=1)
evidence["interval_width"] = evidence["ci_high"] - evidence["ci_low"]
evidence["claim_tier"] = pd.cut(
    evidence["maturity_score"],
    bins=[-0.01, 0.01, 0.34, 0.67, 1.01],
    labels=["illustration", "preliminary", "comparative", "audited"],
)
evidence[
    [
        "experiment_id",
        "system",
        "task",
        "sample_size",
        "quality",
        "maturity_score",
        "claim_tier",
    ]
].head(10)

# %% [markdown]
# ## 2. Raw ranking versus evidence-aware ranking
#
# We use a small neutral prior to shrink noisier estimates and multiply by an evidence
# factor. This is an illustrative score, not a universal formula.

# %%
prior_n = 100
prior_mean = 0.50
evidence["shrunk_quality"] = (
    evidence["sample_size"] * evidence["quality"] + prior_n * prior_mean
) / (evidence["sample_size"] + prior_n)
evidence["evidence_aware_score"] = evidence["shrunk_quality"] * (
    0.55 + 0.45 * evidence["maturity_score"]
)
raw_top = evidence.nlargest(5, "quality")[
    ["experiment_id", "system", "task", "quality", "claim_tier"]
].assign(ranking="raw quality")
aware_top = evidence.nlargest(5, "evidence_aware_score")[
    ["experiment_id", "system", "task", "quality", "claim_tier"]
].assign(ranking="evidence aware")
pd.concat([raw_top, aware_top], ignore_index=True)

# %% [markdown]
# ## 3. Uncertainty belongs on the chart

# %%
plot_frame = evidence.sort_values("evidence_aware_score").tail(14)
fig, ax = plt.subplots(figsize=(9, 6.2))
errors = np.vstack(
    [
        plot_frame["quality"] - plot_frame["ci_low"],
        plot_frame["ci_high"] - plot_frame["quality"],
    ]
)
colors = [
    {
        "illustration": COLORS["slate"],
        "preliminary": COLORS["gold"],
        "comparative": COLORS["blue"],
        "audited": COLORS["green"],
    }[str(tier)]
    for tier in plot_frame["claim_tier"]
]
ax.errorbar(
    plot_frame["quality"],
    np.arange(len(plot_frame)),
    xerr=errors,
    fmt="none",
    ecolor=COLORS["slate"],
    capsize=3,
    alpha=0.8,
)
ax.scatter(plot_frame["quality"], np.arange(len(plot_frame)), c=colors, s=70)
ax.set_yticks(np.arange(len(plot_frame)), plot_frame["experiment_id"] + " · " + plot_frame["system"])
ax.set(title="Quality estimates with 95% intervals", xlabel="Illustrative quality metric")
finish(ax, "Fictional experiment cards")
plt.tight_layout()

# %% [markdown]
# ## 4. Validate the minimum comparison record

# %%
required_fields = {
    "experiment_id",
    "system",
    "task",
    "sample_size",
    "quality",
    "ci_low",
    "ci_high",
    "temporal_split",
    "point_in_time",
    "independent_review",
    "data_status",
}
record_check = pd.DataFrame(
    {
        "required_field": sorted(required_fields),
        "present": [field in evidence.columns for field in sorted(required_fields)],
        "non_null": [
            bool(evidence[field].notna().all()) if field in evidence.columns else False
            for field in sorted(required_fields)
        ],
    }
)
record_check

# %% [markdown]
# ## 5. System-level uncertainty

# %%
system_intervals = []
for system, group in evidence.groupby("system"):
    estimate, low, high = bootstrap_interval(
        group["quality"], confidence=0.95, resamples=1500, seed=1212
    )
    system_intervals.append(
        {
            "system": system,
            "mean_quality": estimate,
            "ci_low": low,
            "ci_high": high,
            "experiments": len(group),
            "mean_maturity": group["maturity_score"].mean(),
        }
    )
system_intervals = pd.DataFrame(system_intervals).sort_values("mean_quality", ascending=False)
system_intervals.round(4)

# %% [markdown]
# ## 6. License the claim, not just the score

# %%
claim_policy = pd.DataFrame(
    [
        ("illustration", "The workflow runs on a fictional example.", "No performance claim."),
        ("preliminary", "The system shows promise on this sample.", "No generalization claim."),
        ("comparative", "The system outperformed the declared baseline under this protocol.", "Protocol-specific claim."),
        ("audited", "The result reproduced under point-in-time, independently reviewed evaluation.", "Still bounded to the tested domain."),
    ],
    columns=["tier", "licensed_statement", "boundary"],
)
claim_policy

# %% [markdown]
# ## 7. Reproducibility record

# %%
registry = {
    "dataset": "fictional evidence cards",
    "experiments": evidence["experiment_id"].tolist(),
    "required_fields_pass": bool(
        record_check[["present", "non_null"]].to_numpy().all()
    ),
    "ranking_rule": "shrunk_quality * (0.55 + 0.45 * maturity_score)",
    "top_evidence_aware": aware_top.iloc[0]["experiment_id"],
}
registry["fingerprint"] = canonical_hash(registry)
registry

# %% [markdown]
# ## Takeaways
#
# - Sample size and uncertainty change how close scores should be interpreted.
# - Random-split or non-point-in-time results cannot license deployment claims.
# - Evidence maturity and task performance are distinct dimensions.
# - A comparison record should be machine-readable and complete before ranking.
#
# **Challenge:** replace the illustrative evidence score with a preregistered multi-criteria
# decision rule and run a sensitivity analysis over its weights.

