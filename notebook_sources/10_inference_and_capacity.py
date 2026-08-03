# %% [markdown]
# # Chapter 10 Lab: Inference, Capacity, Cost, and SLOs
#
# **Runtime:** 2–3 minutes · **Difficulty:** intermediate · **Credentials:** none
#
# Model quality is only one production dimension. This notebook combines illustrative
# serving configurations with queueing pressure, cost, memory, and a quality floor to select
# feasible deployment candidates.

# %% [markdown]
# ## Learning objectives
#
# - distinguish service latency from queueing latency;
# - identify unstable load configurations;
# - build a multi-constraint service-level objective;
# - calculate a quality–cost–latency Pareto frontier; and
# - stress a chosen configuration under a load spike.

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

from finllm_lab.plotting import COLORS, finish, set_finance_theme

set_finance_theme()
configs = pd.read_csv(ROOT / "data" / "infrastructure_configs.csv")
configs.head()

# %% [markdown]
# ## 1. Add queueing delay
#
# For an M/M/1 teaching approximation, waiting time becomes unstable when arrival rate
# reaches service rate. The queue-wait tail satisfies
# `P(W_q > t) = utilization × exp(-(service-arrival)t)`. We add that wait estimate to the
# measured no-queue service p95 as a conservative screening approximation.

# %%
arrival_rate = 10.0

def queue_wait_p95_ms(arrival: float, service: float) -> float:
    if arrival >= service:
        return np.inf
    utilization = arrival / service
    if utilization <= 0.05:
        return 0.0
    return float(np.log(utilization / 0.05) / (service - arrival) * 1000)

configs["utilization"] = arrival_rate / configs["service_rate_rps"]
configs["stable"] = configs["utilization"] < 1
configs["queue_p95_ms"] = [
    queue_wait_p95_ms(arrival_rate, service)
    for service in configs["service_rate_rps"]
]
configs["total_p95_ms"] = configs["p95_latency_ms"] + configs["queue_p95_ms"]
configs[
    [
        "configuration",
        "quality_index",
        "service_rate_rps",
        "utilization",
        "total_p95_ms",
        "cost_per_1k_requests_usd",
        "stable",
    ]
].replace(np.inf, np.nan).round(3)

# %% [markdown]
# ## 2. Apply the production SLO

# %%
quality_floor = 0.985
latency_ceiling = 1_250
cost_ceiling = 2.25
configs["quality_pass"] = configs["quality_index"] >= quality_floor
configs["latency_pass"] = configs["total_p95_ms"] <= latency_ceiling
configs["cost_pass"] = configs["cost_per_1k_requests_usd"] <= cost_ceiling
configs["slo_pass"] = configs[["stable", "quality_pass", "latency_pass", "cost_pass"]].all(axis=1)
feasible = configs[configs["slo_pass"]].sort_values(
    ["cost_per_1k_requests_usd", "total_p95_ms"]
)
feasible[
    ["configuration", "quality_index", "total_p95_ms", "cost_per_1k_requests_usd", "memory_index"]
].round(3)

# %% [markdown]
# ## 3. Compute the Pareto frontier

# %%
def dominated(row: pd.Series, frame: pd.DataFrame) -> bool:
    better_or_equal = (
        (frame["quality_index"] >= row["quality_index"])
        & (frame["total_p95_ms"] <= row["total_p95_ms"])
        & (frame["cost_per_1k_requests_usd"] <= row["cost_per_1k_requests_usd"])
    )
    strictly_better = (
        (frame["quality_index"] > row["quality_index"])
        | (frame["total_p95_ms"] < row["total_p95_ms"])
        | (frame["cost_per_1k_requests_usd"] < row["cost_per_1k_requests_usd"])
    )
    return bool((better_or_equal & strictly_better).any())

stable = configs[configs["stable"]].copy()
stable["pareto"] = [not dominated(row, stable) for _, row in stable.iterrows()]
stable[stable["pareto"]][
    ["configuration", "quality_index", "total_p95_ms", "cost_per_1k_requests_usd"]
].sort_values("cost_per_1k_requests_usd").round(3)

# %%
fig, ax = plt.subplots(figsize=(8.4, 5.0))
scatter = ax.scatter(
    stable["total_p95_ms"],
    stable["cost_per_1k_requests_usd"],
    c=stable["quality_index"],
    s=50 + stable["batch_size"] * 5,
    cmap="viridis",
    alpha=0.85,
)
pareto = stable[stable["pareto"]]
ax.scatter(
    pareto["total_p95_ms"],
    pareto["cost_per_1k_requests_usd"],
    facecolors="none",
    edgecolors=COLORS["coral"],
    s=180,
    linewidths=2,
    label="Pareto candidate",
)
ax.axvline(latency_ceiling, color=COLORS["red"], linestyle="--", label="latency ceiling")
ax.set(title="Serving configurations", xlabel="Total p95 latency (ms)", ylabel="Cost / 1k requests (USD)")
fig.colorbar(scatter, ax=ax, label="Quality index")
ax.legend()
finish(ax, "Illustrative benchmark; circle size increases with batch")
plt.tight_layout()

# %% [markdown]
# ## 4. Stress the best feasible candidate

# %%
if feasible.empty:
    raise RuntimeError("No configuration satisfies the declared SLO")
selected = feasible.iloc[0]
loads = np.linspace(1, selected["service_rate_rps"] * 1.10, 120)
stress_latency = np.array(
    [
        selected["p95_latency_ms"]
        + queue_wait_p95_ms(load, selected["service_rate_rps"])
        for load in loads
    ]
)
fig, ax = plt.subplots()
finite = np.isfinite(stress_latency)
ax.plot(loads[finite], stress_latency[finite], color=COLORS["blue"])
ax.axhline(latency_ceiling, color=COLORS["red"], linestyle="--", label="SLO")
ax.axvline(selected["service_rate_rps"], color=COLORS["coral"], linestyle=":", label="capacity")
ax.set(
    title=f"Load stress: {selected['configuration']}",
    xlabel="Arrival rate (requests / second)",
    ylabel="Approximate p95 latency (ms)",
    ylim=(0, min(5000, np.nanmax(stress_latency[finite]))),
)
ax.legend()
finish(ax)
plt.tight_layout()

# %% [markdown]
# ## 5. Deployment decision record

# %%
decision = {
    "selected_configuration": selected["configuration"],
    "arrival_rate_rps": arrival_rate,
    "quality_floor": quality_floor,
    "latency_ceiling_ms": latency_ceiling,
    "cost_ceiling_usd_per_1k": cost_ceiling,
    "observed_quality": float(selected["quality_index"]),
    "estimated_total_p95_ms": float(selected["total_p95_ms"]),
    "estimated_cost_usd_per_1k": float(selected["cost_per_1k_requests_usd"]),
    "caveat": (
        "Illustrative M/M/1 queue-wait screen with conservative p95 addition; "
        "validate the total-latency distribution with workload replay."
    ),
}
decision

# %% [markdown]
# ## Takeaways
#
# - A fast model can miss its SLO when the queue is unstable.
# - Quantization and batching trade quality, memory, latency, and cost.
# - Feasibility comes before scalar optimization.
# - A production choice needs workload replay, tail measurements, and failure-mode tests.
#
# **Challenge:** extend the model to two replicas with bursty arrivals and a 10-minute
# autoscaling delay.
