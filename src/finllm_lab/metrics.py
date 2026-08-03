from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[int], bins: int = 10
) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    if p.shape != y.shape:
        raise ValueError("probabilities and labels must have the same shape")
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= low) & (p < high if high < 1 else p <= high)
        if mask.any():
            error += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(error)


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2))


def retrieval_metrics(
    ranked_ids: Sequence[Sequence[str]],
    relevant_ids: Sequence[set[str]],
    k: int = 5,
) -> dict[str, float]:
    recalls, reciprocal_ranks, ndcgs = [], [], []
    for ranked, relevant in zip(ranked_ids, relevant_ids, strict=True):
        top = list(ranked[:k])
        hits = [1 if item in relevant else 0 for item in top]
        recalls.append(sum(hits) / max(len(relevant), 1))
        first = next((i + 1 for i, hit in enumerate(hits) if hit), None)
        reciprocal_ranks.append(0.0 if first is None else 1.0 / first)
        dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
        ideal_hits = [1] * min(len(relevant), k)
        idcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(ideal_hits))
        ndcgs.append(dcg / idcg if idcg else 0.0)
    return {
        f"recall@{k}": float(np.mean(recalls)),
        "mrr": float(np.mean(reciprocal_ranks)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
    }


def retrieval_metrics_with_intervals(
    ranked_ids: Sequence[Sequence[str]],
    relevant_ids: Sequence[set[str]],
    k: int = 5,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Per-question retrieval metrics with bootstrap intervals over questions.

    Small benchmarks produce wide intervals. Reporting the point estimate alone
    invites over-reading a difference that the sample cannot support.
    """
    per_question: dict[str, list[float]] = {f"recall@{k}": [], "mrr": [], f"ndcg@{k}": []}
    for ranked, relevant in zip(ranked_ids, relevant_ids, strict=True):
        single = retrieval_metrics([ranked], [relevant], k=k)
        for name, value in single.items():
            per_question[name].append(value)
    return {
        name: bootstrap_interval(
            values, confidence=confidence, resamples=resamples, seed=seed
        )
        for name, values in per_question.items()
    }


def bootstrap_interval(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("values cannot be empty")
    rng = np.random.default_rng(seed)
    estimates = np.array(
        [statistic(rng.choice(values, size=len(values), replace=True)) for _ in range(resamples)]
    )
    alpha = 1 - confidence
    return (
        float(statistic(values)),
        float(np.quantile(estimates, alpha / 2)),
        float(np.quantile(estimates, 1 - alpha / 2)),
    )


def proportion_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation for the small samples used
    throughout this suite, where the naive interval can leave [0, 1].
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    from scipy.stats import norm

    z = float(norm.ppf(1 - (1 - confidence) / 2))
    p = successes / trials
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    half_width = (
        z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    )
    return float(p), float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def sharpe_standard_error(sharpe: float, observations: int, periods: int = 252) -> float:
    """Lo (2002) standard error of an annualized Sharpe ratio under i.i.d. returns.

    The point estimate alone is not evidence; with a few years of daily data the
    interval is typically wide enough to include zero.
    """
    if observations < 2:
        return float("nan")
    per_period = sharpe / math.sqrt(periods)
    return float(math.sqrt(periods) * math.sqrt((1 + 0.5 * per_period**2) / observations))


def annualized_sharpe(returns: Sequence[float], periods: int = 252) -> float:
    """Annualized Sharpe ratio; NaN when the return series has no dispersion.

    Returns NaN rather than 0.0 for a degenerate series: a constant positive
    return has an undefined (not zero) Sharpe ratio, and silently reporting 0.0
    would understate rather than flag the degeneracy.
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return float("nan")
    dispersion = np.std(r, ddof=1)
    if np.isclose(dispersion, 0.0, atol=1e-15):
        return float("nan")
    return float(np.sqrt(periods) * np.mean(r) / dispersion)


def max_drawdown(returns: Sequence[float]) -> float:
    wealth = np.cumprod(1 + np.asarray(returns, dtype=float))
    running = np.maximum.accumulate(wealth)
    return float(np.min(wealth / running - 1))


def release_score(
    metrics: dict[str, float],
    thresholds: dict[str, tuple[str, float]],
) -> tuple[bool, dict[str, bool]]:
    checks = {}
    for name, (direction, threshold) in thresholds.items():
        if name not in metrics:
            raise KeyError(f"release gate requires metric {name!r}, which was not supplied")
        value = metrics[name]
        if direction == "min":
            checks[name] = bool(value >= threshold)
        elif direction == "max":
            checks[name] = bool(value <= threshold)
        else:
            raise ValueError("direction must be 'min' or 'max'")
    return all(checks.values()), checks
