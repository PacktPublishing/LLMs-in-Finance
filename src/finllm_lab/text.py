from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Sequence

import numpy as np

TOKEN_PATTERN = re.compile(
    r"""
    \$?\d+(?:,\d{3})*(?:\.\d+)?(?:%|bps|bn|mn|m|b)?
    |[A-Za-z]+(?:[-'][A-Za-z]+)*
    |\d{1,2}-[KQ]
    |[^\w\s]
    """,
    flags=re.VERBOSE | re.IGNORECASE,
)


def tokenize_financial(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def normalize_numeric_token(token: str) -> str:
    """Strip currency, grouping, and unit decoration from a numeric token.

    ``$1.5``, ``1,500``, and ``8%`` normalize to ``1.5``, ``1500`` and ``8`` so
    that a correct paraphrase is not scored as an unsupported claim merely
    because it dropped a dollar sign.
    """
    cleaned = token.strip().lower().lstrip("$").rstrip("%")
    for suffix in ("bps", "bn", "mn", "m", "b"):
        if cleaned.endswith(suffix) and cleaned[: -len(suffix)].replace(",", "").replace(
            ".", ""
        ).isdigit():
            cleaned = cleaned[: -len(suffix)]
            break
    cleaned = cleaned.replace(",", "")
    if cleaned and cleaned.replace(".", "", 1).isdigit():
        try:
            value = float(cleaned)
        except ValueError:
            return cleaned
        return str(int(value)) if value.is_integer() else str(value)
    return cleaned


def is_numeric_token(token: str) -> bool:
    return any(character.isdigit() for character in token)


def financial_metrics(text: str) -> list[dict]:
    patterns = {
        "currency": r"\$\s?\d+(?:,\d{3})*(?:\.\d+)?\s?(?:bn|mn|m|billion|million)?",
        "percentage": r"[-+]?\d+(?:\.\d+)?\s?%",
        "basis_points": r"[-+]?\d+(?:\.\d+)?\s?(?:bps|basis points?)",
        "multiple": r"\d+(?:\.\d+)?x",
    }
    found = []
    for kind, pattern in patterns.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            found.append(
                {
                    "type": kind,
                    "value": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    return sorted(found, key=lambda item: item["start"])


class NGramLanguageModel:
    def __init__(self, n: int = 2, alpha: float = 0.5):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self.alpha = alpha
        self.context_counts: Counter = Counter()
        self.next_counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        self.vocabulary: set[str] = set()

    def fit(self, texts: Iterable[str]) -> "NGramLanguageModel":
        pad = ["<s>"] * (self.n - 1)
        for text in texts:
            tokens = pad + tokenize_financial(text) + ["</s>"]
            self.vocabulary.update(tokens)
            for i in range(self.n - 1, len(tokens)):
                context = tuple(tokens[i - self.n + 1 : i])
                target = tokens[i]
                self.context_counts[context] += 1
                self.next_counts[context][target] += 1
        return self

    def probability(self, target: str, context: Sequence[str]) -> float:
        """Add-alpha probability. Scoring never mutates the fitted model."""
        context = tuple(context[-(self.n - 1) :]) if self.n > 1 else tuple()
        vocab_size = max(len(self.vocabulary), 1)
        observed = self.next_counts.get(context)
        numerator = (observed[target] if observed is not None else 0) + self.alpha
        denominator = self.context_counts[context] + self.alpha * vocab_size
        return numerator / denominator

    def predict_next(self, context: Sequence[str], k: int = 5) -> list[tuple[str, float]]:
        scores = [(t, self.probability(t, context)) for t in self.vocabulary]
        return sorted(scores, key=lambda pair: (-pair[1], pair[0]))[:k]

    def perplexity(self, text: str) -> float:
        pad = ["<s>"] * (self.n - 1)
        tokens = pad + tokenize_financial(text) + ["</s>"]
        losses = []
        for i in range(self.n - 1, len(tokens)):
            context = tokens[i - self.n + 1 : i]
            losses.append(-math.log(self.probability(tokens[i], context)))
        if not losses:
            return float("nan")
        return float(math.exp(np.mean(losses)))


def stable_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - values.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention over the last two axes.

    Leading axes are treated as batch or head dimensions, so ``[heads, tokens,
    d_k]`` and ``[batch, heads, tokens, d_k]`` behave as expected. Transposing
    only the final two axes matters: ``key.T`` reverses every axis and silently
    computes the wrong contraction whenever the shapes happen to conform.
    """
    query = np.asarray(query, dtype=float)
    key = np.asarray(key, dtype=float)
    value = np.asarray(value, dtype=float)
    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise ValueError("query, key and value must each have at least two dimensions")
    scale = math.sqrt(query.shape[-1])
    logits = query @ np.swapaxes(key, -1, -2) / scale
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != logits.shape[-mask.ndim :]:
            raise ValueError(f"mask shape {mask.shape} does not match logits {logits.shape}")
        logits = np.where(mask, logits, -np.inf)
        if bool((~mask).all(axis=-1).any()):
            raise ValueError("mask leaves at least one query with no admissible key")
    weights = stable_softmax(logits, axis=-1)
    return weights @ value, weights


def multi_head_attention(
    inputs: np.ndarray,
    projections: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_projection: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Full multi-head attention: per-head attention, concatenation, then W_O.

    The concatenation and output projection are the steps that make several
    heads one layer rather than several independent layers.
    """
    contexts, all_weights = [], []
    for w_q, w_k, w_v in projections:
        context, weights = scaled_dot_product_attention(
            inputs @ w_q, inputs @ w_k, inputs @ w_v, mask
        )
        contexts.append(context)
        all_weights.append(weights)
    concatenated = np.concatenate(contexts, axis=-1)
    if concatenated.shape[-1] != output_projection.shape[0]:
        raise ValueError(
            f"concatenated head width {concatenated.shape[-1]} does not match "
            f"output projection input width {output_projection.shape[0]}"
        )
    return concatenated @ output_projection, np.stack(all_weights)
