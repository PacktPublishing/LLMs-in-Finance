from __future__ import annotations

import numpy as np

from .text import stable_softmax


def dpo_loss(
    chosen_logp: np.ndarray,
    rejected_logp: np.ndarray,
    ref_chosen_logp: np.ndarray,
    ref_rejected_logp: np.ndarray,
    beta: float = 0.1,
) -> float:
    model_margin = np.asarray(chosen_logp) - np.asarray(rejected_logp)
    ref_margin = np.asarray(ref_chosen_logp) - np.asarray(ref_rejected_logp)
    logits = beta * (model_margin - ref_margin)
    return float(np.mean(np.logaddexp(0.0, -logits)))


def grpo_advantages(rewards: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [prompts, group]")
    means = rewards.mean(axis=1, keepdims=True)
    scales = rewards.std(axis=1, keepdims=True)
    return (rewards - means) / np.maximum(scales, epsilon)


def ppo_clipped_objective(
    new_logp: np.ndarray,
    old_logp: np.ndarray,
    advantages: np.ndarray,
    clip: float = 0.2,
) -> float:
    """PPO surrogate objective. This is *maximized*; its negation is the loss."""
    ratio = np.exp(np.asarray(new_logp) - np.asarray(old_logp))
    advantages = np.asarray(advantages)
    unclipped = ratio * advantages
    clipped = np.clip(ratio, 1 - clip, 1 + clip) * advantages
    return float(np.mean(np.minimum(unclipped, clipped)))


def discrete_policy(logits: np.ndarray) -> np.ndarray:
    return stable_softmax(np.asarray(logits, dtype=float), axis=-1)


def categorical_kl(policy: np.ndarray, reference: np.ndarray) -> float:
    """Mean KL(policy || reference), summed over the categorical support.

    The divergence is summed over the *last* axis and averaged over any leading
    rows. Summing over the whole array instead makes the result scale with batch
    size, so two runs with different batch sizes cannot be compared.
    """
    p = np.clip(np.asarray(policy, dtype=float), 1e-12, 1)
    q = np.clip(np.asarray(reference, dtype=float), 1e-12, 1)
    if p.shape != q.shape:
        raise ValueError("policy and reference must have the same shape")
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=-1)))
