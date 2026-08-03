from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import annualized_sharpe, max_drawdown


def backtest_signal(
    frame: pd.DataFrame,
    signal_col: str,
    return_col: str = "next_return",
    cost_bps: float = 5.0,
) -> dict:
    """Sign-following backtest with linear turnover costs.

    Precondition: ``signal_col`` on row *t* must be observable at the decision
    time of row *t*, and ``return_col`` on row *t* must be earned *after* that
    decision. This function cannot verify that — run
    ``temporal.availability_audit`` first.
    """
    signal = np.sign(frame[signal_col].to_numpy(dtype=float))
    returns = frame[return_col].to_numpy(dtype=float)
    turnover = np.abs(np.diff(np.r_[0.0, signal]))
    strategy = signal * returns - turnover * cost_bps / 10_000
    return {
        "returns": strategy,
        "sharpe": annualized_sharpe(strategy),
        "max_drawdown": max_drawdown(strategy),
        "mean_daily_bps": float(strategy.mean() * 10_000),
        "turnover": float(turnover.mean()),
        "observations": int(strategy.size),
    }


def herding_index(actions: np.ndarray) -> float:
    """Normalized net direction: |mean(a)| / mean(|a|).

    0 means perfectly offsetting positions, 1 means fully one-sided. Note the
    finite-sample floor: for n independent +/-1 actions the expected value is
    about sqrt(2 / (pi * n)), not 0, so a small population never reads exactly
    balanced. Compare against ``herding_null_baseline(n)``.
    """
    actions = np.asarray(actions, dtype=float)
    if actions.size == 0:
        return 0.0
    return float(abs(actions.mean()) / max(np.mean(abs(actions)), 1e-12))


def herding_null_baseline(n_agents: int) -> float:
    """Expected herding index for n independent, unbiased +/-1 actions."""
    if n_agents < 1:
        raise ValueError("n_agents must be positive")
    return float(np.sqrt(2.0 / (np.pi * n_agents)))


def spectral_abscissa(matrix: np.ndarray) -> float:
    """Largest real part of the eigenvalues: max Re(lambda).

    Stability criterion for a *continuous-time* linearization ``dx/dt = J x``,
    which becomes unstable as the abscissa crosses zero. For a discrete-time map
    ``x_{t+1} = J x_t`` use ``spectral_radius``, whose boundary is one, not zero.
    """
    return float(np.max(np.real(np.linalg.eigvals(np.asarray(matrix, dtype=float)))))


def spectral_radius(matrix: np.ndarray) -> float:
    """Largest eigenvalue modulus: stability criterion for a discrete-time map."""
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(matrix, dtype=float)))))
