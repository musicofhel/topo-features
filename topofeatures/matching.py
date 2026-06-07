"""Wasserstein distance between persistence diagrams.

Machine 3 from topoyolo atlas (Optimal Matching / OT).
Reference: Bubenik & Elchesen — universality theorem for Wasserstein on PDs.
"""

from __future__ import annotations

import numpy as np


def wasserstein_window_drift(
    diagrams_list: list[list[np.ndarray]],
) -> float:
    """Compute mean Wasserstein-1 distance between consecutive windows' H1 diagrams.

    High drift = regime change between consecutive windows.
    Low drift = stationary dynamics.

    Parameters
    ----------
    diagrams_list : list of persistence diagram lists
        One entry per window. Each entry is a list of diagrams [H0, H1, ...].

    Returns
    -------
    float
        Mean W1 drift across consecutive window pairs. 0.0 if < 2 windows.
    """
    if len(diagrams_list) < 2:
        return 0.0

    try:
        from persim import wasserstein as persim_wasserstein
    except ImportError:
        # Fallback: use bottleneck-like proxy with numpy
        return _wasserstein_fallback(diagrams_list)

    drifts = []
    for i in range(1, len(diagrams_list)):
        dgm1_prev = diagrams_list[i - 1][1] if len(diagrams_list[i - 1]) > 1 else np.array([]).reshape(0, 2)
        dgm1_curr = diagrams_list[i][1] if len(diagrams_list[i]) > 1 else np.array([]).reshape(0, 2)

        # Filter to finite lifetimes
        dgm1_prev = _filter_finite(dgm1_prev)
        dgm1_curr = _filter_finite(dgm1_curr)

        if len(dgm1_prev) == 0 and len(dgm1_curr) == 0:
            drifts.append(0.0)
        else:
            drifts.append(persim_wasserstein(dgm1_prev, dgm1_curr))

    return float(np.mean(drifts))


def _filter_finite(dgm: np.ndarray) -> np.ndarray:
    """Keep only finite-lifetime features."""
    if len(dgm) == 0:
        return dgm
    mask = np.isfinite(dgm[:, 1]) & (dgm[:, 1] > dgm[:, 0])
    return dgm[mask]


def _wasserstein_fallback(diagrams_list: list[list[np.ndarray]]) -> float:
    """Simple total-persistence-difference proxy when persim unavailable."""
    totals = []
    for dgms in diagrams_list:
        dgm1 = dgms[1] if len(dgms) > 1 else np.array([]).reshape(0, 2)
        dgm1 = _filter_finite(dgm1)
        if len(dgm1) > 0:
            lt = dgm1[:, 1] - dgm1[:, 0]
            totals.append(float(lt.sum()))
        else:
            totals.append(0.0)

    drifts = [abs(totals[i] - totals[i - 1]) for i in range(1, len(totals))]
    return float(np.mean(drifts)) if drifts else 0.0
