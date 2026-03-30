"""Input validation and preprocessing."""

from __future__ import annotations

import numpy as np


def validate_and_prepare(X, n_components: int) -> np.ndarray:
    """Validate input and return a 2D array ready for PCA/persistence.

    If 1D, applies Takens delay embedding (delay=10, dimension=5).
    If 2D, returns as-is after validation.

    Raises ValueError on NaN, Inf, or empty input.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
        X = X.ravel()
        if len(X) == 0:
            raise ValueError("Input time series is empty.")
        if np.any(~np.isfinite(X)):
            raise ValueError("Input contains NaN or Inf values.")
        return _takens_embed(X, delay=10, dimension=5)

    if X.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got {X.ndim}D.")
    if X.shape[0] == 0:
        raise ValueError("Input time series is empty.")
    if np.any(~np.isfinite(X)):
        raise ValueError("Input contains NaN or Inf values.")
    return X


def _takens_embed(x: np.ndarray, delay: int = 10, dimension: int = 5) -> np.ndarray:
    """Takens delay embedding for a 1D time series.

    Returns array of shape (n_embed, dimension) where
    n_embed = len(x) - (dimension - 1) * delay.
    """
    n_embed = len(x) - (dimension - 1) * delay
    if n_embed < 1:
        raise ValueError(
            f"Time series too short for Takens embedding: "
            f"need at least {(dimension - 1) * delay + 1} points, got {len(x)}."
        )
    embedded = np.zeros((n_embed, dimension))
    for d in range(dimension):
        embedded[:, d] = x[d * delay : d * delay + n_embed]
    return embedded


def is_constant(X: np.ndarray, tol: float = 1e-15) -> bool:
    """Check if the array is effectively constant."""
    return float(np.std(X)) < tol
