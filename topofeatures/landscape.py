"""Persistence landscape computation.

Reference: Bubenik (2015, JMLR) — Statistical Topological Data Analysis
using Persistence Landscapes.
"""

from __future__ import annotations

import numpy as np


def persistence_landscape_norm(
    dgm: np.ndarray,
    resolution: int = 100,
) -> float:
    """Compute L2 norm of the first persistence landscape function.

    The persistence landscape converts a persistence diagram into a sequence
    of piecewise-linear functions. The first landscape function lambda_1(t)
    is the pointwise maximum of all "tent" functions from individual features.

    For each birth-death pair (b, d), the tent function is:
        tent(t) = max(0, min(t - b, d - t))

    lambda_1(t) = max over all features of tent_i(t)

    The L2 norm captures total topological signal strength across all
    filtration scales. Dynamics papers show this detects bifurcation
    transitions because the landscape shape changes qualitatively.

    Parameters
    ----------
    dgm : array, shape (n_features, 2)
        Persistence diagram with columns [birth, death].
    resolution : int
        Number of grid points for numerical integration.

    Returns
    -------
    float
        L2 norm of the first landscape function. Returns 0.0 for empty diagrams.
    """
    if len(dgm) == 0:
        return 0.0

    # Filter to finite lifetimes
    finite_mask = np.isfinite(dgm[:, 1]) & (dgm[:, 1] > dgm[:, 0])
    dgm = dgm[finite_mask]
    if len(dgm) == 0:
        return 0.0

    births = dgm[:, 0]
    deaths = dgm[:, 1]

    t_min = births.min()
    t_max = deaths.max()
    if t_max <= t_min:
        return 0.0

    t_grid = np.linspace(t_min, t_max, resolution)
    dt = t_grid[1] - t_grid[0]

    # Vectorized tent computation: (n_features, resolution)
    # tent_i(t) = max(0, min(t - b_i, d_i - t))
    t_minus_b = t_grid[np.newaxis, :] - births[:, np.newaxis]  # (n, res)
    d_minus_t = deaths[:, np.newaxis] - t_grid[np.newaxis, :]  # (n, res)
    tents = np.maximum(0, np.minimum(t_minus_b, d_minus_t))    # (n, res)

    # First landscape = pointwise max
    lambda_1 = tents.max(axis=0)  # (res,)

    # L2 norm via trapezoidal rule
    norm = np.sqrt(np.trapezoid(lambda_1 ** 2, dx=dt))
    return float(norm)
