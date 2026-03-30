"""Core API: extract topological features from time series."""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd

from topofeatures._validation import is_constant, validate_and_prepare
from topofeatures.persistence import compute_persistence, extract_features

ZERO_FEATURES = {
    "betti_0": 0,
    "betti_1": 0,
    "persistence_entropy": 0.0,
    "max_H1_persistence": 0.0,
    "total_H1_persistence": 0.0,
}

FEATURE_NAMES = list(ZERO_FEATURES.keys())


def extract(X, n_components=3, subsample=400, max_dim=1, seed=42):
    """Extract topological features from a multivariate time series.

    Parameters
    ----------
    X : array-like, shape (n_steps, n_variables) or (n_steps,)
        The time series. If 1D, Takens embedding is applied first
        (delay=10, dimension=5). If 2D, PCA projection is applied.
    n_components : int
        Number of PCA components for dimensionality reduction.
        Ignored if X is 1D (Takens embedding used instead).
    subsample : int
        Max points for persistence computation. Keeps runtime bounded.
    max_dim : int
        Maximum homology dimension (1 = detect loops, 2 = detect voids).
    seed : int
        Random seed for subsampling reproducibility.

    Returns
    -------
    dict with keys:
        'betti_0': int -- number of connected components
        'betti_1': int -- number of loops
        'persistence_entropy': float -- Shannon entropy of H1 lifetimes
        'max_H1_persistence': float -- lifetime of the most persistent loop
        'total_H1_persistence': float -- sum of all H1 lifetimes
    """
    state_matrix = validate_and_prepare(X, n_components)

    if is_constant(state_matrix):
        return dict(ZERO_FEATURES)

    # PCA reduction if needed
    cloud = _pca_reduce(state_matrix, n_components)

    if is_constant(cloud):
        return dict(ZERO_FEATURES)

    # Subsample for bounded runtime
    if len(cloud) > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(cloud), subsample, replace=False)
        cloud = cloud[idx]

    # Persistent homology
    diagrams = compute_persistence(cloud, max_dim=max_dim)
    return extract_features(diagrams)


def extract_many(X_list, **kwargs):
    """Extract features from multiple time series.

    Parameters
    ----------
    X_list : list of array-like
        Each element is a time series (1D or 2D).
    **kwargs
        Passed to extract().

    Returns
    -------
    list of dict
        One feature dict per input time series.
    """
    return [extract(X, **kwargs) for X in X_list]


def extract_windows(X, window_size, step_size, **kwargs):
    """Sliding-window feature extraction.

    Parameters
    ----------
    X : array-like, shape (n_steps, n_variables) or (n_steps,)
        The time series.
    window_size : int
        Number of time steps per window.
    step_size : int
        Step between consecutive windows.
    **kwargs
        Passed to extract().

    Returns
    -------
    features : ndarray, shape (n_windows, 5)
        Feature matrix.
    names : list of str
        Feature names corresponding to columns.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    windows = []
    start = 0
    while start + window_size <= n:
        windows.append((start, start + window_size))
        start += step_size

    if not windows:
        raise ValueError(
            f"window_size={window_size} exceeds series length={n}."
        )

    results = []
    for s, e in windows:
        w = X[s:e] if X.ndim == 1 else X[s:e, :]
        feats = extract(w, **kwargs)
        results.append([feats[k] for k in FEATURE_NAMES])

    return np.array(results), FEATURE_NAMES


def _pca_reduce(states: np.ndarray, n_components: int) -> np.ndarray:
    """PCA reduce (n_steps, n_vars) -> (n_steps, n_components)."""
    if states.shape[1] <= n_components:
        return states

    X = states - states.mean(axis=0)
    _, _, Vt = svd(X, full_matrices=False)
    return X @ Vt[:min(n_components, len(Vt))].T
