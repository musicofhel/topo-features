"""Core API: extract topological features from time series."""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd

from topofeatures._validation import is_constant, validate_and_prepare
from topofeatures.landscape import persistence_landscape_norm
from topofeatures.matching import wasserstein_window_drift as _wasserstein_window_drift
from topofeatures.null import persistence_significance as _persistence_significance
from topofeatures.persistence import compute_persistence, extract_features
from topofeatures.synergy import channel_synergy as _channel_synergy

ZERO_FEATURES = {
    "betti_0": 0,
    "betti_1": 0,
    "persistence_entropy": 0.0,
    "max_H1_persistence": 0.0,
    "total_H1_persistence": 0.0,
    "birth_death_slope": 0.0,
    "persistence_landscape_norm": 0.0,
    "persistence_stability_ratio": 0.0,
    "channel_synergy": 0.0,
    "persistence_significance": 0,
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
        'birth_death_slope': float -- OLS slope of death vs birth in H1
        'persistence_landscape_norm': float -- L2 norm of first H1 landscape
        'persistence_stability_ratio': float -- |total_H1(noisy)-total_H1(clean)|/eps
        'channel_synergy': float -- joint - mean(per-channel) H1 persistence
        'persistence_significance': int -- H1 features exceeding surrogate max
    """
    X_original = np.asarray(X, dtype=float)
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
    features = extract_features(diagrams)

    # Persistence landscape norm (Machine 2: Parameterized Homology)
    dgm1 = diagrams[1] if len(diagrams) > 1 else np.array([]).reshape(0, 2)
    features["persistence_landscape_norm"] = persistence_landscape_norm(dgm1)

    # Persistence stability ratio (Machine 4: Stability)
    # eps = 0.05 * diameter(cloud), averaged over 5 seeds
    diam = np.ptp(cloud, axis=0).max()
    eps = 0.05 * diam
    if eps > 1e-15:
        clean_total = features["total_H1_persistence"]
        noisy_totals = []
        for s in range(5):
            rng_stab = np.random.default_rng(seed + s + 1000)
            noisy_cloud = cloud + rng_stab.normal(0, eps, size=cloud.shape)
            noisy_dgms = compute_persistence(noisy_cloud, max_dim=max_dim)
            noisy_dgm1 = noisy_dgms[1] if len(noisy_dgms) > 1 else np.array([]).reshape(0, 2)
            if len(noisy_dgm1) > 0:
                lt = noisy_dgm1[:, 1] - noisy_dgm1[:, 0]
                lt = lt[np.isfinite(lt) & (lt > 1e-10)]
                noisy_totals.append(float(lt.sum()))
            else:
                noisy_totals.append(0.0)
        mean_diff = np.mean([abs(nt - clean_total) for nt in noisy_totals])
        features["persistence_stability_ratio"] = float(mean_diff / eps)
    else:
        features["persistence_stability_ratio"] = 0.0

    # Channel synergy (Machine 5: Joint-vs-Marginal Excess)
    features["channel_synergy"] = _channel_synergy(
        X_original, cloud, max_dim=max_dim, subsample=subsample, seed=seed,
    )

    # Persistence significance (Machine 6: Null Hypothesis)
    features["persistence_significance"] = _persistence_significance(
        X_original, diagrams, max_dim=max_dim, subsample=subsample,
        n_components=n_components, seed=seed,
    )

    return features


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
    features : ndarray, shape (n_windows, n_features)
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
    window_diagrams = []
    max_dim = kwargs.get("max_dim", 1)
    subsample_n = kwargs.get("subsample", 400)
    n_comp = kwargs.get("n_components", 3)
    seed = kwargs.get("seed", 42)

    for s, e in windows:
        w = X[s:e] if X.ndim == 1 else X[s:e, :]
        feats = extract(w, **kwargs)
        results.append([feats[k] for k in FEATURE_NAMES])

        # Also compute raw diagrams for Wasserstein drift
        from topofeatures._validation import is_constant, validate_and_prepare
        try:
            sm = validate_and_prepare(w, n_comp)
            if not is_constant(sm):
                cloud = _pca_reduce(sm, n_comp)
                if not is_constant(cloud):
                    if len(cloud) > subsample_n:
                        rng = np.random.default_rng(seed)
                        idx = rng.choice(len(cloud), subsample_n, replace=False)
                        cloud = cloud[idx]
                    dgms = compute_persistence(cloud, max_dim=max_dim)
                    window_diagrams.append(dgms)
                else:
                    window_diagrams.append([])
            else:
                window_diagrams.append([])
        except (ValueError, TypeError):
            window_diagrams.append([])

    drift = _wasserstein_window_drift(window_diagrams)

    return np.array(results), FEATURE_NAMES, drift


def _pca_reduce(states: np.ndarray, n_components: int) -> np.ndarray:
    """PCA reduce (n_steps, n_vars) -> (n_steps, n_components)."""
    if states.shape[1] <= n_components:
        return states

    X = states - states.mean(axis=0)
    _, _, Vt = svd(X, full_matrices=False)
    return X @ Vt[:min(n_components, len(Vt))].T
