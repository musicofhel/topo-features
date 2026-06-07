"""Persistence significance via phase-randomized surrogates.

Machine 6 from topoyolo atlas. Implements Theiler Algorithm 1 (Theiler et al.
1992, DOI: 10.1016/0167-2789(92)90102-S): FFT → randomize phases → IFFT.
Preserves power spectrum, destroys temporal structure.

Reference: Vejdemo-Johansson & Mukherjee (2018, arXiv: 1812.06491) —
universal empirical null for PH significance testing.
"""

from __future__ import annotations

import numpy as np


def persistence_significance(
    X: np.ndarray,
    diagrams: list[np.ndarray],
    max_dim: int = 1,
    k: int = 19,
    subsample: int = 400,
    n_components: int = 3,
    seed: int = 42,
) -> int:
    """Count H1 features exceeding surrogate maximum lifetimes.

    Generates K phase-randomized surrogates (Theiler Algorithm 1), computes
    PH for each, and counts how many real H1 features have lifetimes exceeding
    the maximum lifetime across all surrogates. This is a rank-based test
    at p = 1/(K+1).

    K=19 gives p = 0.05; K=99 gives p = 0.01.

    Parameters
    ----------
    X : array, shape (n_steps,) or (n_steps, n_vars)
        Original time series (needed for surrogate generation).
    diagrams : list of ndarray
        Persistence diagrams from the real data.
    max_dim : int
        Maximum homology dimension.
    k : int
        Number of surrogates. Default 19 for p=0.05.
    subsample : int
        Max points for persistence computation.
    n_components : int
        PCA components for multivariate data.
    seed : int
        Random seed.

    Returns
    -------
    int
        Number of H1 features with lifetime exceeding all surrogates' max.
    """
    from topofeatures.persistence import compute_persistence

    rng = np.random.default_rng(seed)

    # Get real H1 lifetimes
    dgm1 = diagrams[1] if len(diagrams) > 1 else np.array([]).reshape(0, 2)
    if len(dgm1) == 0:
        return 0
    real_lifetimes = dgm1[:, 1] - dgm1[:, 0]
    real_lifetimes = real_lifetimes[np.isfinite(real_lifetimes) & (real_lifetimes > 1e-10)]
    if len(real_lifetimes) == 0:
        return 0

    # Generate surrogates and find max lifetime across all
    surrogate_max = 0.0
    for i in range(k):
        surr = _theiler_surrogate(X, rng)
        surr_cloud = _prepare_cloud(surr, n_components, subsample, rng)
        if surr_cloud is None or len(surr_cloud) < 3:
            continue
        surr_dgms = compute_persistence(surr_cloud, max_dim=max_dim)
        surr_dgm1 = surr_dgms[1] if len(surr_dgms) > 1 else np.array([]).reshape(0, 2)
        if len(surr_dgm1) > 0:
            surr_lt = surr_dgm1[:, 1] - surr_dgm1[:, 0]
            surr_lt = surr_lt[np.isfinite(surr_lt) & (surr_lt > 1e-10)]
            if len(surr_lt) > 0:
                surrogate_max = max(surrogate_max, surr_lt.max())

    # Count real features exceeding surrogate max
    return int(np.sum(real_lifetimes > surrogate_max))


def _theiler_surrogate(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Theiler Algorithm 1: FFT → randomize phases → IFFT.

    Preserves power spectrum (autocorrelation structure), destroys
    temporal ordering / nonlinear dynamics.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return _phase_randomize_1d(X, rng)
    # Multivariate: randomize each channel with same phases for cross-correlation preservation
    n = X.shape[0]
    freqs = np.fft.rfftfreq(n)
    phases = rng.uniform(0, 2 * np.pi, size=len(freqs))
    phases[0] = 0  # DC component
    if n % 2 == 0:
        phases[-1] = 0  # Nyquist component
    result = np.empty_like(X)
    for c in range(X.shape[1]):
        ft = np.fft.rfft(X[:, c])
        ft_shifted = ft * np.exp(1j * phases)
        result[:, c] = np.fft.irfft(ft_shifted, n=n)
    return result


def _phase_randomize_1d(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase randomization for a single 1D series."""
    n = len(x)
    ft = np.fft.rfft(x)
    phases = rng.uniform(0, 2 * np.pi, size=len(ft))
    phases[0] = 0
    if n % 2 == 0:
        phases[-1] = 0
    ft_shifted = ft * np.exp(1j * phases)
    return np.fft.irfft(ft_shifted, n=n)


def _prepare_cloud(
    X: np.ndarray,
    n_components: int,
    subsample: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Prepare a point cloud from a time series (same pipeline as core.extract)."""
    from topofeatures._validation import is_constant, validate_and_prepare
    try:
        state_matrix = validate_and_prepare(X, n_components)
    except (ValueError, TypeError):
        return None

    if is_constant(state_matrix):
        return None

    # PCA if needed
    if state_matrix.shape[1] > n_components:
        from numpy.linalg import svd
        centered = state_matrix - state_matrix.mean(axis=0)
        _, _, Vt = svd(centered, full_matrices=False)
        cloud = centered @ Vt[:min(n_components, len(Vt))].T
    else:
        cloud = state_matrix

    if is_constant(cloud):
        return None

    if len(cloud) > subsample:
        idx = rng.choice(len(cloud), subsample, replace=False)
        cloud = cloud[idx]

    return cloud
