"""Channel synergy: joint-vs-marginal topological excess.

Machine 5 from topoyolo atlas. Inspired by Varley et al. (2025) — H2 cavities
in joint fMRI space correlate with information-theoretic synergy at rho = -0.55
to -0.65, absent from individual channels.
"""

from __future__ import annotations

import numpy as np


def channel_synergy(
    X: np.ndarray,
    joint_cloud: np.ndarray,
    max_dim: int = 1,
    delay: int = 10,
    embed_dim: int = 5,
    n_components: int = 3,
    subsample: int = 400,
    seed: int = 42,
) -> float:
    """Compute channel synergy: total_H1(joint) - mean(total_H1(channel_i)).

    Positive = cross-channel topological structure exists beyond what
    individual channels produce. Zero/negative = channels are topologically
    independent.

    Parameters
    ----------
    X : array, shape (n_steps, n_channels)
        Original multivariate time series (must have >= 2 columns).
    joint_cloud : array, shape (n_points, n_dims)
        PCA-reduced joint point cloud (already computed by caller).
    max_dim : int
        Maximum homology dimension.
    delay, embed_dim : int
        Takens embedding parameters for per-channel analysis.
    n_components : int
        PCA components for per-channel clouds.
    subsample : int
        Max points for persistence computation.
    seed : int
        Random seed.

    Returns
    -------
    float
        Channel synergy value. Returns 0.0 for univariate input.
    """
    if X.ndim == 1 or X.shape[1] < 2:
        return 0.0

    from topofeatures.persistence import compute_persistence

    rng = np.random.default_rng(seed)

    # Joint total H1 persistence
    jc = joint_cloud
    if len(jc) > subsample:
        idx = rng.choice(len(jc), subsample, replace=False)
        jc = jc[idx]
    joint_dgms = compute_persistence(jc, max_dim=max_dim)
    joint_dgm1 = joint_dgms[1] if len(joint_dgms) > 1 else np.array([]).reshape(0, 2)
    if len(joint_dgm1) > 0:
        lt = joint_dgm1[:, 1] - joint_dgm1[:, 0]
        lt = lt[np.isfinite(lt) & (lt > 1e-10)]
        joint_total = float(lt.sum())
    else:
        joint_total = 0.0

    # Per-channel total H1 persistence via Takens embedding
    n_channels = X.shape[1]
    channel_totals = []
    for c in range(n_channels):
        series = X[:, c]
        embedded = _takens_embed(series, delay, embed_dim)
        if embedded is None or len(embedded) < 3:
            channel_totals.append(0.0)
            continue

        # PCA reduce if needed
        if embedded.shape[1] > n_components:
            from numpy.linalg import svd
            centered = embedded - embedded.mean(axis=0)
            _, _, Vt = svd(centered, full_matrices=False)
            embedded = centered @ Vt[:min(n_components, len(Vt))].T

        if len(embedded) > subsample:
            idx = rng.choice(len(embedded), subsample, replace=False)
            embedded = embedded[idx]

        ch_dgms = compute_persistence(embedded, max_dim=max_dim)
        ch_dgm1 = ch_dgms[1] if len(ch_dgms) > 1 else np.array([]).reshape(0, 2)
        if len(ch_dgm1) > 0:
            lt = ch_dgm1[:, 1] - ch_dgm1[:, 0]
            lt = lt[np.isfinite(lt) & (lt > 1e-10)]
            channel_totals.append(float(lt.sum()))
        else:
            channel_totals.append(0.0)

    mean_channel = np.mean(channel_totals) if channel_totals else 0.0
    return float(joint_total - mean_channel)


def _takens_embed(series: np.ndarray, delay: int, dim: int) -> np.ndarray | None:
    """Create time-delay embedding of a 1D series."""
    n = len(series)
    needed = (dim - 1) * delay + 1
    if n < needed:
        return None
    rows = n - (dim - 1) * delay
    return np.array([series[i:i + (dim - 1) * delay + 1:delay] for i in range(rows)])
