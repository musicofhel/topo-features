"""Persistent homology computation wrapper."""

from __future__ import annotations

import numpy as np


def compute_persistence(cloud: np.ndarray, max_dim: int = 1) -> list[np.ndarray]:
    """Compute persistent homology diagrams from a point cloud.

    Parameters
    ----------
    cloud : array, shape (n_points, n_dims)
        The point cloud.
    max_dim : int
        Maximum homology dimension.

    Returns
    -------
    list of arrays
        Persistence diagrams for dimensions 0 through max_dim.
        Each diagram has shape (n_features, 2) with columns [birth, death].
    """
    try:
        from ripser import ripser
    except ImportError:
        raise ImportError(
            "ripser is required for persistence computation. "
            "Install it with: pip install ripser"
        )

    result = ripser(cloud, maxdim=max_dim)
    return result["dgms"]


def extract_features(diagrams: list[np.ndarray]) -> dict:
    """Extract topological features from persistence diagrams.

    Returns dict with keys:
        betti_0, betti_1, persistence_entropy, max_H1_persistence,
        total_H1_persistence.
    """
    features = {}

    # H0: connected components
    dgm0 = diagrams[0] if len(diagrams) > 0 else np.array([]).reshape(0, 2)
    if len(dgm0) > 0:
        lifetimes0 = dgm0[:, 1] - dgm0[:, 0]
        # Filter infinite lifetimes and numerical noise
        lifetimes0 = lifetimes0[np.isfinite(lifetimes0) & (lifetimes0 > 1e-10)]
        features["betti_0"] = int(len(lifetimes0))
    else:
        features["betti_0"] = 0

    # H1: loops
    dgm1 = diagrams[1] if len(diagrams) > 1 else np.array([]).reshape(0, 2)
    if len(dgm1) > 0:
        lifetimes1 = dgm1[:, 1] - dgm1[:, 0]
        lifetimes1 = lifetimes1[np.isfinite(lifetimes1) & (lifetimes1 > 1e-10)]
        features["betti_1"] = int(len(lifetimes1))
        if len(lifetimes1) > 0:
            total = float(lifetimes1.sum())
            features["total_H1_persistence"] = total
            features["max_H1_persistence"] = float(lifetimes1.max())
            if total > 0:
                p = lifetimes1 / total
                features["persistence_entropy"] = float(
                    -np.sum(p * np.log(p + 1e-15))
                )
            else:
                features["persistence_entropy"] = 0.0
        else:
            features["total_H1_persistence"] = 0.0
            features["max_H1_persistence"] = 0.0
            features["persistence_entropy"] = 0.0
    else:
        features["betti_1"] = 0
        features["total_H1_persistence"] = 0.0
        features["max_H1_persistence"] = 0.0
        features["persistence_entropy"] = 0.0

    return features
