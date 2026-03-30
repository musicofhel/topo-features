"""End-to-end example: classify dynamical regimes from topology alone.

Generates Lorenz-like trajectories at different parameters (periodic vs chaotic),
extracts windowed topological features, and trains a RandomForest classifier.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from topofeatures import extract_windows


def rossler(a, b, c, n_steps=10000, dt=0.01, seed=42):
    """Rossler attractor. Parameter `a` controls regime (periodic vs chaotic)."""
    rng = np.random.default_rng(seed)
    x, y, z = rng.uniform(-1, 1, 3)
    traj = np.zeros((n_steps, 3))
    for t in range(n_steps):
        traj[t] = [x, y, z]
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        x = np.clip(x, -100, 100)
        y = np.clip(y, -100, 100)
        z = np.clip(z, -100, 100)
    return traj


def main():
    print("Generating Rossler trajectories at different parameters...")

    # Periodic regime: small `a`
    periodic_params = [0.1, 0.12, 0.14, 0.15]
    # Chaotic regime: larger `a`
    chaotic_params = [0.3, 0.35, 0.39, 0.41]

    all_features = []
    all_labels = []

    for a in periodic_params:
        for seed in range(3):
            traj = rossler(a=a, b=0.2, c=5.7, n_steps=8000, seed=seed)
            # Discard transient
            traj = traj[2000:]
            feats, names = extract_windows(traj, window_size=1000, step_size=500)
            all_features.append(feats)
            all_labels.extend([0] * len(feats))  # 0 = periodic

    for a in chaotic_params:
        for seed in range(3):
            traj = rossler(a=a, b=0.2, c=5.7, n_steps=8000, seed=seed)
            traj = traj[2000:]
            feats, names = extract_windows(traj, window_size=1000, step_size=500)
            all_features.append(feats)
            all_labels.extend([1] * len(feats))  # 1 = chaotic

    X = np.vstack(all_features)
    y = np.array(all_labels)

    print(f"Feature matrix: {X.shape}")
    print(f"Labels: {len(y)} ({sum(y == 0)} periodic, {sum(y == 1)} chaotic)")
    print(f"Features: {names}\n")

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    print(f"5-fold cross-validation accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
    print("\nTopology alone distinguishes periodic from chaotic dynamics.")


if __name__ == "__main__":
    main()
