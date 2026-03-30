"""Quickstart: extract topological features from a simple time series."""

import numpy as np

from topofeatures import extract

# A noisy limit cycle (3 variables)
t = np.linspace(0, 10 * np.pi, 2000)
X = np.column_stack([np.sin(t), np.cos(t), np.sin(2 * t)])
X += 0.1 * np.random.default_rng(42).standard_normal(X.shape)

features = extract(X)
print(features)
# Example output:
# {'betti_0': 1, 'betti_1': 2, 'persistence_entropy': 0.68,
#  'max_H1_persistence': 1.23, 'total_H1_persistence': 1.89}
