# topo-features

5 topological features for time series. Works standalone or as a tsfresh plugin.

Extracts persistent homology features from multivariate time series via PCA dimensionality reduction + Vietoris-Rips filtration. Captures geometric structure (loops, connectedness, complexity) that spectral analysis and statistical moments miss.

## Install

```bash
pip install topo-features
```

With tsfresh integration:
```bash
pip install topo-features[tsfresh]
```

## Quickstart

```python
import numpy as np
from topofeatures import extract

t = np.linspace(0, 10 * np.pi, 2000)
X = np.column_stack([np.sin(t), np.cos(t), np.sin(2*t)])
X += 0.1 * np.random.default_rng(42).standard_normal(X.shape)

features = extract(X)
# {'betti_0': 1, 'betti_1': 2, 'persistence_entropy': 0.68,
#  'max_H1_persistence': 1.23, 'total_H1_persistence': 1.89}
```

## tsfresh integration

```python
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from topofeatures.tsfresh_binding import topological_features, TOPO_SETTINGS

settings = ComprehensiveFCParameters()
settings[topological_features] = TOPO_SETTINGS["topological_features"]

features = extract_features(df, default_fc_parameters=settings)
# Now includes betti_0, betti_1, persistence_entropy alongside 700+ tsfresh features
```

## What the features mean

| Feature | Description |
|---------|-------------|
| `betti_0` | Number of connected components in the point cloud |
| `betti_1` | Number of loops (1-cycles) — captures oscillatory structure |
| `persistence_entropy` | Shannon entropy of loop lifetimes — high = many similar loops, low = one dominant loop |
| `max_H1_persistence` | Lifetime of the most persistent loop — how "real" the dominant cycle is |
| `total_H1_persistence` | Sum of all loop lifetimes — total geometric complexity |

## When to use it

Your time series comes from a dynamical system (sensor data, neural recordings, simulations, process monitoring). You want features that capture geometric structure beyond what spectral analysis and statistical moments provide.

## When NOT to use it

- Purely stochastic data (financial returns, white noise)
- Series shorter than ~100 points (after embedding)
- Single-variable data without oscillatory structure

## Validated on

Tested on 20 dynamical systems across the probe battery (18/20 passed, |rho| > 0.6):

| Domain | Systems | Result |
|--------|---------|--------|
| Reservoir computing | ESN (5 input types, 4 sizes, 3 topologies) | 18/20 pass |
| Neural populations | FitzHugh-Nagumo, Hopfield, Spiking STDP | Pass |
| Oscillator networks | Kuramoto synchronization | Pass |
| Ecosystems | Lotka-Volterra (5 species) | Pass |
| Spatiotemporal | Coupled map lattice, reaction-diffusion | Pass |
| Boolean networks | Kauffman NK model | Pass |
| Gene regulation | 10-gene regulatory circuit | Pass |
| Real data | EEG (Sleep-EDF), power grid (PMU), weather | Pass |

Mean Spearman |rho| = 0.76 between topological features and ground-truth system properties.

## Performance

~150ms per series at default settings (subsample=400, PCA to 3 components). Runtime scales with `subsample`, not series length.

## API

```python
# Single series
extract(X, n_components=3, subsample=400, max_dim=1, seed=42) -> dict

# Multiple series
extract_many(X_list, **kwargs) -> list[dict]

# Sliding windows
extract_windows(X, window_size, step_size, **kwargs) -> (ndarray, list[str])
```

**Input**: 1D array (Takens embedding applied automatically) or 2D array (n_steps, n_variables).

## License

MIT
