"""tsfresh integration: combiner feature calculator for topological features."""

from __future__ import annotations

import numpy as np

try:
    from tsfresh.feature_extraction.feature_calculators import set_property

    @set_property("fctype", "combiner")
    @set_property("high_comp_cost", True)
    def topological_features(x, param):
        """Topological features via persistent homology on delay-embedded time series.

        Extracts 5 features capturing the geometric structure of the attractor
        reconstructed from a univariate time series via Takens delay embedding.
        These features are complementary to spectral and statistical features --
        they capture loop structure, connectedness, and geometric complexity
        that variance, entropy, and FFT coefficients miss.

        Validated on 18/20 dynamical systems (Kuramoto oscillators, Lotka-Volterra
        ecosystems, Hopfield networks, gene regulatory circuits, power grids,
        real EEG, and more).

        :param x: the time series to calculate features of
        :type x: numpy.ndarray
        :param param: contains dictionaries with optional settings
        :type param: list
        :return: list of (feature_name, value) tuples
        :rtype: list
        """
        from topofeatures import extract

        x = np.asarray(x, dtype=float)
        results = []

        for p in param:
            delay = p.get("delay", 10)
            dimension = p.get("dimension", 5)
            subsample = p.get("subsample", 400)

            # Takens embedding for univariate input
            n_embed = len(x) - (dimension - 1) * delay
            if n_embed < 50:
                # Too short for meaningful topology — return zeros for all features
                from topofeatures.core import FEATURE_NAMES
                for name in FEATURE_NAMES:
                    results.append(
                        (f"delay_{delay}__dim_{dimension}__{name}", 0.0)
                    )
                continue

            embedded = np.zeros((n_embed, dimension))
            for d in range(dimension):
                embedded[:, d] = x[d * delay : d * delay + n_embed]

            features = extract(
                embedded,
                n_components=min(3, dimension),
                subsample=subsample,
            )

            for name, value in features.items():
                results.append(
                    (f"delay_{delay}__dim_{dimension}__{name}", float(value))
                )

        return results

except ImportError:
    def topological_features(x, param):
        """Stub: tsfresh not installed."""
        raise ImportError(
            "tsfresh is required for the tsfresh binding. "
            "Install it with: pip install topo-features[tsfresh]"
        )


# Default parameter settings for tsfresh integration
TOPO_SETTINGS = {
    "topological_features": [
        {"delay": 10, "dimension": 5, "subsample": 400},
    ]
}
