"""Example: plug topological features into a tsfresh pipeline."""

import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

from topofeatures.tsfresh_binding import TOPO_SETTINGS, topological_features

# Create a small dataset in tsfresh format
rng = np.random.default_rng(42)
records = []
for series_id in range(5):
    n = 2000
    t = np.linspace(0, 10 * np.pi, n)
    freq = 1.0 + 0.5 * series_id
    values = np.sin(freq * t) + 0.1 * rng.standard_normal(n)
    for i in range(n):
        records.append({"id": series_id, "time": i, "value": values[i]})

df = pd.DataFrame(records)

# Add topological features to tsfresh's standard set
settings = ComprehensiveFCParameters()
settings[topological_features] = TOPO_SETTINGS["topological_features"]

features = extract_features(
    df,
    column_id="id",
    column_sort="time",
    default_fc_parameters=settings,
    disable_progressbar=True,
)

# Show just the topological columns
topo_cols = [c for c in features.columns if "topological" in c]
print("Topological features extracted alongside 700+ tsfresh features:\n")
print(features[topo_cols])
