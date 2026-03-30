"""Tests for tsfresh integration."""

import numpy as np
import pytest

tsfresh = pytest.importorskip("tsfresh")

from topofeatures.tsfresh_binding import TOPO_SETTINGS, topological_features  # noqa: E402


class TestTopologicalFeatures:
    @pytest.fixture
    def sine_series(self):
        t = np.linspace(0, 10 * np.pi, 2000)
        return np.sin(t) + 0.05 * np.random.default_rng(42).standard_normal(2000)

    def test_returns_list_of_tuples(self, sine_series):
        param = TOPO_SETTINGS["topological_features"]
        result = topological_features(sine_series, param)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            name, value = item
            assert isinstance(name, str)
            assert isinstance(value, float)

    def test_names_follow_convention(self, sine_series):
        param = TOPO_SETTINGS["topological_features"]
        result = topological_features(sine_series, param)
        names = [name for name, _ in result]
        assert len(names) == 5
        for name in names:
            assert name.startswith("delay_10__dim_5__")

    def test_short_series_returns_zeros(self):
        short = np.ones(30)
        param = TOPO_SETTINGS["topological_features"]
        result = topological_features(short, param)
        assert len(result) == 5
        for _, value in result:
            assert value == 0.0

    def test_extract_features_integration(self):
        """Integration test: run topological_features through tsfresh."""
        import pandas as pd
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters

        rng = np.random.default_rng(42)
        n = 2000
        t = np.linspace(0, 10 * np.pi, n)
        values = np.sin(t) + 0.1 * rng.standard_normal(n)

        df = pd.DataFrame({
            "id": np.ones(n, dtype=int),
            "time": np.arange(n),
            "value": values,
        })

        settings = MinimalFCParameters()
        settings[topological_features] = TOPO_SETTINGS["topological_features"]

        result = extract_features(
            df,
            column_id="id",
            column_sort="time",
            default_fc_parameters=settings,
            disable_progressbar=True,
        )

        topo_cols = [c for c in result.columns if "topological" in c]
        assert len(topo_cols) == 5
