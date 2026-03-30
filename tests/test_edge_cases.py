"""Edge case tests."""

import numpy as np
import pytest

from topofeatures import extract
from topofeatures.core import FEATURE_NAMES, ZERO_FEATURES


class TestConstantInput:
    def test_constant_2d_returns_zeros(self):
        X = np.ones((500, 3))
        result = extract(X)
        assert result == ZERO_FEATURES

    def test_constant_1d_returns_zeros(self):
        x = np.ones(2000)
        result = extract(x)
        assert result == ZERO_FEATURES


class TestShortInput:
    def test_very_short_1d_raises(self):
        """Too short for Takens embedding (need 41 points for delay=10, dim=5)."""
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="too short"):
            extract(x)

    def test_minimal_length_1d_works(self):
        """Exactly 41 points: (5-1)*10 + 1 = 41."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        result = extract(x)
        assert set(result.keys()) == set(FEATURE_NAMES)


class TestNaNInput:
    def test_nan_in_1d_raises(self):
        x = np.ones(500)
        x[100] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            extract(x)

    def test_nan_in_2d_raises(self):
        X = np.ones((500, 3))
        X[100, 1] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            extract(X)

    def test_inf_raises(self):
        x = np.ones(500)
        x[50] = np.inf
        with pytest.raises(ValueError, match="NaN"):
            extract(x)


class TestSingleColumn2D:
    def test_single_column_treated_as_1d(self):
        """(n, 1) array should be treated as 1D and get Takens embedding."""
        rng = np.random.default_rng(42)
        x_1d = rng.standard_normal(2000)
        x_2d = x_1d.reshape(-1, 1)

        r1 = extract(x_1d, seed=42)
        r2 = extract(x_2d, seed=42)

        for k in FEATURE_NAMES:
            assert r1[k] == r2[k]


class TestEmptyInput:
    def test_empty_1d_raises(self):
        with pytest.raises(ValueError, match="empty"):
            extract(np.array([]))

    def test_empty_2d_raises(self):
        with pytest.raises(ValueError, match="empty"):
            extract(np.zeros((0, 3)))


class TestHighDimensional:
    def test_high_dim_pca_reduces(self):
        """50-variable input gets PCA-reduced to 3 components."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((1000, 50))
        result = extract(X, n_components=3)
        assert set(result.keys()) == set(FEATURE_NAMES)
        for v in result.values():
            assert np.isfinite(v)

    def test_fewer_vars_than_components_skips_pca(self):
        """2-variable input with n_components=3 should skip PCA."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 10 * np.pi, 1000)
        X = np.column_stack([np.sin(t), np.cos(t)])
        X += 0.1 * rng.standard_normal(X.shape)
        result = extract(X, n_components=3)
        assert set(result.keys()) == set(FEATURE_NAMES)
