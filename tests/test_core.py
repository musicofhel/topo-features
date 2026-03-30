"""Tests for topofeatures.core.extract()."""

import numpy as np
import pytest

from topofeatures import extract, extract_many, extract_windows
from topofeatures.core import FEATURE_NAMES


@pytest.fixture
def sine_3d():
    """3-variable sine wave with noise, 2000 steps."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 10 * np.pi, 2000)
    X = np.column_stack([np.sin(t), np.cos(t), np.sin(2 * t)])
    return X + 0.1 * rng.standard_normal(X.shape)


@pytest.fixture
def sine_1d():
    """1D sine wave, 2000 steps."""
    t = np.linspace(0, 10 * np.pi, 2000)
    return np.sin(t) + 0.05 * np.random.default_rng(42).standard_normal(2000)


class TestExtract:
    def test_returns_dict_with_5_keys(self, sine_3d):
        result = extract(sine_3d)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(FEATURE_NAMES)

    def test_values_are_finite(self, sine_3d):
        result = extract(sine_3d)
        for v in result.values():
            assert np.isfinite(v)

    def test_betti_values_are_nonneg_ints(self, sine_3d):
        result = extract(sine_3d)
        assert isinstance(result["betti_0"], int)
        assert isinstance(result["betti_1"], int)
        assert result["betti_0"] >= 0
        assert result["betti_1"] >= 0

    def test_persistence_values_nonneg(self, sine_3d):
        result = extract(sine_3d)
        assert result["persistence_entropy"] >= 0.0
        assert result["max_H1_persistence"] >= 0.0
        assert result["total_H1_persistence"] >= 0.0

    def test_1d_input_works(self, sine_1d):
        result = extract(sine_1d)
        assert set(result.keys()) == set(FEATURE_NAMES)
        for v in result.values():
            assert np.isfinite(v)

    def test_2d_input_works(self, sine_3d):
        result = extract(sine_3d)
        assert result["betti_1"] >= 1  # sine should have loops

    def test_reproducible_with_same_seed(self, sine_3d):
        r1 = extract(sine_3d, seed=42)
        r2 = extract(sine_3d, seed=42)
        for k in FEATURE_NAMES:
            assert r1[k] == r2[k]

    def test_different_seeds_similar_features(self, sine_3d):
        r1 = extract(sine_3d, seed=42)
        r2 = extract(sine_3d, seed=99)
        # Betti numbers may differ slightly due to subsampling,
        # but should be in the same ballpark
        assert abs(r1["betti_1"] - r2["betti_1"]) < max(r1["betti_1"], r2["betti_1"]) * 0.5

    def test_max_dim_2(self, sine_3d):
        result = extract(sine_3d, max_dim=2)
        assert set(result.keys()) == set(FEATURE_NAMES)

    def test_small_subsample(self, sine_3d):
        result = extract(sine_3d, subsample=50)
        assert set(result.keys()) == set(FEATURE_NAMES)


class TestExtractMany:
    def test_returns_list_of_dicts(self, sine_3d, sine_1d):
        results = extract_many([sine_3d, sine_1d])
        assert len(results) == 2
        for r in results:
            assert set(r.keys()) == set(FEATURE_NAMES)


class TestExtractWindows:
    def test_returns_correct_shape(self, sine_3d):
        features, names = extract_windows(sine_3d, window_size=500, step_size=250)
        expected_windows = (2000 - 500) // 250 + 1
        assert features.shape == (expected_windows, 5)
        assert len(names) == 5

    def test_feature_names_match(self, sine_3d):
        _, names = extract_windows(sine_3d, window_size=500, step_size=500)
        assert names == FEATURE_NAMES

    def test_1d_windows(self, sine_1d):
        features, _ = extract_windows(sine_1d, window_size=500, step_size=500)
        assert features.shape[1] == 5
        assert features.shape[0] >= 1
