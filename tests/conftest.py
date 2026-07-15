"""
Pytest fixtures: synthetic data generators for unit and integration tests.
"""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def synthetic_subject(rng):
    """Create two synthetic REST runs for connectivity tests."""
    V = 500
    T = 200
    rest_runs = [rng.randn(T, V).astype(np.float32) for _ in range(2)]
    return {"V": V, "rest_runs": rest_runs}
