"""
Pytest configuration and shared fixtures.
"""
import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_times():
    """Generate sample time array."""
    return np.linspace(0, 10, 1000)


@pytest.fixture
def sample_params():
    """Generate sample system parameters."""
    return {
        'gamma': 100.0,
        'omega': 1000.0,
        'n': 1.0,
        'eta': 1.0,
        'kappa': 9.0
    }


@pytest.fixture
def sample_hypothesis_pair():
    """Generate a pair of hypotheses for testing."""
    h0 = [100.0, 0.0, 1.0, 1.0, 9.0]  # [gamma, omega, n, eta, kappa]
    h1 = [429.0, 0.0, 1.0, 1.0, 9.0]
    return [h1, h0]  # [true hypothesis, alternative hypothesis]


@pytest.fixture
def small_dt():
    """Small time step for numerical tests."""
    return 1e-4


@pytest.fixture
def total_time():
    """Total integration time for tests."""
    return 8.0

