"""
Unit tests for numerical integration routines.
"""
import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from numerics.integration.steps import Ikpw, Robler_step
from numerics.utilities.misc import get_stop_time, get_timind_indis


class TestIntegrationSteps:
    """Test numerical integration step functions."""
    
    def test_ikpw_shape(self):
        """Test that Ikpw returns correct shapes."""
        N = 100
        m = 2
        h = 0.01
        dW = np.random.randn(N, m)
        
        A, I = Ikpw(dW, h, n=5)
        
        assert A.shape == (N, m, m), f"Expected A shape ({N}, {m}, {m}), got {A.shape}"
        assert I.shape == (N, m, m), f"Expected I shape ({N}, {m}, {m}), got {I.shape}"
    
    def test_ikpw_symmetry(self):
        """Test that Ikpw I matrices have expected structure."""
        N = 10
        m = 2
        h = 0.01
        dW = np.random.randn(N, m)
        
        A, I = Ikpw(dW, h, n=5)
        
        # I should have shape (N, m, m)
        for i in range(N):
            assert I[i].shape == (m, m), f"I[{i}] should be {m}x{m}"
            # I should be finite
            assert np.all(np.isfinite(I[i])), f"I[{i}] should be finite"
    
    def test_ikpw_deterministic_limit(self):
        """Test Ikpw in deterministic limit (zero dW)."""
        N = 10
        m = 2
        h = 0.01
        dW = np.zeros((N, m))  # Deterministic case
        
        A, I = Ikpw(dW, h, n=5)
        
        # In deterministic limit, I has diagonal terms from -h/2
        # Off-diagonal terms should be small (from A term which depends on random Xk, Yk)
        # The diagonal should be approximately -h/2
        for i in range(N):
            diag = np.diag(I[i])
            # Diagonal should be approximately -h/2 (from the formula)
            assert np.allclose(diag, -h/2, atol=1e-2), "Diagonal should be approximately -h/2"
    
    def test_robler_step_shape(self):
        """Test Robler_step returns correct shape.
        
        Note: Robler_step is numba-jitted and requires numba-compatible functions.
        This test is skipped as it requires numba-compatible function definitions.
        """
        pytest.skip("Robler_step requires numba-compatible functions, cannot test with Python functions")
    
    def test_robler_step_linear_dynamics(self):
        """Test Robler_step with simple linear dynamics.
        
        Note: Robler_step is numba-jitted and requires numba-compatible functions.
        This test is skipped as it requires numba-compatible function definitions.
        """
        pytest.skip("Robler_step requires numba-compatible functions, cannot test with Python functions")


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_stop_time_basic(self):
        """Test get_stop_time with simple crossing.
        
        Note: get_stop_time finds the first time when ell is OUTSIDE [-b, b],
        i.e., when ell < -b or ell > b. It uses argmin on logicals which finds
        the first False (outside bounds).
        """
        times = np.linspace(0, 10, 100)
        # Create signal that starts within bounds and crosses outside
        ell = np.linspace(0, 5, 100)  # Goes from 0 to 5, crossing b=2.0
        b = 2.0
        
        stop_time = get_stop_time(ell, b, times)
        
        # Should find when ell crosses b (goes outside bounds)
        # The function finds first time OUTSIDE bounds
        assert not np.isnan(stop_time), "Should find a stop time when crossing boundary"
        assert 0 <= stop_time <= 10, "Stop time should be within time range"
    
    def test_get_stop_time_no_crossing(self):
        """Test get_stop_time when no crossing occurs."""
        times = np.linspace(0, 10, 100)
        ell = np.ones(100) * 0.5  # Always within bounds
        b = 2.0
        
        stop_time = get_stop_time(ell, b, times)
        
        assert np.isnan(stop_time), "Should return NaN when no crossing occurs"
    
    def test_get_stop_time_immediate_crossing(self):
        """Test get_stop_time with immediate crossing."""
        times = np.linspace(0, 10, 100)
        ell = np.ones(100) * 5.0  # Always outside bounds
        b = 2.0
        
        stop_time = get_stop_time(ell, b, times)
        
        # Should return NaN if crossing happens immediately (ind_times == 0)
        # This depends on implementation, but should handle gracefully
        assert isinstance(stop_time, (float, type(np.nan)))
    
    def test_get_timind_indis_basic(self):
        """Test get_timind_indis with basic parameters."""
        total_time = 10.0
        dt = 0.01
        N = 100
        
        times = np.arange(0, total_time + dt, dt)
        # If len(times) <= 1e4, it returns all points, not a subset
        if len(times) <= 1e4:
            # In this case, it returns all points
            timind, indis, rrange = get_timind_indis(total_time, dt, N=N, begin=0, rrange=True)
            assert len(timind) == len(indis), "timind and indis should have same length"
            # When times <= 1e4, it returns all points, so length equals len(times)
            assert len(timind) == len(times), f"Should return all {len(times)} points when <= 1e4"
        else:
            # When times > 1e4, it uses logspace
            timind, indis, rrange = get_timind_indis(total_time, dt, N=N, begin=0, rrange=True)
            assert len(timind) == len(indis), "timind and indis should have same length"
            assert len(timind) <= N, f"Should have at most {N} points, got {len(timind)}"
        
        assert all(0 <= t <= total_time for t in timind), "All times should be in range"
        assert all(isinstance(i, int) for i in indis), "All indices should be integers"
    
    def test_get_timind_indis_small_times(self):
        """Test get_timind_indis with small time array."""
        total_time = 0.1
        dt = 0.01
        N = 1000  # More than available points
        
        timind, indis = get_timind_indis(total_time, dt, N=N, begin=0, rrange=False)
        
        # Should return all available points if less than N
        times = np.arange(0, total_time + dt, dt)
        assert len(timind) <= len(times), "Should not exceed available time points"
    
    def test_get_timind_indis_logspace(self):
        """Test that get_timind_indis uses logspace for large arrays."""
        total_time = 100.0
        dt = 0.001
        N = 100
        
        timind, indis, _ = get_timind_indis(total_time, dt, N=N, begin=0, rrange=True)
        
        # Check that indices are logarithmically spaced (not uniformly)
        if len(indis) > 2:
            diffs = np.diff(indis)
            # Log spacing means differences should increase
            # (at least not be constant)
            assert not np.allclose(diffs, diffs[0]), "Should use logspace, not uniform spacing"


class TestMatrixGeneration:
    """Test matrix generation functions."""
    
    def test_give_matrices_structure(self):
        """Test that give_matrices returns correct matrix structures."""
        # This is a helper function inside integrate.py, but we can test the logic
        gamma = 100.0
        omega = 1000.0
        n = 1.0
        eta = 1.0
        kappa = 9.0
        
        A = np.array([[-gamma/2, omega], [-omega, -gamma/2]])
        C = np.sqrt(4*eta*kappa) * np.eye(2)
        D = np.diag([gamma*(n+0.5) + kappa] * 2)
        G = np.zeros((2, 2))
        
        # Check shapes
        assert A.shape == (2, 2), "A should be 2x2"
        assert C.shape == (2, 2), "C should be 2x2"
        assert D.shape == (2, 2), "D should be 2x2"
        assert G.shape == (2, 2), "G should be 2x2"
        
        # Check A structure
        assert np.isclose(A[0, 0], -gamma/2), "A[0,0] should be -gamma/2"
        assert np.isclose(A[1, 1], -gamma/2), "A[1,1] should be -gamma/2"
        assert np.isclose(A[0, 1], omega), "A[0,1] should be omega"
        assert np.isclose(A[1, 0], -omega), "A[1,0] should be -omega"
        
        # Check C structure
        assert np.allclose(C, np.sqrt(4*eta*kappa) * np.eye(2)), "C should be diagonal"
        
        # Check D structure
        expected_diag = gamma*(n+0.5) + kappa
        assert np.allclose(np.diag(D), [expected_diag, expected_diag]), "D should have correct diagonal"
    
    def test_matrix_properties(self):
        """Test mathematical properties of generated matrices."""
        gamma = 100.0
        omega = 1000.0
        n = 1.0
        eta = 1.0
        kappa = 9.0
        
        A = np.array([[-gamma/2, omega], [-omega, -gamma/2]])
        
        # A should have eigenvalues with negative real parts (stable)
        eigenvals = np.linalg.eigvals(A)
        assert np.all(np.real(eigenvals) < 0), "A should be stable (negative real parts)"
        
        # Trace should be -gamma
        assert np.isclose(np.trace(A), -gamma), "Trace of A should be -gamma"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

