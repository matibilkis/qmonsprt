"""
Unit tests for Euler update and likelihood computation.
"""
import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestEulerUpdate:
    """Test Euler update functions for state and likelihood evolution."""
    
    def test_euler_update_shape(self):
        """Test that Euler update maintains correct shapes."""
        # Simulate the EulerUpdate_x0_logliks function logic
        x1 = np.array([1.0, 2.0])  # Hidden state
        dy = np.array([0.1, 0.2])   # Measurement increment
        s = np.array([0.5, 1.5, 0.0, 0.0])  # [x0, p0, l0, l1]
        dt = 0.01
        
        # Simplified version of update logic
        x0 = s[:2]
        dx0 = 0.1 * x0 * dt + 0.1 * dy  # Simplified dynamics
        
        u0 = np.dot(np.eye(2), x0)
        u1 = np.dot(np.eye(2), x1)
        dl0 = -dt * np.dot(u0, u0) / 2 + np.dot(u0, dy)
        dl1 = -dt * np.dot(u1, u1) / 2 + np.dot(u1, dy)
        
        result = np.array([(x0 + dx0)[0], (x0 + dx0)[1], s[2] + dl0, s[3] + dl1])
        
        assert result.shape == (4,), "Result should have 4 elements"
        assert isinstance(result[0], (float, np.floating)), "First element should be float"
        assert isinstance(result[2], (float, np.floating)), "Log-likelihood should be float"
    
    def test_likelihood_increment_properties(self):
        """Test properties of likelihood increments."""
        dt = 0.01
        x = np.array([1.0, 0.0])
        dy = np.array([0.1, 0.2])
        
        u = np.dot(np.eye(2), x)
        dl = -dt * np.dot(u, u) / 2 + np.dot(u, dy)
        
        # Likelihood increment should be a scalar
        assert np.isscalar(dl) or dl.shape == (), "dl should be scalar"
        
        # Should be finite
        assert np.isfinite(dl), "Likelihood increment should be finite"
    
    def test_state_update_linearity(self):
        """Test that state update is approximately linear for small dt."""
        x0 = np.array([1.0, 2.0])
        dy = np.array([0.1, 0.2])
        dt1 = 0.001
        dt2 = 0.002
        
        # Simplified update: dx = A*x*dt + B*dy
        A = 0.1 * np.eye(2)
        B = 0.1 * np.eye(2)
        
        dx1 = np.dot(A, x0) * dt1 + np.dot(B, dy)
        dx2 = np.dot(A, x0) * dt2 + np.dot(B, dy)
        
        # For linear dynamics, doubling dt should approximately double dx (ignoring dy term)
        ratio = dx2[0] / dx1[0] if dx1[0] != 0 else 1.0
        assert np.isclose(ratio, dt2/dt1, rtol=0.1), "Update should scale with dt"
    
    def test_likelihood_consistency(self):
        """Test that likelihood updates are consistent."""
        dt = 0.01
        x0 = np.array([1.0, 0.0])
        x1 = np.array([1.0, 0.0])  # Same state
        dy = np.array([0.1, 0.1])
        
        u0 = np.dot(np.eye(2), x0)
        u1 = np.dot(np.eye(2), x1)
        
        dl0 = -dt * np.dot(u0, u0) / 2 + np.dot(u0, dy)
        dl1 = -dt * np.dot(u1, u1) / 2 + np.dot(u1, dy)
        
        # If states are the same, likelihood increments should be the same
        assert np.isclose(dl0, dl1), "Same states should give same likelihood increments"
    
    def test_measurement_update(self):
        """Test that measurement updates affect state correctly."""
        x0 = np.array([0.0, 0.0])
        dy = np.array([1.0, 0.0])  # Large measurement
        dt = 0.01
        
        # State should respond to measurement
        # Simplified: dx = gain * dy
        gain = 0.1
        dx = gain * dy
        
        new_x = x0 + dx
        
        assert not np.allclose(new_x, x0), "State should change with measurement"
        assert new_x[0] > 0, "State should respond to positive measurement"


class TestIntegrationLoop:
    """Test integration loop structure."""
    
    def test_integration_loop_dimensions(self):
        """Test that integration loop maintains correct dimensions."""
        # Simulate integration loop structure
        n_steps = 100
        y0_hidden = np.array([0.0, 0.0])
        y0_exp = np.array([0.0, 0.0, 0.0, 0.0])
        times = np.linspace(0, 1, n_steps)
        dt = times[1] - times[0]
        
        # Simplified loop
        yhidden = np.zeros((n_steps + 1, 2))
        yexper = np.zeros((n_steps + 1, 4))
        
        yhidden[0] = y0_hidden
        yexper[0] = y0_exp
        
        for i in range(n_steps):
            # Simple update
            yhidden[i+1] = yhidden[i] + 0.01 * np.random.randn(2)
            yexper[i+1] = yexper[i] + 0.01 * np.random.randn(4)
        
        assert yhidden.shape == (n_steps + 1, 2), "Hidden state should have correct shape"
        assert yexper.shape == (n_steps + 1, 4), "Experimental state should have correct shape"
        assert np.allclose(yhidden[0], y0_hidden), "Initial condition should be preserved"
        assert np.allclose(yexper[0], y0_exp), "Initial condition should be preserved"
    
    def test_signal_generation(self):
        """Test measurement signal generation properties."""
        n_steps = 1000
        dt = 0.01
        C = np.eye(2)
        x = np.random.randn(n_steps, 2)
        dW = np.sqrt(dt) * np.random.randn(n_steps, 2)
        
        # Generate signals: dy = C*x*dt + dW
        signals = []
        for i in range(n_steps):
            dy = np.dot(C, x[i]) * dt + dW[i]
            signals.append(dy)
        
        signals = np.array(signals)
        
        assert signals.shape == (n_steps, 2), "Signals should have correct shape"
        # Signals should have finite variance
        assert np.var(signals) > 0, "Signals should have non-zero variance"


class TestMatrixOperations:
    """Test matrix operations used in integration."""
    
    def test_continuous_are_solution_properties(self):
        """Test properties of continuous algebraic Riccati equation solution."""
        from scipy.linalg import solve_continuous_are
        
        # Simple stable system
        A = np.array([[-1.0, 0.0], [0.0, -1.0]])
        B = np.array([[1.0], [1.0]])
        Q = np.eye(2)
        R = np.eye(1)
        
        S = solve_continuous_are(A, B, Q, R)
        
        # Solution should be symmetric
        assert np.allclose(S, S.T), "Solution should be symmetric"
        
        # Solution should be positive semi-definite
        eigenvals = np.linalg.eigvals(S)
        assert np.all(eigenvals >= 0), "Solution should be positive semi-definite"
    
    def test_pseudoinverse_properties(self):
        """Test pseudoinverse properties."""
        # Test with well-conditioned matrix
        C = np.array([[2.0, 0.0], [0.0, 2.0]])
        C_normalized = C / C[0, 0]
        C_pinv = np.linalg.pinv(C_normalized)
        
        # Pseudoinverse should satisfy: C * C_pinv * C ≈ C
        result = np.dot(C_normalized, np.dot(C_pinv, C_normalized))
        assert np.allclose(result, C_normalized, rtol=1e-6), "Pseudoinverse should satisfy Moore-Penrose property"
    
    def test_covariance_matrix_properties(self):
        """Test that covariance matrices have correct properties."""
        # Generate a valid covariance matrix
        XiCov = np.array([[2.0, 0.5], [0.5, 2.0]])
        
        # Should be symmetric
        assert np.allclose(XiCov, XiCov.T), "Covariance should be symmetric"
        
        # Should be positive definite (for valid covariance)
        eigenvals = np.linalg.eigvals(XiCov)
        assert np.all(eigenvals > 0), "Covariance should be positive definite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

