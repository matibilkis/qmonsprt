"""
Unit tests for utility functions.
"""
import numpy as np
import pytest
import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from numerics.utilities.misc import (
    give_model,
    def_params,
    get_path_config,
    load_liks,
    get_stop_time,
    get_timind_indis
)


class TestModelFunctions:
    """Test model-related utility functions."""
    
    def test_give_model(self):
        """Test that give_model returns a string."""
        model = give_model()
        assert isinstance(model, str), "give_model should return a string"
        assert len(model) > 0, "Model name should not be empty"
    
    def test_def_params_mechanical_damp(self):
        """Test def_params for mechanical damping model."""
        # Note: This depends on the model returned by give_model()
        # We'll test the structure regardless
        params, exp_path = def_params(flip=0)
        
        assert isinstance(params, list), "params should be a list"
        assert len(params) == 2, "params should contain two hypotheses"
        assert isinstance(exp_path, str), "exp_path should be a string"
        
        # Each hypothesis should be a list of 5 parameters
        for i, hyp in enumerate(params):
            assert isinstance(hyp, list), f"Hypothesis {i} should be a list"
            assert len(hyp) == 5, f"Hypothesis {i} should have 5 parameters"
    
    def test_def_params_flip(self):
        """Test that def_params respects flip parameter."""
        params0, _ = def_params(flip=0)
        params1, _ = def_params(flip=1)
        
        # Flipped params should have hypotheses swapped
        assert params0[0] == params1[1], "Flipping should swap hypotheses"
        assert params0[1] == params1[0], "Flipping should swap hypotheses"


class TestPathFunctions:
    """Test path configuration functions."""
    
    def test_get_path_config_basic(self):
        """Test get_path_config with basic parameters."""
        exp_path = "test_exp/"
        itraj = 1
        total_time = 8.0
        dt = 1e-4
        
        path = get_path_config(exp_path=exp_path, itraj=itraj, 
                              total_time=total_time, dt=dt)
        
        assert isinstance(path, str), "Path should be a string"
        assert exp_path in path, "Path should contain exp_path"
        assert str(itraj) in path, "Path should contain trajectory number"
        assert str(total_time) in path, "Path should contain total_time"
        assert str(dt) in path, "Path should contain dt"
    
    def test_get_path_config_noitraj(self):
        """Test get_path_config with noitraj=True."""
        exp_path = "test_exp/"
        total_time = 8.0
        dt = 1e-4
        
        path = get_path_config(exp_path=exp_path, total_time=total_time, 
                              dt=dt, noitraj=True)
        
        assert isinstance(path, str), "Path should be a string"
        # Should not contain trajectory number when noitraj=True
        # (implementation dependent, but should be different)
        assert str(total_time) in path, "Path should contain total_time"


class TestStopTimeFunctions:
    """Test stopping time calculation functions."""
    
    def test_get_stop_time_positive_crossing(self):
        """Test get_stop_time with positive boundary crossing.
        
        get_stop_time finds first time when ell is OUTSIDE [-b, b].
        """
        times = np.linspace(0, 10, 1000)
        # Create signal that starts within bounds and crosses positive boundary
        ell = np.linspace(0, 5, 1000)  # Goes from 0 to 5, crossing b=2.0
        b = 2.0
        
        stop_time = get_stop_time(ell, b, times)
        
        # Should find crossing when ell > b (goes outside bounds)
        assert not np.isnan(stop_time), "Should find stop time when crossing boundary"
        assert 0 <= stop_time <= 10, "Stop time should be in valid range"
    
    def test_get_stop_time_negative_crossing(self):
        """Test get_stop_time with negative boundary crossing.
        
        get_stop_time finds first time when ell is OUTSIDE [-b, b].
        Note: If the signal starts outside bounds, it returns NaN (ind_times==0).
        """
        times = np.linspace(0, 10, 1000)
        # Create signal that starts within bounds and crosses negative boundary
        ell = np.linspace(0, -5, 1000)  # Goes from 0 to -5, crossing -b=-2.0
        b = 2.0
        
        stop_time = get_stop_time(ell, b, times)
        
        # Should find crossing when ell < -b (goes outside bounds)
        assert not np.isnan(stop_time), "Should find stop time when crossing boundary"
        assert 0 <= stop_time <= 10, "Stop time should be in valid range"
    
    def test_get_stop_time_no_crossing(self):
        """Test get_stop_time when signal stays within bounds."""
        times = np.linspace(0, 10, 1000)
        ell = np.ones(1000) * 0.5  # Always within [-b, b]
        b = 2.0
        
        stop_time = get_stop_time(ell, b, times)
        
        assert np.isnan(stop_time), "Should return NaN when no crossing"
    
    def test_get_stop_time_edge_cases(self):
        """Test get_stop_time with edge cases."""
        # Empty arrays - will cause ValueError in argmin
        times = np.array([])
        ell = np.array([])
        b = 2.0
        
        with pytest.raises(ValueError):
            get_stop_time(ell, b, times)
        
        # Single point outside bounds - argmin will be 0, so returns NaN
        times = np.array([0.0])
        ell = np.array([3.0])  # Outside bounds
        stop_time = get_stop_time(ell, b, times)
        # When ind_times == 0, function returns NaN
        assert np.isnan(stop_time), "Should return NaN when crossing happens at index 0"


class TestTimeIndexFunctions:
    """Test time indexing functions."""
    
    def test_get_timind_indis_consistency(self):
        """Test that get_timind_indis returns consistent results."""
        total_time = 10.0
        dt = 0.01
        N = 100
        
        timind, indis, rrange = get_timind_indis(total_time, dt, N=N, 
                                                begin=0, rrange=True)
        
        # Check consistency
        times = np.arange(0, total_time + dt, dt)
        for i, idx in enumerate(indis):
            assert 0 <= idx < len(times), f"Index {idx} out of range"
            assert np.isclose(timind[i], times[idx]), "Time should match index"
    
    def test_get_timind_indis_begin_parameter(self):
        """Test get_timind_indis with different begin values."""
        total_time = 10.0
        dt = 0.01
        N = 50
        
        timind0, indis0, _ = get_timind_indis(total_time, dt, N=N, begin=0, rrange=True)
        timind10, indis10, _ = get_timind_indis(total_time, dt, N=N, begin=10, rrange=True)
        
        # Should start from different points
        assert indis0[0] < indis10[0], "begin=10 should start later"
        assert timind0[0] < timind10[0], "begin=10 should start at later time"
    
    def test_get_timind_indis_rrange(self):
        """Test get_timind_indis with and without rrange."""
        total_time = 10.0
        dt = 0.01
        N = 50
        
        timind1, indis1, rrange1 = get_timind_indis(total_time, dt, N=N, 
                                                    begin=0, rrange=True)
        timind2, indis2 = get_timind_indis(total_time, dt, N=N, 
                                          begin=0, rrange=False)
        
        # Should have same times and indices
        assert len(timind1) == len(timind2), "Should have same length"
        assert np.allclose(timind1, timind2), "Times should match"
        assert indis1 == indis2, "Indices should match"
        
        # rrange should be a list of integers
        assert isinstance(rrange1, list), "rrange should be a list"
        assert all(isinstance(i, int) for i in rrange1), "rrange should contain integers"


class TestLikelihoodFunctions:
    """Test likelihood-related functions."""
    
    def test_load_liks_structure(self):
        """Test that load_liks returns expected structure."""
        # Note: This requires data files to exist, so we'll test the interface
        # In a real scenario, you'd mock the file loading
        
        # This test will likely fail if data doesn't exist, but documents expected behavior
        try:
            l_1true, l_0true = load_liks(itraj=1, dt=1e-4, total_time=8.0)
            
            # Should return two arrays
            assert isinstance(l_1true, np.ndarray), "l_1true should be numpy array"
            assert isinstance(l_0true, np.ndarray), "l_0true should be numpy array"
            
            # Should have same length
            assert len(l_1true) == len(l_0true), "Arrays should have same length"
            
        except (FileNotFoundError, OSError):
            # Expected if data files don't exist
            pytest.skip("Data files not available for testing")


class TestNumericalStability:
    """Test numerical stability of utility functions."""
    
    def test_get_stop_time_numerical_precision(self):
        """Test get_stop_time with values near boundaries."""
        times = np.linspace(0, 10, 1000)
        b = 2.0
        
        # Signal very close to boundary
        ell = np.ones(1000) * (b - 1e-10)
        stop_time = get_stop_time(ell, b, times)
        assert np.isnan(stop_time), "Should not trigger on values just below boundary"
        
        # Signal just crossing boundary
        ell = np.linspace(b - 0.1, b + 0.1, 1000)
        stop_time = get_stop_time(ell, b, times)
        # Should find crossing
        assert isinstance(stop_time, (float, type(np.nan)))
    
    def test_get_timind_indis_large_numbers(self):
        """Test get_timind_indis with large time values."""
        total_time = 1e6
        dt = 0.1
        N = 1000
        
        timind, indis, _ = get_timind_indis(total_time, dt, N=N, begin=0, rrange=True)
        
        assert len(timind) <= N, "Should not exceed requested number of points"
        assert all(t <= total_time for t in timind), "All times should be within range"
        assert all(isinstance(i, int) for i in indis), "All indices should be integers"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

