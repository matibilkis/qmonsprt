#!/usr/bin/env python3
"""
Simple test runner that can be used to verify tests without pytest.
This is a fallback for environments where pytest is not available.
"""
import sys
import os
import importlib.util

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_test_file(test_file):
    """Run a test file and report results."""
    print(f"\n{'='*60}")
    print(f"Running tests from {test_file}")
    print(f"{'='*60}")
    
    try:
        spec = importlib.util.spec_from_file_location("test_module", test_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✓ {test_file} loaded successfully")
        return True
    except Exception as e:
        print(f"✗ {test_file} failed to load: {e}")
        return False


def main():
    """Run all test files."""
    test_dir = os.path.dirname(__file__)
    test_files = [
        "test_integration.py",
        "test_utilities.py",
        "test_euler_update.py",
    ]
    
    results = []
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        if os.path.exists(test_path):
            success = run_test_file(test_path)
            results.append((test_file, success))
        else:
            print(f"✗ {test_file} not found")
            results.append((test_file, False))
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for test_file, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_file}")
    
    all_passed = all(success for _, success in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

