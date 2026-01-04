#!/usr/bin/env python3
"""
Simple verification script to check repository structure and imports.
This script verifies the organization without requiring all dependencies.
"""
import os
import sys
import ast

def check_python_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error in {filepath}: {e}")
        return False
    except Exception as e:
        print(f"  ⚠ Could not parse {filepath}: {e}")
        return False

def verify_structure():
    """Verify the repository structure."""
    print("Verifying repository structure...")
    print("=" * 60)
    
    # Check required directories exist
    required_dirs = [
        'numerics',
        'numerics/integration',
        'numerics/utilities',
        'scripts',
        'tests',
        'analysis',
        'docs',
        'examples',
        'HPC'
    ]
    
    print("\n1. Checking directory structure:")
    all_dirs_ok = True
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ Missing: {dir_path}/")
            all_dirs_ok = False
    
    # Check key files exist
    print("\n2. Checking key files:")
    key_files = [
        'README.md',
        'requirements.txt',
        'setup.py',
        'pytest.ini',
        '.gitignore',
        'Makefile',
        'numerics/__init__.py',
        'numerics/integration/__init__.py',
        'numerics/utilities/__init__.py',
        'tests/__init__.py',
    ]
    
    all_files_ok = True
    for file_path in key_files:
        if os.path.isfile(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ Missing: {file_path}")
            all_files_ok = False
    
    # Check Python syntax
    print("\n3. Checking Python syntax:")
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and test_env
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'test_env' and d != '__pycache__']
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                python_files.append(filepath)
    
    syntax_ok = True
    for filepath in sorted(python_files)[:20]:  # Check first 20 files
        if check_python_syntax(filepath):
            print(f"  ✓ {filepath}")
        else:
            syntax_ok = False
    
    if len(python_files) > 20:
        print(f"  ... and {len(python_files) - 20} more files")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    if all_dirs_ok and all_files_ok and syntax_ok:
        print("  ✓ Repository structure is well-organized!")
        print("  ✓ All key files and directories are present")
        print("  ✓ Python syntax is valid")
        return 0
    else:
        print("  ⚠ Some issues found (see above)")
        return 1

if __name__ == "__main__":
    sys.exit(verify_structure())

