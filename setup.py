"""
Setup script for qmonsprt package.
"""
from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

setup(
    name="qmonsprt",
    version="1.0.0",
    author="Matias Bilkis, Giulio Gasbarri, Elisabet Roda-Salichs, John Calsamiglia",
    author_email="",
    description="Sequential hypothesis testing for continuously-monitored quantum systems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/matibilkis/qmonsprt",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0,<2.0.0",
        "scipy>=1.5.0",
        "numba>=0.50.0",
        "tqdm>=4.50.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
        ],
    },
)

