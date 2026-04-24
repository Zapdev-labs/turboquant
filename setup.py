#!/usr/bin/env python3
"""
Setup script for FastVQ package.
This enables pip install -e . for development.
"""

from setuptools import find_packages, setup

setup(
    name="fastvq",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    entry_points={
        "console_scripts": [
            "turboquant=turboquant.cli:main",
            "tq=turboquant.cli:main",
            "fastvq=fastvq.cli:main",
        ],
    },
    python_requires=">=3.8",
)
