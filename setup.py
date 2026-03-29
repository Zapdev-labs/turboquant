#!/usr/bin/env python3
"""
Setup script for TurboQuant package.
This enables pip install -e . for development.
"""

from setuptools import setup, find_packages

setup(
    name="turboquant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    entry_points={
        "console_scripts": [
            "turboquant=turboquant.cli:main",
            "tq=turboquant.cli:main",
        ],
    },
    python_requires=">=3.8",
)
