"""
Setup configuration for NPS vs UPS Portfolio Optimization package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nps-ups-portfolio-optimization",
    version="1.0.0",
    author="Quantitative Research Team",
    author_email="quant.research@example.com",
    description="Comprehensive portfolio optimization comparing India's NPS vs UPS pension schemes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/nps-ups-portfolio-optimization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "ruff>=0.0.287",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.2.0",
            "mkdocstrings[python]>=0.22.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nps-ups=nps_ups.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "nps_ups": ["templates/*.html", "data/*.csv"],
    },
    keywords=[
        "finance", "portfolio-optimization", "pension", "nps", "ups", 
        "quantitative-finance", "risk-management", "monte-carlo",
        "efficient-frontier", "pypfopt", "india"
    ],
) 