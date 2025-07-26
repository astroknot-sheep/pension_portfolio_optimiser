"""
NPS vs UPS Portfolio Optimization Package

A comprehensive quantitative finance package for comparing India's National Pension System (NPS)
with the Unified Pension Scheme (UPS) using modern portfolio theory and risk analytics.

Designed for institutional-grade research workflows including:
- Portfolio optimization with PyPortfolioOpt
- Risk analysis and stress testing
- Monte Carlo simulation
- Automated reporting and visualization
"""

__version__ = "1.0.0"
__author__ = "Quantitative Research Team"
__email__ = "quant.research@example.com"

# Core imports
from nps_ups.optimiser import PortfolioOptimizer
from nps_ups.analytics import RiskAnalytics
from nps_ups.simulation import MonteCarloSimulator
from nps_ups.reporting import ReportGenerator

__all__ = [
    "PortfolioOptimizer",
    "RiskAnalytics", 
    "MonteCarloSimulator",
    "ReportGenerator",
] 