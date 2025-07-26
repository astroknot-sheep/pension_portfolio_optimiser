"""
NPS vs UPS Portfolio Optimization Package

A comprehensive quantitative finance platform for comparing India's National Pension System (NPS)
with the Unified Pension Scheme (UPS) using modern portfolio optimization techniques.

This package provides:
- Advanced portfolio optimization with PyPortfolioOpt
- Comprehensive risk analytics (VaR, ES, stress testing)
- Monte Carlo simulation for pension corpus projections
- Economic scenario modeling with VAR/IRF analysis
- Interactive reporting and visualization tools
- CLI interface for automated analysis workflows

Designed for institutional-grade quantitative research with production-ready code quality.
"""

__version__ = "1.0.0"
__author__ = "Quantitative Research Team"
__email__ = "quant.research@example.com"

# Core imports that are safe and don't have dependency issues
from nps_ups.optimiser import PortfolioOptimizer
from nps_ups.analytics import RiskAnalytics
from nps_ups.simulation import MonteCarloSimulator

# Lazy imports for modules with heavy dependencies
def get_report_generator():
    """
    Get ReportGenerator class with lazy loading.
    
    This prevents import issues with PDF generation dependencies
    while still allowing access when needed.
    
    Returns:
        ReportGenerator: The reporting class
    """
    from nps_ups.reporting import ReportGenerator
    return ReportGenerator

def get_data_loader():
    """
    Get DataLoader class with lazy loading.
    
    Returns:
        DataLoader: The data loading class
    """
    from nps_ups.io.data_loader import DataLoader
    return DataLoader

def get_economic_data_provider():
    """
    Get EconomicDataProvider class with lazy loading.
    
    Returns:
        EconomicDataProvider: The economic data provider class
    """
    from nps_ups.io.economic_data import EconomicDataProvider
    return EconomicDataProvider

# For backwards compatibility, create ReportGenerator instance when requested
@property 
def ReportGenerator():
    """Property accessor for ReportGenerator with lazy loading."""
    return get_report_generator()

# Make lazy loaders available
__all__ = [
    "PortfolioOptimizer",
    "RiskAnalytics", 
    "MonteCarloSimulator",
    "get_report_generator",
    "get_data_loader",
    "get_economic_data_provider",
    "create_portfolio_optimizer",
    "create_risk_analytics",
    "create_monte_carlo_simulator",
    "run_quick_analysis",
    "__version__",
    "__author__",
    "__email__"
]

# Package-level convenience functions
def create_portfolio_optimizer(returns_data, **kwargs):
    """
    Create a PortfolioOptimizer instance with data.
    
    Args:
        returns_data: DataFrame of asset returns
        **kwargs: Additional arguments for PortfolioOptimizer
        
    Returns:
        PortfolioOptimizer: Configured optimizer instance
    """
    return PortfolioOptimizer(returns_data, **kwargs)

def create_risk_analytics(returns_data, **kwargs):
    """
    Create a RiskAnalytics instance with data.
    
    Args:
        returns_data: DataFrame of asset returns
        **kwargs: Additional arguments for RiskAnalytics
        
    Returns:
        RiskAnalytics: Configured analytics instance  
    """
    return RiskAnalytics(returns_data, **kwargs)

def create_monte_carlo_simulator(returns_data, **kwargs):
    """
    Create a MonteCarloSimulator instance.
    
    Args:
        returns_data: DataFrame of asset returns
        **kwargs: Additional arguments for MonteCarloSimulator
        
    Returns:
        MonteCarloSimulator: Configured simulator instance
    """
    return MonteCarloSimulator(returns_data, **kwargs)

def run_quick_analysis(returns_data, **kwargs):
    """
    Run a quick portfolio optimization analysis.
    
    Args:
        returns_data: DataFrame of asset returns
        **kwargs: Additional configuration options
        
    Returns:
        dict: Analysis results including optimal weights and metrics
    """
    # Create optimizer
    optimizer = create_portfolio_optimizer(returns_data, **kwargs)
    
    # Run optimization
    results = optimizer.optimize_max_sharpe()
    
    # Add risk analytics
    analytics = create_risk_analytics(returns_data)
    var_es = analytics.calculate_var_es(results['weights'])
    results.update(var_es)
    
    return results 