"""
Monte Carlo simulation module for pension corpus and income forecasting.

Implements:
- Salary growth and contribution modeling
- Portfolio return simulation under multiple scenarios
- NPS vs UPS comparison analysis
- Life-cycle adjustments and rebalancing
- Inflation-adjusted pension income calculations
- Sensitivity analysis and confidence intervals
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class SimulationParameters:
    """Parameters for Monte Carlo simulation."""
    # Employee parameters
    current_age: int = 25
    retirement_age: int = 60
    life_expectancy: int = 75
    current_salary: float = 1_000_000  # Annual salary in INR
    
    # Contribution parameters
    employee_contribution_rate: float = 0.10  # 10% of salary
    employer_contribution_rate: float = 0.10  # 10% of salary (for NPS)
    
    # Economic parameters
    salary_growth_rate: float = 0.08  # 8% annual salary growth
    inflation_rate: float = 0.04  # 4% inflation
    
    # UPS parameters
    ups_pension_rate: float = 0.50  # 50% of last drawn salary
    ups_assured_return: float = 0.075  # 7.5% assured return
    
    # Simulation parameters
    n_simulations: int = 10_000
    rebalance_frequency: int = 12  # Rebalance annually
    random_seed: int = 42


class TemporalFusionTransformerStub(nn.Module):
    """
    Lightweight TFT stub for scenario return generation.
    
    This is a simplified implementation for demonstration.
    In production, would use full TFT with attention mechanisms.
    """
    
    def __init__(self, input_size: int = 5, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """Forward pass through TFT stub."""
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.dropout(attn_out)
        return self.output(out[:, -1, :])  # Return last timestep


class MonteCarloSimulator:
    """
    Comprehensive Monte Carlo simulator for pension analysis.
    
    Simulates:
    - Multiple economic scenarios (base, optimistic, adverse)
    - Stochastic portfolio returns with correlation
    - Salary progression and contribution accumulation
    - Lifecycle rebalancing and withdrawal phases
    - NPS vs UPS outcome comparison
    """
    
    def __init__(
        self,
        returns_data: pd.DataFrame,
        parameters: Optional[SimulationParameters] = None,
        portfolios: Optional[Dict[str, pd.Series]] = None
    ):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            returns_data: Historical returns for assets
            parameters: Simulation parameters
            portfolios: Dictionary of portfolio weights to simulate
        """
        self.returns_data = returns_data.copy()
        self.parameters = parameters or SimulationParameters()
        self.portfolios = portfolios or {}
        
        # Precompute statistics
        self._compute_return_statistics()
        
        # Initialize TFT stub
        self.tft_model = TemporalFusionTransformerStub()
        self._fit_tft_stub()
        
        # Simulation results storage
        self.simulation_results = {}
    
    def _compute_return_statistics(self):
        """Compute return statistics for simulation."""
        self.mean_returns = self.returns_data.mean() * 252  # Annualized
        self.cov_matrix = self.returns_data.cov() * 252  # Annualized
        self.correlation_matrix = self.returns_data.corr()
        
        logger.info(f"Computed return statistics for {len(self.returns_data.columns)} assets")
        logger.info(f"Mean annual return range: {self.mean_returns.min():.2%} - {self.mean_returns.max():.2%}")
    
    def _fit_tft_stub(self):
        """Fit TFT stub for scenario generation (simplified)."""
        # This is a placeholder implementation
        # In production, would train on macroeconomic data
        logger.info("TFT stub initialized for scenario generation")
    
    def generate_scenario_returns(
        self,
        scenario: str = 'base',
        time_horizon: int = 35
    ) -> Dict[str, np.ndarray]:
        """
        Generate scenario-specific return paths using TFT stub.
        
        Args:
            scenario: Scenario type ('base', 'optimistic', 'adverse')
            time_horizon: Number of years to simulate
            
        Returns:
            Dictionary with return paths for each asset
        """
        # Scenario-specific adjustments
        scenario_adjustments = {
            'base': {'mean_adj': 0.0, 'vol_adj': 1.0},
            'optimistic': {'mean_adj': 0.02, 'vol_adj': 0.8},  # +2% return, -20% vol
            'adverse': {'mean_adj': -0.03, 'vol_adj': 1.4}     # -3% return, +40% vol
        }
        
        adj = scenario_adjustments.get(scenario, scenario_adjustments['base'])
        
        # Adjusted parameters
        adjusted_means = self.mean_returns + adj['mean_adj']
        adjusted_cov = self.cov_matrix * (adj['vol_adj'] ** 2)
        
        # Generate correlated return paths
        np.random.seed(self.parameters.random_seed + hash(scenario) % 1000)
        
        return_paths = {}
        n_assets = len(self.returns_data.columns)
        
        for asset_idx, asset in enumerate(self.returns_data.columns):
            # Base trend
            base_trend = np.full(time_horizon, adjusted_means.iloc[asset_idx])
            
            # Add mean reversion and volatility clustering
            returns = []
            current_return = adjusted_means.iloc[asset_idx]
            volatility = np.sqrt(adjusted_cov.iloc[asset_idx, asset_idx])
            
            for year in range(time_horizon):
                # Mean reversion towards long-term average
                mean_reversion = 0.1 * (adjusted_means.iloc[asset_idx] - current_return)
                
                # Volatility clustering (GARCH-like)
                vol_shock = np.random.normal(0, volatility * 0.1)
                volatility = max(0.01, volatility + vol_shock)
                
                # Random shock
                random_shock = np.random.normal(0, volatility)
                
                # TFT-based adjustment (simplified)
                tft_adjustment = self._get_tft_adjustment(scenario, year, time_horizon)
                
                # Combined return
                annual_return = current_return + mean_reversion + random_shock + tft_adjustment
                returns.append(annual_return)
                current_return = annual_return
            
            return_paths[asset] = np.array(returns)
        
        return return_paths
    
    def _get_tft_adjustment(self, scenario: str, year: int, total_years: int) -> float:
        """Get TFT-based scenario adjustment (simplified)."""
        # Declining impact over time
        time_factor = 1 - (year / total_years)
        
        scenario_impacts = {
            'base': 0.0,
            'optimistic': 0.005 * time_factor,  # Gradual improvement
            'adverse': -0.01 * time_factor       # Gradual deterioration
        }
        
        return scenario_impacts.get(scenario, 0.0)
    
    def simulate_salary_growth(self, n_simulations: int = None) -> np.ndarray:
        """
        Simulate salary growth paths with stochastic components.
        
        Args:
            n_simulations: Number of simulation paths
            
        Returns:
            Array of salary paths [simulations x years]
        """
        if n_simulations is None:
            n_simulations = self.parameters.n_simulations
        
        years = self.parameters.retirement_age - self.parameters.current_age
        
        # Salary growth with random shocks
        np.random.seed(self.parameters.random_seed)
        
        # Base growth rate with random variation
        growth_shocks = np.random.normal(
            self.parameters.salary_growth_rate,
            0.02,  # 2% standard deviation
            size=(n_simulations, years)
        )
        
        # Ensure non-negative growth rates
        growth_shocks = np.maximum(growth_shocks, -0.05)  # Max -5% salary decline
        
        # Calculate cumulative salary paths
        salary_paths = np.zeros((n_simulations, years + 1))
        salary_paths[:, 0] = self.parameters.current_salary
        
        for year in range(years):
            salary_paths[:, year + 1] = salary_paths[:, year] * (1 + growth_shocks[:, year])
        
        return salary_paths
    
    def simulate_nps_accumulation(
        self,
        portfolio_weights: pd.Series,
        scenario: str = 'base',
        n_simulations: int = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate NPS corpus accumulation with lifecycle rebalancing.
        
        Args:
            portfolio_weights: Initial portfolio weights
            scenario: Economic scenario
            n_simulations: Number of simulations
            
        Returns:
            Dictionary with corpus accumulation paths and final values
        """
        if n_simulations is None:
            n_simulations = self.parameters.n_simulations
        
        years = self.parameters.retirement_age - self.parameters.current_age
        
        # Generate return and salary scenarios
        return_paths = self.generate_scenario_returns(scenario, years)
        salary_paths = self.simulate_salary_growth(n_simulations)
        
        # Initialize arrays
        corpus_paths = np.zeros((n_simulations, years + 1))
        contribution_paths = np.zeros((n_simulations, years))
        
        for sim in range(n_simulations):
            current_corpus = 0.0
            current_weights = portfolio_weights.copy()
            
            for year in range(years):
                # Calculate annual contribution
                annual_salary = salary_paths[sim, year]
                employee_contrib = annual_salary * self.parameters.employee_contribution_rate
                employer_contrib = annual_salary * self.parameters.employer_contribution_rate
                total_contribution = employee_contrib + employer_contrib
                contribution_paths[sim, year] = total_contribution
                
                # Add contribution to corpus
                current_corpus += total_contribution
                
                # Apply investment returns
                portfolio_return = sum(
                    current_weights[asset] * return_paths[asset][year]
                    for asset in current_weights.index
                    if asset in return_paths
                )
                current_corpus *= (1 + portfolio_return)
                
                # Lifecycle rebalancing (annually)
                current_age = self.parameters.current_age + year
                if year % self.parameters.rebalance_frequency == 0:
                    current_weights = self._lifecycle_rebalance(current_age, portfolio_weights)
                
                corpus_paths[sim, year + 1] = current_corpus
        
        final_corpus = corpus_paths[:, -1]
        
        return {
            'corpus_paths': corpus_paths,
            'contribution_paths': contribution_paths,
            'final_corpus': final_corpus,
            'salary_paths': salary_paths,
            'scenario': scenario,
            'portfolio_weights': portfolio_weights
        }
    
    def _lifecycle_rebalance(
        self,
        current_age: int,
        base_weights: pd.Series
    ) -> pd.Series:
        """
        Perform lifecycle rebalancing based on age.
        
        Args:
            current_age: Current age for rebalancing
            base_weights: Base portfolio weights
            
        Returns:
            Rebalanced portfolio weights
        """
        # Equity percentage based on age (100 - age rule with adjustments)
        target_equity_pct = max(0.20, min(0.80, (100 - current_age) / 100))
        
        # Adjust weights based on asset types
        rebalanced_weights = base_weights.copy()
        
        # Identify equity and fixed income assets
        equity_assets = [asset for asset in base_weights.index if 'E' in asset]
        bond_assets = [asset for asset in base_weights.index if 'C' in asset or 'G' in asset]
        
        if equity_assets and bond_assets:
            # Calculate current allocations
            current_equity_weight = sum(rebalanced_weights[asset] for asset in equity_assets)
            current_bond_weight = sum(rebalanced_weights[asset] for asset in bond_assets)
            
            # Adjust towards target
            if current_equity_weight > 0 and current_bond_weight > 0:
                equity_adjustment = target_equity_pct / current_equity_weight
                bond_adjustment = (1 - target_equity_pct) / current_bond_weight
                
                # Apply adjustments
                for asset in equity_assets:
                    rebalanced_weights[asset] *= equity_adjustment
                for asset in bond_assets:
                    rebalanced_weights[asset] *= bond_adjustment
                
                # Normalize weights
                rebalanced_weights = rebalanced_weights / rebalanced_weights.sum()
        
        return rebalanced_weights
    
    def simulate_ups_benefits(self, n_simulations: int = None) -> Dict[str, np.ndarray]:
        """
        Simulate UPS pension benefits.
        
        Args:
            n_simulations: Number of simulations
            
        Returns:
            Dictionary with UPS benefit calculations
        """
        if n_simulations is None:
            n_simulations = self.parameters.n_simulations
        
        # Generate salary paths
        salary_paths = self.simulate_salary_growth(n_simulations)
        
        # UPS benefits are based on last drawn salary
        final_salaries = salary_paths[:, -1]
        
        # Monthly pension = 50% of last drawn salary / 12
        monthly_pension = final_salaries * self.parameters.ups_pension_rate / 12
        
        # Calculate equivalent corpus needed to generate this pension
        # Using annuity formula with life expectancy
        years_in_retirement = self.parameters.life_expectancy - self.parameters.retirement_age
        discount_rate = self.parameters.ups_assured_return
        
        # Present value of annuity
        annuity_factor = (1 - (1 + discount_rate) ** (-years_in_retirement)) / discount_rate
        equivalent_corpus = monthly_pension * 12 * annuity_factor
        
        return {
            'final_salaries': final_salaries,
            'monthly_pension': monthly_pension,
            'annual_pension': monthly_pension * 12,
            'equivalent_corpus': equivalent_corpus,
            'years_in_retirement': years_in_retirement
        }
    
    def run_comprehensive_simulation(
        self,
        scenarios: List[str] = ['base', 'optimistic', 'adverse']
    ) -> Dict[str, Dict]:
        """
        Run comprehensive simulation across multiple scenarios and portfolios.
        
        Args:
            scenarios: List of economic scenarios to simulate
            
        Returns:
            Dictionary with simulation results for all scenarios and portfolios
        """
        logger.info(f"Running comprehensive simulation with {self.parameters.n_simulations} paths")
        
        results = {
            'parameters': self.parameters,
            'scenarios': {},
            'ups_baseline': self.simulate_ups_benefits()
        }
        
        # Run simulations for each scenario and portfolio
        for scenario in scenarios:
            logger.info(f"Simulating scenario: {scenario}")
            scenario_results = {'portfolios': {}}
            
            for portfolio_name, weights in self.portfolios.items():
                logger.info(f"  Portfolio: {portfolio_name}")
                
                nps_results = self.simulate_nps_accumulation(
                    weights, scenario, self.parameters.n_simulations
                )
                scenario_results['portfolios'][portfolio_name] = nps_results
            
            results['scenarios'][scenario] = scenario_results
        
        # Calculate summary statistics
        results['summary'] = self._calculate_summary_statistics(results)
        
        self.simulation_results = results
        return results
    
    def _calculate_summary_statistics(self, results: Dict) -> Dict:
        """Calculate summary statistics across simulations."""
        summary = {
            'ups_stats': {},
            'nps_stats': {},
            'comparison': {}
        }
        
        # UPS statistics
        ups_data = results['ups_baseline']
        summary['ups_stats'] = {
            'mean_equivalent_corpus': np.mean(ups_data['equivalent_corpus']),
            'median_equivalent_corpus': np.median(ups_data['equivalent_corpus']),
            'std_equivalent_corpus': np.std(ups_data['equivalent_corpus']),
            'mean_annual_pension': np.mean(ups_data['annual_pension']),
            'median_annual_pension': np.median(ups_data['annual_pension'])
        }
        
        # NPS statistics by scenario and portfolio
        for scenario, scenario_data in results['scenarios'].items():
            summary['nps_stats'][scenario] = {}
            
            for portfolio_name, portfolio_data in scenario_data['portfolios'].items():
                final_corpus = portfolio_data['final_corpus']
                
                # Calculate pension income assuming 4% withdrawal rate
                annual_pension = final_corpus * 0.04
                
                portfolio_stats = {
                    'mean_final_corpus': np.mean(final_corpus),
                    'median_final_corpus': np.median(final_corpus),
                    'std_final_corpus': np.std(final_corpus),
                    'percentile_10': np.percentile(final_corpus, 10),
                    'percentile_25': np.percentile(final_corpus, 25),
                    'percentile_75': np.percentile(final_corpus, 75),
                    'percentile_90': np.percentile(final_corpus, 90),
                    'mean_annual_pension': np.mean(annual_pension),
                    'median_annual_pension': np.median(annual_pension),
                    'probability_beats_ups': np.mean(
                        final_corpus > ups_data['equivalent_corpus']
                    )
                }
                
                summary['nps_stats'][scenario][portfolio_name] = portfolio_stats
        
        return summary
    
    def calculate_sensitivity_analysis(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        base_portfolio: str,
        scenario: str = 'base',
        n_samples: int = 100
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on key parameters.
        
        Args:
            parameter_ranges: Dictionary of parameter names and their ranges
            base_portfolio: Base portfolio for analysis
            scenario: Scenario to analyze
            n_samples: Number of samples for each parameter
            
        Returns:
            DataFrame with sensitivity analysis results
        """
        if base_portfolio not in self.portfolios:
            raise ValueError(f"Portfolio {base_portfolio} not found")
        
        sensitivity_results = []
        base_weights = self.portfolios[base_portfolio]
        
        for param_name, (min_val, max_val) in parameter_ranges.items():
            logger.info(f"Analyzing sensitivity to {param_name}")
            
            param_values = np.linspace(min_val, max_val, n_samples)
            
            for param_value in param_values:
                # Temporarily modify parameter
                original_value = getattr(self.parameters, param_name)
                setattr(self.parameters, param_name, param_value)
                
                # Run limited simulation
                nps_results = self.simulate_nps_accumulation(
                    base_weights, scenario, n_simulations=1000
                )
                
                # Store results
                sensitivity_results.append({
                    'parameter': param_name,
                    'parameter_value': param_value,
                    'mean_final_corpus': np.mean(nps_results['final_corpus']),
                    'median_final_corpus': np.median(nps_results['final_corpus']),
                    'std_final_corpus': np.std(nps_results['final_corpus'])
                })
                
                # Restore original parameter
                setattr(self.parameters, param_name, original_value)
        
        return pd.DataFrame(sensitivity_results)
    
    def export_simulation_results(self, filename: str = "simulation_results.xlsx") -> None:
        """Export simulation results to Excel file."""
        if not self.simulation_results:
            logger.warning("No simulation results to export. Run simulation first.")
            return
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary statistics
            summary_df = pd.DataFrame(self.simulation_results['summary']['nps_stats'])
            summary_df.to_excel(writer, sheet_name='NPS_Summary')
            
            # UPS statistics
            ups_stats = pd.DataFrame([self.simulation_results['summary']['ups_stats']])
            ups_stats.to_excel(writer, sheet_name='UPS_Summary')
            
            # Detailed results for each scenario
            for scenario, scenario_data in self.simulation_results['scenarios'].items():
                for portfolio_name, portfolio_data in scenario_data['portfolios'].items():
                    sheet_name = f"{scenario}_{portfolio_name}"[:31]  # Excel limit
                    
                    # Create summary DataFrame
                    results_df = pd.DataFrame({
                        'Final_Corpus': portfolio_data['final_corpus'],
                        'Total_Contributions': portfolio_data['contribution_paths'].sum(axis=1)
                    })
                    
                    results_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"Simulation results exported to {filename}")


# Convenience function for quick simulation
def run_quick_simulation(
    returns_data: pd.DataFrame,
    portfolios: Dict[str, pd.Series],
    n_simulations: int = 1000
) -> Dict:
    """Quick simulation runner for testing."""
    params = SimulationParameters(n_simulations=n_simulations)
    simulator = MonteCarloSimulator(returns_data, params, portfolios)
    return simulator.run_comprehensive_simulation() 