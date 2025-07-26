"""
Portfolio optimization module using PyPortfolioOpt for NPS/UPS analysis.

Implements:
- Mean-variance optimization with Markowitz efficient frontier
- Maximum Sharpe ratio portfolios
- Risk parity and minimum volatility portfolios  
- Custom lifecycle allocation strategies
- Scenario-based robust optimization
- CVaR (Conditional Value at Risk) optimization
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, CLA, EfficientCVaR
from pypfopt import risk_models, expected_returns
from pypfopt.objective_functions import L2_reg
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from scipy.optimize import minimize
import cvxpy as cp

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class PortfolioOptimizer:
    """
    Comprehensive portfolio optimization engine for pension fund analysis.
    
    Supports multiple optimization methods:
    - Mean-variance optimization
    - Maximum Sharpe ratio
    - Minimum volatility  
    - Risk parity
    - CVaR optimization
    - Lifecycle allocation strategies
    """
    
    # NPS allocation constraints based on PFRDA guidelines
    NPS_CONSTRAINTS = {
        'Aggressive': {'E': (0.50, 0.75), 'C': (0.15, 0.35), 'G': (0.10, 0.25)},
        'Moderate': {'E': (0.25, 0.50), 'C': (0.25, 0.50), 'G': (0.25, 0.50)},
        'Conservative': {'E': (0.10, 0.25), 'C': (0.25, 0.40), 'G': (0.35, 0.65)}
    }
    
    def __init__(
        self,
        returns_data: pd.DataFrame,
        risk_free_rate: float = 0.07,
        frequency: int = 252,
        covariance_method: str = 'ledoit_wolf'
    ):
        """
        Initialize the PortfolioOptimizer.
        
        Args:
            returns_data: DataFrame with return series for each asset
            risk_free_rate: Risk-free rate (default: 7% for India)
            frequency: Return frequency for annualization (default: 252 for daily)
            covariance_method: Method for covariance estimation ('sample', 'ledoit_wolf', 'oas')
        """
        self.returns_data = returns_data.copy()
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        self.covariance_method = covariance_method
        
        # Precompute expected returns and covariance
        self._compute_risk_return_estimates()
        
        # Store optimization results
        self.optimization_results = {}
        
    def _compute_risk_return_estimates(self):
        """Compute expected returns and covariance matrix."""
        # Expected returns using CAPM with shrinkage
        # Clean the data first
        clean_data = self.returns_data.dropna().fillna(0)
        
        # Expected returns using mean historical return with validation
        try:
            self.expected_returns = expected_returns.mean_historical_return(
                clean_data, frequency=self.frequency
            )
            
            # Handle any remaining NaN values
            if self.expected_returns.isnull().any():
                logger.warning("NaN values detected in expected returns, using simple mean")
                self.expected_returns = clean_data.mean() * self.frequency
                
        except Exception as e:
            logger.warning(f"PyPortfolioOpt expected returns failed: {e}, using simple mean")
            self.expected_returns = clean_data.mean() * self.frequency        
        # Covariance matrix with regularization
        if self.covariance_method == 'ledoit_wolf':
            self.cov_matrix = risk_models.CovarianceShrinkage(self.returns_data).ledoit_wolf()
        elif self.covariance_method == 'oas':
            self.cov_matrix = risk_models.CovarianceShrinkage(self.returns_data).oracle_approximating()
        else:
            self.cov_matrix = risk_models.sample_cov(clean_data, frequency=self.frequency)
        
        logger.info(f"Computed expected returns for {len(self.expected_returns)} assets")
        # Create aliases for PyPortfolioOpt compatibility
        self.mu = self.expected_returns
        self.S = self.cov_matrix
        logger.info(f"Mean annual return: {self.expected_returns.mean():.2%}")
        logger.info(f"Portfolio volatility range: {np.sqrt(np.diag(self.cov_matrix)).min():.2%} - {np.sqrt(np.diag(self.cov_matrix)).max():.2%}")
    
    def optimize_max_sharpe(
        self,
        target_allocation: Optional[str] = None,
        l2_gamma: float = 0.1
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Optimize for maximum Sharpe ratio portfolio.
        
        Args:
            target_allocation: NPS allocation type ('Aggressive', 'Moderate', 'Conservative')
            l2_gamma: L2 regularization parameter to prevent extreme weights
            
        Returns:
            Dictionary with optimized weights, performance metrics
        """
        ef = EfficientFrontier(
            self.expected_returns,
            self.cov_matrix,
            weight_bounds=(0, 1)  # Long-only constraint
        )
        
        # Add allocation constraints if specified
        if target_allocation and target_allocation in self.NPS_CONSTRAINTS:
            constraints = self.NPS_CONSTRAINTS[target_allocation]
            for asset, (min_weight, max_weight) in constraints.items():
                if asset in self.expected_returns.index:
                    ef.add_constraint(lambda w, asset=asset: w[asset] >= min_weight)
                    ef.add_constraint(lambda w, asset=asset: w[asset] <= max_weight)
        
        # Add L2 regularization to prevent extreme weights
        ef.add_objective(L2_reg, gamma=l2_gamma)
        
        # Optimize for maximum Sharpe ratio
        weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        performance = ef.portfolio_performance(
            risk_free_rate=self.risk_free_rate,
            verbose=False
        )
        
        result = {
            'weights': pd.Series(weights),
            'expected_return': performance[0],
            'volatility': performance[1], 
            'sharpe_ratio': performance[2],
            'allocation_type': target_allocation or 'Unconstrained'
        }
        
        self.optimization_results['max_sharpe'] = result
        return result
    
    def optimize_min_volatility(
        self,
        target_allocation: Optional[str] = None
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Optimize for minimum volatility portfolio.
        
        Args:
            target_allocation: NPS allocation type constraints
            
        Returns:
            Dictionary with optimized weights, performance metrics
        """
        ef = EfficientFrontier(
            self.expected_returns,
            self.cov_matrix,
            weight_bounds=(0, 1)
        )
        
        # Add allocation constraints if specified
        if target_allocation and target_allocation in self.NPS_CONSTRAINTS:
            constraints = self.NPS_CONSTRAINTS[target_allocation]
            for asset, (min_weight, max_weight) in constraints.items():
                if asset in self.expected_returns.index:
                    ef.add_constraint(lambda w, asset=asset: w[asset] >= min_weight)
                    ef.add_constraint(lambda w, asset=asset: w[asset] <= max_weight)
        
        weights = ef.min_volatility()
        performance = ef.portfolio_performance(
            risk_free_rate=self.risk_free_rate,
            verbose=False
        )
        
        result = {
            'weights': pd.Series(weights),
            'expected_return': performance[0],
            'volatility': performance[1],
            'sharpe_ratio': performance[2],
            'allocation_type': target_allocation or 'Unconstrained'
        }
        
        self.optimization_results['min_volatility'] = result
        return result
    
    def optimize_target_return(
        self,
        target_return: float,
        target_allocation: Optional[str] = None
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Optimize for target return with minimum risk.
        
        Args:
            target_return: Target annual return
            target_allocation: NPS allocation type constraints
            
        Returns:
            Dictionary with optimized weights, performance metrics
        """
        ef = EfficientFrontier(
            self.expected_returns,
            self.cov_matrix,
            weight_bounds=(0, 1)
        )
        
        # Add allocation constraints
        if target_allocation and target_allocation in self.NPS_CONSTRAINTS:
            constraints = self.NPS_CONSTRAINTS[target_allocation]
            for asset, (min_weight, max_weight) in constraints.items():
                if asset in self.expected_returns.index:
                    ef.add_constraint(lambda w, asset=asset: w[asset] >= min_weight)
                    ef.add_constraint(lambda w, asset=asset: w[asset] <= max_weight)
        
        weights = ef.efficient_return(target_return)
        performance = ef.portfolio_performance(
            risk_free_rate=self.risk_free_rate,
            verbose=False
        )
        
        result = {
            'weights': pd.Series(weights),
            'expected_return': performance[0],
            'volatility': performance[1],
            'sharpe_ratio': performance[2],
            'target_return': target_return,
            'allocation_type': target_allocation or 'Unconstrained'
        }
        
        return result
    
    def compute_efficient_frontier(
        self,
        num_portfolios: int = 50,
        target_allocation: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute efficient frontier with multiple risk-return points.
        
        Args:
            num_portfolios: Number of portfolios on the frontier
            target_allocation: NPS allocation constraints
            
        Returns:
            DataFrame with weights, returns, volatilities for each portfolio
        """
        # Get return range
        min_ret = self.expected_returns.min()
        max_ret = self.expected_returns.max()
        
        # Create target return range
        target_returns = np.linspace(min_ret * 0.5, max_ret * 0.9, num_portfolios)
        
        frontier_results = []
        
        for target_ret in target_returns:
            try:
                result = self.optimize_target_return(target_ret, target_allocation)
                frontier_results.append({
                    'target_return': target_ret,
                    'expected_return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    **result['weights'].to_dict()
                })
            except Exception as e:
                logger.warning(f"Failed to optimize for return {target_ret:.2%}: {e}")
                continue
        
        if not frontier_results:
            raise ValueError("Failed to compute efficient frontier")
        
        frontier_df = pd.DataFrame(frontier_results)
        self.optimization_results['efficient_frontier'] = frontier_df
        
        return frontier_df
    
    def optimize_risk_parity(self) -> Dict[str, Union[float, pd.Series]]:
        """
        Optimize for risk parity (equal risk contribution) portfolio.
        
        Returns:
            Dictionary with optimized weights, performance metrics
        """
        n_assets = len(self.expected_returns)
        
        def risk_parity_objective(weights, cov_matrix):
            """Risk parity objective function."""
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights / portfolio_vol
            contrib = weights * marginal_contrib
            target_contrib = np.ones(n_assets) / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Optimization constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
        ]
        bounds = [(0.01, 0.5) for _ in range(n_assets)]  # Reasonable bounds
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            args=(self.cov_matrix.values,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning("Risk parity optimization failed, using equal weights")
            weights = pd.Series(x0, index=self.expected_returns.index)
        else:
            weights = pd.Series(result.x, index=self.expected_returns.index)
        
        # Calculate performance metrics
        expected_return = weights @ self.expected_returns
        volatility = np.sqrt(weights @ self.cov_matrix @ weights)
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility
        
        result_dict = {
            'weights': weights,
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'allocation_type': 'Risk Parity'
        }
        
        self.optimization_results['risk_parity'] = result_dict
        return result_dict
    
    def optimize_cvar(
        self,
        confidence_level: float = 0.05,
        target_allocation: Optional[str] = None
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Optimize portfolio using Conditional Value at Risk (CVaR).
        
        Args:
            confidence_level: CVaR confidence level (default: 5%)
            target_allocation: NPS allocation constraints
            
        Returns:
            Dictionary with optimized weights, performance metrics
        """
        try:
            ec = EfficientCVaR(
                self.expected_returns,
                self.returns_data,
                confidence_level=confidence_level
            )
            
            # Add allocation constraints
            if target_allocation and target_allocation in self.NPS_CONSTRAINTS:
                constraints = self.NPS_CONSTRAINTS[target_allocation]
                for asset, (min_weight, max_weight) in constraints.items():
                    if asset in self.expected_returns.index:
                        ec.add_constraint(lambda w, asset=asset: w[asset] >= min_weight)
                        ec.add_constraint(lambda w, asset=asset: w[asset] <= max_weight)
            
            weights = ec.min_cvar()
            performance = ec.portfolio_performance(verbose=False)
            
            result = {
                'weights': pd.Series(weights),
                'expected_return': performance[0],
                'cvar': performance[1],
                'allocation_type': target_allocation or 'Unconstrained',
                'confidence_level': confidence_level
            }
            
            self.optimization_results['cvar'] = result
            return result
            
        except Exception as e:
            logger.error(f"CVaR optimization failed: {e}")
            # Fallback to minimum volatility
            return self.optimize_min_volatility(target_allocation)
    
    def generate_lifecycle_portfolios(
        self,
        ages: List[int] = [25, 35, 45, 55, 60]
    ) -> Dict[int, Dict[str, float]]:
        """
        Generate lifecycle allocation portfolios based on age.
        
        Implements target-date fund logic:
        - Younger ages: Higher equity allocation
        - Older ages: More conservative allocation
        
        Args:
            ages: List of ages to generate portfolios for
            
        Returns:
            Dictionary mapping age to portfolio allocation
        """
        lifecycle_portfolios = {}
        
        for age in ages:
            # Rule of thumb: Equity allocation = (100 - age)% with adjustments
            base_equity_pct = max(0.20, min(0.80, (100 - age) / 100))
            
            # Determine allocation category based on equity percentage
            if base_equity_pct >= 0.60:
                allocation_type = 'Aggressive'
            elif base_equity_pct >= 0.40:
                allocation_type = 'Moderate'
            else:
                allocation_type = 'Conservative'
            
            # Get constraints for this allocation type
            constraints = self.NPS_CONSTRAINTS[allocation_type]
            
            # Optimize portfolio with lifecycle-adjusted constraints
            try:
                # Adjust constraints based on age-specific equity target
                adjusted_constraints = constraints.copy()
                equity_range = constraints.get('E', (0.1, 0.8))
                target_equity = max(equity_range[0], min(equity_range[1], base_equity_pct))
                
                # Optimize with target equity allocation
                ef = EfficientFrontier(
                    self.expected_returns,
                    self.cov_matrix,
                    weight_bounds=(0, 1)
                )
                
                # Add soft constraint for target equity allocation
                if 'E' in self.expected_returns.index:
                    ef.add_constraint(lambda w: w['E'] >= target_equity * 0.9)
                    ef.add_constraint(lambda w: w['E'] <= target_equity * 1.1)
                
                weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
                performance = ef.portfolio_performance(
                    risk_free_rate=self.risk_free_rate,
                    verbose=False
                )
                
                lifecycle_portfolios[age] = {
                    'weights': pd.Series(weights),
                    'expected_return': performance[0],
                    'volatility': performance[1],
                    'sharpe_ratio': performance[2],
                    'allocation_type': allocation_type,
                    'target_equity_pct': target_equity
                }
                
            except Exception as e:
                logger.warning(f"Lifecycle optimization failed for age {age}: {e}")
                # Use simple rule-based allocation as fallback
                weights = self._simple_lifecycle_allocation(age)
                lifecycle_portfolios[age] = {
                    'weights': weights,
                    'allocation_type': allocation_type,
                    'target_equity_pct': base_equity_pct
                }
        
        self.optimization_results['lifecycle'] = lifecycle_portfolios
        return lifecycle_portfolios
    
    def _simple_lifecycle_allocation(self, age: int) -> pd.Series:
        """Simple rule-based lifecycle allocation."""
        equity_pct = max(0.20, min(0.80, (100 - age) / 100))
        bond_pct = (1 - equity_pct) * 0.7  # 70% of non-equity in corporate bonds
        govt_pct = (1 - equity_pct) * 0.3  # 30% in government securities
        
        # Map to available assets
        weights = {}
        for asset in self.expected_returns.index:
            if 'E' in asset:  # Equity schemes
                weights[asset] = equity_pct / len([a for a in self.expected_returns.index if 'E' in a])
            elif 'C' in asset:  # Corporate bond schemes  
                weights[asset] = bond_pct / len([a for a in self.expected_returns.index if 'C' in a])
            elif 'G' in asset:  # Government securities
                weights[asset] = govt_pct / len([a for a in self.expected_returns.index if 'G' in a])
            else:
                weights[asset] = 0.0
        
        return pd.Series(weights)
    
    def get_allocation_summary(self) -> pd.DataFrame:
        """
        Get summary of all optimization results.
        
        Returns:
            DataFrame comparing different optimization strategies
        """
        summary_data = []
        
        for strategy, result in self.optimization_results.items():
            if strategy == 'efficient_frontier':
                continue  # Skip frontier data
                
            if strategy == 'lifecycle':
                # Summarize lifecycle portfolios
                for age, portfolio in result.items():
                    summary_data.append({
                        'strategy': f'Lifecycle_{age}',
                        'expected_return': portfolio.get('expected_return', np.nan),
                        'volatility': portfolio.get('volatility', np.nan), 
                        'sharpe_ratio': portfolio.get('sharpe_ratio', np.nan),
                        'allocation_type': portfolio.get('allocation_type', ''),
                        'equity_weight': portfolio['weights'].get('E', 0) if 'weights' in portfolio else 0
                    })
            else:
                summary_data.append({
                    'strategy': strategy,
                    'expected_return': result.get('expected_return', np.nan),
                    'volatility': result.get('volatility', np.nan),
                    'sharpe_ratio': result.get('sharpe_ratio', np.nan),
                    'allocation_type': result.get('allocation_type', ''),
                    'equity_weight': result['weights'].get('E', 0) if 'weights' in result else 0
                })
        
        return pd.DataFrame(summary_data).round(4)
    
    def clean_weights(self, weights: pd.Series, threshold: float = 0.01) -> pd.Series:
        """
        Clean portfolio weights by removing tiny allocations.
        
        Args:
            weights: Portfolio weights
            threshold: Minimum weight threshold
            
        Returns:
            Cleaned weights that sum to 1
        """
        # Remove weights below threshold
        cleaned = weights.copy()
        cleaned[cleaned < threshold] = 0
        
        # Renormalize to sum to 1
        if cleaned.sum() > 0:
            cleaned = cleaned / cleaned.sum()
        
        return cleaned 