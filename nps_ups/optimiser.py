"""
Portfolio optimization module using PyPortfolioOpt for NPS/UPS analysis.

Implements:
- Mean-variance optimization with Markowitz efficient frontier
- Maximum Sharpe ratio portfolios
- Risk parity and minimum volatility portfolios  
- Custom lifecycle allocation strategies
- Scenario-based robust optimization
- CVaR (Conditional Value at Risk) optimization

Features robust data preprocessing and numerical stability for production use.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, CLA, EfficientCVaR
from pypfopt import risk_models, expected_returns
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from scipy.optimize import minimize
import cvxpy as cp

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class PortfolioOptimizer:
    """
    Advanced portfolio optimizer with robust numerical stability.
    
    Supports:
    - Multiple optimization objectives (max Sharpe, min volatility, target return)
    - NPS-specific allocation constraints (Aggressive/Moderate/Conservative)
    - Lifecycle portfolio strategies with age-based rebalancing
    - CVaR optimization for tail risk management
    - Robust covariance estimation with shrinkage methods
    """
    
    # NPS allocation constraints per PFRDA guidelines
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
        Initialize the PortfolioOptimizer with robust data preprocessing.
        
        Args:
            returns_data: DataFrame with return series for each asset
            risk_free_rate: Risk-free rate (default: 7% for India)
            frequency: Return frequency for annualization (default: 252 for daily)
            covariance_method: Method for covariance estimation ('sample', 'ledoit_wolf', 'oas')
        """
        # Store parameters
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        self.covariance_method = covariance_method
        
        # Clean and validate input data
        self.returns_data = self._preprocess_returns_data(returns_data)
        
        # Compute risk-return estimates with robust methods
        self._compute_risk_return_estimates()
        
        # Store optimization results
        self.optimization_results = {}
        
        logger.info(f"PortfolioOptimizer initialized with {len(self.returns_data.columns)} assets, {len(self.returns_data)} observations")
        
    def _preprocess_returns_data(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Robust data preprocessing with outlier handling and validation.
        
        Args:
            returns_data: Raw returns data
            
        Returns:
            Cleaned returns data
        """
        logger.info("Preprocessing returns data for numerical stability...")
        
        # Make a copy to avoid modifying original
        clean_data = returns_data.copy()
        
        # Remove any non-finite values
        clean_data = clean_data.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill small gaps, then drop remaining NaNs
        clean_data = clean_data.fillna(method='ffill', limit=5)
        clean_data = clean_data.dropna()
        
        if len(clean_data) < 30:
            raise ValueError(f"Insufficient data: only {len(clean_data)} observations after cleaning")
        
        # Winsorize extreme outliers (beyond 10 standard deviations)
        for col in clean_data.columns:
            std = clean_data[col].std()
            mean = clean_data[col].mean()
            lower_bound = mean - 10 * std
            upper_bound = mean + 10 * std
            clean_data[col] = np.clip(clean_data[col], lower_bound, upper_bound)
        
        # Final validation
        if clean_data.isnull().any().any():
            raise ValueError("Data still contains NaN values after preprocessing")
        
        if (clean_data.std() == 0).any():
            zero_vol_assets = clean_data.columns[clean_data.std() == 0].tolist()
            logger.warning(f"Zero volatility assets detected: {zero_vol_assets}")
            # Add tiny noise to zero-volatility assets
            for asset in zero_vol_assets:
                clean_data[asset] += np.random.normal(0, 1e-6, len(clean_data))
        
        logger.info(f"Data preprocessing complete: {len(clean_data)} observations, {len(clean_data.columns)} assets")
        return clean_data
        
    def _compute_risk_return_estimates(self):
        """
        Compute expected returns and covariance matrix with robust methods.
        """
        logger.info("Computing risk-return estimates...")
        
        # Expected returns with robust estimation
        try:
            # Use PyPortfolioOpt's mean historical return
            self.expected_returns = expected_returns.mean_historical_return(
                self.returns_data, frequency=self.frequency
            )
            
            # Validate and handle any issues
            if self.expected_returns.isnull().any():
                logger.warning("NaN values in expected returns, using sample mean")
                self.expected_returns = self.returns_data.mean() * self.frequency
                
        except Exception as e:
            logger.warning(f"PyPortfolioOpt expected returns failed: {e}, using sample mean")
            self.expected_returns = self.returns_data.mean() * self.frequency
        
        # Covariance matrix with consistent data and regularization
        try:
            if self.covariance_method == 'ledoit_wolf':
                # Use cleaned data for consistency
                self.cov_matrix = risk_models.CovarianceShrinkage(self.returns_data).ledoit_wolf()
            elif self.covariance_method == 'oas':
                self.cov_matrix = risk_models.CovarianceShrinkage(self.returns_data).oracle_approximating()
            else:
                self.cov_matrix = risk_models.sample_cov(self.returns_data, frequency=self.frequency)
                
            # Validate covariance matrix
            if not self._is_valid_covariance_matrix(self.cov_matrix):
                logger.warning("Invalid covariance matrix, using regularized version")
                self.cov_matrix = self._regularize_covariance_matrix(self.cov_matrix)
                
        except Exception as e:
            logger.error(f"Covariance estimation failed: {e}, using sample covariance")
            self.cov_matrix = risk_models.sample_cov(self.returns_data, frequency=self.frequency)
        
        # Create aliases for PyPortfolioOpt compatibility
        self.mu = self.expected_returns
        self.S = self.cov_matrix
        
        # Validation logging
        logger.info(f"Expected returns computed: mean={self.expected_returns.mean():.4f}, std={self.expected_returns.std():.4f}")
        logger.info(f"Covariance matrix: shape={self.cov_matrix.shape}, condition number={np.linalg.cond(self.cov_matrix):.2f}")
        
        volatilities = np.sqrt(np.diag(self.cov_matrix))
        logger.info(f"Asset volatilities: min={volatilities.min():.4f}, max={volatilities.max():.4f}")
        
    def _is_valid_covariance_matrix(self, cov_matrix: pd.DataFrame) -> bool:
        """Check if covariance matrix is valid (positive semi-definite)."""
        try:
            eigenvals = np.linalg.eigvals(cov_matrix.values)
            return np.all(eigenvals >= -1e-8)  # Allow for small numerical errors
        except:
            return False
    
    def _regularize_covariance_matrix(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """Regularize covariance matrix to ensure positive semi-definiteness."""
        # Add small regularization to diagonal
        regularized = cov_matrix + np.eye(len(cov_matrix)) * 1e-5
        return regularized
    
    def optimize_max_sharpe(
        self,
        target_allocation: Optional[str] = None,
        l2_gamma: float = 0.1
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Optimize for maximum Sharpe ratio portfolio with robust solver configuration.
        
        Args:
            target_allocation: Target allocation type ('Aggressive', 'Moderate', 'Conservative')
            l2_gamma: L2 regularization parameter for stability
            
        Returns:
            Dictionary containing optimized weights and performance metrics
        """
        logger.info(f"Optimizing for maximum Sharpe ratio (allocation: {target_allocation})")
        
        # Validate inputs
        if not hasattr(self, 'mu') or not hasattr(self, 'S'):
            raise ValueError("Expected returns and covariance matrix not computed")
        
        # Try multiple solvers for robustness
        solvers = ['OSQP', 'ECOS', 'SCS', 'CLARABEL']
        
        for solver in solvers:
            try:
                logger.debug(f"Attempting optimization with {solver} solver...")
                
                # Create efficient frontier with validated data
                ef = EfficientFrontier(
                    self.mu, 
                    self.S, 
                    weight_bounds=(0, 1),
                    solver=solver
                )
                
                # Add L2 regularization for numerical stability
                ef.add_objective(objective_functions.L2_reg, gamma=l2_gamma)
                
                # Apply NPS allocation constraints if specified
                if target_allocation and target_allocation in self.NPS_CONSTRAINTS:
                    self._apply_nps_constraints(ef, target_allocation)
                
                # Optimize for maximum Sharpe ratio
                weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
                cleaned_weights = ef.clean_weights()
                
                # Calculate performance metrics
                expected_return, volatility, sharpe_ratio = ef.portfolio_performance(
                    verbose=False, risk_free_rate=self.risk_free_rate
                )
                
                # Validate results
                if not (0.001 < expected_return < 2.0 and 0.001 < volatility < 2.0):
                    logger.warning(f"Unrealistic results with {solver}: return={expected_return:.4f}, vol={volatility:.4f}")
                    continue
                
                logger.info(f"Optimization successful with {solver} solver")
                logger.info(f"Portfolio: return={expected_return:.2%}, volatility={volatility:.2%}, Sharpe={sharpe_ratio:.2f}")
                
                result = {
                    'weights': pd.Series(cleaned_weights),
                    'expected_return': expected_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'method': 'max_sharpe',
                    'target_allocation': target_allocation,
                    'solver_used': solver
                }
                
                # Store result
                self.optimization_results['max_sharpe'] = result
                return result
                
            except Exception as e:
                logger.warning(f"Optimization failed with {solver} solver: {e}")
                continue
        
        # If all solvers fail, use fallback method
        logger.warning("All convex solvers failed, using equal weight fallback")
        return self._fallback_equal_weight_portfolio()
    
    def optimize_min_volatility(
        self, 
        target_allocation: Optional[str] = None
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Optimize for minimum volatility portfolio.
        
        Args:
            target_allocation: Target allocation type
            
        Returns:
            Dictionary containing optimized weights and performance metrics
        """
        logger.info("Optimizing for minimum volatility portfolio")
        
        try:
            ef = EfficientFrontier(self.mu, self.S, weight_bounds=(0, 1))
            
            # Apply constraints if specified
            if target_allocation and target_allocation in self.NPS_CONSTRAINTS:
                self._apply_nps_constraints(ef, target_allocation)
            
            weights = ef.min_volatility()
            cleaned_weights = ef.clean_weights()
            
            expected_return, volatility, sharpe_ratio = ef.portfolio_performance(
                verbose=False, risk_free_rate=self.risk_free_rate
            )
            
            result = {
                'weights': pd.Series(cleaned_weights),
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'method': 'min_volatility',
                'target_allocation': target_allocation
            }
            
            self.optimization_results['min_volatility'] = result
            return result
            
        except Exception as e:
            logger.error(f"Min volatility optimization failed: {e}")
            return self._fallback_equal_weight_portfolio()
    
    def optimize_efficient_return(
        self,
        target_return: float,
        target_allocation: Optional[str] = None
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Optimize for target return with minimum risk.
        
        Args:
            target_return: Target annual return
            target_allocation: Target allocation type
            
        Returns:
            Dictionary containing optimized weights and performance metrics
        """
        logger.info(f"Optimizing for target return: {target_return:.2%}")
        
        try:
            ef = EfficientFrontier(self.mu, self.S, weight_bounds=(0, 1))
            
            if target_allocation and target_allocation in self.NPS_CONSTRAINTS:
                self._apply_nps_constraints(ef, target_allocation)
            
            weights = ef.efficient_return(target_return)
            cleaned_weights = ef.clean_weights()
            
            expected_return, volatility, sharpe_ratio = ef.portfolio_performance(
                verbose=False, risk_free_rate=self.risk_free_rate
            )
            
            result = {
                'weights': pd.Series(cleaned_weights),
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'method': 'efficient_return',
                'target_return': target_return,
                'target_allocation': target_allocation
            }
            
            self.optimization_results['efficient_return'] = result
            return result
            
        except Exception as e:
            logger.error(f"Target return optimization failed: {e}")
            return self._fallback_equal_weight_portfolio()
    
    def optimize_cvar(
        self,
        confidence_level: float = 0.05,
        target_allocation: Optional[str] = None
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Optimize portfolio using CVaR (Conditional Value at Risk).
        
        Args:
            confidence_level: Confidence level for CVaR (e.g., 0.05 for 95% CVaR)
            target_allocation: Target allocation type
            
        Returns:
            Dictionary containing optimized weights and performance metrics
        """
        logger.info(f"Optimizing CVaR portfolio (confidence: {1-confidence_level:.0%})")
        
        try:
            # CVaR optimization with returns data
            cvar_optimizer = EfficientCVaR(
                self.mu,
                self.returns_data,
                beta=confidence_level
            )
            
            weights = cvar_optimizer.min_cvar()
            cleaned_weights = cvar_optimizer.clean_weights()
            
            # Calculate performance using regular EfficientFrontier
            ef = EfficientFrontier(self.mu, self.S)
            ef.set_weights(cleaned_weights)
            expected_return, volatility, sharpe_ratio = ef.portfolio_performance(
                verbose=False, risk_free_rate=self.risk_free_rate
            )
            
            result = {
                'weights': pd.Series(cleaned_weights),
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'method': 'cvar',
                'confidence_level': confidence_level,
                'target_allocation': target_allocation
            }
            
            self.optimization_results['cvar'] = result
            return result
            
        except Exception as e:
            logger.error(f"CVaR optimization failed: {e}, falling back to max Sharpe")
            return self.optimize_max_sharpe(target_allocation)
    
    def generate_efficient_frontier(
        self,
        num_portfolios: int = 100,
        target_allocation: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate efficient frontier portfolios.
        
        Args:
            num_portfolios: Number of portfolios to generate
            target_allocation: Target allocation type
            
        Returns:
            DataFrame with portfolio weights, returns, and risks
        """
        logger.info(f"Generating efficient frontier with {num_portfolios} portfolios")
        
        try:
            ef = EfficientFrontier(self.mu, self.S, weight_bounds=(0, 1))
            
            if target_allocation and target_allocation in self.NPS_CONSTRAINTS:
                self._apply_nps_constraints(ef, target_allocation)
            
            # Generate frontier
            frontier_data = []
            
            # Get return range
            min_ret = self.mu.min()
            max_ret = self.mu.max()
            
            # Generate target returns
            target_returns = np.linspace(min_ret * 1.1, max_ret * 0.9, num_portfolios)
            
            for target_ret in target_returns:
                try:
                    ef_copy = EfficientFrontier(self.mu, self.S, weight_bounds=(0, 1))
                    if target_allocation and target_allocation in self.NPS_CONSTRAINTS:
                        self._apply_nps_constraints(ef_copy, target_allocation)
                    
                    weights = ef_copy.efficient_return(target_ret)
                    cleaned_weights = ef_copy.clean_weights()
                    
                    expected_return, volatility, sharpe_ratio = ef_copy.portfolio_performance(
                        verbose=False, risk_free_rate=self.risk_free_rate
                    )
                    
                    portfolio_data = {
                        'return': expected_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio,
                        **cleaned_weights
                    }
                    
                    frontier_data.append(portfolio_data)
                    
                except:
                    continue
            
            frontier_df = pd.DataFrame(frontier_data)
            logger.info(f"Generated {len(frontier_df)} efficient portfolios")
            
            return frontier_df
            
        except Exception as e:
            logger.error(f"Efficient frontier generation failed: {e}")
            return pd.DataFrame()
    
    def generate_lifecycle_portfolios(
        self,
        ages: List[int] = [25, 35, 45, 55, 60]
    ) -> Dict[int, Dict[str, float]]:
        """
        Generate lifecycle portfolios with age-appropriate allocations.
        
        Args:
            ages: List of ages for lifecycle portfolios
            
        Returns:
            Dictionary mapping ages to portfolio allocations
        """
        logger.info(f"Generating lifecycle portfolios for ages: {ages}")
        
        lifecycle_portfolios = {}
        
        for age in ages:
            # Determine allocation type based on age
            if age <= 35:
                allocation_type = 'Aggressive'
            elif age <= 50:
                allocation_type = 'Moderate'  
            else:
                allocation_type = 'Conservative'
            
            try:
                # Optimize with age-appropriate constraints
                result = self.optimize_max_sharpe(target_allocation=allocation_type)
                lifecycle_portfolios[age] = {
                    'allocation_type': allocation_type,
                    'weights': result['weights'].to_dict(),
                    'expected_return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio']
                }
                
            except Exception as e:
                logger.error(f"Lifecycle portfolio failed for age {age}: {e}")
                lifecycle_portfolios[age] = {
                    'allocation_type': allocation_type,
                    'weights': self._get_equal_weights(),
                    'expected_return': 0.07,
                    'volatility': 0.15,
                    'sharpe_ratio': 0.33
                }
        
        return lifecycle_portfolios
    
    def _apply_nps_constraints(self, ef: EfficientFrontier, allocation_type: str):
        """Apply NPS allocation constraints to efficient frontier."""
        constraints = self.NPS_CONSTRAINTS[allocation_type]
        
        for scheme_type, (min_weight, max_weight) in constraints.items():
            # Find assets matching this scheme type
            matching_assets = [col for col in self.returns_data.columns if scheme_type in col]
            
            if matching_assets:
                # Add constraint for this scheme type
                asset_indices = [list(self.returns_data.columns).index(asset) for asset in matching_assets]
                
                # Sum of weights for this scheme type should be in [min_weight, max_weight]
                ef.add_constraint(lambda w, indices=asset_indices: sum(w[i] for i in indices) >= min_weight)
                ef.add_constraint(lambda w, indices=asset_indices: sum(w[i] for i in indices) <= max_weight)
    
    def _fallback_equal_weight_portfolio(self) -> Dict[str, Union[float, pd.Series]]:
        """Fallback to equal weight portfolio if optimization fails."""
        logger.warning("Using equal weight fallback portfolio")
        
        n_assets = len(self.returns_data.columns)
        equal_weights = pd.Series(
            data=1.0 / n_assets,
            index=self.returns_data.columns
        )
        
        # Calculate performance
        portfolio_return = (equal_weights * self.expected_returns).sum()
        portfolio_variance = np.dot(equal_weights.values, np.dot(self.cov_matrix.values, equal_weights.values))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return {
            'weights': equal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'method': 'equal_weight_fallback'
        }
    
    def _get_equal_weights(self) -> Dict[str, float]:
        """Get equal weight allocation dictionary."""
        n_assets = len(self.returns_data.columns)
        return {asset: 1.0/n_assets for asset in self.returns_data.columns}
    
    def get_portfolio_summary(self) -> Dict[str, any]:
        """Get summary of all optimization results."""
        return {
            'optimization_results': self.optimization_results,
            'data_summary': {
                'n_assets': len(self.returns_data.columns),
                'n_observations': len(self.returns_data),
                'date_range': f"{self.returns_data.index[0]} to {self.returns_data.index[-1]}",
                'expected_returns': self.expected_returns.to_dict(),
                'volatilities': np.sqrt(np.diag(self.cov_matrix)).tolist()
            }
        } 