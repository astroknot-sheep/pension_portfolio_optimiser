"""
Risk analytics module for comprehensive portfolio risk assessment.

Provides:
- Value at Risk (VaR) and Expected Shortfall (ES) calculations
- Vector Autoregression (VAR) modeling for inflation sensitivity
- Impulse Response Functions (IRF) for shock analysis
- Stress testing and scenario analysis
- Risk attribution and decomposition
- Performance metrics and risk-adjusted returns
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, t
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class RiskAnalytics:
    """
    Comprehensive risk analytics for pension portfolio analysis.
    
    Capabilities:
    - Historical and parametric VaR/ES
    - Monte Carlo simulation for risk metrics
    - VAR modeling for macroeconomic relationships
    - Stress testing under various scenarios
    - Risk attribution and factor decomposition
    """
    
    def __init__(self, returns_data: pd.DataFrame, confidence_levels: List[float] = [0.95, 0.99]):
        """
        Initialize RiskAnalytics.
        
        Args:
            returns_data: DataFrame with asset return series
            confidence_levels: Confidence levels for VaR calculation
        """
        self.returns_data = returns_data.copy()
        self.confidence_levels = confidence_levels
        self.var_results = {}
        self.stress_test_results = {}
        
        # Precompute basic statistics
        self._compute_basic_stats()
    
    def _compute_basic_stats(self):
        """Compute basic statistical properties of returns."""
        self.mean_returns = self.returns_data.mean()
        self.std_returns = self.returns_data.std()
        self.skewness = self.returns_data.skew()
        self.kurtosis = self.returns_data.kurtosis()
        
        # Correlation matrix
        self.correlation_matrix = self.returns_data.corr()
        
        logger.info(f"Computed statistics for {len(self.returns_data.columns)} assets")
        logger.info(f"Average annual return: {self.mean_returns.mean() * 252:.2%}")
        logger.info(f"Average volatility: {self.std_returns.mean() * np.sqrt(252):.2%}")
    
    def calculate_var_es(
        self,
        portfolio_weights: pd.Series,
        method: str = 'historical',
        confidence_levels: Optional[List[float]] = None,
        holding_period: int = 1
    ) -> Dict[str, Dict[float, float]]:
        """
        Calculate Value at Risk (VaR) and Expected Shortfall (ES) for portfolio.
        
        Args:
            portfolio_weights: Portfolio weights
            method: Calculation method ('historical', 'parametric', 'monte_carlo')
            confidence_levels: Confidence levels (default: class levels)
            holding_period: Holding period in days
            
        Returns:
            Dictionary with VaR and ES for each confidence level
        """
        if confidence_levels is None:
            confidence_levels = self.confidence_levels
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns_data * portfolio_weights).sum(axis=1)
        
        # Scale for holding period
        if holding_period > 1:
            portfolio_returns = portfolio_returns * np.sqrt(holding_period)
        
        results = {'VaR': {}, 'ES': {}}
        
        for confidence_level in confidence_levels:
            alpha = 1 - confidence_level
            
            if method == 'historical':
                var, es = self._historical_var_es(portfolio_returns, alpha)
            elif method == 'parametric':
                var, es = self._parametric_var_es(portfolio_returns, alpha)
            elif method == 'monte_carlo':
                var, es = self._monte_carlo_var_es(portfolio_returns, alpha)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results['VaR'][confidence_level] = var
            results['ES'][confidence_level] = es
        
        self.var_results[f'{method}_{holding_period}d'] = results
        return results
    
    def _historical_var_es(self, returns: pd.Series, alpha: float) -> Tuple[float, float]:
        """Calculate historical VaR and ES."""
        sorted_returns = returns.sort_values()
        index = int(alpha * len(sorted_returns))
        
        var = -sorted_returns.iloc[index]
        es = -sorted_returns.iloc[:index].mean()
        
        return var, es
    
    def _parametric_var_es(self, returns: pd.Series, alpha: float) -> Tuple[float, float]:
        """Calculate parametric VaR and ES assuming normal distribution."""
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Check for normality (simple test)
        if abs(returns.skew()) > 1 or returns.kurtosis() > 4:
            logger.warning("Returns show significant non-normality, consider alternative methods")
        
        var = -(mean_return + norm.ppf(alpha) * std_return)
        es = -(mean_return - (norm.pdf(norm.ppf(alpha)) / alpha) * std_return)
        
        return var, es
    
    def _monte_carlo_var_es(
        self, 
        returns: pd.Series, 
        alpha: float, 
        n_simulations: int = 10000
    ) -> Tuple[float, float]:
        """Calculate Monte Carlo VaR and ES."""
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate random scenarios
        np.random.seed(42)
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        
        # Calculate VaR and ES
        sorted_returns = np.sort(simulated_returns)
        index = int(alpha * n_simulations)
        
        var = -sorted_returns[index]
        es = -sorted_returns[:index].mean()
        
        return var, es
    
    def run_var_model(
        self,
        macro_data: pd.DataFrame,
        lags: int = 4,
        variables: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Run Vector Autoregression (VAR) model for macroeconomic analysis.
        
        Args:
            macro_data: DataFrame with macroeconomic variables
            lags: Number of lags to include
            variables: Specific variables to include (default: all)
            
        Returns:
            Dictionary with VAR model results and diagnostics
        """
        if variables is None:
            variables = macro_data.columns.tolist()
        
        # Prepare data
        model_data = macro_data[variables].dropna()
        
        # Check stationarity (simple ADF test surrogate)
        diff_data = model_data.diff().dropna()
        
        # Fit VAR model
        try:
            var_model = VAR(diff_data)
            var_fitted = var_model.fit(lags)
            
            # Model diagnostics
            residuals = var_fitted.resid
            
            # Ljung-Box test for serial correlation
            lb_tests = {}
            for col in residuals.columns:
                try:
                    lb_stat, lb_pvalue = acorr_ljungbox(
                        residuals[col], 
                        lags=min(10, len(residuals)//4),
                        return_df=False
                    )
                    lb_tests[col] = {'statistic': lb_stat[-1], 'p_value': lb_pvalue[-1]}
                except:
                    lb_tests[col] = {'statistic': np.nan, 'p_value': np.nan}
            
            results = {
                'model': var_fitted,
                'summary': str(var_fitted.summary()),
                'aic': var_fitted.aic,
                'bic': var_fitted.bic,
                'residuals': residuals,
                'ljung_box_tests': lb_tests,
                'variables': variables,
                'lags': lags
            }
            
            logger.info(f"VAR model fitted with {lags} lags, AIC: {var_fitted.aic:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"VAR model fitting failed: {e}")
            return {'error': str(e)}
    
    def calculate_impulse_responses(
        self,
        var_results: Dict,
        periods: int = 24,
        shock_size: float = 1.0
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate Impulse Response Functions (IRF) from VAR model.
        
        Args:
            var_results: Results from run_var_model
            periods: Number of periods for IRF
            shock_size: Size of shock (in standard deviations)
            
        Returns:
            Dictionary with IRF DataFrames for each variable
        """
        if 'model' not in var_results:
            raise ValueError("VAR model not found in results")
        
        var_model = var_results['model']
        
        try:
            # Calculate IRFs
            irf = var_model.irf(periods)
            
            # Extract IRF matrices
            irfs = {}
            variables = var_results['variables']
            
            for i, shock_var in enumerate(variables):
                irf_matrix = []
                for period in range(periods):
                    responses = irf.irfs[period][:, i] * shock_size
                    irf_matrix.append(responses)
                
                irf_df = pd.DataFrame(
                    irf_matrix, 
                    columns=[f'Response_{var}' for var in variables],
                    index=range(periods)
                )
                irfs[f'Shock_to_{shock_var}'] = irf_df
            
            # Calculate cumulative IRFs
            cumulative_irfs = {}
            for shock_var, irf_df in irfs.items():
                cumulative_irfs[f'Cumulative_{shock_var}'] = irf_df.cumsum()
            
            irfs.update(cumulative_irfs)
            
            logger.info(f"Calculated IRFs for {len(variables)} variables over {periods} periods")
            return irfs
            
        except Exception as e:
            logger.error(f"IRF calculation failed: {e}")
            return {}
    
    def stress_test_portfolio(
        self,
        portfolio_weights: pd.Series,
        scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform stress testing under various scenarios.
        
        Args:
            portfolio_weights: Portfolio weights
            scenarios: Dictionary of stress scenarios
                     Format: {'scenario_name': {'asset_name': shock_percentage}}
            
        Returns:
            Dictionary with stress test results
        """
        stress_results = {}
        
        # Baseline portfolio statistics
        portfolio_returns = (self.returns_data * portfolio_weights).sum(axis=1)
        baseline_return = portfolio_returns.mean() * 252
        baseline_vol = portfolio_returns.std() * np.sqrt(252)
        
        stress_results['baseline'] = {
            'annual_return': baseline_return,
            'annual_volatility': baseline_vol,
            'sharpe_ratio': baseline_return / baseline_vol if baseline_vol > 0 else 0
        }
        
        for scenario_name, shocks in scenarios.items():
            stressed_returns = self.returns_data.copy()
            
            # Apply shocks to specified assets
            for asset, shock_pct in shocks.items():
                if asset in stressed_returns.columns:
                    # Apply one-time shock to returns
                    stressed_returns[asset] *= (1 + shock_pct)
            
            # Calculate stressed portfolio performance
            stressed_portfolio_returns = (stressed_returns * portfolio_weights).sum(axis=1)
            stressed_annual_return = stressed_portfolio_returns.mean() * 252
            stressed_annual_vol = stressed_portfolio_returns.std() * np.sqrt(252)
            
            # Calculate impact
            return_impact = stressed_annual_return - baseline_return
            vol_impact = stressed_annual_vol - baseline_vol
            
            stress_results[scenario_name] = {
                'annual_return': stressed_annual_return,
                'annual_volatility': stressed_annual_vol,
                'sharpe_ratio': stressed_annual_return / stressed_annual_vol if stressed_annual_vol > 0 else 0,
                'return_impact': return_impact,
                'volatility_impact': vol_impact,
                'total_impact': return_impact - vol_impact  # Risk-adjusted impact
            }
        
        self.stress_test_results = stress_results
        return stress_results
    
    def calculate_risk_attribution(
        self,
        portfolio_weights: pd.Series,
        factor_loadings: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.Series]:
        """
        Perform risk attribution analysis.
        
        Args:
            portfolio_weights: Portfolio weights
            factor_loadings: Factor loadings matrix (if None, use PCA)
            
        Returns:
            Dictionary with risk attribution results
        """
        # Portfolio returns
        portfolio_returns = (self.returns_data * portfolio_weights).sum(axis=1)
        portfolio_variance = portfolio_returns.var()
        
        # If no factor loadings provided, use PCA
        if factor_loadings is None:
            pca = PCA(n_components=min(5, len(self.returns_data.columns)))
            pca.fit(self.returns_data.fillna(0))
            
            factor_loadings = pd.DataFrame(
                pca.components_.T,
                index=self.returns_data.columns,
                columns=[f'Factor_{i+1}' for i in range(pca.n_components_)]
            )
        
        # Calculate factor exposures
        factor_exposures = portfolio_weights @ factor_loadings
        
        # Risk attribution
        asset_contributions = {}
        factor_contributions = {}
        
        # Asset-level risk contribution (marginal contribution to risk)
        covariance_matrix = self.returns_data.cov()
        marginal_risk = (covariance_matrix @ portfolio_weights) / np.sqrt(portfolio_variance)
        asset_risk_contrib = portfolio_weights * marginal_risk
        
        # Normalize to sum to 100%
        asset_risk_contrib = asset_risk_contrib / asset_risk_contrib.sum()
        
        # Factor-level risk contribution
        if len(factor_loadings.columns) > 0:
            factor_covariance = factor_loadings.T @ covariance_matrix @ factor_loadings
            factor_marginal_risk = (factor_covariance @ factor_exposures) / np.sqrt(portfolio_variance)
            factor_risk_contrib = factor_exposures * factor_marginal_risk
            factor_risk_contrib = factor_risk_contrib / factor_risk_contrib.sum()
        else:
            factor_risk_contrib = pd.Series()
        
        return {
            'asset_risk_contribution': asset_risk_contrib,
            'factor_risk_contribution': factor_risk_contrib,
            'factor_exposures': factor_exposures,
            'factor_loadings': factor_loadings
        }
    
    def calculate_performance_metrics(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: Optional[pd.Series] = None,
        risk_free_rate: float = 0.07
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio_weights: Portfolio weights
            benchmark_weights: Benchmark weights (if None, use equal weights)
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary with performance metrics
        """
        # Portfolio returns
        portfolio_returns = (self.returns_data * portfolio_weights).sum(axis=1)
        
        # Benchmark returns
        if benchmark_weights is None:
            benchmark_weights = pd.Series(
                1/len(self.returns_data.columns), 
                index=self.returns_data.columns
            )
        benchmark_returns = (self.returns_data * benchmark_weights).sum(axis=1)
        
        # Annualized metrics
        portfolio_annual_return = portfolio_returns.mean() * 252
        portfolio_annual_vol = portfolio_returns.std() * np.sqrt(252)
        benchmark_annual_return = benchmark_returns.mean() * 252
        benchmark_annual_vol = benchmark_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = (portfolio_annual_return - risk_free_rate) / portfolio_annual_vol
        
        # Information ratio (tracking error)
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = (portfolio_annual_return - benchmark_annual_return) / tracking_error if tracking_error > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (portfolio_annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else np.inf
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calmar ratio
        calmar_ratio = portfolio_annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        # Beta relative to benchmark
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
        
        # Alpha
        alpha = portfolio_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
        
        return {
            'annual_return': portfolio_annual_return,
            'annual_volatility': portfolio_annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'beta': beta,
            'alpha': alpha,
            'benchmark_return': benchmark_annual_return,
            'benchmark_volatility': benchmark_annual_vol
        }
    
    def generate_risk_report(
        self,
        portfolio_weights: pd.Series,
        portfolio_name: str = "Portfolio"
    ) -> Dict[str, any]:
        """
        Generate comprehensive risk report.
        
        Args:
            portfolio_weights: Portfolio weights
            portfolio_name: Name for the portfolio
            
        Returns:
            Dictionary with complete risk analysis
        """
        report = {
            'portfolio_name': portfolio_name,
            'weights': portfolio_weights,
            'timestamp': pd.Timestamp.now()
        }
        
        # VaR and ES analysis
        report['var_analysis'] = {}
        for method in ['historical', 'parametric']:
            try:
                var_results = self.calculate_var_es(portfolio_weights, method=method)
                report['var_analysis'][method] = var_results
            except Exception as e:
                logger.warning(f"VaR calculation failed for {method}: {e}")
        
        # Performance metrics
        try:
            report['performance_metrics'] = self.calculate_performance_metrics(portfolio_weights)
        except Exception as e:
            logger.warning(f"Performance metrics calculation failed: {e}")
        
        # Risk attribution
        try:
            report['risk_attribution'] = self.calculate_risk_attribution(portfolio_weights)
        except Exception as e:
            logger.warning(f"Risk attribution calculation failed: {e}")
        
        # Basic statistics
        portfolio_returns = (self.returns_data * portfolio_weights).sum(axis=1)
        report['basic_stats'] = {
            'mean_daily_return': portfolio_returns.mean(),
            'std_daily_return': portfolio_returns.std(),
            'skewness': portfolio_returns.skew(),
            'kurtosis': portfolio_returns.kurtosis(),
            'min_return': portfolio_returns.min(),
            'max_return': portfolio_returns.max()
        }
        
        return report 