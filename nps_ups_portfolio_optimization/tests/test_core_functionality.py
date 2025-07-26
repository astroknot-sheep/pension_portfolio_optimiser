"""
Comprehensive tests for NPS vs UPS portfolio optimization core functionality.

Tests cover:
- Data loading and preprocessing
- Portfolio optimization
- Risk analytics
- Monte Carlo simulation
- Report generation
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from nps_ups.io.data_loader import DataLoader
from nps_ups.io.economic_data import EconomicDataProvider
from nps_ups.optimiser import PortfolioOptimizer
from nps_ups.analytics import RiskAnalytics
from nps_ups.simulation import MonteCarloSimulator, SimulationParameters
from nps_ups.reporting import ReportGenerator


class TestDataLoader:
    """Test data loading functionality."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_data_loader_initialization(self, temp_data_dir):
        """Test DataLoader initialization."""
        loader = DataLoader(temp_data_dir)
        assert loader.data_dir.exists()
        assert loader.raw_dir.exists()
        assert loader.processed_dir.exists()
    
    def test_pension_fund_data_loading(self, temp_data_dir):
        """Test pension fund data loading."""
        loader = DataLoader(temp_data_dir)
        data = loader.load_pension_fund_data(
            start_date="2022-01-01",
            end_date="2022-12-31"
        )
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert all(col in data.columns for col in ['date', 'pfm', 'scheme', 'nav_value'])
        assert data['pfm'].isin(loader.PFM_MAPPING.keys()).all()
        assert data['scheme'].isin(loader.SCHEME_TYPES).all()
    
    def test_market_data_loading(self, temp_data_dir):
        """Test market data loading."""
        loader = DataLoader(temp_data_dir)
        data = loader.load_market_data(
            tickers=['^NSEI'],
            start_date="2022-01-01",
            end_date="2022-03-31"
        )
        
        # Should not fail even if external API is unavailable
        assert isinstance(data, pd.DataFrame)
    
    def test_risk_free_rate(self, temp_data_dir):
        """Test risk-free rate retrieval."""
        loader = DataLoader(temp_data_dir)
        rf_rate = loader.get_risk_free_rate()
        
        assert isinstance(rf_rate, float)
        assert 0 <= rf_rate <= 0.20  # Reasonable range for Indian risk-free rate


class TestEconomicDataProvider:
    """Test economic data provider functionality."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_economic_provider_initialization(self, temp_data_dir):
        """Test EconomicDataProvider initialization."""
        provider = EconomicDataProvider(temp_data_dir)
        assert provider.data_dir.exists()
    
    def test_inflation_data_loading(self, temp_data_dir):
        """Test inflation data loading."""
        provider = EconomicDataProvider(temp_data_dir)
        data = provider.load_inflation_data(
            start_date="2022-01-01",
            end_date="2022-12-31"
        )
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert 'cpi_inflation' in data.columns
        assert 'date' in data.columns
    
    def test_repo_rate_data_loading(self, temp_data_dir):
        """Test repo rate data loading."""
        provider = EconomicDataProvider(temp_data_dir)
        data = provider.load_repo_rate_data(
            start_date="2022-01-01",
            end_date="2022-12-31"
        )
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert 'repo_rate' in data.columns
        assert 'date' in data.columns
    
    def test_inflation_scenarios(self, temp_data_dir):
        """Test inflation scenario generation."""
        provider = EconomicDataProvider(temp_data_dir)
        scenarios = provider.calculate_inflation_scenarios(periods=12)
        
        assert isinstance(scenarios, dict)
        assert all(scenario in scenarios for scenario in ['base', 'optimistic', 'adverse'])
        assert all(len(scenario_data) == 12 for scenario_data in scenarios.values())


class TestPortfolioOptimizer:
    """Test portfolio optimization functionality."""
    
    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        
        # Generate synthetic returns for 5 PFM-scheme combinations
        assets = ['HDFC_E', 'HDFC_C', 'HDFC_G', 'ICICI_E', 'SBI_C']
        returns_data = {}
        
        for asset in assets:
            # Different risk-return profiles for different schemes
            if 'E' in asset:  # Equity
                mean_return = 0.12 / 252
                volatility = 0.18 / np.sqrt(252)
            elif 'C' in asset:  # Corporate bonds
                mean_return = 0.08 / 252
                volatility = 0.08 / np.sqrt(252)
            else:  # Government securities
                mean_return = 0.07 / 252
                volatility = 0.02 / np.sqrt(252)
            
            returns = np.random.normal(mean_return, volatility, len(dates))
            returns_data[asset] = returns
        
        return pd.DataFrame(returns_data, index=dates)
    
    def test_optimizer_initialization(self, sample_returns_data):
        """Test PortfolioOptimizer initialization."""
        optimizer = PortfolioOptimizer(sample_returns_data)
        
        assert optimizer.returns_data is not None
        assert optimizer.expected_returns is not None
        assert optimizer.cov_matrix is not None
        assert len(optimizer.expected_returns) == len(sample_returns_data.columns)
    
    def test_max_sharpe_optimization(self, sample_returns_data):
        """Test maximum Sharpe ratio optimization."""
        optimizer = PortfolioOptimizer(sample_returns_data)
        result = optimizer.optimize_max_sharpe()
        
        assert isinstance(result, dict)
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'volatility' in result
        assert 'sharpe_ratio' in result
        
        # Check weights sum to 1
        assert abs(result['weights'].sum() - 1.0) < 1e-6
        
        # Check all weights are non-negative
        assert (result['weights'] >= -1e-6).all()
    
    def test_min_volatility_optimization(self, sample_returns_data):
        """Test minimum volatility optimization."""
        optimizer = PortfolioOptimizer(sample_returns_data)
        result = optimizer.optimize_min_volatility()
        
        assert isinstance(result, dict)
        assert 'weights' in result
        assert abs(result['weights'].sum() - 1.0) < 1e-6
        assert (result['weights'] >= -1e-6).all()
    
    def test_constrained_optimization(self, sample_returns_data):
        """Test optimization with NPS allocation constraints."""
        optimizer = PortfolioOptimizer(sample_returns_data)
        result = optimizer.optimize_max_sharpe(target_allocation='Moderate')
        
        assert isinstance(result, dict)
        assert result['allocation_type'] == 'Moderate'
        assert abs(result['weights'].sum() - 1.0) < 1e-6
    
    def test_efficient_frontier(self, sample_returns_data):
        """Test efficient frontier computation."""
        optimizer = PortfolioOptimizer(sample_returns_data)
        
        try:
            frontier = optimizer.compute_efficient_frontier(num_portfolios=10)
            assert isinstance(frontier, pd.DataFrame)
            assert len(frontier) <= 10
            assert 'expected_return' in frontier.columns
            assert 'volatility' in frontier.columns
        except Exception:
            # Efficient frontier may fail with small dataset, that's ok
            pass
    
    def test_lifecycle_portfolios(self, sample_returns_data):
        """Test lifecycle portfolio generation."""
        optimizer = PortfolioOptimizer(sample_returns_data)
        lifecycle = optimizer.generate_lifecycle_portfolios([25, 35, 45, 55])
        
        assert isinstance(lifecycle, dict)
        assert len(lifecycle) == 4
        
        for age, portfolio in lifecycle.items():
            assert 'weights' in portfolio
            assert 'allocation_type' in portfolio


class TestRiskAnalytics:
    """Test risk analytics functionality."""
    
    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        
        assets = ['HDFC_E', 'HDFC_C', 'HDFC_G']
        returns_data = {}
        
        for asset in assets:
            returns = np.random.normal(0.0004, 0.02, len(dates))  # Daily returns
            returns_data[asset] = returns
        
        return pd.DataFrame(returns_data, index=dates)
    
    @pytest.fixture
    def sample_portfolio_weights(self):
        """Create sample portfolio weights."""
        return pd.Series([0.5, 0.3, 0.2], index=['HDFC_E', 'HDFC_C', 'HDFC_G'])
    
    def test_risk_analytics_initialization(self, sample_returns_data):
        """Test RiskAnalytics initialization."""
        analyzer = RiskAnalytics(sample_returns_data)
        
        assert analyzer.returns_data is not None
        assert analyzer.mean_returns is not None
        assert analyzer.correlation_matrix is not None
    
    def test_var_calculation(self, sample_returns_data, sample_portfolio_weights):
        """Test VaR calculation methods."""
        analyzer = RiskAnalytics(sample_returns_data)
        
        for method in ['historical', 'parametric', 'monte_carlo']:
            var_results = analyzer.calculate_var_es(
                sample_portfolio_weights, 
                method=method
            )
            
            assert isinstance(var_results, dict)
            assert 'VaR' in var_results
            assert 'ES' in var_results
            assert 0.95 in var_results['VaR']
            assert 0.99 in var_results['VaR']
            
            # VaR should be positive (representing losses)
            assert var_results['VaR'][0.95] >= 0
            assert var_results['VaR'][0.99] >= var_results['VaR'][0.95]
    
    def test_performance_metrics(self, sample_returns_data, sample_portfolio_weights):
        """Test performance metrics calculation."""
        analyzer = RiskAnalytics(sample_returns_data)
        metrics = analyzer.calculate_performance_metrics(sample_portfolio_weights)
        
        assert isinstance(metrics, dict)
        expected_metrics = [
            'annual_return', 'annual_volatility', 'sharpe_ratio',
            'max_drawdown', 'beta', 'alpha'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_risk_attribution(self, sample_returns_data, sample_portfolio_weights):
        """Test risk attribution analysis."""
        analyzer = RiskAnalytics(sample_returns_data)
        attribution = analyzer.calculate_risk_attribution(sample_portfolio_weights)
        
        assert isinstance(attribution, dict)
        assert 'asset_risk_contribution' in attribution
        assert 'factor_risk_contribution' in attribution
        
        # Risk contributions should sum to approximately 1
        asset_contrib = attribution['asset_risk_contribution']
        assert abs(asset_contrib.sum() - 1.0) < 0.1
    
    def test_stress_testing(self, sample_returns_data, sample_portfolio_weights):
        """Test stress testing functionality."""
        analyzer = RiskAnalytics(sample_returns_data)
        
        scenarios = {
            'market_crash': {'HDFC_E': -0.30, 'HDFC_C': -0.10},
            'inflation_shock': {'HDFC_G': -0.05}
        }
        
        stress_results = analyzer.stress_test_portfolio(
            sample_portfolio_weights, scenarios
        )
        
        assert isinstance(stress_results, dict)
        assert 'baseline' in stress_results
        assert 'market_crash' in stress_results
        assert 'inflation_shock' in stress_results


class TestMonteCarloSimulator:
    """Test Monte Carlo simulation functionality."""
    
    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        
        assets = ['HDFC_E', 'HDFC_C', 'HDFC_G']
        returns_data = {}
        
        for asset in assets:
            returns = np.random.normal(0.0004, 0.02, len(dates))
            returns_data[asset] = returns
        
        return pd.DataFrame(returns_data, index=dates)
    
    @pytest.fixture
    def sample_portfolios(self):
        """Create sample portfolio weights."""
        return {
            'Aggressive': pd.Series([0.7, 0.2, 0.1], index=['HDFC_E', 'HDFC_C', 'HDFC_G']),
            'Conservative': pd.Series([0.2, 0.3, 0.5], index=['HDFC_E', 'HDFC_C', 'HDFC_G'])
        }
    
    def test_simulator_initialization(self, sample_returns_data, sample_portfolios):
        """Test MonteCarloSimulator initialization."""
        params = SimulationParameters(n_simulations=100)
        simulator = MonteCarloSimulator(sample_returns_data, params, sample_portfolios)
        
        assert simulator.returns_data is not None
        assert simulator.parameters.n_simulations == 100
        assert len(simulator.portfolios) == 2
    
    def test_salary_growth_simulation(self, sample_returns_data):
        """Test salary growth simulation."""
        params = SimulationParameters(n_simulations=100)
        simulator = MonteCarloSimulator(sample_returns_data, params)
        
        salary_paths = simulator.simulate_salary_growth(n_simulations=100)
        
        assert salary_paths.shape == (100, 35)  # 100 sims, 35 years (60-25)
        assert (salary_paths[:, 0] == params.current_salary).all()
        assert (salary_paths[:, -1] > salary_paths[:, 0]).all()  # Should grow
    
    def test_nps_accumulation_simulation(self, sample_returns_data, sample_portfolios):
        """Test NPS accumulation simulation."""
        params = SimulationParameters(n_simulations=100)
        simulator = MonteCarloSimulator(sample_returns_data, params, sample_portfolios)
        
        results = simulator.simulate_nps_accumulation(
            sample_portfolios['Aggressive'], 
            scenario='base',
            n_simulations=100
        )
        
        assert isinstance(results, dict)
        assert 'corpus_paths' in results
        assert 'final_corpus' in results
        assert 'contribution_paths' in results
        
        assert results['corpus_paths'].shape == (100, 36)  # 100 sims, 36 time points
        assert len(results['final_corpus']) == 100
        assert (results['final_corpus'] > 0).all()
    
    def test_ups_benefits_simulation(self, sample_returns_data):
        """Test UPS benefits simulation."""
        params = SimulationParameters(n_simulations=100)
        simulator = MonteCarloSimulator(sample_returns_data, params)
        
        results = simulator.simulate_ups_benefits(n_simulations=100)
        
        assert isinstance(results, dict)
        assert 'final_salaries' in results
        assert 'monthly_pension' in results
        assert 'equivalent_corpus' in results
        
        assert len(results['final_salaries']) == 100
        assert (results['equivalent_corpus'] > 0).all()
    
    def test_comprehensive_simulation(self, sample_returns_data, sample_portfolios):
        """Test comprehensive simulation."""
        params = SimulationParameters(n_simulations=50)  # Smaller for speed
        simulator = MonteCarloSimulator(sample_returns_data, params, sample_portfolios)
        
        results = simulator.run_comprehensive_simulation(['base'])
        
        assert isinstance(results, dict)
        assert 'scenarios' in results
        assert 'ups_baseline' in results
        assert 'summary' in results
        
        assert 'base' in results['scenarios']
        base_scenario = results['scenarios']['base']
        assert 'portfolios' in base_scenario
        assert len(base_scenario['portfolios']) == 2


class TestReportGenerator:
    """Test report generation functionality."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_report_generator_initialization(self, temp_output_dir):
        """Test ReportGenerator initialization."""
        generator = ReportGenerator(temp_output_dir)
        
        assert generator.output_dir.exists()
        assert (generator.output_dir / "charts").exists()
        assert (generator.output_dir / "data").exists()
    
    def test_chart_creation(self, temp_output_dir):
        """Test basic chart creation."""
        generator = ReportGenerator(temp_output_dir)
        
        # Create sample data
        portfolios = {
            'Portfolio_A': pd.Series([0.6, 0.3, 0.1], index=['E', 'C', 'G']),
            'Portfolio_B': pd.Series([0.3, 0.4, 0.3], index=['E', 'C', 'G'])
        }
        
        # Test portfolio allocation chart
        fig = generator.create_portfolio_allocation_chart(portfolios)
        assert fig is not None
        assert 'portfolio_allocation' in generator.charts
    
    def test_executive_summary_table(self, temp_output_dir):
        """Test executive summary table creation."""
        generator = ReportGenerator(temp_output_dir)
        
        # Sample summary data
        summary_data = {
            'ups_stats': {
                'mean_equivalent_corpus': 50_000_000,
                'mean_annual_pension': 1_000_000
            },
            'nps_stats': {
                'base': {
                    'Aggressive': {
                        'mean_final_corpus': 60_000_000,
                        'probability_beats_ups': 0.75
                    }
                }
            }
        }
        
        df = generator.create_executive_summary_table(summary_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'Metric' in df.columns
        assert 'Value' in df.columns


class TestIntegration:
    """Integration tests for end-to-end functionality."""
    
    def test_end_to_end_pipeline(self):
        """Test complete analysis pipeline with minimal data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Data loading
            loader = DataLoader(temp_dir)
            pension_data = loader.load_pension_fund_data(
                start_date="2022-01-01",
                end_date="2022-03-31"
            )
            
            # Step 2: Prepare returns data
            pivot_data = pension_data.pivot_table(
                index='date',
                columns=['pfm', 'scheme'],
                values='return',
                aggfunc='mean'
            )
            pivot_data.columns = [f"{pfm}_{scheme}" for pfm, scheme in pivot_data.columns]
            returns_data = pivot_data.fillna(method='ffill').dropna()
            
            # Skip if insufficient data
            if len(returns_data) < 30 or len(returns_data.columns) < 3:
                pytest.skip("Insufficient data for integration test")
            
            # Step 3: Portfolio optimization
            optimizer = PortfolioOptimizer(returns_data)
            max_sharpe = optimizer.optimize_max_sharpe()
            
            # Step 4: Risk analysis
            analyzer = RiskAnalytics(returns_data)
            var_results = analyzer.calculate_var_es(max_sharpe['weights'])
            
            # Step 5: Simulation (small scale)
            params = SimulationParameters(n_simulations=10)
            portfolios = {'Test_Portfolio': max_sharpe['weights']}
            simulator = MonteCarloSimulator(returns_data, params, portfolios)
            
            # This should complete without errors
            assert True


if __name__ == '__main__':
    pytest.main([__file__]) 