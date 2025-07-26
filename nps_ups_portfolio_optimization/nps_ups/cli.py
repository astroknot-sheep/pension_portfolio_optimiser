"""
Command-line interface for NPS vs UPS portfolio optimization analysis.

Provides a comprehensive CLI for running the complete research pipeline:
- Data loading and preprocessing
- Portfolio optimization
- Risk analytics
- Monte Carlo simulation
- Report generation

Usage:
    python -m nps_ups.cli run-analysis
    python -m nps_ups.cli --help
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import time

import click
import pandas as pd
from tqdm import tqdm

from nps_ups.io.data_loader import DataLoader
from nps_ups.io.economic_data import EconomicDataProvider
from nps_ups.optimiser import PortfolioOptimizer
from nps_ups.analytics import RiskAnalytics
from nps_ups.simulation import MonteCarloSimulator, SimulationParameters
from nps_ups.reporting import ReportGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('nps_ups_analysis.log')
    ]
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--data-dir', default='data', help='Data directory path')
@click.option('--output-dir', default='output', help='Output directory path')
@click.pass_context
def cli(ctx, verbose, data_dir, output_dir):
    """
    NPS vs UPS Portfolio Optimization CLI
    
    A comprehensive quantitative finance tool for comparing India's National Pension System (NPS)
    with the Unified Pension Scheme (UPS) using modern portfolio theory and risk analytics.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    ctx.ensure_object(dict)
    ctx.obj['data_dir'] = data_dir
    ctx.obj['output_dir'] = output_dir
    
    # Ensure directories exist
    Path(data_dir).mkdir(exist_ok=True)
    Path(output_dir).mkdir(exist_ok=True)


@cli.command()
@click.option('--start-date', default='2019-01-01', help='Start date for data loading (YYYY-MM-DD)')
@click.option('--end-date', default=None, help='End date for data loading (YYYY-MM-DD)')
@click.option('--force-refresh', is_flag=True, help='Force refresh of cached data')
@click.pass_context
def load_data(ctx, start_date, end_date, force_refresh):
    """Load and cache pension fund and economic data."""
    logger.info("Starting data loading process...")
    
    data_dir = ctx.obj['data_dir']
    
    # Initialize data loaders
    data_loader = DataLoader(data_dir)
    economic_provider = EconomicDataProvider(data_dir)
    
    with tqdm(total=4, desc="Loading data") as pbar:
        # Load pension fund data
        pbar.set_description("Loading pension fund data")
        pension_data = data_loader.load_pension_fund_data(
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )
        pbar.update(1)
        
        # Load economic data
        pbar.set_description("Loading inflation data")
        inflation_data = economic_provider.load_inflation_data(
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )
        pbar.update(1)
        
        pbar.set_description("Loading repo rate data")
        repo_data = economic_provider.load_repo_rate_data(
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )
        pbar.update(1)
        
        pbar.set_description("Loading market indices")
        market_data = data_loader.load_market_data(
            tickers=['^NSEI', '^BSESN'],
            start_date=start_date,
            end_date=end_date
        )
        pbar.update(1)
    
    logger.info(f"Data loading completed successfully!")
    logger.info(f"Pension fund records: {len(pension_data)}")
    logger.info(f"Economic data points: {len(inflation_data)}")
    
    click.echo(f"✅ Data loaded and cached in {data_dir}")


@cli.command()
@click.option('--scenarios', multiple=True, default=['base', 'optimistic', 'adverse'],
              help='Economic scenarios to simulate')
@click.option('--n-simulations', default=10000, help='Number of Monte Carlo simulations')
@click.option('--current-age', default=25, help='Current age of employee')
@click.option('--retirement-age', default=60, help='Retirement age')
@click.option('--current-salary', default=1000000, help='Current annual salary (INR)')
@click.pass_context
def run_analysis(ctx, scenarios, n_simulations, current_age, retirement_age, current_salary):
    """Run complete portfolio optimization and simulation analysis."""
    start_time = time.time()
    logger.info("Starting comprehensive NPS vs UPS analysis...")
    
    data_dir = ctx.obj['data_dir']
    output_dir = ctx.obj['output_dir']
    
    try:
        # Initialize components
        with tqdm(total=8, desc="Initializing analysis") as pbar:
            
            # Step 1: Load data
            pbar.set_description("Loading pension fund data")
            data_loader = DataLoader(data_dir)
            pension_data = data_loader.load_pension_fund_data()
            pbar.update(1)
            
            # Step 2: Prepare returns data
            pbar.set_description("Preparing returns data")
            returns_data = prepare_returns_data(pension_data)
            pbar.update(1)
            
            # Step 3: Portfolio optimization
            pbar.set_description("Running portfolio optimization")
            optimizer = PortfolioOptimizer(returns_data)
            optimization_results = run_portfolio_optimization(optimizer)
            pbar.update(1)
            
            # Step 4: Risk analytics
            pbar.set_description("Computing risk analytics")
            risk_analyzer = RiskAnalytics(returns_data)
            risk_results = compute_risk_analytics(risk_analyzer, optimization_results)
            pbar.update(1)
            
            # Step 5: Monte Carlo simulation
            pbar.set_description("Running Monte Carlo simulation")
            sim_params = SimulationParameters(
                current_age=current_age,
                retirement_age=retirement_age,
                current_salary=current_salary,
                n_simulations=n_simulations
            )
            simulator = MonteCarloSimulator(
                returns_data, 
                sim_params, 
                optimization_results.get('portfolios', {})
            )
            simulation_results = simulator.run_comprehensive_simulation(list(scenarios))
            pbar.update(1)
            
            # Step 6: Performance metrics
            pbar.set_description("Computing performance metrics")
            performance_metrics = compute_performance_metrics(
                risk_analyzer, optimization_results
            )
            pbar.update(1)
            
            # Step 7: Generate reports
            pbar.set_description("Generating reports")
            report_generator = ReportGenerator(output_dir)
            html_path, pdf_path = report_generator.create_comprehensive_report(
                optimization_results,
                simulation_results,
                risk_results,
                performance_metrics
            )
            pbar.update(1)
            
            # Step 8: Export data
            pbar.set_description("Exporting results")
            export_results(
                output_dir, 
                optimization_results, 
                simulation_results, 
                risk_results
            )
            pbar.update(1)
        
        # Summary
        elapsed_time = time.time() - start_time
        logger.info(f"Analysis completed successfully in {elapsed_time:.1f} seconds!")
        
        click.echo("\n" + "="*60)
        click.echo("🎉 NPS vs UPS Analysis Complete!")
        click.echo("="*60)
        click.echo(f"📊 HTML Report: {html_path}")
        click.echo(f"📋 PDF Report: {pdf_path}")
        click.echo(f"📁 Output Directory: {output_dir}")
        click.echo(f"⏱️  Total Runtime: {elapsed_time:.1f} seconds")
        
        # Key findings summary
        if simulation_results and 'summary' in simulation_results:
            summary = simulation_results['summary']
            ups_corpus = summary.get('ups_stats', {}).get('mean_equivalent_corpus', 0)
            
            click.echo("\n📈 Key Findings:")
            click.echo(f"   UPS Equivalent Corpus: ₹{ups_corpus/1e6:.2f} Million")
            
            nps_base = summary.get('nps_stats', {}).get('base', {})
            for portfolio, stats in nps_base.items():
                nps_corpus = stats.get('mean_final_corpus', 0)
                prob_beats = stats.get('probability_beats_ups', 0)
                click.echo(f"   {portfolio} Portfolio: ₹{nps_corpus/1e6:.2f} Million ({prob_beats:.1%} vs UPS)")
        
        click.echo("\n💡 Next Steps:")
        click.echo("   1. Review the generated reports for detailed analysis")
        click.echo("   2. Customize parameters and re-run if needed")
        click.echo("   3. Share findings with stakeholders")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        click.echo(f"❌ Analysis failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--portfolio-type', type=click.Choice(['aggressive', 'moderate', 'conservative']),
              default='moderate', help='Portfolio allocation type')
@click.option('--target-return', type=float, help='Target annual return (e.g., 0.12 for 12%)')
@click.pass_context
def optimize_portfolio(ctx, portfolio_type, target_return):
    """Run portfolio optimization with specific constraints."""
    logger.info(f"Running portfolio optimization for {portfolio_type} allocation...")
    
    data_dir = ctx.obj['data_dir']
    
    # Load data
    data_loader = DataLoader(data_dir)
    pension_data = data_loader.load_pension_fund_data()
    returns_data = prepare_returns_data(pension_data)
    
    # Optimize
    optimizer = PortfolioOptimizer(returns_data)
    
    if target_return:
        result = optimizer.optimize_target_return(target_return, portfolio_type.title())
        click.echo(f"Optimized for {target_return:.1%} target return:")
    else:
        result = optimizer.optimize_max_sharpe(portfolio_type.title())
        click.echo(f"Optimized for maximum Sharpe ratio ({portfolio_type}):")
    
    # Display results
    click.echo("\nPortfolio Weights:")
    for asset, weight in result['weights'].items():
        if weight > 0.01:  # Only show weights > 1%
            click.echo(f"  {asset}: {weight:.1%}")
    
    click.echo(f"\nExpected Return: {result['expected_return']:.2%}")
    click.echo(f"Volatility: {result['volatility']:.2%}")
    click.echo(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")


@cli.command()
@click.option('--portfolio-weights', help='JSON string of portfolio weights')
@click.option('--confidence-level', default=0.95, help='VaR confidence level')
@click.pass_context
def risk_analysis(ctx, portfolio_weights, confidence_level):
    """Run risk analysis for a specific portfolio."""
    import json
    
    logger.info("Running risk analysis...")
    
    data_dir = ctx.obj['data_dir']
    
    # Load data
    data_loader = DataLoader(data_dir)
    pension_data = data_loader.load_pension_fund_data()
    returns_data = prepare_returns_data(pension_data)
    
    # Parse portfolio weights
    if portfolio_weights:
        weights_dict = json.loads(portfolio_weights)
        weights = pd.Series(weights_dict)
    else:
        # Use equal weights as default
        weights = pd.Series(1/len(returns_data.columns), index=returns_data.columns)
    
    # Risk analysis
    risk_analyzer = RiskAnalytics(returns_data, [confidence_level])
    
    # VaR analysis
    var_results = risk_analyzer.calculate_var_es(weights)
    click.echo(f"\nValue at Risk ({confidence_level:.0%} confidence):")
    for method, results in var_results.items():
        var_val = results['VaR'][confidence_level] * 100
        es_val = results['ES'][confidence_level] * 100
        click.echo(f"  {method}: VaR = {var_val:.2f}%, ES = {es_val:.2f}%")
    
    # Performance metrics
    perf_metrics = risk_analyzer.calculate_performance_metrics(weights)
    click.echo(f"\nPerformance Metrics:")
    click.echo(f"  Annual Return: {perf_metrics['annual_return']:.2%}")
    click.echo(f"  Annual Volatility: {perf_metrics['annual_volatility']:.2%}")
    click.echo(f"  Sharpe Ratio: {perf_metrics['sharpe_ratio']:.3f}")
    click.echo(f"  Max Drawdown: {perf_metrics['max_drawdown']:.2%}")


def prepare_returns_data(pension_data: pd.DataFrame) -> pd.DataFrame:
    """Prepare returns data for analysis."""
    # Pivot data to get returns by PFM and scheme
    pivot_data = pension_data.pivot_table(
        index='date',
        columns=['pfm', 'scheme'],
        values='return',
        aggfunc='mean'
    )
    
    # Flatten column names
    pivot_data.columns = [f"{pfm}_{scheme}" for pfm, scheme in pivot_data.columns]
    
    # Forward fill missing values and drop NaN
    returns_data = pivot_data.fillna(method='ffill').dropna()
    
    return returns_data


def run_portfolio_optimization(optimizer: PortfolioOptimizer) -> dict:
    """Run comprehensive portfolio optimization."""
    results = {
        'portfolios': {},
        'optimal_portfolios': {},
        'efficient_frontier': None
    }
    
    # Optimize different strategies
    strategies = ['Aggressive', 'Moderate', 'Conservative']
    
    for strategy in strategies:
        # Max Sharpe optimization
        max_sharpe = optimizer.optimize_max_sharpe(strategy)
        results['optimal_portfolios'][f'Max_Sharpe_{strategy}'] = max_sharpe
        results['portfolios'][f'Max_Sharpe_{strategy}'] = max_sharpe['weights']
        
        # Min volatility optimization  
        min_vol = optimizer.optimize_min_volatility(strategy)
        results['optimal_portfolios'][f'Min_Vol_{strategy}'] = min_vol
        results['portfolios'][f'Min_Vol_{strategy}'] = min_vol['weights']
    
    # Risk parity portfolio
    risk_parity = optimizer.optimize_risk_parity()
    results['optimal_portfolios']['Risk_Parity'] = risk_parity
    results['portfolios']['Risk_Parity'] = risk_parity['weights']
    
    # Lifecycle portfolios
    lifecycle_portfolios = optimizer.generate_lifecycle_portfolios()
    for age, portfolio in lifecycle_portfolios.items():
        portfolio_name = f'Lifecycle_Age_{age}'
        results['optimal_portfolios'][portfolio_name] = portfolio
        results['portfolios'][portfolio_name] = portfolio['weights']
    
    # Efficient frontier (for moderate allocation)
    try:
        efficient_frontier = optimizer.compute_efficient_frontier(
            num_portfolios=50, target_allocation='Moderate'
        )
        results['efficient_frontier'] = efficient_frontier
    except Exception as e:
        logger.warning(f"Efficient frontier computation failed: {e}")
    
    return results


def compute_risk_analytics(
    risk_analyzer: RiskAnalytics, 
    optimization_results: dict
) -> dict:
    """Compute comprehensive risk analytics."""
    risk_results = {}
    
    for portfolio_name, weights in optimization_results.get('portfolios', {}).items():
        risk_results[portfolio_name] = risk_analyzer.generate_risk_report(
            weights, portfolio_name
        )
    
    return risk_results


def compute_performance_metrics(
    risk_analyzer: RiskAnalytics,
    optimization_results: dict
) -> dict:
    """Compute performance metrics for all portfolios."""
    performance_metrics = {}
    
    for portfolio_name, weights in optimization_results.get('portfolios', {}).items():
        performance_metrics[portfolio_name] = risk_analyzer.calculate_performance_metrics(
            weights
        )
    
    return performance_metrics


def export_results(
    output_dir: str,
    optimization_results: dict,
    simulation_results: dict,
    risk_results: dict
) -> None:
    """Export results to various formats."""
    output_path = Path(output_dir)
    
    # Export portfolio weights
    if 'portfolios' in optimization_results:
        weights_df = pd.DataFrame(optimization_results['portfolios'])
        weights_df.to_csv(output_path / 'portfolio_weights.csv')
    
    # Export efficient frontier
    if 'efficient_frontier' in optimization_results and optimization_results['efficient_frontier'] is not None:
        optimization_results['efficient_frontier'].to_csv(output_path / 'efficient_frontier.csv')
    
    # Export simulation summary
    if simulation_results and 'summary' in simulation_results:
        summary_df = pd.DataFrame(simulation_results['summary'])
        summary_df.to_json(output_path / 'simulation_summary.json', indent=2)
    
    logger.info(f"Results exported to {output_dir}")


if __name__ == '__main__':
    cli() 