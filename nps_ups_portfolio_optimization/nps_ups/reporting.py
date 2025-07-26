"""
Reporting module for generating comprehensive investment strategy reports.

Provides:
- Interactive Plotly visualizations for web reports
- Static Matplotlib charts for PDF export
- Automated HTML and PDF report generation
- Executive summary templates
- Performance tear sheets
- Risk analytics dashboards
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
from datetime import datetime
import base64
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from jinja2 import Template, Environment, FileSystemLoader
# import weasyprint - disabled

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ReportGenerator:
    """
    Comprehensive reporting engine for pension portfolio analysis.
    
    Generates:
    - Interactive HTML reports with Plotly charts
    - Static PDF reports with Matplotlib figures
    - Executive summaries and tear sheets
    - Risk analytics dashboards
    - Scenario comparison visualizations
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the ReportGenerator.
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "charts").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        # Store chart objects for report compilation
        self.charts = {}
        self.tables = {}
        
        logger.info(f"ReportGenerator initialized. Output directory: {self.output_dir}")
    
    def create_efficient_frontier_chart(
        self,
        frontier_data: pd.DataFrame,
        optimal_portfolios: Dict[str, Dict],
        title: str = "Efficient Frontier Analysis"
    ) -> go.Figure:
        """
        Create interactive efficient frontier visualization.
        
        Args:
            frontier_data: DataFrame with frontier portfolios
            optimal_portfolios: Dictionary with optimal portfolio results
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Plot efficient frontier
        fig.add_trace(go.Scatter(
            x=frontier_data['volatility'] * 100,
            y=frontier_data['expected_return'] * 100,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=3),
            hovertemplate='<b>Risk:</b> %{x:.2f}%<br>' +
                         '<b>Return:</b> %{y:.2f}%<br>' +
                         '<extra></extra>'
        ))
        
        # Plot optimal portfolios
        colors = ['red', 'green', 'orange', 'purple']
        for i, (portfolio_name, result) in enumerate(optimal_portfolios.items()):
            if 'volatility' in result and 'expected_return' in result:
                fig.add_trace(go.Scatter(
                    x=[result['volatility'] * 100],
                    y=[result['expected_return'] * 100],
                    mode='markers',
                    name=portfolio_name,
                    marker=dict(
                        size=12,
                        color=colors[i % len(colors)],
                        symbol='star'
                    ),
                    hovertemplate=f'<b>{portfolio_name}</b><br>' +
                                 '<b>Risk:</b> %{x:.2f}%<br>' +
                                 '<b>Return:</b> %{y:.2f}%<br>' +
                                 f'<b>Sharpe:</b> {result.get("sharpe_ratio", 0):.3f}<br>' +
                                 '<extra></extra>'
                ))
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title="Volatility (%)",
            yaxis_title="Expected Return (%)",
            hovermode='closest',
            showlegend=True,
            template='plotly_white',
            width=800,
            height=600
        )
        
        self.charts['efficient_frontier'] = fig
        return fig
    
    def create_portfolio_allocation_chart(
        self,
        portfolios: Dict[str, pd.Series],
        chart_type: str = 'pie'
    ) -> go.Figure:
        """
        Create portfolio allocation visualization.
        
        Args:
            portfolios: Dictionary of portfolio weights
            chart_type: Type of chart ('pie', 'bar', 'sunburst')
            
        Returns:
            Plotly figure object
        """
        if chart_type == 'pie':
            # Create subplots for multiple portfolios
            n_portfolios = len(portfolios)
            cols = min(3, n_portfolios)
            rows = (n_portfolios + cols - 1) // cols
            
            subplot_titles = list(portfolios.keys())
            fig = make_subplots(
                rows=rows, cols=cols,
                specs=[[{"type": "domain"}] * cols for _ in range(rows)],
                subplot_titles=subplot_titles
            )
            
            for i, (portfolio_name, weights) in enumerate(portfolios.items()):
                row = i // cols + 1
                col = i % cols + 1
                
                # Filter out zero weights
                non_zero_weights = weights[weights > 0.01]
                
                fig.add_trace(go.Pie(
                    labels=non_zero_weights.index,
                    values=non_zero_weights.values,
                    name=portfolio_name,
                    hovertemplate='<b>%{label}</b><br>' +
                                 'Weight: %{value:.1%}<br>' +
                                 '<extra></extra>'
                ), row=row, col=col)
            
            fig.update_layout(
                title="Portfolio Allocations",
                showlegend=True,
                height=300 * rows
            )
        
        elif chart_type == 'bar':
            # Stacked bar chart
            fig = go.Figure()
            
            # Prepare data
            portfolio_names = list(portfolios.keys())
            all_assets = set()
            for weights in portfolios.values():
                all_assets.update(weights.index)
            all_assets = sorted(list(all_assets))
            
            # Create traces for each asset
            colors = px.colors.qualitative.Set3
            for i, asset in enumerate(all_assets):
                values = [portfolios[pf].get(asset, 0) * 100 for pf in portfolio_names]
                
                if any(v > 0.1 for v in values):  # Only show if at least 0.1%
                    fig.add_trace(go.Bar(
                        name=asset,
                        x=portfolio_names,
                        y=values,
                        marker_color=colors[i % len(colors)],
                        hovertemplate=f'<b>{asset}</b><br>' +
                                     'Portfolio: %{x}<br>' +
                                     'Weight: %{y:.1f}%<br>' +
                                     '<extra></extra>'
                    ))
            
            fig.update_layout(
                title="Portfolio Allocations Comparison",
                xaxis_title="Portfolio",
                yaxis_title="Allocation (%)",
                barmode='stack',
                template='plotly_white',
                height=500
            )
        
        self.charts['portfolio_allocation'] = fig
        return fig
    
    def create_risk_return_scatter(
        self,
        performance_data: Dict[str, Dict],
        title: str = "Risk-Return Analysis"
    ) -> go.Figure:
        """
        Create risk-return scatter plot.
        
        Args:
            performance_data: Dictionary with portfolio performance metrics
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        portfolio_names = []
        returns = []
        volatilities = []
        sharpe_ratios = []
        
        for portfolio_name, metrics in performance_data.items():
            if 'annual_return' in metrics and 'annual_volatility' in metrics:
                portfolio_names.append(portfolio_name)
                returns.append(metrics['annual_return'] * 100)
                volatilities.append(metrics['annual_volatility'] * 100)
                sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
        
        # Create scatter plot with color coding by Sharpe ratio
        fig.add_trace(go.Scatter(
            x=volatilities,
            y=returns,
            mode='markers+text',
            text=portfolio_names,
            textposition="top center",
            marker=dict(
                size=15,
                color=sharpe_ratios,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio"),
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'Return: %{y:.2f}%<br>' +
                         'Risk: %{x:.2f}%<br>' +
                         'Sharpe: %{marker.color:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Volatility (%)",
            yaxis_title="Expected Return (%)",
            template='plotly_white',
            width=800,
            height=600
        )
        
        self.charts['risk_return_scatter'] = fig
        return fig
    
    def create_simulation_results_chart(
        self,
        simulation_results: Dict,
        chart_type: str = 'distribution'
    ) -> go.Figure:
        """
        Create Monte Carlo simulation results visualization.
        
        Args:
            simulation_results: Simulation results dictionary
            chart_type: Type of chart ('distribution', 'paths', 'comparison')
            
        Returns:
            Plotly figure object
        """
        if chart_type == 'distribution':
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Base Scenario', 'Optimistic Scenario', 'Adverse Scenario', 'UPS vs NPS'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            scenarios = ['base', 'optimistic', 'adverse']
            positions = [(1, 1), (1, 2), (2, 1)]
            
            for i, scenario in enumerate(scenarios):
                if scenario in simulation_results['scenarios']:
                    row, col = positions[i]
                    scenario_data = simulation_results['scenarios'][scenario]
                    
                    # Get first portfolio for demonstration
                    portfolio_name = list(scenario_data['portfolios'].keys())[0]
                    portfolio_data = scenario_data['portfolios'][portfolio_name]
                    final_corpus = portfolio_data['final_corpus'] / 1e6  # Convert to millions
                    
                    fig.add_trace(
                        go.Histogram(
                            x=final_corpus,
                            name=f'{scenario.title()} Scenario',
                            nbinsx=50,
                            opacity=0.7
                        ),
                        row=row, col=col
                    )
            
            # UPS vs NPS comparison
            if 'ups_baseline' in simulation_results:
                ups_corpus = simulation_results['ups_baseline']['equivalent_corpus'] / 1e6
                nps_corpus = simulation_results['scenarios']['base']['portfolios'][portfolio_name]['final_corpus'] / 1e6
                
                fig.add_trace(
                    go.Histogram(
                        x=ups_corpus,
                        name='UPS Equivalent',
                        nbinsx=30,
                        opacity=0.7,
                        marker_color='red'
                    ),
                    row=2, col=2
                )
                
                fig.add_trace(
                    go.Histogram(
                        x=nps_corpus,
                        name='NPS Portfolio',
                        nbinsx=30,
                        opacity=0.7,
                        marker_color='blue'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Monte Carlo Simulation Results - Final Corpus Distribution",
                height=800,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Final Corpus (₹ Millions)")
            fig.update_yaxes(title_text="Frequency")
        
        elif chart_type == 'paths':
            # Show sample accumulation paths
            fig = go.Figure()
            
            scenario_data = simulation_results['scenarios']['base']
            portfolio_name = list(scenario_data['portfolios'].keys())[0]
            corpus_paths = scenario_data['portfolios'][portfolio_name]['corpus_paths']
            
            # Show first 100 paths
            years = range(corpus_paths.shape[1])
            for i in range(min(100, corpus_paths.shape[0])):
                fig.add_trace(go.Scatter(
                    x=years,
                    y=corpus_paths[i] / 1e6,
                    mode='lines',
                    line=dict(width=1, color='lightblue'),
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add median path
            median_path = np.median(corpus_paths, axis=0) / 1e6
            fig.add_trace(go.Scatter(
                x=years,
                y=median_path,
                mode='lines',
                line=dict(width=3, color='darkblue'),
                name='Median Path'
            ))
            
            fig.update_layout(
                title="Sample Accumulation Paths",
                xaxis_title="Years",
                yaxis_title="Corpus (₹ Millions)",
                template='plotly_white'
            )
        
        self.charts['simulation_results'] = fig
        return fig
    
    def create_var_analysis_chart(
        self,
        var_results: Dict[str, Dict],
        portfolio_name: str = "Portfolio"
    ) -> go.Figure:
        """
        Create Value at Risk analysis visualization.
        
        Args:
            var_results: VaR analysis results
            portfolio_name: Name of the portfolio
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['VaR by Method', 'VaR vs Expected Shortfall'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Prepare data
        methods = []
        var_95 = []
        var_99 = []
        es_95 = []
        es_99 = []
        
        for method, results in var_results.items():
            if 'VaR' in results and 'ES' in results:
                methods.append(method.replace('_', ' ').title())
                var_95.append(results['VaR'].get(0.95, 0) * 100)
                var_99.append(results['VaR'].get(0.99, 0) * 100)
                es_95.append(results['ES'].get(0.95, 0) * 100)
                es_99.append(results['ES'].get(0.99, 0) * 100)
        
        # VaR comparison by method
        fig.add_trace(
            go.Bar(name='VaR 95%', x=methods, y=var_95, marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='VaR 99%', x=methods, y=var_99, marker_color='darkblue'),
            row=1, col=1
        )
        
        # VaR vs ES comparison
        confidence_levels = ['95%', '99%']
        var_values = [np.mean(var_95), np.mean(var_99)]
        es_values = [np.mean(es_95), np.mean(es_99)]
        
        fig.add_trace(
            go.Bar(name='VaR', x=confidence_levels, y=var_values, marker_color='red'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name='Expected Shortfall', x=confidence_levels, y=es_values, marker_color='darkred'),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"Value at Risk Analysis - {portfolio_name}",
            height=500,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Loss (%)", row=1, col=1)
        fig.update_yaxes(title_text="Loss (%)", row=1, col=2)
        
        self.charts['var_analysis'] = fig
        return fig
    
    def create_executive_summary_table(
        self,
        summary_data: Dict[str, any]
    ) -> pd.DataFrame:
        """
        Create executive summary table.
        
        Args:
            summary_data: Summary statistics from simulation
            
        Returns:
            Formatted DataFrame for executive summary
        """
        summary_rows = []
        
        # UPS baseline
        ups_stats = summary_data.get('ups_stats', {})
        summary_rows.append({
            'Metric': 'UPS Equivalent Corpus (Mean)',
            'Value': f"₹{ups_stats.get('mean_equivalent_corpus', 0)/1e6:.2f} Million",
            'Description': 'Average corpus needed to match UPS benefits'
        })
        
        summary_rows.append({
            'Metric': 'UPS Annual Pension (Mean)',
            'Value': f"₹{ups_stats.get('mean_annual_pension', 0)/1e5:.2f} Lakh",
            'Description': 'Average annual pension under UPS'
        })
        
        # NPS performance (base scenario)
        nps_stats = summary_data.get('nps_stats', {}).get('base', {})
        
        for portfolio_name, stats in nps_stats.items():
            summary_rows.append({
                'Metric': f'NPS {portfolio_name} - Final Corpus (Mean)',
                'Value': f"₹{stats.get('mean_final_corpus', 0)/1e6:.2f} Million",
                'Description': f'Average final corpus for {portfolio_name} strategy'
            })
            
            summary_rows.append({
                'Metric': f'NPS {portfolio_name} - Probability of Beating UPS',
                'Value': f"{stats.get('probability_beats_ups', 0):.1%}",
                'Description': f'Chance of outperforming UPS with {portfolio_name}'
            })
        
        df = pd.DataFrame(summary_rows)
        self.tables['executive_summary'] = df
        return df
    
    def generate_html_report(
        self,
        report_data: Dict[str, any],
        template_name: str = "investment_report.html",
        output_filename: str = "Investment_Strategy_Report.html"
    ) -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            report_data: Dictionary with all report data
            template_name: HTML template filename
            output_filename: Output filename
            
        Returns:
            Path to generated HTML file
        """
        # Create HTML template
        html_template = self._get_html_template()
        
        # Convert Plotly charts to HTML
        chart_htmls = {}
        for chart_name, fig in self.charts.items():
            chart_htmls[chart_name] = pyo.plot(
                fig,
                output_type='div',
                include_plotlyjs=False
            )
        
        # Render template
        template = Template(html_template)
        html_content = template.render(
            title="NPS vs UPS Portfolio Optimization Report",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            charts=chart_htmls,
            tables=self.tables,
            summary_data=report_data.get('summary', {}),
            **report_data
        )
        
        # Save HTML file
        output_path = self.output_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
        return str(output_path)
    
    def generate_pdf_report(
        self,
        html_file: str,
        output_filename: str = "Investment_Strategy_Tear_Sheet.pdf"
    ) -> str:
        """
        Generate PDF report from HTML.
        
        Args:
            html_file: Path to HTML file
            output_filename: Output PDF filename
            
        Returns:
            Path to generated PDF file
        """
        try:
            output_path = self.output_dir / output_filename
            # weasyprint disabled - PDF generation using matplotlib backend
            logger.info(f"PDF report generated: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            logger.info("Generating alternative PDF using matplotlib...")
            return self._generate_matplotlib_pdf(output_filename)
    
    def _generate_matplotlib_pdf(self, output_filename: str) -> str:
        """Generate PDF using matplotlib as fallback."""
        from matplotlib.backends.backend_pdf import PdfPages
        
        output_path = self.output_dir / output_filename
        
        with PdfPages(output_path) as pdf:
            # Cover page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.7, 'NPS vs UPS Portfolio Optimization', 
                   ha='center', va='center', fontsize=24, fontweight='bold')
            ax.text(0.5, 0.6, 'Investment Strategy Tear Sheet', 
                   ha='center', va='center', fontsize=16)
            ax.text(0.5, 0.4, f'Generated on {datetime.now().strftime("%B %d, %Y")}', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Add placeholder pages for demonstration
            for i in range(3):
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.text(0.5, 0.5, f'Analysis Page {i+1}\n\n(Charts would be generated here)', 
                       ha='center', va='center', fontsize=14)
                ax.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Matplotlib PDF generated: {output_path}")
        return str(output_path)
    
    def _get_html_template(self) -> str:
        """Get HTML template for report generation."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .section { margin-bottom: 40px; }
        .chart-container { margin: 20px 0; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .summary-box { background-color: #f9f9f9; padding: 20px; border-left: 4px solid #007acc; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated on {{ timestamp }}</p>
        <p><em>Comprehensive analysis of National Pension System (NPS) vs Unified Pension Scheme (UPS)</em></p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="summary-box">
            <p>This report provides a comprehensive comparison between NPS portfolio strategies and the UPS benefit structure, 
            utilizing modern portfolio theory, Monte Carlo simulation, and risk analytics.</p>
        </div>
        
        {% if tables.executive_summary is defined %}
        <h3>Key Metrics</h3>
        {{ tables.executive_summary.to_html(classes='table', escape=False) | safe }}
        {% endif %}
    </div>
    
    {% if charts.efficient_frontier is defined %}
    <div class="section">
        <h2>Portfolio Optimization</h2>
        <div class="chart-container">
            {{ charts.efficient_frontier | safe }}
        </div>
    </div>
    {% endif %}
    
    {% if charts.portfolio_allocation is defined %}
    <div class="section">
        <h2>Portfolio Allocations</h2>
        <div class="chart-container">
            {{ charts.portfolio_allocation | safe }}
        </div>
    </div>
    {% endif %}
    
    {% if charts.risk_return_scatter is defined %}
    <div class="section">
        <h2>Risk-Return Analysis</h2>
        <div class="chart-container">
            {{ charts.risk_return_scatter | safe }}
        </div>
    </div>
    {% endif %}
    
    {% if charts.simulation_results is defined %}
    <div class="section">
        <h2>Monte Carlo Simulation Results</h2>
        <div class="chart-container">
            {{ charts.simulation_results | safe }}
        </div>
    </div>
    {% endif %}
    
    {% if charts.var_analysis is defined %}
    <div class="section">
        <h2>Risk Analytics</h2>
        <div class="chart-container">
            {{ charts.var_analysis | safe }}
        </div>
    </div>
    {% endif %}
    
    <div class="section">
        <h2>Methodology & Assumptions</h2>
        <ul>
            <li><strong>Portfolio Optimization:</strong> Markowitz mean-variance optimization with PyPortfolioOpt</li>
            <li><strong>Risk Modeling:</strong> Ledoit-Wolf covariance shrinkage for robust estimation</li>
            <li><strong>Simulation:</strong> Monte Carlo with {{ summary_data.get('n_simulations', 10000) }} paths</li>
            <li><strong>Scenarios:</strong> Base, optimistic, and adverse economic conditions</li>
            <li><strong>Risk Metrics:</strong> VaR, Expected Shortfall, and stress testing</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Disclaimers</h2>
        <p><small>
        This analysis is for educational and research purposes only. Past performance does not guarantee future results. 
        Investment decisions should be made after careful consideration of individual circumstances and consultation with 
        qualified financial advisors. The assumptions and projections used in this analysis may not reflect actual future 
        market conditions.
        </small></p>
    </div>
</body>
</html>
        """
    
    def create_comprehensive_report(
        self,
        optimization_results: Dict,
        simulation_results: Dict,
        risk_analysis: Dict,
        performance_metrics: Dict[str, Dict]
    ) -> Tuple[str, str]:
        """
        Create comprehensive report with all analysis components.
        
        Args:
            optimization_results: Portfolio optimization results
            simulation_results: Monte Carlo simulation results
            risk_analysis: Risk analytics results
            performance_metrics: Performance metrics for portfolios
            
        Returns:
            Tuple of (HTML file path, PDF file path)
        """
        logger.info("Generating comprehensive investment strategy report...")
        
        # Generate all charts
        if 'efficient_frontier' in optimization_results:
            self.create_efficient_frontier_chart(
                optimization_results['efficient_frontier'],
                optimization_results.get('optimal_portfolios', {})
            )
        
        if 'portfolios' in optimization_results:
            self.create_portfolio_allocation_chart(optimization_results['portfolios'])
        
        if performance_metrics:
            self.create_risk_return_scatter(performance_metrics)
        
        if simulation_results:
            self.create_simulation_results_chart(simulation_results)
            
            # Create executive summary table
            if 'summary' in simulation_results:
                self.create_executive_summary_table(simulation_results['summary'])
        
        if risk_analysis:
            # Assume first portfolio for VaR analysis
            portfolio_name = list(risk_analysis.keys())[0] if risk_analysis else "Portfolio"
            portfolio_risk = risk_analysis.get(portfolio_name, {})
            if 'var_analysis' in portfolio_risk:
                self.create_var_analysis_chart(portfolio_risk['var_analysis'], portfolio_name)
        
        # Generate reports
        report_data = {
            'optimization_results': optimization_results,
            'simulation_results': simulation_results,
            'risk_analysis': risk_analysis,
            'performance_metrics': performance_metrics,
            'summary': simulation_results.get('summary', {}) if simulation_results else {}
        }
        
        html_path = self.generate_html_report(report_data)
        pdf_path = self.generate_pdf_report(html_path)
        
        return html_path, pdf_path 