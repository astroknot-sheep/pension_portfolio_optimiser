"""
Advanced reporting and visualization module for NPS vs UPS portfolio optimization.

This module provides comprehensive reporting capabilities including:
- Interactive Plotly dashboards for web exploration  
- Static matplotlib charts for PDF export
- HTML report generation with embedded visualizations
- PDF tearsheet creation using matplotlib backend
- Executive summary generation with key insights
- Performance attribution and risk decomposition charts

The module supports multiple output formats and provides both programmatic
and template-based report generation for institutional-grade presentations.
"""

import logging
import os
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from plotly.subplots import make_subplots

# PDF Generation - Multiple backends for reliability
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    warnings.warn("ReportLab not available. PDF generation will use matplotlib backend only.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
plt.style.use('seaborn-v0_8-whitegrid')

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Advanced report generator for portfolio optimization results.
    
    Supports multiple output formats including interactive HTML, static PDF,
    and programmatic data exports. Designed for institutional-grade presentations
    with customizable templates and professional styling.
    """
    
    def __init__(self, output_dir: str = "output", template_dir: Optional[str] = None):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save generated reports
            template_dir: Directory containing custom Jinja2 templates
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Jinja2 templating
        if template_dir and Path(template_dir).exists():
            self.template_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(template_dir)
            )
        else:
            # Use built-in templates
            self.template_env = jinja2.Environment(
                loader=jinja2.DictLoader(self._get_default_templates())
            )
        
        # Configure plotting
        self.plotly_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
        }
        
        # Report metadata
        self.report_metadata = {
            'generated_at': datetime.now(),
            'generator': 'NPS vs UPS Portfolio Optimization System',
            'version': '1.0.0'
        }
        
        logger.info(f"ReportGenerator initialized with output directory: {self.output_dir}")
    
    def generate_portfolio_tearsheet(
        self,
        portfolio_results: Dict[str, Any],
        risk_metrics: Dict[str, Any],
        benchmark_comparison: Optional[Dict[str, Any]] = None,
        save_pdf: bool = True,
        save_html: bool = True
    ) -> Dict[str, str]:
        """
        Generate a comprehensive portfolio tearsheet.
        
        Args:
            portfolio_results: Portfolio optimization results
            risk_metrics: Risk analytics results  
            benchmark_comparison: Optional benchmark comparison data
            save_pdf: Whether to save PDF version
            save_html: Whether to save HTML version
            
        Returns:
            Dictionary with paths to generated files
        """
        logger.info("Generating portfolio tearsheet...")
        
        # Create visualization components
        plots = self._create_tearsheet_plots(
            portfolio_results, risk_metrics, benchmark_comparison
        )
        
        # Generate summary statistics
        summary_stats = self._calculate_summary_statistics(
            portfolio_results, risk_metrics
        )
        
        # Create executive summary
        executive_summary = self._generate_executive_summary(
            portfolio_results, risk_metrics, summary_stats
        )
        
        generated_files = {}
        
        # Generate HTML report
        if save_html:
            html_path = self._generate_html_tearsheet(
                plots, summary_stats, executive_summary
            )
            generated_files['html'] = str(html_path)
            logger.info(f"HTML tearsheet saved to: {html_path}")
        
        # Generate PDF report  
        if save_pdf:
            pdf_path = self._generate_pdf_tearsheet(
                plots, summary_stats, executive_summary
            )
            generated_files['pdf'] = str(pdf_path)
            logger.info(f"PDF tearsheet saved to: {pdf_path}")
        
        return generated_files
    
    def create_efficient_frontier_plot(
        self,
        frontier_data: pd.DataFrame,
        optimal_portfolios: Dict[str, Dict],
        interactive: bool = True
    ) -> Union[go.Figure, Figure]:
        """
        Create an interactive efficient frontier visualization.
        
        Args:
            frontier_data: DataFrame with risk/return points
            optimal_portfolios: Dictionary of optimal portfolio allocations
            interactive: Whether to create interactive Plotly chart
            
        Returns:
            Plotly or matplotlib figure
        """
        if interactive:
            return self._create_plotly_efficient_frontier(frontier_data, optimal_portfolios)
        else:
            return self._create_matplotlib_efficient_frontier(frontier_data, optimal_portfolios)
    
    def create_portfolio_composition_chart(
        self,
        weights: Dict[str, float],
        allocation_type: str = "Optimal",
        interactive: bool = True
    ) -> Union[go.Figure, Figure]:
        """
        Create portfolio composition visualization.
        
        Args:
            weights: Portfolio weights dictionary
            allocation_type: Type of allocation (for title)
            interactive: Whether to create interactive chart
            
        Returns:
            Plotly or matplotlib figure
        """
        if interactive:
            return self._create_plotly_composition_chart(weights, allocation_type)
        else:
            return self._create_matplotlib_composition_chart(weights, allocation_type)
    
    def create_risk_decomposition_chart(
        self,
        risk_attribution: Dict[str, float],
        interactive: bool = True
    ) -> Union[go.Figure, Figure]:
        """
        Create risk decomposition visualization.
        
        Args:
            risk_attribution: Risk contribution by asset
            interactive: Whether to create interactive chart
            
        Returns:
            Plotly or matplotlib figure
        """
        if interactive:
            return self._create_plotly_risk_decomposition(risk_attribution)
        else:
            return self._create_matplotlib_risk_decomposition(risk_attribution)
    
    def create_performance_metrics_table(
        self,
        metrics: Dict[str, float],
        benchmark_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Create HTML table of performance metrics.
        
        Args:
            metrics: Portfolio performance metrics
            benchmark_metrics: Optional benchmark metrics for comparison
            
        Returns:
            HTML table string
        """
        # Create DataFrame for easier manipulation
        data = {'Portfolio': metrics}
        if benchmark_metrics:
            data['Benchmark'] = benchmark_metrics
            data['Excess'] = {k: metrics.get(k, 0) - benchmark_metrics.get(k, 0) 
                            for k in metrics.keys()}
        
        df = pd.DataFrame(data).round(4)
        
        # Convert to HTML with styling
        html = df.to_html(
            classes='table table-striped table-hover',
            table_id='performance-metrics',
            escape=False
        )
        
        return html
    
    def _create_tearsheet_plots(
        self,
        portfolio_results: Dict[str, Any],
        risk_metrics: Dict[str, Any],
        benchmark_comparison: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Create all plots for the tearsheet."""
        plots = {}
        
        # Efficient frontier plot
        if 'frontier_data' in portfolio_results:
            frontier_fig = self.create_efficient_frontier_plot(
                portfolio_results['frontier_data'],
                portfolio_results.get('optimal_portfolios', {}),
                interactive=True
            )
            plots['efficient_frontier'] = pyo.plot(
                frontier_fig, output_type='div', include_plotlyjs=False
            )
        
        # Portfolio composition
        if 'weights' in portfolio_results:
            composition_fig = self.create_portfolio_composition_chart(
                portfolio_results['weights'],
                interactive=True
            )
            plots['portfolio_composition'] = pyo.plot(
                composition_fig, output_type='div', include_plotlyjs=False
            )
        
        # Risk decomposition
        if 'risk_attribution' in risk_metrics:
            risk_fig = self.create_risk_decomposition_chart(
                risk_metrics['risk_attribution'],
                interactive=True
            )
            plots['risk_decomposition'] = pyo.plot(
                risk_fig, output_type='div', include_plotlyjs=False
            )
        
        # Performance over time (if available)
        if 'cumulative_returns' in portfolio_results:
            perf_fig = self._create_performance_chart(
                portfolio_results['cumulative_returns'],
                benchmark_comparison
            )
            plots['performance_chart'] = pyo.plot(
                perf_fig, output_type='div', include_plotlyjs=False
            )
        
        return plots
    
    def _create_plotly_efficient_frontier(
        self,
        frontier_data: pd.DataFrame,
        optimal_portfolios: Dict[str, Dict]
    ) -> go.Figure:
        """Create Plotly efficient frontier chart."""
        fig = go.Figure()
        
        # Add efficient frontier curve
        fig.add_trace(go.Scatter(
            x=frontier_data['volatility'],
            y=frontier_data['return'],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Risk:</b> %{x:.2%}<br><b>Return:</b> %{y:.2%}<extra></extra>'
        ))
        
        # Add optimal portfolios
        colors = ['red', 'green', 'orange', 'purple']
        for i, (name, portfolio) in enumerate(optimal_portfolios.items()):
            if 'risk' in portfolio and 'return' in portfolio:
                fig.add_trace(go.Scatter(
                    x=[portfolio['risk']],
                    y=[portfolio['return']],
                    mode='markers',
                    name=name,
                    marker=dict(size=12, color=colors[i % len(colors)]),
                    hovertemplate=f'<b>{name}</b><br>Risk: %{{x:.2%}}<br>Return: %{{y:.2%}}<extra></extra>'
                ))
        
        fig.update_layout(
            title='Efficient Frontier Analysis',
            xaxis_title='Risk (Volatility)',
            yaxis_title='Expected Return',
            template='plotly_white',
            hovermode='closest',
            width=800,
            height=600
        )
        
        return fig
    
    def _create_plotly_composition_chart(
        self,
        weights: Dict[str, float],
        allocation_type: str
    ) -> go.Figure:
        """Create Plotly portfolio composition chart."""
        # Filter out zero or very small weights
        filtered_weights = {k: v for k, v in weights.items() if abs(v) > 0.001}
        
        labels = list(filtered_weights.keys())
        values = list(filtered_weights.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>Weight: %{value:.2%}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'{allocation_type} Portfolio Composition',
            template='plotly_white',
            width=700,
            height=500
        )
        
        return fig
    
    def _create_plotly_risk_decomposition(
        self,
        risk_attribution: Dict[str, float]
    ) -> go.Figure:
        """Create Plotly risk decomposition chart."""
        assets = list(risk_attribution.keys())
        contributions = list(risk_attribution.values())
        
        fig = go.Figure(data=[go.Bar(
            x=assets,
            y=contributions,
            marker_color='lightblue',
            hovertemplate='<b>%{x}</b><br>Risk Contribution: %{y:.2%}<extra></extra>'
        )])
        
        fig.update_layout(
            title='Risk Attribution by Asset',
            xaxis_title='Assets',
            yaxis_title='Risk Contribution',
            template='plotly_white',
            width=800,
            height=500
        )
        
        return fig
    
    def _create_performance_chart(
        self,
        cumulative_returns: pd.Series,
        benchmark_comparison: Optional[Dict[str, Any]]
    ) -> go.Figure:
        """Create performance over time chart."""
        fig = go.Figure()
        
        # Add portfolio performance
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=2)
        ))
        
        # Add benchmark if available
        if benchmark_comparison and 'cumulative_returns' in benchmark_comparison:
            benchmark_returns = benchmark_comparison['cumulative_returns']
            fig.add_trace(go.Scatter(
                x=benchmark_returns.index,
                y=benchmark_returns.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='gray', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Cumulative Performance Comparison',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            template='plotly_white',
            hovermode='x unified',
            width=800,
            height=500
        )
        
        return fig
    
    def _create_matplotlib_efficient_frontier(
        self,
        frontier_data: pd.DataFrame,
        optimal_portfolios: Dict[str, Dict]
    ) -> Figure:
        """Create matplotlib efficient frontier chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot efficient frontier
        ax.plot(frontier_data['volatility'], frontier_data['return'], 
                'b-', linewidth=2, label='Efficient Frontier')
        
        # Plot optimal portfolios
        colors = ['red', 'green', 'orange', 'purple']
        for i, (name, portfolio) in enumerate(optimal_portfolios.items()):
            if 'risk' in portfolio and 'return' in portfolio:
                ax.scatter(portfolio['risk'], portfolio['return'], 
                          c=colors[i % len(colors)], s=100, label=name, zorder=5)
        
        ax.set_xlabel('Risk (Volatility)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Efficient Frontier Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _create_matplotlib_composition_chart(
        self,
        weights: Dict[str, float],
        allocation_type: str
    ) -> Figure:
        """Create matplotlib portfolio composition chart."""
        # Filter out zero or very small weights
        filtered_weights = {k: v for k, v in weights.items() if abs(v) > 0.001}
        
        labels = list(filtered_weights.keys())
        values = list(filtered_weights.values())
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10}
        )
        
        ax.set_title(f'{allocation_type} Portfolio Composition')
        
        return fig
    
    def _create_matplotlib_risk_decomposition(
        self,
        risk_attribution: Dict[str, float]
    ) -> Figure:
        """Create matplotlib risk decomposition chart."""
        assets = list(risk_attribution.keys())
        contributions = list(risk_attribution.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(assets, contributions, color='lightblue', edgecolor='navy')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}', ha='center', va='bottom')
        
        ax.set_xlabel('Assets')
        ax.set_ylabel('Risk Contribution')
        ax.set_title('Risk Attribution by Asset')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def _generate_html_tearsheet(
        self,
        plots: Dict[str, str],
        summary_stats: Dict[str, Any],
        executive_summary: str
    ) -> Path:
        """Generate HTML tearsheet."""
        template = self.template_env.get_template('tearsheet.html')
        
        html_content = template.render(
            plots=plots,
            summary_stats=summary_stats,
            executive_summary=executive_summary,
            metadata=self.report_metadata,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        html_path = self.output_dir / 'Investment_Strategy_Report.html'
        html_path.write_text(html_content, encoding='utf-8')
        
        return html_path
    
    def _generate_pdf_tearsheet(
        self,
        plots: Dict[str, str],
        summary_stats: Dict[str, Any],
        executive_summary: str
    ) -> Path:
        """Generate PDF tearsheet using matplotlib backend."""
        pdf_path = self.output_dir / 'Investment_Strategy_Tear_Sheet.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Page 1: Executive Summary and Key Metrics
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('NPS vs UPS Portfolio Optimization - Investment Strategy Tear Sheet', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            # Add executive summary text
            plt.figtext(0.1, 0.85, 'Executive Summary', fontsize=14, fontweight='bold')
            plt.figtext(0.1, 0.75, executive_summary, fontsize=10, wrap=True, 
                       verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
            
            # Add key metrics table
            if 'performance_metrics' in summary_stats:
                self._add_metrics_table_to_plot(fig, summary_stats['performance_metrics'], 0.45)
            
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Portfolio Composition (if available)
            if 'weights' in summary_stats:
                fig = self._create_matplotlib_composition_chart(
                    summary_stats['weights'], 'Optimal'
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Page 3: Risk Analysis (if available)
            if 'risk_attribution' in summary_stats:
                fig = self._create_matplotlib_risk_decomposition(
                    summary_stats['risk_attribution']
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        logger.info(f"PDF tearsheet generated: {pdf_path}")
        return pdf_path
    
    def _add_metrics_table_to_plot(
        self,
        fig: Figure,
        metrics: Dict[str, float],
        y_position: float = 0.5
    ):
        """Add metrics table to matplotlib figure."""
        # Create table data
        table_data = []
        for metric, value in metrics.items():
            if isinstance(value, float):
                if metric.lower() in ['return', 'volatility', 'sharpe', 'sortino']:
                    formatted_value = f"{value:.2%}" if abs(value) < 1 else f"{value:.2f}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            table_data.append([metric.replace('_', ' ').title(), formatted_value])
        
        # Add table to figure
        plt.figtext(0.1, y_position, 'Key Performance Metrics', 
                   fontsize=12, fontweight='bold')
        
        table_text = '\n'.join([f"{row[0]}: {row[1]}" for row in table_data])
        plt.figtext(0.1, y_position - 0.05, table_text, fontsize=9,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray"))
    
    def _calculate_summary_statistics(
        self,
        portfolio_results: Dict[str, Any],
        risk_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for the report."""
        summary = {
            'weights': portfolio_results.get('weights', {}),
            'performance_metrics': {},
            'risk_metrics': risk_metrics.copy()
        }
        
        # Extract performance metrics
        if 'expected_return' in portfolio_results:
            summary['performance_metrics']['Expected Annual Return'] = portfolio_results['expected_return']
        
        if 'volatility' in portfolio_results:
            summary['performance_metrics']['Annual Volatility'] = portfolio_results['volatility']
        
        if 'sharpe_ratio' in portfolio_results:
            summary['performance_metrics']['Sharpe Ratio'] = portfolio_results['sharpe_ratio']
        
        # Add risk metrics
        if 'var_95' in risk_metrics:
            summary['performance_metrics']['VaR (95%)'] = risk_metrics['var_95']
        
        if 'expected_shortfall_95' in risk_metrics:
            summary['performance_metrics']['Expected Shortfall (95%)'] = risk_metrics['expected_shortfall_95']
        
        return summary
    
    def _generate_executive_summary(
        self,
        portfolio_results: Dict[str, Any],
        risk_metrics: Dict[str, Any],
        summary_stats: Dict[str, Any]
    ) -> str:
        """Generate executive summary text."""
        
        # Extract key metrics
        expected_return = portfolio_results.get('expected_return', 0)
        volatility = portfolio_results.get('volatility', 0)
        sharpe_ratio = portfolio_results.get('sharpe_ratio', 0)
        
        # Get top holdings
        weights = portfolio_results.get('weights', {})
        top_holdings = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        summary = f"""
This investment strategy analysis compares the National Pension System (NPS) with the Unified Pension Scheme (UPS) 
using modern portfolio optimization techniques.

KEY FINDINGS:
• Expected Annual Return: {expected_return:.2%}
• Annual Volatility: {volatility:.2%}  
• Sharpe Ratio: {sharpe_ratio:.2f}

PORTFOLIO COMPOSITION:
The optimized portfolio allocates capital across pension fund schemes with the following top holdings:
"""
        
        for asset, weight in top_holdings:
            summary += f"• {asset}: {weight:.1%}\n"
        
        summary += f"""
RISK ASSESSMENT:
The portfolio demonstrates {self._assess_risk_level(volatility)} risk characteristics with robust diversification 
across pension fund managers and scheme types. Risk management is enhanced through systematic allocation 
constraints aligned with PFRDA guidelines.

RECOMMENDATION:
{self._generate_recommendation(expected_return, sharpe_ratio, volatility)}
"""
        
        return summary.strip()
    
    def _assess_risk_level(self, volatility: float) -> str:
        """Assess risk level based on volatility."""
        if volatility < 0.10:
            return "conservative"
        elif volatility < 0.20:
            return "moderate"
        else:
            return "aggressive"
    
    def _generate_recommendation(
        self,
        expected_return: float,
        sharpe_ratio: float,
        volatility: float
    ) -> str:
        """Generate investment recommendation."""
        if sharpe_ratio > 1.0:
            return "The optimized NPS portfolio demonstrates superior risk-adjusted returns and is recommended for long-term pension savings."
        elif sharpe_ratio > 0.5:
            return "The portfolio shows acceptable risk-adjusted performance suitable for moderate risk tolerance investors."
        else:
            return "Consider the guaranteed UPS option for risk-averse investors or review allocation constraints."
    
    def _get_default_templates(self) -> Dict[str, str]:
        """Get default HTML templates."""
        return {
            'tearsheet.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Investment Strategy Report - NPS vs UPS Portfolio Optimization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; }
        .metric-card { border-left: 4px solid #007bff; }
        .chart-container { margin: 2rem 0; }
        .executive-summary { background-color: #f8f9fa; border-radius: 0.5rem; padding: 1.5rem; }
    </style>
</head>
<body>
    <div class="header text-center">
        <h1>Investment Strategy Report</h1>
        <h2>NPS vs UPS Portfolio Optimization</h2>
        <p>Generated on {{ timestamp }} | {{ metadata.generator }} v{{ metadata.version }}</p>
    </div>
    
    <div class="container mt-4">
        <div class="row">
            <div class="col-12">
                <div class="executive-summary">
                    <h3><i class="fas fa-chart-line"></i> Executive Summary</h3>
                    <pre style="white-space: pre-wrap; font-family: inherit;">{{ executive_summary }}</pre>
                </div>
            </div>
        </div>
        
        {% if plots.efficient_frontier %}
        <div class="row chart-container">
            <div class="col-12">
                <h3>Efficient Frontier Analysis</h3>
                {{ plots.efficient_frontier|safe }}
            </div>
        </div>
        {% endif %}
        
        {% if plots.portfolio_composition %}
        <div class="row chart-container">
            <div class="col-12">
                <h3>Portfolio Composition</h3>
                {{ plots.portfolio_composition|safe }}
            </div>
        </div>
        {% endif %}
        
        {% if plots.risk_decomposition %}
        <div class="row chart-container">
            <div class="col-12">
                <h3>Risk Attribution</h3>
                {{ plots.risk_decomposition|safe }}
            </div>
        </div>
        {% endif %}
        
        {% if plots.performance_chart %}
        <div class="row chart-container">
            <div class="col-12">
                <h3>Performance Analysis</h3>
                {{ plots.performance_chart|safe }}
            </div>
        </div>
        {% endif %}
        
        <div class="row mt-5">
            <div class="col-12">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Key Performance Metrics</h5>
                        {% if summary_stats.performance_metrics %}
                            <div class="row">
                                {% for metric, value in summary_stats.performance_metrics.items() %}
                                <div class="col-md-4 mb-3">
                                    <div class="card metric-card">
                                        <div class="card-body">
                                            <h6 class="card-subtitle mb-2 text-muted">{{ metric }}</h6>
                                            <h4 class="card-text">{{ "%.2f%%" % (value * 100) if value < 1 and metric != 'Sharpe Ratio' else "%.2f" % value }}</h4>
                                        </div>
                                    </div>
                                </div>
                                {% endfor %}
                            </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <footer class="bg-light text-center py-3 mt-5">
        <p>&copy; 2025 NPS vs UPS Portfolio Optimization System. Generated by {{ metadata.generator }}.</p>
    </footer>
</body>
</html>
            '''
        }

    def save_plotly_chart(
        self,
        fig: go.Figure,
        filename: str,
        format: str = 'html'
    ) -> Path:
        """
        Save Plotly chart to file.
        
        Args:
            fig: Plotly figure
            filename: Output filename (without extension)
            format: Output format ('html', 'png', 'pdf', 'svg')
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{filename}.{format}"
        
        if format == 'html':
            fig.write_html(str(output_path), config=self.plotly_config)
        elif format == 'png':
            fig.write_image(str(output_path), width=1200, height=800)
        elif format == 'pdf':
            fig.write_image(str(output_path), width=1200, height=800)
        elif format == 'svg':
            fig.write_image(str(output_path), width=1200, height=800)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Chart saved to: {output_path}")
        return output_path

    def export_data(
        self,
        data: Dict[str, Any],
        filename: str,
        format: str = 'json'
    ) -> Path:
        """
        Export data to file.
        
        Args:
            data: Data dictionary to export
            filename: Output filename (without extension)
            format: Output format ('json', 'csv', 'excel')
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{filename}.{format}"
        
        if format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == 'csv':
            if isinstance(data, dict) and all(isinstance(v, (list, pd.Series)) for v in data.values()):
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False)
            else:
                raise ValueError("Data must be dictionary of lists/Series for CSV export")
        elif format == 'excel':
            if isinstance(data, dict):
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    for sheet_name, sheet_data in data.items():
                        if isinstance(sheet_data, pd.DataFrame):
                            sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
                        elif isinstance(sheet_data, dict):
                            pd.DataFrame([sheet_data]).to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                raise ValueError("Data must be dictionary for Excel export")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data exported to: {output_path}")
        return output_path 