# NPS vs UPS Portfolio Optimization - Setup & Run Guide

## Quick Start (< 5 minutes)

### 1. Environment Setup
```bash
# Ensure Python 3.11 is installed
python --version  # Should show Python 3.11.x

# Clone/navigate to project directory
cd nps_ups_portfolio_optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### 2. Run Complete Analysis
```bash
# Option A: Using CLI (Recommended)
python -m nps_ups run-analysis

# Option B: Using Makefile
make run-fast  # Quick version (1,000 simulations)
make run       # Full version (10,000 simulations)

# Option C: Using Jupyter Notebook
jupyter notebook notebooks/run_all.ipynb
```

### 3. View Results
```bash
# Check output directory
ls -la output/

# Open reports
open output/Investment_Strategy_Report.html      # Interactive report
open output/Investment_Strategy_Tear_Sheet.pdf   # PDF summary
```

## Detailed Setup Instructions

### Prerequisites
- **Python 3.11** (required for latest type hints and performance)
- **Git** (for cloning repository)
- **Make** (optional, for development commands)

### System Dependencies (Linux/macOS)
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libcairo2-dev libpango1.0-dev libgdk-pixbuf2.0-dev libffi-dev

# macOS (using Homebrew)
brew install cairo pango gdk-pixbuf libffi
```

### Installation Options

#### Option 1: Standard Installation
```bash
pip install -e .
```

#### Option 2: Development Installation (with testing tools)
```bash
pip install -e ".[dev,docs]"
make dev-setup  # Sets up pre-commit hooks
```

#### Option 3: Minimal Installation (core functionality only)
```bash
pip install numpy pandas PyPortfolioOpt matplotlib plotly click
pip install -e . --no-deps
```

## Usage Examples

### CLI Interface
```bash
# Load and cache data
python -m nps_ups load-data --start-date 2019-01-01

# Run analysis with custom parameters
python -m nps_ups run-analysis \
    --scenarios base optimistic adverse \
    --n-simulations 5000 \
    --current-age 30 \
    --retirement-age 60 \
    --current-salary 1500000

# Individual portfolio optimization
python -m nps_ups optimize-portfolio --portfolio-type aggressive

# Risk analysis for specific portfolio
python -m nps_ups risk-analysis --confidence-level 0.95
```

### Python API
```python
from nps_ups import PortfolioOptimizer, MonteCarloSimulator
from nps_ups.io import DataLoader

# Load data
loader = DataLoader()
data = loader.load_pension_fund_data()

# Optimize portfolio
optimizer = PortfolioOptimizer(data)
result = optimizer.optimize_max_sharpe(target_allocation='Moderate')

# Run Monte Carlo simulation
simulator = MonteCarloSimulator(data)
results = simulator.run_comprehensive_simulation(['base'])
```

## Configuration

### Environment Variables (Optional)
```bash
# Data source API keys (if available)
export PFRDA_API_KEY="your_pfrda_key"
export RBI_API_KEY="your_rbi_key"

# Custom directories
export NPS_UPS_DATA_DIR="./custom_data"
export NPS_UPS_OUTPUT_DIR="./custom_output"
```

### Analysis Parameters
Edit `nps_ups/simulation.py` to modify default parameters:
```python
@dataclass
class SimulationParameters:
    current_age: int = 25           # Starting age
    retirement_age: int = 60        # Retirement age
    current_salary: float = 1_000_000  # Annual salary (INR)
    n_simulations: int = 10_000     # Monte Carlo paths
    # ... other parameters
```

## Expected Output

### Generated Files
```
output/
├── Investment_Strategy_Report.html         # Interactive web report
├── Investment_Strategy_Tear_Sheet.pdf      # Executive summary PDF
├── portfolio_weights.csv                   # Optimized allocations
├── efficient_frontier.csv                  # Risk-return frontier
├── simulation_summary.json                 # Monte Carlo results
└── charts/                                 # Individual chart files
    ├── efficient_frontier.png
    ├── portfolio_allocation.png
    └── monte_carlo_results.png
```

### Performance Metrics
The analysis generates:
- **Portfolio Optimization**: 10+ optimized portfolios (Aggressive, Moderate, Conservative, Lifecycle)
- **Risk Analytics**: VaR, stress tests, performance metrics for each portfolio
- **Monte Carlo Results**: 10,000 simulation paths across 3 economic scenarios
- **Comparison Analysis**: Probability of NPS strategies beating UPS benefits

## Troubleshooting

### Common Issues

#### 1. PDF Generation Fails
```bash
# Issue: WeasyPrint dependencies missing
# Solution: Install system dependencies
sudo apt-get install libcairo2-dev libpango1.0-dev  # Linux
brew install cairo pango  # macOS
```

#### 2. Memory Issues with Large Simulations
```bash
# Issue: Out of memory with 10,000 simulations
# Solution: Reduce simulation count
python -m nps_ups run-analysis --n-simulations 1000
```

#### 3. Import Errors
```bash
# Issue: Module not found
# Solution: Install in editable mode
pip install -e .

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 4. Data Download Issues
```bash
# Issue: Network/API errors
# Solution: Use offline mode (sample data is included)
python -m nps_ups run-analysis  # Will use sample data automatically
```

### Performance Optimization

#### For Faster Execution
```bash
# Reduce simulations
python -m nps_ups run-analysis --n-simulations 1000

# Run single scenario
python -m nps_ups run-analysis --scenarios base

# Skip expensive optimizations
# Edit nps_ups/optimiser.py and comment out CVaR optimization
```

#### For Maximum Accuracy
```bash
# Increase simulations
python -m nps_ups run-analysis --n-simulations 50000

# Use all scenarios
python -m nps_ups run-analysis --scenarios base optimistic adverse

# Enable parallel processing (if available)
export OMP_NUM_THREADS=4
```

## Testing

### Run Test Suite
```bash
# Full test suite
make test

# Fast tests only
make test-fast

# Specific test category
pytest tests/ -m unit
pytest tests/ -m integration
```

### Code Quality Checks
```bash
# All quality checks
make quality

# Individual checks
make lint        # Linting
make format      # Code formatting
make type-check  # Type checking
```

## Development

### Development Workflow
```bash
# Setup development environment
make dev-setup

# Make changes to code
# ...

# Run quality checks
make quality

# Run tests
make test-fast

# Test analysis pipeline
make run-fast
```

### Adding New Features
1. **Data Sources**: Extend `nps_ups/io/` modules
2. **Optimization Strategies**: Add methods to `nps_ups/optimiser.py`
3. **Risk Models**: Implement in `nps_ups/analytics.py`
4. **Visualizations**: Add charts to `nps_ups/reporting.py`

## Support

### Getting Help
1. **Documentation**: See README.md for detailed project overview
2. **Code Examples**: Check `notebooks/run_all.ipynb` for usage examples
3. **API Reference**: Run `make docs` to build API documentation
4. **Issues**: Report bugs or feature requests on GitHub

### Performance Benchmarks
- **Full Analysis**: < 10 minutes on modern laptop
- **Quick Analysis**: < 2 minutes with 1,000 simulations
- **Memory Usage**: ~2GB peak for 10,000 simulations
- **Output Size**: ~50MB for complete analysis

---

**Happy Analyzing! 📊🚀**

*This setup guide covers installation, usage, and troubleshooting for the NPS vs UPS Portfolio Optimization project. For detailed methodology and results interpretation, see the main README.md file.* 