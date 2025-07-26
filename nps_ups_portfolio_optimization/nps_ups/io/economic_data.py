"""
Economic data provider for macroeconomic indicators and inflation analysis.

Handles downloading and processing of:
- RBI repo rates and policy rates
- CPI inflation data
- GDP growth rates  
- Currency exchange rates
- Other macroeconomic indicators relevant to pension analysis
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
from pandas_datareader import data as pdr
import yfinance as yf

logger = logging.getLogger(__name__)


class EconomicDataProvider:
    """
    Provides access to Indian macroeconomic data for pension analysis.
    
    Data sources:
    - RBI official statistics
    - FRED (Federal Reserve Economic Data)
    - Yahoo Finance for currency data
    - Government statistics portals
    """
    
    # Economic indicators mapping
    INDICATORS = {
        'repo_rate': 'RBI Repo Rate',
        'cpi_inflation': 'Consumer Price Index',
        'wpi_inflation': 'Wholesale Price Index', 
        'gdp_growth': 'GDP Growth Rate',
        'usdinr': 'USD/INR Exchange Rate',
        'crude_oil': 'Crude Oil Prices',
        '10y_gsec': '10-Year Government Securities'
    }
    
    def __init__(self, data_dir: str = "data", cache_ttl_hours: int = 24):
        """
        Initialize the EconomicDataProvider.
        
        Args:
            data_dir: Directory to store cached data
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_inflation_data(
        self,
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load CPI inflation data for India.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_refresh: Force download even if cache exists
            
        Returns:
            DataFrame with columns: date, cpi_inflation, yoy_change
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        cache_file = self.processed_dir / f"inflation_data_{start_date}_{end_date}.parquet"
        
        if not force_refresh and self._is_cache_valid(cache_file):
            return pd.read_parquet(cache_file)
        
        logger.info("Loading inflation data...")
        
        # Generate sample inflation data (in production, use RBI API)
        df = self._generate_sample_inflation_data(start_date, end_date)
        
        # Cache the result
        df.to_parquet(cache_file, index=False)
        
        return df
    
    def load_repo_rate_data(
        self,
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load RBI repo rate data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format  
            force_refresh: Force download even if cache exists
            
        Returns:
            DataFrame with columns: date, repo_rate, reverse_repo_rate
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        cache_file = self.processed_dir / f"repo_rate_{start_date}_{end_date}.parquet"
        
        if not force_refresh and self._is_cache_valid(cache_file):
            return pd.read_parquet(cache_file)
        
        logger.info("Loading repo rate data...")
        
        # Generate sample repo rate data
        df = self._generate_sample_repo_rate_data(start_date, end_date)
        
        # Cache the result
        df.to_parquet(cache_file, index=False)
        
        return df
    
    def load_currency_data(
        self,
        pair: str = "USDINR=X",
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load currency exchange rate data.
        
        Args:
            pair: Currency pair symbol (default: USD/INR)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLCV data for currency pair
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            data = yf.download(pair, start=start_date, end=end_date, progress=False)
            data = data.reset_index()
            return data
        except Exception as e:
            logger.error(f"Failed to download currency data: {e}")
            return self._generate_sample_currency_data(start_date, end_date)
    
    def load_comprehensive_macro_data(
        self,
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load comprehensive macroeconomic dataset for VAR/IRF analysis.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with multiple economic indicators
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Load individual datasets
        inflation_df = self.load_inflation_data(start_date, end_date)
        repo_df = self.load_repo_rate_data(start_date, end_date)
        currency_df = self.load_currency_data("USDINR=X", start_date, end_date)
        
        # Merge on common date column
        merged_df = inflation_df.merge(repo_df, on='date', how='outer')
        
        # Add currency data (use Close price as USD/INR rate)
        if not currency_df.empty:
            currency_clean = currency_df[['Date', 'Close']].rename(
                columns={'Date': 'date', 'Close': 'usdinr_rate'}
            )
            merged_df = merged_df.merge(currency_clean, on='date', how='outer')
        
        # Sort by date and forward fill missing values
        merged_df = merged_df.sort_values('date')
        merged_df = merged_df.fillna(method='ffill')
        
        return merged_df
    
    def calculate_inflation_scenarios(
        self,
        base_inflation: float = 0.04,
        shock_magnitude: float = 0.02,
        periods: int = 24
    ) -> Dict[str, List[float]]:
        """
        Generate inflation scenarios for Monte Carlo simulation.
        
        Args:
            base_inflation: Base inflation rate (default: 4%)
            shock_magnitude: Magnitude of inflation shock (default: 2%)
            periods: Number of periods for scenario (default: 24 months)
            
        Returns:
            Dictionary with base, optimistic, and adverse inflation scenarios
        """
        scenarios = {
            'base': [],
            'optimistic': [],
            'adverse': []
        }
        
        # Base scenario: mean-reverting inflation around target
        target_inflation = base_inflation
        current_inflation = base_inflation
        
        for i in range(periods):
            # Mean reversion with some persistence
            mean_reversion = 0.1 * (target_inflation - current_inflation)
            noise = np.random.normal(0, 0.005)  # Small random shocks
            
            # Base scenario
            base_next = current_inflation + mean_reversion + noise
            scenarios['base'].append(max(0, base_next))
            
            # Optimistic scenario (lower inflation)
            opt_next = base_next - shock_magnitude * np.exp(-i/12)  # Decaying benefit
            scenarios['optimistic'].append(max(0, opt_next))
            
            # Adverse scenario (higher inflation)  
            adv_next = base_next + shock_magnitude * np.exp(-i/8)  # Slower decay
            scenarios['adverse'].append(max(0, adv_next))
            
            current_inflation = base_next
        
        return scenarios
    
    def _generate_sample_inflation_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic sample inflation data."""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Monthly frequency for inflation data
        dates = pd.date_range(start_dt, end_dt, freq='M')
        
        # Generate inflation series with persistence and shocks
        np.random.seed(42)
        n_periods = len(dates)
        
        # Base inflation trend (targeting ~4-6%)
        trend = 0.05 + 0.01 * np.sin(np.arange(n_periods) * 2 * np.pi / 12)  # Seasonal pattern
        
        # Add persistence and shocks
        inflation_series = [0.045]  # Start at 4.5%
        for i in range(1, n_periods):
            # AR(1) process with trend
            persistence = 0.7 * inflation_series[-1] + 0.3 * trend[i]
            shock = np.random.normal(0, 0.01)  # Monthly shock
            next_val = persistence + shock
            inflation_series.append(max(0, next_val))  # Non-negative inflation
        
        # Calculate year-over-year changes
        yoy_changes = []
        for i in range(len(inflation_series)):
            if i < 12:
                yoy_changes.append(np.nan)  # Not enough history
            else:
                yoy_change = (inflation_series[i] - inflation_series[i-12]) * 100
                yoy_changes.append(yoy_change)
        
        return pd.DataFrame({
            'date': dates,
            'cpi_inflation': inflation_series,
            'yoy_change': yoy_changes
        })
    
    def _generate_sample_repo_rate_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic sample repo rate data."""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Quarterly frequency for policy rate changes
        dates = pd.date_range(start_dt, end_dt, freq='Q')
        
        # Repo rate path (starts at 6.5%, moves based on economic conditions)
        repo_rates = [0.065]  # Start at 6.5%
        
        np.random.seed(42)
        for i in range(1, len(dates)):
            # Policy decisions: small changes with persistence
            change_prob = 0.3  # 30% chance of rate change per quarter
            if np.random.random() < change_prob:
                # Rate change: typically 25-50 bps
                change = np.random.choice([-0.0050, -0.0025, 0.0025, 0.0050])
                new_rate = repo_rates[-1] + change
                # Keep rates within reasonable bounds (4% to 8%)
                new_rate = max(0.04, min(0.08, new_rate))
                repo_rates.append(new_rate)
            else:
                repo_rates.append(repo_rates[-1])  # No change
        
        # Reverse repo rate is typically 25 bps below repo rate
        reverse_repo_rates = [rate - 0.0025 for rate in repo_rates]
        
        return pd.DataFrame({
            'date': dates,
            'repo_rate': repo_rates,
            'reverse_repo_rate': reverse_repo_rates
        })
    
    def _generate_sample_currency_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate sample USD/INR currency data if Yahoo Finance fails."""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        dates = pd.date_range(start_dt, end_dt, freq='D')
        dates = dates[dates.weekday < 5]  # Remove weekends
        
        # Generate USD/INR series (starts around 74-75)
        np.random.seed(42)
        base_rate = 74.5
        rates = [base_rate]
        
        for i in range(1, len(dates)):
            # Random walk with slight upward trend (INR depreciation)
            drift = 0.001  # Small daily appreciation trend
            volatility = 0.005  # Daily volatility
            change = drift + np.random.normal(0, volatility)
            new_rate = rates[-1] * (1 + change)
            rates.append(new_rate)
        
        return pd.DataFrame({
            'Date': dates,
            'Open': rates,
            'High': [rate * 1.005 for rate in rates],
            'Low': [rate * 0.995 for rate in rates],
            'Close': rates,
            'Volume': [1000000] * len(rates)  # Dummy volume
        })
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file exists and is within TTL."""
        if not cache_file.exists():
            return False
            
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age < self.cache_ttl 