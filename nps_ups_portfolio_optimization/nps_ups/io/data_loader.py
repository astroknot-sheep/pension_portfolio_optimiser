"""
Data loader module for NPS pension fund NAV data.

Provides functionality to download, cache, and preprocess NAV data from various
Pension Fund Managers (PFMs) including HDFC, ICICI, SBI, UTI, and LIC.
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles downloading and caching of pension fund NAV data.
    
    Supports multiple data sources:
    - PFRDA official API (when available)
    - NSE data feeds
    - Yahoo Finance for indices
    - Local CSV fallback data
    """
    
    # Pension Fund Manager mapping
    PFM_MAPPING = {
        'HDFC': 'HDFC Pension Fund',
        'ICICI': 'ICICI Prudential Pension Fund', 
        'SBI': 'SBI Pension Fund',
        'UTI': 'UTI Retirement Solutions',
        'LIC': 'LIC Pension Fund'
    }
    
    # Scheme types for NPS
    SCHEME_TYPES = ['E', 'C', 'G']  # Equity, Corporate Bond, Government Securities
    
    def __init__(self, data_dir: str = "data", cache_ttl_hours: int = 24):
        """
        Initialize the DataLoader.
        
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
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def load_pension_fund_data(
        self,
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load comprehensive pension fund NAV data for all PFMs and schemes.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: today)
            force_refresh: Force download even if cache exists
            
        Returns:
            DataFrame with columns: date, pfm, scheme, nav_value
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        cache_file = self.processed_dir / f"pension_nav_{start_date}_{end_date}.parquet"
        
        # Check cache validity
        if not force_refresh and self._is_cache_valid(cache_file):
            logger.info(f"Loading cached pension fund data from {cache_file}")
            return pd.read_parquet(cache_file)
        
        logger.info("Downloading fresh pension fund data...")
        
        # Try multiple data sources
        data_frames = []
        
        # Method 1: Try PFRDA API (mock implementation)
        try:
            df_pfrda = self._fetch_pfrda_data(start_date, end_date)
            if not df_pfrda.empty:
                data_frames.append(df_pfrda)
        except Exception as e:
            logger.warning(f"PFRDA API failed: {e}")
        
        # Method 2: Use sample data if API fails
        if not data_frames:
            logger.info("Using sample pension fund data...")
            df_sample = self._generate_sample_pension_data(start_date, end_date)
            data_frames.append(df_sample)
        
        # Combine and process data
        combined_df = pd.concat(data_frames, ignore_index=True)
        processed_df = self._process_pension_data(combined_df)
        
        # Cache the result
        processed_df.to_parquet(cache_file, index=False)
        
        return processed_df
    
    def load_market_data(
        self,
        tickers: List[str],
        start_date: str = "2019-01-01", 
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load market index data for benchmarking.
        
        Args:
            tickers: List of ticker symbols (e.g., ['^NSEI', '^BSESN'])
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLCV data for each ticker
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        cache_file = self.processed_dir / f"market_data_{'_'.join(tickers)}_{start_date}_{end_date}.parquet"
        
        if self._is_cache_valid(cache_file):
            return pd.read_parquet(cache_file)
        
        logger.info(f"Downloading market data for {tickers}")
        
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            
            if len(tickers) == 1:
                # Single ticker - flatten column structure
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
                data = data.reset_index()
            else:
                # Multiple tickers - keep hierarchical structure
                data = data.reset_index()
            
            data.to_parquet(cache_file, index=False)
            return data
            
        except Exception as e:
            logger.error(f"Failed to download market data: {e}")
            return pd.DataFrame()
    
    def _fetch_pfrda_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data from PFRDA API (mock implementation).
        
        In production, this would connect to the actual PFRDA API.
        """
        # Mock API endpoint (not real)
        api_url = "https://api.pfrda.gov.in/nav/historical"
        
        # This is a placeholder - real implementation would make API calls
        # For now, return empty DataFrame to trigger sample data generation
        return pd.DataFrame()
    
    def _generate_sample_pension_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate realistic sample pension fund NAV data for demonstration.
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Create daily date range
        dates = pd.date_range(start_dt, end_dt, freq='D')
        
        # Remove weekends (markets closed)
        dates = dates[dates.weekday < 5]
        
        data_rows = []
        
        # Base NAV values and volatilities for different schemes
        nav_configs = {
            'E': {'base_nav': 25.0, 'annual_return': 0.12, 'volatility': 0.18},
            'C': {'base_nav': 20.0, 'annual_return': 0.08, 'volatility': 0.08}, 
            'G': {'base_nav': 15.0, 'annual_return': 0.07, 'volatility': 0.02}
        }
        
        np.random.seed(42)  # For reproducible results
        
        for pfm in self.PFM_MAPPING.keys():
            for scheme in self.SCHEME_TYPES:
                config = nav_configs[scheme]
                
                # Generate price series using geometric Brownian motion
                dt = 1/252  # Daily time step
                n_steps = len(dates)
                
                # Generate random walks
                random_shocks = np.random.normal(0, 1, n_steps)
                
                # Calculate returns
                drift = (config['annual_return'] - 0.5 * config['volatility']**2) * dt
                diffusion = config['volatility'] * np.sqrt(dt) * random_shocks
                
                # Compute NAV series
                log_returns = drift + diffusion
                cumulative_returns = np.cumsum(log_returns)
                nav_values = config['base_nav'] * np.exp(cumulative_returns)
                
                # Add PFM-specific adjustments
                pfm_factor = {'HDFC': 1.02, 'ICICI': 1.01, 'SBI': 1.0, 'UTI': 0.99, 'LIC': 0.98}
                nav_values *= pfm_factor[pfm]
                
                # Create records
                for date, nav in zip(dates, nav_values):
                    data_rows.append({
                        'date': date,
                        'pfm': pfm,
                        'scheme': scheme,
                        'nav_value': round(nav, 4)
                    })
        
        return pd.DataFrame(data_rows)
    
    def _process_pension_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and validate pension fund data.
        """
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values(['pfm', 'scheme', 'date']).reset_index(drop=True)
        
        # Calculate returns
        df['return'] = df.groupby(['pfm', 'scheme'])['nav_value'].pct_change()
        
        # Remove outliers (returns > 50% or < -50%)
        df = df[df['return'].between(-0.5, 0.5)].copy()
        
        # Forward fill missing values (up to 5 days)
        df = df.groupby(['pfm', 'scheme']).apply(
            lambda x: x.fillna(method='ffill', limit=5)
        ).reset_index(drop=True)
        
        return df
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file exists and is within TTL."""
        if not cache_file.exists():
            return False
            
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age < self.cache_ttl
    
    def get_risk_free_rate(self, date: Optional[str] = None) -> float:
        """
        Get current or historical risk-free rate (10-year G-Sec yield).
        
        Args:
            date: Date in YYYY-MM-DD format (default: latest)
            
        Returns:
            Risk-free rate as decimal (e.g., 0.07 for 7%)
        """
        # Default risk-free rate for India (7% based on recent 10Y G-Sec)
        # In production, fetch from RBI or Bloomberg API
        return 0.07
    
    def export_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """Export DataFrame to CSV in the raw data directory."""
        file_path = self.raw_dir / filename
        df.to_csv(file_path, index=False)
        logger.info(f"Data exported to {file_path}")


# Convenience function for quick data loading
def load_pension_data(start_date: str = "2019-01-01", **kwargs) -> pd.DataFrame:
    """Quick function to load pension fund data."""
    loader = DataLoader()
    return loader.load_pension_fund_data(start_date=start_date, **kwargs) 