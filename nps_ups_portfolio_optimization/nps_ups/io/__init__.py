"""
Data ingestion and management module for NPS/UPS portfolio optimization.

Handles downloading, caching, and preprocessing of:
- Pension fund NAV data (HDFC, ICICI, SBI, UTI, LIC)
- Economic indicators (inflation, repo rates)
- Market indices and risk-free rates
"""

from nps_ups.io.data_loader import DataLoader
from nps_ups.io.economic_data import EconomicDataProvider

__all__ = ["DataLoader", "EconomicDataProvider"] 