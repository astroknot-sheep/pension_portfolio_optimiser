"""
Main entry point for the NPS vs UPS portfolio optimization package.

Allows running the package as a module:
    python -m nps_ups
    python -m nps_ups.run
"""

from nps_ups.cli import cli

if __name__ == '__main__':
    cli() 