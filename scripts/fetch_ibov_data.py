#!/usr/bin/env python3
"""
Fetch Extended IBOVESPA Data
============================
Fetches maximum available IBOVESPA historical data including price and volatility.

Uses yfinance to download data from Yahoo Finance, which provides data
back to the early 1990s for IBOVESPA (^BVSP).

Usage:
    python scripts/fetch_ibov_data.py

Output:
    data/raw/ibovespa_data.csv

Author: Claude Code for CAEN/UFC Master's Thesis
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas and numpy required")
    sys.exit(1)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Setup paths
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_DIR = _SCRIPT_DIR.parent
DATA_DIR = _PROJECT_DIR / 'data' / 'raw'
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / 'ibovespa_data.csv'


def calculate_realized_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Calculate annualized realized volatility."""
    return returns.rolling(window).std() * np.sqrt(252) * 100


def fetch_ibovespa_yfinance():
    """Fetch IBOVESPA data from Yahoo Finance."""
    if not YFINANCE_AVAILABLE:
        print("  yfinance not available. Install with: pip install yfinance")
        return None

    print("  Fetching IBOVESPA (^BVSP) from Yahoo Finance...")

    try:
        # Fetch maximum available history
        ibov = yf.Ticker("^BVSP")
        hist = ibov.history(period="max")

        if hist is None or len(hist) == 0:
            print("    No data returned")
            return None

        print(f"    Downloaded {len(hist)} daily observations")
        print(f"    Period: {hist.index.min().date()} to {hist.index.max().date()}")

        # Prepare DataFrame
        df = pd.DataFrame({
            'date': hist.index,
            'open': hist['Open'],
            'high': hist['High'],
            'low': hist['Low'],
            'close': hist['Close'],
            'volume': hist['Volume'],
        })
        df = df.reset_index(drop=True)

        # Calculate returns
        df['return'] = df['close'].pct_change()

        # Calculate realized volatility at different windows
        df['realized_vol_5d'] = calculate_realized_volatility(df['return'], 5)
        df['realized_vol_21d'] = calculate_realized_volatility(df['return'], 21)
        df['realized_vol_30d'] = calculate_realized_volatility(df['return'], 30)

        # Calculate intraday range (high-low)/close
        df['intraday_range'] = (df['high'] - df['low']) / df['close'] * 100

        return df

    except Exception as e:
        print(f"    Error: {e}")
        return None


def fetch_ewz_yfinance():
    """Fetch EWZ (Brazil ETF) as alternative."""
    if not YFINANCE_AVAILABLE:
        return None

    print("  Fetching EWZ (Brazil ETF) as backup...")

    try:
        ewz = yf.Ticker("EWZ")
        hist = ewz.history(period="max")

        if hist is None or len(hist) == 0:
            return None

        print(f"    Downloaded {len(hist)} daily observations")

        df = pd.DataFrame({
            'date': hist.index,
            'open': hist['Open'],
            'high': hist['High'],
            'low': hist['Low'],
            'close': hist['Close'],
            'volume': hist['Volume'],
        })
        df = df.reset_index(drop=True)

        df['return'] = df['close'].pct_change()
        df['realized_vol_5d'] = calculate_realized_volatility(df['return'], 5)
        df['realized_vol_21d'] = calculate_realized_volatility(df['return'], 21)
        df['realized_vol_30d'] = calculate_realized_volatility(df['return'], 30)
        df['intraday_range'] = (df['high'] - df['low']) / df['close'] * 100

        return df

    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    print("=" * 60)
    print("IBOVESPA DATA FETCHER")
    print("=" * 60)

    df = None
    source = None

    # Try IBOVESPA first
    print("\n1. Attempting to fetch IBOVESPA (^BVSP)...")
    df = fetch_ibovespa_yfinance()
    if df is not None:
        source = "Yahoo Finance (^BVSP)"

    # Fallback to EWZ
    if df is None:
        print("\n2. Attempting to fetch EWZ (Brazil ETF)...")
        df = fetch_ewz_yfinance()
        if df is not None:
            source = "Yahoo Finance (EWZ)"

    if df is not None:
        # Save to CSV
        df.to_csv(OUTPUT_FILE, index=False)

        print("\n" + "=" * 60)
        print(f"SUCCESS: Data saved to {OUTPUT_FILE}")
        print("=" * 60)
        print(f"\nSource: {source}")
        print(f"Period: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"Observations: {len(df)}")
        print(f"\nVolatility Statistics (21-day):")
        vol = df['realized_vol_21d'].dropna()
        print(f"  Mean: {vol.mean():.1f}%")
        print(f"  Min:  {vol.min():.1f}%")
        print(f"  Max:  {vol.max():.1f}%")
        print(f"  Current: {vol.iloc[-1]:.1f}%")

        return True
    else:
        print("\nFailed to fetch IBOVESPA data.")
        print("Please install yfinance: pip install yfinance")
        return False


if __name__ == '__main__':
    main()
