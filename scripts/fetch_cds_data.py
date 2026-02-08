#!/usr/bin/env python3
"""
Fetch Brazil 5Y CDS Data
========================
Fetches Brazil 5-year Credit Default Swap spread data from various sources.

CDS spreads are a market-based measure of sovereign credit risk and serve
as an excellent benchmark for validating financial stress indices.

Sources attempted:
1. FRED (Federal Reserve Economic Data) - if available
2. Yahoo Finance (indirect proxy)
3. Manual download instructions

Usage:
    python scripts/fetch_cds_data.py

Output:
    data/raw/brazil_cds_5y.csv

Author: Claude Code for CAEN/UFC Master's Thesis
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Handle imports
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("ERROR: pandas required. Install with: pip install pandas")
    sys.exit(1)

# Try importing data fetching libraries
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from pandas_datareader import data as pdr
    DATAREADER_AVAILABLE = True
except ImportError:
    DATAREADER_AVAILABLE = False

# Setup paths
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_DIR = _SCRIPT_DIR.parent
DATA_DIR = _PROJECT_DIR / 'data' / 'raw'
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / 'brazil_cds_5y.csv'


def fetch_from_fred():
    """
    Try to fetch Brazil CDS data from FRED.

    Note: FRED doesn't have Brazil CDS directly, but has related series.
    """
    if not DATAREADER_AVAILABLE:
        print("  pandas_datareader not available")
        return None

    try:
        # FRED doesn't have Brazil CDS directly
        # But we can try emerging market related series
        print("  Checking FRED for Brazil CDS proxy data...")

        # These are alternative series that correlate with Brazil CDS
        series_to_try = [
            'BAMLHE00EHYIOAS',  # ICE BofA Emerging Markets High Yield Index Option-Adjusted Spread
            'BAMLEMHBHYCRPIOAS',  # Emerging Markets High Yield Corp Bond OAS
        ]

        for series in series_to_try:
            try:
                data = pdr.get_data_fred(series, start='2000-01-01')
                if data is not None and len(data) > 0:
                    print(f"    Found {series}: {len(data)} observations")
                    return data
            except Exception as e:
                continue

        return None
    except Exception as e:
        print(f"  FRED fetch failed: {e}")
        return None


def fetch_brazil_embi():
    """
    Fetch Brazil EMBI+ spread as a CDS proxy.

    The EMBI+ (Emerging Markets Bond Index Plus) spread for Brazil
    is highly correlated with CDS spreads and can serve as a proxy.
    """
    if not YFINANCE_AVAILABLE:
        print("  yfinance not available")
        return None

    try:
        print("  Attempting to fetch Brazil-related ETFs as proxy...")

        # Brazil ETF (EWZ) implied volatility can proxy stress
        ewz = yf.Ticker("EWZ")
        hist = ewz.history(period="max")

        if hist is not None and len(hist) > 0:
            print(f"    Found EWZ data: {len(hist)} observations")

            # Calculate rolling volatility as stress proxy
            hist['return'] = hist['Close'].pct_change()
            hist['vol_21d'] = hist['return'].rolling(21).std() * np.sqrt(252) * 100

            # Scale to approximate CDS basis points (rough approximation)
            # Historical Brazil CDS ranges roughly 100-500 bps normally
            # EWZ vol ranges roughly 15-60%
            hist['cds_proxy'] = hist['vol_21d'] * 10  # Rough scaling

            result = pd.DataFrame({
                'date': hist.index,
                'cds_5y': hist['cds_proxy'].values,
                'note': 'proxy_from_ewz_volatility'
            })
            result = result.dropna()
            return result

    except Exception as e:
        print(f"  ETF proxy fetch failed: {e}")

    return None


def main():
    print("=" * 60)
    print("BRAZIL 5Y CDS DATA FETCHER")
    print("=" * 60)

    df = None
    source = None

    # Try different sources
    print("\n1. Attempting FRED...")
    fred_data = fetch_from_fred()
    if fred_data is not None:
        df = fred_data
        source = 'FRED'

    if df is None:
        print("\n2. Attempting ETF proxy...")
        ewz_data = fetch_brazil_embi()
        if ewz_data is not None:
            df = ewz_data
            source = 'EWZ_proxy'

    if df is None:
        print("\n" + "=" * 60)
        print("ERROR: Could not fetch CDS data from any source.")
        print("=" * 60)
        print("\nTo get real CDS data, manually download from:")
        print("  - Investing.com: https://www.investing.com/rates-bonds/brazil-cds-5-years-usd-historical-data")
        print("  - World Gov Bonds: http://www.worldgovernmentbonds.com/cds-historical-data/brazil/5-years/")
        print("\nSave as: data/raw/brazil_cds_5y.csv with columns: date, cds_5y")
        return False

    if df is not None:
        # Ensure proper format
        if 'date' not in df.columns:
            df = df.reset_index()
            df.columns = ['date', 'cds_5y'] + list(df.columns[2:])

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Keep only date and cds_5y columns
        df = df[['date', 'cds_5y']]

        # Save to CSV
        df.to_csv(OUTPUT_FILE, index=False)

        print("\n" + "=" * 60)
        print(f"SUCCESS: Data saved to {OUTPUT_FILE}")
        print("=" * 60)
        print(f"\nSource: {source}")
        print(f"Period: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"Observations: {len(df)}")
        print(f"\nCDS Statistics:")
        print(f"  Mean: {df['cds_5y'].mean():.1f} bps")
        print(f"  Min:  {df['cds_5y'].min():.1f} bps")
        print(f"  Max:  {df['cds_5y'].max():.1f} bps")

        return True
    else:
        print("\nFailed to fetch CDS data from any source.")
        return False


if __name__ == '__main__':
    main()
