#!/usr/bin/env python3
"""
Data Collection Script
======================
Collects real data from Google Trends and Yahoo Finance.

Usage:
    # Dictionary method (25 queries)
    python scripts/collect_data.py

    # ML method (104 queries)
    python scripts/collect_data.py --ml

    # Custom date range
    python scripts/collect_data.py --start-date 2015-01-01 --end-date 2025-12-31

Output:
    data/raw/google_trends_data.csv (or google_trends_ml.csv with --ml)
    data/raw/ibovespa_data.csv
"""

import argparse
import io
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

# Handle both script execution and interactive/REPL usage
try:
    _SCRIPT_DIR = Path(__file__).parent
    _PROJECT_DIR = _SCRIPT_DIR.parent
except NameError:
    _PROJECT_DIR = Path.cwd()
    _SCRIPT_DIR = _PROJECT_DIR / 'scripts'

sys.path.insert(0, str(_PROJECT_DIR))

from scripts.dictionaries import (
    STRESS_QUERIES_GT,
    ML_QUERIES_ALL,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = _PROJECT_DIR / 'data' / 'raw'
DATA_DIR.mkdir(parents=True, exist_ok=True)

GOOGLE_TRENDS_DELAY = 3  # seconds between batches


# =============================================================================
# GOOGLE TRENDS DATA COLLECTION
# =============================================================================

def collect_google_trends(
    queries: List[str],
    start_date: str,
    end_date: str,
    geo: str = 'BR',
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Collect Google Trends Search Volume Index (SVI) for queries.
    Uses pytrends library (free, no API key required).
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("  ERROR: pytrends not installed.")
        print("  Install with: pip install pytrends")
        return pd.DataFrame()

    print(f"  Collecting data for {len(queries)} queries...")
    print(f"  Region: {geo}, Period: {start_date} to {end_date}")

    pytrends = TrendReq(hl='pt-BR', tz=180)
    all_data = []
    timeframe = f'{start_date} {end_date}'

    # Process in batches of 5 (Google Trends API limit)
    total_batches = (len(queries) + 4) // 5
    for i in range(0, len(queries), 5):
        batch = queries[i:i+5]
        batch_num = i // 5 + 1

        print(f"    Batch {batch_num}/{total_batches}: {batch}")

        try:
            pytrends.build_payload(batch, timeframe=timeframe, geo=geo, cat=0)
            df = pytrends.interest_over_time()

            if not df.empty:
                if 'isPartial' in df.columns:
                    df = df.drop('isPartial', axis=1)
                all_data.append(df)
                print(f"      Retrieved {len(df)} weeks")
            else:
                print(f"      No data returned")

        except Exception as e:
            print(f"      Error: {e}")

        if batch_num < total_batches:
            time.sleep(GOOGLE_TRENDS_DELAY)

    if not all_data:
        print("  ERROR: No data collected from Google Trends")
        return pd.DataFrame()

    # Combine all batches
    result = pd.concat(all_data, axis=1)
    result = result.loc[:, ~result.columns.duplicated()]
    result.index.name = 'date'
    result = result.reset_index()

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"  Saved {len(result)} weeks to {output_path}")

    return result


# =============================================================================
# IBOVESPA DATA COLLECTION
# =============================================================================

def collect_ibovespa(
    start_date: str,
    end_date: str,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Download IBOVESPA data from Yahoo Finance or alternative sources.
    Calculates returns and realized volatility.
    """
    df = None
    source = None

    # Try yfinance first
    try:
        import yfinance as yf
        print(f"  Downloading IBOVESPA (^BVSP) from Yahoo Finance...")
        ticker = yf.Ticker('^BVSP')
        df = ticker.history(start=start_date, end=end_date)
        if not df.empty:
            source = "Yahoo Finance (^BVSP)"
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date', 'Open': 'open', 'High': 'high',
                'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            })
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
    except ImportError:
        print("  yfinance not available, trying alternative sources...")
    except Exception as e:
        print(f"  Yahoo Finance failed: {e}")

    # Try stooq.com for EWZ (Brazil ETF) as backup
    if df is None or df.empty:
        try:
            import requests
            import io
            print("  Downloading EWZ (Brazil ETF) from stooq.com...")
            start_fmt = start_date.replace('-', '')
            end_fmt = end_date.replace('-', '')
            url = f'https://stooq.com/q/d/l/?s=ewz.us&d1={start_fmt}&d2={end_fmt}&i=d'
            response = requests.get(url, timeout=30)
            if response.status_code == 200 and len(response.content) > 100:
                df = pd.read_csv(io.StringIO(response.text))
                if len(df) > 10:
                    source = "stooq.com (EWZ)"
                    df.columns = [c.lower() for c in df.columns]
                    df = df.sort_values('date')
        except Exception as e:
            print(f"  stooq.com failed: {e}")

    if df is None or df.empty:
        print("  ERROR: Could not fetch IBOVESPA data from any source")
        return pd.DataFrame()

    # Calculate returns and volatility
    df['return'] = df['close'].pct_change()
    df['realized_vol_5d'] = df['return'].rolling(5).std() * np.sqrt(252) * 100
    df['realized_vol_21d'] = df['return'].rolling(21).std() * np.sqrt(252) * 100
    df['realized_vol_30d'] = df['return'].rolling(30).std() * np.sqrt(252) * 100
    df['intraday_range'] = (df['high'] - df['low']) / df['close'] * 100

    # Ensure required columns exist
    cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'return',
            'realized_vol_5d', 'realized_vol_21d', 'realized_vol_30d', 'intraday_range']
    df = df[[c for c in cols if c in df.columns]]

    print(f"    Source: {source}")
    print(f"    Downloaded {len(df):,} trading days")
    print(f"    Period: {df['date'].min()} to {df['date'].max()}")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect Google Trends and IBOVESPA data for FSI',
    )
    parser.add_argument('--ml', action='store_true',
                        help='Collect expanded queries for ML approach (104 queries)')
    parser.add_argument('--start-date', type=str, default='2008-01-01',
                        help='Start date (default: 2008-01-01)')
    parser.add_argument('--end-date', type=str, default='2025-12-31',
                        help='End date (default: 2025-12-31)')

    args = parser.parse_args()

    # Select query set based on method
    if args.ml:
        queries = ML_QUERIES_ALL
        gt_file = 'google_trends_ml.csv'
        method = "ML (García et al. 2023)"
        next_script = "ml_fsi.py"
    else:
        queries = STRESS_QUERIES_GT
        gt_file = 'google_trends_data.csv'
        method = "Dictionary (Da et al. 2011)"
        next_script = "run_fsi.py"

    print("=" * 60)
    print("DATA COLLECTION")
    print(f"Method: {method}")
    print("=" * 60)
    print(f"\nPeriod: {args.start_date} to {args.end_date}")
    print(f"Queries: {len(queries)}")
    print(f"Output: {DATA_DIR}")

    # Collect IBOVESPA
    print("\n" + "-" * 40)
    print("IBOVESPA Market Data")
    print("-" * 40)
    ibov = collect_ibovespa(args.start_date, args.end_date, DATA_DIR / 'ibovespa_data.csv')

    # Collect Google Trends
    print("\n" + "-" * 40)
    print(f"Google Trends ({len(queries)} queries)")
    print("-" * 40)
    gt = collect_google_trends(queries, args.start_date, args.end_date, output_path=DATA_DIR / gt_file)

    # Summary
    print("\n" + "=" * 60)
    if not ibov.empty and not gt.empty:
        print("DATA COLLECTION COMPLETE")
        print("=" * 60)
        print(f"\nFiles saved to: {DATA_DIR}/")
        print(f"  - {gt_file} ({len(gt)} weeks)")
        print(f"  - ibovespa_data.csv ({len(ibov)} days)")
        print(f"\nNext step: python scripts/{next_script}")
    else:
        print("DATA COLLECTION FAILED")
        print("=" * 60)
        print("\nCheck your internet connection and try again.")
    print("=" * 60)


if __name__ == '__main__':
    main()
