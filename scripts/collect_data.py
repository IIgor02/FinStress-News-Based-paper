#!/usr/bin/env python3
"""
Data Collection Script for FSI
==============================
Collects data from two sources:
1. Google Trends - Search volume index (pytrends)
2. IBOVESPA - Market data (yfinance)

Usage:
    # Collect real data
    python scripts/collect_data.py

    # Custom date range
    python scripts/collect_data.py --start-date 2015-01-01 --end-date 2025-12-31

    # Generate synthetic data for testing (no internet needed)
    python scripts/collect_data.py --synthetic

Output:
    data/raw/google_trends_data.csv
    data/raw/ibovespa_data.csv
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

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
    QUERY_WEIGHTS,
    CRISIS_EPISODES,
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
    start_date: str = '2015-01-01',
    end_date: str = '2025-12-31',
    geo: str = 'BR',
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Collect Google Trends Search Volume Index (SVI) for stress queries.
    Uses pytrends library (no authentication required).
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("  pytrends not installed. Install with: pip install pytrends")
        print("  Generating synthetic data instead.")
        return generate_synthetic_google_trends(queries, start_date, end_date, output_path)

    print(f"  Collecting Google Trends data for {len(queries)} queries...")
    print(f"  Region: {geo}, Period: {start_date} to {end_date}")

    pytrends = TrendReq(hl='pt-BR', tz=180)
    all_data = []
    timeframe = f'{start_date} {end_date}'

    # Process in batches of 5 (Google Trends API limit)
    for i in range(0, len(queries), 5):
        batch = queries[i:i+5]
        batch_num = i // 5 + 1
        total_batches = (len(queries) + 4) // 5

        print(f"    Batch {batch_num}/{total_batches}: {batch}")

        try:
            pytrends.build_payload(batch, timeframe=timeframe, geo=geo, cat=0)
            df = pytrends.interest_over_time()

            if not df.empty:
                if 'isPartial' in df.columns:
                    df = df.drop('isPartial', axis=1)
                all_data.append(df)
                print(f"      Retrieved {len(df)} data points")
            else:
                print(f"      No data returned for batch")

        except Exception as e:
            print(f"      Error: {e}")

        time.sleep(GOOGLE_TRENDS_DELAY)

    if not all_data:
        print("  No Google Trends data collected. Using synthetic data.")
        return generate_synthetic_google_trends(queries, start_date, end_date, output_path)

    result = pd.concat(all_data, axis=1)
    result = result.loc[:, ~result.columns.duplicated()]
    result.index.name = 'date'
    result = result.reset_index()
    result['source'] = 'google_trends'

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"  Saved {len(result)} weeks to {output_path}")

    return result


def generate_synthetic_google_trends(
    queries: List[str],
    start_date: str = '2015-01-01',
    end_date: str = '2025-12-31',
    output_path: Optional[str] = None,
    seed: int = 45,
) -> pd.DataFrame:
    """Generate synthetic Google Trends data for testing."""
    np.random.seed(seed)

    dates = pd.date_range(start_date, end_date, freq='W-SUN')
    n_weeks = len(dates)
    data = {'date': dates}

    for query in queries:
        base = np.random.uniform(15, 35)
        svi = np.ones(n_weeks) * base
        svi += np.linspace(0, 5, n_weeks)  # trend
        svi += 5 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)  # seasonal

        # Crisis spikes
        for (start, end), _ in CRISIS_EPISODES.items():
            mask = (dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))
            svi[mask] += np.random.uniform(30, 70)

        svi += np.random.normal(0, 5, n_weeks)
        svi = np.clip(svi, 0, 100)

        weight = QUERY_WEIGHTS.get(query, 1.0)
        svi = np.clip(svi * (0.8 + 0.4 * weight), 0, 100)
        data[query] = svi.astype(int)

    df = pd.DataFrame(data)
    df['source'] = 'synthetic'

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  Saved {len(df)} weeks of synthetic data to {output_path}")

    return df


# =============================================================================
# IBOVESPA DATA COLLECTION
# =============================================================================

def collect_ibovespa_data(
    start_date: str,
    end_date: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Download IBOVESPA data from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed. Install with: pip install yfinance")
        print("  Generating synthetic market data.")
        return generate_synthetic_market(start_date, end_date, output_path)

    print(f"  Downloading IBOVESPA (^BVSP) data...")

    try:
        ticker = yf.Ticker('^BVSP')
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            print("    No data returned. Using synthetic data.")
            return generate_synthetic_market(start_date, end_date, output_path)

        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'date', 'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume',
        })

        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)

        # Calculate returns and volatility
        df['return'] = df['close'].pct_change()
        df['realized_vol_5d'] = df['return'].rolling(5).std() * np.sqrt(252) * 100
        df['realized_vol_21d'] = df['return'].rolling(21).std() * np.sqrt(252) * 100
        df['realized_vol_30d'] = df['return'].rolling(30).std() * np.sqrt(252) * 100
        df['intraday_range'] = (df['high'] - df['low']) / df['close'] * 100

        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'return',
                 'realized_vol_5d', 'realized_vol_21d', 'realized_vol_30d', 'intraday_range']]

        print(f"    Downloaded {len(df):,} trading days")
        print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")

        if output_path:
            df.to_csv(output_path, index=False)
            print(f"  Saved to {output_path}")

        return df

    except Exception as e:
        print(f"    Error: {e}")
        return generate_synthetic_market(start_date, end_date, output_path)


def generate_synthetic_market(
    start_date: str,
    end_date: str,
    output_path: Optional[str] = None,
    seed: int = 44,
) -> pd.DataFrame:
    """Generate synthetic market data mimicking IBOVESPA."""
    np.random.seed(seed)

    dates = pd.date_range(start_date, end_date, freq='B')
    n_days = len(dates)
    base_vol = 0.015
    returns = []

    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        vol_multiplier = 1.0

        for (start, end), _ in CRISIS_EPISODES.items():
            if start <= date_str <= end:
                vol_multiplier = 2.5
                break

        returns.append(np.random.normal(0.0002, base_vol * vol_multiplier))

    prices = [100000]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    returns_series = pd.Series(returns)

    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(5e9, 15e9, n_days),
        'return': returns,
        'realized_vol_5d': returns_series.rolling(5).std() * np.sqrt(252) * 100,
        'realized_vol_21d': returns_series.rolling(21).std() * np.sqrt(252) * 100,
        'realized_vol_30d': returns_series.rolling(30).std() * np.sqrt(252) * 100,
    })
    df['intraday_range'] = (df['high'] - df['low']) / df['close'] * 100

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  Saved {len(df):,} days of synthetic market data to {output_path}")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect data for Financial Stress Index (Google Trends + IBOVESPA)',
    )
    parser.add_argument('--synthetic', action='store_true', help='Generate synthetic data (no internet)')
    parser.add_argument('--ml', action='store_true', help='Collect expanded queries for ML approach (~100 queries)')
    parser.add_argument('--start-date', type=str, default='2015-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default='2025-12-31', help='End date')

    args = parser.parse_args()

    # Select query set
    if args.ml:
        queries = ML_QUERIES_ALL
        gt_file = 'google_trends_ml.csv'
        method_name = "ML (García et al. 2023)"
    else:
        queries = STRESS_QUERIES_GT
        gt_file = 'google_trends_data.csv'
        method_name = "Dictionary (Da et al. 2011)"

    print("=" * 60)
    print("DATA COLLECTION FOR FSI")
    print(f"Method: {method_name}")
    print("=" * 60)
    print(f"\nDate range: {args.start_date} to {args.end_date}")
    print(f"Queries: {len(queries)}")
    print(f"Output: {DATA_DIR}")

    # IBOVESPA
    print("\n" + "-" * 40)
    print("IBOVESPA Market Data")
    print("-" * 40)

    if args.synthetic:
        generate_synthetic_market(args.start_date, args.end_date, DATA_DIR / 'ibovespa_data.csv')
    else:
        collect_ibovespa_data(args.start_date, args.end_date, DATA_DIR / 'ibovespa_data.csv')

    # Google Trends
    print("\n" + "-" * 40)
    print(f"Google Trends Data ({len(queries)} queries)")
    print("-" * 40)

    if args.synthetic:
        generate_synthetic_google_trends(queries, args.start_date, args.end_date, DATA_DIR / gt_file)
    else:
        collect_google_trends(queries, args.start_date, args.end_date, output_path=DATA_DIR / gt_file)

    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)
    print(f"\nFiles saved to: {DATA_DIR}/")
    print(f"  - {gt_file}")
    print("  - ibovespa_data.csv")
    if args.ml:
        print("\nNext step: python scripts/ml_fsi.py")
    else:
        print("\nNext step: python scripts/run_fsi.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
