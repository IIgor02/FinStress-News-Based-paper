#!/usr/bin/env python3
"""
Data Collection Script for FSI
==============================
Script 2 of 3: Data Collection

Collects data from three sources:
1. Twitter/X - Social media posts (snscrape or API v2)
2. Google Trends - Search volume index (pytrends)
3. IBOVESPA - Market data (yfinance)

Usage:
    # Collect all data (real data from APIs)
    python scripts/collect_data.py --all

    # Collect only Google Trends data (free, no API key)
    python scripts/collect_data.py --google-trends

    # Collect only Twitter data
    python scripts/collect_data.py --twitter

    # Generate synthetic data for testing
    python scripts/collect_data.py --synthetic

    # Custom date range
    python scripts/collect_data.py --all --start-date 2015-01-01 --end-date 2025-12-31

Output:
    data/raw/twitter_data.csv
    data/raw/google_trends_data.csv
    data/raw/ibovespa_data.csv
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.dictionaries import (
    STRESS_QUERIES_GT,
    QUERY_WEIGHTS,
    CRISIS_EPISODES,
    FINANCIAL_TERMS,
    STRESS_TERMS,
    NEGATIVE_TERMS,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Twitter search queries
TWITTER_SEARCH_QUERIES = [
    'ibovespa', 'bolsa de valores brasil', 'mercado financeiro',
    'crise financeira brasil', 'dólar real', 'selic copom',
    'petrobras vale ações', 'investimentos brasil',
]

# Rate limiting
GOOGLE_TRENDS_DELAY = 3  # seconds between batches
TWITTER_DELAY = 2


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

    Args:
        queries: List of search queries (max 5 per batch due to API limit)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        geo: Geographic region (BR = Brazil)
        output_path: Path to save CSV output

    Returns:
        DataFrame with weekly SVI for each query
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("❌ pytrends not installed. Install with: pip install pytrends")
        print("   Generating synthetic Google Trends data instead.")
        return generate_synthetic_google_trends(
            queries, start_date, end_date, output_path
        )

    print(f"  Collecting Google Trends data for {len(queries)} queries...")
    print(f"  Region: {geo}, Period: {start_date} to {end_date}")

    pytrends = TrendReq(hl='pt-BR', tz=180)  # Portuguese Brazil, GMT-3
    all_data = []
    timeframe = f'{start_date} {end_date}'

    # Process in batches of 5 (Google Trends API limit)
    for i in range(0, len(queries), 5):
        batch = queries[i:i+5]
        batch_num = i // 5 + 1
        total_batches = (len(queries) + 4) // 5

        print(f"    Batch {batch_num}/{total_batches}: {batch}")

        try:
            pytrends.build_payload(
                batch,
                timeframe=timeframe,
                geo=geo,
                cat=0,  # All categories
            )

            df = pytrends.interest_over_time()

            if not df.empty:
                # Remove the isPartial column
                if 'isPartial' in df.columns:
                    df = df.drop('isPartial', axis=1)
                all_data.append(df)
                print(f"      ✓ Retrieved {len(df)} data points")
            else:
                print(f"      ⚠ No data returned for batch")

        except Exception as e:
            print(f"      ❌ Error: {e}")

        # Rate limiting
        time.sleep(GOOGLE_TRENDS_DELAY)

    if not all_data:
        print("  ⚠ No Google Trends data collected. Using synthetic data.")
        return generate_synthetic_google_trends(
            queries, start_date, end_date, output_path
        )

    # Combine all batches
    result = pd.concat(all_data, axis=1)

    # Handle duplicate columns (from overlapping batches)
    result = result.loc[:, ~result.columns.duplicated()]

    result.index.name = 'date'
    result = result.reset_index()

    # Add metadata
    result['source'] = 'google_trends'

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"  💾 Saved {len(result)} weeks of Google Trends data to {output_path}")

    return result


def generate_synthetic_google_trends(
    queries: List[str],
    start_date: str = '2015-01-01',
    end_date: str = '2025-12-31',
    output_path: Optional[str] = None,
    seed: int = 45,
) -> pd.DataFrame:
    """
    Generate synthetic Google Trends data for testing.

    Simulates realistic SVI patterns with:
    - Base level around 20-40
    - Spikes during crisis periods (up to 100)
    - Seasonal variation
    - Random noise

    Args:
        queries: List of query names
        start_date: Start date
        end_date: End date
        output_path: Path to save output
        seed: Random seed

    Returns:
        DataFrame with synthetic SVI data
    """
    np.random.seed(seed)

    # Generate weekly date range
    dates = pd.date_range(start_date, end_date, freq='W-SUN')
    n_weeks = len(dates)

    data = {'date': dates}

    for query in queries:
        # Base level with random variation
        base = np.random.uniform(15, 35)
        svi = np.ones(n_weeks) * base

        # Add trend component
        trend = np.linspace(0, 5, n_weeks)
        svi += trend

        # Add seasonal component (higher in Q1, Q4)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
        svi += seasonal

        # Add spikes during crisis periods
        for (start, end), label in CRISIS_EPISODES.items():
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)

            mask = (dates >= start_dt) & (dates <= end_dt)
            spike_size = np.random.uniform(30, 70)
            svi[mask] += spike_size

        # Add random noise
        noise = np.random.normal(0, 5, n_weeks)
        svi += noise

        # Clip to 0-100 range
        svi = np.clip(svi, 0, 100)

        # Apply query weight to add variation between queries
        weight = QUERY_WEIGHTS.get(query, 1.0)
        svi = svi * (0.8 + 0.4 * weight)
        svi = np.clip(svi, 0, 100)

        data[query] = svi.astype(int)

    df = pd.DataFrame(data)
    df['source'] = 'synthetic'

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  💾 Saved {len(df)} weeks of synthetic Google Trends data to {output_path}")

    return df


# =============================================================================
# TWITTER DATA COLLECTION
# =============================================================================

def collect_twitter_snscrape(
    queries: List[str],
    start_date: str,
    end_date: str,
    max_tweets_per_query: int = 10000,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Collect Twitter data using snscrape (no API key needed).

    Args:
        queries: List of search queries
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_tweets_per_query: Maximum tweets per query
        output_path: Path to save output

    Returns:
        DataFrame with collected tweets
    """
    try:
        import snscrape.modules.twitter as sntwitter
    except ImportError:
        print("❌ snscrape not installed. Install with: pip install snscrape")
        print("   Generating synthetic Twitter data instead.")
        return generate_synthetic_twitter(start_date, end_date, output_path=output_path)

    all_tweets = []

    for query in queries:
        print(f"  Searching Twitter for: '{query}'...")
        search_query = f"{query} lang:pt since:{start_date} until:{end_date}"

        try:
            tweet_count = 0
            for tweet in sntwitter.TwitterSearchScraper(search_query).get_items():
                if tweet_count >= max_tweets_per_query:
                    break

                all_tweets.append({
                    'date': tweet.date,
                    'text': tweet.rawContent,
                    'followers': tweet.user.followersCount,
                    'user_id': tweet.user.username,
                    'likes': tweet.likeCount,
                    'retweets': tweet.retweetCount,
                    'query': query,
                    'platform': 'twitter',
                })
                tweet_count += 1

            print(f"    Collected {tweet_count:,} tweets")
            time.sleep(TWITTER_DELAY)

        except Exception as e:
            print(f"    ⚠ Error: {e}")

    if not all_tweets:
        print("  ⚠ No tweets collected. Using synthetic data.")
        return generate_synthetic_twitter(start_date, end_date, output_path=output_path)

    df = pd.DataFrame(all_tweets)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['text', 'user_id', 'date'])
    df = df.sort_values('date').reset_index(drop=True)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  💾 Saved {len(df):,} tweets to {output_path}")

    return df


# =============================================================================
# SYNTHETIC TWITTER DATA
# =============================================================================

TWEET_TEMPLATES = {
    'stress': [
        "A {stress} no {financial} está gerando {negative} entre investidores",
        "{stress} no {financial} causa {negative} nas ações",
        "Analistas alertam para {stress} com {negative} no {financial}",
        "{financial} em {negative} com {stress} generalizado",
        "URGENTE: {financial} sofre {negative} em meio a {stress}",
        "{stress} afeta {financial} e gera {negative}",
        "Com a {stress} no {financial}, investidores enfrentam {negative}",
        "{financial} despenca em dia de {stress} e {negative}",
        "Temor de {stress} causa {negative} no {financial}",
        "{negative} no {financial} após sinais de {stress}",
    ],
    'financial_only': [
        "{financial} fecha em alta nesta sessão",
        "Investidores otimistas com {financial}",
        "{financial} tem dia de ganhos expressivos",
        "Analistas recomendam compra de ações do {financial}",
        "{financial} supera expectativas do mercado",
        "Bons resultados impulsionam {financial}",
        "{financial} atinge nova máxima histórica",
        "Otimismo com {financial} anima investidores",
    ],
    'non_financial': [
        "Que dia lindo hoje! Bom fim de semana a todos",
        "Alguém mais viu o jogo ontem? Que partida!",
        "Recomendo muito esse restaurante novo",
        "Feriado chegando, hora de descansar",
        "Que filme incrível, vale muito a pena",
        "Bom dia pessoal! Vamos que vamos",
    ],
}

SAMPLE_FINANCIAL = ['mercado', 'bolsa', 'Ibovespa', 'mercado financeiro', 'setor bancário', 'economia', 'investimentos']
SAMPLE_STRESS = ['crise', 'risco', 'volatilidade', 'instabilidade', 'incerteza', 'pânico', 'turbulência']
SAMPLE_NEGATIVE = ['queda', 'perdas', 'prejuízo', 'desvalorização', 'retração', 'problemas', 'piora']


def get_stress_probability(date: datetime, base_prob: float = 0.15) -> float:
    """Get stress probability based on date (higher during crises)."""
    date_str = date.strftime('%Y-%m-%d')

    for (start, end), label in CRISIS_EPISODES.items():
        if start <= date_str <= end:
            return min(base_prob * 3.5, 0.6)

    return base_prob


def generate_tweet(category: str) -> str:
    """Generate a synthetic tweet."""
    template = random.choice(TWEET_TEMPLATES[category])

    tweet = template.format(
        financial=random.choice(SAMPLE_FINANCIAL),
        stress=random.choice(SAMPLE_STRESS),
        negative=random.choice(SAMPLE_NEGATIVE),
    )

    if random.random() < 0.25:
        hashtags = ['#economia', '#mercado', '#investimentos', '#bolsa', '#ibovespa']
        tweet += ' ' + random.choice(hashtags)

    return tweet


def generate_synthetic_twitter(
    start_date: str,
    end_date: str,
    posts_per_day: Tuple[int, int] = (80, 180),
    stress_ratio: float = 0.15,
    financial_ratio: float = 0.65,
    output_path: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic Twitter data for testing."""
    random.seed(seed)
    np.random.seed(seed)

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    data = []
    current_date = start

    while current_date <= end:
        n_posts = random.randint(*posts_per_day)
        stress_prob = get_stress_probability(current_date, stress_ratio)

        for _ in range(n_posts):
            if random.random() < financial_ratio:
                category = 'stress' if random.random() < stress_prob else 'financial_only'
            else:
                category = 'non_financial'

            text = generate_tweet(category)
            followers = int(100 * (1 - random.random()) ** (-1/1.5))
            followers = min(followers, 5_000_000)

            hour = random.randint(6, 23)
            minute = random.randint(0, 59)
            timestamp = current_date.replace(hour=hour, minute=minute)

            data.append({
                'date': timestamp,
                'text': text,
                'followers': followers,
                'user_id': f'user_{random.randint(1, 100000)}',
                'likes': random.randint(0, 1000),
                'retweets': random.randint(0, 200),
                'platform': 'twitter',
                '_category': category,
            })

        current_date += timedelta(days=1)

    df = pd.DataFrame(data)
    df = df.sort_values('date').reset_index(drop=True)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  💾 Saved {len(df):,} synthetic tweets to {output_path}")

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
        print("❌ yfinance not installed. Install with: pip install yfinance")
        print("   Generating synthetic market data.")
        return generate_synthetic_market(start_date, end_date, output_path)

    print(f"  Downloading IBOVESPA (^BVSP) data...")

    try:
        ticker = yf.Ticker('^BVSP')
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            print("    ⚠ No data returned. Using synthetic data.")
            return generate_synthetic_market(start_date, end_date, output_path)

        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
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
            print(f"  💾 Saved to {output_path}")

        return df

    except Exception as e:
        print(f"    ❌ Error: {e}")
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

    for i, date in enumerate(dates):
        date_str = date.strftime('%Y-%m-%d')
        vol_multiplier = 1.0

        for (start, end), _ in CRISIS_EPISODES.items():
            if start <= date_str <= end:
                vol_multiplier = 2.5
                break

        daily_vol = base_vol * vol_multiplier
        daily_return = np.random.normal(0.0002, daily_vol)
        returns.append(daily_return)

    initial_price = 100000
    prices = [initial_price]
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
        print(f"  💾 Saved {len(df):,} days of synthetic market data to {output_path}")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect data for Financial Stress Index',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate synthetic data for testing
    python scripts/collect_data.py --synthetic

    # Collect Google Trends data (free, no API key)
    python scripts/collect_data.py --google-trends

    # Collect all data sources
    python scripts/collect_data.py --all

Environment Variables (optional):
    TWITTER_BEARER_TOKEN - For Twitter API v2
        """
    )

    parser.add_argument('--all', action='store_true', help='Collect all data sources')
    parser.add_argument('--twitter', action='store_true', help='Collect Twitter data')
    parser.add_argument('--google-trends', action='store_true', help='Collect Google Trends data')
    parser.add_argument('--ibovespa', action='store_true', help='Collect IBOVESPA market data')
    parser.add_argument('--synthetic', action='store_true', help='Generate synthetic data')
    parser.add_argument('--start-date', type=str, default='2015-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default='2025-12-31', help='End date')
    parser.add_argument('--max-tweets', type=int, default=10000, help='Max tweets per query')

    args = parser.parse_args()

    # Default to synthetic if nothing specified
    if not any([args.all, args.twitter, args.google_trends, args.ibovespa, args.synthetic]):
        args.synthetic = True
        args.ibovespa = True

    print("=" * 60)
    print("DATA COLLECTION FOR FSI")
    print("Twitter + Google Trends + IBOVESPA")
    print("=" * 60)
    print(f"\nDate range: {args.start_date} to {args.end_date}")
    print(f"Output: {DATA_DIR}")

    # IBOVESPA
    if args.ibovespa or args.all or args.synthetic:
        print("\n" + "-" * 40)
        print("📈 IBOVESPA Market Data")
        print("-" * 40)

        market_df = collect_ibovespa_data(
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=DATA_DIR / 'ibovespa_data.csv',
        )

        if market_df.empty and args.synthetic:
            market_df = generate_synthetic_market(
                args.start_date, args.end_date,
                output_path=DATA_DIR / 'ibovespa_data.csv',
            )

    # Google Trends
    if args.google_trends or args.all:
        print("\n" + "-" * 40)
        print("🔍 Google Trends Data")
        print("-" * 40)

        gt_df = collect_google_trends(
            queries=STRESS_QUERIES_GT,
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=DATA_DIR / 'google_trends_data.csv',
        )

    elif args.synthetic:
        print("\n" + "-" * 40)
        print("🔍 Google Trends Data (Synthetic)")
        print("-" * 40)

        gt_df = generate_synthetic_google_trends(
            queries=STRESS_QUERIES_GT,
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=DATA_DIR / 'google_trends_data.csv',
        )

    # Twitter
    if args.twitter or args.all:
        print("\n" + "-" * 40)
        print("🐦 Twitter Data")
        print("-" * 40)

        twitter_df = collect_twitter_snscrape(
            queries=TWITTER_SEARCH_QUERIES,
            start_date=args.start_date,
            end_date=args.end_date,
            max_tweets_per_query=args.max_tweets,
            output_path=DATA_DIR / 'twitter_data.csv',
        )

    elif args.synthetic:
        print("\n" + "-" * 40)
        print("🐦 Twitter Data (Synthetic)")
        print("-" * 40)

        twitter_df = generate_synthetic_twitter(
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=DATA_DIR / 'twitter_data.csv',
        )

    # Summary
    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)
    print(f"\nFiles saved to: {DATA_DIR}/")
    print("  - twitter_data.csv")
    print("  - google_trends_data.csv")
    print("  - ibovespa_data.csv")
    print("\nNext step: Run the FSI calculation script:")
    print("  python scripts/run_fsi.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
