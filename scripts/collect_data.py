#!/usr/bin/env python3
"""
Data Collection Script for FSI
==============================
Script 1 of 3: Data Collection

Collects social media data from Twitter/X and Reddit for FSI calculation.
Also downloads IBOVESPA market data from Yahoo Finance.

Data Sources:
1. Twitter/X - Using snscrape (no API key needed) or Twitter API v2
2. Reddit - Using PRAW (Reddit API) or Pushshift
3. IBOVESPA - Using yfinance (Yahoo Finance)

Usage:
    # Collect all data (Twitter, Reddit, IBOVESPA)
    python scripts/collect_data.py --all --start-date 2020-01-01 --end-date 2024-12-31

    # Collect only Twitter data
    python scripts/collect_data.py --twitter --start-date 2020-01-01 --end-date 2024-12-31

    # Collect only Reddit data
    python scripts/collect_data.py --reddit --start-date 2020-01-01 --end-date 2024-12-31

    # Collect only IBOVESPA data
    python scripts/collect_data.py --ibovespa --start-date 2020-01-01 --end-date 2024-12-31

    # Generate synthetic data for testing
    python scripts/collect_data.py --synthetic --start-date 2020-01-01 --end-date 2024-12-31

Output:
    data/raw/twitter_data.csv
    data/raw/reddit_data.csv
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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.dictionaries import FINANCIAL_TERMS, STRESS_TERMS, NEGATIVE_TERMS, CRISIS_EPISODES


# =============================================================================
# CONFIGURATION
# =============================================================================

# Output paths
DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Twitter search queries (Portuguese financial terms)
TWITTER_SEARCH_QUERIES = [
    'ibovespa',
    'bolsa de valores brasil',
    'mercado financeiro brasil',
    'ações brasileiras',
    'investimentos brasil',
    'crise financeira brasil',
    'dólar real',
    'selic copom',
    'petrobras vale',
    'banco central brasil',
]

# Reddit subreddits to scrape
REDDIT_SUBREDDITS = [
    'investimentos',
    'brasil',
    'farialimabets',
    'financas',
    'bolsapt',
    'economia',
    'mercadofinanceiro',
    'breconomia',
]

# Rate limiting
TWITTER_DELAY = 2  # seconds between requests
REDDIT_DELAY = 1   # seconds between requests


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

    Note: snscrape may have limitations and requires installation:
        pip install snscrape

    Args:
        queries: List of search queries
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_tweets_per_query: Maximum tweets to collect per query
        output_path: Path to save CSV output

    Returns:
        DataFrame with collected tweets
    """
    try:
        import snscrape.modules.twitter as sntwitter
    except ImportError:
        print("❌ snscrape not installed. Install with: pip install snscrape")
        print("   Falling back to synthetic data generation.")
        return pd.DataFrame()

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
            print(f"    ⚠️ Error: {e}")
            continue

    if not all_tweets:
        print("  ⚠️ No tweets collected")
        return pd.DataFrame()

    df = pd.DataFrame(all_tweets)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['text', 'user_id', 'date'])
    df = df.sort_values('date').reset_index(drop=True)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  💾 Saved {len(df):,} tweets to {output_path}")

    return df


def collect_twitter_api(
    queries: List[str],
    start_date: str,
    end_date: str,
    bearer_token: str,
    max_tweets_per_query: int = 10000,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Collect Twitter data using official Twitter API v2.

    Requires Twitter Developer account and bearer token.
    Set TWITTER_BEARER_TOKEN environment variable.

    Args:
        queries: List of search queries
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        bearer_token: Twitter API bearer token
        max_tweets_per_query: Maximum tweets per query
        output_path: Path to save CSV output

    Returns:
        DataFrame with collected tweets
    """
    try:
        import tweepy
    except ImportError:
        print("❌ tweepy not installed. Install with: pip install tweepy")
        return pd.DataFrame()

    if not bearer_token:
        print("❌ Twitter bearer token not provided.")
        print("   Set TWITTER_BEARER_TOKEN environment variable.")
        return pd.DataFrame()

    client = tweepy.Client(bearer_token=bearer_token)
    all_tweets = []

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    for query in queries:
        print(f"  Searching Twitter API for: '{query}'...")

        try:
            tweets = tweepy.Paginator(
                client.search_recent_tweets,
                query=f"{query} lang:pt -is:retweet",
                tweet_fields=['created_at', 'public_metrics', 'author_id'],
                user_fields=['public_metrics'],
                expansions=['author_id'],
                max_results=100,
            ).flatten(limit=max_tweets_per_query)

            tweet_count = 0
            for tweet in tweets:
                all_tweets.append({
                    'date': tweet.created_at,
                    'text': tweet.text,
                    'followers': tweet.author.public_metrics.get('followers_count', 0) if hasattr(tweet, 'author') else 0,
                    'user_id': tweet.author_id,
                    'likes': tweet.public_metrics.get('like_count', 0),
                    'retweets': tweet.public_metrics.get('retweet_count', 0),
                    'query': query,
                    'platform': 'twitter',
                })
                tweet_count += 1

            print(f"    Collected {tweet_count:,} tweets")
            time.sleep(TWITTER_DELAY)

        except Exception as e:
            print(f"    ⚠️ Error: {e}")
            continue

    if not all_tweets:
        return pd.DataFrame()

    df = pd.DataFrame(all_tweets)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['text', 'user_id'])
    df = df.sort_values('date').reset_index(drop=True)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  💾 Saved {len(df):,} tweets to {output_path}")

    return df


# =============================================================================
# REDDIT DATA COLLECTION
# =============================================================================

def collect_reddit_praw(
    subreddits: List[str],
    start_date: str,
    end_date: str,
    client_id: str,
    client_secret: str,
    user_agent: str = 'FSI-Collector/1.0',
    max_posts_per_subreddit: int = 10000,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Collect Reddit data using PRAW (Python Reddit API Wrapper).

    Requires Reddit API credentials:
    - Create app at: https://www.reddit.com/prefs/apps
    - Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables

    Args:
        subreddits: List of subreddit names
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        client_id: Reddit API client ID
        client_secret: Reddit API client secret
        user_agent: User agent string
        max_posts_per_subreddit: Maximum posts per subreddit
        output_path: Path to save CSV output

    Returns:
        DataFrame with collected posts
    """
    try:
        import praw
    except ImportError:
        print("❌ praw not installed. Install with: pip install praw")
        return pd.DataFrame()

    if not client_id or not client_secret:
        print("❌ Reddit API credentials not provided.")
        print("   Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")
        return pd.DataFrame()

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    start_timestamp = datetime.strptime(start_date, '%Y-%m-%d').timestamp()
    end_timestamp = datetime.strptime(end_date, '%Y-%m-%d').timestamp()

    all_posts = []

    for subreddit_name in subreddits:
        print(f"  Collecting from r/{subreddit_name}...")

        try:
            subreddit = reddit.subreddit(subreddit_name)
            post_count = 0

            # Get posts from different time periods
            for submission in subreddit.new(limit=max_posts_per_subreddit):
                created_time = submission.created_utc

                if start_timestamp <= created_time <= end_timestamp:
                    # Combine title and selftext
                    text = submission.title
                    if submission.selftext:
                        text += ' ' + submission.selftext

                    all_posts.append({
                        'date': datetime.fromtimestamp(created_time),
                        'text': text,
                        'upvotes': submission.score,
                        'subreddit': subreddit_name,
                        'num_comments': submission.num_comments,
                        'post_id': submission.id,
                        'platform': 'reddit',
                    })
                    post_count += 1

            print(f"    Collected {post_count:,} posts")
            time.sleep(REDDIT_DELAY)

        except Exception as e:
            print(f"    ⚠️ Error: {e}")
            continue

    if not all_posts:
        return pd.DataFrame()

    df = pd.DataFrame(all_posts)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['post_id'])
    df = df.sort_values('date').reset_index(drop=True)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  💾 Saved {len(df):,} posts to {output_path}")

    return df


# =============================================================================
# IBOVESPA DATA COLLECTION
# =============================================================================

def collect_ibovespa_data(
    start_date: str,
    end_date: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download IBOVESPA data from Yahoo Finance.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_path: Path to save CSV output

    Returns:
        DataFrame with IBOVESPA data
    """
    try:
        import yfinance as yf
    except ImportError:
        print("❌ yfinance not installed. Install with: pip install yfinance")
        return pd.DataFrame()

    print(f"  Downloading IBOVESPA (^BVSP) data...")

    try:
        ticker = yf.Ticker('^BVSP')
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            print("    ⚠️ No data returned from Yahoo Finance")
            return pd.DataFrame()

        # Process data
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        })

        # Remove timezone info if present
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)

        # Calculate returns
        df['return'] = df['close'].pct_change()

        # Calculate realized volatility (various windows)
        df['realized_vol_5d'] = df['return'].rolling(window=5).std() * np.sqrt(252) * 100
        df['realized_vol_21d'] = df['return'].rolling(window=21).std() * np.sqrt(252) * 100
        df['realized_vol_30d'] = df['return'].rolling(window=30).std() * np.sqrt(252) * 100
        df['realized_vol_63d'] = df['return'].rolling(window=63).std() * np.sqrt(252) * 100

        # Calculate intraday volatility (high-low range)
        df['intraday_range'] = (df['high'] - df['low']) / df['close'] * 100

        # Keep relevant columns
        df = df[[
            'date', 'open', 'high', 'low', 'close', 'volume',
            'return', 'realized_vol_5d', 'realized_vol_21d',
            'realized_vol_30d', 'realized_vol_63d', 'intraday_range'
        ]]

        print(f"    Downloaded {len(df):,} trading days")
        print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"    Volatility range (30d): {df['realized_vol_30d'].min():.1f}% - {df['realized_vol_30d'].max():.1f}%")

        if output_path:
            df.to_csv(output_path, index=False)
            print(f"  💾 Saved to {output_path}")

        return df

    except Exception as e:
        print(f"    ❌ Error downloading data: {e}")
        return pd.DataFrame()


# =============================================================================
# SYNTHETIC DATA GENERATION (for testing)
# =============================================================================

# Sample tweet templates
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
        "{financial} fecha com {negative} devido à {stress}",
        "Investidores preocupados com {stress} e {negative} no {financial}",
        "Alarme de {stress} pressiona {financial} com {negative}",
        "{stress} global atinge {financial} brasileiro com {negative} forte",
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
        "Blue chips lideram altas na {financial}",
        "Perspectiva positiva para o {financial}",
        "Fluxo estrangeiro impulsiona {financial}",
        "{financial} recupera perdas da semana anterior",
    ],
    'non_financial': [
        "Que dia lindo hoje! Bom fim de semana a todos",
        "Alguém mais viu o jogo ontem? Que partida!",
        "Recomendo muito esse restaurante novo",
        "Feriado chegando, hora de descansar",
        "Que filme incrível, vale muito a pena",
        "Bom dia pessoal! Vamos que vamos",
        "Saudades de viajar, quando isso vai acabar",
        "Essa receita ficou deliciosa",
        "Tempo bom para praia hoje",
        "Meu novo livro favorito chegou",
        "Domingo de descanso com a família",
        "Mais uma semana começando",
    ],
}

SAMPLE_FINANCIAL = ['mercado', 'bolsa', 'Ibovespa', 'mercado financeiro', 'setor bancário', 'economia', 'investimentos', 'ações']
SAMPLE_STRESS = ['crise', 'risco', 'volatilidade', 'instabilidade', 'incerteza', 'pânico', 'turbulência', 'tensão']
SAMPLE_NEGATIVE = ['queda', 'perdas', 'prejuízo', 'desvalorização', 'retração', 'problemas', 'piora', 'deterioração']


def get_stress_probability(date: datetime, base_prob: float = 0.15) -> float:
    """Get stress probability based on date (higher during crisis periods)."""
    date_str = date.strftime('%Y-%m-%d')

    for (start, end), label in CRISIS_EPISODES.items():
        if start <= date_str <= end:
            return min(base_prob * 3.5, 0.6)  # Higher stress during crises

    return base_prob


def generate_tweet(category: str) -> str:
    """Generate a synthetic tweet from templates."""
    template = random.choice(TWEET_TEMPLATES[category])

    tweet = template.format(
        financial=random.choice(SAMPLE_FINANCIAL),
        stress=random.choice(SAMPLE_STRESS),
        negative=random.choice(SAMPLE_NEGATIVE),
    )

    # Add hashtags sometimes
    if random.random() < 0.25:
        hashtags = ['#economia', '#mercado', '#investimentos', '#bolsa', '#ibovespa', '#brasil']
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

            # Generate follower count (power law distribution)
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


def generate_synthetic_reddit(
    start_date: str,
    end_date: str,
    posts_per_day: Tuple[int, int] = (30, 80),
    stress_ratio: float = 0.20,
    financial_ratio: float = 0.75,
    output_path: Optional[str] = None,
    seed: int = 43,
) -> pd.DataFrame:
    """Generate synthetic Reddit data for testing."""
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

            # Reddit posts tend to be longer
            if random.random() < 0.5:
                text += f" {generate_tweet(category)}"

            upvotes = int(np.random.exponential(50 if category == 'stress' else 30))
            upvotes = min(upvotes, 10000)

            hour = random.randint(8, 22)
            minute = random.randint(0, 59)
            timestamp = current_date.replace(hour=hour, minute=minute)

            data.append({
                'date': timestamp,
                'text': text,
                'upvotes': upvotes,
                'subreddit': random.choice(REDDIT_SUBREDDITS),
                'num_comments': random.randint(0, 200),
                'post_id': f'post_{random.randint(1, 1000000)}',
                'platform': 'reddit',
                '_category': category,
            })

        current_date += timedelta(days=1)

    df = pd.DataFrame(data)
    df = df.sort_values('date').reset_index(drop=True)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  💾 Saved {len(df):,} synthetic Reddit posts to {output_path}")

    return df


def generate_synthetic_market(
    start_date: str,
    end_date: str,
    output_path: Optional[str] = None,
    seed: int = 44,
) -> pd.DataFrame:
    """Generate synthetic market data mimicking IBOVESPA."""
    np.random.seed(seed)

    dates = pd.date_range(start_date, end_date, freq='B')  # Business days
    n_days = len(dates)

    base_vol = 0.015  # 1.5% daily volatility
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
        'realized_vol_5d': returns_series.rolling(window=5).std() * np.sqrt(252) * 100,
        'realized_vol_21d': returns_series.rolling(window=21).std() * np.sqrt(252) * 100,
        'realized_vol_30d': returns_series.rolling(window=30).std() * np.sqrt(252) * 100,
        'realized_vol_63d': returns_series.rolling(window=63).std() * np.sqrt(252) * 100,
    })

    df['intraday_range'] = (df['high'] - df['low']) / df['close'] * 100

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  💾 Saved {len(df):,} days of synthetic market data to {output_path}")

    return df


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect data for Financial Stress Index calculation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate synthetic data for testing
    python scripts/collect_data.py --synthetic --start-date 2020-01-01 --end-date 2024-12-31

    # Collect only IBOVESPA data (always available)
    python scripts/collect_data.py --ibovespa --start-date 2020-01-01 --end-date 2024-12-31

    # Collect all data (requires API credentials)
    python scripts/collect_data.py --all --start-date 2020-01-01 --end-date 2024-12-31

Environment Variables (for real data collection):
    TWITTER_BEARER_TOKEN - Twitter API v2 bearer token
    REDDIT_CLIENT_ID     - Reddit API client ID
    REDDIT_CLIENT_SECRET - Reddit API client secret
        """
    )

    parser.add_argument('--all', action='store_true', help='Collect all data sources')
    parser.add_argument('--twitter', action='store_true', help='Collect Twitter data')
    parser.add_argument('--reddit', action='store_true', help='Collect Reddit data')
    parser.add_argument('--ibovespa', action='store_true', help='Collect IBOVESPA market data')
    parser.add_argument('--synthetic', action='store_true', help='Generate synthetic data for testing')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--max-tweets', type=int, default=10000, help='Max tweets per query')
    parser.add_argument('--max-posts', type=int, default=10000, help='Max Reddit posts per subreddit')

    args = parser.parse_args()

    # Default to synthetic if nothing specified
    if not any([args.all, args.twitter, args.reddit, args.ibovespa, args.synthetic]):
        args.synthetic = True
        args.ibovespa = True  # Always try to get real market data

    print("=" * 60)
    print("DATA COLLECTION FOR FINANCIAL STRESS INDEX")
    print("=" * 60)
    print(f"\nDate range: {args.start_date} to {args.end_date}")
    print(f"Output directory: {DATA_DIR}")

    # Collect IBOVESPA data (always try real data first)
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
            print("  Generating synthetic market data...")
            market_df = generate_synthetic_market(
                start_date=args.start_date,
                end_date=args.end_date,
                output_path=DATA_DIR / 'ibovespa_data.csv',
            )

    # Collect or generate Twitter data
    if args.twitter or args.all:
        print("\n" + "-" * 40)
        print("🐦 Twitter Data")
        print("-" * 40)

        bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')

        if bearer_token:
            twitter_df = collect_twitter_api(
                queries=TWITTER_SEARCH_QUERIES,
                start_date=args.start_date,
                end_date=args.end_date,
                bearer_token=bearer_token,
                max_tweets_per_query=args.max_tweets,
                output_path=DATA_DIR / 'twitter_data.csv',
            )
        else:
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

    # Collect or generate Reddit data
    if args.reddit or args.all:
        print("\n" + "-" * 40)
        print("👽 Reddit Data")
        print("-" * 40)

        client_id = os.environ.get('REDDIT_CLIENT_ID')
        client_secret = os.environ.get('REDDIT_CLIENT_SECRET')

        reddit_df = collect_reddit_praw(
            subreddits=REDDIT_SUBREDDITS,
            start_date=args.start_date,
            end_date=args.end_date,
            client_id=client_id,
            client_secret=client_secret,
            max_posts_per_subreddit=args.max_posts,
            output_path=DATA_DIR / 'reddit_data.csv',
        )

    elif args.synthetic:
        print("\n" + "-" * 40)
        print("👽 Reddit Data (Synthetic)")
        print("-" * 40)

        reddit_df = generate_synthetic_reddit(
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=DATA_DIR / 'reddit_data.csv',
        )

    # Summary
    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)
    print(f"\nFiles saved to: {DATA_DIR}/")
    print("  - twitter_data.csv")
    print("  - reddit_data.csv")
    print("  - ibovespa_data.csv")
    print("\nNext step: Run the FSI calculation script:")
    print("  python scripts/run_fsi.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
