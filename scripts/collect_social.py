#!/usr/bin/env python3
"""
Twitter Data Collection for Financial Stress Index
====================================================
Collects tweets from Brazilian financial news accounts using twikit.

Target accounts (Brazilian financial news):
- @CNNBrasil, @JornalOGlobo, @valoreconomico, @estadaborges, etc.

Requirements:
    pip install twikit

Usage:
    python scripts/collect_social.py
    python scripts/collect_social.py --max-tweets 2000

Output:
    data/raw/twitter_posts.csv
    data/raw/social_combined.csv
"""

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd

try:
    _SCRIPT_DIR = Path(__file__).parent
    _PROJECT_DIR = _SCRIPT_DIR.parent
except NameError:
    _PROJECT_DIR = Path.cwd()
    _SCRIPT_DIR = _PROJECT_DIR / 'scripts'

sys.path.insert(0, str(_PROJECT_DIR))

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = _PROJECT_DIR / 'data' / 'raw'
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = _PROJECT_DIR / 'data' / 'social_config.json'

# Twitter accounts to follow (Brazilian financial news)
# Expanded list for maximum coverage
TWITTER_ACCOUNTS = [
    # Major news outlets
    'CNNBrasil',
    'JornalOGlobo',
    'Aboragesil',          # O Globo economia
    'valoreconomico',
    'EstadaoEconomia',
    'folaborahalves',      # Folha related
    'UaborlEconomia',

    # Financial specialized
    'InfoMoney',
    'exaborae',
    'BloombergLinea',
    'maboroneycnn',        # CNN economia

    # Institutions
    'BancoCentralBR',      # Banco Central do Brasil
    'B3_Oficial',          # B3 Stock Exchange
    'Aborrafin',           # Abrafin
    'feaboraban',          # Febraban

    # Financial journalists/analysts
    'juliaaborarlos',
    'ThiaborgoResende',

    # Business
    'PipelineValor',
    'NeoFeed',
]

# Keywords to filter relevant posts
FINANCIAL_KEYWORDS = [
    'ibovespa', 'bolsa', 'ação', 'ações', 'mercado',
    'dólar', 'dolar', 'real', 'economia', 'crise',
    'inflação', 'selic', 'juros', 'banco central',
    'recessão', 'pib', 'desemprego', 'bc',
    'queda', 'alta', 'bull', 'bear',
    'b3', 'bovespa', 'tesouro', 'cdi',
]

REQUEST_DELAY = 3  # Seconds between requests


# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

def load_config() -> Dict:
    """Load Twitter credentials from config file."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


# =============================================================================
# TWITTER SCRAPER (using twikit)
# =============================================================================

async def scrape_twitter_user_tweets(client, username: str, max_tweets: int = 500) -> List[Dict]:
    """Get tweets from a specific user."""
    tweets_data = []

    try:
        # Get user
        user = await client.get_user_by_screen_name(username)
        if not user:
            return []

        # Get tweets - twikit returns tweets in batches
        cursor = None
        tweets_fetched = 0

        while tweets_fetched < max_tweets:
            try:
                if cursor:
                    user_tweets = await client.get_user_tweets(user.id, 'Tweets', cursor=cursor)
                else:
                    user_tweets = await client.get_user_tweets(user.id, 'Tweets')

                if not user_tweets:
                    break

                for tweet in user_tweets:
                    try:
                        # Get tweet date
                        tweet_date = tweet.created_at
                        if hasattr(tweet_date, 'strftime'):
                            date_str = tweet_date.strftime('%Y-%m-%d')
                            year = tweet_date.year
                        else:
                            # Parse string date
                            date_str = str(tweet_date)[:10]
                            year = int(date_str[:4]) if len(date_str) >= 4 else 2020

                        # Skip if before 2011
                        if year < 2011:
                            continue

                        tweets_data.append({
                            'text': tweet.text,
                            'date': date_str,
                            'user': username,
                            'likes': getattr(tweet, 'favorite_count', 0) or 0,
                            'retweets': getattr(tweet, 'retweet_count', 0) or 0,
                            'tweet_id': str(tweet.id),
                            'url': f"https://twitter.com/{username}/status/{tweet.id}",
                            'source': 'twitter',
                        })
                        tweets_fetched += 1

                    except Exception:
                        continue

                # Get next cursor for pagination
                if hasattr(user_tweets, 'cursor'):
                    cursor = user_tweets.cursor
                elif hasattr(user_tweets, 'next_cursor'):
                    cursor = user_tweets.next_cursor
                else:
                    # Try to get cursor from last tweet
                    cursor = None
                    break

                if not cursor:
                    break

                await asyncio.sleep(REQUEST_DELAY)

            except Exception as e:
                if 'rate' in str(e).lower():
                    print(f"\n      Rate limited, waiting 60s...")
                    await asyncio.sleep(60)
                else:
                    break

    except Exception as e:
        print(f"error: {str(e)[:50]}")

    return tweets_data


async def search_twitter_keywords(client, keywords: List[str], max_tweets: int = 1000) -> List[Dict]:
    """Search Twitter for financial keywords."""
    tweets_data = []
    seen_ids = set()

    for keyword in keywords[:10]:  # Limit keywords
        try:
            print(f"      Searching: '{keyword}'...", end=" ", flush=True)

            # Search tweets
            search_results = await client.search_tweet(keyword, 'Latest', count=100)

            keyword_tweets = 0
            for tweet in search_results:
                try:
                    tweet_id = str(tweet.id)
                    if tweet_id in seen_ids:
                        continue
                    seen_ids.add(tweet_id)

                    tweet_date = tweet.created_at
                    if hasattr(tweet_date, 'strftime'):
                        date_str = tweet_date.strftime('%Y-%m-%d')
                    else:
                        date_str = str(tweet_date)[:10]

                    # Get username
                    username = getattr(tweet.user, 'screen_name', 'unknown') if hasattr(tweet, 'user') else 'unknown'

                    tweets_data.append({
                        'text': tweet.text,
                        'date': date_str,
                        'user': username,
                        'likes': getattr(tweet, 'favorite_count', 0) or 0,
                        'retweets': getattr(tweet, 'retweet_count', 0) or 0,
                        'tweet_id': tweet_id,
                        'url': f"https://twitter.com/{username}/status/{tweet_id}",
                        'source': 'twitter',
                        'keyword': keyword,
                    })
                    keyword_tweets += 1

                except Exception:
                    continue

            print(f"{keyword_tweets} tweets")
            await asyncio.sleep(REQUEST_DELAY)

            if len(tweets_data) >= max_tweets:
                break

        except Exception as e:
            if 'rate' in str(e).lower():
                print(f"rate limited, waiting...")
                await asyncio.sleep(60)
            else:
                print(f"error")

    return tweets_data


async def scrape_twitter_async(config: Dict, accounts: List[str],
                                max_tweets_per_account: int = 500,
                                search_keywords: bool = True) -> List[Dict]:
    """
    Main async function to scrape Twitter.
    """
    try:
        from twikit import Client
    except ImportError:
        print("  ERROR: twikit not installed. Run: pip install twikit")
        return []

    all_tweets = []

    try:
        # Initialize client
        client = Client('en-US')

        # Check for existing cookies
        cookies_file = DATA_DIR / 'twitter_cookies.json'

        if cookies_file.exists():
            print("  Loading existing Twitter session...")
            try:
                client.load_cookies(str(cookies_file))
                # Test if session is valid
                print("  Session loaded successfully")
            except Exception:
                print("  Session expired, logging in again...")
                cookies_file.unlink()

        if not cookies_file.exists():
            print("  Logging into Twitter...")
            print(f"    Username: {config.get('username', 'N/A')}")

            await client.login(
                auth_info_1=config.get('username', ''),
                auth_info_2=config.get('email', ''),
                password=config.get('password', ''),
            )
            client.save_cookies(str(cookies_file))
            print("  Login successful, cookies saved")

        # Scrape each account
        print(f"\n  Scraping {len(accounts)} Twitter accounts...")

        for account in accounts:
            print(f"    @{account}...", end=" ", flush=True)

            try:
                account_tweets = await scrape_twitter_user_tweets(client, account, max_tweets_per_account)
                all_tweets.extend(account_tweets)
                print(f"{len(account_tweets)} tweets")

            except Exception as e:
                error_msg = str(e)
                if 'not found' in error_msg.lower() or '404' in error_msg:
                    print("not found")
                elif 'rate' in error_msg.lower():
                    print("rate limited, waiting...")
                    await asyncio.sleep(60)
                else:
                    print(f"error: {error_msg[:30]}")

            await asyncio.sleep(REQUEST_DELAY)

        # Also search for keywords
        if search_keywords:
            print(f"\n  Searching Twitter for financial keywords...")
            keyword_tweets = await search_twitter_keywords(client, FINANCIAL_KEYWORDS, max_tweets=500)
            all_tweets.extend(keyword_tweets)

    except Exception as e:
        print(f"\n  Twitter error: {e}")
        import traceback
        traceback.print_exc()

    return all_tweets


def scrape_twitter(config: Dict, accounts: List[str],
                   max_tweets_per_account: int = 500) -> pd.DataFrame:
    """Wrapper for async Twitter scraper."""
    print("\n" + "=" * 50)
    print("TWITTER DATA COLLECTION")
    print("=" * 50)

    if not config.get('username'):
        print("  ERROR: Twitter credentials not configured")
        print(f"  Edit {CONFIG_FILE} with your credentials")
        return pd.DataFrame()

    # Run async scraper
    tweets = asyncio.run(scrape_twitter_async(config, accounts, max_tweets_per_account))

    if not tweets:
        print("\n  No tweets collected")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(tweets)

    # Remove duplicates
    if 'tweet_id' in df.columns:
        df = df.drop_duplicates(subset=['tweet_id'])
    elif 'url' in df.columns:
        df = df.drop_duplicates(subset=['url'])

    # Parse dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date', ascending=True)

    return df


# =============================================================================
# FILTERING
# =============================================================================

def filter_financial_posts(df: pd.DataFrame, keywords: List[str] = None) -> pd.DataFrame:
    """Filter posts to only include financially relevant content."""
    if df.empty:
        return df

    keywords = keywords or FINANCIAL_KEYWORDS

    # Create regex pattern
    pattern = '|'.join(keywords)

    # Filter by text content
    mask = df['text'].str.lower().str.contains(pattern, regex=True, na=False)

    filtered = df[mask].copy()
    print(f"\n  Filtered to {len(filtered)} financially relevant tweets")

    return filtered


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect Twitter data for Financial Stress Index',
    )
    parser.add_argument('--max-tweets', type=int, default=500,
                        help='Max tweets per account (default: 500)')
    parser.add_argument('--no-filter', action='store_true',
                        help='Skip financial keyword filtering')
    parser.add_argument('--accounts', type=str, default=None,
                        help='Comma-separated list of specific accounts')

    args = parser.parse_args()

    print("=" * 60)
    print("SOCIAL MEDIA DATA COLLECTION (Twitter)")
    print("=" * 60)

    # Load config
    config = load_config()
    twitter_config = config.get('twitter', {})

    if not twitter_config.get('username'):
        print(f"\n  ERROR: Config not found at {CONFIG_FILE}")
        print("  Create the file with your Twitter credentials")
        return

    # Determine accounts to scrape
    if args.accounts:
        accounts = [a.strip() for a in args.accounts.split(',')]
    else:
        accounts = TWITTER_ACCOUNTS

    print(f"\nAccounts to scrape: {len(accounts)}")
    print(f"Max tweets per account: {args.max_tweets}")

    # Scrape Twitter
    twitter_df = scrape_twitter(twitter_config, accounts, args.max_tweets)

    if twitter_df.empty:
        print("\n  ERROR: No data collected")
        return

    # Filter for financial content
    if not args.no_filter:
        print("\n" + "-" * 40)
        print("FILTERING")
        print("-" * 40)
        filtered_df = filter_financial_posts(twitter_df)
    else:
        filtered_df = twitter_df

    # Save
    print("\n" + "-" * 40)
    print("SAVING")
    print("-" * 40)

    # Save all tweets
    twitter_path = DATA_DIR / 'twitter_posts.csv'
    twitter_df.to_csv(twitter_path, index=False)
    print(f"  All tweets: {twitter_path} ({len(twitter_df)} tweets)")

    # Save filtered tweets
    combined_path = DATA_DIR / 'social_combined.csv'
    filtered_df.to_csv(combined_path, index=False)
    print(f"  Filtered: {combined_path} ({len(filtered_df)} tweets)")

    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"\nTotal tweets: {len(twitter_df)}")
    print(f"Financial-related: {len(filtered_df)}")

    if not twitter_df.empty:
        print(f"Date range: {twitter_df['date'].min().date()} to {twitter_df['date'].max().date()}")

        print(f"\nBy account:")
        account_counts = twitter_df['user'].value_counts().head(10)
        for account, count in account_counts.items():
            print(f"  @{account}: {count}")

        print(f"\nBy year:")
        twitter_df['year'] = twitter_df['date'].dt.year
        year_counts = twitter_df.groupby('year').size()
        for year in sorted(year_counts.index):
            print(f"  {year}: {year_counts[year]}")

    print(f"\nNext step: python scripts/social_fsi.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
