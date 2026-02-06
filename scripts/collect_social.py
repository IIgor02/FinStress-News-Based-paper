#!/usr/bin/env python3
"""
Social Media Data Collection for Financial Stress Index
=========================================================
Collects posts from Twitter/X and Reddit for FSI analysis.

Twitter Sources (Brazilian financial news accounts):
- @CNNBrasil, @JornalOGlobo, @EstadaoEconomia, @valoreconomico, etc.

Reddit Sources:
- r/farialimabets (Brazilian financial memes/discussion)
- r/investimentos (Brazilian investments)

Requirements:
    pip install twikit praw asyncio

Usage:
    # Twitter only (requires credentials)
    python scripts/collect_social.py --twitter

    # Reddit only (requires API credentials)
    python scripts/collect_social.py --reddit

    # Both
    python scripts/collect_social.py --all

Configuration:
    Create a config file at data/social_config.json with credentials

Output:
    data/raw/twitter_posts.csv
    data/raw/reddit_posts.csv
    data/raw/social_combined.csv
"""

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime, timedelta
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
TWITTER_ACCOUNTS = [
    'CNNBrasil',
    'JornalOGlobo',
    'estadaborges',  # Estadão Economia
    'valoreconomico',
    'folaborges',    # Folha Mercado
    'InfoaborageMoney',
    'exaborame',
    'BCB_aborr',     # Banco Central
    'B3_Oficial',    # B3 Stock Exchange
]

# Reddit subreddits to follow
REDDIT_SUBREDDITS = [
    'farialimabets',
    'investimentos',
    'brasil',
]

# Keywords to filter relevant posts
FINANCIAL_KEYWORDS = [
    'ibovespa', 'bolsa', 'ação', 'ações', 'mercado',
    'dólar', 'dolar', 'real', 'economia', 'crise',
    'inflação', 'selic', 'juros', 'bc', 'banco central',
    'recessão', 'pib', 'desemprego', 'falência',
    'queda', 'alta', 'bull', 'bear', 'crash',
]

REQUEST_DELAY = 2


# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

def load_config() -> Dict:
    """Load social media credentials from config file."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_config(config: Dict):
    """Save config to file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def create_sample_config():
    """Create sample config file."""
    sample = {
        "twitter": {
            "username": "your_twitter_username",
            "email": "your_email@example.com",
            "password": "your_password",
            "_note": "Use a secondary account, not your personal one"
        },
        "reddit": {
            "client_id": "your_reddit_app_client_id",
            "client_secret": "your_reddit_app_secret",
            "user_agent": "FSI-Scraper/1.0",
            "_note": "Create app at https://www.reddit.com/prefs/apps"
        }
    }

    if not CONFIG_FILE.exists():
        save_config(sample)
        print(f"  Created sample config at {CONFIG_FILE}")
        print("  Please edit with your credentials")

    return sample


# =============================================================================
# TWITTER SCRAPER (using twikit)
# =============================================================================

async def scrape_twitter_async(config: Dict, accounts: List[str],
                                start_year: int, max_tweets: int = 1000) -> List[Dict]:
    """
    Scrape tweets from specified accounts using twikit.

    Args:
        config: Twitter credentials (username, email, password)
        accounts: List of Twitter usernames to scrape
        start_year: Only get tweets from this year onwards
        max_tweets: Maximum tweets per account

    Returns:
        List of tweet dictionaries
    """
    try:
        from twikit import Client
    except ImportError:
        print("  ERROR: twikit not installed. Run: pip install twikit")
        return []

    tweets_data = []

    try:
        # Initialize client
        client = Client('en-US')

        # Check for existing cookies
        cookies_file = DATA_DIR / 'twitter_cookies.json'

        if cookies_file.exists():
            print("  Loading existing Twitter session...")
            client.load_cookies(str(cookies_file))
        else:
            print("  Logging into Twitter...")
            await client.login(
                auth_info_1=config.get('username', ''),
                auth_info_2=config.get('email', ''),
                password=config.get('password', ''),
            )
            client.save_cookies(str(cookies_file))
            print("  Login successful, cookies saved")

        # Scrape each account
        for account in accounts:
            print(f"    Scraping @{account}...", end=" ", flush=True)

            try:
                # Get user
                user = await client.get_user_by_screen_name(account)
                if not user:
                    print("not found")
                    continue

                # Get tweets
                user_tweets = await client.get_user_tweets(user.id, 'Tweets', count=max_tweets)

                account_tweets = 0
                for tweet in user_tweets:
                    try:
                        # Parse date
                        tweet_date = tweet.created_at
                        if hasattr(tweet_date, 'year'):
                            if tweet_date.year < start_year:
                                continue
                            date_str = tweet_date.strftime('%Y-%m-%d')
                        else:
                            date_str = str(tweet_date)[:10]

                        tweets_data.append({
                            'text': tweet.text,
                            'date': date_str,
                            'user': account,
                            'likes': getattr(tweet, 'favorite_count', 0),
                            'retweets': getattr(tweet, 'retweet_count', 0),
                            'url': f"https://twitter.com/{account}/status/{tweet.id}",
                            'source': 'twitter',
                        })
                        account_tweets += 1

                    except Exception:
                        continue

                print(f"{account_tweets} tweets")

            except Exception as e:
                print(f"error: {e}")

            await asyncio.sleep(REQUEST_DELAY)

    except Exception as e:
        print(f"  Twitter error: {e}")

    return tweets_data


def scrape_twitter(config: Dict, accounts: List[str],
                   start_year: int = 2011, max_tweets: int = 1000) -> pd.DataFrame:
    """Wrapper for async Twitter scraper."""
    print("\n  Scraping Twitter...")

    if not config.get('username') or config.get('username') == 'your_twitter_username':
        print("  ERROR: Twitter credentials not configured")
        print(f"  Edit {CONFIG_FILE} with your credentials")
        return pd.DataFrame()

    tweets = asyncio.run(scrape_twitter_async(config, accounts, start_year, max_tweets))

    if not tweets:
        return pd.DataFrame()

    df = pd.DataFrame(tweets)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date', ascending=True)

    return df


# =============================================================================
# REDDIT SCRAPER (using PRAW)
# =============================================================================

def scrape_reddit_praw(config: Dict, subreddits: List[str],
                        start_year: int = 2011, max_posts: int = 1000) -> pd.DataFrame:
    """
    Scrape Reddit posts using PRAW (Python Reddit API Wrapper).

    Args:
        config: Reddit API credentials
        subreddits: List of subreddit names
        start_year: Only get posts from this year onwards
        max_posts: Maximum posts per subreddit

    Returns:
        DataFrame with posts
    """
    try:
        import praw
    except ImportError:
        print("  ERROR: praw not installed. Run: pip install praw")
        return pd.DataFrame()

    print("\n  Scraping Reddit with PRAW...")

    if not config.get('client_id') or config.get('client_id') == 'your_reddit_app_client_id':
        print("  ERROR: Reddit API credentials not configured")
        print(f"  1. Go to https://www.reddit.com/prefs/apps")
        print("  2. Create a 'script' type app")
        print(f"  3. Edit {CONFIG_FILE} with client_id and client_secret")
        return pd.DataFrame()

    posts_data = []

    try:
        reddit = praw.Reddit(
            client_id=config['client_id'],
            client_secret=config['client_secret'],
            user_agent=config.get('user_agent', 'FSI-Scraper/1.0'),
        )

        start_timestamp = datetime(start_year, 1, 1).timestamp()

        for subreddit_name in subreddits:
            print(f"    Scraping r/{subreddit_name}...", end=" ", flush=True)

            try:
                subreddit = reddit.subreddit(subreddit_name)

                # Get posts from different time periods
                post_count = 0
                for time_filter in ['all', 'year', 'month']:
                    for post in subreddit.top(time_filter=time_filter, limit=max_posts // 3):
                        try:
                            if post.created_utc < start_timestamp:
                                continue

                            post_date = datetime.fromtimestamp(post.created_utc)

                            posts_data.append({
                                'text': f"{post.title} {post.selftext}",
                                'title': post.title,
                                'date': post_date.strftime('%Y-%m-%d'),
                                'user': str(post.author),
                                'score': post.score,
                                'comments': post.num_comments,
                                'url': f"https://reddit.com{post.permalink}",
                                'subreddit': subreddit_name,
                                'source': 'reddit',
                            })
                            post_count += 1

                        except Exception:
                            continue

                print(f"{post_count} posts")

            except Exception as e:
                print(f"error: {e}")

            time.sleep(REQUEST_DELAY)

    except Exception as e:
        print(f"  Reddit error: {e}")

    if not posts_data:
        return pd.DataFrame()

    df = pd.DataFrame(posts_data)
    df = df.drop_duplicates(subset=['url'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date', ascending=True)

    return df


# =============================================================================
# REDDIT SCRAPER (direct, no API needed)
# =============================================================================

def scrape_reddit_direct(subreddits: List[str], start_year: int = 2011,
                          max_posts: int = 500) -> pd.DataFrame:
    """
    Scrape Reddit directly using old.reddit.com (no API key needed).
    Limited but works without credentials.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("  ERROR: requests/beautifulsoup not installed")
        return pd.DataFrame()

    print("\n  Scraping Reddit directly (limited)...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }

    posts_data = []
    session = requests.Session()
    session.headers.update(headers)

    for subreddit_name in subreddits:
        print(f"    Scraping r/{subreddit_name}...", end=" ", flush=True)

        try:
            # Use old.reddit.com for easier scraping
            url = f"https://old.reddit.com/r/{subreddit_name}/top/?t=all"

            response = session.get(url, timeout=30)
            if response.status_code != 200:
                print(f"error {response.status_code}")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find posts
            posts = soup.select('div.thing.link')
            post_count = 0

            for post in posts[:max_posts]:
                try:
                    # Title
                    title_elem = post.select_one('a.title')
                    if not title_elem:
                        continue
                    title = title_elem.get_text(strip=True)

                    # URL
                    permalink = post.get('data-permalink', '')
                    if permalink:
                        post_url = f"https://reddit.com{permalink}"
                    else:
                        post_url = title_elem.get('href', '')

                    # Score
                    score_elem = post.select_one('div.score.unvoted')
                    score = score_elem.get('title', '0') if score_elem else '0'
                    try:
                        score = int(score)
                    except:
                        score = 0

                    # Date (approximate from data-timestamp or use current)
                    timestamp = post.get('data-timestamp', '')
                    if timestamp:
                        try:
                            post_date = datetime.fromtimestamp(int(timestamp) / 1000)
                            date_str = post_date.strftime('%Y-%m-%d')
                        except:
                            date_str = datetime.now().strftime('%Y-%m-%d')
                    else:
                        date_str = datetime.now().strftime('%Y-%m-%d')

                    # Author
                    author_elem = post.select_one('a.author')
                    author = author_elem.get_text(strip=True) if author_elem else 'unknown'

                    posts_data.append({
                        'text': title,
                        'title': title,
                        'date': date_str,
                        'user': author,
                        'score': score,
                        'url': post_url,
                        'subreddit': subreddit_name,
                        'source': 'reddit',
                    })
                    post_count += 1

                except Exception:
                    continue

            print(f"{post_count} posts")

        except Exception as e:
            print(f"error: {e}")

        time.sleep(REQUEST_DELAY)

    if not posts_data:
        return pd.DataFrame()

    df = pd.DataFrame(posts_data)
    df = df.drop_duplicates(subset=['url'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date', ascending=True)

    return df


# =============================================================================
# FILTERING AND PROCESSING
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
    print(f"  Filtered to {len(filtered)} financially relevant posts")

    return filtered


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect social media data for Financial Stress Index',
    )
    parser.add_argument('--twitter', action='store_true', help='Scrape Twitter')
    parser.add_argument('--reddit', action='store_true', help='Scrape Reddit')
    parser.add_argument('--all', action='store_true', help='Scrape all sources')
    parser.add_argument('--start-year', type=int, default=2011, help='Start year')
    parser.add_argument('--no-filter', action='store_true', help='Skip financial keyword filtering')
    parser.add_argument('--setup', action='store_true', help='Create sample config file')

    args = parser.parse_args()

    print("=" * 60)
    print("SOCIAL MEDIA DATA COLLECTION")
    print("=" * 60)

    # Setup mode
    if args.setup:
        create_sample_config()
        return

    # Load config
    config = load_config()
    if not config:
        print(f"\n  Config file not found at {CONFIG_FILE}")
        print("  Run: python scripts/collect_social.py --setup")
        create_sample_config()
        return

    # Determine what to scrape
    scrape_twitter = args.twitter or args.all
    scrape_reddit = args.reddit or args.all

    if not scrape_twitter and not scrape_reddit:
        print("\n  No source specified. Use --twitter, --reddit, or --all")
        return

    all_data = []

    # Twitter
    if scrape_twitter:
        print("\n" + "-" * 40)
        print("TWITTER")
        print("-" * 40)
        twitter_config = config.get('twitter', {})
        twitter_df = scrape_twitter(twitter_config, TWITTER_ACCOUNTS, args.start_year)

        if not twitter_df.empty:
            twitter_path = DATA_DIR / 'twitter_posts.csv'
            twitter_df.to_csv(twitter_path, index=False)
            print(f"  Saved {len(twitter_df)} tweets to {twitter_path}")
            all_data.append(twitter_df)

    # Reddit
    if scrape_reddit:
        print("\n" + "-" * 40)
        print("REDDIT")
        print("-" * 40)
        reddit_config = config.get('reddit', {})

        # Try PRAW first, fall back to direct scraping
        if reddit_config.get('client_id') and reddit_config.get('client_id') != 'your_reddit_app_client_id':
            reddit_df = scrape_reddit_praw(reddit_config, REDDIT_SUBREDDITS, args.start_year)
        else:
            print("  Reddit API not configured, using direct scraping (limited)")
            reddit_df = scrape_reddit_direct(REDDIT_SUBREDDITS, args.start_year)

        if not reddit_df.empty:
            reddit_path = DATA_DIR / 'reddit_posts.csv'
            reddit_df.to_csv(reddit_path, index=False)
            print(f"  Saved {len(reddit_df)} posts to {reddit_path}")
            all_data.append(reddit_df)

    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # Filter for financial content
        if not args.no_filter:
            print("\n" + "-" * 40)
            print("FILTERING")
            print("-" * 40)
            combined_df = filter_financial_posts(combined_df)

        # Save combined
        combined_path = DATA_DIR / 'social_combined.csv'
        combined_df.to_csv(combined_path, index=False)

        # Summary
        print("\n" + "=" * 60)
        print("SOCIAL MEDIA COLLECTION COMPLETE")
        print("=" * 60)
        print(f"\nTotal posts: {len(combined_df)}")

        if not combined_df.empty:
            print(f"Date range: {combined_df['date'].min().date()} to {combined_df['date'].max().date()}")

            print(f"\nBy source:")
            for source, count in combined_df['source'].value_counts().items():
                print(f"  - {source}: {count}")

            if 'subreddit' in combined_df.columns:
                print(f"\nBy subreddit:")
                reddit_only = combined_df[combined_df['source'] == 'reddit']
                for sub, count in reddit_only['subreddit'].value_counts().items():
                    print(f"  - r/{sub}: {count}")

        print(f"\nOutput files:")
        print(f"  - {combined_path}")
        print(f"\nNext: python scripts/social_fsi.py")
    else:
        print("\n  No data collected")

    print("=" * 60)


if __name__ == '__main__':
    main()
