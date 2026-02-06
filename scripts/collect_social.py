#!/usr/bin/env python3
"""
Twitter Data Collection for Financial Stress Index
====================================================
Collects tweets from Brazilian financial news accounts using Selenium.

Uses Selenium with a real browser to bypass Cloudflare protection.

Target accounts (Brazilian financial news):
- @CNNBrasil, @JornalOGlobo, @valoreconomico, etc.

Requirements:
    pip install selenium webdriver-manager

Usage:
    python scripts/collect_social.py
    python scripts/collect_social.py --accounts CNNBrasil,valoreconomico

Output:
    data/raw/twitter_posts.csv
    data/raw/social_combined.csv
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
from urllib.parse import quote

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
# Verified real accounts
TWITTER_ACCOUNTS = [
    # Major news outlets
    'CNNBrasil',
    'JornalOGlobo',
    'valoreconomico',
    'EstadaoEconomia',
    'folaborahaes',
    'UOLEconomia',

    # Financial specialized
    'InfoMoney',
    'exaborae',
    'BloombergLinea',

    # Institutions
    'BancoCentralBR',      # Banco Central do Brasil
    'B3_Oficial',          # B3 Stock Exchange
    'Aborrafin',
    'febraboraan',

    # Business
    'PipelineValor',
    'NeoFeed',
    'maboroneycnn',
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

# Search queries for Twitter (Portuguese financial terms)
SEARCH_QUERIES = [
    'ibovespa',
    'bolsa brasil',
    'dólar real',
    'selic taxa',
    'economia brasileira',
    'crise econômica brasil',
    'mercado financeiro',
    'banco central brasil',
    'inflação brasil',
]

REQUEST_DELAY = 2  # Seconds between requests


# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

def load_config() -> Dict:
    """Load Twitter credentials from config file."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# =============================================================================
# SELENIUM-BASED TWITTER SCRAPER
# =============================================================================

def setup_selenium_driver():
    """Set up Selenium WebDriver with Chrome."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        print("  ERROR: selenium or webdriver-manager not installed")
        print("  Run: pip install selenium webdriver-manager")
        return None

    options = Options()
    # Run in headless mode for automation
    # options.add_argument('--headless')  # Comment out for debugging
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    # Disable automation flags
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        # Remove webdriver flag
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        return driver
    except Exception as e:
        print(f"  ERROR setting up Chrome driver: {e}")
        return None


def twitter_login(driver, username: str, email: str, password: str) -> bool:
    """Log into Twitter using Selenium."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    try:
        print("  Opening Twitter login page...")
        driver.get('https://twitter.com/i/flow/login')
        time.sleep(3)

        # Wait for username field
        wait = WebDriverWait(driver, 20)

        # Enter username
        print(f"  Entering username: {username}")
        username_input = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, 'input[autocomplete="username"]')
        ))
        username_input.send_keys(username)
        username_input.send_keys(Keys.RETURN)
        time.sleep(2)

        # Check for email verification step
        try:
            email_input = WebDriverWait(driver, 5).until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'input[data-testid="ocfEnterTextTextInput"]')
            ))
            print(f"  Entering email verification: {email}")
            email_input.send_keys(email)
            email_input.send_keys(Keys.RETURN)
            time.sleep(2)
        except:
            pass  # No email verification needed

        # Enter password
        print("  Entering password...")
        password_input = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, 'input[name="password"]')
        ))
        password_input.send_keys(password)
        password_input.send_keys(Keys.RETURN)
        time.sleep(5)

        # Check if login was successful
        if 'home' in driver.current_url or 'twitter.com' in driver.current_url:
            print("  Login successful!")
            return True
        else:
            print(f"  Login may have failed. Current URL: {driver.current_url}")
            return False

    except Exception as e:
        print(f"  Login error: {e}")
        return False


def scrape_user_tweets_selenium(driver, username: str, max_tweets: int = 20000) -> List[Dict]:
    """Scrape tweets from a user's profile using Selenium."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    tweets_data = []
    seen_ids = set()

    try:
        # Go to user's profile
        url = f'https://twitter.com/{username}'
        driver.get(url)
        time.sleep(3)

        # Check if user exists
        if "This account doesn" in driver.page_source or "doesn't exist" in driver.page_source:
            return []

        # Scroll and collect tweets
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0
        max_scroll_attempts = 100  # Allow more scrolling

        while len(tweets_data) < max_tweets and scroll_attempts < max_scroll_attempts:
            # Find tweet articles
            try:
                articles = driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')

                for article in articles:
                    try:
                        # Get tweet text
                        text_elem = article.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]')
                        text = text_elem.text if text_elem else ""

                        if not text:
                            continue

                        # Get tweet link (contains ID)
                        links = article.find_elements(By.CSS_SELECTOR, 'a[href*="/status/"]')
                        tweet_url = ""
                        tweet_id = ""
                        for link in links:
                            href = link.get_attribute('href')
                            if '/status/' in href:
                                tweet_url = href
                                match = re.search(r'/status/(\d+)', href)
                                if match:
                                    tweet_id = match.group(1)
                                break

                        if not tweet_id or tweet_id in seen_ids:
                            continue
                        seen_ids.add(tweet_id)

                        # Get time element
                        time_elem = article.find_element(By.CSS_SELECTOR, 'time')
                        date_str = time_elem.get_attribute('datetime')[:10] if time_elem else ""

                        if not date_str:
                            continue

                        # Get engagement metrics
                        likes = 0
                        retweets = 0
                        try:
                            metrics = article.find_elements(By.CSS_SELECTOR, 'div[data-testid="like"] span, div[data-testid="retweet"] span')
                            for m in metrics:
                                val = m.text.replace(',', '').replace('K', '000').replace('M', '000000')
                                if val.isdigit():
                                    if 'like' in m.find_element(By.XPATH, '..').get_attribute('data-testid'):
                                        likes = int(val)
                                    else:
                                        retweets = int(val)
                        except:
                            pass

                        tweets_data.append({
                            'text': text,
                            'date': date_str,
                            'user': username,
                            'likes': likes,
                            'retweets': retweets,
                            'tweet_id': tweet_id,
                            'url': tweet_url,
                            'source': 'twitter',
                        })

                    except Exception:
                        continue

            except Exception:
                pass

            # Scroll down
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(REQUEST_DELAY)

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                scroll_attempts += 1
                if scroll_attempts > 5:
                    break
            else:
                scroll_attempts = 0
            last_height = new_height

    except Exception as e:
        print(f"error: {str(e)[:40]}")

    return tweets_data


def search_twitter_selenium(driver, query: str, max_tweets: int = 500) -> List[Dict]:
    """Search Twitter for a query using Selenium."""
    from selenium.webdriver.common.by import By

    tweets_data = []
    seen_ids = set()

    try:
        # Go to search page
        encoded_query = quote(query)
        url = f'https://twitter.com/search?q={encoded_query}&src=typed_query&f=live'
        driver.get(url)
        time.sleep(3)

        # Scroll and collect tweets
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0

        while len(tweets_data) < max_tweets and scroll_attempts < 20:
            try:
                articles = driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')

                for article in articles:
                    try:
                        # Get tweet text
                        text_elem = article.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]')
                        text = text_elem.text if text_elem else ""

                        if not text:
                            continue

                        # Get username
                        user_links = article.find_elements(By.CSS_SELECTOR, 'a[href^="/"]')
                        username = "unknown"
                        for link in user_links:
                            href = link.get_attribute('href')
                            if href and '/' in href and '/status/' not in href:
                                username = href.split('/')[-1]
                                if username and not username.startswith('search'):
                                    break

                        # Get tweet link
                        links = article.find_elements(By.CSS_SELECTOR, 'a[href*="/status/"]')
                        tweet_url = ""
                        tweet_id = ""
                        for link in links:
                            href = link.get_attribute('href')
                            if '/status/' in href:
                                tweet_url = href
                                match = re.search(r'/status/(\d+)', href)
                                if match:
                                    tweet_id = match.group(1)
                                break

                        if not tweet_id or tweet_id in seen_ids:
                            continue
                        seen_ids.add(tweet_id)

                        # Get time
                        time_elem = article.find_element(By.CSS_SELECTOR, 'time')
                        date_str = time_elem.get_attribute('datetime')[:10] if time_elem else ""

                        if not date_str:
                            continue

                        tweets_data.append({
                            'text': text,
                            'date': date_str,
                            'user': username,
                            'likes': 0,
                            'retweets': 0,
                            'tweet_id': tweet_id,
                            'url': tweet_url,
                            'source': 'twitter',
                            'keyword': query,
                        })

                    except Exception:
                        continue

            except Exception:
                pass

            # Scroll
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(REQUEST_DELAY)

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                scroll_attempts += 1
            else:
                scroll_attempts = 0
            last_height = new_height

    except Exception as e:
        print(f"error: {str(e)[:40]}")

    return tweets_data


def scrape_twitter_selenium(config: Dict, accounts: List[str],
                            search_keywords: bool = True) -> pd.DataFrame:
    """Main function to scrape Twitter using Selenium."""
    print("\n" + "=" * 50)
    print("TWITTER DATA COLLECTION (Selenium)")
    print("=" * 50)

    # Set up driver
    print("\n  Setting up Chrome browser...")
    driver = setup_selenium_driver()

    if not driver:
        return pd.DataFrame()

    all_tweets = []

    try:
        # Login
        username = config.get('username', '')
        email = config.get('email', '')
        password = config.get('password', '')

        if username and password:
            login_success = twitter_login(driver, username, email, password)
            if not login_success:
                print("  WARNING: Login may have failed, continuing anyway...")
        else:
            print("  WARNING: No credentials provided, scraping public data only")

        # Scrape each account
        print(f"\n  Scraping {len(accounts)} Twitter accounts...")

        for i, account in enumerate(accounts):
            print(f"    [{i+1}/{len(accounts)}] @{account}...", end=" ", flush=True)

            try:
                account_tweets = scrape_user_tweets_selenium(driver, account)
                all_tweets.extend(account_tweets)
                print(f"{len(account_tweets)} tweets")

            except Exception as e:
                print(f"error: {str(e)[:30]}")

            time.sleep(REQUEST_DELAY)

        # Search for keywords
        if search_keywords:
            print(f"\n  Searching Twitter for financial keywords...")
            for query in SEARCH_QUERIES[:5]:  # Limit searches
                print(f"    Searching: '{query}'...", end=" ", flush=True)
                try:
                    search_tweets = search_twitter_selenium(driver, query, max_tweets=200)
                    all_tweets.extend(search_tweets)
                    print(f"{len(search_tweets)} tweets")
                except Exception as e:
                    print(f"error")
                time.sleep(REQUEST_DELAY)

    finally:
        driver.quit()

    if not all_tweets:
        print("\n  No tweets collected")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(all_tweets)

    # Remove duplicates
    if 'tweet_id' in df.columns:
        df = df.drop_duplicates(subset=['tweet_id'])

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
    parser.add_argument('--no-filter', action='store_true',
                        help='Skip financial keyword filtering')
    parser.add_argument('--accounts', type=str, default=None,
                        help='Comma-separated list of specific accounts')
    parser.add_argument('--no-search', action='store_true',
                        help='Skip keyword searches')

    args = parser.parse_args()

    print("=" * 60)
    print("SOCIAL MEDIA DATA COLLECTION (Twitter)")
    print("=" * 60)

    # Load config
    config = load_config()
    twitter_config = config.get('twitter', {})

    if not twitter_config.get('username'):
        print(f"\n  WARNING: No credentials at {CONFIG_FILE}")
        print("  Will attempt to scrape public data only")
        twitter_config = {}

    # Determine accounts to scrape
    if args.accounts:
        accounts = [a.strip() for a in args.accounts.split(',')]
    else:
        accounts = TWITTER_ACCOUNTS

    print(f"\nAccounts to scrape: {len(accounts)}")

    # Scrape Twitter
    twitter_df = scrape_twitter_selenium(
        twitter_config,
        accounts,
        search_keywords=not args.no_search
    )

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
