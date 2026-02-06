#!/usr/bin/env python3
"""
Historical News Collection Script for Brazilian Financial News
================================================================
Scrapes historical news from Folha de São Paulo (Mercado section).

The Folha search archive works with requests/BeautifulSoup.
G1 and Valor search pages use JavaScript rendering, so we use their
main pages and RSS feeds instead.

Based on:
- Folha: https://github.com/ruanrf/webscraper-folha

Usage:
    python scripts/collect_news.py
    python scripts/collect_news.py --start-year 2015 --end-year 2024
    python scripts/collect_news.py --keywords "crise,recessão"

Output:
    data/raw/news_combined.csv
"""

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from urllib.parse import quote, urljoin

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

REQUEST_DELAY = 2
MAX_RETRIES = 3

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
}

# Financial/economic keywords for search
DEFAULT_KEYWORDS = [
    'crise financeira brasil',
    'ibovespa queda',
    'bolsa valores brasil',
    'recessão economia',
    'inflação alta',
    'dólar sobe',
    'banco central juros',
    'mercado financeiro crise',
]

MAX_ARTICLES_PER_KEYWORD = 200


# =============================================================================
# HTTP HELPERS
# =============================================================================

def get_session():
    """Get requests session."""
    try:
        import requests
        session = requests.Session()
        session.headers.update(HEADERS)
        return session
    except ImportError:
        print("ERROR: requests not installed. Run: pip install requests")
        return None


def fetch_url(session, url: str) -> Optional[str]:
    """Fetch URL with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 429:  # Rate limited
                print(f"    Rate limited, waiting...")
                time.sleep(10)
            else:
                return None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(REQUEST_DELAY * (attempt + 1))
    return None


def get_soup(html: str):
    """Parse HTML."""
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, 'html.parser')
    except ImportError:
        print("ERROR: BeautifulSoup not installed. Run: pip install beautifulsoup4")
        return None


# =============================================================================
# FOLHA DE SÃO PAULO - SEARCH ARCHIVE (WORKS!)
# =============================================================================

def scrape_folha_search(session, keyword: str, start_year: int, end_year: int) -> List[Dict]:
    """
    Scrape Folha using their search archive.

    URL format (from working repository):
    https://search.folha.uol.com.br/search?q=QUERY&periodo=personalizado&sd=DD%2FMM%2FYYYY&ed=DD%2FMM%2FYYYY&site=todos&sr=OFFSET

    The search returns links to articles. We look for <a> tags with href containing folha.uol.com.br
    """
    articles = []

    # URL-encode the dates (DD/MM/YYYY -> DD%2FMM%2FYYYY)
    start_date = f"01%2F01%2F{start_year}"
    end_date = f"31%2F12%2F{end_year}"

    print(f"      Searching: '{keyword}'")

    for page in range(0, 10):  # Max 10 pages per keyword
        offset = page * 25 + 1

        url = (
            f"https://search.folha.uol.com.br/search?"
            f"q={quote(keyword)}&periodo=personalizado"
            f"&sd={start_date}&ed={end_date}&site=todos&sr={offset}"
        )

        html = fetch_url(session, url)
        if not html:
            break

        soup = get_soup(html)
        if not soup:
            break

        # Find all links - the working code looks for <a> tags with href patterns
        # Folha articles have URLs like: folha.uol.com.br/mercado/YYYY/MM/...
        links = soup.find_all('a', href=True)

        found_articles = 0
        for link in links:
            href = link.get('href', '')

            # Filter for Folha article URLs (contain date pattern in path)
            if 'folha.uol.com.br' not in href:
                continue

            # Must have date pattern like /2020/01/ or /2019/12/
            date_match = re.search(r'/(\d{4})/(\d{2})/', href)
            if not date_match:
                continue

            # Skip non-article pages
            if '/autor/' in href or '/colunistas/' in href or '/sobre/' in href:
                continue

            # Get title from link text
            title = link.get_text(strip=True)
            if not title or len(title) < 20:
                # Try to find title in parent elements
                parent = link.find_parent(['h2', 'h3', 'div'])
                if parent:
                    title = parent.get_text(strip=True)

            if not title or len(title) < 20:
                continue

            # Extract date from URL
            year, month = date_match.group(1), date_match.group(2)
            date_str = f"{year}-{month}-15"  # Approximate to mid-month

            articles.append({
                'title': title[:500],
                'url': href,
                'date': date_str,
                'source': 'folha',
                'keyword': keyword,
            })
            found_articles += 1

        if found_articles == 0:
            break  # No more results

        time.sleep(REQUEST_DELAY)

        if len(articles) >= MAX_ARTICLES_PER_KEYWORD:
            break

    return articles


def scrape_folha_archive(session, start_year: int, end_year: int,
                          keywords: List[str]) -> List[Dict]:
    """Scrape Folha for all keywords."""
    all_articles = []

    print(f"\n  Scraping Folha de S. Paulo ({start_year}-{end_year})...")

    for keyword in keywords:
        articles = scrape_folha_search(session, keyword, start_year, end_year)
        all_articles.extend(articles)
        print(f"        Found {len(articles)} articles")
        time.sleep(REQUEST_DELAY)

    # Remove duplicates
    seen = set()
    unique = []
    for a in all_articles:
        if a['url'] not in seen:
            seen.add(a['url'])
            unique.append(a)

    print(f"    Total: {len(unique)} unique articles from Folha")
    return unique


# =============================================================================
# G1 - DIRECT PAGE SCRAPING (search uses JavaScript)
# =============================================================================

def scrape_g1_main(session) -> List[Dict]:
    """
    Scrape G1 latest news (search pages use JavaScript).

    Based on: https://github.com/leviobrabo/G1-news-scraping
    Uses div.bastian-feed-item and a.feed-post-link selectors.
    """
    articles = []

    urls = [
        'https://g1.globo.com/economia/',
        'https://g1.globo.com/economia/noticia/',
    ]

    print(f"\n  Scraping G1 Economia (latest news)...")

    for url in urls:
        html = fetch_url(session, url)
        if not html:
            continue

        soup = get_soup(html)
        if not soup:
            continue

        # Find article containers (from working repository)
        items = soup.select('div.bastian-feed-item')
        if not items:
            items = soup.select('article, div.feed-post')

        for item in items:
            try:
                # Find title link
                link = item.select_one('a.feed-post-link')
                if not link:
                    link = item.select_one('a[href*="/noticia/"]')
                if not link:
                    continue

                title = link.get_text(strip=True)
                href = link.get('href', '')

                if not title or not href:
                    continue

                # Extract date from URL
                date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', href)
                if date_match:
                    date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                else:
                    date_str = datetime.now().strftime('%Y-%m-%d')

                articles.append({
                    'title': title[:500],
                    'url': href,
                    'date': date_str,
                    'source': 'g1',
                    'keyword': 'economia',
                })
            except Exception:
                continue

        time.sleep(REQUEST_DELAY)

    # Remove duplicates
    seen = set()
    unique = [a for a in articles if not (a['url'] in seen or seen.add(a['url']))]

    print(f"    Found {len(unique)} articles from G1")
    return unique


# =============================================================================
# VALOR ECONÔMICO - DIRECT PAGE SCRAPING
# =============================================================================

def scrape_valor_main(session) -> List[Dict]:
    """
    Scrape Valor Econômico main pages.
    """
    articles = []

    urls = [
        'https://valor.globo.com/',
        'https://valor.globo.com/financas/',
        'https://valor.globo.com/brasil/',
        'https://valor.globo.com/empresas/',
    ]

    print(f"\n  Scraping Valor Econômico (latest news)...")

    for url in urls:
        html = fetch_url(session, url)
        if not html:
            continue

        soup = get_soup(html)
        if not soup:
            continue

        # Find all article links with date pattern in URL
        links = soup.find_all('a', href=True)

        for link in links:
            href = link.get('href', '')

            # Filter for Valor article URLs
            if 'valor.globo.com' not in href:
                continue

            date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', href)
            if not date_match:
                continue

            title = link.get_text(strip=True)
            if not title or len(title) < 20:
                continue

            date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"

            articles.append({
                'title': title[:500],
                'url': href,
                'date': date_str,
                'source': 'valor',
                'keyword': 'economia',
            })

        time.sleep(REQUEST_DELAY)

    # Remove duplicates
    seen = set()
    unique = [a for a in articles if not (a['url'] in seen or seen.add(a['url']))]

    print(f"    Found {len(unique)} articles from Valor")
    return unique


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect news from Brazilian financial news sources',
    )
    parser.add_argument('--start-year', type=int, default=2011,
                        help='Start year for Folha search (default: 2011)')
    parser.add_argument('--end-year', type=int, default=2025,
                        help='End year for Folha search (default: 2025)')
    parser.add_argument('--keywords', type=str, default=None,
                        help='Comma-separated keywords for Folha search')
    parser.add_argument('--folha-only', action='store_true',
                        help='Only scrape Folha (has historical archive)')

    args = parser.parse_args()

    keywords = DEFAULT_KEYWORDS
    if args.keywords:
        keywords = [k.strip() for k in args.keywords.split(',')]

    print("=" * 60)
    print("NEWS COLLECTION FOR FINANCIAL STRESS INDEX")
    print("=" * 60)
    print(f"\nFolha search: {args.start_year} to {args.end_year}")
    print(f"Keywords: {keywords[:3]}... ({len(keywords)} total)")
    print(f"Output: {DATA_DIR}")

    session = get_session()
    if not session:
        return

    all_articles = []

    # Folha - Has working historical search archive
    print("\n" + "-" * 40)
    print("FOLHA DE SÃO PAULO (Historical Archive)")
    print("-" * 40)
    folha_articles = scrape_folha_archive(session, args.start_year, args.end_year, keywords)
    all_articles.extend(folha_articles)

    if not args.folha_only:
        # G1 and Valor - Only latest news (search uses JavaScript)
        print("\n" + "-" * 40)
        print("G1 & VALOR (Latest News Only)")
        print("-" * 40)
        print("Note: Search pages use JavaScript, collecting from main pages only")

        g1_articles = scrape_g1_main(session)
        all_articles.extend(g1_articles)

        valor_articles = scrape_valor_main(session)
        all_articles.extend(valor_articles)

    if not all_articles:
        print("\n  ERROR: No articles collected")
        return

    # Create DataFrame
    df = pd.DataFrame(all_articles)
    df = df.drop_duplicates(subset=['url'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date', ascending=True)

    # Save
    combined_path = DATA_DIR / 'news_combined.csv'
    df.to_csv(combined_path, index=False)
    print(f"\n  Saved {len(df)} articles to {combined_path}")

    # Save per-source
    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        source_path = DATA_DIR / f'news_{source}.csv'
        source_df.to_csv(source_path, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("NEWS COLLECTION COMPLETE")
    print("=" * 60)
    print(f"\nTotal: {len(df)} articles")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    print(f"\nBy source:")
    for source, count in df['source'].value_counts().items():
        print(f"  - {source}: {count}")

    print(f"\nBy year:")
    df['year'] = df['date'].dt.year
    for year in sorted(df['year'].unique())[-10:]:  # Show last 10 years
        print(f"  - {year}: {(df['year'] == year).sum()}")

    print(f"\nNext: python scripts/news_fsi.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
