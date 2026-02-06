#!/usr/bin/env python3
"""
Historical News Collection Script for Brazilian Financial News
================================================================
Scrapes historical news from G1 Economia, Valor Econômico, and Folha Mercado.
Uses search/archive functionality to access articles from 2011 onwards.

Based on methodologies from:
- G1: https://github.com/leviobrabo/G1-news-scraping
- Valor: https://github.com/vmesel/valor_alert
- Folha: https://github.com/ruanrf/webscraper-folha

Usage:
    # Scrape all sources (default: 2011 to present)
    python scripts/collect_news.py

    # Specific date range
    python scripts/collect_news.py --start-year 2015 --end-year 2024

    # Specific source
    python scripts/collect_news.py --source folha

    # Custom keywords
    python scripts/collect_news.py --keywords "crise,recessão"

Output:
    data/raw/news_g1.csv
    data/raw/news_valor.csv
    data/raw/news_folha.csv
    data/raw/news_combined.csv
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
from urllib.parse import urljoin, quote, urlencode

import pandas as pd

# Handle both script execution and interactive usage
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

# Request delays to avoid rate limiting
REQUEST_DELAY = 2  # seconds between requests
MAX_RETRIES = 3

# User agent to mimic browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
}

# Default search keywords for financial/economic news
DEFAULT_KEYWORDS = [
    'economia brasileira',
    'crise financeira',
    'bolsa de valores',
    'ibovespa',
    'mercado financeiro',
    'banco central',
    'inflação brasil',
    'taxa selic',
    'dólar real',
    'recessão',
]

# Maximum articles per source per year (to avoid excessive scraping)
MAX_ARTICLES_PER_YEAR = 500


# =============================================================================
# HTTP HELPERS
# =============================================================================

def get_session():
    """Get requests session with headers."""
    try:
        import requests
        session = requests.Session()
        session.headers.update(HEADERS)
        return session
    except ImportError:
        print("ERROR: requests not installed. Install with: pip install requests")
        return None


def fetch_url(session, url: str, timeout: int = 30) -> Optional[str]:
    """Fetch URL with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(REQUEST_DELAY * (attempt + 1))
            else:
                return None
    return None


def get_soup(html: str):
    """Parse HTML with BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, 'html.parser')
    except ImportError:
        print("ERROR: BeautifulSoup not installed. Install with: pip install beautifulsoup4")
        return None


# =============================================================================
# FOLHA DE SÃO PAULO - ARCHIVE SEARCH
# =============================================================================

def scrape_folha_archive(session, start_year: int, end_year: int,
                         keywords: List[str] = None) -> List[Dict]:
    """
    Scrape Folha de São Paulo using their search archive.

    Folha search URL format:
    https://search.folha.uol.com.br/search?q=QUERY&site=todos&periodo=personalizado&sd=DD/MM/YYYY&ed=DD/MM/YYYY&sr=OFFSET

    Each page has 25 results.
    """
    articles = []
    keywords = keywords or DEFAULT_KEYWORDS

    print(f"\n  Scraping Folha de S. Paulo archive ({start_year}-{end_year})...")

    for year in range(start_year, end_year + 1):
        year_articles = []
        print(f"    Year {year}...", end=" ", flush=True)

        for keyword in keywords[:5]:  # Limit keywords per year
            start_date = f"01/01/{year}"
            end_date = f"31/12/{year}"

            # Iterate through pages
            for page in range(1, 21):  # Max 20 pages per keyword per year
                offset = (page - 1) * 25 + 1

                url = (
                    f"https://search.folha.uol.com.br/search?"
                    f"q={quote(keyword)}&site=online&periodo=personalizado"
                    f"&sd={start_date}&ed={end_date}&sr={offset}"
                )

                html = fetch_url(session, url)
                if not html:
                    break

                soup = get_soup(html)
                if not soup:
                    break

                # Find search results
                results = soup.select('ol.c-search li.c-search__result')
                if not results:
                    # Try alternative selector
                    results = soup.select('div.search-result, li.search-result-item')

                if not results:
                    break

                for item in results:
                    try:
                        link = item.select_one('a.c-search__result__link, a')
                        if not link:
                            continue

                        title_elem = item.select_one('h2.c-search__result__title, h3, span.title')
                        title = title_elem.get_text(strip=True) if title_elem else link.get_text(strip=True)

                        url_article = link.get('href', '')
                        if not url_article or 'folha.uol.com.br' not in url_article:
                            continue

                        # Extract date from URL or metadata
                        date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url_article)
                        if date_match:
                            date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                        else:
                            # Try to find date in text
                            date_elem = item.select_one('time, span.c-search__result__date')
                            if date_elem:
                                date_text = date_elem.get_text(strip=True)
                                # Parse DD/MM/YYYY or similar
                                dm = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_text)
                                if dm:
                                    date_str = f"{dm.group(3)}-{dm.group(2).zfill(2)}-{dm.group(1).zfill(2)}"
                                else:
                                    date_str = f"{year}-01-01"
                            else:
                                date_str = f"{year}-01-01"

                        # Summary
                        summary_elem = item.select_one('p.c-search__result__summary, p')
                        summary = summary_elem.get_text(strip=True) if summary_elem else ''

                        year_articles.append({
                            'title': title[:500],
                            'url': url_article,
                            'date': date_str,
                            'summary': summary[:500],
                            'source': 'folha',
                            'keyword': keyword,
                        })
                    except Exception:
                        continue

                time.sleep(REQUEST_DELAY)

                # Stop if we got enough articles
                if len(year_articles) >= MAX_ARTICLES_PER_YEAR:
                    break

            if len(year_articles) >= MAX_ARTICLES_PER_YEAR:
                break

        # Remove duplicates within year
        seen_urls = set()
        unique = []
        for a in year_articles:
            if a['url'] not in seen_urls:
                seen_urls.add(a['url'])
                unique.append(a)

        articles.extend(unique)
        print(f"{len(unique)} articles")

    print(f"    Total: {len(articles)} articles from Folha")
    return articles


# =============================================================================
# VALOR ECONÔMICO - ARCHIVE SEARCH
# =============================================================================

def scrape_valor_archive(session, start_year: int, end_year: int,
                         keywords: List[str] = None) -> List[Dict]:
    """
    Scrape Valor Econômico using their search.

    Valor search URL format:
    https://valor.globo.com/busca/?q=QUERY&page=PAGE&order=recent&from=YYYY-MM-DDTHH:MM:SS-0300&to=YYYY-MM-DDTHH:MM:SS-0300
    """
    articles = []
    keywords = keywords or DEFAULT_KEYWORDS

    print(f"\n  Scraping Valor Econômico archive ({start_year}-{end_year})...")

    for year in range(start_year, end_year + 1):
        year_articles = []
        print(f"    Year {year}...", end=" ", flush=True)

        for keyword in keywords[:5]:
            from_date = f"{year}-01-01T00:00:00-0300"
            to_date = f"{year}-12-31T23:59:59-0300"

            for page in range(1, 21):
                url = (
                    f"https://valor.globo.com/busca/?"
                    f"q={quote(keyword)}&page={page}&order=recent"
                    f"&from={quote(from_date)}&to={quote(to_date)}"
                )

                html = fetch_url(session, url)
                if not html:
                    break

                soup = get_soup(html)
                if not soup:
                    break

                # Find search results - Globo sites use widget--info
                results = soup.select('li.widget--info')
                if not results:
                    results = soup.select('div.search-result, article')

                if not results:
                    break

                for item in results:
                    try:
                        link = item.select_one('a')
                        if not link:
                            continue

                        href = link.get('href', '')
                        if not href:
                            continue

                        # Only keep valor.globo.com URLs
                        if 'valor.globo.com' not in href and not href.startswith('/'):
                            continue

                        if href.startswith('/'):
                            href = f"https://valor.globo.com{href}"

                        # Filter for article URLs (contain date pattern)
                        if not re.search(r'/\d{4}/\d{2}/\d{2}/', href):
                            continue

                        title_elem = item.select_one('div.widget--info__title, h2, h3')
                        title = title_elem.get_text(strip=True) if title_elem else link.get_text(strip=True)

                        if len(title) < 15:
                            continue

                        # Extract date from URL
                        date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', href)
                        date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}" if date_match else f"{year}-01-01"

                        year_articles.append({
                            'title': title[:500],
                            'url': href,
                            'date': date_str,
                            'summary': '',
                            'source': 'valor',
                            'keyword': keyword,
                        })
                    except Exception:
                        continue

                time.sleep(REQUEST_DELAY)

                if len(year_articles) >= MAX_ARTICLES_PER_YEAR:
                    break

            if len(year_articles) >= MAX_ARTICLES_PER_YEAR:
                break

        # Remove duplicates
        seen_urls = set()
        unique = []
        for a in year_articles:
            if a['url'] not in seen_urls:
                seen_urls.add(a['url'])
                unique.append(a)

        articles.extend(unique)
        print(f"{len(unique)} articles")

    print(f"    Total: {len(articles)} articles from Valor")
    return articles


# =============================================================================
# G1 - ARCHIVE SEARCH
# =============================================================================

def scrape_g1_archive(session, start_year: int, end_year: int,
                      keywords: List[str] = None) -> List[Dict]:
    """
    Scrape G1 Economia using their search.

    G1 search URL format:
    https://g1.globo.com/busca/?q=QUERY&page=PAGE&order=recent&species=notícias&from=YYYY-MM-DDTHH:MM:SS-0300&to=YYYY-MM-DDTHH:MM:SS-0300
    """
    articles = []
    keywords = keywords or DEFAULT_KEYWORDS

    print(f"\n  Scraping G1 Economia archive ({start_year}-{end_year})...")

    for year in range(start_year, end_year + 1):
        year_articles = []
        print(f"    Year {year}...", end=" ", flush=True)

        for keyword in keywords[:5]:
            from_date = f"{year}-01-01T00:00:00-0300"
            to_date = f"{year}-12-31T23:59:59-0300"

            for page in range(1, 21):
                url = (
                    f"https://g1.globo.com/busca/?"
                    f"q={quote(keyword)}&page={page}&order=recent&species=not%C3%ADcias"
                    f"&from={quote(from_date)}&to={quote(to_date)}"
                )

                html = fetch_url(session, url)
                if not html:
                    break

                soup = get_soup(html)
                if not soup:
                    break

                results = soup.select('li.widget--info')
                if not results:
                    results = soup.select('div.search-result, article')

                if not results:
                    break

                for item in results:
                    try:
                        link = item.select_one('a')
                        if not link:
                            continue

                        href = link.get('href', '')
                        if not href:
                            continue

                        # Filter for G1 economy URLs
                        if 'g1.globo.com' not in href and 'economia' not in href.lower():
                            continue

                        title_elem = item.select_one('div.widget--info__title, h2, h3')
                        title = title_elem.get_text(strip=True) if title_elem else link.get_text(strip=True)

                        if len(title) < 15:
                            continue

                        # Extract date
                        date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', href)
                        if date_match:
                            date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                        else:
                            # Try meta date
                            meta_elem = item.select_one('div.widget--info__meta, span.timestamp')
                            if meta_elem:
                                meta_text = meta_elem.get_text(strip=True)
                                dm = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', meta_text)
                                if dm:
                                    date_str = f"{dm.group(3)}-{dm.group(2).zfill(2)}-{dm.group(1).zfill(2)}"
                                else:
                                    date_str = f"{year}-01-01"
                            else:
                                date_str = f"{year}-01-01"

                        year_articles.append({
                            'title': title[:500],
                            'url': href,
                            'date': date_str,
                            'summary': '',
                            'source': 'g1',
                            'keyword': keyword,
                        })
                    except Exception:
                        continue

                time.sleep(REQUEST_DELAY)

                if len(year_articles) >= MAX_ARTICLES_PER_YEAR:
                    break

            if len(year_articles) >= MAX_ARTICLES_PER_YEAR:
                break

        # Remove duplicates
        seen_urls = set()
        unique = []
        for a in year_articles:
            if a['url'] not in seen_urls:
                seen_urls.add(a['url'])
                unique.append(a)

        articles.extend(unique)
        print(f"{len(unique)} articles")

    print(f"    Total: {len(articles)} articles from G1")
    return articles


# =============================================================================
# MAIN COLLECTION
# =============================================================================

def collect_all_news(start_year: int, end_year: int,
                     keywords: List[str] = None,
                     sources: List[str] = None) -> pd.DataFrame:
    """
    Collect news from all sources for the given year range.
    """
    session = get_session()
    if not session:
        return pd.DataFrame()

    all_articles = []
    sources = sources or ['folha', 'valor', 'g1']

    if 'folha' in sources:
        articles = scrape_folha_archive(session, start_year, end_year, keywords)
        all_articles.extend(articles)

    if 'valor' in sources:
        articles = scrape_valor_archive(session, start_year, end_year, keywords)
        all_articles.extend(articles)

    if 'g1' in sources:
        articles = scrape_g1_archive(session, start_year, end_year, keywords)
        all_articles.extend(articles)

    if not all_articles:
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)

    # Remove duplicates by URL
    df = df.drop_duplicates(subset=['url'])

    # Parse and sort by date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date', ascending=True)

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect historical news from Brazilian financial news sources',
    )
    parser.add_argument('--start-year', type=int, default=2011,
                        help='Start year (default: 2011)')
    parser.add_argument('--end-year', type=int, default=2025,
                        help='End year (default: 2025)')
    parser.add_argument('--source', type=str, choices=['g1', 'valor', 'folha', 'all'],
                        default='all', help='News source to scrape (default: all)')
    parser.add_argument('--keywords', type=str, default=None,
                        help='Comma-separated keywords to search for')

    args = parser.parse_args()

    keywords = None
    if args.keywords:
        keywords = [k.strip() for k in args.keywords.split(',')]
    else:
        keywords = DEFAULT_KEYWORDS

    sources = ['folha', 'valor', 'g1'] if args.source == 'all' else [args.source]

    print("=" * 60)
    print("HISTORICAL NEWS COLLECTION")
    print("=" * 60)
    print(f"\nPeriod: {args.start_year} to {args.end_year}")
    print(f"Sources: {', '.join(sources)}")
    print(f"Keywords: {keywords[:5]}...")  # Show first 5
    print(f"Output: {DATA_DIR}")

    print("\n" + "-" * 40)
    print("Collecting articles...")
    print("-" * 40)

    df = collect_all_news(args.start_year, args.end_year, keywords, sources)

    if df.empty:
        print("\n  WARNING: No articles collected")
        print("  Check your internet connection or try different keywords")
        return

    # Save combined file
    combined_path = DATA_DIR / 'news_combined.csv'
    df.to_csv(combined_path, index=False)
    print(f"\n  Saved {len(df)} total articles to {combined_path}")

    # Save per-source files
    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        source_path = DATA_DIR / f'news_{source}.csv'
        source_df.to_csv(source_path, index=False)
        print(f"  Saved {len(source_df)} articles to {source_path}")

    # Summary
    print("\n" + "=" * 60)
    print("NEWS COLLECTION COMPLETE")
    print("=" * 60)
    print(f"\nTotal articles: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    print(f"\nArticles by source:")
    for source, count in df['source'].value_counts().items():
        print(f"  - {source}: {count}")

    print(f"\nArticles by year:")
    df['year'] = df['date'].dt.year
    for year in sorted(df['year'].unique()):
        count = (df['year'] == year).sum()
        print(f"  - {year}: {count}")

    print(f"\nNext step: python scripts/news_fsi.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
