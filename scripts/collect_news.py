#!/usr/bin/env python3
"""
News Collection Script for Brazilian Financial News
====================================================
Scrapes news from G1 Economia, Valor Econômico, and Folha Mercado.

Based on methodologies from:
- G1: https://github.com/leviobrabo/G1-news-scraping
- Valor: https://github.com/vmesel/valor_alert
- Folha: https://github.com/ruanrf/webscraper-folha

Usage:
    # Scrape all sources
    python scripts/collect_news.py

    # Scrape specific source
    python scripts/collect_news.py --source g1
    python scripts/collect_news.py --source valor
    python scripts/collect_news.py --source folha

    # Search with keywords
    python scripts/collect_news.py --keywords "crise,inflação"

    # Specify number of pages
    python scripts/collect_news.py --pages 10

Output:
    data/raw/news_g1.csv
    data/raw/news_valor.csv
    data/raw/news_folha.csv
    data/raw/news_combined.csv
"""

import argparse
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
from urllib.parse import urljoin, quote

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

# URL configurations for each source
SOURCES = {
    'g1': {
        'name': 'G1 Economia',
        'base_url': 'https://g1.globo.com',
        'economia_url': 'https://g1.globo.com/economia/',
        'search_url': 'https://g1.globo.com/busca/?q={query}&page={page}&order=recent&species=notícias',
    },
    'valor': {
        'name': 'Valor Econômico',
        'base_url': 'https://valor.globo.com',
        'economia_url': 'https://valor.globo.com/',
        'search_url': 'https://valor.globo.com/busca/?q={query}&page={page}&order=recent',
    },
    'folha': {
        'name': 'Folha de S. Paulo',
        'base_url': 'https://www.folha.uol.com.br',
        'economia_url': 'https://www.folha.uol.com.br/mercado/',
        'search_url': 'https://search.folha.uol.com.br/search?q={query}&site=todos&periodo=todos&sr={offset}',
    },
}


# =============================================================================
# SCRAPER BASE CLASS
# =============================================================================

class NewsScraper:
    """Base class for news scrapers."""

    def __init__(self, source: str):
        self.source = source
        self.config = SOURCES[source]
        self.session = None

    def _get_session(self):
        """Get or create requests session."""
        if self.session is None:
            try:
                import requests
                self.session = requests.Session()
                self.session.headers.update(HEADERS)
            except ImportError:
                print("ERROR: requests not installed. Install with: pip install requests")
                return None
        return self.session

    def _get_soup(self, url: str):
        """Fetch URL and return BeautifulSoup object."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            print("ERROR: BeautifulSoup not installed. Install with: pip install beautifulsoup4")
            return None

        session = self._get_session()
        if not session:
            return None

        for attempt in range(MAX_RETRIES):
            try:
                response = session.get(url, timeout=30)
                response.raise_for_status()
                return BeautifulSoup(response.text, 'html.parser')
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(REQUEST_DELAY * (attempt + 1))
                else:
                    print(f"  Error fetching {url}: {e}")
                    return None
        return None

    def scrape_latest(self, pages: int = 5) -> List[Dict]:
        """Scrape latest news articles. Override in subclasses."""
        raise NotImplementedError

    def search(self, query: str, pages: int = 5) -> List[Dict]:
        """Search for news with keyword. Override in subclasses."""
        raise NotImplementedError


# =============================================================================
# G1 SCRAPER
# =============================================================================

class G1Scraper(NewsScraper):
    """Scraper for G1 Economia news."""

    def __init__(self):
        super().__init__('g1')

    def scrape_latest(self, pages: int = 5) -> List[Dict]:
        """Scrape latest G1 Economia articles."""
        articles = []

        # Scrape main economia page
        soup = self._get_soup(self.config['economia_url'])
        if soup:
            articles.extend(self._parse_g1_page(soup))

        print(f"    Found {len(articles)} articles from G1")
        return articles

    def search(self, query: str, pages: int = 5) -> List[Dict]:
        """Search G1 for articles matching query."""
        articles = []

        for page in range(1, pages + 1):
            url = self.config['search_url'].format(query=quote(query), page=page)
            print(f"    Searching G1 page {page}...")

            soup = self._get_soup(url)
            if soup:
                page_articles = self._parse_g1_search(soup)
                if not page_articles:
                    break
                articles.extend(page_articles)

            time.sleep(REQUEST_DELAY)

        print(f"    Found {len(articles)} articles from G1 search")
        return articles

    def _parse_g1_page(self, soup) -> List[Dict]:
        """Parse G1 news page for articles."""
        articles = []

        # G1 uses various article containers
        selectors = [
            'div.bastian-feed-item',
            'div.feed-post-body',
            'article',
            'div.post-item',
        ]

        for selector in selectors:
            items = soup.select(selector)
            for item in items:
                try:
                    # Try to find title link
                    link = item.select_one('a.feed-post-link, a.post-link, a[href*="/noticia/"]')
                    if not link:
                        continue

                    title = link.get_text(strip=True)
                    url = link.get('href', '')
                    if not url.startswith('http'):
                        url = urljoin(self.config['base_url'], url)

                    # Try to find date
                    date_elem = item.select_one('span.feed-post-datetime, time, span.timestamp')
                    date_str = date_elem.get_text(strip=True) if date_elem else ''

                    # Try to find summary
                    summary_elem = item.select_one('div.feed-post-body-resumo, p.summary')
                    summary = summary_elem.get_text(strip=True) if summary_elem else ''

                    if title and url:
                        articles.append({
                            'title': title,
                            'url': url,
                            'date': self._parse_date(date_str),
                            'summary': summary,
                            'source': 'g1',
                        })
                except Exception:
                    continue

        return articles

    def _parse_g1_search(self, soup) -> List[Dict]:
        """Parse G1 search results."""
        articles = []

        items = soup.select('li.widget--info')
        for item in items:
            try:
                link = item.select_one('a')
                if not link:
                    continue

                title_elem = item.select_one('div.widget--info__title')
                title = title_elem.get_text(strip=True) if title_elem else link.get_text(strip=True)
                url = link.get('href', '')

                date_elem = item.select_one('div.widget--info__meta')
                date_str = date_elem.get_text(strip=True) if date_elem else ''

                if title and url:
                    articles.append({
                        'title': title,
                        'url': url,
                        'date': self._parse_date(date_str),
                        'summary': '',
                        'source': 'g1',
                    })
            except Exception:
                continue

        return articles

    def _parse_date(self, date_str: str) -> str:
        """Parse date string to ISO format."""
        if not date_str:
            return datetime.now().strftime('%Y-%m-%d')

        # Try various Portuguese date formats
        date_str = date_str.lower().strip()

        if 'hoje' in date_str or 'há' in date_str:
            return datetime.now().strftime('%Y-%m-%d')
        if 'ontem' in date_str:
            return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        # Try DD/MM/YYYY or DD/MM
        match = re.search(r'(\d{1,2})/(\d{1,2})(?:/(\d{4}))?', date_str)
        if match:
            day, month = int(match.group(1)), int(match.group(2))
            year = int(match.group(3)) if match.group(3) else datetime.now().year
            try:
                return datetime(year, month, day).strftime('%Y-%m-%d')
            except ValueError:
                pass

        return datetime.now().strftime('%Y-%m-%d')


# =============================================================================
# VALOR SCRAPER
# =============================================================================

class ValorScraper(NewsScraper):
    """Scraper for Valor Econômico news."""

    def __init__(self):
        super().__init__('valor')

    def scrape_latest(self, pages: int = 5) -> List[Dict]:
        """Scrape latest Valor Econômico articles."""
        articles = []

        # Scrape main page
        soup = self._get_soup(self.config['economia_url'])
        if soup:
            articles.extend(self._parse_valor_page(soup))

        # Also try different sections
        sections = ['financas', 'brasil', 'mundo', 'empresas']
        for section in sections:
            url = f"{self.config['base_url']}/{section}/"
            soup = self._get_soup(url)
            if soup:
                articles.extend(self._parse_valor_page(soup))
            time.sleep(REQUEST_DELAY)

        # Remove duplicates
        seen_urls = set()
        unique_articles = []
        for article in articles:
            if article['url'] not in seen_urls:
                seen_urls.add(article['url'])
                unique_articles.append(article)

        print(f"    Found {len(unique_articles)} articles from Valor")
        return unique_articles

    def search(self, query: str, pages: int = 5) -> List[Dict]:
        """Search Valor for articles matching query."""
        articles = []

        for page in range(1, pages + 1):
            url = self.config['search_url'].format(query=quote(query), page=page)
            print(f"    Searching Valor page {page}...")

            soup = self._get_soup(url)
            if soup:
                page_articles = self._parse_valor_search(soup)
                if not page_articles:
                    break
                articles.extend(page_articles)

            time.sleep(REQUEST_DELAY)

        print(f"    Found {len(articles)} articles from Valor search")
        return articles

    def _parse_valor_page(self, soup) -> List[Dict]:
        """Parse Valor page for articles."""
        articles = []

        # Find all article links
        links = soup.find_all('a', href=True)
        for link in links:
            try:
                href = link.get('href', '')

                # Filter for article URLs (contain date pattern)
                if not re.search(r'/\d{4}/\d{2}/\d{2}/', href):
                    continue
                if 'valor.globo.com' not in href and not href.startswith('/'):
                    continue

                title = link.get_text(strip=True)
                if not title or len(title) < 20:  # Skip short text
                    continue

                url = href if href.startswith('http') else urljoin(self.config['base_url'], href)

                # Extract date from URL
                date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
                date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}" if date_match else datetime.now().strftime('%Y-%m-%d')

                articles.append({
                    'title': title,
                    'url': url,
                    'date': date_str,
                    'summary': '',
                    'source': 'valor',
                })
            except Exception:
                continue

        return articles

    def _parse_valor_search(self, soup) -> List[Dict]:
        """Parse Valor search results."""
        return self._parse_valor_page(soup)


# =============================================================================
# FOLHA SCRAPER
# =============================================================================

class FolhaScraper(NewsScraper):
    """Scraper for Folha de S. Paulo (Mercado section)."""

    def __init__(self):
        super().__init__('folha')

    def scrape_latest(self, pages: int = 5) -> List[Dict]:
        """Scrape latest Folha Mercado articles."""
        articles = []

        # Scrape mercado section
        soup = self._get_soup(self.config['economia_url'])
        if soup:
            articles.extend(self._parse_folha_page(soup))

        print(f"    Found {len(articles)} articles from Folha")
        return articles

    def search(self, query: str, pages: int = 5) -> List[Dict]:
        """Search Folha for articles matching query."""
        articles = []

        for page in range(1, pages + 1):
            offset = (page - 1) * 25 + 1
            url = self.config['search_url'].format(query=quote(query), offset=offset)
            print(f"    Searching Folha page {page}...")

            soup = self._get_soup(url)
            if soup:
                page_articles = self._parse_folha_search(soup)
                if not page_articles:
                    break
                articles.extend(page_articles)

            time.sleep(REQUEST_DELAY)

        print(f"    Found {len(articles)} articles from Folha search")
        return articles

    def _parse_folha_page(self, soup) -> List[Dict]:
        """Parse Folha page for articles."""
        articles = []

        # Find article containers
        selectors = [
            'div.c-headline',
            'article',
            'div.c-main-headline',
        ]

        for selector in selectors:
            items = soup.select(selector)
            for item in items:
                try:
                    link = item.select_one('a[href*="folha.uol.com.br"]')
                    if not link:
                        link = item.select_one('a')
                    if not link:
                        continue

                    title_elem = item.select_one('h2, h3, span.c-headline__title')
                    title = title_elem.get_text(strip=True) if title_elem else link.get_text(strip=True)
                    url = link.get('href', '')

                    if not url.startswith('http'):
                        url = urljoin(self.config['base_url'], url)

                    # Try to find date
                    time_elem = item.select_one('time')
                    date_str = time_elem.get('datetime', '') if time_elem else ''
                    if not date_str:
                        date_str = datetime.now().strftime('%Y-%m-%d')
                    else:
                        date_str = date_str[:10]  # Get just the date part

                    if title and url and 'folha.uol.com.br' in url:
                        articles.append({
                            'title': title,
                            'url': url,
                            'date': date_str,
                            'summary': '',
                            'source': 'folha',
                        })
                except Exception:
                    continue

        return articles

    def _parse_folha_search(self, soup) -> List[Dict]:
        """Parse Folha search results."""
        articles = []

        items = soup.select('ol.c-search li.c-search__result')
        if not items:
            items = soup.select('div.search-result')

        for item in items:
            try:
                link = item.select_one('a')
                if not link:
                    continue

                title = link.get_text(strip=True)
                url = link.get('href', '')

                if title and url:
                    articles.append({
                        'title': title,
                        'url': url,
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'summary': '',
                        'source': 'folha',
                    })
            except Exception:
                continue

        return articles


# =============================================================================
# MAIN COLLECTION FUNCTIONS
# =============================================================================

def collect_all_news(pages: int = 5, keywords: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Collect news from all sources.

    Args:
        pages: Number of pages to scrape per source
        keywords: Optional list of keywords to search for

    Returns:
        DataFrame with all collected articles
    """
    all_articles = []

    scrapers = {
        'g1': G1Scraper(),
        'valor': ValorScraper(),
        'folha': FolhaScraper(),
    }

    for source_name, scraper in scrapers.items():
        print(f"\n  Scraping {scraper.config['name']}...")

        if keywords:
            for keyword in keywords:
                articles = scraper.search(keyword, pages)
                all_articles.extend(articles)
                time.sleep(REQUEST_DELAY)
        else:
            articles = scraper.scrape_latest(pages)
            all_articles.extend(articles)

        time.sleep(REQUEST_DELAY * 2)

    if not all_articles:
        print("\n  WARNING: No articles collected")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(all_articles)

    # Remove duplicates
    df = df.drop_duplicates(subset=['url'])

    # Sort by date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date', ascending=False)

    return df


def collect_from_source(source: str, pages: int = 5, keywords: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Collect news from a specific source.

    Args:
        source: Source name ('g1', 'valor', or 'folha')
        pages: Number of pages to scrape
        keywords: Optional list of keywords to search for

    Returns:
        DataFrame with collected articles
    """
    scrapers = {
        'g1': G1Scraper,
        'valor': ValorScraper,
        'folha': FolhaScraper,
    }

    if source not in scrapers:
        print(f"ERROR: Unknown source '{source}'. Use: g1, valor, or folha")
        return pd.DataFrame()

    scraper = scrapers[source]()
    articles = []

    print(f"\n  Scraping {scraper.config['name']}...")

    if keywords:
        for keyword in keywords:
            articles.extend(scraper.search(keyword, pages))
            time.sleep(REQUEST_DELAY)
    else:
        articles.extend(scraper.scrape_latest(pages))

    if not articles:
        print("\n  WARNING: No articles collected")
        return pd.DataFrame()

    df = pd.DataFrame(articles)
    df = df.drop_duplicates(subset=['url'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date', ascending=False)

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect news from Brazilian financial news sources',
    )
    parser.add_argument('--source', type=str, choices=['g1', 'valor', 'folha', 'all'],
                        default='all', help='News source to scrape (default: all)')
    parser.add_argument('--keywords', type=str, default=None,
                        help='Comma-separated keywords to search for')
    parser.add_argument('--pages', type=int, default=5,
                        help='Number of pages to scrape (default: 5)')

    args = parser.parse_args()

    keywords = None
    if args.keywords:
        keywords = [k.strip() for k in args.keywords.split(',')]

    print("=" * 60)
    print("NEWS COLLECTION")
    print("=" * 60)
    print(f"\nSource: {args.source}")
    print(f"Pages: {args.pages}")
    if keywords:
        print(f"Keywords: {keywords}")
    print(f"Output: {DATA_DIR}")

    print("\n" + "-" * 40)
    print("Collecting articles...")
    print("-" * 40)

    if args.source == 'all':
        df = collect_all_news(args.pages, keywords)

        if not df.empty:
            # Save combined file
            combined_path = DATA_DIR / 'news_combined.csv'
            df.to_csv(combined_path, index=False)

            # Save per-source files
            for source in df['source'].unique():
                source_df = df[df['source'] == source]
                source_path = DATA_DIR / f'news_{source}.csv'
                source_df.to_csv(source_path, index=False)
                print(f"  Saved {len(source_df)} articles to {source_path}")

            print(f"\n  Saved {len(df)} total articles to {combined_path}")
    else:
        df = collect_from_source(args.source, args.pages, keywords)

        if not df.empty:
            output_path = DATA_DIR / f'news_{args.source}.csv'
            df.to_csv(output_path, index=False)
            print(f"\n  Saved {len(df)} articles to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    if df.empty:
        print("NEWS COLLECTION FAILED")
        print("=" * 60)
        print("\nNo articles collected. Check your internet connection.")
    else:
        print("NEWS COLLECTION COMPLETE")
        print("=" * 60)
        print(f"\nTotal articles: {len(df)}")
        print(f"Date range: {df['date'].min().date() if pd.notna(df['date'].min()) else 'N/A'} to {df['date'].max().date() if pd.notna(df['date'].max()) else 'N/A'}")
        print(f"\nArticles by source:")
        for source, count in df['source'].value_counts().items():
            print(f"  - {source}: {count}")
        print(f"\nNext step: python scripts/news_fsi.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
