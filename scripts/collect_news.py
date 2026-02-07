#!/usr/bin/env python3
"""
News Collection with Selenium Support
======================================
Scrapes historical news from Brazilian financial news sources.
Uses Selenium for JavaScript-rendered pages (G1, Valor).

Requirements:
    pip install selenium webdriver-manager

Usage:
    python scripts/collect_news.py
    python scripts/collect_news.py --start-year 2011 --end-year 2025
    python scripts/collect_news.py --no-selenium  # Skip G1/Valor search

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

REQUEST_DELAY = 2
MAX_RETRIES = 3

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
}

# Financial/economic keywords for search
DEFAULT_KEYWORDS = [
    'crise financeira',
    'crise econômica',
    'recessão brasil',
    'ibovespa',
    'bolsa valores',
    'mercado financeiro',
    'banco central',
    'inflação',
    'taxa selic',
    'dólar',
]

MAX_ARTICLES_PER_KEYWORD = 300
MAX_PAGES_PER_SEARCH = 15


# =============================================================================
# HTTP SESSION (for non-JS pages)
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
            elif response.status_code == 429:
                print(f"    Rate limited, waiting...")
                time.sleep(10)
            else:
                return None
        except Exception:
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
# SELENIUM DRIVER
# =============================================================================

def get_selenium_driver():
    """Initialize Selenium WebDriver with Chrome."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options

        # Try to use webdriver-manager for automatic driver management
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())
        except ImportError:
            print("  Note: webdriver-manager not installed, using system chromedriver")
            service = None

        options = Options()
        options.add_argument('--headless')  # Run in background
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument(f'--user-agent={HEADERS["User-Agent"]}')
        options.add_argument('--lang=pt-BR')

        # Disable images for faster loading
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)

        if service:
            driver = webdriver.Chrome(service=service, options=options)
        else:
            driver = webdriver.Chrome(options=options)

        driver.set_page_load_timeout(60)
        return driver

    except ImportError:
        print("\n  ERROR: Selenium not installed.")
        print("  Install with: pip install selenium webdriver-manager")
        return None
    except Exception as e:
        print(f"\n  ERROR initializing Selenium: {e}")
        print("  Make sure Chrome browser is installed.")
        return None


def selenium_fetch(driver, url: str, wait_time: int = 5) -> Optional[str]:
    """Fetch URL using Selenium and return page source."""
    try:
        driver.get(url)
        time.sleep(wait_time)  # Wait for JavaScript to render

        # Scroll down to load more content
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        return driver.page_source
    except Exception as e:
        print(f"    Selenium error: {e}")
        return None


# =============================================================================
# FOLHA DE SÃO PAULO - HISTORICAL ARCHIVE (per year)
# =============================================================================

def scrape_folha_year(session, keyword: str, year: int) -> List[Dict]:
    """Scrape Folha for a specific year using correct CSS selectors."""
    articles = []

    # Search within a single year
    start_date = f"01/01/{year}"
    end_date = f"31/12/{year}"

    for page in range(0, MAX_PAGES_PER_SEARCH):
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

        # Use correct CSS selector: ol.c-search > li
        search_results = soup.select('ol.c-search > li')

        if not search_results:
            break

        found_articles = 0
        for item in search_results:
            try:
                # Get title from h2.c-headline__title
                title_elem = item.select_one('h2.c-headline__title')
                if not title_elem:
                    continue
                title = title_elem.get_text(strip=True)

                if not title or len(title) < 10:
                    continue

                # Get URL from the content link (not image link)
                link_elem = item.select_one('div.c-headline__content a[href]')
                if not link_elem:
                    link_elem = item.select_one('a[href*="folha.uol.com.br"]')
                if not link_elem:
                    continue

                href = link_elem.get('href', '')
                if 'folha.uol.com.br' not in href:
                    continue

                # Get date from time element or URL
                date_elem = item.select_one('time.c-headline__dateline')
                if date_elem:
                    # Parse date like "6.fev.2026 às 9h29"
                    date_text = date_elem.get('datetime', '') or date_elem.get_text(strip=True)
                    date_str = parse_folha_date(date_text, year)
                else:
                    # Extract from URL pattern /YYYY/MM/
                    date_match = re.search(r'/(\d{4})/(\d{2})/', href)
                    if date_match:
                        date_str = f"{date_match.group(1)}-{date_match.group(2)}-15"
                    else:
                        date_str = f"{year}-06-15"

                articles.append({
                    'title': title[:500],
                    'url': href,
                    'date': date_str,
                    'source': 'folha',
                    'keyword': keyword,
                })
                found_articles += 1

            except Exception:
                continue

        if found_articles == 0:
            break

        time.sleep(REQUEST_DELAY)

        if len(articles) >= MAX_ARTICLES_PER_KEYWORD:
            break

    return articles


def parse_folha_date(date_text: str, default_year: int) -> str:
    """Parse Folha date format like '6.fev.2026 às 9h29' to YYYY-MM-DD."""
    if not date_text:
        return f"{default_year}-06-15"

    # Month mapping
    months = {
        'jan': '01', 'fev': '02', 'mar': '03', 'abr': '04',
        'mai': '05', 'jun': '06', 'jul': '07', 'ago': '08',
        'set': '09', 'out': '10', 'nov': '11', 'dez': '12'
    }

    # Try to parse "DD.MMM.YYYY" format
    match = re.search(r'(\d{1,2})\.(\w{3})\.(\d{4})', date_text.lower())
    if match:
        day = match.group(1).zfill(2)
        month = months.get(match.group(2), '06')
        year = match.group(3)
        return f"{year}-{month}-{day}"

    # Try YYYY-MM-DD format
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    return f"{default_year}-06-15"


def scrape_folha_archive(session, start_year: int, end_year: int,
                          keywords: List[str]) -> List[Dict]:
    """Scrape Folha historical archive year by year."""
    all_articles = []

    print(f"\n  Scraping Folha de S. Paulo ({start_year}-{end_year})...")

    for year in range(start_year, end_year + 1):
        year_articles = []
        print(f"    Year {year}...", end=" ", flush=True)

        for keyword in keywords[:5]:  # Limit keywords per year
            articles = scrape_folha_year(session, keyword, year)
            year_articles.extend(articles)
            time.sleep(REQUEST_DELAY)

        # Remove duplicates within year
        seen = set()
        unique = [a for a in year_articles if not (a['url'] in seen or seen.add(a['url']))]
        all_articles.extend(unique)
        print(f"{len(unique)} articles")

    # Final deduplication
    seen = set()
    unique = [a for a in all_articles if not (a['url'] in seen or seen.add(a['url']))]

    print(f"    Total: {len(unique)} unique articles from Folha")
    return unique


# =============================================================================
# G1 - SELENIUM SEARCH
# =============================================================================

def scrape_g1_selenium(driver, start_year: int, end_year: int,
                        keywords: List[str]) -> List[Dict]:
    """Scrape G1 using Selenium for JavaScript-rendered search."""
    all_articles = []

    print(f"\n  Scraping G1 with Selenium ({start_year}-{end_year})...")

    for year in range(start_year, end_year + 1):
        year_articles = []
        print(f"    Year {year}...", end=" ", flush=True)

        for keyword in keywords[:5]:  # Use same number as Folha
            from_date = f"{year}-01-01T00:00:00-0300"
            to_date = f"{year}-12-31T23:59:59-0300"

            for page in range(1, 11):  # Max 10 pages per keyword
                url = (
                    f"https://g1.globo.com/busca/?"
                    f"q={quote(keyword)}&page={page}&order=recent&species=not%C3%ADcias"
                    f"&from={quote(from_date)}&to={quote(to_date)}"
                )

                html = selenium_fetch(driver, url, wait_time=3)
                if not html:
                    break

                soup = get_soup(html)
                if not soup:
                    break

                # Find search results
                results = soup.select('li.widget--info')
                if not results:
                    # Try finding any links with /noticia/
                    results = soup.find_all('a', href=lambda h: h and '/noticia/' in h)

                if not results:
                    break

                found = 0
                for item in results:
                    try:
                        # Handle both selector types
                        if hasattr(item, 'select_one'):
                            link = item.select_one('a') or item
                        else:
                            link = item

                        href = link.get('href', '')
                        if not href or '/noticia/' not in href:
                            continue

                        # Get title
                        if hasattr(item, 'select_one'):
                            title_elem = item.select_one('div.widget--info__title')
                            title = title_elem.get_text(strip=True) if title_elem else link.get_text(strip=True)
                        else:
                            title = link.get_text(strip=True)

                        if not title or len(title) < 15:
                            continue

                        # Extract date
                        date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', href)
                        if date_match:
                            date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                        else:
                            date_str = f"{year}-06-15"

                        year_articles.append({
                            'title': title[:500],
                            'url': href,
                            'date': date_str,
                            'source': 'g1',
                            'keyword': keyword,
                        })
                        found += 1
                    except Exception:
                        continue

                if found == 0:
                    break

                time.sleep(REQUEST_DELAY)

        # Deduplicate year
        seen = set()
        unique = [a for a in year_articles if not (a['url'] in seen or seen.add(a['url']))]
        all_articles.extend(unique)
        print(f"{len(unique)} articles")

    # Final deduplication
    seen = set()
    unique = [a for a in all_articles if not (a['url'] in seen or seen.add(a['url']))]

    print(f"    Total: {len(unique)} articles from G1")
    return unique


# =============================================================================
# VALOR - SELENIUM SEARCH
# =============================================================================

def scrape_valor_selenium(driver, start_year: int, end_year: int,
                           keywords: List[str]) -> List[Dict]:
    """Scrape Valor Econômico using Selenium."""
    all_articles = []

    print(f"\n  Scraping Valor Econômico with Selenium ({start_year}-{end_year})...")

    for year in range(start_year, end_year + 1):
        year_articles = []
        print(f"    Year {year}...", end=" ", flush=True)

        for keyword in keywords[:5]:  # Use same number as Folha
            from_date = f"{year}-01-01T00:00:00-0300"
            to_date = f"{year}-12-31T23:59:59-0300"

            for page in range(1, 11):  # Max 10 pages per keyword
                url = (
                    f"https://valor.globo.com/busca/?"
                    f"q={quote(keyword)}&page={page}&order=recent"
                    f"&from={quote(from_date)}&to={quote(to_date)}"
                )

                html = selenium_fetch(driver, url, wait_time=3)
                if not html:
                    break

                soup = get_soup(html)
                if not soup:
                    break

                # Find search results
                results = soup.select('li.widget--info')
                if not results:
                    # Try finding any links with valor.globo.com and date pattern
                    results = soup.find_all('a', href=lambda h: h and 'valor.globo.com' in h and re.search(r'/\d{4}/\d{2}/\d{2}/', h))

                if not results:
                    break

                found = 0
                for item in results:
                    try:
                        if hasattr(item, 'select_one'):
                            link = item.select_one('a') or item
                        else:
                            link = item

                        href = link.get('href', '')
                        if not href or 'valor.globo.com' not in href:
                            continue

                        date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', href)
                        if not date_match:
                            continue

                        if hasattr(item, 'select_one'):
                            title_elem = item.select_one('div.widget--info__title')
                            title = title_elem.get_text(strip=True) if title_elem else link.get_text(strip=True)
                        else:
                            title = link.get_text(strip=True)

                        if not title or len(title) < 15:
                            continue

                        date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"

                        year_articles.append({
                            'title': title[:500],
                            'url': href,
                            'date': date_str,
                            'source': 'valor',
                            'keyword': keyword,
                        })
                        found += 1
                    except Exception:
                        continue

                if found == 0:
                    break

                time.sleep(REQUEST_DELAY)

        seen = set()
        unique = [a for a in year_articles if not (a['url'] in seen or seen.add(a['url']))]
        all_articles.extend(unique)
        print(f"{len(unique)} articles")

    seen = set()
    unique = [a for a in all_articles if not (a['url'] in seen or seen.add(a['url']))]

    print(f"    Total: {len(unique)} articles from Valor")
    return unique


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect news from Brazilian financial news sources',
    )
    parser.add_argument('--start-year', type=int, default=2011,
                        help='Start year (default: 2011)')
    parser.add_argument('--end-year', type=int, default=2025,
                        help='End year (default: 2025)')
    parser.add_argument('--keywords', type=str, default=None,
                        help='Comma-separated keywords')
    parser.add_argument('--no-selenium', action='store_true',
                        help='Skip Selenium scraping (G1/Valor search)')
    parser.add_argument('--folha-only', action='store_true',
                        help='Only scrape Folha')

    args = parser.parse_args()

    keywords = DEFAULT_KEYWORDS
    if args.keywords:
        keywords = [k.strip() for k in args.keywords.split(',')]

    print("=" * 60)
    print("NEWS COLLECTION FOR FINANCIAL STRESS INDEX")
    print("=" * 60)
    print(f"\nPeriod: {args.start_year} to {args.end_year}")
    print(f"Keywords: {keywords[:3]}... ({len(keywords)} total)")
    print(f"Output: {DATA_DIR}")

    session = get_session()
    if not session:
        return

    all_articles = []
    driver = None

    try:
        # FOLHA - Historical archive (works without Selenium)
        print("\n" + "-" * 40)
        print("FOLHA DE SÃO PAULO")
        print("-" * 40)
        folha_articles = scrape_folha_archive(session, args.start_year, args.end_year, keywords)
        all_articles.extend(folha_articles)

        if not args.folha_only and not args.no_selenium:
            # G1 and VALOR - Require Selenium
            print("\n" + "-" * 40)
            print("G1 & VALOR (Selenium)")
            print("-" * 40)

            driver = get_selenium_driver()
            if driver:
                g1_articles = scrape_g1_selenium(driver, args.start_year, args.end_year, keywords)
                all_articles.extend(g1_articles)

                valor_articles = scrape_valor_selenium(driver, args.start_year, args.end_year, keywords)
                all_articles.extend(valor_articles)
            else:
                print("\n  Selenium not available, skipping G1/Valor search")
                print("  Install with: pip install selenium webdriver-manager")

    finally:
        if driver:
            driver.quit()

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
    year_counts = df.groupby('year').size()
    for year in sorted(year_counts.index):
        print(f"  - {year}: {year_counts[year]}")

    print(f"\nNext: python scripts/news_fsi.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
