#!/usr/bin/env python3
"""
Debug script to investigate news scraping issues.
Run this to see what HTML is being returned from each source.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
}

def test_folha():
    """Test Folha search."""
    print("\n" + "=" * 60)
    print("TESTING FOLHA DE SÃO PAULO")
    print("=" * 60)

    # Test different URL formats
    urls = [
        "https://search.folha.uol.com.br/search?q=economia&site=online&periodo=personalizado&sd=01/01/2020&ed=31/12/2020&sr=1",
        "https://search.folha.uol.com.br/search?q=economia&site=todos",
        "https://www.folha.uol.com.br/mercado/",
    ]

    session = requests.Session()
    session.headers.update(HEADERS)

    for url in urls:
        print(f"\n--- Testing: {url[:80]}...")
        try:
            response = session.get(url, timeout=30)
            print(f"Status: {response.status_code}")
            print(f"URL after redirects: {response.url}")
            print(f"Content length: {len(response.text)} chars")

            soup = BeautifulSoup(response.text, 'html.parser')

            # Try various selectors
            selectors = [
                'ol.c-search li',
                'div.c-search',
                'li.c-search__result',
                'div.search-result',
                'article',
                'div.c-headline',
                'a[href*="/mercado/"]',
                'a[href*="folha.uol.com.br"]',
            ]

            print("\nTrying selectors:")
            for sel in selectors:
                items = soup.select(sel)
                print(f"  {sel}: {len(items)} matches")
                if items and len(items) < 5:
                    for item in items[:3]:
                        text = item.get_text(strip=True)[:100]
                        print(f"    -> {text}")

            # Show page title
            title = soup.find('title')
            print(f"\nPage title: {title.get_text() if title else 'None'}")

            # Check for JavaScript requirement
            if 'javascript' in response.text.lower() and len(soup.find_all('a')) < 10:
                print("\n⚠️  Page might require JavaScript rendering!")

        except Exception as e:
            print(f"Error: {e}")


def test_valor():
    """Test Valor search."""
    print("\n" + "=" * 60)
    print("TESTING VALOR ECONÔMICO")
    print("=" * 60)

    urls = [
        "https://valor.globo.com/busca/?q=economia&page=1&order=recent",
        "https://valor.globo.com/busca/?q=economia",
        "https://valor.globo.com/",
    ]

    session = requests.Session()
    session.headers.update(HEADERS)

    for url in urls:
        print(f"\n--- Testing: {url[:80]}...")
        try:
            response = session.get(url, timeout=30)
            print(f"Status: {response.status_code}")
            print(f"URL after redirects: {response.url}")
            print(f"Content length: {len(response.text)} chars")

            soup = BeautifulSoup(response.text, 'html.parser')

            selectors = [
                'li.widget--info',
                'div.widget--info',
                'article',
                'div.feed-post',
                'a[href*="valor.globo.com"]',
                'a[href*="/2"]',  # URLs with dates like /2024/
            ]

            print("\nTrying selectors:")
            for sel in selectors:
                items = soup.select(sel)
                print(f"  {sel}: {len(items)} matches")

            # Find all links with dates in URL
            date_links = soup.find_all('a', href=lambda h: h and '/202' in h)
            print(f"\n  Links with /202 in URL: {len(date_links)}")
            for link in date_links[:5]:
                print(f"    -> {link.get('href', '')[:80]}")

            title = soup.find('title')
            print(f"\nPage title: {title.get_text() if title else 'None'}")

            if 'javascript' in response.text.lower() and len(soup.find_all('a')) < 10:
                print("\n⚠️  Page might require JavaScript rendering!")

        except Exception as e:
            print(f"Error: {e}")


def test_g1():
    """Test G1 search."""
    print("\n" + "=" * 60)
    print("TESTING G1")
    print("=" * 60)

    urls = [
        "https://g1.globo.com/busca/?q=economia&page=1&order=recent",
        "https://g1.globo.com/busca/?q=economia",
        "https://g1.globo.com/economia/",
    ]

    session = requests.Session()
    session.headers.update(HEADERS)

    for url in urls:
        print(f"\n--- Testing: {url[:80]}...")
        try:
            response = session.get(url, timeout=30)
            print(f"Status: {response.status_code}")
            print(f"URL after redirects: {response.url}")
            print(f"Content length: {len(response.text)} chars")

            soup = BeautifulSoup(response.text, 'html.parser')

            selectors = [
                'li.widget--info',
                'div.widget--info',
                'article',
                'div.feed-post',
                'div.bastian-feed-item',
                'a[href*="g1.globo.com"]',
                'a[href*="/noticia/"]',
            ]

            print("\nTrying selectors:")
            for sel in selectors:
                items = soup.select(sel)
                print(f"  {sel}: {len(items)} matches")

            # Find all links with dates in URL
            date_links = soup.find_all('a', href=lambda h: h and '/noticia/' in h)
            print(f"\n  Links with /noticia/ in URL: {len(date_links)}")
            for link in date_links[:5]:
                print(f"    -> {link.get('href', '')[:80]}")

            title = soup.find('title')
            print(f"\nPage title: {title.get_text() if title else 'None'}")

            if 'javascript' in response.text.lower() and len(soup.find_all('a')) < 10:
                print("\n⚠️  Page might require JavaScript rendering!")

        except Exception as e:
            print(f"Error: {e}")


def show_raw_html_sample(url: str):
    """Show a sample of raw HTML for manual inspection."""
    print("\n" + "=" * 60)
    print(f"RAW HTML SAMPLE: {url[:50]}...")
    print("=" * 60)

    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        response = session.get(url, timeout=30)
        html = response.text

        # Show first 3000 chars
        print("\nFirst 3000 characters of HTML:")
        print("-" * 40)
        print(html[:3000])
        print("-" * 40)
        print(f"\n... ({len(html)} total characters)")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    print("NEWS SCRAPING DEBUG TOOL")
    print("This will test each news source and show what's being returned.")

    test_folha()
    test_valor()
    test_g1()

    # Uncomment to see raw HTML:
    # show_raw_html_sample("https://valor.globo.com/busca/?q=economia")

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)
    print("\nIf you see '⚠️ Page might require JavaScript rendering!'")
    print("it means the site uses JavaScript to load content dynamically.")
    print("In that case, we need to use Selenium or Playwright instead.")
