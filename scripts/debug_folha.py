#!/usr/bin/env python3
"""
Debug script to find the correct way to scrape Folha.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import re

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
}

def test_folha_urls():
    """Test different URL formats for Folha search."""
    session = requests.Session()
    session.headers.update(HEADERS)

    print("=" * 60)
    print("TESTING FOLHA URL FORMATS")
    print("=" * 60)

    # Different URL formats to try
    test_urls = [
        # Format 1: URL-encoded dates (my current approach)
        "https://search.folha.uol.com.br/search?q=crise&periodo=personalizado&sd=01%2F01%2F2020&ed=31%2F12%2F2020&site=todos&sr=1",

        # Format 2: Plain dates (from the working repository)
        "https://search.folha.uol.com.br/search?q=crise&periodo=personalizado&sd=01/01/2020&ed=31/12/2020&site=todos&sr=1",

        # Format 3: Without date filter
        "https://search.folha.uol.com.br/search?q=crise+financeira&site=todos&sr=1",

        # Format 4: With site=online instead of todos
        "https://search.folha.uol.com.br/search?q=economia&site=online&sr=1",
    ]

    for i, url in enumerate(test_urls, 1):
        print(f"\n--- Test {i}: {url[:80]}...")

        try:
            response = session.get(url, timeout=30)
            print(f"Status: {response.status_code}")
            print(f"Final URL: {response.url[:100]}")

            soup = BeautifulSoup(response.text, 'html.parser')

            # Count all folha links
            all_folha_links = soup.find_all('a', href=lambda h: h and 'folha.uol.com.br' in h)
            print(f"Total folha links: {len(all_folha_links)}")

            # Count links with date patterns
            date_pattern_links = []
            for link in all_folha_links:
                href = link.get('href', '')
                if re.search(r'/\d{4}/\d{2}/', href):
                    date_pattern_links.append(link)

            print(f"Links with /YYYY/MM/ pattern: {len(date_pattern_links)}")

            # Show first 5 examples
            print("\nFirst 5 article links:")
            for link in date_pattern_links[:5]:
                href = link.get('href', '')
                title = link.get_text(strip=True)[:60]
                print(f"  URL: {href[:80]}")
                print(f"  Title: {title}")
                print()

        except Exception as e:
            print(f"Error: {e}")


def test_folha_selectors():
    """Test what CSS selectors work for Folha search results."""
    session = requests.Session()
    session.headers.update(HEADERS)

    print("\n" + "=" * 60)
    print("TESTING FOLHA CSS SELECTORS")
    print("=" * 60)

    url = "https://search.folha.uol.com.br/search?q=economia&site=todos&sr=1"

    response = session.get(url, timeout=30)
    soup = BeautifulSoup(response.text, 'html.parser')

    print(f"\nURL: {url}")
    print(f"Status: {response.status_code}")

    # Try various selectors
    selectors = [
        'ol.c-search > li',
        'ol.c-search li',
        'li.c-search__result',
        'div.c-search__results li',
        'ol li',
        'div.sr a',  # sr = search results?
    ]

    print("\nTesting selectors:")
    for sel in selectors:
        try:
            items = soup.select(sel)
            print(f"  {sel}: {len(items)} matches")
            if items and len(items) > 0:
                # Show first item's structure
                first = items[0]
                links = first.find_all('a', href=True)
                print(f"    First item has {len(links)} links")
                for link in links[:2]:
                    print(f"      -> {link.get('href', '')[:60]}")
        except Exception as e:
            print(f"  {sel}: Error - {e}")

    # Try to find the actual search result container
    print("\n\nLooking for search result containers:")

    # Find any ol or ul that might contain results
    for tag in soup.find_all(['ol', 'ul']):
        classes = tag.get('class', [])
        if classes:
            li_count = len(tag.find_all('li', recursive=False))
            if li_count > 3:
                print(f"  <{tag.name} class='{' '.join(classes)}'> has {li_count} direct <li> children")

                # Check first li
                first_li = tag.find('li')
                if first_li:
                    links = first_li.find_all('a', href=True)
                    print(f"    First <li> has {len(links)} links")


def analyze_search_result_structure():
    """Deep dive into the HTML structure of search results."""
    session = requests.Session()
    session.headers.update(HEADERS)

    print("\n" + "=" * 60)
    print("ANALYZING SEARCH RESULT HTML STRUCTURE")
    print("=" * 60)

    url = "https://search.folha.uol.com.br/search?q=ibovespa&site=todos&sr=1"

    response = session.get(url, timeout=30)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find ol.c-search which had 25 matches
    ol = soup.select_one('ol.c-search')
    if ol:
        print(f"\nFound ol.c-search")
        lis = ol.find_all('li', recursive=False)
        print(f"Direct <li> children: {len(lis)}")

        if lis:
            print("\n--- First <li> structure ---")
            first_li = lis[0]
            print(first_li.prettify()[:2000])

            print("\n--- Extracting data from first <li> ---")

            # Find all links
            links = first_li.find_all('a', href=True)
            for link in links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                if 'folha.uol.com.br' in href and len(text) > 10:
                    print(f"Link: {href}")
                    print(f"Text: {text[:100]}")
                    print()
    else:
        print("ol.c-search not found!")

        # Try to find any container with multiple article links
        print("\nSearching for containers with article links...")
        for container in soup.find_all(['div', 'ol', 'ul', 'section']):
            article_links = container.find_all('a', href=lambda h: h and 'folha.uol.com.br' in h and '/20' in h)
            if len(article_links) >= 5:
                classes = container.get('class', [])
                tag = container.name
                print(f"<{tag} class='{' '.join(classes) if classes else 'none'}'> has {len(article_links)} article links")


if __name__ == '__main__':
    test_folha_urls()
    test_folha_selectors()
    analyze_search_result_structure()

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)
