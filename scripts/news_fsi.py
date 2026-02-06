#!/usr/bin/env python3
"""
News-Based Financial Stress Index (FSI)
========================================
Calculates a news-based FSI from Brazilian financial news articles
using the three-way co-occurrence method (Baker et al. 2019 style).

Methodology:
1. Collect news articles from G1, Valor, Folha
2. For each article, check for co-occurrence of:
   - Financial terms (from dictionaries.py)
   - Stress terms (from dictionaries.py)
   - Negative terms (from dictionaries.py)
3. Calculate stress scores based on term density
4. Aggregate to weekly FSI

Usage:
    python scripts/news_fsi.py

    # Use existing news data
    python scripts/news_fsi.py --data data/raw/news_combined.csv

    # Scrape fresh news
    python scripts/news_fsi.py --scrape --pages 10

Output:
    output/news_fsi.csv
    output/news_fsi_weekly.csv
    output/news_fsi_analysis.png
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Handle both script execution and interactive usage
try:
    _SCRIPT_DIR = Path(__file__).parent
    _PROJECT_DIR = _SCRIPT_DIR.parent
except NameError:
    _PROJECT_DIR = Path.cwd()
    _SCRIPT_DIR = _PROJECT_DIR / 'scripts'

sys.path.insert(0, str(_PROJECT_DIR))

from scripts.dictionaries import (
    FINANCIAL_TERMS,
    STRESS_TERMS,
    NEGATIVE_TERMS,
    DICTIONARIES,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = _PROJECT_DIR / 'data' / 'raw'
OUTPUT_DIR = _PROJECT_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum terms required for an article to be considered "stress-related"
MIN_FINANCIAL_TERMS = 1
MIN_STRESS_TERMS = 1
MIN_NEGATIVE_TERMS = 1


# =============================================================================
# TEXT PROCESSING
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for term matching."""
    if not text:
        return ''
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def count_term_matches(text: str, terms: set) -> Tuple[int, List[str]]:
    """
    Count how many dictionary terms appear in text.

    Returns:
        Tuple of (count, list of matched terms)
    """
    text_normalized = normalize_text(text)
    matched = []

    for term in terms:
        term_lower = term.lower()
        # Check for whole word match
        pattern = r'\b' + re.escape(term_lower) + r'\b'
        if re.search(pattern, text_normalized):
            matched.append(term)

    return len(matched), matched


def analyze_article(title: str, summary: str = '') -> Dict:
    """
    Analyze a single article for financial stress content.

    Returns:
        Dictionary with analysis results
    """
    # Combine title and summary for analysis
    full_text = f"{title} {summary}"

    financial_count, financial_terms = count_term_matches(full_text, DICTIONARIES['financial'])
    stress_count, stress_terms = count_term_matches(full_text, DICTIONARIES['stress'])
    negative_count, negative_terms = count_term_matches(full_text, DICTIONARIES['negative'])

    # Check for three-way co-occurrence (Baker et al. style)
    has_cooccurrence = (
        financial_count >= MIN_FINANCIAL_TERMS and
        stress_count >= MIN_STRESS_TERMS and
        negative_count >= MIN_NEGATIVE_TERMS
    )

    # Calculate stress score
    # Score = product of term counts (gives weight to articles with multiple matches)
    if has_cooccurrence:
        stress_score = (financial_count * stress_count * negative_count) ** (1/3)  # Geometric mean
    else:
        stress_score = 0

    return {
        'financial_count': financial_count,
        'stress_count': stress_count,
        'negative_count': negative_count,
        'total_terms': financial_count + stress_count + negative_count,
        'has_cooccurrence': has_cooccurrence,
        'stress_score': stress_score,
        'financial_terms': ', '.join(financial_terms[:5]),  # Top 5
        'stress_terms': ', '.join(stress_terms[:5]),
        'negative_terms': ', '.join(negative_terms[:5]),
    }


# =============================================================================
# FSI CALCULATION
# =============================================================================

def calculate_article_fsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze all articles and calculate per-article stress scores.

    Args:
        df: DataFrame with 'title', 'date', and optionally 'summary' columns

    Returns:
        DataFrame with added analysis columns
    """
    print(f"\n  Analyzing {len(df)} articles...")

    results = []
    for idx, row in df.iterrows():
        title = row.get('title', '')
        summary = row.get('summary', '')

        analysis = analyze_article(title, summary)
        results.append(analysis)

    # Add analysis results to DataFrame
    analysis_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), analysis_df], axis=1)

    # Summary statistics
    stress_articles = df['has_cooccurrence'].sum()
    print(f"    Articles with stress co-occurrence: {stress_articles} ({100*stress_articles/len(df):.1f}%)")

    return df


def aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate article-level data to weekly FSI.

    Args:
        df: DataFrame with article analysis and 'date' column

    Returns:
        DataFrame with weekly FSI
    """
    df = df.copy()

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Create week column (ISO week)
    df['week'] = df['date'].dt.to_period('W').dt.start_time

    # Aggregate by week
    weekly = df.groupby('week').agg({
        'stress_score': ['sum', 'mean', 'count'],
        'has_cooccurrence': 'sum',
        'financial_count': 'sum',
        'stress_count': 'sum',
        'negative_count': 'sum',
        'source': lambda x: ', '.join(x.unique()),
    })

    # Flatten column names
    weekly.columns = [
        'stress_sum', 'stress_mean', 'article_count',
        'stress_articles', 'financial_terms', 'stress_terms',
        'negative_terms', 'sources',
    ]

    weekly = weekly.reset_index()
    weekly = weekly.rename(columns={'week': 'date'})

    # Calculate FSI as normalized stress sum
    # FSI = total stress score / number of articles (density)
    weekly['fsi_raw'] = weekly['stress_sum'] / weekly['article_count']

    # Alternative: proportion of stress articles
    weekly['fsi_proportion'] = weekly['stress_articles'] / weekly['article_count']

    # Standardize to mean=100
    fsi_mean = weekly['fsi_raw'].mean()
    fsi_std = weekly['fsi_raw'].std()
    if fsi_std > 0:
        weekly['news_fsi'] = 100 + 25 * (weekly['fsi_raw'] - fsi_mean) / fsi_std
    else:
        weekly['news_fsi'] = 100

    return weekly


def standardize_fsi(fsi_series: pd.Series, vol_series: pd.Series = None) -> pd.Series:
    """
    Standardize FSI to mean=100 scale.
    If volatility is provided, ensure positive correlation.
    """
    fsi = fsi_series.copy()

    # If volatility provided, check correlation
    if vol_series is not None:
        fsi_aligned = fsi.dropna()
        vol_aligned = vol_series.reindex(fsi_aligned.index).dropna()
        common_idx = fsi_aligned.index.intersection(vol_aligned.index)

        if len(common_idx) > 10:
            corr = fsi_aligned.loc[common_idx].corr(vol_aligned.loc[common_idx])
            if corr < 0:
                print(f"    Note: Flipping FSI sign (correlation with volatility was {corr:.3f})")
                fsi = -fsi

    # Standardize to mean=100, std~25
    fsi_mean = fsi.mean()
    fsi_std = fsi.std()
    if fsi_std > 0:
        fsi_standardized = 100 + 25 * (fsi - fsi_mean) / fsi_std
    else:
        fsi_standardized = pd.Series(100, index=fsi.index)

    return fsi_standardized


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_news_fsi(weekly_df: pd.DataFrame, ibov_df: pd.DataFrame = None,
                   output_path: Path = None) -> None:
    """
    Create visualization of news-based FSI.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  WARNING: matplotlib not installed, skipping visualization")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: News FSI
    ax1 = axes[0]
    ax1.plot(weekly_df['date'], weekly_df['news_fsi'], 'b-', linewidth=1.5, label='News FSI')
    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(weekly_df['date'], 100, weekly_df['news_fsi'],
                     where=weekly_df['news_fsi'] > 100, alpha=0.3, color='red', label='High Stress')
    ax1.fill_between(weekly_df['date'], 100, weekly_df['news_fsi'],
                     where=weekly_df['news_fsi'] <= 100, alpha=0.3, color='green', label='Low Stress')
    ax1.set_ylabel('News FSI')
    ax1.set_title('News-Based Financial Stress Index (Brazil)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Article counts
    ax2 = axes[1]
    ax2.bar(weekly_df['date'], weekly_df['article_count'], alpha=0.5, label='Total Articles', width=5)
    ax2.bar(weekly_df['date'], weekly_df['stress_articles'], alpha=0.7, label='Stress Articles', width=5)
    ax2.set_ylabel('Article Count')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: If IBOVESPA data available, show comparison
    if ibov_df is not None and not ibov_df.empty:
        ax3 = axes[2]

        # Resample IBOV to weekly
        ibov = ibov_df.copy()
        ibov['date'] = pd.to_datetime(ibov['date'])
        ibov = ibov.set_index('date')

        if 'realized_vol_21d' in ibov.columns:
            weekly_vol = ibov['realized_vol_21d'].resample('W').mean()
            ax3.plot(weekly_vol.index, weekly_vol.values, 'orange', linewidth=1.5, label='IBOV Volatility (21d)')
            ax3.set_ylabel('Realized Volatility (%)')
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
    else:
        ax3 = axes[2]
        ax3.text(0.5, 0.5, 'IBOVESPA data not available',
                 ha='center', va='center', transform=ax3.transAxes, fontsize=12, color='gray')
        ax3.set_ylabel('Realized Volatility (%)')

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Calculate news-based Financial Stress Index',
    )
    parser.add_argument('--data', type=str, default=None,
                        help='Path to news CSV file (default: data/raw/news_combined.csv)')
    parser.add_argument('--scrape', action='store_true',
                        help='Scrape fresh news data before analysis')
    parser.add_argument('--pages', type=int, default=5,
                        help='Pages to scrape if --scrape is used (default: 5)')
    parser.add_argument('--keywords', type=str, default=None,
                        help='Comma-separated keywords for scraping')

    args = parser.parse_args()

    print("=" * 60)
    print("NEWS-BASED FINANCIAL STRESS INDEX")
    print("=" * 60)

    # Step 1: Get news data
    print("\n" + "-" * 40)
    print("STEP 1: Loading News Data")
    print("-" * 40)

    if args.scrape:
        # Scrape fresh data
        from scripts.collect_news import collect_all_news

        keywords = None
        if args.keywords:
            keywords = [k.strip() for k in args.keywords.split(',')]

        print(f"  Scraping fresh news data (pages={args.pages})...")
        news_df = collect_all_news(pages=args.pages, keywords=keywords)

        if not news_df.empty:
            news_path = DATA_DIR / 'news_combined.csv'
            news_df.to_csv(news_path, index=False)
            print(f"  Saved {len(news_df)} articles to {news_path}")
    else:
        # Load existing data
        if args.data:
            news_path = Path(args.data)
        else:
            news_path = DATA_DIR / 'news_combined.csv'

        if not news_path.exists():
            print(f"\n  ERROR: News data not found at {news_path}")
            print("  Run: python scripts/collect_news.py first")
            print("  Or use: python scripts/news_fsi.py --scrape")
            return

        news_df = pd.read_csv(news_path)
        print(f"  Loaded {len(news_df)} articles from {news_path}")

    if news_df.empty:
        print("\n  ERROR: No news data available")
        return

    # Step 2: Analyze articles
    print("\n" + "-" * 40)
    print("STEP 2: Analyzing Articles")
    print("-" * 40)

    analyzed_df = calculate_article_fsi(news_df)

    # Save article-level analysis
    article_path = OUTPUT_DIR / 'news_fsi_articles.csv'
    analyzed_df.to_csv(article_path, index=False)
    print(f"  Saved article analysis to {article_path}")

    # Step 3: Aggregate to weekly
    print("\n" + "-" * 40)
    print("STEP 3: Computing Weekly FSI")
    print("-" * 40)

    weekly_df = aggregate_to_weekly(analyzed_df)
    print(f"  Computed FSI for {len(weekly_df)} weeks")

    # Try to load IBOVESPA data for correlation
    ibov_path = DATA_DIR / 'ibovespa_data.csv'
    ibov_df = None
    if ibov_path.exists():
        ibov_df = pd.read_csv(ibov_path)
        print(f"  Loaded IBOVESPA data ({len(ibov_df)} days)")

        # Standardize FSI with volatility check
        if 'realized_vol_21d' in ibov_df.columns:
            ibov_df['date'] = pd.to_datetime(ibov_df['date'])
            vol_series = ibov_df.set_index('date')['realized_vol_21d']
            weekly_df = weekly_df.set_index('date')
            weekly_df['news_fsi'] = standardize_fsi(weekly_df['news_fsi'], vol_series)
            weekly_df = weekly_df.reset_index()

    # Save weekly FSI
    weekly_path = OUTPUT_DIR / 'news_fsi_weekly.csv'
    weekly_df.to_csv(weekly_path, index=False)
    print(f"  Saved weekly FSI to {weekly_path}")

    # Step 4: Visualization
    print("\n" + "-" * 40)
    print("STEP 4: Visualization")
    print("-" * 40)

    plot_path = OUTPUT_DIR / 'news_fsi_analysis.png'
    plot_news_fsi(weekly_df, ibov_df, plot_path)

    # Summary
    print("\n" + "=" * 60)
    print("NEWS FSI CALCULATION COMPLETE")
    print("=" * 60)

    print(f"\n📊 FSI Statistics:")
    print(f"   Period: {weekly_df['date'].min().date()} to {weekly_df['date'].max().date()}")
    print(f"   Weeks: {len(weekly_df)}")
    print(f"   Mean FSI: {weekly_df['news_fsi'].mean():.2f}")
    print(f"   Std FSI: {weekly_df['news_fsi'].std():.2f}")
    print(f"   Max FSI: {weekly_df['news_fsi'].max():.2f} ({weekly_df.loc[weekly_df['news_fsi'].idxmax(), 'date'].date()})")
    print(f"   Min FSI: {weekly_df['news_fsi'].min():.2f} ({weekly_df.loc[weekly_df['news_fsi'].idxmin(), 'date'].date()})")

    if ibov_df is not None and 'realized_vol_21d' in ibov_df.columns:
        # Calculate correlation with volatility
        ibov_weekly = ibov_df.set_index('date')['realized_vol_21d'].resample('W').mean()
        fsi_for_corr = weekly_df.set_index('date')['news_fsi']
        common_idx = fsi_for_corr.index.intersection(ibov_weekly.index)
        if len(common_idx) > 5:
            corr = fsi_for_corr.loc[common_idx].corr(ibov_weekly.loc[common_idx])
            print(f"\n   Correlation with IBOV volatility: {corr:.3f}")

    print(f"\n📁 Output files:")
    print(f"   - {article_path}")
    print(f"   - {weekly_path}")
    print(f"   - {plot_path}")

    print("=" * 60)


if __name__ == '__main__':
    main()
