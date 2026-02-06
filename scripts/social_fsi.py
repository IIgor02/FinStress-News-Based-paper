#!/usr/bin/env python3
"""
Social Media-Based Financial Stress Index
==========================================
Calculates FSI from Twitter and Reddit posts using sentiment analysis
and keyword frequency.

Methodology:
1. Load social media posts (Twitter, Reddit)
2. Analyze sentiment and stress-related keywords
3. Aggregate to weekly FSI

Usage:
    python scripts/social_fsi.py

Output:
    output/social_fsi_weekly.csv
    output/social_fsi_analysis.png
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    _SCRIPT_DIR = Path(__file__).parent
    _PROJECT_DIR = _SCRIPT_DIR.parent
except NameError:
    _PROJECT_DIR = Path.cwd()
    _SCRIPT_DIR = _PROJECT_DIR / 'scripts'

sys.path.insert(0, str(_PROJECT_DIR))

from scripts.dictionaries import (
    STRESS_TERMS,
    NEGATIVE_TERMS,
    FINANCIAL_TERMS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = _PROJECT_DIR / 'data' / 'raw'
OUTPUT_DIR = _PROJECT_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Stress keywords specific to social media
SOCIAL_STRESS_KEYWORDS = [
    # Crisis
    'crise', 'crash', 'colapso', 'desastre', 'caos',
    'pânico', 'panico', 'medo', 'terror', 'apocalipse',

    # Market stress
    'queda', 'despencou', 'derreteu', 'afundou', 'perdeu',
    'prejuízo', 'prejuizo', 'perda', 'vermelho', 'negativo',

    # Economic stress
    'recessão', 'recessao', 'depressão', 'depressao',
    'inflação', 'inflacao', 'hiperinflação', 'estagflação',
    'desemprego', 'demissão', 'demissao', 'layoff',

    # Financial stress
    'falência', 'falencia', 'quebra', 'calote', 'default',
    'dívida', 'divida', 'endividado', 'insolvente',

    # Sentiment
    'preocupado', 'preocupação', 'ansiedade', 'nervoso',
    'pessimista', 'negativo', 'ruim', 'péssimo', 'horrível',

    # Reddit/Twitter specific
    'loss porn', 'perdi tudo', 'fudeu', 'ferrou', 'lascou',
    'rip', 'f', 'gg', 'acabou',
]

POSITIVE_KEYWORDS = [
    'alta', 'subiu', 'ganho', 'lucro', 'verde', 'positivo',
    'otimista', 'recuperação', 'crescimento', 'boom',
    'bull', 'moon', 'foguete', 'to the moon',
]


# =============================================================================
# SENTIMENT ANALYSIS
# =============================================================================

def count_keywords(text: str, keywords: List[str]) -> int:
    """Count keyword occurrences in text."""
    if not text:
        return 0

    text_lower = text.lower()
    count = 0

    for keyword in keywords:
        # Word boundary matching
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        matches = re.findall(pattern, text_lower)
        count += len(matches)

    return count


def analyze_post_sentiment(text: str) -> Dict:
    """
    Analyze sentiment of a social media post.

    Returns dict with:
    - stress_score: Number of stress keywords
    - positive_score: Number of positive keywords
    - net_sentiment: positive - stress (higher = more positive)
    - stress_ratio: stress / (stress + positive + 1)
    """
    stress_count = count_keywords(text, SOCIAL_STRESS_KEYWORDS)
    positive_count = count_keywords(text, POSITIVE_KEYWORDS)

    # Also check dictionaries
    stress_count += count_keywords(text, list(STRESS_TERMS)[:50])
    stress_count += count_keywords(text, list(NEGATIVE_TERMS)[:50])

    net_sentiment = positive_count - stress_count
    stress_ratio = stress_count / (stress_count + positive_count + 1)

    return {
        'stress_score': stress_count,
        'positive_score': positive_count,
        'net_sentiment': net_sentiment,
        'stress_ratio': stress_ratio,
    }


def analyze_posts(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze all posts for sentiment."""
    print(f"  Analyzing {len(df)} posts...")

    results = []
    for _, row in df.iterrows():
        text = str(row.get('text', ''))
        sentiment = analyze_post_sentiment(text)
        results.append(sentiment)

    sentiment_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)

    # Summary
    stress_posts = (df['stress_score'] > 0).sum()
    print(f"    Posts with stress keywords: {stress_posts} ({100*stress_posts/len(df):.1f}%)")

    return df


# =============================================================================
# FSI CALCULATION
# =============================================================================

def calculate_social_fsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate weekly FSI from social media data.

    FSI = average stress_ratio * volume weight
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Aggregate by week
    df['week'] = df['date'].dt.to_period('W').dt.start_time

    weekly = df.groupby('week').agg({
        'stress_score': ['sum', 'mean'],
        'positive_score': ['sum', 'mean'],
        'stress_ratio': 'mean',
        'net_sentiment': 'mean',
        'text': 'count',  # Post volume
    })

    # Flatten columns
    weekly.columns = [
        'stress_sum', 'stress_mean',
        'positive_sum', 'positive_mean',
        'stress_ratio', 'net_sentiment',
        'post_count',
    ]

    weekly = weekly.reset_index()
    weekly = weekly.rename(columns={'week': 'date'})

    # Calculate FSI
    # Higher stress_ratio = higher FSI
    # Also weight by post volume (more posts during stress periods)
    weekly['volume_weight'] = weekly['post_count'] / weekly['post_count'].mean()
    weekly['fsi_raw'] = weekly['stress_ratio'] * (1 + np.log1p(weekly['volume_weight']))

    # Standardize to mean=100
    fsi_mean = weekly['fsi_raw'].mean()
    fsi_std = weekly['fsi_raw'].std()

    if fsi_std > 0:
        weekly['social_fsi'] = 100 + 25 * (weekly['fsi_raw'] - fsi_mean) / fsi_std
    else:
        weekly['social_fsi'] = 100

    return weekly


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_social_fsi(weekly_df: pd.DataFrame, output_path: Path = None):
    """Create visualization of social media FSI."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  WARNING: matplotlib not installed")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Social FSI
    ax1 = axes[0]
    ax1.plot(weekly_df['date'], weekly_df['social_fsi'], 'purple', linewidth=1.5)
    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(weekly_df['date'], 100, weekly_df['social_fsi'],
                     where=weekly_df['social_fsi'] > 100, alpha=0.3, color='red')
    ax1.fill_between(weekly_df['date'], 100, weekly_df['social_fsi'],
                     where=weekly_df['social_fsi'] <= 100, alpha=0.3, color='green')
    ax1.set_ylabel('Social FSI')
    ax1.set_title('Social Media Financial Stress Index')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Post volume
    ax2 = axes[1]
    ax2.bar(weekly_df['date'], weekly_df['post_count'], alpha=0.6, width=5)
    ax2.set_ylabel('Post Count')
    ax2.set_title('Social Media Activity Volume')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Sentiment breakdown
    ax3 = axes[2]
    ax3.plot(weekly_df['date'], weekly_df['stress_mean'], 'r-', label='Stress Score', alpha=0.7)
    ax3.plot(weekly_df['date'], weekly_df['positive_mean'], 'g-', label='Positive Score', alpha=0.7)
    ax3.set_ylabel('Average Score')
    ax3.set_title('Sentiment Breakdown')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Calculate FSI from social media data',
    )
    parser.add_argument('--data', type=str, default=None,
                        help='Path to social media CSV')

    args = parser.parse_args()

    print("=" * 60)
    print("SOCIAL MEDIA FINANCIAL STRESS INDEX")
    print("=" * 60)

    # Load data
    print("\n" + "-" * 40)
    print("Loading Data")
    print("-" * 40)

    if args.data:
        data_path = Path(args.data)
    else:
        data_path = DATA_DIR / 'social_combined.csv'

    if not data_path.exists():
        # Try individual files
        dfs = []
        for source in ['twitter_posts.csv', 'reddit_posts.csv']:
            path = DATA_DIR / source
            if path.exists():
                dfs.append(pd.read_csv(path))
                print(f"  Loaded {path.name}")

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            print(f"  ERROR: No social media data found")
            print(f"  Run: python scripts/collect_social.py first")
            return
    else:
        df = pd.read_csv(data_path)
        print(f"  Loaded {len(df)} posts from {data_path}")

    if df.empty:
        print("  ERROR: No data to analyze")
        return

    # Analyze sentiment
    print("\n" + "-" * 40)
    print("Analyzing Sentiment")
    print("-" * 40)

    analyzed_df = analyze_posts(df)

    # Calculate weekly FSI
    print("\n" + "-" * 40)
    print("Calculating Weekly FSI")
    print("-" * 40)

    weekly_df = calculate_social_fsi(analyzed_df)
    print(f"  Computed FSI for {len(weekly_df)} weeks")

    # Save
    weekly_path = OUTPUT_DIR / 'social_fsi_weekly.csv'
    weekly_df.to_csv(weekly_path, index=False)
    print(f"  Saved to {weekly_path}")

    # Visualize
    print("\n" + "-" * 40)
    print("Visualization")
    print("-" * 40)

    plot_path = OUTPUT_DIR / 'social_fsi_analysis.png'
    plot_social_fsi(weekly_df, plot_path)

    # Summary
    print("\n" + "=" * 60)
    print("SOCIAL FSI COMPLETE")
    print("=" * 60)

    print(f"\nFSI Statistics:")
    print(f"  Period: {weekly_df['date'].min().date()} to {weekly_df['date'].max().date()}")
    print(f"  Weeks: {len(weekly_df)}")
    print(f"  Mean: {weekly_df['social_fsi'].mean():.2f}")
    print(f"  Std: {weekly_df['social_fsi'].std():.2f}")
    print(f"  Max: {weekly_df['social_fsi'].max():.2f}")
    print(f"  Min: {weekly_df['social_fsi'].min():.2f}")

    print(f"\nOutput files:")
    print(f"  - {weekly_path}")
    print(f"  - {plot_path}")

    print("=" * 60)


if __name__ == '__main__':
    main()
