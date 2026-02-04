#!/usr/bin/env python3
"""
Financial Stress Index Calculator
=================================
Script 3 of 3: FSI Calculation, Validation, and Visualization

This script:
1. Loads collected data (Twitter, Reddit, IBOVESPA)
2. Classifies posts using three-way co-occurrence
3. Calculates daily and monthly FSI with weighting
4. Validates against IBOVESPA volatility
5. Generates visualizations

Usage:
    # Run with default settings
    python scripts/run_fsi.py

    # Run with custom parameters
    python scripts/run_fsi.py --min-posts 10 --twitter-weight 0.7 --reddit-weight 0.3

    # Run without generating plots (faster)
    python scripts/run_fsi.py --no-plots

Output:
    output/results/fsi_twitter_daily.csv
    output/results/fsi_reddit_daily.csv
    output/results/fsi_combined_daily.csv
    output/results/fsi_monthly.csv
    output/plots/fsi_timeseries.png
    output/plots/fsi_vs_volatility.png
    output/plots/fsi_scatter.png
    output/plots/fsi_monthly_comparison.png
"""

import argparse
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from unidecode import unidecode

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.dictionaries import (
    DICTIONARIES,
    FINANCIAL_TERMS,
    STRESS_TERMS,
    NEGATIVE_TERMS,
    CRISIS_EPISODES,
    get_dictionary_stats,
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# CONFIGURATION / TUNABLE PARAMETERS
# =============================================================================

class FSIConfig:
    """Configuration parameters for FSI calculation."""

    # Minimum posts per day to calculate valid FSI
    MIN_POSTS_PER_DAY: int = 5

    # Platform weighting for combined FSI
    TWITTER_WEIGHT: float = 0.6
    REDDIT_WEIGHT: float = 0.4

    # Standardization
    TARGET_MEAN: float = 100.0
    TARGET_STD: Optional[float] = None  # None = preserve original std

    # Volatility column to use for validation
    VOLATILITY_COLUMN: str = 'realized_vol_30d'

    # Target correlation range (from Baker et al. 2019)
    TARGET_CORRELATION_MIN: float = 0.60
    TARGET_CORRELATION_MAX: float = 0.90

    # Rolling window for smoothing (0 = no smoothing)
    SMOOTHING_WINDOW: int = 7

    # File paths
    DATA_DIR: Path = Path(__file__).parent.parent / 'data' / 'raw'
    OUTPUT_DIR: Path = Path(__file__).parent.parent / 'output' / 'results'
    PLOTS_DIR: Path = Path(__file__).parent.parent / 'output' / 'plots'


# Global config instance
CONFIG = FSIConfig()


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

# Portuguese stopwords
STOPWORDS = {
    'a', 'ao', 'aos', 'as', 'da', 'das', 'de', 'do', 'dos', 'e', 'em',
    'na', 'nas', 'no', 'nos', 'o', 'os', 'para', 'por', 'que', 'um', 'uma',
}


def preprocess_text(text: str) -> str:
    """Preprocess text for dictionary matching."""
    if not isinstance(text, str):
        return ''

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Remove mentions
    text = re.sub(r'@\w+', '', text)

    # Convert hashtags to words
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove RT prefix
    text = re.sub(r'^rt\s+', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_tokens(text: str) -> Set[str]:
    """Extract tokens from preprocessed text."""
    if not text:
        return set()

    words = text.split()
    tokens = set(words)

    # Also add accent-normalized versions
    tokens.update(unidecode(w) for w in words)

    return tokens


# =============================================================================
# CLASSIFICATION (Three-Way Co-occurrence)
# =============================================================================

def find_matches(
    tokens: Set[str],
    text: str,
    dictionary: Set[str],
) -> List[str]:
    """Find dictionary terms in tokens/text."""
    matches = []

    for term in dictionary:
        if ' ' in term:
            # Multi-word term: check in original text
            if term in text or unidecode(term) in text:
                matches.append(term)
        else:
            # Single word: check in tokens
            if term in tokens or unidecode(term) in tokens:
                matches.append(term)

    return matches


def classify_post(text: str) -> Dict:
    """
    Classify a single post using three-way co-occurrence.

    Returns:
        Dictionary with classification results
    """
    processed = preprocess_text(text)
    tokens = extract_tokens(processed)

    # Find matches in each category
    financial_matches = find_matches(tokens, processed, DICTIONARIES['financial'])
    stress_matches = find_matches(tokens, processed, DICTIONARIES['stress'])
    negative_matches = find_matches(tokens, processed, DICTIONARIES['negative'])

    is_financial = len(financial_matches) > 0
    is_stress = (
        len(financial_matches) > 0 and
        len(stress_matches) > 0 and
        len(negative_matches) > 0
    )

    return {
        'is_financial': is_financial,
        'is_stress': is_stress,
        'n_financial': len(financial_matches),
        'n_stress': len(stress_matches),
        'n_negative': len(negative_matches),
        'financial_terms': financial_matches[:5],  # Limit for memory
        'stress_terms': stress_matches[:5],
        'negative_terms': negative_matches[:5],
    }


def classify_dataframe(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """Classify all posts in a DataFrame."""
    print(f"    Classifying {len(df):,} posts...")

    df = df.copy()
    results = df[text_column].apply(classify_post)

    df['is_financial'] = results.apply(lambda x: x['is_financial'])
    df['is_stress'] = results.apply(lambda x: x['is_stress'])
    df['n_financial'] = results.apply(lambda x: x['n_financial'])
    df['n_stress'] = results.apply(lambda x: x['n_stress'])
    df['n_negative'] = results.apply(lambda x: x['n_negative'])

    return df


# =============================================================================
# FSI CALCULATION
# =============================================================================

def calculate_weight(value: float, platform: str = 'twitter') -> float:
    """Calculate log-based weight for engagement metric."""
    return np.log1p(max(0, value))


def calculate_daily_fsi(
    df: pd.DataFrame,
    date_column: str = 'date',
    weight_column: Optional[str] = None,
    platform: str = 'twitter',
) -> pd.DataFrame:
    """Calculate daily FSI from classified data."""

    df = df.copy()
    df['date_only'] = pd.to_datetime(df[date_column]).dt.date

    # Calculate weights
    if weight_column and weight_column in df.columns:
        df['weight'] = df[weight_column].apply(lambda x: calculate_weight(x, platform))
    else:
        df['weight'] = 1.0

    # Filter to financial posts
    financial_df = df[df['is_financial']].copy()

    if len(financial_df) == 0:
        return pd.DataFrame(columns=['date', 'fsi_raw', 'n_financial', 'n_stress'])

    # Group by date
    daily_stats = []
    for date, group in financial_df.groupby('date_only'):
        total_weight = group['weight'].sum()
        stress_weight = group[group['is_stress']]['weight'].sum()

        fsi_raw = (stress_weight / total_weight * 100) if total_weight > 0 else np.nan

        daily_stats.append({
            'date': pd.Timestamp(date),
            'fsi_raw': fsi_raw,
            'n_financial': len(group),
            'n_stress': group['is_stress'].sum(),
            'total_weight': total_weight,
            'stress_weight': stress_weight,
        })

    daily_df = pd.DataFrame(daily_stats)
    daily_df = daily_df.sort_values('date').reset_index(drop=True)

    # Filter days with insufficient data
    mask = daily_df['n_financial'] >= CONFIG.MIN_POSTS_PER_DAY
    daily_df.loc[~mask, 'fsi_raw'] = np.nan

    return daily_df


def standardize_fsi(fsi_series: pd.Series) -> pd.Series:
    """Standardize FSI to target mean."""
    valid = fsi_series.dropna()

    if len(valid) == 0:
        return fsi_series

    mean = valid.mean()
    std = valid.std()

    if std == 0:
        return pd.Series(CONFIG.TARGET_MEAN, index=fsi_series.index)

    # Z-score and rescale
    standardized = (fsi_series - mean) / std
    standardized = standardized * std + CONFIG.TARGET_MEAN

    return standardized


def calculate_monthly_fsi(daily_df: pd.DataFrame, fsi_column: str = 'fsi') -> pd.DataFrame:
    """Aggregate daily FSI to monthly."""
    df = daily_df.copy()
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')

    monthly = df.groupby('month').agg({
        fsi_column: 'mean',
        'n_financial': 'sum',
        'n_stress': 'sum',
    }).reset_index()

    monthly['month'] = monthly['month'].dt.to_timestamp()
    monthly = monthly.rename(columns={fsi_column: 'fsi'})

    return monthly


def combine_platforms(
    twitter_daily: pd.DataFrame,
    reddit_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Combine FSI from Twitter and Reddit."""
    combined = pd.merge(
        twitter_daily[['date', 'fsi_raw', 'n_financial', 'n_stress']],
        reddit_daily[['date', 'fsi_raw', 'n_financial', 'n_stress']],
        on='date',
        how='outer',
        suffixes=('_twitter', '_reddit'),
    )

    # Weighted combination
    tw = CONFIG.TWITTER_WEIGHT
    rd = CONFIG.REDDIT_WEIGHT

    # Handle cases where one platform is missing
    twitter_present = combined['fsi_raw_twitter'].notna()
    reddit_present = combined['fsi_raw_reddit'].notna()
    both_present = twitter_present & reddit_present

    combined['fsi_raw'] = np.nan
    combined.loc[both_present, 'fsi_raw'] = (
        combined.loc[both_present, 'fsi_raw_twitter'] * tw +
        combined.loc[both_present, 'fsi_raw_reddit'] * rd
    )
    combined.loc[twitter_present & ~reddit_present, 'fsi_raw'] = combined.loc[twitter_present & ~reddit_present, 'fsi_raw_twitter']
    combined.loc[reddit_present & ~twitter_present, 'fsi_raw'] = combined.loc[reddit_present & ~twitter_present, 'fsi_raw_reddit']

    # Sum posts
    combined['n_financial'] = combined['n_financial_twitter'].fillna(0) + combined['n_financial_reddit'].fillna(0)
    combined['n_stress'] = combined['n_stress_twitter'].fillna(0) + combined['n_stress_reddit'].fillna(0)

    combined = combined.sort_values('date').reset_index(drop=True)

    return combined


# =============================================================================
# VALIDATION
# =============================================================================

def calculate_correlation(fsi: pd.Series, vol: pd.Series) -> Dict:
    """Calculate correlation metrics."""
    aligned = pd.DataFrame({'fsi': fsi, 'vol': vol}).dropna()

    if len(aligned) < 10:
        return {'pearson': np.nan, 'spearman': np.nan, 'n': len(aligned)}

    pearson_r, pearson_p = stats.pearsonr(aligned['fsi'], aligned['vol'])
    spearman_r, spearman_p = stats.spearmanr(aligned['fsi'], aligned['vol'])

    return {
        'pearson': pearson_r,
        'pearson_pvalue': pearson_p,
        'spearman': spearman_r,
        'spearman_pvalue': spearman_p,
        'n': len(aligned),
    }


def validate_fsi(
    daily_fsi: pd.DataFrame,
    market_data: pd.DataFrame,
    monthly_fsi: Optional[pd.DataFrame] = None,
) -> Dict:
    """Validate FSI against market volatility."""
    results = {}

    # Merge for daily correlation
    daily_merged = pd.merge(
        daily_fsi[['date', 'fsi']],
        market_data[['date', CONFIG.VOLATILITY_COLUMN]],
        on='date',
        how='inner',
    )

    results['daily'] = calculate_correlation(
        daily_merged['fsi'],
        daily_merged[CONFIG.VOLATILITY_COLUMN],
    )

    # Monthly correlation
    if monthly_fsi is not None:
        market_data['month'] = pd.to_datetime(market_data['date']).dt.to_period('M')
        monthly_vol = market_data.groupby('month')[CONFIG.VOLATILITY_COLUMN].mean().reset_index()
        monthly_vol['month'] = monthly_vol['month'].dt.to_timestamp()

        monthly_merged = pd.merge(
            monthly_fsi.rename(columns={'month': 'date'}),
            monthly_vol.rename(columns={'month': 'date'}),
            on='date',
            how='inner',
        )

        results['monthly'] = calculate_correlation(
            monthly_merged['fsi'],
            monthly_merged[CONFIG.VOLATILITY_COLUMN],
        )

    # FSI statistics
    results['fsi_stats'] = {
        'mean': daily_fsi['fsi'].mean(),
        'std': daily_fsi['fsi'].std(),
        'min': daily_fsi['fsi'].min(),
        'max': daily_fsi['fsi'].max(),
        'n_days': len(daily_fsi),
    }

    return results


def print_validation_report(results: Dict):
    """Print formatted validation report."""
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    print("\nCorrelation with Market Volatility:")
    print("-" * 40)

    if 'daily' in results:
        d = results['daily']
        in_range = CONFIG.TARGET_CORRELATION_MIN <= abs(d['pearson']) <= CONFIG.TARGET_CORRELATION_MAX
        status = "✓" if in_range else "○"
        print(f"  Daily (n={d['n']}):")
        print(f"    Pearson:  r = {d['pearson']:.3f} (p = {d['pearson_pvalue']:.4f}) {status}")
        print(f"    Spearman: ρ = {d['spearman']:.3f} (p = {d['spearman_pvalue']:.4f})")

    if 'monthly' in results:
        m = results['monthly']
        in_range = CONFIG.TARGET_CORRELATION_MIN <= abs(m['pearson']) <= CONFIG.TARGET_CORRELATION_MAX
        status = "✓" if in_range else "○"
        print(f"\n  Monthly (n={m['n']}):")
        print(f"    Pearson:  r = {m['pearson']:.3f} (p = {m['pearson_pvalue']:.4f}) {status}")
        print(f"    Spearman: ρ = {m['spearman']:.3f} (p = {m['spearman_pvalue']:.4f})")

    print(f"\n  Target Range: {CONFIG.TARGET_CORRELATION_MIN:.2f} - {CONFIG.TARGET_CORRELATION_MAX:.2f}")

    if 'fsi_stats' in results:
        s = results['fsi_stats']
        print(f"\nFSI Statistics:")
        print("-" * 40)
        print(f"  Mean:  {s['mean']:.2f}")
        print(f"  Std:   {s['std']:.2f}")
        print(f"  Range: [{s['min']:.2f}, {s['max']:.2f}]")
        print(f"  Days:  {s['n_days']}")

    print("=" * 60)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_fsi_timeseries(
    daily_fsi: pd.DataFrame,
    save_path: str,
    title: str = 'Financial Stress Index (Combined)',
):
    """Plot FSI time series with crisis markers."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot FSI
    ax.plot(daily_fsi['date'], daily_fsi['fsi'], color='#2E86AB', linewidth=1, label='FSI')

    # Add rolling average
    if len(daily_fsi) > 30 and CONFIG.SMOOTHING_WINDOW > 0:
        rolling = daily_fsi['fsi'].rolling(window=CONFIG.SMOOTHING_WINDOW, center=True).mean()
        ax.plot(daily_fsi['date'], rolling, color='#E94F37', linewidth=2, label=f'{CONFIG.SMOOTHING_WINDOW}-day MA', alpha=0.8)

    # Mark crisis periods
    for (start, end), label in CRISIS_EPISODES.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        if daily_fsi['date'].min() <= end_dt and daily_fsi['date'].max() >= start_dt:
            ax.axvspan(start_dt, end_dt, alpha=0.15, color='red')
            mid = start_dt + (end_dt - start_dt) / 2
            ax.text(mid, ax.get_ylim()[1] * 0.97, label, ha='center', va='top', fontsize=7, rotation=45)

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('FSI', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_fsi_vs_volatility(
    daily_fsi: pd.DataFrame,
    market_data: pd.DataFrame,
    save_path: str,
):
    """Plot FSI overlaid with market volatility."""
    merged = pd.merge(
        daily_fsi[['date', 'fsi']],
        market_data[['date', CONFIG.VOLATILITY_COLUMN]],
        on='date',
        how='inner',
    )

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # FSI on left axis
    color1 = '#2E86AB'
    ax1.plot(merged['date'], merged['fsi'], color=color1, linewidth=1.5, label='FSI')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('FSI', color=color1, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Volatility on right axis
    ax2 = ax1.twinx()
    color2 = '#E94F37'
    ax2.plot(merged['date'], merged[CONFIG.VOLATILITY_COLUMN], color=color2, linewidth=1.5, label='Volatility', alpha=0.7)
    ax2.set_ylabel('Realized Volatility (%)', color=color2, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Correlation in title
    corr = merged['fsi'].corr(merged[CONFIG.VOLATILITY_COLUMN])
    ax1.set_title(f'FSI vs Market Volatility\n(Correlation: r = {corr:.3f})', fontsize=13, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_scatter(
    daily_fsi: pd.DataFrame,
    market_data: pd.DataFrame,
    save_path: str,
):
    """Plot scatter correlation."""
    merged = pd.merge(
        daily_fsi[['date', 'fsi']],
        market_data[['date', CONFIG.VOLATILITY_COLUMN]],
        on='date',
    ).dropna()

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(merged[CONFIG.VOLATILITY_COLUMN], merged['fsi'], alpha=0.4, s=15, color='#2E86AB')

    # Regression line
    z = np.polyfit(merged[CONFIG.VOLATILITY_COLUMN], merged['fsi'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged[CONFIG.VOLATILITY_COLUMN].min(), merged[CONFIG.VOLATILITY_COLUMN].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2, label='Regression')

    corr = merged['fsi'].corr(merged[CONFIG.VOLATILITY_COLUMN])

    ax.set_xlabel('Market Volatility (%)', fontsize=11)
    ax.set_ylabel('FSI', fontsize=11)
    ax.set_title(f'FSI vs Volatility Scatter\n(r = {corr:.3f})', fontsize=13, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_monthly_comparison(
    twitter_monthly: pd.DataFrame,
    reddit_monthly: pd.DataFrame,
    combined_monthly: pd.DataFrame,
    save_path: str,
):
    """Plot monthly FSI comparison across platforms."""
    fig, ax = plt.subplots(figsize=(14, 6))

    if twitter_monthly is not None and len(twitter_monthly) > 0:
        ax.plot(twitter_monthly['month'], twitter_monthly['fsi'], marker='o', markersize=4, label='Twitter', alpha=0.8)

    if reddit_monthly is not None and len(reddit_monthly) > 0:
        ax.plot(reddit_monthly['month'], reddit_monthly['fsi'], marker='s', markersize=4, label='Reddit', alpha=0.8)

    if combined_monthly is not None and len(combined_monthly) > 0:
        ax.plot(combined_monthly['month'], combined_monthly['fsi'], marker='D', markersize=4, linewidth=2, label='Combined', color='black')

    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('FSI', fontsize=11)
    ax.set_title('Monthly FSI by Platform', fontsize=13, fontweight='bold')
    ax.legend()

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all data files."""
    twitter_path = CONFIG.DATA_DIR / 'twitter_data.csv'
    reddit_path = CONFIG.DATA_DIR / 'reddit_data.csv'
    market_path = CONFIG.DATA_DIR / 'ibovespa_data.csv'

    twitter_df = pd.DataFrame()
    reddit_df = pd.DataFrame()
    market_df = pd.DataFrame()

    if twitter_path.exists():
        twitter_df = pd.read_csv(twitter_path)
        twitter_df['date'] = pd.to_datetime(twitter_df['date'])
        print(f"  Twitter: {len(twitter_df):,} posts loaded")
    else:
        print(f"  Twitter: File not found ({twitter_path})")

    if reddit_path.exists():
        reddit_df = pd.read_csv(reddit_path)
        reddit_df['date'] = pd.to_datetime(reddit_df['date'])
        print(f"  Reddit: {len(reddit_df):,} posts loaded")
    else:
        print(f"  Reddit: File not found ({reddit_path})")

    if market_path.exists():
        market_df = pd.read_csv(market_path)
        market_df['date'] = pd.to_datetime(market_df['date'])
        print(f"  IBOVESPA: {len(market_df):,} trading days loaded")
    else:
        print(f"  IBOVESPA: File not found ({market_path})")

    return twitter_df, reddit_df, market_df


def save_results(
    twitter_daily: pd.DataFrame,
    reddit_daily: pd.DataFrame,
    combined_daily: pd.DataFrame,
    twitter_monthly: pd.DataFrame,
    reddit_monthly: pd.DataFrame,
    combined_monthly: pd.DataFrame,
):
    """Save all results to CSV files."""
    CONFIG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if twitter_daily is not None and len(twitter_daily) > 0:
        twitter_daily.to_csv(CONFIG.OUTPUT_DIR / 'fsi_twitter_daily.csv', index=False)
        print(f"  Saved: {CONFIG.OUTPUT_DIR / 'fsi_twitter_daily.csv'}")

    if reddit_daily is not None and len(reddit_daily) > 0:
        reddit_daily.to_csv(CONFIG.OUTPUT_DIR / 'fsi_reddit_daily.csv', index=False)
        print(f"  Saved: {CONFIG.OUTPUT_DIR / 'fsi_reddit_daily.csv'}")

    if combined_daily is not None and len(combined_daily) > 0:
        combined_daily.to_csv(CONFIG.OUTPUT_DIR / 'fsi_combined_daily.csv', index=False)
        print(f"  Saved: {CONFIG.OUTPUT_DIR / 'fsi_combined_daily.csv'}")

    # Combined monthly
    monthly_df = pd.DataFrame()
    if combined_monthly is not None:
        monthly_df['month'] = combined_monthly['month']
        monthly_df['fsi_combined'] = combined_monthly['fsi']
    if twitter_monthly is not None:
        if 'month' not in monthly_df.columns:
            monthly_df['month'] = twitter_monthly['month']
        monthly_df['fsi_twitter'] = twitter_monthly['fsi']
    if reddit_monthly is not None:
        if 'month' not in monthly_df.columns:
            monthly_df['month'] = reddit_monthly['month']
        monthly_df['fsi_reddit'] = reddit_monthly['fsi']

    if len(monthly_df) > 0:
        monthly_df.to_csv(CONFIG.OUTPUT_DIR / 'fsi_monthly.csv', index=False)
        print(f"  Saved: {CONFIG.OUTPUT_DIR / 'fsi_monthly.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Financial Stress Index',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--min-posts', type=int, default=5, help='Minimum posts per day')
    parser.add_argument('--twitter-weight', type=float, default=0.6, help='Twitter weight in combined FSI')
    parser.add_argument('--reddit-weight', type=float, default=0.4, help='Reddit weight in combined FSI')
    parser.add_argument('--smoothing', type=int, default=7, help='Smoothing window (0=none)')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')

    args = parser.parse_args()

    # Update config
    CONFIG.MIN_POSTS_PER_DAY = args.min_posts
    CONFIG.TWITTER_WEIGHT = args.twitter_weight
    CONFIG.REDDIT_WEIGHT = args.reddit_weight
    CONFIG.SMOOTHING_WINDOW = args.smoothing

    # Header
    print("=" * 60)
    print("FINANCIAL STRESS INDEX - CALCULATION")
    print("Baker et al. (2019) Methodology")
    print("=" * 60)

    # Dictionary stats
    stats = get_dictionary_stats()
    print(f"\n📚 Dictionary: {stats['financial_terms']} financial, {stats['stress_terms']} stress, {stats['negative_terms']} negative terms")

    # Configuration
    print(f"\n⚙️  Configuration:")
    print(f"   Min posts/day: {CONFIG.MIN_POSTS_PER_DAY}")
    print(f"   Twitter weight: {CONFIG.TWITTER_WEIGHT}")
    print(f"   Reddit weight: {CONFIG.REDDIT_WEIGHT}")
    print(f"   Smoothing window: {CONFIG.SMOOTHING_WINDOW}")

    # Load data
    print(f"\n📁 Loading data...")
    twitter_df, reddit_df, market_df = load_data()

    if twitter_df.empty and reddit_df.empty:
        print("\n❌ No social media data found. Run collect_data.py first:")
        print("   python scripts/collect_data.py --synthetic")
        return

    # Process Twitter
    twitter_daily = None
    twitter_monthly = None
    if not twitter_df.empty:
        print(f"\n🐦 Processing Twitter...")
        twitter_classified = classify_dataframe(twitter_df)
        n_fin = twitter_classified['is_financial'].sum()
        n_stress = twitter_classified['is_stress'].sum()
        print(f"    Financial: {n_fin:,} ({n_fin/len(twitter_classified)*100:.1f}%)")
        print(f"    Stress: {n_stress:,} ({n_stress/n_fin*100:.1f}% of financial)")

        twitter_daily = calculate_daily_fsi(twitter_classified, weight_column='followers', platform='twitter')
        twitter_daily['fsi'] = standardize_fsi(twitter_daily['fsi_raw'])
        twitter_monthly = calculate_monthly_fsi(twitter_daily)

    # Process Reddit
    reddit_daily = None
    reddit_monthly = None
    if not reddit_df.empty:
        print(f"\n👽 Processing Reddit...")
        reddit_classified = classify_dataframe(reddit_df)
        n_fin = reddit_classified['is_financial'].sum()
        n_stress = reddit_classified['is_stress'].sum()
        print(f"    Financial: {n_fin:,} ({n_fin/len(reddit_classified)*100:.1f}%)")
        print(f"    Stress: {n_stress:,} ({n_stress/n_fin*100:.1f}% of financial)")

        reddit_daily = calculate_daily_fsi(reddit_classified, weight_column='upvotes', platform='reddit')
        reddit_daily['fsi'] = standardize_fsi(reddit_daily['fsi_raw'])
        reddit_monthly = calculate_monthly_fsi(reddit_daily)

    # Combine platforms
    combined_daily = None
    combined_monthly = None

    if twitter_daily is not None and reddit_daily is not None:
        print(f"\n🔗 Combining platforms...")
        combined_daily = combine_platforms(twitter_daily, reddit_daily)
        combined_daily['fsi'] = standardize_fsi(combined_daily['fsi_raw'])
        combined_monthly = calculate_monthly_fsi(combined_daily)
    elif twitter_daily is not None:
        combined_daily = twitter_daily
        combined_monthly = twitter_monthly
    elif reddit_daily is not None:
        combined_daily = reddit_daily
        combined_monthly = reddit_monthly

    # Validate
    if combined_daily is not None and not market_df.empty:
        print(f"\n📊 Validating against IBOVESPA volatility...")
        validation = validate_fsi(combined_daily, market_df, combined_monthly)
        print_validation_report(validation)

    # Save results
    print(f"\n💾 Saving results...")
    save_results(twitter_daily, reddit_daily, combined_daily, twitter_monthly, reddit_monthly, combined_monthly)

    # Generate plots
    if not args.no_plots and combined_daily is not None:
        print(f"\n📈 Generating visualizations...")
        CONFIG.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        plot_fsi_timeseries(combined_daily, CONFIG.PLOTS_DIR / 'fsi_timeseries.png')

        if not market_df.empty:
            plot_fsi_vs_volatility(combined_daily, market_df, CONFIG.PLOTS_DIR / 'fsi_vs_volatility.png')
            plot_scatter(combined_daily, market_df, CONFIG.PLOTS_DIR / 'fsi_scatter.png')

        if twitter_monthly is not None or reddit_monthly is not None:
            plot_monthly_comparison(twitter_monthly, reddit_monthly, combined_monthly, CONFIG.PLOTS_DIR / 'fsi_monthly_comparison.png')

    # Summary
    print("\n" + "=" * 60)
    print("✅ FSI CALCULATION COMPLETE")
    print("=" * 60)
    print(f"\n   Results: {CONFIG.OUTPUT_DIR}/")
    print(f"   Plots:   {CONFIG.PLOTS_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
