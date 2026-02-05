#!/usr/bin/env python3
"""
Financial Stress Index Calculator
=================================
Script 3 of 3: FSI Calculation, Validation, and Visualization

Calculates FSI from two sources:
1. Twitter - Daily FSI using three-way co-occurrence (Baker et al. 2019)
2. Google Trends - Weekly FSSVI using search volume (Da et al. 2011)

Combined Index = weighted average of Twitter + Google Trends FSI

Usage:
    # Run with default settings
    python scripts/run_fsi.py

    # Custom parameters
    python scripts/run_fsi.py --twitter-weight 0.5 --gt-weight 0.5 --aggregation average

    # Skip plots (faster)
    python scripts/run_fsi.py --no-plots

Output:
    output/results/fsi_twitter_daily.csv
    output/results/fsi_google_trends_weekly.csv
    output/results/fsi_combined_weekly.csv
    output/results/fsi_monthly.csv
    output/plots/fsi_timeseries.png
    output/plots/fsi_vs_volatility.png
    output/plots/fsi_components.png
"""

import argparse
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from unidecode import unidecode

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.dictionaries import (
    DICTIONARIES,
    FINANCIAL_TERMS,
    STRESS_TERMS,
    NEGATIVE_TERMS,
    STRESS_QUERIES_GT,
    QUERY_WEIGHTS,
    CRISIS_EPISODES,
    get_dictionary_stats,
    get_query_weight,
)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# CONFIGURATION
# =============================================================================

class FSIConfig:
    """Configuration for FSI calculation."""

    # Minimum data requirements
    MIN_POSTS_PER_DAY: int = 5
    MIN_QUERIES_FOR_GT: int = 3  # Minimum queries with data

    # Platform weights for combined FSI
    TWITTER_WEIGHT: float = 0.5
    GT_WEIGHT: float = 0.5

    # Standardization
    TARGET_MEAN: float = 100.0

    # Volatility column for validation
    VOLATILITY_COLUMN: str = 'realized_vol_30d'

    # Target correlation range
    TARGET_CORRELATION_MIN: float = 0.60
    TARGET_CORRELATION_MAX: float = 0.90

    # Smoothing window
    SMOOTHING_WINDOW: int = 4  # weeks for GT

    # Google Trends aggregation method
    GT_AGGREGATION: str = 'weighted_average'  # 'average', 'weighted_average', 'pca'

    # File paths
    DATA_DIR: Path = Path(__file__).parent.parent / 'data' / 'raw'
    OUTPUT_DIR: Path = Path(__file__).parent.parent / 'output' / 'results'
    PLOTS_DIR: Path = Path(__file__).parent.parent / 'output' / 'plots'


CONFIG = FSIConfig()


# =============================================================================
# TEXT PREPROCESSING (Twitter)
# =============================================================================

def preprocess_text(text: str) -> str:
    """Preprocess text for dictionary matching."""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'^rt\s+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_tokens(text: str) -> Set[str]:
    """Extract tokens from text."""
    if not text:
        return set()
    words = text.split()
    tokens = set(words)
    tokens.update(unidecode(w) for w in words)
    return tokens


# =============================================================================
# TWITTER FSI CALCULATION (Three-Way Co-occurrence)
# =============================================================================

def find_matches(tokens: Set[str], text: str, dictionary: Set[str]) -> List[str]:
    """Find dictionary terms in tokens/text."""
    matches = []
    for term in dictionary:
        if ' ' in term:
            if term in text or unidecode(term) in text:
                matches.append(term)
        else:
            if term in tokens or unidecode(term) in tokens:
                matches.append(term)
    return matches


def classify_post(text: str) -> Dict:
    """Classify a post using three-way co-occurrence."""
    processed = preprocess_text(text)
    tokens = extract_tokens(processed)

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
    }


def classify_twitter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Classify all Twitter posts."""
    print(f"    Classifying {len(df):,} tweets...")
    df = df.copy()
    results = df['text'].apply(classify_post)
    df['is_financial'] = results.apply(lambda x: x['is_financial'])
    df['is_stress'] = results.apply(lambda x: x['is_stress'])
    return df


def calculate_twitter_daily_fsi(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily Twitter FSI."""
    df = df.copy()
    df['date_only'] = pd.to_datetime(df['date']).dt.date

    # Calculate weights
    if 'followers' in df.columns:
        df['weight'] = np.log1p(df['followers'].clip(lower=0))
    else:
        df['weight'] = 1.0

    financial_df = df[df['is_financial']].copy()

    if len(financial_df) == 0:
        return pd.DataFrame(columns=['date', 'fsi_raw', 'n_financial', 'n_stress'])

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
        })

    daily_df = pd.DataFrame(daily_stats)
    daily_df = daily_df.sort_values('date').reset_index(drop=True)

    # Filter days with insufficient data
    mask = daily_df['n_financial'] >= CONFIG.MIN_POSTS_PER_DAY
    daily_df.loc[~mask, 'fsi_raw'] = np.nan

    return daily_df


# =============================================================================
# GOOGLE TRENDS FSI CALCULATION (Search Volume Index)
# =============================================================================

def calculate_google_trends_fsi(
    gt_df: pd.DataFrame,
    method: str = 'weighted_average',
) -> pd.DataFrame:
    """
    Calculate Financial Stress Search Volume Index (FSSVI) from Google Trends.

    Args:
        gt_df: DataFrame with Google Trends data (date + query columns)
        method: 'average', 'weighted_average', or 'pca'

    Returns:
        DataFrame with weekly FSSVI
    """
    df = gt_df.copy()

    # Identify query columns (exclude date and source)
    query_cols = [c for c in df.columns if c not in ['date', 'source', 'isPartial']]

    if len(query_cols) < CONFIG.MIN_QUERIES_FOR_GT:
        print(f"    ⚠ Only {len(query_cols)} queries found (need {CONFIG.MIN_QUERIES_FOR_GT})")
        return pd.DataFrame(columns=['date', 'fssvi_raw'])

    print(f"    Processing {len(query_cols)} Google Trends queries...")

    if method == 'average':
        # Simple average of all query SVIs
        df['fssvi_raw'] = df[query_cols].mean(axis=1)

    elif method == 'weighted_average':
        # Weighted average using query weights from dictionaries
        weights = {q: get_query_weight(q) for q in query_cols}
        total_weight = sum(weights.values())

        weighted_sum = sum(df[q] * weights.get(q, 1.0) for q in query_cols)
        df['fssvi_raw'] = weighted_sum / total_weight

    elif method == 'pca':
        # Extract first principal component
        from sklearn.decomposition import PCA

        query_data = df[query_cols].dropna()
        if len(query_data) < 10:
            print("    ⚠ Insufficient data for PCA, using weighted average")
            return calculate_google_trends_fsi(gt_df, 'weighted_average')

        # Standardize
        query_std = (query_data - query_data.mean()) / query_data.std()

        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(query_std).flatten()

        print(f"    PCA variance explained: {pca.explained_variance_ratio_[0]:.1%}")

        # Rescale to 0-100 range
        pc1_scaled = (pc1 - pc1.min()) / (pc1.max() - pc1.min()) * 100
        df.loc[query_data.index, 'fssvi_raw'] = pc1_scaled

    else:
        raise ValueError(f"Unknown method: {method}")

    # Select output columns
    result = df[['date', 'fssvi_raw']].copy()
    result['date'] = pd.to_datetime(result['date'])
    result = result.sort_values('date').reset_index(drop=True)

    return result


# =============================================================================
# STANDARDIZATION
# =============================================================================

def standardize_fsi(fsi_series: pd.Series) -> pd.Series:
    """Standardize FSI to target mean."""
    valid = fsi_series.dropna()
    if len(valid) == 0:
        return fsi_series

    mean = valid.mean()
    std = valid.std()

    if std == 0:
        return pd.Series(CONFIG.TARGET_MEAN, index=fsi_series.index)

    standardized = (fsi_series - mean) / std
    standardized = standardized * std + CONFIG.TARGET_MEAN

    return standardized


# =============================================================================
# COMBINING TWITTER + GOOGLE TRENDS
# =============================================================================

def combine_twitter_and_google_trends(
    twitter_daily: pd.DataFrame,
    gt_weekly: pd.DataFrame,
    twitter_weight: float = 0.5,
    gt_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Combine Twitter FSI (daily) with Google Trends FSI (weekly).

    Args:
        twitter_daily: Daily Twitter FSI with 'date' and 'fsi' columns
        gt_weekly: Weekly Google Trends FSI with 'date' and 'fssvi' columns
        twitter_weight: Weight for Twitter (0-1)
        gt_weight: Weight for Google Trends (0-1)

    Returns:
        Combined weekly FSI DataFrame
    """
    # Aggregate Twitter to weekly
    twitter_daily = twitter_daily.copy()
    twitter_daily['week'] = pd.to_datetime(twitter_daily['date']).dt.to_period('W')

    twitter_weekly = twitter_daily.groupby('week').agg({
        'fsi': 'mean',
        'n_financial': 'sum',
        'n_stress': 'sum',
    }).reset_index()
    twitter_weekly['date'] = twitter_weekly['week'].dt.to_timestamp()
    twitter_weekly = twitter_weekly.drop('week', axis=1)

    # Merge
    gt_weekly = gt_weekly.copy()
    gt_weekly['date'] = pd.to_datetime(gt_weekly['date'])

    combined = pd.merge(
        twitter_weekly[['date', 'fsi']].rename(columns={'fsi': 'fsi_twitter'}),
        gt_weekly[['date', 'fssvi']].rename(columns={'fssvi': 'fsi_gt'}),
        on='date',
        how='outer',
    )

    # Calculate combined FSI
    twitter_present = combined['fsi_twitter'].notna()
    gt_present = combined['fsi_gt'].notna()
    both_present = twitter_present & gt_present

    combined['fsi_combined'] = np.nan
    combined.loc[both_present, 'fsi_combined'] = (
        combined.loc[both_present, 'fsi_twitter'] * twitter_weight +
        combined.loc[both_present, 'fsi_gt'] * gt_weight
    )

    # If only one source available, use it
    combined.loc[twitter_present & ~gt_present, 'fsi_combined'] = (
        combined.loc[twitter_present & ~gt_present, 'fsi_twitter']
    )
    combined.loc[gt_present & ~twitter_present, 'fsi_combined'] = (
        combined.loc[gt_present & ~twitter_present, 'fsi_gt']
    )

    combined = combined.sort_values('date').reset_index(drop=True)

    return combined


def calculate_monthly_fsi(weekly_df: pd.DataFrame, fsi_col: str = 'fsi') -> pd.DataFrame:
    """Aggregate weekly FSI to monthly."""
    df = weekly_df.copy()
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')

    monthly = df.groupby('month').agg({
        fsi_col: 'mean',
    }).reset_index()

    monthly['month'] = monthly['month'].dt.to_timestamp()
    monthly = monthly.rename(columns={fsi_col: 'fsi'})

    return monthly


# =============================================================================
# VALIDATION
# =============================================================================

def calculate_correlation(fsi: pd.Series, vol: pd.Series) -> Dict:
    """Calculate correlation metrics."""
    aligned = pd.DataFrame({'fsi': fsi, 'vol': vol}).dropna()

    if len(aligned) < 10:
        return {
            'pearson': np.nan,
            'pearson_pvalue': np.nan,
            'spearman': np.nan,
            'spearman_pvalue': np.nan,
            'n': len(aligned),
        }

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
    weekly_fsi: pd.DataFrame,
    market_data: pd.DataFrame,
    fsi_col: str = 'fsi',
) -> Dict:
    """Validate FSI against market volatility."""
    results = {}

    # Aggregate market data to weekly
    market_data = market_data.copy()
    market_data['week'] = pd.to_datetime(market_data['date']).dt.to_period('W')
    market_weekly = market_data.groupby('week')[CONFIG.VOLATILITY_COLUMN].mean().reset_index()

    # Add week period to FSI data for matching
    weekly_fsi = weekly_fsi.copy()
    weekly_fsi['week'] = pd.to_datetime(weekly_fsi['date']).dt.to_period('W')

    # Merge on week period instead of exact date
    merged = pd.merge(
        weekly_fsi[['week', fsi_col]],
        market_weekly[['week', CONFIG.VOLATILITY_COLUMN]],
        on='week',
        how='inner',
    )

    results['weekly'] = calculate_correlation(
        merged[fsi_col],
        merged[CONFIG.VOLATILITY_COLUMN],
    )

    # Monthly
    monthly_fsi = calculate_monthly_fsi(weekly_fsi, fsi_col)
    market_data['month'] = pd.to_datetime(market_data['date']).dt.to_period('M')
    market_monthly = market_data.groupby('month')[CONFIG.VOLATILITY_COLUMN].mean().reset_index()
    market_monthly['month'] = market_monthly['month'].dt.to_timestamp()

    monthly_merged = pd.merge(
        monthly_fsi.rename(columns={'month': 'date'}),
        market_monthly.rename(columns={'month': 'date'}),
        on='date',
        how='inner',
    )

    results['monthly'] = calculate_correlation(
        monthly_merged['fsi'],
        monthly_merged[CONFIG.VOLATILITY_COLUMN],
    )

    # FSI stats
    results['fsi_stats'] = {
        'mean': weekly_fsi[fsi_col].mean(),
        'std': weekly_fsi[fsi_col].std(),
        'min': weekly_fsi[fsi_col].min(),
        'max': weekly_fsi[fsi_col].max(),
        'n_weeks': len(weekly_fsi),
    }

    return results


def print_validation_report(results: Dict, source: str = 'Combined'):
    """Print validation report."""
    print(f"\n{'=' * 60}")
    print(f"VALIDATION: {source} FSI")
    print('=' * 60)

    print("\nCorrelation with IBOVESPA Volatility:")
    print("-" * 40)

    if 'weekly' in results:
        w = results['weekly']
        if w['n'] < 10 or np.isnan(w['pearson']):
            print(f"  Weekly (n={w['n']}): Insufficient data for correlation")
        else:
            in_range = CONFIG.TARGET_CORRELATION_MIN <= abs(w['pearson']) <= CONFIG.TARGET_CORRELATION_MAX
            status = "✓" if in_range else "○"
            print(f"  Weekly (n={w['n']}):")
            print(f"    Pearson:  r = {w['pearson']:.3f} (p = {w['pearson_pvalue']:.4f}) {status}")
            print(f"    Spearman: ρ = {w['spearman']:.3f} (p = {w['spearman_pvalue']:.4f})")

    if 'monthly' in results:
        m = results['monthly']
        if m['n'] < 10 or np.isnan(m['pearson']):
            print(f"\n  Monthly (n={m['n']}): Insufficient data for correlation")
        else:
            in_range = CONFIG.TARGET_CORRELATION_MIN <= abs(m['pearson']) <= CONFIG.TARGET_CORRELATION_MAX
            status = "✓" if in_range else "○"
            print(f"\n  Monthly (n={m['n']}):")
            print(f"    Pearson:  r = {m['pearson']:.3f} (p = {m['pearson_pvalue']:.4f}) {status}")
            print(f"    Spearman: ρ = {m['spearman']:.3f} (p = {m['spearman_pvalue']:.4f})")

    print(f"\n  Target: {CONFIG.TARGET_CORRELATION_MIN:.2f} - {CONFIG.TARGET_CORRELATION_MAX:.2f}")

    if 'fsi_stats' in results:
        s = results['fsi_stats']
        print(f"\nFSI Statistics:")
        print(f"  Mean:  {s['mean']:.2f}")
        print(f"  Std:   {s['std']:.2f}")
        print(f"  Range: [{s['min']:.2f}, {s['max']:.2f}]")
        print(f"  Weeks: {s['n_weeks']}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_fsi_timeseries(
    weekly_fsi: pd.DataFrame,
    fsi_col: str = 'fsi',
    title: str = 'Financial Stress Index',
    save_path: str = None,
):
    """Plot FSI time series with crisis markers."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(weekly_fsi['date'], weekly_fsi[fsi_col], color='#2E86AB', linewidth=1.5, label='FSI')

    # Smoothing
    if len(weekly_fsi) > CONFIG.SMOOTHING_WINDOW:
        rolling = weekly_fsi[fsi_col].rolling(window=CONFIG.SMOOTHING_WINDOW, center=True).mean()
        ax.plot(weekly_fsi['date'], rolling, color='#E94F37', linewidth=2,
                label=f'{CONFIG.SMOOTHING_WINDOW}-week MA', alpha=0.8)

    # Crisis markers
    for (start, end), label in CRISIS_EPISODES.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if weekly_fsi['date'].min() <= end_dt and weekly_fsi['date'].max() >= start_dt:
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
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_fsi_vs_volatility(
    weekly_fsi: pd.DataFrame,
    market_data: pd.DataFrame,
    fsi_col: str = 'fsi',
    save_path: str = None,
):
    """Plot FSI vs market volatility."""
    # Aggregate market to weekly
    market_data = market_data.copy()
    market_data['week'] = pd.to_datetime(market_data['date']).dt.to_period('W')
    market_weekly = market_data.groupby('week')[CONFIG.VOLATILITY_COLUMN].mean().reset_index()
    market_weekly['date'] = market_weekly['week'].dt.to_timestamp()

    merged = pd.merge(
        weekly_fsi[['date', fsi_col]],
        market_weekly[['date', CONFIG.VOLATILITY_COLUMN]],
        on='date',
        how='inner',
    )

    fig, ax1 = plt.subplots(figsize=(14, 6))

    color1 = '#2E86AB'
    ax1.plot(merged['date'], merged[fsi_col], color=color1, linewidth=1.5, label='FSI')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('FSI', color=color1, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = '#E94F37'
    ax2.plot(merged['date'], merged[CONFIG.VOLATILITY_COLUMN], color=color2,
             linewidth=1.5, label='Volatility', alpha=0.7)
    ax2.set_ylabel('Volatility (%)', color=color2, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color2)

    corr = merged[fsi_col].corr(merged[CONFIG.VOLATILITY_COLUMN])
    ax1.set_title(f'FSI vs Market Volatility (r = {corr:.3f})', fontsize=13, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_fsi_components(
    combined_df: pd.DataFrame,
    save_path: str = None,
):
    """Plot Twitter vs Google Trends FSI components."""
    fig, ax = plt.subplots(figsize=(14, 6))

    if 'fsi_twitter' in combined_df.columns:
        ax.plot(combined_df['date'], combined_df['fsi_twitter'],
                color='#1DA1F2', linewidth=1.5, label='Twitter FSI', alpha=0.8)

    if 'fsi_gt' in combined_df.columns:
        ax.plot(combined_df['date'], combined_df['fsi_gt'],
                color='#4285F4', linewidth=1.5, label='Google Trends FSI', alpha=0.8)

    if 'fsi_combined' in combined_df.columns:
        ax.plot(combined_df['date'], combined_df['fsi_combined'],
                color='#E94F37', linewidth=2, label='Combined FSI')

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('FSI', fontsize=11)
    ax.set_title('FSI Components: Twitter vs Google Trends', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all data files."""
    twitter_path = CONFIG.DATA_DIR / 'twitter_data.csv'
    gt_path = CONFIG.DATA_DIR / 'google_trends_data.csv'
    market_path = CONFIG.DATA_DIR / 'ibovespa_data.csv'

    twitter_df = pd.DataFrame()
    gt_df = pd.DataFrame()
    market_df = pd.DataFrame()

    if twitter_path.exists():
        twitter_df = pd.read_csv(twitter_path)
        twitter_df['date'] = pd.to_datetime(twitter_df['date'])
        print(f"  Twitter: {len(twitter_df):,} posts loaded")
    else:
        print(f"  Twitter: File not found")

    if gt_path.exists():
        gt_df = pd.read_csv(gt_path)
        gt_df['date'] = pd.to_datetime(gt_df['date'])
        query_cols = [c for c in gt_df.columns if c not in ['date', 'source', 'isPartial']]
        print(f"  Google Trends: {len(gt_df)} weeks, {len(query_cols)} queries")
    else:
        print(f"  Google Trends: File not found")

    if market_path.exists():
        market_df = pd.read_csv(market_path)
        market_df['date'] = pd.to_datetime(market_df['date'])
        print(f"  IBOVESPA: {len(market_df):,} trading days")
    else:
        print(f"  IBOVESPA: File not found")

    return twitter_df, gt_df, market_df


def save_results(
    twitter_daily: pd.DataFrame,
    gt_weekly: pd.DataFrame,
    combined_weekly: pd.DataFrame,
    monthly_fsi: pd.DataFrame,
):
    """Save results to CSV files."""
    CONFIG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if twitter_daily is not None and len(twitter_daily) > 0:
        twitter_daily.to_csv(CONFIG.OUTPUT_DIR / 'fsi_twitter_daily.csv', index=False)
        print(f"  Saved: fsi_twitter_daily.csv")

    if gt_weekly is not None and len(gt_weekly) > 0:
        gt_weekly.to_csv(CONFIG.OUTPUT_DIR / 'fsi_google_trends_weekly.csv', index=False)
        print(f"  Saved: fsi_google_trends_weekly.csv")

    if combined_weekly is not None and len(combined_weekly) > 0:
        combined_weekly.to_csv(CONFIG.OUTPUT_DIR / 'fsi_combined_weekly.csv', index=False)
        print(f"  Saved: fsi_combined_weekly.csv")

    if monthly_fsi is not None and len(monthly_fsi) > 0:
        monthly_fsi.to_csv(CONFIG.OUTPUT_DIR / 'fsi_monthly.csv', index=False)
        print(f"  Saved: fsi_monthly.csv")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Calculate Financial Stress Index from Twitter + Google Trends',
    )

    parser.add_argument('--min-posts', type=int, default=5, help='Min tweets per day')
    parser.add_argument('--twitter-weight', type=float, default=0.5, help='Twitter weight')
    parser.add_argument('--gt-weight', type=float, default=0.5, help='Google Trends weight')
    parser.add_argument('--aggregation', type=str, default='weighted_average',
                        choices=['average', 'weighted_average', 'pca'],
                        help='Google Trends aggregation method')
    parser.add_argument('--no-plots', action='store_true', help='Skip plots')

    args = parser.parse_args()

    # Update config
    CONFIG.MIN_POSTS_PER_DAY = args.min_posts
    CONFIG.TWITTER_WEIGHT = args.twitter_weight
    CONFIG.GT_WEIGHT = args.gt_weight
    CONFIG.GT_AGGREGATION = args.aggregation

    print("=" * 60)
    print("FINANCIAL STRESS INDEX CALCULATION")
    print("Twitter + Google Trends (Baker et al. 2019 + Da et al. 2011)")
    print("=" * 60)

    stats = get_dictionary_stats()
    print(f"\n📚 Dictionaries: {stats['total_unique']} terms")
    print(f"🔍 Google Trends: {stats['google_trends_queries']} queries")

    print(f"\n⚙️  Configuration:")
    print(f"   Twitter weight: {CONFIG.TWITTER_WEIGHT}")
    print(f"   Google Trends weight: {CONFIG.GT_WEIGHT}")
    print(f"   GT aggregation: {CONFIG.GT_AGGREGATION}")

    # Load data
    print(f"\n📁 Loading data...")
    twitter_df, gt_df, market_df = load_data()

    twitter_daily = None
    twitter_weekly = None
    gt_weekly = None
    combined_weekly = None

    # Process Twitter
    if not twitter_df.empty:
        print(f"\n🐦 Processing Twitter...")
        twitter_classified = classify_twitter_dataframe(twitter_df)
        n_fin = twitter_classified['is_financial'].sum()
        n_stress = twitter_classified['is_stress'].sum()
        print(f"    Financial: {n_fin:,} ({n_fin/len(twitter_classified)*100:.1f}%)")
        print(f"    Stress: {n_stress:,} ({n_stress/n_fin*100:.1f}% of financial)")

        twitter_daily = calculate_twitter_daily_fsi(twitter_classified)
        twitter_daily['fsi'] = standardize_fsi(twitter_daily['fsi_raw'])

    # Process Google Trends
    if not gt_df.empty:
        print(f"\n🔍 Processing Google Trends...")
        gt_weekly = calculate_google_trends_fsi(gt_df, method=CONFIG.GT_AGGREGATION)
        gt_weekly['fssvi'] = standardize_fsi(gt_weekly['fssvi_raw'])

    # Combine
    if twitter_daily is not None and gt_weekly is not None:
        print(f"\n🔗 Combining Twitter + Google Trends...")
        combined_weekly = combine_twitter_and_google_trends(
            twitter_daily, gt_weekly,
            twitter_weight=CONFIG.TWITTER_WEIGHT,
            gt_weight=CONFIG.GT_WEIGHT,
        )
        combined_weekly['fsi'] = standardize_fsi(combined_weekly['fsi_combined'])
    elif twitter_daily is not None:
        # Aggregate Twitter to weekly
        twitter_daily_copy = twitter_daily.copy()
        twitter_daily_copy['week'] = pd.to_datetime(twitter_daily_copy['date']).dt.to_period('W')
        combined_weekly = twitter_daily_copy.groupby('week').agg({'fsi': 'mean'}).reset_index()
        combined_weekly['date'] = combined_weekly['week'].dt.to_timestamp()
        combined_weekly = combined_weekly.drop('week', axis=1)
    elif gt_weekly is not None:
        combined_weekly = gt_weekly.rename(columns={'fssvi': 'fsi'})

    # Validate
    if combined_weekly is not None and not market_df.empty:
        print(f"\n📊 Validating...")

        # Validate combined
        validation = validate_fsi(combined_weekly, market_df, 'fsi')
        print_validation_report(validation, 'Combined')

        # Validate individual components
        if twitter_daily is not None:
            tw_weekly = twitter_daily.copy()
            tw_weekly['week'] = pd.to_datetime(tw_weekly['date']).dt.to_period('W')
            tw_weekly = tw_weekly.groupby('week').agg({'fsi': 'mean'}).reset_index()
            tw_weekly['date'] = tw_weekly['week'].dt.to_timestamp()
            tw_validation = validate_fsi(tw_weekly, market_df, 'fsi')
            print_validation_report(tw_validation, 'Twitter')

        if gt_weekly is not None:
            gt_validation = validate_fsi(gt_weekly.rename(columns={'fssvi': 'fsi'}), market_df, 'fsi')
            print_validation_report(gt_validation, 'Google Trends')

    # Monthly
    monthly_fsi = None
    if combined_weekly is not None:
        monthly_fsi = calculate_monthly_fsi(combined_weekly, 'fsi')

    # Save
    print(f"\n💾 Saving results...")
    save_results(twitter_daily, gt_weekly, combined_weekly, monthly_fsi)

    # Plots
    if not args.no_plots and combined_weekly is not None:
        print(f"\n📈 Generating plots...")
        CONFIG.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        plot_fsi_timeseries(
            combined_weekly, 'fsi', 'Financial Stress Index (Combined)',
            CONFIG.PLOTS_DIR / 'fsi_timeseries.png'
        )

        if not market_df.empty:
            plot_fsi_vs_volatility(
                combined_weekly, market_df, 'fsi',
                CONFIG.PLOTS_DIR / 'fsi_vs_volatility.png'
            )

        if 'fsi_twitter' in combined_weekly.columns or 'fsi_gt' in combined_weekly.columns:
            plot_fsi_components(combined_weekly, CONFIG.PLOTS_DIR / 'fsi_components.png')

    print("\n" + "=" * 60)
    print("✅ FSI CALCULATION COMPLETE")
    print("=" * 60)
    print(f"   Results: {CONFIG.OUTPUT_DIR}/")
    print(f"   Plots:   {CONFIG.PLOTS_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
