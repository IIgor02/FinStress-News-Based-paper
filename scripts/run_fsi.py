#!/usr/bin/env python3
"""
Financial Stress Index Calculator
=================================
Calculates FSI from Google Trends search volume data (Da et al. 2011).

Usage:
    python scripts/run_fsi.py
    python scripts/run_fsi.py --aggregation pca
    python scripts/run_fsi.py --no-plots

Output:
    output/results/fsi_weekly.csv
    output/results/fsi_monthly.csv
    output/plots/fsi_timeseries.png
    output/plots/fsi_vs_volatility.png
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

# Handle both script execution and interactive/REPL usage
try:
    _SCRIPT_DIR = Path(__file__).parent
    _PROJECT_DIR = _SCRIPT_DIR.parent
except NameError:
    _PROJECT_DIR = Path.cwd()
    _SCRIPT_DIR = _PROJECT_DIR / 'scripts'

sys.path.insert(0, str(_PROJECT_DIR))

from scripts.dictionaries import (
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

class Config:
    """Configuration for FSI calculation."""
    MIN_QUERIES: int = 3
    TARGET_MEAN: float = 100.0
    VOLATILITY_COLUMN: str = 'realized_vol_30d'
    TARGET_CORR_MIN: float = 0.60
    TARGET_CORR_MAX: float = 0.90
    SMOOTHING_WINDOW: int = 4
    AGGREGATION: str = 'weighted_average'

    DATA_DIR: Path = _PROJECT_DIR / 'data' / 'raw'
    OUTPUT_DIR: Path = _PROJECT_DIR / 'output' / 'results'
    PLOTS_DIR: Path = _PROJECT_DIR / 'output' / 'plots'


CONFIG = Config()


# =============================================================================
# FSI CALCULATION
# =============================================================================

def calculate_fsi(gt_df: pd.DataFrame, method: str = 'weighted_average') -> pd.DataFrame:
    """
    Calculate Financial Stress Index from Google Trends data.

    Methods:
    - average: Simple average of all query SVIs
    - weighted_average: Weighted by tier (1.5 for crisis, 1.2 for market, 1.0 for others)
    - pca: First principal component
    """
    df = gt_df.copy()
    query_cols = [c for c in df.columns if c not in ['date', 'source', 'isPartial']]

    if len(query_cols) < CONFIG.MIN_QUERIES:
        print(f"    Only {len(query_cols)} queries found (need {CONFIG.MIN_QUERIES})")
        return pd.DataFrame(columns=['date', 'fsi_raw'])

    print(f"    Processing {len(query_cols)} queries with '{method}' method...")

    if method == 'average':
        df['fsi_raw'] = df[query_cols].mean(axis=1)

    elif method == 'weighted_average':
        weights = {q: get_query_weight(q) for q in query_cols}
        total_weight = sum(weights.values())
        weighted_sum = sum(df[q] * weights.get(q, 1.0) for q in query_cols)
        df['fsi_raw'] = weighted_sum / total_weight

    elif method == 'pca':
        from sklearn.decomposition import PCA

        query_data = df[query_cols].dropna()
        if len(query_data) < 10:
            print("    Insufficient data for PCA, using weighted average")
            return calculate_fsi(gt_df, 'weighted_average')

        query_std = (query_data - query_data.mean()) / query_data.std()
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(query_std).flatten()
        print(f"    PCA variance explained: {pca.explained_variance_ratio_[0]:.1%}")

        pc1_scaled = (pc1 - pc1.min()) / (pc1.max() - pc1.min()) * 100
        df.loc[query_data.index, 'fsi_raw'] = pc1_scaled

    else:
        raise ValueError(f"Unknown method: {method}")

    result = df[['date', 'fsi_raw']].copy()
    result['date'] = pd.to_datetime(result['date'])
    result = result.sort_values('date').reset_index(drop=True)

    return result


def standardize_fsi(fsi_series: pd.Series) -> pd.Series:
    """Standardize FSI to target mean (100)."""
    valid = fsi_series.dropna()
    if len(valid) == 0:
        return fsi_series

    mean, std = valid.mean(), valid.std()
    if std == 0:
        return pd.Series(CONFIG.TARGET_MEAN, index=fsi_series.index)

    return (fsi_series - mean) / std * std + CONFIG.TARGET_MEAN


def calculate_monthly_fsi(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weekly FSI to monthly."""
    df = weekly_df.copy()
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    monthly = df.groupby('month')['fsi'].mean().reset_index()
    monthly['month'] = monthly['month'].dt.to_timestamp()
    return monthly


# =============================================================================
# VALIDATION
# =============================================================================

def calculate_correlation(fsi: pd.Series, vol: pd.Series) -> Dict:
    """Calculate correlation metrics."""
    aligned = pd.DataFrame({'fsi': fsi, 'vol': vol}).dropna()

    if len(aligned) < 10:
        return {'pearson': np.nan, 'pearson_p': np.nan, 'spearman': np.nan, 'spearman_p': np.nan, 'n': len(aligned)}

    pearson_r, pearson_p = stats.pearsonr(aligned['fsi'], aligned['vol'])
    spearman_r, spearman_p = stats.spearmanr(aligned['fsi'], aligned['vol'])

    return {'pearson': pearson_r, 'pearson_p': pearson_p, 'spearman': spearman_r, 'spearman_p': spearman_p, 'n': len(aligned)}


def validate_fsi(weekly_fsi: pd.DataFrame, market_data: pd.DataFrame) -> Dict:
    """Validate FSI against market volatility."""
    results = {}

    # Weekly validation
    market = market_data.copy()
    market['week'] = pd.to_datetime(market['date']).dt.to_period('W')
    market_weekly = market.groupby('week')[CONFIG.VOLATILITY_COLUMN].mean().reset_index()

    fsi = weekly_fsi.copy()
    fsi['week'] = pd.to_datetime(fsi['date']).dt.to_period('W')

    merged = pd.merge(fsi[['week', 'fsi']], market_weekly[['week', CONFIG.VOLATILITY_COLUMN]], on='week', how='inner')
    results['weekly'] = calculate_correlation(merged['fsi'], merged[CONFIG.VOLATILITY_COLUMN])

    # Monthly validation
    monthly_fsi = calculate_monthly_fsi(weekly_fsi)
    market['month'] = pd.to_datetime(market['date']).dt.to_period('M')
    market_monthly = market.groupby('month')[CONFIG.VOLATILITY_COLUMN].mean().reset_index()
    market_monthly['month'] = market_monthly['month'].dt.to_timestamp()

    monthly_merged = pd.merge(
        monthly_fsi.rename(columns={'month': 'date'}),
        market_monthly.rename(columns={'month': 'date'}),
        on='date', how='inner'
    )
    results['monthly'] = calculate_correlation(monthly_merged['fsi'], monthly_merged[CONFIG.VOLATILITY_COLUMN])

    # Stats
    results['stats'] = {
        'mean': weekly_fsi['fsi'].mean(),
        'std': weekly_fsi['fsi'].std(),
        'min': weekly_fsi['fsi'].min(),
        'max': weekly_fsi['fsi'].max(),
        'n_weeks': len(weekly_fsi),
    }

    return results


def print_validation(results: Dict):
    """Print validation report."""
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print("\nCorrelation with IBOVESPA Volatility:")
    print("-" * 40)

    w = results['weekly']
    if w['n'] >= 10 and not np.isnan(w['pearson']):
        status = "OK" if CONFIG.TARGET_CORR_MIN <= abs(w['pearson']) <= CONFIG.TARGET_CORR_MAX else ""
        print(f"  Weekly (n={w['n']}):")
        print(f"    Pearson:  r = {w['pearson']:.3f} (p = {w['pearson_p']:.4f}) {status}")
        print(f"    Spearman: r = {w['spearman']:.3f} (p = {w['spearman_p']:.4f})")
    else:
        print(f"  Weekly (n={w['n']}): Insufficient data")

    m = results['monthly']
    if m['n'] >= 10 and not np.isnan(m['pearson']):
        status = "OK" if CONFIG.TARGET_CORR_MIN <= abs(m['pearson']) <= CONFIG.TARGET_CORR_MAX else ""
        print(f"\n  Monthly (n={m['n']}):")
        print(f"    Pearson:  r = {m['pearson']:.3f} (p = {m['pearson_p']:.4f}) {status}")
        print(f"    Spearman: r = {m['spearman']:.3f} (p = {m['spearman_p']:.4f})")
    else:
        print(f"\n  Monthly (n={m['n']}): Insufficient data")

    print(f"\n  Target correlation: {CONFIG.TARGET_CORR_MIN:.2f} - {CONFIG.TARGET_CORR_MAX:.2f}")

    s = results['stats']
    print(f"\nFSI Statistics:")
    print(f"  Mean:  {s['mean']:.2f}")
    print(f"  Std:   {s['std']:.2f}")
    print(f"  Range: [{s['min']:.2f}, {s['max']:.2f}]")
    print(f"  Weeks: {s['n_weeks']}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_fsi_timeseries(weekly_fsi: pd.DataFrame, save_path: str = None):
    """Plot FSI time series with crisis markers."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(weekly_fsi['date'], weekly_fsi['fsi'], color='#2E86AB', linewidth=1.5, label='FSI')

    if len(weekly_fsi) > CONFIG.SMOOTHING_WINDOW:
        rolling = weekly_fsi['fsi'].rolling(window=CONFIG.SMOOTHING_WINDOW, center=True).mean()
        ax.plot(weekly_fsi['date'], rolling, color='#E94F37', linewidth=2,
                label=f'{CONFIG.SMOOTHING_WINDOW}-week MA', alpha=0.8)

    for (start, end), label in CRISIS_EPISODES.items():
        start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
        if weekly_fsi['date'].min() <= end_dt and weekly_fsi['date'].max() >= start_dt:
            ax.axvspan(start_dt, end_dt, alpha=0.15, color='red')
            mid = start_dt + (end_dt - start_dt) / 2
            ax.text(mid, ax.get_ylim()[1] * 0.97, label, ha='center', va='top', fontsize=7, rotation=45)

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('FSI', fontsize=11)
    ax.set_title('Financial Stress Index - Brazil (Google Trends)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_fsi_vs_volatility(weekly_fsi: pd.DataFrame, market_data: pd.DataFrame, save_path: str = None):
    """Plot FSI vs market volatility."""
    market = market_data.copy()
    market['week'] = pd.to_datetime(market['date']).dt.to_period('W')
    market_weekly = market.groupby('week')[CONFIG.VOLATILITY_COLUMN].mean().reset_index()
    market_weekly['date'] = market_weekly['week'].dt.to_timestamp()

    merged = pd.merge(weekly_fsi[['date', 'fsi']], market_weekly[['date', CONFIG.VOLATILITY_COLUMN]], on='date', how='inner')

    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(merged['date'], merged['fsi'], color='#2E86AB', linewidth=1.5, label='FSI')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('FSI', color='#2E86AB', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#2E86AB')

    ax2 = ax1.twinx()
    ax2.plot(merged['date'], merged[CONFIG.VOLATILITY_COLUMN], color='#E94F37', linewidth=1.5, label='Volatility', alpha=0.7)
    ax2.set_ylabel('Volatility (%)', color='#E94F37', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#E94F37')

    corr = merged['fsi'].corr(merged[CONFIG.VOLATILITY_COLUMN])
    ax1.set_title(f'FSI vs IBOVESPA Volatility (r = {corr:.3f})', fontsize=13, fontweight='bold')

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


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load Google Trends and IBOVESPA data."""
    gt_path = CONFIG.DATA_DIR / 'google_trends_data.csv'
    market_path = CONFIG.DATA_DIR / 'ibovespa_data.csv'

    gt_df = pd.DataFrame()
    market_df = pd.DataFrame()

    if gt_path.exists():
        gt_df = pd.read_csv(gt_path)
        gt_df['date'] = pd.to_datetime(gt_df['date'])
        query_cols = [c for c in gt_df.columns if c not in ['date', 'source', 'isPartial']]
        print(f"  Google Trends: {len(gt_df)} weeks, {len(query_cols)} queries")
    else:
        print(f"  Google Trends: File not found at {gt_path}")

    if market_path.exists():
        market_df = pd.read_csv(market_path)
        market_df['date'] = pd.to_datetime(market_df['date'])
        print(f"  IBOVESPA: {len(market_df):,} trading days")
    else:
        print(f"  IBOVESPA: File not found at {market_path}")

    return gt_df, market_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Calculate Financial Stress Index from Google Trends')
    parser.add_argument('--aggregation', type=str, default='weighted_average',
                        choices=['average', 'weighted_average', 'pca'], help='Aggregation method')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')

    args = parser.parse_args()
    CONFIG.AGGREGATION = args.aggregation

    print("=" * 60)
    print("FINANCIAL STRESS INDEX (Da et al. 2011)")
    print("=" * 60)

    stats = get_dictionary_stats()
    print(f"\nGoogle Trends queries: {stats['google_trends_queries']}")
    print(f"Aggregation method: {CONFIG.AGGREGATION}")

    # Load data
    print(f"\nLoading data...")
    gt_df, market_df = load_data()

    if gt_df.empty:
        print("\nERROR: No Google Trends data found. Run collect_data.py first.")
        return

    # Calculate FSI
    print(f"\nCalculating FSI...")
    weekly_fsi = calculate_fsi(gt_df, method=CONFIG.AGGREGATION)
    weekly_fsi['fsi'] = standardize_fsi(weekly_fsi['fsi_raw'])

    # Validate
    if not market_df.empty:
        validation = validate_fsi(weekly_fsi, market_df)
        print_validation(validation)

    # Monthly
    monthly_fsi = calculate_monthly_fsi(weekly_fsi)

    # Save
    print(f"\nSaving results...")
    CONFIG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    weekly_fsi.to_csv(CONFIG.OUTPUT_DIR / 'fsi_weekly.csv', index=False)
    print(f"  Saved: fsi_weekly.csv")

    monthly_fsi.to_csv(CONFIG.OUTPUT_DIR / 'fsi_monthly.csv', index=False)
    print(f"  Saved: fsi_monthly.csv")

    # Plots
    if not args.no_plots:
        print(f"\nGenerating plots...")
        CONFIG.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        plot_fsi_timeseries(weekly_fsi, CONFIG.PLOTS_DIR / 'fsi_timeseries.png')

        if not market_df.empty:
            plot_fsi_vs_volatility(weekly_fsi, market_df, CONFIG.PLOTS_DIR / 'fsi_vs_volatility.png')

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Results: {CONFIG.OUTPUT_DIR}/")
    print(f"  Plots:   {CONFIG.PLOTS_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
