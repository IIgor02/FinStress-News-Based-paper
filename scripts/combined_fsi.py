#!/usr/bin/env python3
"""
Combined Financial Stress Index
================================
Combines multiple FSI methodologies:
1. Dictionary-based FSI (Google Trends) - Da et al. (2011)
2. ML-based FSI (LASSO) - García et al. (2023)
3. News-based FSI (Article analysis) - Baker et al. (2019) style

Usage:
    # Run all methods and combine
    python scripts/combined_fsi.py

    # Specific combination methods
    python scripts/combined_fsi.py --method average
    python scripts/combined_fsi.py --method pca

Output:
    output/combined_fsi.csv
    output/combined_fsi_analysis.png
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Handle both script execution and interactive usage
try:
    _SCRIPT_DIR = Path(__file__).parent
    _PROJECT_DIR = _SCRIPT_DIR.parent
except NameError:
    _PROJECT_DIR = Path.cwd()
    _SCRIPT_DIR = _PROJECT_DIR / 'scripts'

sys.path.insert(0, str(_PROJECT_DIR))

from scripts.dictionaries import CRISIS_EPISODES


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = _PROJECT_DIR / 'data' / 'raw'
OUTPUT_DIR = _PROJECT_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOAD INDIVIDUAL FSI SERIES
# =============================================================================

def load_dictionary_fsi() -> Optional[pd.DataFrame]:
    """Load dictionary-based FSI from run_fsi.py output."""
    fsi_path = OUTPUT_DIR / 'results' / 'fsi_weekly.csv'

    if fsi_path.exists():
        df = pd.read_csv(fsi_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'fsi': 'dict_fsi'})
        print(f"  Dictionary FSI: {len(df)} weeks")
        return df[['date', 'dict_fsi']]

    print("  Dictionary FSI: Not found (run scripts/run_fsi.py)")
    return None


def load_ml_fsi() -> Optional[pd.DataFrame]:
    """Load ML-based FSI from ml_fsi.py output."""
    fsi_path = OUTPUT_DIR / 'ml_fsi_weekly.csv'

    if fsi_path.exists():
        df = pd.read_csv(fsi_path)
        df['date'] = pd.to_datetime(df['date'])
        if 'ml_fsi' in df.columns:
            print(f"  ML FSI: {len(df)} weeks")
            return df[['date', 'ml_fsi']]
        elif 'fsi' in df.columns:
            df = df.rename(columns={'fsi': 'ml_fsi'})
            print(f"  ML FSI: {len(df)} weeks")
            return df[['date', 'ml_fsi']]

    print("  ML FSI: Not found (run scripts/ml_fsi.py)")
    return None


def load_news_fsi() -> Optional[pd.DataFrame]:
    """Load news-based FSI from news_fsi.py output."""
    fsi_path = OUTPUT_DIR / 'news_fsi_weekly.csv'

    if fsi_path.exists():
        df = pd.read_csv(fsi_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'news_fsi': 'news_fsi'})
        print(f"  News FSI: {len(df)} weeks")
        return df[['date', 'news_fsi']]

    print("  News FSI: Not found (run scripts/news_fsi.py)")
    return None


def load_ibovespa() -> Optional[pd.DataFrame]:
    """Load IBOVESPA data for validation."""
    ibov_path = DATA_DIR / 'ibovespa_data.csv'

    if ibov_path.exists():
        df = pd.read_csv(ibov_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"  IBOVESPA: {len(df)} days")
        return df

    print("  IBOVESPA: Not found")
    return None


# =============================================================================
# COMBINATION METHODS
# =============================================================================

def standardize_series(series: pd.Series) -> pd.Series:
    """
    Standardize a series to 0-1 scale using min-max normalization.

    0 = minimum stress, 1 = maximum stress
    """
    valid = series.dropna()
    if len(valid) < 2:
        return series

    min_val, max_val = valid.min(), valid.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)

    # Min-max normalization to 0-1
    standardized = (series - min_val) / (max_val - min_val)
    return standardized.clip(0, 1)


def combine_simple_average(fsi_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine FSI series using simple average."""
    # Merge all series
    combined = None
    for name, df in fsi_dict.items():
        if combined is None:
            combined = df.copy()
        else:
            combined = combined.merge(df, on='date', how='outer')

    if combined is None:
        return pd.DataFrame()

    combined = combined.sort_values('date')

    # Get FSI columns
    fsi_cols = [c for c in combined.columns if c != 'date' and 'fsi' in c.lower()]

    if len(fsi_cols) == 0:
        return combined

    # Standardize each series
    for col in fsi_cols:
        combined[col] = standardize_series(combined[col])

    # Calculate average
    combined['combined_fsi'] = combined[fsi_cols].mean(axis=1)

    return combined


def combine_weighted_average(fsi_dict: Dict[str, pd.DataFrame],
                              weights: Dict[str, float] = None) -> pd.DataFrame:
    """Combine FSI series using weighted average."""
    if weights is None:
        # Default weights (higher for theory-based methods)
        weights = {
            'dict_fsi': 0.4,    # Dictionary method (well-established)
            'ml_fsi': 0.35,    # ML method (data-driven)
            'news_fsi': 0.25,  # News method (newer)
        }

    combined = combine_simple_average(fsi_dict)

    if combined.empty:
        return combined

    fsi_cols = [c for c in combined.columns if c != 'date' and c != 'combined_fsi' and 'fsi' in c.lower()]

    if len(fsi_cols) == 0:
        return combined

    # Calculate weighted average
    weighted_sum = 0
    total_weight = 0

    for col in fsi_cols:
        w = weights.get(col, 1.0 / len(fsi_cols))
        valid_mask = combined[col].notna()
        weighted_sum = weighted_sum + combined[col].fillna(0) * w * valid_mask.astype(float)
        total_weight = total_weight + w * valid_mask.astype(float)

    combined['combined_fsi'] = weighted_sum / total_weight.replace(0, np.nan)

    return combined


def combine_pca(fsi_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine FSI series using PCA (first principal component)."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  WARNING: scikit-learn not installed, using simple average")
        return combine_simple_average(fsi_dict)

    combined = combine_simple_average(fsi_dict)

    if combined.empty:
        return combined

    fsi_cols = [c for c in combined.columns if c != 'date' and c != 'combined_fsi' and 'fsi' in c.lower()]

    if len(fsi_cols) < 2:
        return combined

    # Prepare data for PCA
    X = combined[fsi_cols].dropna()

    if len(X) < 10:
        print("  WARNING: Insufficient data for PCA, using simple average")
        return combined

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X_scaled).flatten()

    print(f"    PCA variance explained: {pca.explained_variance_ratio_[0]:.1%}")

    # Store PC1 and standardize
    combined.loc[X.index, 'combined_fsi'] = standardize_series(pd.Series(pc1, index=X.index))

    return combined


def combine_dynamic(fsi_dict: Dict[str, pd.DataFrame],
                    ibov_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Dynamic combination: weight each FSI by its recent correlation with volatility.
    """
    combined = combine_simple_average(fsi_dict)

    if combined.empty or ibov_df is None:
        return combined

    fsi_cols = [c for c in combined.columns if c != 'date' and c != 'combined_fsi' and 'fsi' in c.lower()]

    if len(fsi_cols) == 0:
        return combined

    # Resample IBOV to weekly
    ibov = ibov_df.copy()
    ibov = ibov.set_index('date')

    if 'realized_vol_21d' not in ibov.columns:
        return combine_weighted_average(fsi_dict)

    vol_weekly = ibov['realized_vol_21d'].resample('W').mean()

    # Calculate rolling correlation for each FSI
    window = 52  # 1 year
    combined = combined.set_index('date')

    for col in fsi_cols:
        combined[f'{col}_corr'] = combined[col].rolling(window, min_periods=26).corr(vol_weekly)

    # Calculate dynamic weights (higher weight = higher correlation)
    corr_cols = [f'{c}_corr' for c in fsi_cols]
    combined['total_corr'] = combined[corr_cols].abs().sum(axis=1)

    # Calculate weighted FSI
    weighted_fsi = 0
    for col in fsi_cols:
        weight = combined[f'{col}_corr'].abs() / combined['total_corr'].replace(0, np.nan)
        weighted_fsi = weighted_fsi + combined[col].fillna(0) * weight.fillna(0)

    combined['combined_fsi'] = standardize_series(weighted_fsi)

    # Clean up temp columns
    combined = combined.drop(columns=corr_cols + ['total_corr'], errors='ignore')
    combined = combined.reset_index()

    return combined


# =============================================================================
# VALIDATION
# =============================================================================

def validate_combined_fsi(combined_df: pd.DataFrame, ibov_df: pd.DataFrame) -> Dict:
    """Validate combined FSI against market volatility."""
    from scipy import stats

    results = {}

    if ibov_df is None or 'realized_vol_21d' not in ibov_df.columns:
        return results

    # Resample IBOV to weekly
    ibov = ibov_df.copy()
    ibov['week'] = ibov['date'].dt.to_period('W').dt.start_time
    vol_weekly = ibov.groupby('week')['realized_vol_21d'].mean().reset_index()
    vol_weekly.columns = ['date', 'volatility']

    # Merge
    merged = combined_df.merge(vol_weekly, on='date', how='inner')

    if len(merged) < 10:
        return results

    # Calculate correlation for each FSI
    fsi_cols = [c for c in merged.columns if 'fsi' in c.lower()]

    for col in fsi_cols:
        valid = merged[[col, 'volatility']].dropna()
        if len(valid) >= 10:
            corr, pval = stats.pearsonr(valid[col], valid['volatility'])
            results[col] = {'correlation': corr, 'p_value': pval, 'n': len(valid)}

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_combined_fsi(combined_df: pd.DataFrame, ibov_df: pd.DataFrame = None,
                       output_path: Path = None) -> None:
    """Create visualization comparing all FSI series."""
    if not MATPLOTLIB_AVAILABLE:
        print("  WARNING: matplotlib not installed, skipping visualization")
        return

    fsi_cols = [c for c in combined_df.columns if 'fsi' in c.lower() and c != 'combined_fsi']
    n_plots = 2 + (1 if ibov_df is not None else 0)

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)

    if n_plots == 2:
        axes = list(axes)

    colors = {'dict_fsi': '#2E86AB', 'ml_fsi': '#E94F37', 'news_fsi': '#4ECDC4'}
    labels = {'dict_fsi': 'Dictionary (Da et al.)', 'ml_fsi': 'ML (García et al.)', 'news_fsi': 'News (Baker et al.)'}

    # Plot 1: Individual FSI series (0-1 scale)
    ax1 = axes[0]
    for col in fsi_cols:
        ax1.plot(combined_df['date'], combined_df[col],
                 color=colors.get(col, 'gray'), alpha=0.6, linewidth=1,
                 label=labels.get(col, col))
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('FSI (0-1 scale)')
    ax1.set_ylim(0, 1)
    ax1.set_title('Individual FSI Components (0=No Stress, 1=Maximum Stress)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Add crisis shading
    for (start, end), label in CRISIS_EPISODES.items():
        start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
        if combined_df['date'].min() <= end_dt and combined_df['date'].max() >= start_dt:
            ax1.axvspan(start_dt, end_dt, alpha=0.1, color='red')

    # Plot 2: Combined FSI (0-1 scale)
    ax2 = axes[1]
    ax2.plot(combined_df['date'], combined_df['combined_fsi'],
             color='black', linewidth=2, label='Combined FSI')
    ax2.fill_between(combined_df['date'], 0.5, combined_df['combined_fsi'],
                     where=combined_df['combined_fsi'] > 0.5, alpha=0.3, color='red')
    ax2.fill_between(combined_df['date'], 0.5, combined_df['combined_fsi'],
                     where=combined_df['combined_fsi'] <= 0.5, alpha=0.3, color='green')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Combined FSI (0-1 scale)')
    ax2.set_ylim(0, 1)
    ax2.set_title('Combined Financial Stress Index (0=No Stress, 1=Maximum Stress)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Add crisis shading
    for (start, end), label in CRISIS_EPISODES.items():
        start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
        if combined_df['date'].min() <= end_dt and combined_df['date'].max() >= start_dt:
            ax2.axvspan(start_dt, end_dt, alpha=0.1, color='red')

    # Plot 3: Comparison with volatility (if available)
    if ibov_df is not None and len(axes) > 2:
        ax3 = axes[2]

        # Resample IBOV to weekly
        ibov = ibov_df.copy()
        ibov = ibov.set_index('date')

        if 'realized_vol_21d' in ibov.columns:
            vol_weekly = ibov['realized_vol_21d'].resample('W').mean()
            ax3.plot(vol_weekly.index, vol_weekly.values, color='orange',
                     linewidth=1.5, label='IBOV Volatility (21d)', alpha=0.8)

        ax3_twin = ax3.twinx()
        ax3_twin.plot(combined_df['date'], combined_df['combined_fsi'],
                      color='black', linewidth=1.5, alpha=0.7, label='Combined FSI')
        ax3_twin.set_ylabel('Combined FSI', color='black')

        ax3.set_ylabel('Volatility (%)', color='orange')
        ax3.set_title('Combined FSI vs Market Volatility')
        ax3.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Format x-axis
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
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
        description='Combine multiple FSI methodologies',
    )
    parser.add_argument('--method', type=str, default='weighted',
                        choices=['average', 'weighted', 'pca', 'dynamic'],
                        help='Combination method (default: weighted)')

    args = parser.parse_args()

    print("=" * 60)
    print("COMBINED FINANCIAL STRESS INDEX")
    print("=" * 60)

    # Load individual FSI series
    print("\n" + "-" * 40)
    print("Loading FSI Components")
    print("-" * 40)

    fsi_dict = {}

    dict_fsi = load_dictionary_fsi()
    if dict_fsi is not None:
        fsi_dict['dict_fsi'] = dict_fsi

    ml_fsi = load_ml_fsi()
    if ml_fsi is not None:
        fsi_dict['ml_fsi'] = ml_fsi

    news_fsi = load_news_fsi()
    if news_fsi is not None:
        fsi_dict['news_fsi'] = news_fsi

    ibov_df = load_ibovespa()

    if len(fsi_dict) == 0:
        print("\n  ERROR: No FSI data found. Run individual scripts first:")
        print("    - python scripts/run_fsi.py (Dictionary FSI)")
        print("    - python scripts/ml_fsi.py (ML FSI)")
        print("    - python scripts/news_fsi.py (News FSI)")
        return

    print(f"\n  Found {len(fsi_dict)} FSI series: {list(fsi_dict.keys())}")

    # Combine FSI series
    print("\n" + "-" * 40)
    print(f"Combining FSI (method: {args.method})")
    print("-" * 40)

    if args.method == 'average':
        combined_df = combine_simple_average(fsi_dict)
    elif args.method == 'weighted':
        combined_df = combine_weighted_average(fsi_dict)
    elif args.method == 'pca':
        combined_df = combine_pca(fsi_dict)
    elif args.method == 'dynamic':
        combined_df = combine_dynamic(fsi_dict, ibov_df)
    else:
        combined_df = combine_weighted_average(fsi_dict)

    if combined_df.empty:
        print("\n  ERROR: Failed to combine FSI series")
        return

    print(f"  Combined {len(combined_df)} weeks of data")

    # Validate
    if ibov_df is not None:
        print("\n" + "-" * 40)
        print("Validation")
        print("-" * 40)

        validation = validate_combined_fsi(combined_df, ibov_df)

        if validation:
            print("\n  Correlation with IBOVESPA Volatility:")
            for fsi_name, metrics in validation.items():
                r = metrics['correlation']
                p = metrics['p_value']
                n = metrics['n']
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                print(f"    {fsi_name:15s}: r = {r:+.3f} (p = {p:.4f}) {sig} (n={n})")

    # Save
    print("\n" + "-" * 40)
    print("Saving Results")
    print("-" * 40)

    output_path = OUTPUT_DIR / 'combined_fsi.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    # Visualization
    print("\n" + "-" * 40)
    print("Visualization")
    print("-" * 40)

    plot_path = OUTPUT_DIR / 'combined_fsi_analysis.png'
    plot_combined_fsi(combined_df, ibov_df, plot_path)

    # Summary
    print("\n" + "=" * 60)
    print("COMBINED FSI COMPLETE")
    print("=" * 60)

    fsi_cols = [c for c in combined_df.columns if 'fsi' in c.lower()]
    print(f"\n📊 Combined FSI Statistics:")
    print(f"   Period: {combined_df['date'].min().date()} to {combined_df['date'].max().date()}")
    print(f"   Weeks: {len(combined_df)}")

    if 'combined_fsi' in combined_df.columns:
        cfsi = combined_df['combined_fsi'].dropna()
        print(f"\n   Combined FSI:")
        print(f"     Mean: {cfsi.mean():.2f}")
        print(f"     Std:  {cfsi.std():.2f}")
        print(f"     Max:  {cfsi.max():.2f} ({combined_df.loc[cfsi.idxmax(), 'date'].date()})")
        print(f"     Min:  {cfsi.min():.2f} ({combined_df.loc[cfsi.idxmin(), 'date'].date()})")

    print(f"\n📁 Output files:")
    print(f"   - {output_path}")
    print(f"   - {plot_path}")

    print("=" * 60)


if __name__ == '__main__':
    main()
