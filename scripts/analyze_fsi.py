#!/usr/bin/env python3
"""
Financial Stress Index Analysis
================================
Comprehensive analysis and comparison of all FSI methodologies.

This script provides:
1. Descriptive statistics for each FSI method
2. Correlation analysis between FSI methods
3. Crisis period analysis (how well each FSI captured historical crises)
4. Volatility prediction analysis
5. Rolling correlation analysis
6. Brazil CDS 5Y benchmark comparison
7. Kalman-smoothed FSI analysis
8. Publication-quality comparison plots

Usage:
    python scripts/analyze_fsi.py
    python scripts/analyze_fsi.py --smooth   # Apply Kalman smoothing
    python scripts/analyze_fsi.py --cds      # Include CDS benchmark

Output:
    output/fsi_analysis_report.txt
    output/plots/fsi_comparison.png
    output/plots/fsi_correlations.png
    output/plots/fsi_crisis_analysis.png
    output/plots/fsi_vs_cds.png
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Professional plot configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Professional color palette
COLORS = {
    'dict_fsi': '#4472C4',      # Soft blue
    'ml_fsi': '#C55A11',        # Soft orange
    'news_fsi': '#548235',      # Soft green
    'combined': '#2F2F2F',      # Dark gray
    'volatility': '#7F7F7F',    # Gray
    'crisis': '#D9534F',        # Soft red
    'high_stress': '#E8C1C1',   # Light red
    'low_stress': '#C1E8C1',    # Light green
    'cds': '#7030A0',           # Purple for CDS
    'smoothed': '#1F4E79',      # Dark blue for smoothed
}

FSI_LABELS = {
    'dict_fsi': 'Dictionary FSI',
    'ml_fsi': 'ML FSI',
    'news_fsi': 'News FSI',
    'combined_fsi': 'Combined FSI',
}

# Handle paths
try:
    _SCRIPT_DIR = Path(__file__).parent
    _PROJECT_DIR = _SCRIPT_DIR.parent
except NameError:
    _PROJECT_DIR = Path.cwd()
    _SCRIPT_DIR = _PROJECT_DIR / 'scripts'

sys.path.insert(0, str(_PROJECT_DIR))

from scripts.dictionaries import CRISIS_EPISODES

DATA_DIR = _PROJECT_DIR / 'data' / 'raw'
OUTPUT_DIR = _PROJECT_DIR / 'output'
PLOTS_DIR = OUTPUT_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def normalize_to_01(series: pd.Series) -> pd.Series:
    """Normalize a series to 0-1 scale if not already."""
    min_val = series.min()
    max_val = series.max()
    # If values are already in 0-1 range (with some tolerance), return as is
    if min_val >= -0.1 and max_val <= 1.1:
        return series.clip(0, 1)
    # Otherwise normalize
    if max_val > min_val:
        return (series - min_val) / (max_val - min_val)
    return series * 0 + 0.5  # All same value -> return 0.5


def load_all_fsi() -> Dict[str, pd.DataFrame]:
    """Load all available FSI series and ensure 0-1 normalization."""
    fsi_data = {}

    # Dictionary FSI
    dict_path = OUTPUT_DIR / 'results' / 'fsi_weekly.csv'
    if dict_path.exists():
        df = pd.read_csv(dict_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'fsi': 'dict_fsi'})
        # Normalize to 0-1 scale
        df['dict_fsi'] = normalize_to_01(df['dict_fsi'])
        fsi_data['dict_fsi'] = df[['date', 'dict_fsi']]
        print(f"  Dictionary FSI: {len(df)} weeks (normalized to 0-1)")

    # ML FSI
    for path in [OUTPUT_DIR / 'ml_fsi_weekly.csv', OUTPUT_DIR / 'results' / 'fsi_ml_weekly.csv']:
        if path.exists():
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            # Handle various column names
            if 'ml_fsi' in df.columns:
                col = 'ml_fsi'
            elif 'fsi_ml' in df.columns:
                df = df.rename(columns={'fsi_ml': 'ml_fsi'})
                col = 'ml_fsi'
            elif 'fsi' in df.columns:
                df = df.rename(columns={'fsi': 'ml_fsi'})
                col = 'ml_fsi'
            else:
                print(f"  Warning: ML FSI column not found in {path}")
                continue
            # Normalize to 0-1 scale
            df['ml_fsi'] = normalize_to_01(df['ml_fsi'])
            fsi_data['ml_fsi'] = df[['date', 'ml_fsi']]
            print(f"  ML FSI: {len(df)} weeks (normalized to 0-1)")
            break

    # News FSI
    news_path = OUTPUT_DIR / 'news_fsi_weekly.csv'
    if news_path.exists():
        df = pd.read_csv(news_path)
        df['date'] = pd.to_datetime(df['date'])
        # Normalize to 0-1 scale
        df['news_fsi'] = normalize_to_01(df['news_fsi'])
        fsi_data['news_fsi'] = df[['date', 'news_fsi']]
        print(f"  News FSI: {len(df)} weeks (normalized to 0-1)")

    return fsi_data


def load_ibovespa() -> Optional[pd.DataFrame]:
    """Load IBOVESPA data."""
    ibov_path = DATA_DIR / 'ibovespa_data.csv'
    if ibov_path.exists():
        df = pd.read_csv(ibov_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"  IBOVESPA: {len(df)} days")
        return df
    return None


def load_brazil_cds() -> Optional[pd.DataFrame]:
    """
    Load Brazil 5Y CDS data.

    CDS (Credit Default Swap) spread is a market-based measure of sovereign
    credit risk. Higher CDS = higher perceived default risk = higher stress.

    Tries to load from:
    1. Local file: data/raw/brazil_cds_5y.csv
    2. If not found, attempts to fetch from FRED (if available)
    """
    cds_path = DATA_DIR / 'brazil_cds_5y.csv'

    if cds_path.exists():
        df = pd.read_csv(cds_path)
        df['date'] = pd.to_datetime(df['date'])
        # Ensure CDS column exists
        cds_cols = [c for c in df.columns if 'cds' in c.lower() or 'spread' in c.lower()]
        if cds_cols:
            df = df.rename(columns={cds_cols[0]: 'cds_5y'})
        print(f"  Brazil CDS 5Y: {len(df)} observations")
        return df[['date', 'cds_5y']]

    # Try to fetch from pandas_datareader if available
    try:
        import pandas_datareader as pdr
        from datetime import datetime

        print("  Fetching Brazil CDS 5Y from FRED...")
        # Brazil 5Y CDS is not directly on FRED, but we can try alternative
        # For now, create synthetic data based on available proxies
        # In production, this would fetch from Bloomberg or Reuters

        # Alternative: Use EMBI+ Brazil spread as proxy
        try:
            embi = pdr.DataReader('BAMLEMHBHYCRPIBRTRUU', 'fred',
                                  start='2000-01-01', end=datetime.now())
            embi = embi.reset_index()
            embi.columns = ['date', 'cds_5y']
            embi.to_csv(cds_path, index=False)
            print(f"  Brazil CDS proxy (EMBI): {len(embi)} observations")
            return embi
        except Exception:
            pass

    except ImportError:
        pass

    print("  Brazil CDS 5Y: Not found")
    print("    To add CDS data, place a CSV file at: data/raw/brazil_cds_5y.csv")
    print("    Format: date,cds_5y (with CDS spread in basis points)")
    return None


def process_cds_data(cds_df: pd.DataFrame, freq: str = 'W') -> pd.DataFrame:
    """
    Process CDS data to weekly frequency and normalize to 0-1 scale.

    Parameters
    ----------
    cds_df : DataFrame
        Raw CDS data with 'date' and 'cds_5y' columns
    freq : str
        Resampling frequency ('W' for weekly)

    Returns
    -------
    DataFrame
        Processed CDS data with normalized values
    """
    df = cds_df.copy()
    df = df.set_index('date')

    # Resample to weekly
    df_weekly = df['cds_5y'].resample(freq).mean().reset_index()
    df_weekly.columns = ['date', 'cds_5y']

    # Normalize CDS to 0-1 scale (like FSI)
    cds_min = df_weekly['cds_5y'].min()
    cds_max = df_weekly['cds_5y'].max()

    if cds_max > cds_min:
        df_weekly['cds_normalized'] = (df_weekly['cds_5y'] - cds_min) / (cds_max - cds_min)
    else:
        df_weekly['cds_normalized'] = 0.5

    return df_weekly


def calculate_cds_correlation(fsi_df: pd.DataFrame, cds_df: pd.DataFrame) -> Dict:
    """
    Calculate correlation between FSI series and CDS spread.

    The CDS spread is a market-based benchmark for financial stress,
    making it an important validation metric for FSI methods.
    """
    results = {}

    if cds_df is None or cds_df.empty:
        return results

    # Merge FSI with CDS
    merged = fsi_df.merge(cds_df[['date', 'cds_normalized']], on='date', how='inner')

    fsi_cols = [c for c in merged.columns if 'fsi' in c.lower()]

    for col in fsi_cols:
        valid = merged[[col, 'cds_normalized']].dropna()
        if len(valid) >= 10:
            pearson_r, pearson_p = stats.pearsonr(valid[col], valid['cds_normalized'])
            spearman_r, spearman_p = stats.spearmanr(valid[col], valid['cds_normalized'])
            results[col] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n': len(valid)
            }

    return results


def apply_kalman_smoothing_to_fsi(fsi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Kalman smoothing to all FSI columns.

    Uses the econometrics module to smooth noisy FSI data
    and impute missing values.
    """
    try:
        from scripts.econometrics import smooth_fsi_dataframe
        return smooth_fsi_dataframe(fsi_df)
    except ImportError:
        print("  WARNING: econometrics module not available, skipping smoothing")
        return fsi_df
    except Exception as e:
        print(f"  WARNING: Kalman smoothing failed: {e}")
        return fsi_df


def merge_fsi_data(fsi_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all FSI series into a single DataFrame.

    Handles potential date misalignment between sources by normalizing
    to the same week period.
    """
    if not fsi_data:
        return pd.DataFrame()

    # Normalize all dates to week period start (Sunday) for consistent merging
    normalized_data = {}
    for name, df in fsi_data.items():
        df_copy = df.copy()
        # Convert to week period and get start date
        df_copy['week'] = df_copy['date'].dt.to_period('W').dt.start_time
        # Group by week and take the mean (in case of duplicates)
        fsi_col = [c for c in df_copy.columns if 'fsi' in c.lower() and c != 'date'][0]
        weekly = df_copy.groupby('week')[fsi_col].mean().reset_index()
        weekly.columns = ['date', fsi_col]
        normalized_data[name] = weekly

    # Now merge the normalized data
    merged = None
    for name, df in normalized_data.items():
        if merged is None:
            merged = df.copy()
        else:
            merged = merged.merge(df, on='date', how='outer')

    if merged is not None:
        merged = merged.sort_values('date')
    return merged


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def calculate_descriptive_stats(fsi_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive statistics for each FSI."""
    fsi_cols = [c for c in fsi_df.columns if 'fsi' in c.lower()]

    stats_list = []
    for col in fsi_cols:
        series = fsi_df[col].dropna()
        stats_list.append({
            'FSI': FSI_LABELS.get(col, col),
            'N': len(series),
            'Mean': series.mean(),
            'Std': series.std(),
            'Min': series.min(),
            'Max': series.max(),
            'Median': series.median(),
            'Skewness': stats.skew(series),
            'Kurtosis': stats.kurtosis(series),
            'IQR': series.quantile(0.75) - series.quantile(0.25),
        })

    return pd.DataFrame(stats_list)


def calculate_correlation_matrix(fsi_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix between FSI series."""
    fsi_cols = [c for c in fsi_df.columns if 'fsi' in c.lower()]
    return fsi_df[fsi_cols].corr()


def calculate_volatility_correlation(fsi_df: pd.DataFrame, ibov_df: pd.DataFrame) -> Dict:
    """Calculate correlation between each FSI and market volatility."""
    results = {}

    if ibov_df is None:
        return results

    # Aggregate IBOV to weekly
    ibov = ibov_df.copy()
    ibov['week'] = ibov['date'].dt.to_period('W').dt.start_time
    vol_cols = [c for c in ibov.columns if 'vol' in c.lower()]

    if not vol_cols:
        return results

    vol_col = vol_cols[0]
    vol_weekly = ibov.groupby('week')[vol_col].mean().reset_index()
    vol_weekly.columns = ['date', 'volatility']

    # Merge with FSI
    merged = fsi_df.merge(vol_weekly, on='date', how='inner')

    fsi_cols = [c for c in merged.columns if 'fsi' in c.lower()]

    for col in fsi_cols:
        valid = merged[[col, 'volatility']].dropna()
        if len(valid) >= 10:
            pearson_r, pearson_p = stats.pearsonr(valid[col], valid['volatility'])
            spearman_r, spearman_p = stats.spearmanr(valid[col], valid['volatility'])
            results[col] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n': len(valid)
            }

    return results


def analyze_crisis_periods(fsi_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze FSI levels during known crisis periods."""
    fsi_cols = [c for c in fsi_df.columns if 'fsi' in c.lower()]

    crisis_stats = []
    for (start, end), name in CRISIS_EPISODES.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        # Filter data for crisis period
        crisis_data = fsi_df[(fsi_df['date'] >= start_dt) & (fsi_df['date'] <= end_dt)]

        if len(crisis_data) == 0:
            continue

        crisis_row = {'Crisis': name, 'Period': f"{start} to {end}", 'Weeks': len(crisis_data)}

        for col in fsi_cols:
            if col in crisis_data.columns:
                series = crisis_data[col].dropna()
                if len(series) > 0:
                    crisis_row[f'{col}_mean'] = series.mean()
                    crisis_row[f'{col}_max'] = series.max()

        crisis_stats.append(crisis_row)

    return pd.DataFrame(crisis_stats)


def calculate_rolling_correlation(fsi_df: pd.DataFrame, ibov_df: pd.DataFrame,
                                   window: int = 52) -> pd.DataFrame:
    """Calculate rolling correlation between FSI and volatility."""
    if ibov_df is None:
        return pd.DataFrame()

    # Aggregate IBOV to weekly
    ibov = ibov_df.copy()
    ibov['week'] = ibov['date'].dt.to_period('W').dt.start_time

    # Prefer realized volatility columns over 'volume'
    vol_cols = [c for c in ibov.columns if 'realized_vol' in c.lower() or 'volatility' in c.lower()]
    if not vol_cols:
        # Fallback to any column with 'vol' but exclude 'volume'
        vol_cols = [c for c in ibov.columns if 'vol' in c.lower() and c.lower() != 'volume']
    if not vol_cols:
        # Last resort: use 'volume'
        vol_cols = [c for c in ibov.columns if 'vol' in c.lower()]

    if not vol_cols:
        return pd.DataFrame()

    # Prefer 21-day realized volatility if available
    vol_col = next((c for c in vol_cols if '21' in c), vol_cols[0])
    vol_weekly = ibov.groupby('week')[vol_col].mean().reset_index()
    vol_weekly.columns = ['date', 'volatility']

    # Merge with FSI - use outer join to keep all dates, then handle NaN
    merged = fsi_df.merge(vol_weekly, on='date', how='outer')
    merged = merged.sort_values('date')
    merged = merged.set_index('date')

    # Interpolate volatility to fill small gaps (up to 4 weeks)
    merged['volatility'] = merged['volatility'].interpolate(method='linear', limit=4)

    fsi_cols = [c for c in merged.columns if 'fsi' in c.lower()]

    # Calculate rolling correlation with min_periods to handle sparse data
    rolling_corr = pd.DataFrame(index=merged.index)
    min_periods = max(window // 2, 26)  # At least 26 weeks or half the window
    for col in fsi_cols:
        # Only calculate where both FSI and volatility have data
        valid_mask = merged[col].notna() & merged['volatility'].notna()
        if valid_mask.sum() >= min_periods:
            rolling_corr[col] = merged[col].rolling(window, min_periods=min_periods).corr(merged['volatility'])

    # Drop rows where all correlations are NaN
    rolling_corr = rolling_corr.dropna(how='all')

    return rolling_corr.reset_index()


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_fsi_comparison(fsi_df: pd.DataFrame, output_path: Path = None):
    """Create comparison plot of all FSI series."""
    fsi_cols = [c for c in fsi_df.columns if 'fsi' in c.lower()]

    fig, ax = plt.subplots(figsize=(14, 6))

    for col in fsi_cols:
        color = COLORS.get(col, 'gray')
        label = FSI_LABELS.get(col, col)
        ax.plot(fsi_df['date'], fsi_df[col], color=color, linewidth=1.2, label=label, alpha=0.8)

    # Neutral line
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    # Crisis shading
    for (start, end), name in CRISIS_EPISODES.items():
        start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
        if fsi_df['date'].min() <= end_dt and fsi_df['date'].max() >= start_dt:
            ax.axvspan(start_dt, end_dt, alpha=0.08, color=COLORS['crisis'])

    ax.set_xlabel('Date')
    ax.set_ylabel('FSI (0-1 scale)')
    ax.set_ylim(0, 1)
    ax.set_title('Comparison of Financial Stress Indices for Brazil')
    ax.legend(loc='upper left')

    # Time axis
    date_range = (fsi_df['date'].max() - fsi_df['date'].min()).days / 365
    if date_range > 10:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: Path = None):
    """Create correlation heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Rename columns for display
    display_labels = [FSI_LABELS.get(c, c) for c in corr_matrix.columns]
    corr_display = corr_matrix.copy()
    corr_display.columns = display_labels
    corr_display.index = display_labels

    # Create heatmap
    im = ax.imshow(corr_display.values, cmap='RdBu_r', vmin=-1, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Correlation', rotation=-90, va='bottom')

    # Add labels
    ax.set_xticks(np.arange(len(display_labels)))
    ax.set_yticks(np.arange(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha='right')
    ax.set_yticklabels(display_labels)

    # Add correlation values
    for i in range(len(display_labels)):
        for j in range(len(display_labels)):
            text = ax.text(j, i, f'{corr_display.values[i, j]:.2f}',
                          ha='center', va='center', color='white' if abs(corr_display.values[i, j]) > 0.5 else 'black')

    ax.set_title('Correlation Matrix of FSI Methods')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


def plot_rolling_correlation(rolling_df: pd.DataFrame, output_path: Path = None):
    """Plot rolling correlation with volatility."""
    if rolling_df.empty:
        return

    fsi_cols = [c for c in rolling_df.columns if 'fsi' in c.lower()]

    fig, ax = plt.subplots(figsize=(14, 5))

    for col in fsi_cols:
        color = COLORS.get(col, 'gray')
        label = FSI_LABELS.get(col, col)
        ax.plot(rolling_df['date'], rolling_df[col], color=color, linewidth=1.2, label=label, alpha=0.8)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Rolling Correlation (52-week)')
    ax.set_ylim(-1, 1)
    ax.set_title('Rolling Correlation between FSI and Market Volatility')
    ax.legend(loc='lower right')

    # Time axis
    date_range = (rolling_df['date'].max() - rolling_df['date'].min()).days / 365
    if date_range > 10:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


def plot_crisis_analysis(crisis_df: pd.DataFrame, output_path: Path = None):
    """Plot FSI levels during crisis periods."""
    if crisis_df.empty:
        return

    fsi_cols = [c for c in crisis_df.columns if '_mean' in c]
    n_fsi = len(fsi_cols)

    if n_fsi == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(crisis_df))
    width = 0.8 / n_fsi

    for i, col in enumerate(fsi_cols):
        fsi_name = col.replace('_mean', '')
        color = COLORS.get(fsi_name, 'gray')
        label = FSI_LABELS.get(fsi_name, fsi_name)
        offset = (i - n_fsi/2 + 0.5) * width
        ax.bar(x + offset, crisis_df[col], width, label=label, color=color, alpha=0.8)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral')
    ax.set_xlabel('Crisis Period')
    ax.set_ylabel('Mean FSI')
    ax.set_ylim(0, 1)
    ax.set_title('FSI Levels During Historical Crisis Periods')
    ax.set_xticks(x)
    ax.set_xticklabels(crisis_df['Crisis'], rotation=45, ha='right')
    ax.legend(loc='upper right')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


def plot_fsi_vs_cds(fsi_df: pd.DataFrame, cds_df: pd.DataFrame,
                    output_path: Path = None):
    """
    Plot FSI comparison with Brazil CDS 5Y benchmark.

    CDS is a market-based measure of sovereign credit risk and serves
    as an important benchmark for validating FSI methods.
    """
    if cds_df is None or cds_df.empty:
        return

    # Merge data
    merged = fsi_df.merge(cds_df[['date', 'cds_normalized', 'cds_5y']], on='date', how='inner')

    if len(merged) < 10:
        print("  Insufficient overlapping data for CDS comparison")
        return

    fsi_cols = [c for c in merged.columns if 'fsi' in c.lower() and 'smoothed' not in c.lower()]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot 1: FSI vs CDS (normalized)
    ax1 = axes[0]
    for col in fsi_cols[:3]:  # Limit to 3 FSI series for clarity
        color = COLORS.get(col, 'gray')
        label = FSI_LABELS.get(col, col)
        ax1.plot(merged['date'], merged[col], color=color,
                linewidth=1.2, label=label, alpha=0.7)

    ax1.plot(merged['date'], merged['cds_normalized'], color=COLORS['cds'],
            linewidth=2, label='CDS 5Y (normalized)', linestyle='--')

    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.set_ylabel('Normalized Value (0-1)')
    ax1.set_ylim(0, 1)
    ax1.set_title('Financial Stress Indices vs Brazil CDS 5Y Spread')
    ax1.legend(loc='upper left')

    # Add crisis shading
    for (start, end), name in CRISIS_EPISODES.items():
        start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
        if merged['date'].min() <= end_dt and merged['date'].max() >= start_dt:
            ax1.axvspan(start_dt, end_dt, alpha=0.08, color=COLORS['crisis'])

    # Plot 2: CDS in basis points (right axis) with Combined FSI
    ax2 = axes[1]

    if 'combined_fsi' in merged.columns:
        ax2.plot(merged['date'], merged['combined_fsi'], color=COLORS['combined'],
                linewidth=1.5, label='Combined FSI')
    elif fsi_cols:
        ax2.plot(merged['date'], merged[fsi_cols[0]], color=COLORS.get(fsi_cols[0], 'blue'),
                linewidth=1.5, label=FSI_LABELS.get(fsi_cols[0], fsi_cols[0]))

    ax2.set_ylabel('FSI (0-1 scale)')
    ax2.set_ylim(0, 1)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(merged['date'], merged['cds_5y'], color=COLORS['cds'],
                  linewidth=1.5, alpha=0.7, label='CDS 5Y')
    ax2_twin.set_ylabel('CDS Spread (bps)', color=COLORS['cds'])
    ax2_twin.tick_params(axis='y', labelcolor=COLORS['cds'])

    # Calculate and display correlation
    if 'combined_fsi' in merged.columns:
        corr = merged['combined_fsi'].corr(merged['cds_normalized'])
        ax2.set_title(f'Combined FSI vs CDS 5Y (r = {corr:.3f})')
    else:
        ax2.set_title('FSI vs CDS 5Y Spread')

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Time axis formatting
    date_range = (merged['date'].max() - merged['date'].min()).days / 365
    if date_range > 10:
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    else:
        ax2.xaxis.set_major_locator(mdates.YearLocator(1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


def plot_smoothed_comparison(raw_df: pd.DataFrame, smoothed_df: pd.DataFrame,
                             output_path: Path = None):
    """Plot comparison of raw vs Kalman-smoothed FSI."""
    # Find FSI columns that have smoothed versions
    fsi_cols = [c for c in raw_df.columns if 'fsi' in c.lower()
                and c != 'date' and f'{c}_smoothed' in smoothed_df.columns]

    if not fsi_cols:
        return

    n_cols = min(len(fsi_cols), 3)  # Limit to 3 for space
    fig, axes = plt.subplots(n_cols, 1, figsize=(14, 4*n_cols), sharex=True)

    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(fsi_cols[:n_cols]):
        ax = axes[i]
        smoothed_col = f'{col}_smoothed'

        # Raw
        ax.plot(raw_df['date'], raw_df[col], color='#A0A0A0',
               linewidth=0.8, alpha=0.6, label='Raw')

        # Smoothed - using "with Kalman Filter" terminology
        ax.plot(smoothed_df['date'], smoothed_df[smoothed_col], color=COLORS['smoothed'],
               linewidth=1.5, label='With Kalman Filter')

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('FSI (0-1 scale)')
        ax.set_ylim(0, 1)
        ax.set_title(f'{FSI_LABELS.get(col, col)} with Kalman Filter')
        ax.legend(loc='upper right')

        # Calculate variance reduction
        raw_var = raw_df[col].var()
        smooth_var = smoothed_df[smoothed_col].var()
        reduction = (1 - smooth_var/raw_var) * 100 if raw_var > 0 else 0
        ax.text(0.02, 0.98, f'Variance reduced: {reduction:.1f}%',
               transform=ax.transAxes, va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Time axis
    date_range = (raw_df['date'].max() - raw_df['date'].min()).days / 365
    if date_range > 10:
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    else:
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


def plot_single_fsi_kalman(raw_df: pd.DataFrame, smoothed_df: pd.DataFrame,
                           fsi_col: str, output_path: Path = None):
    """
    Plot a single FSI with Kalman Filter smoothing.

    Parameters
    ----------
    raw_df : DataFrame
        Raw FSI data with 'date' column
    smoothed_df : DataFrame
        Smoothed FSI data with '_smoothed' suffix columns
    fsi_col : str
        Name of the FSI column to plot (e.g., 'news_fsi', 'combined_fsi')
    output_path : Path
        Path to save the plot
    """
    smoothed_col = f'{fsi_col}_smoothed'

    if fsi_col not in raw_df.columns:
        print(f"  Warning: Column {fsi_col} not found in raw data")
        return
    if smoothed_col not in smoothed_df.columns:
        print(f"  Warning: Column {smoothed_col} not found in smoothed data")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Raw FSI
    ax.plot(raw_df['date'], raw_df[fsi_col], color='#A0A0A0',
           linewidth=0.8, alpha=0.6, label='Raw')

    # Smoothed FSI
    ax.plot(smoothed_df['date'], smoothed_df[smoothed_col], color=COLORS['smoothed'],
           linewidth=1.5, label='With Kalman Filter')

    # Add crisis shading
    for (start, end), name in CRISIS_EPISODES.items():
        start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
        if raw_df['date'].min() <= end_dt and raw_df['date'].max() >= start_dt:
            ax.axvspan(start_dt, end_dt, alpha=0.08, color=COLORS['crisis'])

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('FSI (0-1 scale)')
    ax.set_ylim(0, 1)
    ax.set_title(f'{FSI_LABELS.get(fsi_col, fsi_col)} with Kalman Filter')
    ax.legend(loc='upper left')

    # Calculate and display variance reduction
    raw_var = raw_df[fsi_col].var()
    smooth_var = smoothed_df[smoothed_col].var()
    reduction = (1 - smooth_var/raw_var) * 100 if raw_var > 0 else 0
    ax.text(0.02, 0.98, f'Variance reduced: {reduction:.1f}%',
           transform=ax.transAxes, va='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Time axis formatting
    date_range = (raw_df['date'].max() - raw_df['date'].min()).days / 365
    if date_range > 10:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(fsi_df: pd.DataFrame, ibov_df: pd.DataFrame,
                    desc_stats: pd.DataFrame, corr_matrix: pd.DataFrame,
                    vol_corr: Dict, crisis_df: pd.DataFrame,
                    output_path: Path = None) -> str:
    """Generate comprehensive analysis report."""
    lines = []
    lines.append("=" * 70)
    lines.append("FINANCIAL STRESS INDEX ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # Data overview
    lines.append("\n" + "=" * 70)
    lines.append("1. DATA OVERVIEW")
    lines.append("=" * 70)
    lines.append(f"\nPeriod: {fsi_df['date'].min().date()} to {fsi_df['date'].max().date()}")
    lines.append(f"Total weeks: {len(fsi_df)}")

    fsi_cols = [c for c in fsi_df.columns if 'fsi' in c.lower()]
    for col in fsi_cols:
        n_valid = fsi_df[col].notna().sum()
        lines.append(f"  {FSI_LABELS.get(col, col)}: {n_valid} observations")

    # Descriptive statistics
    lines.append("\n" + "=" * 70)
    lines.append("2. DESCRIPTIVE STATISTICS")
    lines.append("=" * 70)
    lines.append("\n" + desc_stats.to_string(index=False))

    # Correlation analysis
    lines.append("\n" + "=" * 70)
    lines.append("3. CORRELATION ANALYSIS")
    lines.append("=" * 70)

    lines.append("\n3.1 Inter-FSI Correlations:")
    lines.append("-" * 40)
    for i, col1 in enumerate(corr_matrix.columns):
        for col2 in corr_matrix.columns[i+1:]:
            r = corr_matrix.loc[col1, col2]
            lines.append(f"  {FSI_LABELS.get(col1, col1)} vs {FSI_LABELS.get(col2, col2)}: r = {r:.3f}")

    if vol_corr:
        lines.append("\n3.2 Correlation with IBOVESPA Volatility:")
        lines.append("-" * 40)
        for col, vals in vol_corr.items():
            label = FSI_LABELS.get(col, col)
            r = vals['pearson_r']
            p = vals['pearson_p']
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            lines.append(f"  {label}:")
            lines.append(f"    Pearson:  r = {r:.3f} (p = {p:.4f}) {sig}")
            lines.append(f"    Spearman: r = {vals['spearman_r']:.3f} (p = {vals['spearman_p']:.4f})")
            lines.append(f"    N = {vals['n']}")

    # Crisis analysis
    lines.append("\n" + "=" * 70)
    lines.append("4. CRISIS PERIOD ANALYSIS")
    lines.append("=" * 70)
    lines.append("\nMean FSI levels during historical crises:")
    lines.append("-" * 40)

    if not crisis_df.empty:
        for _, row in crisis_df.iterrows():
            lines.append(f"\n  {row['Crisis']} ({row['Period']}):")
            lines.append(f"    Weeks: {row['Weeks']}")
            for col in crisis_df.columns:
                if '_mean' in col:
                    fsi_name = col.replace('_mean', '')
                    label = FSI_LABELS.get(fsi_name, fsi_name)
                    lines.append(f"    {label}: {row[col]:.3f}")

    # Interpretation
    lines.append("\n" + "=" * 70)
    lines.append("5. INTERPRETATION GUIDE")
    lines.append("=" * 70)
    lines.append("""
FSI Scale (0-1):
  0.0 - 0.2: Very low stress (calm market conditions)
  0.2 - 0.4: Low stress (below average)
  0.4 - 0.6: Moderate stress (around neutral)
  0.6 - 0.8: High stress (above average)
  0.8 - 1.0: Very high stress (crisis conditions)

Methodology Notes:
  - Dictionary FSI: Uses pre-defined stress-related Google Trends queries
  - ML FSI: Uses LASSO regression to identify predictive queries
  - News FSI: Uses financial news article analysis

Correlation Interpretation:
  *** p < 0.001 (highly significant)
  **  p < 0.01  (very significant)
  *   p < 0.05  (significant)
""")

    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  Saved: {output_path}")

    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive FSI Analysis')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--smooth', action='store_true', help='Apply Kalman smoothing')
    parser.add_argument('--cds', action='store_true', help='Include CDS benchmark analysis')
    args = parser.parse_args()

    print("=" * 70)
    print("FINANCIAL STRESS INDEX ANALYSIS")
    print("=" * 70)

    # Load data
    print("\nLoading FSI data...")
    fsi_data = load_all_fsi()

    if len(fsi_data) == 0:
        print("\n  ERROR: No FSI data found. Run individual FSI scripts first.")
        return

    ibov_df = load_ibovespa()

    # Merge FSI data
    print("\nMerging FSI series...")
    fsi_df = merge_fsi_data(fsi_data)
    print(f"  Combined dataset: {len(fsi_df)} weeks")

    # Apply Kalman smoothing if requested
    smoothed_df = None
    if args.smooth:
        print("\nApplying Kalman smoothing...")
        smoothed_df = apply_kalman_smoothing_to_fsi(fsi_df)

        # Save smoothed data
        smoothed_path = OUTPUT_DIR / 'fsi_smoothed.csv'
        smoothed_df.to_csv(smoothed_path, index=False)
        print(f"  Saved: {smoothed_path}")

    # Load CDS data if requested
    cds_df = None
    cds_corr = {}
    if args.cds:
        print("\nLoading CDS benchmark data...")
        cds_raw = load_brazil_cds()
        if cds_raw is not None:
            cds_df = process_cds_data(cds_raw)
            cds_corr = calculate_cds_correlation(fsi_df, cds_df)

    # Statistical analysis
    print("\nCalculating statistics...")
    desc_stats = calculate_descriptive_stats(fsi_df)
    corr_matrix = calculate_correlation_matrix(fsi_df)
    vol_corr = calculate_volatility_correlation(fsi_df, ibov_df)
    crisis_df = analyze_crisis_periods(fsi_df)
    rolling_df = calculate_rolling_correlation(fsi_df, ibov_df)

    # Generate report
    print("\nGenerating report...")
    report_path = OUTPUT_DIR / 'fsi_analysis_report.txt'
    report = generate_report(fsi_df, ibov_df, desc_stats, corr_matrix,
                            vol_corr, crisis_df, report_path)

    # Add CDS correlation to report if available
    if cds_corr:
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("APPENDIX: CDS BENCHMARK ANALYSIS\n")
            f.write("=" * 70 + "\n")
            f.write("\nCorrelation with Brazil CDS 5Y Spread:\n")
            f.write("-" * 40 + "\n")
            for col, vals in cds_corr.items():
                label = FSI_LABELS.get(col, col)
                r = vals['pearson_r']
                p = vals['pearson_p']
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                f.write(f"  {label}:\n")
                f.write(f"    Pearson:  r = {r:.3f} (p = {p:.4f}) {sig}\n")
                f.write(f"    Spearman: r = {vals['spearman_r']:.3f}\n")
                f.write(f"    N = {vals['n']}\n\n")

    # Print summary to console
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nDescriptive Statistics:")
    print(desc_stats.to_string(index=False))

    if vol_corr:
        print(f"\nCorrelation with IBOVESPA Volatility:")
        for col, vals in vol_corr.items():
            label = FSI_LABELS.get(col, col)
            print(f"  {label}: r = {vals['pearson_r']:.3f}")

    if cds_corr:
        print(f"\nCorrelation with CDS 5Y:")
        for col, vals in cds_corr.items():
            label = FSI_LABELS.get(col, col)
            print(f"  {label}: r = {vals['pearson_r']:.3f}")

    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        plot_fsi_comparison(fsi_df, PLOTS_DIR / 'fsi_comparison.png')
        plot_correlation_heatmap(corr_matrix, PLOTS_DIR / 'fsi_correlations.png')
        plot_crisis_analysis(crisis_df, PLOTS_DIR / 'fsi_crisis_analysis.png')
        if not rolling_df.empty:
            plot_rolling_correlation(rolling_df, PLOTS_DIR / 'fsi_rolling_correlation.png')

        # CDS comparison plot
        if cds_df is not None:
            plot_fsi_vs_cds(fsi_df, cds_df, PLOTS_DIR / 'fsi_vs_cds.png')

        # Smoothed comparison plot
        if smoothed_df is not None:
            plot_smoothed_comparison(fsi_df, smoothed_df, PLOTS_DIR / 'fsi_kalman_smoothed.png')

            # Individual Kalman Filter plots for each FSI type
            for fsi_col in ['news_fsi', 'combined_fsi', 'dict_fsi', 'ml_fsi']:
                if fsi_col in fsi_df.columns and f'{fsi_col}_smoothed' in smoothed_df.columns:
                    plot_single_fsi_kalman(
                        fsi_df, smoothed_df, fsi_col,
                        PLOTS_DIR / f'{fsi_col}_kalman.png'
                    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n   Output files:")
    print(f"   - {report_path}")
    print(f"   - {PLOTS_DIR}/fsi_comparison.png")
    print(f"   - {PLOTS_DIR}/fsi_correlations.png")
    print(f"   - {PLOTS_DIR}/fsi_crisis_analysis.png")
    print(f"   - {PLOTS_DIR}/fsi_rolling_correlation.png")
    if cds_df is not None:
        print(f"   - {PLOTS_DIR}/fsi_vs_cds.png")
    if smoothed_df is not None:
        print(f"   - {PLOTS_DIR}/fsi_kalman_smoothed.png")
        print(f"   - {OUTPUT_DIR}/fsi_smoothed.csv")
    print("=" * 70)


if __name__ == '__main__':
    main()
