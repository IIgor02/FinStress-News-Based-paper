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
6. Publication-quality comparison plots

Usage:
    python scripts/analyze_fsi.py

Output:
    output/fsi_analysis_report.txt
    output/plots/fsi_comparison.png
    output/plots/fsi_correlations.png
    output/plots/fsi_crisis_analysis.png
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
}

FSI_LABELS = {
    'dict_fsi': 'Dictionary FSI (Da et al.)',
    'ml_fsi': 'ML FSI (García et al.)',
    'news_fsi': 'News FSI (Baker et al.)',
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

def load_all_fsi() -> Dict[str, pd.DataFrame]:
    """Load all available FSI series."""
    fsi_data = {}

    # Dictionary FSI
    dict_path = OUTPUT_DIR / 'results' / 'fsi_weekly.csv'
    if dict_path.exists():
        df = pd.read_csv(dict_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'fsi': 'dict_fsi'})
        fsi_data['dict_fsi'] = df[['date', 'dict_fsi']]
        print(f"  Dictionary FSI: {len(df)} weeks")

    # ML FSI
    for path in [OUTPUT_DIR / 'ml_fsi_weekly.csv', OUTPUT_DIR / 'results' / 'fsi_ml_weekly.csv']:
        if path.exists():
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            if 'ml_fsi' in df.columns:
                fsi_data['ml_fsi'] = df[['date', 'ml_fsi']]
            elif 'fsi' in df.columns:
                df = df.rename(columns={'fsi': 'ml_fsi'})
                fsi_data['ml_fsi'] = df[['date', 'ml_fsi']]
            print(f"  ML FSI: {len(df)} weeks")
            break

    # News FSI
    news_path = OUTPUT_DIR / 'news_fsi_weekly.csv'
    if news_path.exists():
        df = pd.read_csv(news_path)
        df['date'] = pd.to_datetime(df['date'])
        fsi_data['news_fsi'] = df[['date', 'news_fsi']]
        print(f"  News FSI: {len(df)} weeks")

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


def merge_fsi_data(fsi_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all FSI series into a single DataFrame."""
    merged = None
    for name, df in fsi_data.items():
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
    vol_cols = [c for c in ibov.columns if 'vol' in c.lower()]

    if not vol_cols:
        return pd.DataFrame()

    vol_col = vol_cols[0]
    vol_weekly = ibov.groupby('week')[vol_col].mean().reset_index()
    vol_weekly.columns = ['date', 'volatility']

    # Merge with FSI
    merged = fsi_df.merge(vol_weekly, on='date', how='inner')
    merged = merged.set_index('date')

    fsi_cols = [c for c in merged.columns if 'fsi' in c.lower()]

    # Calculate rolling correlation
    rolling_corr = pd.DataFrame(index=merged.index)
    for col in fsi_cols:
        rolling_corr[col] = merged[col].rolling(window).corr(merged['volatility'])

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
  - Dictionary FSI (Da et al.): Uses pre-defined stress-related Google Trends queries
  - ML FSI (Garcia et al.): Uses LASSO regression to identify predictive queries
  - News FSI (Baker et al.): Uses financial news article analysis

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

    # Print summary to console
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nDescriptive Statistics:")
    print(desc_stats.to_string(index=False))

    if vol_corr:
        print(f"\nCorrelation with Volatility:")
        for col, vals in vol_corr.items():
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

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n   Output files:")
    print(f"   - {report_path}")
    print(f"   - {PLOTS_DIR}/fsi_comparison.png")
    print(f"   - {PLOTS_DIR}/fsi_correlations.png")
    print(f"   - {PLOTS_DIR}/fsi_crisis_analysis.png")
    print(f"   - {PLOTS_DIR}/fsi_rolling_correlation.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
