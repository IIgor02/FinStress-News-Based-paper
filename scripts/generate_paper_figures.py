#!/usr/bin/env python3
"""
Generate Publication-Quality Figures
=====================================
Generates all figures for the paper with standardized academic styling.

This script should be run after all FSI indices have been calculated.
It produces publication-ready figures in grayscale/minimal color scheme.

Usage:
    python scripts/generate_paper_figures.py

Output:
    output/figures/  - Publication-ready figures
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

# Import styling
from scripts.plot_style import (
    apply_style, COLORS, LINE_STYLES, FIGSIZE,
    save_figure, add_zero_line, add_crisis_shading,
    format_date_axis, add_confidence_band, plot_irf,
    CRISIS_PERIODS
)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Apply academic style
apply_style()

# Output directory for figures
FIGURES_DIR = PROJECT_DIR / 'output' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_aligned_data():
    """Load the aligned monthly FSI data."""
    path = PROJECT_DIR / 'output' / 'fsi_monthly_aligned.csv'
    if not path.exists():
        raise FileNotFoundError(f"Aligned data not found: {path}\nRun: python scripts/causality_analysis.py")

    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_weekly_fsi():
    """Load weekly FSI data."""
    path = PROJECT_DIR / 'output' / 'results' / 'fsi_weekly.csv'
    if path.exists():
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None


# =============================================================================
# FIGURE 1: FSI Time Series (Main Index)
# =============================================================================

def figure_1_fsi_timeseries(df):
    """
    Figure 1: Dictionary-Based Financial Stress Index for Brazil

    Shows the main FSI time series with crisis period shading.
    """
    print("\nGenerating Figure 1: FSI Time Series...")

    fig, ax = plt.subplots(figsize=FIGSIZE['double'])

    # Main FSI line
    ax.plot(df['date'], df['dict_fsi'], color=COLORS['black'],
            linewidth=1.2, label='Dictionary FSI', zorder=3)

    # 12-month moving average
    ma = df['dict_fsi'].rolling(12, min_periods=1).mean()
    ax.plot(df['date'], ma, color=COLORS['dark_gray'],
            linewidth=1.8, linestyle='--', label='12-Month MA', zorder=4)

    # Neutral line
    ax.axhline(y=0.5, color=COLORS['gray'], linestyle=':',
               linewidth=0.8, alpha=0.7, label='Neutral (0.5)')

    # Crisis shading
    for (start, end), label in CRISIS_PERIODS.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if df['date'].min() <= end_dt and df['date'].max() >= start_dt:
            ax.axvspan(start_dt, end_dt, alpha=0.15,
                       color=COLORS['fill_dark'], zorder=0)

    ax.set_xlabel('Date')
    ax.set_ylabel('Financial Stress Index (0-1)')
    ax.set_title('Figure 1: Dictionary-Based Financial Stress Index for Brazil')
    ax.set_ylim(0, 1.05)

    # Format x-axis
    years = (df['date'].max() - df['date'].min()).days / 365
    format_date_axis(ax, years)

    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)

    save_figure(fig, 'figure_1_fsi_timeseries', FIGURES_DIR)
    plt.close()


# =============================================================================
# FIGURE 2: FSI vs CDS Comparison
# =============================================================================

def figure_2_fsi_vs_cds(df):
    """
    Figure 2: FSI vs Brazil 5-Year CDS Spread

    Dual-axis plot comparing FSI with CDS as validation.
    """
    print("Generating Figure 2: FSI vs CDS...")

    fig, ax1 = plt.subplots(figsize=FIGSIZE['double'])

    # FSI (left axis)
    line1, = ax1.plot(df['date'], df['dict_fsi'], color=COLORS['black'],
                      linewidth=1.5, label='Dictionary FSI', zorder=3)
    ax1.set_ylabel('Financial Stress Index (0-1)', color=COLORS['black'])
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='y', labelcolor=COLORS['black'])

    # CDS (right axis)
    ax2 = ax1.twinx()
    line2, = ax2.plot(df['date'], df['cds_5y'], color=COLORS['gray'],
                      linewidth=1.5, linestyle='--', label='CDS 5Y (bps)', zorder=2)
    ax2.set_ylabel('CDS Spread (basis points)', color=COLORS['gray'])
    ax2.tick_params(axis='y', labelcolor=COLORS['gray'])

    # Crisis shading
    for (start, end), label in CRISIS_PERIODS.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if df['date'].min() <= end_dt and df['date'].max() >= start_dt:
            ax1.axvspan(start_dt, end_dt, alpha=0.15,
                        color=COLORS['fill_dark'], zorder=0)

    # Calculate correlation
    corr = df['dict_fsi'].corr(df['cds_5y'])

    ax1.set_xlabel('Date')
    ax1.set_title(f'Figure 2: Financial Stress Index vs Brazil 5-Year CDS Spread (r = {corr:.2f})')

    # Format x-axis
    years = (df['date'].max() - df['date'].min()).days / 365
    format_date_axis(ax1, years)

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', framealpha=0.95)

    ax1.grid(True, alpha=0.3)

    save_figure(fig, 'figure_2_fsi_vs_cds', FIGURES_DIR)
    plt.close()


# =============================================================================
# FIGURE 3: Impulse Response Function (Main Result)
# =============================================================================

def figure_3_irf_main(df):
    """
    Figure 3: Impulse Response Function - FSI Shock to CDS

    Main econometric result using Local Projections.
    """
    print("Generating Figure 3: IRF (Main Result)...")

    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    df = df.copy()

    # Calculate transformations
    df['cds_log_return'] = np.log(df['cds_5y'] / df['cds_5y'].shift(1))
    df['fsi_change'] = df['dict_fsi'].diff()

    # Standardize shock
    shock_std = df['fsi_change'].std()
    df['fsi_shock'] = df['fsi_change'] / shock_std

    max_horizon = 12
    n_lags = 4

    horizons = np.arange(max_horizon + 1)
    irfs = []
    ses = []

    for h in range(max_horizon + 1):
        # Cumulative response
        df['y_forward'] = df['cds_log_return'].rolling(h + 1).sum().shift(-h)

        X_cols = ['fsi_shock']
        for lag in range(1, n_lags + 1):
            df[f'cds_lag_{lag}'] = df['cds_log_return'].shift(lag)
            X_cols.append(f'cds_lag_{lag}')

        reg_df = df[['y_forward'] + X_cols].dropna()

        if len(reg_df) < 30:
            irfs.append(np.nan)
            ses.append(np.nan)
            continue

        y = reg_df['y_forward']
        X = add_constant(reg_df[X_cols])

        model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': h + 1})

        irfs.append(model.params['fsi_shock'])
        ses.append(model.bse['fsi_shock'])

    irfs = np.array(irfs)
    ses = np.array(ses)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE['irf'])

    plot_irf(ax, horizons, irfs, ses,
             title='Figure 3: Impulse Response - FSI Shock to CDS Spread',
             ylabel='Cumulative CDS Log-Return Response')

    ax.set_xlabel('Months After Shock')

    # Add annotation for cumulative effect
    cum_irf = np.nansum(irfs)
    sig_months = np.sum(np.abs(irfs / ses) > 1.645)

    textstr = f'Cumulative Effect: {cum_irf:.2f}\nSignificant Months: {sig_months}/13'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9,
                 edgecolor=COLORS['light_gray'])
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    save_figure(fig, 'figure_3_irf_main', FIGURES_DIR)
    plt.close()

    return irfs, ses


# =============================================================================
# FIGURE 4: Methodology Comparison
# =============================================================================

def figure_4_methodology_comparison(df):
    """
    Figure 4: Comparison of FSI Construction Methods

    Shows why dictionary-based FSI outperforms ML and combined approaches.
    """
    print("Generating Figure 4: Methodology Comparison...")

    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    df = df.copy()
    df['cds_log_return'] = np.log(df['cds_5y'] / df['cds_5y'].shift(1))

    methods = [
        ('dict_fsi', 'Dictionary FSI'),
        ('ml_fsi', 'ML FSI'),
        ('combined_fsi', 'Combined FSI'),
    ]

    max_horizon = 12
    n_lags = 4

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for idx, (col, name) in enumerate(methods):
        ax = axes[idx]

        df[f'{col}_change'] = df[col].diff()
        shock_std = df[f'{col}_change'].std()
        df[f'{col}_shock'] = df[f'{col}_change'] / shock_std

        horizons = np.arange(max_horizon + 1)
        irfs = []
        ses = []

        for h in range(max_horizon + 1):
            df['y_forward'] = df['cds_log_return'].rolling(h + 1).sum().shift(-h)

            X_cols = [f'{col}_shock']
            for lag in range(1, n_lags + 1):
                df[f'cds_lag_{lag}'] = df['cds_log_return'].shift(lag)
                X_cols.append(f'cds_lag_{lag}')

            reg_df = df[['y_forward'] + X_cols].dropna()

            if len(reg_df) < 30:
                irfs.append(np.nan)
                ses.append(np.nan)
                continue

            y = reg_df['y_forward']
            X = add_constant(reg_df[X_cols])

            model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': h + 1})

            irfs.append(model.params[f'{col}_shock'])
            ses.append(model.bse[f'{col}_shock'])

        irfs = np.array(irfs)
        ses = np.array(ses)

        # Plot
        ax.plot(horizons, irfs, color=COLORS['black'], linewidth=2,
                marker='o', markersize=4, zorder=4)

        # 95% CI
        ax.fill_between(horizons, irfs - 1.96*ses, irfs + 1.96*ses,
                        alpha=0.2, color=COLORS['gray'], zorder=2)
        # 90% CI
        ax.fill_between(horizons, irfs - 1.645*ses, irfs + 1.645*ses,
                        alpha=0.3, color=COLORS['dark_gray'], zorder=3)

        add_zero_line(ax)

        cum_irf = np.nansum(irfs)
        sig_months = np.sum(np.abs(irfs / ses) > 1.645)

        ax.set_title(f'{name}\n(Cum: {cum_irf:.2f}, Sig: {sig_months})')
        ax.set_xlabel('Months')
        if idx == 0:
            ax.set_ylabel('CDS Log-Return Response')

        ax.grid(True, alpha=0.3)

    plt.suptitle('Figure 4: Comparison of FSI Construction Methods',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    save_figure(fig, 'figure_4_methodology_comparison', FIGURES_DIR)
    plt.close()


# =============================================================================
# FIGURE 5: Crisis Period Analysis
# =============================================================================

def figure_5_crisis_analysis(df):
    """
    Figure 5: FSI Behavior During Crisis Periods

    Detailed view of FSI during major financial stress episodes.
    """
    print("Generating Figure 5: Crisis Period Analysis...")

    crises = [
        ('Global Financial Crisis', '2008-01-01', '2010-06-30'),
        ('Brazilian Recession', '2014-06-01', '2017-06-30'),
        ('COVID-19 Pandemic', '2019-09-01', '2021-03-31'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for idx, (title, start, end) in enumerate(crises):
        ax = axes[idx]

        mask = (df['date'] >= start) & (df['date'] <= end)
        crisis_df = df[mask].copy()

        # FSI
        ax.plot(crisis_df['date'], crisis_df['dict_fsi'],
                color=COLORS['black'], linewidth=1.5, label='FSI')

        # Normalized CDS for comparison
        cds_norm = (crisis_df['cds_5y'] - crisis_df['cds_5y'].min()) / \
                   (crisis_df['cds_5y'].max() - crisis_df['cds_5y'].min())
        ax.plot(crisis_df['date'], cds_norm,
                color=COLORS['gray'], linewidth=1.5, linestyle='--',
                label='CDS (normalized)')

        # Neutral line
        ax.axhline(y=0.5, color=COLORS['light_gray'], linestyle=':',
                   linewidth=0.8, alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel('Date')
        if idx == 0:
            ax.set_ylabel('Index Value (0-1)')

        # Format dates
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Figure 5: FSI Behavior During Major Crisis Periods',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    save_figure(fig, 'figure_5_crisis_analysis', FIGURES_DIR)
    plt.close()


# =============================================================================
# FIGURE 6: Predictive Correlation Analysis
# =============================================================================

def figure_6_predictive_correlation(df):
    """
    Figure 6: Predictive Correlation Analysis

    Shows FSI's ability to predict future CDS movements.
    """
    print("Generating Figure 6: Predictive Correlation...")

    df = df.copy()
    df['cds_log_return'] = np.log(df['cds_5y'] / df['cds_5y'].shift(1))
    df['fsi_change'] = df['dict_fsi'].diff()

    # Calculate correlations at different leads
    max_lead = 12
    correlations = []
    p_values = []

    for lead in range(max_lead + 1):
        df['cds_future'] = df['cds_log_return'].shift(-lead)
        valid = df[['fsi_change', 'cds_future']].dropna()

        if len(valid) > 20:
            corr, pval = stats.pearsonr(valid['fsi_change'], valid['cds_future'])
            correlations.append(corr)
            p_values.append(pval)
        else:
            correlations.append(np.nan)
            p_values.append(np.nan)

    leads = np.arange(max_lead + 1)
    correlations = np.array(correlations)
    p_values = np.array(p_values)

    fig, ax = plt.subplots(figsize=FIGSIZE['single'])

    # Bar plot
    colors = [COLORS['dark_gray'] if p < 0.05 else COLORS['light_gray']
              for p in p_values]

    bars = ax.bar(leads, correlations, color=colors, edgecolor=COLORS['black'],
                  linewidth=0.5, alpha=0.8)

    add_zero_line(ax)

    ax.set_xlabel('Forecast Horizon (Months)')
    ax.set_ylabel('Correlation')
    ax.set_title('Figure 6: FSI Change Correlation with Future CDS Returns')
    ax.set_xticks(leads)

    # Add significance legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['dark_gray'], edgecolor=COLORS['black'],
              label='Significant (p < 0.05)'),
        Patch(facecolor=COLORS['light_gray'], edgecolor=COLORS['black'],
              label='Not Significant'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.grid(True, alpha=0.3, axis='y')

    save_figure(fig, 'figure_6_predictive_correlation', FIGURES_DIR)
    plt.close()


# =============================================================================
# TABLE 1: Summary Statistics
# =============================================================================

def table_1_summary_statistics(df):
    """
    Generate summary statistics table for the paper.
    """
    print("Generating Table 1: Summary Statistics...")

    # FSI statistics
    stats_dict = {
        'Variable': [],
        'Mean': [],
        'Std. Dev.': [],
        'Min': [],
        'Max': [],
        'Observations': [],
    }

    variables = [
        ('dict_fsi', 'Dictionary FSI'),
        ('ml_fsi', 'ML FSI'),
        ('combined_fsi', 'Combined FSI'),
        ('cds_5y', 'CDS 5Y (bps)'),
        ('volatility', 'IBOV Volatility (%)'),
    ]

    for col, name in variables:
        if col in df.columns:
            stats_dict['Variable'].append(name)
            stats_dict['Mean'].append(f"{df[col].mean():.3f}")
            stats_dict['Std. Dev.'].append(f"{df[col].std():.3f}")
            stats_dict['Min'].append(f"{df[col].min():.3f}")
            stats_dict['Max'].append(f"{df[col].max():.3f}")
            stats_dict['Observations'].append(str(len(df[col].dropna())))

    stats_df = pd.DataFrame(stats_dict)

    # Save as CSV
    stats_df.to_csv(FIGURES_DIR / 'table_1_summary_statistics.csv', index=False)

    # Print formatted table
    print("\n  Table 1: Summary Statistics")
    print("  " + "=" * 70)
    print(stats_df.to_string(index=False))
    print("  " + "=" * 70)

    return stats_df


# =============================================================================
# TABLE 2: IRF Results
# =============================================================================

def table_2_irf_results(df):
    """
    Generate IRF results table for the paper.
    """
    print("Generating Table 2: IRF Results...")

    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    df = df.copy()
    df['cds_log_return'] = np.log(df['cds_5y'] / df['cds_5y'].shift(1))

    methods = [
        ('dict_fsi', 'Dictionary FSI'),
        ('ml_fsi', 'ML FSI'),
        ('combined_fsi', 'Combined FSI'),
    ]

    results = []
    max_horizon = 12
    n_lags = 4

    for col, name in methods:
        df[f'{col}_change'] = df[col].diff()
        shock_std = df[f'{col}_change'].std()
        df[f'{col}_shock'] = df[f'{col}_change'] / shock_std

        irfs = []
        pvals = []

        for h in range(max_horizon + 1):
            df['y_forward'] = df['cds_log_return'].rolling(h + 1).sum().shift(-h)

            X_cols = [f'{col}_shock']
            for lag in range(1, n_lags + 1):
                df[f'cds_lag_{lag}'] = df['cds_log_return'].shift(lag)
                X_cols.append(f'cds_lag_{lag}')

            reg_df = df[['y_forward'] + X_cols].dropna()

            if len(reg_df) < 30:
                irfs.append(np.nan)
                pvals.append(np.nan)
                continue

            y = reg_df['y_forward']
            X = add_constant(reg_df[X_cols])

            model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': h + 1})

            irfs.append(model.params[f'{col}_shock'])
            pvals.append(model.pvalues[f'{col}_shock'])

        results.append({
            'Method': name,
            'Cumulative IRF': f"{np.nansum(irfs):.3f}",
            'Significant Months (p<0.1)': int(np.sum(np.array(pvals) < 0.1)),
            'Peak Response': f"{np.nanmax(np.abs(irfs)):.3f}",
            'Peak Month': int(np.nanargmax(np.abs(irfs))),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(FIGURES_DIR / 'table_2_irf_results.csv', index=False)

    print("\n  Table 2: IRF Results Summary")
    print("  " + "=" * 70)
    print(results_df.to_string(index=False))
    print("  " + "=" * 70)

    return results_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate all publication figures."""
    print("=" * 70)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 70)
    print(f"\nOutput directory: {FIGURES_DIR}")

    # Load data
    try:
        df = load_aligned_data()
        print(f"Loaded {len(df)} monthly observations")
        print(f"Period: {df['date'].min().date()} to {df['date'].max().date()}")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return

    # Generate figures
    print("\n" + "-" * 50)
    print("FIGURES")
    print("-" * 50)

    figure_1_fsi_timeseries(df)
    figure_2_fsi_vs_cds(df)
    irfs, ses = figure_3_irf_main(df)
    figure_4_methodology_comparison(df)
    figure_5_crisis_analysis(df)
    figure_6_predictive_correlation(df)

    # Generate tables
    print("\n" + "-" * 50)
    print("TABLES")
    print("-" * 50)

    table_1_summary_statistics(df)
    table_2_irf_results(df)

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nGenerated files in: {FIGURES_DIR}")
    print("\nFigures:")
    for f in sorted(FIGURES_DIR.glob('figure_*.png')):
        print(f"  - {f.name}")
    print("\nTables:")
    for f in sorted(FIGURES_DIR.glob('table_*.csv')):
        print(f"  - {f.name}")
    print("=" * 70)


if __name__ == '__main__':
    main()
