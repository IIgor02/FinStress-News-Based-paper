#!/usr/bin/env python3
"""
Generate Publication-Quality Figures
=====================================
Generates all figures for the paper with standardized academic styling.

This script is SELF-CONTAINED - it creates all necessary data and figures
without requiring other scripts to be run first.

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
from matplotlib.patches import Patch

# Apply academic style
apply_style()

# Output directories
FIGURES_DIR = PROJECT_DIR / 'output' / 'figures'
OUTPUT_DIR = PROJECT_DIR / 'output'
DATA_DIR = PROJECT_DIR / 'data' / 'raw'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# FSI Index configurations
FSI_INDICES = {
    'dict_fsi': {'name': 'Dictionary FSI', 'short': 'dict'},
    'ml_fsi': {'name': 'ML FSI', 'short': 'ml'},
    'combined_fsi': {'name': 'Combined FSI', 'short': 'combined'},
}


# =============================================================================
# DATA CREATION (SELF-CONTAINED)
# =============================================================================

def create_aligned_monthly_data():
    """
    Create aligned monthly FSI data from raw sources.
    This ensures generate_paper_figures.py works standalone.
    """
    print("Creating aligned monthly data...")

    # Load FSI data
    fsi_path = OUTPUT_DIR / 'combined_fsi.csv'
    if not fsi_path.exists():
        # Try alternative paths
        for alt_path in [OUTPUT_DIR / 'combined_fsi_weighted.csv',
                         OUTPUT_DIR / 'results' / 'fsi_weekly.csv']:
            if alt_path.exists():
                fsi_path = alt_path
                break

    if not fsi_path.exists():
        raise FileNotFoundError(f"No FSI data found. Run: python scripts/run_fsi.py && python scripts/combined_fsi.py")

    fsi_df = pd.read_csv(fsi_path)
    fsi_df['date'] = pd.to_datetime(fsi_df['date'])

    # Aggregate to monthly
    fsi_df['month'] = fsi_df['date'].dt.to_period('M')
    fsi_monthly = fsi_df.groupby('month').mean(numeric_only=True).reset_index()
    fsi_monthly['date'] = fsi_monthly['month'].dt.to_timestamp()
    fsi_monthly = fsi_monthly.drop(columns=['month'])

    # Load CDS data
    cds_loaded = False

    # Try Investing.com format
    cds_investing = PROJECT_DIR / 'Brasil CDS 5 Anos USD - Visão Geral.csv'
    if cds_investing.exists():
        try:
            cds_df = pd.read_csv(cds_investing, encoding='utf-8-sig')
            cds_df = cds_df.rename(columns={'Data': 'date', 'Último': 'cds_5y'})
            cds_df['date'] = pd.to_datetime(cds_df['date'], format='%d.%m.%Y')
            cds_df['cds_5y'] = cds_df['cds_5y'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            cds_df['cds_5y'] = pd.to_numeric(cds_df['cds_5y'], errors='coerce')
            cds_df = cds_df.dropna(subset=['cds_5y'])
            cds_loaded = True
        except:
            pass

    # Try standard format
    if not cds_loaded:
        cds_path = DATA_DIR / 'brazil_cds_5y.csv'
        if cds_path.exists():
            cds_df = pd.read_csv(cds_path)
            cds_df['date'] = pd.to_datetime(cds_df['date'])
            cds_loaded = True

    if cds_loaded:
        cds_df['month'] = cds_df['date'].dt.to_period('M')
        cds_monthly = cds_df.groupby('month')['cds_5y'].mean().reset_index()
        cds_monthly['date'] = cds_monthly['month'].dt.to_timestamp()
        cds_monthly = cds_monthly.drop(columns=['month'])
        fsi_monthly = fsi_monthly.merge(cds_monthly, on='date', how='outer')

    # Load IBOVESPA volatility
    ibov_path = DATA_DIR / 'ibovespa_data.csv'
    if ibov_path.exists():
        ibov_df = pd.read_csv(ibov_path)
        ibov_df['date'] = pd.to_datetime(ibov_df['date'], utc=True).dt.tz_localize(None)
        ibov_df['month'] = ibov_df['date'].dt.to_period('M')
        vol_monthly = ibov_df.groupby('month')['realized_vol_21d'].mean().reset_index()
        vol_monthly['date'] = vol_monthly['month'].dt.to_timestamp()
        vol_monthly = vol_monthly.rename(columns={'realized_vol_21d': 'volatility'})
        vol_monthly = vol_monthly.drop(columns=['month'])
        fsi_monthly = fsi_monthly.merge(vol_monthly, on='date', how='outer')

    # Sort and filter
    fsi_monthly = fsi_monthly.sort_values('date')
    fsi_monthly = fsi_monthly[fsi_monthly['date'] >= '2008-01-01']
    fsi_monthly = fsi_monthly.dropna(subset=['dict_fsi'] if 'dict_fsi' in fsi_monthly.columns else fsi_monthly.columns[1:2])

    # Save
    output_path = OUTPUT_DIR / 'fsi_monthly_aligned.csv'
    fsi_monthly.to_csv(output_path, index=False)
    print(f"  Saved: {output_path} ({len(fsi_monthly)} observations)")

    return fsi_monthly


def load_aligned_data():
    """Load the aligned monthly FSI data, creating it if necessary."""
    path = OUTPUT_DIR / 'fsi_monthly_aligned.csv'

    if not path.exists():
        return create_aligned_monthly_data()

    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


# =============================================================================
# IMPROVED LOCAL PROJECTIONS IRF
# =============================================================================

def compute_lp_irf_improved(df, shock_var, response_var, max_horizon=12, n_lags=2):
    """
    Compute Local Projections IRF with improved specification.

    Key improvements:
    - Uses first differences for non-stationary variables
    - Cumulative IRF for meaningful economic interpretation
    - HAC standard errors with appropriate bandwidth
    - Fewer lags to preserve degrees of freedom
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    data = df.copy()

    # Transform to stationary (first differences)
    if response_var == 'cds_5y':
        data['response'] = np.log(data[response_var]).diff() * 100  # Log returns in %
    else:
        data['response'] = data[response_var].diff()

    data['shock'] = data[shock_var].diff()

    # Standardize shock for comparability
    shock_std = data['shock'].std()
    if shock_std > 0:
        data['shock'] = data['shock'] / shock_std

    data = data.dropna()

    horizons = np.arange(max_horizon + 1)
    irfs = []
    ses = []
    pvals = []

    for h in horizons:
        # CUMULATIVE response: sum of responses from t to t+h
        data['y_cum'] = data['response'].rolling(h + 1).sum().shift(-h)

        # Build regression
        X_cols = ['shock']

        # Add lagged response as control
        for lag in range(1, n_lags + 1):
            data[f'response_lag{lag}'] = data['response'].shift(lag)
            X_cols.append(f'response_lag{lag}')

        # Prepare data
        reg_df = data[['y_cum'] + X_cols].dropna()

        if len(reg_df) < 30:
            irfs.append(np.nan)
            ses.append(np.nan)
            pvals.append(np.nan)
            continue

        y = reg_df['y_cum']
        X = add_constant(reg_df[X_cols])

        try:
            # OLS with HAC standard errors
            model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': max(1, h + 1)})

            irfs.append(model.params['shock'])
            ses.append(model.bse['shock'])
            pvals.append(model.pvalues['shock'])
        except:
            irfs.append(np.nan)
            ses.append(np.nan)
            pvals.append(np.nan)

    return {
        'horizons': horizons,
        'irf': np.array(irfs),
        'se': np.array(ses),
        'pval': np.array(pvals),
        'cum_irf': np.nansum(irfs),
        'sig_months': np.sum(np.array(pvals) < 0.1)
    }


# =============================================================================
# FIGURE 1: FSI Time Series (Individual and Combined versions)
# =============================================================================

def figure_1_fsi_timeseries_single(df, fsi_col, fsi_name, suffix):
    """Figure 1: Single FSI Time Series"""
    if fsi_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE['double'])

    # Main FSI line
    ax.plot(df['date'], df[fsi_col], color=COLORS['black'],
            linewidth=1.0, label=fsi_name, zorder=3)

    # 12-month moving average
    ma = df[fsi_col].rolling(12, min_periods=1).mean()
    ax.plot(df['date'], ma, color=COLORS['dark_gray'],
            linewidth=2, linestyle='--', label='12-Month MA', zorder=4)

    # Neutral line
    ax.axhline(y=0.5, color=COLORS['gray'], linestyle=':',
               linewidth=0.8, alpha=0.7)

    # Crisis shading
    for (start, end), label in CRISIS_PERIODS.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if df['date'].min() <= end_dt and df['date'].max() >= start_dt:
            ax.axvspan(start_dt, end_dt, alpha=0.15,
                       color=COLORS['fill_dark'], zorder=0)

    ax.set_xlabel('Date')
    ax.set_ylabel('Financial Stress Index')
    ax.set_ylim(0, 1.05)

    years = (df['date'].max() - df['date'].min()).days / 365
    format_date_axis(ax, years)

    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)

    save_figure(fig, f'figure_1_fsi_timeseries_{suffix}', FIGURES_DIR)
    plt.close()


def figure_1_fsi_timeseries_all(df):
    """Figure 1: All FSI Indices Combined"""
    fig, ax = plt.subplots(figsize=FIGSIZE['double'])

    line_configs = [
        ('dict_fsi', 'Dictionary FSI', COLORS['black'], '-'),
        ('ml_fsi', 'ML FSI', COLORS['dark_gray'], '--'),
        ('combined_fsi', 'Combined FSI', COLORS['gray'], ':'),
    ]

    for fsi_col, label, color, linestyle in line_configs:
        if fsi_col in df.columns:
            ax.plot(df['date'], df[fsi_col], color=color,
                    linewidth=1.5, linestyle=linestyle, label=label, zorder=3)

    # Neutral line
    ax.axhline(y=0.5, color=COLORS['light_gray'], linestyle=':',
               linewidth=0.8, alpha=0.7)

    # Crisis shading
    for (start, end), label in CRISIS_PERIODS.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if df['date'].min() <= end_dt and df['date'].max() >= start_dt:
            ax.axvspan(start_dt, end_dt, alpha=0.15,
                       color=COLORS['fill_dark'], zorder=0)

    ax.set_xlabel('Date')
    ax.set_ylabel('Financial Stress Index')
    ax.set_ylim(0, 1.05)

    years = (df['date'].max() - df['date'].min()).days / 365
    format_date_axis(ax, years)

    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)

    save_figure(fig, 'figure_1_fsi_timeseries_all', FIGURES_DIR)
    plt.close()


def figure_1_fsi_with_volatility(df, fsi_col, fsi_name, suffix):
    """Figure 1: FSI with IBOV Volatility overlay"""
    if fsi_col not in df.columns or 'volatility' not in df.columns:
        return

    fig, ax1 = plt.subplots(figsize=FIGSIZE['double'])

    # FSI (left axis)
    line1, = ax1.plot(df['date'], df[fsi_col], color=COLORS['black'],
                      linewidth=1.5, label=fsi_name)
    ax1.set_ylabel('Financial Stress Index', color=COLORS['black'])
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='y', labelcolor=COLORS['black'])

    # Volatility (right axis)
    ax2 = ax1.twinx()
    line2, = ax2.plot(df['date'], df['volatility'], color=COLORS['gray'],
                      linewidth=1.5, linestyle='--', label='IBOV Volatility')
    ax2.set_ylabel('IBOV Realized Volatility (%)', color=COLORS['gray'])
    ax2.tick_params(axis='y', labelcolor=COLORS['gray'])

    # Crisis shading
    for (start, end), label in CRISIS_PERIODS.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if df['date'].min() <= end_dt and df['date'].max() >= start_dt:
            ax1.axvspan(start_dt, end_dt, alpha=0.15,
                        color=COLORS['fill_dark'], zorder=0)

    ax1.set_xlabel('Date')

    years = (df['date'].max() - df['date'].min()).days / 365
    format_date_axis(ax1, years)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', framealpha=0.95)

    ax1.grid(True, alpha=0.3)

    save_figure(fig, f'figure_1_fsi_volatility_{suffix}', FIGURES_DIR)
    plt.close()


# =============================================================================
# FIGURE 2: FSI vs CDS (Individual and Combined versions)
# =============================================================================

def figure_2_fsi_vs_cds_single(df, fsi_col, fsi_name, suffix):
    """Figure 2: Single FSI vs CDS"""
    if fsi_col not in df.columns or 'cds_5y' not in df.columns:
        return

    fig, ax1 = plt.subplots(figsize=FIGSIZE['double'])

    # FSI (left axis)
    line1, = ax1.plot(df['date'], df[fsi_col], color=COLORS['black'],
                      linewidth=1.5, label=fsi_name)
    ax1.set_ylabel('Financial Stress Index', color=COLORS['black'])
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='y', labelcolor=COLORS['black'])

    # CDS (right axis)
    ax2 = ax1.twinx()
    line2, = ax2.plot(df['date'], df['cds_5y'], color=COLORS['gray'],
                      linewidth=1.5, linestyle='--', label='CDS 5Y (bps)')
    ax2.set_ylabel('CDS Spread (basis points)', color=COLORS['gray'])
    ax2.tick_params(axis='y', labelcolor=COLORS['gray'])

    # Crisis shading
    for (start, end), label in CRISIS_PERIODS.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if df['date'].min() <= end_dt and df['date'].max() >= start_dt:
            ax1.axvspan(start_dt, end_dt, alpha=0.15,
                        color=COLORS['fill_dark'], zorder=0)

    corr = df[fsi_col].corr(df['cds_5y'])

    ax1.set_xlabel('Date')

    years = (df['date'].max() - df['date'].min()).days / 365
    format_date_axis(ax1, years)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', framealpha=0.95)

    # Correlation annotation
    textstr = f'Correlation: {corr:.2f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9,
                 edgecolor=COLORS['light_gray'])
    ax1.text(0.97, 0.97, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    ax1.grid(True, alpha=0.3)

    save_figure(fig, f'figure_2_fsi_vs_cds_{suffix}', FIGURES_DIR)
    plt.close()


def figure_2_fsi_vs_cds_all(df):
    """Figure 2: All FSI Indices vs CDS"""
    if 'cds_5y' not in df.columns:
        return

    fig, ax1 = plt.subplots(figsize=FIGSIZE['double'])

    line_configs = [
        ('dict_fsi', 'Dictionary FSI', COLORS['black'], '-'),
        ('ml_fsi', 'ML FSI', COLORS['dark_gray'], '--'),
        ('combined_fsi', 'Combined FSI', COLORS['gray'], ':'),
    ]

    lines = []
    for fsi_col, label, color, linestyle in line_configs:
        if fsi_col in df.columns:
            line, = ax1.plot(df['date'], df[fsi_col], color=color,
                            linewidth=1.5, linestyle=linestyle, label=label)
            lines.append(line)

    ax1.set_ylabel('Financial Stress Index', color=COLORS['black'])
    ax1.set_ylim(0, 1.05)

    # CDS (right axis)
    ax2 = ax1.twinx()
    line_cds, = ax2.plot(df['date'], df['cds_5y'], color=COLORS['light_gray'],
                         linewidth=2, linestyle='-', label='CDS 5Y (bps)', alpha=0.7)
    ax2.set_ylabel('CDS Spread (basis points)', color=COLORS['light_gray'])
    ax2.tick_params(axis='y', labelcolor=COLORS['light_gray'])

    # Crisis shading
    for (start, end), label in CRISIS_PERIODS.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if df['date'].min() <= end_dt and df['date'].max() >= start_dt:
            ax1.axvspan(start_dt, end_dt, alpha=0.15,
                        color=COLORS['fill_dark'], zorder=0)

    ax1.set_xlabel('Date')

    years = (df['date'].max() - df['date'].min()).days / 365
    format_date_axis(ax1, years)

    all_lines = lines + [line_cds]
    labels = [l.get_label() for l in all_lines]
    ax1.legend(all_lines, labels, loc='upper left', framealpha=0.95)

    ax1.grid(True, alpha=0.3)

    save_figure(fig, 'figure_2_fsi_vs_cds_all', FIGURES_DIR)
    plt.close()


def figure_2_fsi_vs_cds_volatility(df, fsi_col, fsi_name, suffix):
    """Figure 2: FSI vs CDS with Volatility"""
    if fsi_col not in df.columns or 'cds_5y' not in df.columns or 'volatility' not in df.columns:
        return

    fig, ax1 = plt.subplots(figsize=FIGSIZE['double'])

    # FSI (left axis)
    line1, = ax1.plot(df['date'], df[fsi_col], color=COLORS['black'],
                      linewidth=1.5, label=fsi_name)
    ax1.set_ylabel('Financial Stress Index', color=COLORS['black'])
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='y', labelcolor=COLORS['black'])

    # CDS and Volatility (right axis - normalized)
    ax2 = ax1.twinx()

    # Normalize CDS and volatility to 0-1 for comparison
    cds_norm = (df['cds_5y'] - df['cds_5y'].min()) / (df['cds_5y'].max() - df['cds_5y'].min())
    vol_norm = (df['volatility'] - df['volatility'].min()) / (df['volatility'].max() - df['volatility'].min())

    line2, = ax2.plot(df['date'], cds_norm, color=COLORS['gray'],
                      linewidth=1.5, linestyle='--', label='CDS 5Y (normalized)')
    line3, = ax2.plot(df['date'], vol_norm, color=COLORS['light_gray'],
                      linewidth=1.5, linestyle=':', label='IBOV Volatility (normalized)')
    ax2.set_ylabel('Normalized Value', color=COLORS['gray'])
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='y', labelcolor=COLORS['gray'])

    # Crisis shading
    for (start, end), label in CRISIS_PERIODS.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if df['date'].min() <= end_dt and df['date'].max() >= start_dt:
            ax1.axvspan(start_dt, end_dt, alpha=0.15,
                        color=COLORS['fill_dark'], zorder=0)

    ax1.set_xlabel('Date')

    years = (df['date'].max() - df['date'].min()).days / 365
    format_date_axis(ax1, years)

    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', framealpha=0.95)

    ax1.grid(True, alpha=0.3)

    save_figure(fig, f'figure_2_fsi_cds_volatility_{suffix}', FIGURES_DIR)
    plt.close()


# =============================================================================
# FIGURE 3: MAIN IRF RESULT
# =============================================================================

def figure_3_irf_main(df):
    """Figure 3: Impulse Response Function - FSI Shock to CDS"""
    print("Generating Figure 3: IRF (Main Result)...")

    if 'cds_5y' not in df.columns:
        print("  Skipping: CDS data not available")
        return None, None

    # Compute improved LP-IRF
    result = compute_lp_irf_improved(df, 'dict_fsi', 'cds_5y',
                                      max_horizon=12, n_lags=2)

    horizons = result['horizons']
    irfs = result['irf']
    ses = result['se']
    pvals = result['pval']

    fig, ax = plt.subplots(figsize=FIGSIZE['irf'])

    # Plot IRF with confidence bands
    ax.plot(horizons, irfs, color=COLORS['black'], linewidth=2,
            marker='o', markersize=5, zorder=4, label='Cumulative IRF')

    # 95% CI
    ax.fill_between(horizons, irfs - 1.96*ses, irfs + 1.96*ses,
                    alpha=0.15, color=COLORS['gray'], zorder=2, label='95% CI')

    # 90% CI
    ax.fill_between(horizons, irfs - 1.645*ses, irfs + 1.645*ses,
                    alpha=0.25, color=COLORS['dark_gray'], zorder=3, label='90% CI')

    add_zero_line(ax)

    # Mark significant points
    for i, (h, irf_val, pval) in enumerate(zip(horizons, irfs, pvals)):
        if pval < 0.1:
            ax.plot(h, irf_val, 'o', color=COLORS['black'], markersize=8,
                    markerfacecolor='white', markeredgewidth=2, zorder=5)

    ax.set_xlabel('Months After Shock')
    ax.set_ylabel('Cumulative CDS Response (%)')

    # Annotation
    cum_irf = result['cum_irf']
    sig_months = result['sig_months']
    textstr = f'Cumulative Effect: {cum_irf:.2f}%\nSignificant: {sig_months}/13 months'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9,
                 edgecolor=COLORS['light_gray'])
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    save_figure(fig, 'figure_3_irf_main', FIGURES_DIR)
    plt.close()

    return irfs, ses


# =============================================================================
# FIGURE 4: METHODOLOGY COMPARISON
# =============================================================================

def figure_4_methodology_comparison(df):
    """Figure 4: Comparison of FSI Construction Methods"""
    print("Generating Figure 4: Methodology Comparison...")

    if 'cds_5y' not in df.columns:
        print("  Skipping: CDS data not available")
        return

    # Only compare dict_fsi vs combined_fsi (the key comparison)
    methods = []
    if 'dict_fsi' in df.columns:
        methods.append(('dict_fsi', 'Dictionary FSI'))
    if 'ml_fsi' in df.columns:
        methods.append(('ml_fsi', 'ML FSI'))
    if 'combined_fsi' in df.columns:
        methods.append(('combined_fsi', 'Combined FSI'))

    if len(methods) == 0:
        print("  Skipping: No FSI methods available")
        return

    fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 4.5), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for idx, (col, name) in enumerate(methods):
        ax = axes[idx]

        result = compute_lp_irf_improved(df, col, 'cds_5y', max_horizon=12, n_lags=2)

        horizons = result['horizons']
        irfs = result['irf']
        ses = result['se']

        # Plot
        ax.plot(horizons, irfs, color=COLORS['black'], linewidth=2,
                marker='o', markersize=4, zorder=4)

        ax.fill_between(horizons, irfs - 1.96*ses, irfs + 1.96*ses,
                        alpha=0.15, color=COLORS['gray'], zorder=2)
        ax.fill_between(horizons, irfs - 1.645*ses, irfs + 1.645*ses,
                        alpha=0.25, color=COLORS['dark_gray'], zorder=3)

        add_zero_line(ax)

        cum_irf = result['cum_irf']
        sig = result['sig_months']

        # Annotation box instead of title
        textstr = f'{name}\nCumulative: {cum_irf:.1f}%\nSignificant: {sig}/13'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9,
                     edgecolor=COLORS['light_gray'])
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        ax.set_xlabel('Months')
        if idx == 0:
            ax.set_ylabel('Cumulative CDS Response (%)')

        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_figure(fig, 'figure_4_methodology_comparison', FIGURES_DIR)
    plt.close()


# =============================================================================
# FIGURE 5: CRISIS PERIOD ANALYSIS (Individual and Combined versions)
# =============================================================================

def figure_5_crisis_analysis_single(df, fsi_col, fsi_name, suffix):
    """Figure 5: Single FSI During Crisis Periods"""
    if fsi_col not in df.columns:
        return

    crises = [
        ('Global Financial Crisis', '2008-01-01', '2010-06-30'),
        ('Brazilian Recession', '2014-06-01', '2017-06-30'),
        ('COVID-19 Pandemic', '2019-09-01', '2021-03-31'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for idx, (crisis_name, start, end) in enumerate(crises):
        ax = axes[idx]

        mask = (df['date'] >= start) & (df['date'] <= end)
        crisis_df = df[mask].copy()

        if len(crisis_df) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            continue

        # FSI
        ax.plot(crisis_df['date'], crisis_df[fsi_col],
                color=COLORS['black'], linewidth=1.5, label=fsi_name)

        # Normalized CDS
        if 'cds_5y' in crisis_df.columns:
            cds_min = crisis_df['cds_5y'].min()
            cds_max = crisis_df['cds_5y'].max()
            if cds_max > cds_min:
                cds_norm = (crisis_df['cds_5y'] - cds_min) / (cds_max - cds_min)
                ax.plot(crisis_df['date'], cds_norm,
                        color=COLORS['gray'], linewidth=1.5, linestyle='--',
                        label='CDS (normalized)')

        ax.axhline(y=0.5, color=COLORS['light_gray'], linestyle=':',
                   linewidth=0.8, alpha=0.7)

        # Crisis name as annotation
        ax.text(0.5, 0.02, crisis_name, transform=ax.transAxes, fontsize=10,
                ha='center', va='bottom', fontweight='normal')

        ax.set_xlabel('Date')
        if idx == 0:
            ax.set_ylabel('Index Value')

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_figure(fig, f'figure_5_crisis_analysis_{suffix}', FIGURES_DIR)
    plt.close()


def figure_5_crisis_analysis_all(df):
    """Figure 5: All FSI Indices During Crisis Periods"""
    crises = [
        ('Global Financial Crisis', '2008-01-01', '2010-06-30'),
        ('Brazilian Recession', '2014-06-01', '2017-06-30'),
        ('COVID-19 Pandemic', '2019-09-01', '2021-03-31'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    line_configs = [
        ('dict_fsi', 'Dictionary FSI', COLORS['black'], '-'),
        ('ml_fsi', 'ML FSI', COLORS['dark_gray'], '--'),
        ('combined_fsi', 'Combined FSI', COLORS['gray'], ':'),
    ]

    for idx, (crisis_name, start, end) in enumerate(crises):
        ax = axes[idx]

        mask = (df['date'] >= start) & (df['date'] <= end)
        crisis_df = df[mask].copy()

        if len(crisis_df) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            continue

        # All FSI indices
        for fsi_col, label, color, linestyle in line_configs:
            if fsi_col in crisis_df.columns:
                ax.plot(crisis_df['date'], crisis_df[fsi_col],
                        color=color, linewidth=1.5, linestyle=linestyle, label=label)

        ax.axhline(y=0.5, color=COLORS['light_gray'], linestyle=':',
                   linewidth=0.8, alpha=0.7)

        # Crisis name as annotation
        ax.text(0.5, 0.02, crisis_name, transform=ax.transAxes, fontsize=10,
                ha='center', va='bottom', fontweight='normal')

        ax.set_xlabel('Date')
        if idx == 0:
            ax.set_ylabel('Index Value')

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_figure(fig, 'figure_5_crisis_analysis_all', FIGURES_DIR)
    plt.close()


def figure_5_crisis_with_volatility(df, fsi_col, fsi_name, suffix):
    """Figure 5: FSI with Volatility During Crisis Periods"""
    if fsi_col not in df.columns or 'volatility' not in df.columns:
        return

    crises = [
        ('Global Financial Crisis', '2008-01-01', '2010-06-30'),
        ('Brazilian Recession', '2014-06-01', '2017-06-30'),
        ('COVID-19 Pandemic', '2019-09-01', '2021-03-31'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for idx, (crisis_name, start, end) in enumerate(crises):
        ax = axes[idx]

        mask = (df['date'] >= start) & (df['date'] <= end)
        crisis_df = df[mask].copy()

        if len(crisis_df) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            continue

        # FSI
        ax.plot(crisis_df['date'], crisis_df[fsi_col],
                color=COLORS['black'], linewidth=1.5, label=fsi_name)

        # Normalized Volatility
        vol_min = crisis_df['volatility'].min()
        vol_max = crisis_df['volatility'].max()
        if vol_max > vol_min:
            vol_norm = (crisis_df['volatility'] - vol_min) / (vol_max - vol_min)
            ax.plot(crisis_df['date'], vol_norm,
                    color=COLORS['gray'], linewidth=1.5, linestyle='--',
                    label='IBOV Volatility (normalized)')

        ax.axhline(y=0.5, color=COLORS['light_gray'], linestyle=':',
                   linewidth=0.8, alpha=0.7)

        # Crisis name as annotation
        ax.text(0.5, 0.02, crisis_name, transform=ax.transAxes, fontsize=10,
                ha='center', va='bottom', fontweight='normal')

        ax.set_xlabel('Date')
        if idx == 0:
            ax.set_ylabel('Index Value')

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_figure(fig, f'figure_5_crisis_volatility_{suffix}', FIGURES_DIR)
    plt.close()


# =============================================================================
# FIGURE 6: GRANGER CAUSALITY
# =============================================================================

def figure_6_granger_causality(df):
    """Figure 6: Simplified Granger Causality Results"""
    print("Generating Figure 6: Granger Causality...")

    from statsmodels.tsa.stattools import grangercausalitytests

    # Only test key relationships: FSI ↔ CDS
    if 'cds_5y' not in df.columns:
        print("  Skipping: CDS data not available")
        return

    # Transform to stationary
    data = df.copy()
    data['cds_ret'] = np.log(data['cds_5y']).diff() * 100
    data['fsi_diff'] = data['dict_fsi'].diff()
    data = data.dropna()

    max_lag = 4

    # Test FSI → CDS
    try:
        test_fsi_cds = grangercausalitytests(data[['cds_ret', 'fsi_diff']], maxlag=max_lag, verbose=False)
        pval_fsi_cds = min([test_fsi_cds[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)])
    except:
        pval_fsi_cds = 1.0

    # Test CDS → FSI
    try:
        test_cds_fsi = grangercausalitytests(data[['fsi_diff', 'cds_ret']], maxlag=max_lag, verbose=False)
        pval_cds_fsi = min([test_cds_fsi[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)])
    except:
        pval_cds_fsi = 1.0

    # Create simple bar chart
    fig, ax = plt.subplots(figsize=FIGSIZE['single'])

    relationships = ['FSI → CDS', 'CDS → FSI']
    pvalues = [pval_fsi_cds, pval_cds_fsi]

    # Convert to -log10(p) for visualization (higher = more significant)
    significance = [-np.log10(max(p, 1e-10)) for p in pvalues]

    colors = [COLORS['dark_gray'] if p < 0.05 else COLORS['light_gray'] for p in pvalues]

    bars = ax.bar(relationships, significance, color=colors, edgecolor=COLORS['black'], linewidth=0.8)

    # Add significance thresholds
    ax.axhline(y=-np.log10(0.05), color=COLORS['gray'], linestyle='--',
               linewidth=1, label='p = 0.05')
    ax.axhline(y=-np.log10(0.01), color=COLORS['dark_gray'], linestyle=':',
               linewidth=1, label='p = 0.01')

    # Add p-value labels
    for bar, pval in zip(bars, pvalues):
        height = bar.get_height()
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'p={pval:.3f}{sig}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('-log₁₀(p-value)\n(higher = more significant)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    save_figure(fig, 'figure_6_granger_causality', FIGURES_DIR)
    plt.close()


# =============================================================================
# FIGURE 7: REGIME ANALYSIS (Individual and Combined versions)
# =============================================================================

def figure_7_regime_analysis_single(df, fsi_col, fsi_name, suffix):
    """Figure 7: Financial Stress Regimes for Single FSI"""
    if fsi_col not in df.columns:
        return

    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError:
        print("  Skipping: statsmodels MarkovRegression not available")
        return

    # Use specified FSI for regime analysis
    fsi_data = df[fsi_col].dropna()
    dates = df.loc[fsi_data.index, 'date']

    if len(fsi_data) < 50:
        print(f"  Skipping {suffix}: Insufficient data")
        return

    # Fit Markov Switching Model (2 regimes)
    try:
        model = MarkovRegression(fsi_data.values, k_regimes=2,
                                  switching_variance=True)
        results = model.fit(disp=False)

        # Get smoothed probabilities for crisis regime
        # Determine which regime is "crisis" (higher mean)
        regime_means = [results.params[f'const[{i}]'] for i in range(2)]
        crisis_regime = 0 if regime_means[0] > regime_means[1] else 1

        # Get smoothed probabilities (handle different statsmodels versions)
        try:
            smoothed_probs = results.smoothed_marginal_probabilities[:, crisis_regime]
        except:
            smoothed_probs = results.smoothed_marginal_probabilities[crisis_regime]

        # Smooth for cleaner visualization
        smoothed_probs = pd.Series(smoothed_probs).rolling(4, min_periods=1, center=True).mean().values

    except Exception as e:
        # Fallback: use rolling z-score threshold
        rolling_mean = fsi_data.rolling(12, min_periods=1).mean()
        rolling_std = fsi_data.rolling(12, min_periods=1).std().fillna(0.1)
        z_score = (fsi_data - rolling_mean) / rolling_std
        # High stress = FSI significantly above rolling mean
        smoothed_probs = (z_score > 0.5).astype(float).rolling(4, min_periods=1).mean().values

    # Create single clean figure
    fig, ax = plt.subplots(figsize=FIGSIZE['double'])

    # Plot FSI
    ax.plot(dates, fsi_data.values, color=COLORS['black'],
            linewidth=1.2, label=fsi_name, zorder=3)

    # 12-month moving average
    ma = pd.Series(fsi_data.values).rolling(12, min_periods=1).mean()
    ax.plot(dates, ma, color=COLORS['dark_gray'],
            linewidth=2, linestyle='--', label='12-Month MA', zorder=4)

    # Shade high stress periods (smoothed probability > 0.5)
    crisis_mask = smoothed_probs > 0.5

    # Find contiguous crisis periods
    in_crisis = False
    crisis_start = None
    for i, (date, is_crisis) in enumerate(zip(dates, crisis_mask)):
        if is_crisis and not in_crisis:
            crisis_start = date
            in_crisis = True
        elif not is_crisis and in_crisis:
            ax.axvspan(crisis_start, dates.iloc[i-1], alpha=0.2,
                       color=COLORS['fill_dark'], zorder=0)
            in_crisis = False
    # Handle if ends in crisis
    if in_crisis:
        ax.axvspan(crisis_start, dates.iloc[-1], alpha=0.2,
                   color=COLORS['fill_dark'], zorder=0)

    # Add neutral line
    ax.axhline(y=0.5, color=COLORS['gray'], linestyle=':',
               linewidth=0.8, alpha=0.7)

    ax.set_xlabel('Date')
    ax.set_ylabel('Financial Stress Index')
    ax.set_ylim(0, 1.05)

    years = (dates.max() - dates.min()).days / 365
    format_date_axis(ax, years)

    # Add custom legend entry for shading
    legend_elements = [
        plt.Line2D([0], [0], color=COLORS['black'], linewidth=1.2, label=fsi_name),
        plt.Line2D([0], [0], color=COLORS['dark_gray'], linewidth=2, linestyle='--', label='12-Month MA'),
        Patch(facecolor=COLORS['fill_dark'], alpha=0.2, label='High Stress Regime'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95)

    ax.grid(True, alpha=0.3)

    save_figure(fig, f'figure_7_regime_analysis_{suffix}', FIGURES_DIR)
    plt.close()


def figure_7_regime_analysis_all(df):
    """Figure 7: Regime Analysis Comparison for All FSI Indices"""
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError:
        print("  Skipping: statsmodels MarkovRegression not available")
        return

    fsi_cols = ['dict_fsi', 'ml_fsi', 'combined_fsi']
    available_cols = [col for col in fsi_cols if col in df.columns]

    if len(available_cols) == 0:
        return

    fig, axes = plt.subplots(len(available_cols), 1, figsize=(14, 4*len(available_cols)), sharex=True)
    if len(available_cols) == 1:
        axes = [axes]

    line_configs = {
        'dict_fsi': ('Dictionary FSI', COLORS['black']),
        'ml_fsi': ('ML FSI', COLORS['dark_gray']),
        'combined_fsi': ('Combined FSI', COLORS['gray']),
    }

    for idx, fsi_col in enumerate(available_cols):
        ax = axes[idx]
        fsi_name, color = line_configs[fsi_col]

        fsi_data = df[fsi_col].dropna()
        dates = df.loc[fsi_data.index, 'date']

        if len(fsi_data) < 50:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax.transAxes)
            continue

        # Fit model or use threshold fallback
        try:
            model = MarkovRegression(fsi_data.values, k_regimes=2, switching_variance=True)
            results = model.fit(disp=False)
            regime_means = [results.params[f'const[{i}]'] for i in range(2)]
            crisis_regime = 0 if regime_means[0] > regime_means[1] else 1
            try:
                smoothed_probs = results.smoothed_marginal_probabilities[:, crisis_regime]
            except:
                smoothed_probs = results.smoothed_marginal_probabilities[crisis_regime]
            smoothed_probs = pd.Series(smoothed_probs).rolling(4, min_periods=1, center=True).mean().values
        except:
            rolling_mean = fsi_data.rolling(12, min_periods=1).mean()
            rolling_std = fsi_data.rolling(12, min_periods=1).std().fillna(0.1)
            z_score = (fsi_data - rolling_mean) / rolling_std
            smoothed_probs = (z_score > 0.5).astype(float).rolling(4, min_periods=1).mean().values

        # Plot
        ax.plot(dates, fsi_data.values, color=color, linewidth=1.2, label=fsi_name, zorder=3)

        # Shade high stress
        crisis_mask = smoothed_probs > 0.5
        in_crisis = False
        crisis_start = None
        for i, (date, is_crisis) in enumerate(zip(dates, crisis_mask)):
            if is_crisis and not in_crisis:
                crisis_start = date
                in_crisis = True
            elif not is_crisis and in_crisis:
                ax.axvspan(crisis_start, dates.iloc[i-1], alpha=0.2, color=COLORS['fill_dark'], zorder=0)
                in_crisis = False
        if in_crisis:
            ax.axvspan(crisis_start, dates.iloc[-1], alpha=0.2, color=COLORS['fill_dark'], zorder=0)

        ax.axhline(y=0.5, color=COLORS['gray'], linestyle=':', linewidth=0.8, alpha=0.7)
        ax.set_ylabel(fsi_name)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        legend_elements = [
            plt.Line2D([0], [0], color=color, linewidth=1.2, label=fsi_name),
            Patch(facecolor=COLORS['fill_dark'], alpha=0.2, label='High Stress Regime'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95)

    axes[-1].set_xlabel('Date')
    years = (df['date'].max() - df['date'].min()).days / 365
    format_date_axis(axes[-1], years)

    plt.tight_layout()

    save_figure(fig, 'figure_7_regime_analysis_all', FIGURES_DIR)
    plt.close()


def figure_7_regime_with_volatility(df, fsi_col, fsi_name, suffix):
    """Figure 7: Regime Analysis with IBOV Volatility overlay"""
    if fsi_col not in df.columns or 'volatility' not in df.columns:
        return

    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError:
        return

    fsi_data = df[fsi_col].dropna()
    dates = df.loc[fsi_data.index, 'date']

    if len(fsi_data) < 50:
        return

    # Fit model or use threshold fallback
    try:
        model = MarkovRegression(fsi_data.values, k_regimes=2, switching_variance=True)
        results = model.fit(disp=False)
        regime_means = [results.params[f'const[{i}]'] for i in range(2)]
        crisis_regime = 0 if regime_means[0] > regime_means[1] else 1
        try:
            smoothed_probs = results.smoothed_marginal_probabilities[:, crisis_regime]
        except:
            smoothed_probs = results.smoothed_marginal_probabilities[crisis_regime]
        smoothed_probs = pd.Series(smoothed_probs).rolling(4, min_periods=1, center=True).mean().values
    except:
        rolling_mean = fsi_data.rolling(12, min_periods=1).mean()
        rolling_std = fsi_data.rolling(12, min_periods=1).std().fillna(0.1)
        z_score = (fsi_data - rolling_mean) / rolling_std
        smoothed_probs = (z_score > 0.5).astype(float).rolling(4, min_periods=1).mean().values

    fig, ax1 = plt.subplots(figsize=FIGSIZE['double'])

    # FSI (left axis)
    line1, = ax1.plot(dates, fsi_data.values, color=COLORS['black'],
                      linewidth=1.5, label=fsi_name, zorder=3)
    ax1.set_ylabel('Financial Stress Index', color=COLORS['black'])
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='y', labelcolor=COLORS['black'])

    # Shade high stress periods
    crisis_mask = smoothed_probs > 0.5
    in_crisis = False
    crisis_start = None
    for i, (date, is_crisis) in enumerate(zip(dates, crisis_mask)):
        if is_crisis and not in_crisis:
            crisis_start = date
            in_crisis = True
        elif not is_crisis and in_crisis:
            ax1.axvspan(crisis_start, dates.iloc[i-1], alpha=0.2, color=COLORS['fill_dark'], zorder=0)
            in_crisis = False
    if in_crisis:
        ax1.axvspan(crisis_start, dates.iloc[-1], alpha=0.2, color=COLORS['fill_dark'], zorder=0)

    # Volatility (right axis)
    ax2 = ax1.twinx()
    vol_data = df.loc[fsi_data.index, 'volatility']
    line2, = ax2.plot(dates, vol_data, color=COLORS['gray'],
                      linewidth=1.5, linestyle='--', label='IBOV Volatility', zorder=2)
    ax2.set_ylabel('IBOV Realized Volatility (%)', color=COLORS['gray'])
    ax2.tick_params(axis='y', labelcolor=COLORS['gray'])

    ax1.set_xlabel('Date')

    years = (dates.max() - dates.min()).days / 365
    format_date_axis(ax1, years)

    # Combined legend
    legend_elements = [
        plt.Line2D([0], [0], color=COLORS['black'], linewidth=1.5, label=fsi_name),
        plt.Line2D([0], [0], color=COLORS['gray'], linewidth=1.5, linestyle='--', label='IBOV Volatility'),
        Patch(facecolor=COLORS['fill_dark'], alpha=0.2, label='High Stress Regime'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', framealpha=0.95)

    ax1.grid(True, alpha=0.3)

    save_figure(fig, f'figure_7_regime_volatility_{suffix}', FIGURES_DIR)
    plt.close()


# =============================================================================
# TABLES
# =============================================================================

def generate_tables(df):
    """Generate summary tables"""
    print("\nGenerating Tables...")

    # Table 1: Summary Statistics
    stats_dict = {
        'Variable': [],
        'Mean': [],
        'Std. Dev.': [],
        'Min': [],
        'Max': [],
        'N': [],
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
            stats_dict['N'].append(str(len(df[col].dropna())))

    stats_df = pd.DataFrame(stats_dict)
    stats_df.to_csv(FIGURES_DIR / 'table_1_summary_statistics.csv', index=False)

    print("\n  Table 1: Summary Statistics")
    print("  " + "=" * 60)
    print(stats_df.to_string(index=False))

    # Table 2: IRF Results
    if 'cds_5y' in df.columns:
        methods = []
        for col, name in [('dict_fsi', 'Dictionary FSI'), ('ml_fsi', 'ML FSI'), ('combined_fsi', 'Combined FSI')]:
            if col in df.columns:
                result = compute_lp_irf_improved(df, col, 'cds_5y', max_horizon=12, n_lags=2)
                methods.append({
                    'Method': name,
                    'Cumulative IRF (%)': f"{result['cum_irf']:.2f}",
                    'Significant Months': f"{result['sig_months']}/13",
                })

        if methods:
            results_df = pd.DataFrame(methods)
            results_df.to_csv(FIGURES_DIR / 'table_2_irf_results.csv', index=False)

            print("\n  Table 2: IRF Results")
            print("  " + "=" * 60)
            print(results_df.to_string(index=False))

    print("  " + "=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate all publication figures."""
    print("=" * 70)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 70)
    print(f"\nOutput directory: {FIGURES_DIR}")

    # Load data (creates if necessary)
    try:
        df = load_aligned_data()
        print(f"Loaded {len(df)} monthly observations")
        print(f"Period: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"Variables: {[c for c in df.columns if c != 'date']}")
    except Exception as e:
        print(f"\nERROR: {e}")
        return

    # Generate figures
    print("\n" + "-" * 50)
    print("FIGURES")
    print("-" * 50)

    # Figure 1: FSI Time Series
    print("\nGenerating Figure 1 variants: FSI Time Series...")
    for fsi_col, config in FSI_INDICES.items():
        if fsi_col in df.columns:
            figure_1_fsi_timeseries_single(df, fsi_col, config['name'], config['short'])
            figure_1_fsi_with_volatility(df, fsi_col, config['name'], config['short'])
    figure_1_fsi_timeseries_all(df)

    # Figure 2: FSI vs CDS
    print("Generating Figure 2 variants: FSI vs CDS...")
    for fsi_col, config in FSI_INDICES.items():
        if fsi_col in df.columns:
            figure_2_fsi_vs_cds_single(df, fsi_col, config['name'], config['short'])
            figure_2_fsi_vs_cds_volatility(df, fsi_col, config['name'], config['short'])
    figure_2_fsi_vs_cds_all(df)

    # Figure 3: IRF (main result)
    figure_3_irf_main(df)

    # Figure 4: Methodology Comparison
    figure_4_methodology_comparison(df)

    # Figure 5: Crisis Analysis
    print("Generating Figure 5 variants: Crisis Analysis...")
    for fsi_col, config in FSI_INDICES.items():
        if fsi_col in df.columns:
            figure_5_crisis_analysis_single(df, fsi_col, config['name'], config['short'])
            figure_5_crisis_with_volatility(df, fsi_col, config['name'], config['short'])
    figure_5_crisis_analysis_all(df)

    # Figure 6: Granger Causality
    figure_6_granger_causality(df)

    # Figure 7: Regime Analysis
    print("Generating Figure 7 variants: Regime Analysis...")
    for fsi_col, config in FSI_INDICES.items():
        if fsi_col in df.columns:
            figure_7_regime_analysis_single(df, fsi_col, config['name'], config['short'])
            figure_7_regime_with_volatility(df, fsi_col, config['name'], config['short'])
    figure_7_regime_analysis_all(df)

    # Generate tables
    print("\n" + "-" * 50)
    print("TABLES")
    print("-" * 50)

    generate_tables(df)

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
