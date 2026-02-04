"""
Validation and Visualization for Financial Stress Index
========================================================
Validates FSI against market volatility and creates visualizations.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from .dictionaries import CRISIS_EPISODES


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def calculate_correlation_metrics(
    fsi: pd.Series,
    volatility: pd.Series,
) -> Dict[str, float]:
    """
    Calculate correlation metrics between FSI and market volatility.

    Args:
        fsi: FSI time series
        volatility: Market volatility time series

    Returns:
        Dictionary with correlation metrics
    """
    # Align series
    aligned = pd.DataFrame({
        'fsi': fsi,
        'vol': volatility,
    }).dropna()

    if len(aligned) < 10:
        return {
            'pearson': np.nan,
            'pearson_pvalue': np.nan,
            'spearman': np.nan,
            'spearman_pvalue': np.nan,
            'n_observations': len(aligned),
        }

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(aligned['fsi'], aligned['vol'])

    # Spearman correlation (rank-based, more robust)
    spearman_r, spearman_p = stats.spearmanr(aligned['fsi'], aligned['vol'])

    return {
        'pearson': pearson_r,
        'pearson_pvalue': pearson_p,
        'spearman': spearman_r,
        'spearman_pvalue': spearman_p,
        'n_observations': len(aligned),
    }


def validate_fsi(
    daily_fsi: pd.DataFrame,
    market_data: pd.DataFrame,
    monthly_fsi: Optional[pd.DataFrame] = None,
    fsi_column: str = 'fsi',
    vol_column: str = 'realized_vol_30d',
) -> Dict[str, Dict]:
    """
    Comprehensive validation of FSI against market data.

    Args:
        daily_fsi: Daily FSI DataFrame with 'date' and fsi_column
        market_data: Market data with 'date' and vol_column
        monthly_fsi: Optional monthly FSI DataFrame
        fsi_column: Name of FSI column
        vol_column: Name of volatility column

    Returns:
        Dictionary with validation results
    """
    results = {}

    # Merge daily data
    daily_merged = pd.merge(
        daily_fsi[['date', fsi_column]],
        market_data[['date', vol_column]],
        on='date',
        how='inner',
    )

    # Daily correlations
    results['daily'] = calculate_correlation_metrics(
        daily_merged[fsi_column],
        daily_merged[vol_column],
    )

    # Monthly correlations
    if monthly_fsi is not None:
        # Aggregate market data to monthly
        market_data['month'] = pd.to_datetime(market_data['date']).dt.to_period('M')
        monthly_vol = market_data.groupby('month')[vol_column].mean().reset_index()
        monthly_vol['month'] = monthly_vol['month'].dt.to_timestamp()

        monthly_merged = pd.merge(
            monthly_fsi[['month', 'fsi']].rename(columns={'month': 'date'}),
            monthly_vol.rename(columns={'month': 'date'}),
            on='date',
            how='inner',
        )

        results['monthly'] = calculate_correlation_metrics(
            monthly_merged['fsi'],
            monthly_merged[vol_column],
        )

    # Descriptive statistics
    results['fsi_stats'] = {
        'mean': daily_fsi[fsi_column].mean(),
        'std': daily_fsi[fsi_column].std(),
        'min': daily_fsi[fsi_column].min(),
        'max': daily_fsi[fsi_column].max(),
        'n_days': len(daily_fsi),
    }

    return results


def print_validation_report(
    results: Dict,
    target_range: Tuple[float, float] = (0.70, 0.85),
) -> None:
    """
    Print formatted validation report.

    Args:
        results: Validation results from validate_fsi
        target_range: Target correlation range (from literature)
    """
    print("\n" + "=" * 60)
    print("FINANCIAL STRESS INDEX - VALIDATION REPORT")
    print("=" * 60)

    print("\nCorrelation with Market Volatility:")
    print("-" * 40)

    if 'daily' in results:
        d = results['daily']
        status = "✓" if target_range[0] <= abs(d['pearson']) <= target_range[1] else "○"
        print(f"  Daily (n={d['n_observations']}):")
        print(f"    Pearson:  r = {d['pearson']:.3f} (p = {d['pearson_pvalue']:.4f}) {status}")
        print(f"    Spearman: ρ = {d['spearman']:.3f} (p = {d['spearman_pvalue']:.4f})")

    if 'monthly' in results:
        m = results['monthly']
        status = "✓" if target_range[0] <= abs(m['pearson']) <= target_range[1] else "○"
        print(f"\n  Monthly (n={m['n_observations']}):")
        print(f"    Pearson:  r = {m['pearson']:.3f} (p = {m['pearson_pvalue']:.4f}) {status}")
        print(f"    Spearman: ρ = {m['spearman']:.3f} (p = {m['spearman_pvalue']:.4f})")

    print(f"\n  Target Range: {target_range[0]:.2f} - {target_range[1]:.2f}")
    print("  (Based on Baker et al. 2019 VIX correlations)")

    if 'fsi_stats' in results:
        s = results['fsi_stats']
        print(f"\nFSI Descriptive Statistics:")
        print("-" * 40)
        print(f"  Mean:  {s['mean']:.2f}")
        print(f"  Std:   {s['std']:.2f}")
        print(f"  Range: [{s['min']:.2f}, {s['max']:.2f}]")
        print(f"  Days:  {s['n_days']}")

    print("\n" + "=" * 60)


def plot_fsi_timeseries(
    daily_fsi: pd.DataFrame,
    fsi_column: str = 'fsi',
    title: str = 'Financial Stress Index - Daily',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    mark_crises: bool = True,
) -> plt.Figure:
    """
    Plot FSI time series with crisis episode markers.

    Args:
        daily_fsi: DataFrame with date and FSI
        fsi_column: Name of FSI column
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
        mark_crises: Whether to shade crisis periods

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot FSI
    ax.plot(
        daily_fsi['date'],
        daily_fsi[fsi_column],
        color='#2E86AB',
        linewidth=1,
        label='FSI',
    )

    # Add rolling average
    if len(daily_fsi) > 30:
        rolling = daily_fsi[fsi_column].rolling(window=30, center=True).mean()
        ax.plot(
            daily_fsi['date'],
            rolling,
            color='#E94F37',
            linewidth=2,
            label='30-day MA',
            alpha=0.8,
        )

    # Mark crisis periods
    if mark_crises:
        for (start, end), label in CRISIS_EPISODES.items():
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)

            # Check if crisis falls within data range
            if (daily_fsi['date'].min() <= end_dt and
                daily_fsi['date'].max() >= start_dt):

                ax.axvspan(start_dt, end_dt, alpha=0.2, color='red', label=None)

                # Add label at top
                mid_date = start_dt + (end_dt - start_dt) / 2
                ax.text(
                    mid_date,
                    ax.get_ylim()[1] * 0.98,
                    label,
                    ha='center',
                    va='top',
                    fontsize=8,
                    rotation=45,
                )

    # Formatting
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('FSI', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')

    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_fsi_vs_volatility(
    daily_fsi: pd.DataFrame,
    market_data: pd.DataFrame,
    fsi_column: str = 'fsi',
    vol_column: str = 'realized_vol_30d',
    title: str = 'FSI vs Market Volatility',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Plot FSI overlaid with market volatility.

    Args:
        daily_fsi: DataFrame with date and FSI
        market_data: DataFrame with date and volatility
        fsi_column: Name of FSI column
        vol_column: Name of volatility column
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Merge data
    merged = pd.merge(
        daily_fsi[['date', fsi_column]],
        market_data[['date', vol_column]],
        on='date',
        how='inner',
    )

    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot FSI
    color1 = '#2E86AB'
    ax1.plot(
        merged['date'],
        merged[fsi_column],
        color=color1,
        linewidth=1.5,
        label='FSI',
    )
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('FSI', color=color1, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Create second y-axis for volatility
    ax2 = ax1.twinx()
    color2 = '#E94F37'
    ax2.plot(
        merged['date'],
        merged[vol_column],
        color=color2,
        linewidth=1.5,
        label='Volatility',
        alpha=0.7,
    )
    ax2.set_ylabel('Realized Volatility (%)', color=color2, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Calculate correlation for title
    corr = merged[fsi_column].corr(merged[vol_column])
    ax1.set_title(f'{title}\n(Correlation: r = {corr:.3f})', fontsize=13, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.xticks(rotation=45)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_term_frequency(
    classified_df: pd.DataFrame,
    term_type: str = 'stress',
    top_n: int = 20,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot frequency of matched terms.

    Args:
        classified_df: DataFrame with classification results
        term_type: Type of terms to plot ('stress', 'financial', 'negative')
        top_n: Number of top terms to show
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # This requires the matched terms to be stored during classification
    # For now, we'll create a placeholder
    title = title or f'Top {top_n} {term_type.title()} Terms'

    fig, ax = plt.subplots(figsize=figsize)

    # Placeholder - in real usage, you'd aggregate matched terms
    ax.text(
        0.5, 0.5,
        f'Term frequency plot\n(requires matched terms from classification)',
        ha='center', va='center',
        fontsize=12,
        transform=ax.transAxes,
    )

    ax.set_title(title, fontsize=13, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_scatter_correlation(
    fsi: pd.Series,
    volatility: pd.Series,
    title: str = 'FSI vs Volatility Scatter',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Plot scatter plot of FSI vs volatility with regression line.

    Args:
        fsi: FSI series
        volatility: Volatility series
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Align and clean data
    df = pd.DataFrame({'fsi': fsi, 'vol': volatility}).dropna()

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(df['vol'], df['fsi'], alpha=0.5, s=20, color='#2E86AB')

    # Regression line
    z = np.polyfit(df['vol'], df['fsi'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['vol'].min(), df['vol'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2, label='Regression')

    # Correlation
    corr = df['fsi'].corr(df['vol'])

    ax.set_xlabel('Market Volatility (%)', fontsize=11)
    ax.set_ylabel('FSI', fontsize=11)
    ax.set_title(f'{title}\n(r = {corr:.3f})', fontsize=13, fontweight='bold')
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def create_all_plots(
    daily_fsi: pd.DataFrame,
    market_data: pd.DataFrame,
    output_dir: str = 'output/plots',
    fsi_column: str = 'fsi',
    vol_column: str = 'realized_vol_30d',
) -> List[plt.Figure]:
    """
    Create all standard validation plots.

    Args:
        daily_fsi: Daily FSI DataFrame
        market_data: Market data DataFrame
        output_dir: Directory to save plots
        fsi_column: Name of FSI column
        vol_column: Name of volatility column

    Returns:
        List of matplotlib Figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    figures = []

    # 1. FSI time series
    fig1 = plot_fsi_timeseries(
        daily_fsi,
        fsi_column=fsi_column,
        save_path=f'{output_dir}/fsi_timeseries.png',
    )
    figures.append(fig1)

    # 2. FSI vs Volatility
    fig2 = plot_fsi_vs_volatility(
        daily_fsi,
        market_data,
        fsi_column=fsi_column,
        vol_column=vol_column,
        save_path=f'{output_dir}/fsi_vs_volatility.png',
    )
    figures.append(fig2)

    # 3. Scatter correlation
    merged = pd.merge(
        daily_fsi[['date', fsi_column]],
        market_data[['date', vol_column]],
        on='date',
    )
    fig3 = plot_scatter_correlation(
        merged[fsi_column],
        merged[vol_column],
        save_path=f'{output_dir}/fsi_scatter.png',
    )
    figures.append(fig3)

    return figures


if __name__ == '__main__':
    print("Validation module loaded.")
    print("Use validate_fsi() to validate FSI against market data.")
    print("Use create_all_plots() to generate all visualizations.")
