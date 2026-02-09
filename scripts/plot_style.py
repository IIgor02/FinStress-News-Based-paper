#!/usr/bin/env python3
"""
Academic Plot Styling Configuration
====================================
Standardized plot styling for publication-quality figures.

Uses grayscale/minimal color palette following academic conventions.
All figures use consistent fonts, sizes, and formatting.

Usage:
    from scripts.plot_style import apply_style, COLORS, save_figure

    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE['single'])
    # ... create plot ...
    save_figure(fig, 'figure_name', output_dir)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

# =============================================================================
# ACADEMIC COLOR PALETTE (Grayscale + minimal accent)
# =============================================================================

COLORS = {
    # Primary grayscale
    'black': '#000000',
    'dark_gray': '#404040',
    'gray': '#808080',
    'light_gray': '#B0B0B0',
    'very_light_gray': '#D0D0D0',
    'white': '#FFFFFF',

    # For filled regions (very subtle)
    'fill_dark': '#606060',
    'fill_light': '#E0E0E0',
    'fill_positive': '#C0C0C0',
    'fill_negative': '#E8E8E8',

    # Accent colors (use sparingly)
    'accent_dark': '#2C2C2C',
    'accent_medium': '#666666',

    # Line styles for multiple series
    'series_1': '#000000',  # Black (solid)
    'series_2': '#404040',  # Dark gray (dashed)
    'series_3': '#808080',  # Gray (dotted)
    'series_4': '#B0B0B0',  # Light gray (dash-dot)

    # Special markers
    'crisis': '#404040',
    'confidence': '#D0D0D0',
    'zero_line': '#808080',
    'grid': '#E0E0E0',
}

# Line styles for multiple series
LINE_STYLES = {
    'series_1': '-',       # Solid
    'series_2': '--',      # Dashed
    'series_3': ':',       # Dotted
    'series_4': '-.',      # Dash-dot
}

# =============================================================================
# FIGURE SIZES (inches) - Standard academic sizes
# =============================================================================

FIGSIZE = {
    'single': (7, 4.5),         # Single column
    'double': (14, 4.5),        # Double column / wide
    'tall': (7, 7),             # Square-ish
    'panel_2x1': (7, 8),        # 2 rows, 1 column
    'panel_2x2': (10, 8),       # 2x2 grid
    'panel_3x1': (7, 10),       # 3 rows, 1 column
    'panel_1x2': (12, 4.5),     # 1 row, 2 columns
    'irf': (8, 5),              # IRF plots
    'comparison': (12, 8),      # Comparison plots
}

# =============================================================================
# STYLE APPLICATION
# =============================================================================

def apply_style():
    """Apply academic plot style globally."""
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif'],
        'font.size': 10,

        # Axes
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'normal',
        'axes.linewidth': 0.8,
        'axes.edgecolor': COLORS['dark_gray'],
        'axes.labelcolor': COLORS['black'],
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.facecolor': COLORS['white'],

        # Grid
        'grid.linewidth': 0.5,
        'grid.alpha': 0.4,
        'grid.color': COLORS['grid'],
        'grid.linestyle': '-',

        # Ticks
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.color': COLORS['dark_gray'],
        'ytick.color': COLORS['dark_gray'],
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,

        # Legend
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': COLORS['light_gray'],
        'legend.facecolor': COLORS['white'],

        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 5,

        # Figure
        'figure.dpi': 150,
        'figure.facecolor': COLORS['white'],
        'figure.edgecolor': COLORS['white'],
        'figure.autolayout': False,

        # Saving
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.facecolor': COLORS['white'],
        'savefig.edgecolor': COLORS['white'],

        # Text
        'text.color': COLORS['black'],

        # Math text
        'mathtext.fontset': 'dejavuserif',
    })


def save_figure(fig, name: str, output_dir: Path, formats: list = None):
    """
    Save figure in publication-quality format.

    Args:
        fig: matplotlib figure
        name: filename without extension
        output_dir: output directory path
        formats: list of formats (default: ['png', 'pdf'])
    """
    if formats is None:
        formats = ['png']  # PDF optional for final version

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        filepath = output_dir / f"{name}.{fmt}"
        fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  Saved: {filepath}")


# =============================================================================
# COMMON PLOT ELEMENTS
# =============================================================================

def add_zero_line(ax, orientation='horizontal'):
    """Add a zero reference line."""
    if orientation == 'horizontal':
        ax.axhline(y=0, color=COLORS['zero_line'], linestyle='--',
                   linewidth=0.8, alpha=0.7, zorder=1)
    else:
        ax.axvline(x=0, color=COLORS['zero_line'], linestyle='--',
                   linewidth=0.8, alpha=0.7, zorder=1)


def add_crisis_shading(ax, crisis_periods: dict, alpha=0.15):
    """
    Add vertical shading for crisis periods.

    Args:
        ax: matplotlib axes
        crisis_periods: dict of {(start, end): label}
        alpha: transparency of shading
    """
    import pandas as pd

    xlim = ax.get_xlim()

    for (start, end), label in crisis_periods.items():
        start_dt = mdates.date2num(pd.to_datetime(start))
        end_dt = mdates.date2num(pd.to_datetime(end))

        # Only add if within plot range
        if end_dt >= xlim[0] and start_dt <= xlim[1]:
            ax.axvspan(start_dt, end_dt, alpha=alpha,
                       color=COLORS['fill_dark'], zorder=0)


def format_date_axis(ax, data_years: float = None):
    """
    Format x-axis for time series data.

    Args:
        ax: matplotlib axes
        data_years: approximate years of data (for auto-formatting)
    """
    if data_years is None:
        data_years = 10  # Default assumption

    if data_years > 12:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    elif data_years > 5:
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')


def add_confidence_band(ax, x, y, se, confidence=0.95, color=None, alpha=0.2):
    """
    Add confidence band around a line.

    Args:
        ax: matplotlib axes
        x: x values
        y: y values (point estimates)
        se: standard errors
        confidence: confidence level (default 0.95)
        color: fill color (default: fill_light)
        alpha: transparency
    """
    from scipy import stats

    if color is None:
        color = COLORS['fill_light']

    # Z-score for confidence level
    z = stats.norm.ppf((1 + confidence) / 2)

    lower = y - z * se
    upper = y + z * se

    ax.fill_between(x, lower, upper, alpha=alpha, color=color,
                    edgecolor='none', zorder=2)


# =============================================================================
# SPECIALIZED PLOT FUNCTIONS
# =============================================================================

def plot_irf(ax, horizons, irf, se, title=None, ylabel=None,
             show_significance=True, sig_level=0.1):
    """
    Create a standardized IRF plot.

    Args:
        ax: matplotlib axes
        horizons: array of horizon values
        irf: array of IRF point estimates
        se: array of standard errors
        title: plot title
        ylabel: y-axis label
        show_significance: whether to mark significant periods
        sig_level: significance level for marking
    """
    from scipy import stats

    # Point estimates
    ax.plot(horizons, irf, color=COLORS['black'], linewidth=2,
            marker='o', markersize=4, zorder=4, label='IRF')

    # 95% confidence band
    z_95 = 1.96
    lower_95 = irf - z_95 * se
    upper_95 = irf + z_95 * se
    ax.fill_between(horizons, lower_95, upper_95,
                    alpha=0.2, color=COLORS['gray'],
                    edgecolor='none', zorder=2, label='95% CI')

    # 90% confidence band (darker)
    z_90 = 1.645
    lower_90 = irf - z_90 * se
    upper_90 = irf + z_90 * se
    ax.fill_between(horizons, lower_90, upper_90,
                    alpha=0.3, color=COLORS['dark_gray'],
                    edgecolor='none', zorder=3, label='90% CI')

    # Zero line
    add_zero_line(ax)

    # Mark significant periods
    if show_significance:
        t_stats = np.abs(irf / se)
        t_critical = stats.t.ppf(1 - sig_level/2, df=100)  # approx
        significant = t_stats > t_critical

        for i, (h, is_sig) in enumerate(zip(horizons, significant)):
            if is_sig:
                ax.plot(h, irf[i], 'o', color=COLORS['black'],
                        markersize=7, markerfacecolor=COLORS['white'],
                        markeredgewidth=1.5, zorder=5)

    if title:
        ax.set_title(title, fontweight='normal')
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_xlabel('Months')
    ax.legend(loc='best', framealpha=0.9)


def plot_time_series(ax, dates, values, label=None, color=None,
                     linestyle='-', add_ma=False, ma_window=12):
    """
    Plot a time series with optional moving average.

    Args:
        ax: matplotlib axes
        dates: datetime array
        values: values array
        label: series label
        color: line color
        linestyle: line style
        add_ma: whether to add moving average
        ma_window: moving average window
    """
    import pandas as pd

    if color is None:
        color = COLORS['black']

    ax.plot(dates, values, color=color, linestyle=linestyle,
            linewidth=1.5, label=label, zorder=3)

    if add_ma:
        ma = pd.Series(values).rolling(ma_window, min_periods=1).mean()
        ma_label = f'{label} ({ma_window}-period MA)' if label else f'{ma_window}-period MA'
        ax.plot(dates, ma, color=COLORS['dark_gray'], linestyle='--',
                linewidth=1.2, alpha=0.8, label=ma_label, zorder=4)


def create_dual_axis_plot(fig, ax1, dates, y1, y2,
                          y1_label, y2_label,
                          title=None):
    """
    Create a dual-axis time series plot.

    Args:
        fig: matplotlib figure
        ax1: primary axes
        dates: datetime array
        y1: primary y values
        y2: secondary y values
        y1_label: primary y-axis label
        y2_label: secondary y-axis label
        title: plot title
    """
    # Primary axis
    line1 = ax1.plot(dates, y1, color=COLORS['black'], linewidth=1.5,
                     label=y1_label, zorder=3)
    ax1.set_ylabel(y1_label, color=COLORS['black'])
    ax1.tick_params(axis='y', labelcolor=COLORS['black'])

    # Secondary axis
    ax2 = ax1.twinx()
    line2 = ax2.plot(dates, y2, color=COLORS['gray'], linewidth=1.5,
                     linestyle='--', label=y2_label, zorder=2)
    ax2.set_ylabel(y2_label, color=COLORS['gray'])
    ax2.tick_params(axis='y', labelcolor=COLORS['gray'])

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    if title:
        ax1.set_title(title)

    return ax2


# =============================================================================
# CRISIS PERIODS (Brazilian economic history)
# =============================================================================

CRISIS_PERIODS = {
    ('2008-09-15', '2009-03-31'): 'Global Financial Crisis',
    ('2014-01-01', '2016-12-31'): 'Brazilian Recession',
    ('2020-03-01', '2020-06-30'): 'COVID-19 Pandemic',
}


# =============================================================================
# INITIALIZATION
# =============================================================================

# Apply style on import
apply_style()


if __name__ == '__main__':
    # Demo plot
    print("Academic Plot Style - Demo")
    print("=" * 50)

    fig, ax = plt.subplots(figsize=FIGSIZE['single'])

    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    ax.plot(x, y1, color=COLORS['series_1'], linestyle=LINE_STYLES['series_1'],
            label='Series 1 (sin)')
    ax.plot(x, y2, color=COLORS['series_2'], linestyle=LINE_STYLES['series_2'],
            label='Series 2 (cos)')

    add_zero_line(ax)

    ax.set_xlabel('X-axis Label')
    ax.set_ylabel('Y-axis Label')
    ax.set_title('Demo: Academic Plot Style')
    ax.legend()

    demo_path = Path(__file__).parent.parent / 'output' / 'plots'
    save_figure(fig, 'style_demo', demo_path)
    plt.close()

    print("\nStyle configuration loaded successfully.")
