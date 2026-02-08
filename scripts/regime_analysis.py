#!/usr/bin/env python3
"""
Financial Stress Regime Analysis
=================================
Implements Markov Switching Model for identifying financial stress regimes.

Mathematical Framework (Hamilton, 1989):
    The model assumes mean (μ) and variance (σ) depend on a latent regime state S_t ∈ {0, 1}:

    y_t = μ_{S_t} + σ_{S_t} * ε_t,    ε_t ~ N(0, 1)

    Where:
    - S_t = 0: Calm regime (low mean, low variance)
    - S_t = 1: Crisis regime (high mean, high variance)

    Regime transitions follow a Markov chain with transition matrix P:

    P = | p_00  p_01 |
        | p_10  p_11 |

    Where p_ij = P(S_t = j | S_{t-1} = i)

Output:
    - Regime probabilities time series
    - Transition probability matrix
    - Regime-specific statistics
    - Classification of historical periods

References:
    - Hamilton, J.D. (1989): A New Approach to the Economic Analysis of
      Nonstationary Time Series and the Business Cycle
    - Hamilton, J.D. (1994): Time Series Analysis, Chapter 22
    - Kim & Nelson (1999): State-Space Models with Regime Switching

Usage:
    python scripts/regime_analysis.py
    python scripts/regime_analysis.py --fsi combined_fsi
    python scripts/regime_analysis.py --smooth  # Use Kalman-smoothed data

Author: Claude Code for CAEN/UFC Master's Thesis
"""

import argparse
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Statsmodels for Markov Switching
try:
    import statsmodels.api as sm
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Markov Switching will not work.")

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

# Color palette for regimes
COLORS = {
    'calm': '#548235',          # Green for calm regime
    'crisis': '#D9534F',        # Red for crisis regime
    'fsi': '#4472C4',           # Blue for FSI
    'probability': '#7030A0',   # Purple for probabilities
    'neutral': '#808080',       # Gray
    'ci_fill': '#E8C1C1',       # Light red for CI
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

OUTPUT_DIR = _PROJECT_DIR / 'output'
PLOTS_DIR = OUTPUT_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Analysis start date (standardize to 2008 onwards)
ANALYSIS_START_DATE = pd.Timestamp('2008-01-01')


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RegimeResults:
    """Results from Markov Switching Model estimation."""
    # Regime probabilities
    smoothed_probs: np.ndarray      # P(S_t = 1 | all data)
    filtered_probs: np.ndarray      # P(S_t = 1 | data up to t)

    # Regime parameters
    mu_calm: float                  # Mean in calm regime
    mu_crisis: float                # Mean in crisis regime
    sigma_calm: float               # Std dev in calm regime
    sigma_crisis: float             # Std dev in crisis regime

    # Transition probabilities
    p_calm_calm: float              # P(calm | calm)
    p_calm_crisis: float            # P(crisis | calm)
    p_crisis_calm: float            # P(calm | crisis)
    p_crisis_crisis: float          # P(crisis | crisis)

    # Expected durations
    duration_calm: float            # Expected duration of calm regime
    duration_crisis: float          # Expected duration of crisis regime

    # Model diagnostics
    log_likelihood: float
    aic: float
    bic: float
    n_obs: int

    # Dates
    dates: pd.DatetimeIndex


# =============================================================================
# MARKOV SWITCHING MODEL
# =============================================================================

class MarkovSwitchingFSI:
    """
    Markov Switching Model for Financial Stress Index.

    Estimates a two-regime model where:
    - Regime 0 (Calm): Low mean, low variance
    - Regime 1 (Crisis): High mean, high variance

    The model uses switching mean and variance to capture the asymmetric
    behavior of financial stress during calm and crisis periods.
    """

    def __init__(self, k_regimes: int = 2, switching_variance: bool = True):
        """
        Initialize Markov Switching Model.

        Parameters
        ----------
        k_regimes : int
            Number of regimes (default: 2 for calm/crisis)
        switching_variance : bool
            Whether variance switches between regimes (default: True)
        """
        self.k_regimes = k_regimes
        self.switching_variance = switching_variance
        self.model = None
        self.results = None
        self._fitted = False

    def fit(self, fsi: pd.Series) -> 'MarkovSwitchingFSI':
        """
        Fit the Markov Switching Model.

        Parameters
        ----------
        fsi : Series
            FSI time series (should be stationary, 0-1 scale OK)

        Returns
        -------
        self
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for Markov Switching")

        # Prepare data - remove NaN
        data = fsi.dropna()

        if len(data) < 50:
            raise ValueError("Need at least 50 observations for regime estimation")

        print(f"  Fitting Markov Switching Model...")
        print(f"    Observations: {len(data)}")
        print(f"    Period: {data.index.min()} to {data.index.max()}")

        # Fit Markov Switching Model
        # Using MarkovRegression with constant only (intercept switching)
        self.model = MarkovRegression(
            data,
            k_regimes=self.k_regimes,
            trend='c',  # Constant/intercept only
            switching_variance=self.switching_variance,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.results = self.model.fit(maxiter=500, disp=False)

        self._fitted = True
        print(f"    Log-likelihood: {self.results.llf:.2f}")
        print(f"    AIC: {self.results.aic:.2f}")
        print(f"    BIC: {self.results.bic:.2f}")

        return self

    def get_results(self) -> RegimeResults:
        """
        Extract results from fitted model.

        Returns
        -------
        RegimeResults
            Dataclass with all regime analysis results
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")

        # Get regime probabilities
        smoothed = self.results.smoothed_marginal_probabilities
        filtered = self.results.filtered_marginal_probabilities

        # Get parameters
        params = self.results.params

        # For 2-regime model with switching variance:
        # params structure depends on model specification
        # Typically: [const_0, const_1, sigma_0, sigma_1, p[0->0], p[1->1]]

        try:
            # Extract regime means using regime-specific approach
            # The model stores parameters in a specific order depending on specification
            # Try to get regime-specific means from the model summary

            # Method 1: Try accessing regime-specific constants directly
            if hasattr(self.results, 'expected_durations'):
                # Some versions have this attribute
                pass

            # Get the number of parameters
            n_params = len(params)

            # For 2-regime model with switching variance:
            # Typically: [const[0], const[1], sigma[0], sigma[1], ...]
            # But parameter ordering varies by statsmodels version

            # Try to extract means from the param names if available
            param_names = self.results.param_names if hasattr(self.results, 'param_names') else []

            mu_0, mu_1 = None, None
            sigma_0, sigma_1 = None, None

            # Search for const/intercept parameters
            for i, name in enumerate(param_names):
                name_lower = name.lower()
                if 'const' in name_lower or 'intercept' in name_lower:
                    if '[0]' in name or 'regime 0' in name_lower:
                        mu_0 = params[i]
                    elif '[1]' in name or 'regime 1' in name_lower:
                        mu_1 = params[i]
                if 'sigma' in name_lower or 'variance' in name_lower:
                    if '[0]' in name or 'regime 0' in name_lower:
                        sigma_0 = np.exp(params[i]) if params[i] < 10 else params[i]
                    elif '[1]' in name or 'regime 1' in name_lower:
                        sigma_1 = np.exp(params[i]) if params[i] < 10 else params[i]

            # Fallback: Use positional extraction if named extraction failed
            if mu_0 is None or mu_1 is None:
                mu_0 = float(params[0])
                mu_1 = float(params[1]) if n_params > 1 else float(params[0])

            if sigma_0 is None or sigma_1 is None:
                if self.switching_variance and n_params >= 4:
                    sigma_0 = np.exp(float(params[2])) if float(params[2]) < 10 else float(params[2])
                    sigma_1 = np.exp(float(params[3])) if float(params[3]) < 10 else float(params[3])
                elif n_params >= 3:
                    sigma_0 = sigma_1 = np.exp(float(params[2])) if float(params[2]) < 10 else float(params[2])
                else:
                    # Estimate from data
                    sigma_0 = sigma_1 = float(self.model.endog.std())

            # Convert to Python floats
            mu_0, mu_1 = float(mu_0), float(mu_1)
            sigma_0, sigma_1 = float(sigma_0), float(sigma_1)

            # Identify which regime is "crisis" (higher mean = more stress)
            if mu_0 > mu_1:
                # Swap so regime 1 is crisis
                mu_calm, mu_crisis = mu_1, mu_0
                sigma_calm, sigma_crisis = sigma_1, sigma_0
                smoothed_crisis = smoothed[0]
                filtered_crisis = filtered[0]
            else:
                mu_calm, mu_crisis = mu_0, mu_1
                sigma_calm, sigma_crisis = sigma_0, sigma_1
                smoothed_crisis = smoothed[1] if smoothed.shape[0] > 1 else smoothed[0]
                filtered_crisis = filtered[1] if filtered.shape[0] > 1 else filtered[0]

        except Exception as e:
            warnings.warn(f"Parameter extraction error: {e}. Using empirical estimates.")
            # Use empirical estimates based on data quantiles
            data_values = self.model.endog.values if hasattr(self.model.endog, 'values') else self.model.endog
            mu_calm = float(np.percentile(data_values, 25))  # Lower quartile for calm
            mu_crisis = float(np.percentile(data_values, 75))  # Upper quartile for crisis
            sigma_calm = float(np.std(data_values[data_values < np.median(data_values)]))
            sigma_crisis = float(np.std(data_values[data_values >= np.median(data_values)]))
            smoothed_crisis = smoothed[0]
            filtered_crisis = filtered[0]

        # Get transition matrix
        trans_mat = self.results.regime_transition

        # Expected durations (ergodic)
        # Duration = 1 / (1 - p_ii)
        # Extract scalar values from transition matrix
        p_00 = float(trans_mat[0, 0].item() if hasattr(trans_mat[0, 0], 'item') else trans_mat[0, 0])
        p_11 = float(trans_mat[1, 1].item() if hasattr(trans_mat[1, 1], 'item') else trans_mat[1, 1])
        duration_calm = 1 / (1 - p_00) if p_00 < 1 else float('inf')
        duration_crisis = 1 / (1 - p_11) if p_11 < 1 else float('inf')

        # Get dates
        dates = self.model.endog.index if hasattr(self.model.endog, 'index') else None
        if dates is None:
            dates = pd.RangeIndex(len(smoothed_crisis))

        # Convert all transition matrix values to Python floats
        p_calm_calm = float(trans_mat[0, 0].item() if hasattr(trans_mat[0, 0], 'item') else trans_mat[0, 0])
        p_calm_crisis = float(trans_mat[0, 1].item() if hasattr(trans_mat[0, 1], 'item') else trans_mat[0, 1])
        p_crisis_calm = float(trans_mat[1, 0].item() if hasattr(trans_mat[1, 0], 'item') else trans_mat[1, 0])
        p_crisis_crisis = float(trans_mat[1, 1].item() if hasattr(trans_mat[1, 1], 'item') else trans_mat[1, 1])

        return RegimeResults(
            smoothed_probs=smoothed_crisis,
            filtered_probs=filtered_crisis,
            mu_calm=mu_calm,
            mu_crisis=mu_crisis,
            sigma_calm=sigma_calm,
            sigma_crisis=sigma_crisis,
            p_calm_calm=p_calm_calm,
            p_calm_crisis=p_calm_crisis,
            p_crisis_calm=p_crisis_calm,
            p_crisis_crisis=p_crisis_crisis,
            duration_calm=duration_calm,
            duration_crisis=duration_crisis,
            log_likelihood=self.results.llf,
            aic=self.results.aic,
            bic=self.results.bic,
            n_obs=len(self.model.endog),
            dates=dates,
        )


# =============================================================================
# REGIME CLASSIFICATION
# =============================================================================

def classify_regimes(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Classify periods into regimes based on crisis probability.

    Parameters
    ----------
    probs : array
        Crisis regime probabilities
    threshold : float
        Classification threshold (default: 0.5)

    Returns
    -------
    array
        Binary classification (0=calm, 1=crisis)
    """
    return (probs > threshold).astype(int)


def identify_crisis_periods(dates: pd.DatetimeIndex, probs: np.ndarray,
                            threshold: float = 0.7,
                            min_duration: int = 4) -> List[Dict]:
    """
    Identify distinct crisis periods from regime probabilities.

    Parameters
    ----------
    dates : DatetimeIndex
        Date index
    probs : array
        Crisis probabilities
    threshold : float
        Probability threshold for crisis classification
    min_duration : int
        Minimum weeks for a valid crisis period

    Returns
    -------
    list of dict
        Identified crisis periods with start, end, duration, mean probability
    """
    # Ensure dates is a proper DatetimeIndex
    if not isinstance(dates, pd.DatetimeIndex):
        # If it's a RangeIndex or other, we can't identify crisis periods properly
        # Return empty list
        if isinstance(dates, pd.RangeIndex):
            warnings.warn("Dates are RangeIndex, cannot identify crisis periods by date")
            return []
        # Try to convert to DatetimeIndex
        try:
            dates = pd.DatetimeIndex(dates)
        except Exception:
            warnings.warn("Could not convert dates to DatetimeIndex")
            return []

    crisis_mask = probs > threshold
    periods = []

    in_crisis = False
    start_idx = None

    for i, is_crisis in enumerate(crisis_mask):
        if is_crisis and not in_crisis:
            # Start of crisis
            in_crisis = True
            start_idx = i
        elif not is_crisis and in_crisis:
            # End of crisis
            in_crisis = False
            if i - start_idx >= min_duration:
                periods.append({
                    'start': pd.Timestamp(dates[start_idx]),
                    'end': pd.Timestamp(dates[i-1]),
                    'duration_weeks': i - start_idx,
                    'mean_prob': float(probs[start_idx:i].mean()),
                    'max_prob': float(probs[start_idx:i].max()),
                })

    # Handle ongoing crisis at end
    if in_crisis and len(dates) - start_idx >= min_duration:
        periods.append({
            'start': pd.Timestamp(dates[start_idx]),
            'end': pd.Timestamp(dates[-1]),
            'duration_weeks': len(dates) - start_idx,
            'mean_prob': float(probs[start_idx:].mean()),
            'max_prob': float(probs[start_idx:].max()),
        })

    return periods


def compare_with_known_crises(identified: List[Dict], known: Dict) -> pd.DataFrame:
    """
    Compare model-identified crises with known historical crises.

    Parameters
    ----------
    identified : list
        Model-identified crisis periods
    known : dict
        Known crisis periods from CRISIS_EPISODES

    Returns
    -------
    DataFrame
        Comparison table
    """
    results = []

    for (start, end), name in known.items():
        known_start = pd.to_datetime(start)
        known_end = pd.to_datetime(end)

        # Check if any identified period overlaps
        matched = False
        for period in identified:
            # Ensure period dates are Timestamps
            period_start = pd.Timestamp(period['start'])
            period_end = pd.Timestamp(period['end'])

            if (period_start <= known_end and period_end >= known_start):
                matched = True
                overlap_start = max(period_start, known_start)
                overlap_end = min(period_end, known_end)
                overlap_days = (overlap_end - overlap_start).days

                results.append({
                    'Crisis': name,
                    'Known_Start': known_start.date(),
                    'Known_End': known_end.date(),
                    'Detected': 'Yes',
                    'Detected_Start': period_start.date(),
                    'Detected_End': period_end.date(),
                    'Overlap_Days': overlap_days,
                    'Mean_Prob': period['mean_prob'],
                })
                break

        if not matched:
            results.append({
                'Crisis': name,
                'Known_Start': known_start.date(),
                'Known_End': known_end.date(),
                'Detected': 'No',
                'Detected_Start': None,
                'Detected_End': None,
                'Overlap_Days': 0,
                'Mean_Prob': None,
            })

    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_regime_analysis(fsi: pd.Series, results: RegimeResults,
                          output_path: Path = None):
    """
    Create comprehensive regime analysis visualization.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    dates = results.dates
    if not isinstance(dates, pd.DatetimeIndex):
        dates = fsi.dropna().index

    # Plot 1: FSI with regime coloring
    ax1 = axes[0]
    fsi_clean = fsi.dropna()

    ax1.plot(dates, fsi_clean.values, color=COLORS['fsi'], linewidth=1.2, label='FSI')
    ax1.axhline(y=results.mu_calm, color=COLORS['calm'], linestyle='--',
               alpha=0.7, label=f'Calm mean ({results.mu_calm:.2f})')
    ax1.axhline(y=results.mu_crisis, color=COLORS['crisis'], linestyle='--',
               alpha=0.7, label=f'Crisis mean ({results.mu_crisis:.2f})')

    # Shade crisis periods
    crisis_mask = results.smoothed_probs > 0.5
    for i in range(len(crisis_mask)):
        if crisis_mask[i]:
            ax1.axvspan(dates[i], dates[min(i+1, len(dates)-1)],
                       alpha=0.15, color=COLORS['crisis'])

    ax1.set_ylabel('FSI (0-1 scale)')
    ax1.set_ylim(0, 1)
    ax1.set_title('Financial Stress Index with Regime Classification')
    ax1.legend(loc='upper left')

    # Plot 2: Crisis probability
    ax2 = axes[1]
    ax2.fill_between(dates, 0, results.smoothed_probs,
                    alpha=0.5, color=COLORS['probability'], label='P(Crisis)')
    ax2.plot(dates, results.smoothed_probs, color=COLORS['probability'], linewidth=1)
    ax2.axhline(y=0.5, color=COLORS['neutral'], linestyle='--', alpha=0.5)

    ax2.set_ylabel('P(Crisis Regime)')
    ax2.set_ylim(0, 1)
    ax2.set_title('Smoothed Crisis Regime Probability')
    ax2.legend(loc='upper right')

    # Add known crisis markers
    for (start, end), name in CRISIS_EPISODES.items():
        start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
        if dates.min() <= end_dt and dates.max() >= start_dt:
            ax2.axvspan(start_dt, end_dt, alpha=0.1, color='black')

    # Plot 3: Regime classification
    ax3 = axes[2]
    regime_class = classify_regimes(results.smoothed_probs)

    ax3.fill_between(dates, 0, regime_class,
                    where=regime_class == 1,
                    alpha=0.7, color=COLORS['crisis'], label='Crisis Regime')
    ax3.fill_between(dates, 0, 1-regime_class,
                    where=regime_class == 0,
                    alpha=0.7, color=COLORS['calm'], label='Calm Regime')

    ax3.set_ylabel('Regime')
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.25, 0.75])
    ax3.set_yticklabels(['Calm', 'Crisis'])
    ax3.set_title('Regime Classification (threshold = 0.5)')
    ax3.legend(loc='upper right')

    # Time axis formatting
    date_range = (dates.max() - dates.min()).days / 365
    if date_range > 10:
        ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    else:
        ax3.xaxis.set_major_locator(mdates.YearLocator(1))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


def plot_transition_matrix(results: RegimeResults, output_path: Path = None):
    """
    Plot transition probability matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    trans_mat = np.array([
        [results.p_calm_calm, results.p_calm_crisis],
        [results.p_crisis_calm, results.p_crisis_crisis]
    ])

    im = ax.imshow(trans_mat, cmap='Blues', vmin=0, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Transition Probability', rotation=-90, va='bottom')

    # Labels
    labels = ['Calm', 'Crisis']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')

    # Add values
    for i in range(2):
        for j in range(2):
            color = 'white' if trans_mat[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{trans_mat[i, j]:.3f}',
                   ha='center', va='center', color=color, fontsize=12)

    ax.set_title('Markov Transition Probability Matrix')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_regime_report(fsi: pd.Series, results: RegimeResults,
                           comparison_df: pd.DataFrame,
                           output_path: Path = None) -> str:
    """
    Generate comprehensive regime analysis report.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("MARKOV SWITCHING REGIME ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # Model summary
    lines.append("\n" + "=" * 70)
    lines.append("1. MODEL SPECIFICATION")
    lines.append("=" * 70)
    lines.append("""
    Model: Two-Regime Markov Switching Model (Hamilton, 1989)

    Observation equation:
        y_t = μ_{S_t} + σ_{S_t} * ε_t,    ε_t ~ N(0, 1)

    Where S_t ∈ {0, 1} follows a first-order Markov chain.

    Estimation: Maximum Likelihood (EM algorithm)
    """)

    # Regime parameters
    lines.append("\n" + "=" * 70)
    lines.append("2. REGIME PARAMETERS")
    lines.append("=" * 70)

    # Duration values are already Python floats from get_results()
    dur_calm = results.duration_calm
    dur_crisis = results.duration_crisis
    dur_calm_str = '∞' if dur_calm == float('inf') else f'{dur_calm:.1f}'
    dur_crisis_str = '∞' if dur_crisis == float('inf') else f'{dur_crisis:.1f}'

    lines.append("\n  CALM REGIME (S=0):")
    lines.append(f"    Mean (μ₀):           {results.mu_calm:.4f}")
    lines.append(f"    Std Dev (σ₀):        {results.sigma_calm:.4f}")
    lines.append(f"    Expected Duration:   {dur_calm_str} weeks")

    lines.append("\n  CRISIS REGIME (S=1):")
    lines.append(f"    Mean (μ₁):           {results.mu_crisis:.4f}")
    lines.append(f"    Std Dev (σ₁):        {results.sigma_crisis:.4f}")
    lines.append(f"    Expected Duration:   {dur_crisis_str} weeks")

    # Transition matrix
    lines.append("\n" + "=" * 70)
    lines.append("3. TRANSITION PROBABILITY MATRIX")
    lines.append("=" * 70)
    lines.append("""
    P = | P(Calm→Calm)   P(Calm→Crisis)  |
        | P(Crisis→Calm) P(Crisis→Crisis)|
    """)
    lines.append(f"    P = | {results.p_calm_calm:.4f}  {results.p_calm_crisis:.4f} |")
    lines.append(f"        | {results.p_crisis_calm:.4f}  {results.p_crisis_crisis:.4f} |")

    # Model fit
    lines.append("\n" + "=" * 70)
    lines.append("4. MODEL FIT")
    lines.append("=" * 70)
    lines.append(f"    Observations:     {results.n_obs}")
    lines.append(f"    Log-Likelihood:   {results.log_likelihood:.2f}")
    lines.append(f"    AIC:              {results.aic:.2f}")
    lines.append(f"    BIC:              {results.bic:.2f}")

    # Crisis detection
    lines.append("\n" + "=" * 70)
    lines.append("5. HISTORICAL CRISIS DETECTION")
    lines.append("=" * 70)

    if not comparison_df.empty:
        lines.append("\n" + comparison_df.to_string(index=False))

        detected = comparison_df['Detected'].value_counts()
        n_detected = detected.get('Yes', 0)
        n_total = len(comparison_df)
        lines.append(f"\n    Detection rate: {n_detected}/{n_total} ({100*n_detected/n_total:.1f}%)")

    # Interpretation
    lines.append("\n" + "=" * 70)
    lines.append("6. INTERPRETATION")
    lines.append("=" * 70)
    # Format durations for interpretation section
    dur_calm_int = '∞' if dur_calm == float('inf') else f'{dur_calm:.0f}'
    dur_crisis_int = '∞' if dur_crisis == float('inf') else f'{dur_crisis:.0f}'

    lines.append(f"""
    The Markov Switching Model identifies two distinct regimes:

    1. CALM REGIME:
       - Mean FSI: {results.mu_calm:.3f} (lower stress)
       - Volatility: {results.sigma_calm:.3f}
       - Tends to persist for ~{dur_calm_int} weeks on average

    2. CRISIS REGIME:
       - Mean FSI: {results.mu_crisis:.3f} (higher stress)
       - Volatility: {results.sigma_crisis:.3f}
       - Tends to persist for ~{dur_crisis_int} weeks on average

    The smoothed probabilities provide the optimal estimate of being in each
    regime at each point in time, conditioning on all available data.

    High crisis probability (>0.7) indicates elevated financial stress with
    high confidence. These periods should align with known economic crises.
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
    parser = argparse.ArgumentParser(description='Markov Switching Regime Analysis')
    parser.add_argument('--fsi', type=str, default='combined_fsi',
                        help='FSI column to analyze (default: combined_fsi)')
    parser.add_argument('--smooth', action='store_true',
                        help='Use Kalman-smoothed FSI data')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Crisis probability threshold (default: 0.7)')
    args = parser.parse_args()

    print("=" * 70)
    print("MARKOV SWITCHING REGIME ANALYSIS")
    print("Hamilton (1989) Two-Regime Model")
    print("=" * 70)

    if not STATSMODELS_AVAILABLE:
        print("\n  ERROR: statsmodels required for Markov Switching analysis")
        print("  Install with: pip install statsmodels")
        return

    # Load FSI data
    print("\nLoading FSI data...")

    if args.smooth:
        fsi_path = OUTPUT_DIR / 'fsi_smoothed.csv'
        fsi_col = f'{args.fsi}_smoothed'
    else:
        fsi_path = OUTPUT_DIR / 'combined_fsi.csv'
        fsi_col = args.fsi

    if not fsi_path.exists():
        # Try alternative paths
        alt_paths = [
            OUTPUT_DIR / 'combined_fsi_weighted.csv',
            OUTPUT_DIR / 'results' / 'fsi_weekly.csv',
        ]
        for alt in alt_paths:
            if alt.exists():
                fsi_path = alt
                break

    if not fsi_path.exists():
        print(f"\n  ERROR: FSI data not found at {fsi_path}")
        print("  Run: python scripts/combined_fsi.py first")
        return

    df = pd.read_csv(fsi_path)
    df['date'] = pd.to_datetime(df['date'])
    # Filter to 2008 onwards
    df = df[df['date'] >= ANALYSIS_START_DATE]
    df = df.set_index('date')
    print(f"  Loaded {len(df)} observations from {fsi_path.name} (2008+)")

    # Select FSI column
    if fsi_col not in df.columns:
        # Try to find a suitable column
        fsi_cols = [c for c in df.columns if 'fsi' in c.lower()]
        if fsi_cols:
            fsi_col = fsi_cols[0]
        else:
            print(f"  ERROR: Column '{fsi_col}' not found")
            print(f"  Available columns: {list(df.columns)}")
            return

    fsi = df[fsi_col]
    print(f"  Using column: {fsi_col}")

    # Fit Markov Switching Model
    print("\nFitting Markov Switching Model...")
    model = MarkovSwitchingFSI(k_regimes=2, switching_variance=True)

    try:
        model.fit(fsi)
        results = model.get_results()
    except Exception as e:
        print(f"\n  ERROR: Model fitting failed: {e}")
        return

    # Print summary
    print("\n" + "-" * 40)
    print("Regime Parameters:")
    print("-" * 40)
    print(f"  Calm Regime:   μ = {results.mu_calm:.3f}, σ = {results.sigma_calm:.3f}")
    print(f"  Crisis Regime: μ = {results.mu_crisis:.3f}, σ = {results.sigma_crisis:.3f}")
    # Duration values are already Python floats from get_results()
    dur_calm = results.duration_calm
    dur_crisis = results.duration_crisis
    dur_calm_str = '∞' if dur_calm == float('inf') else f'{dur_calm:.1f}'
    dur_crisis_str = '∞' if dur_crisis == float('inf') else f'{dur_crisis:.1f}'
    print(f"  Expected durations: Calm = {dur_calm_str} weeks, Crisis = {dur_crisis_str} weeks")

    # Identify crisis periods
    print("\nIdentifying crisis periods...")
    identified = identify_crisis_periods(
        results.dates, results.smoothed_probs,
        threshold=args.threshold, min_duration=4
    )
    print(f"  Found {len(identified)} crisis periods")

    # Compare with known crises
    comparison_df = compare_with_known_crises(identified, CRISIS_EPISODES)

    # Generate report
    print("\nGenerating report...")
    report = generate_regime_report(
        fsi, results, comparison_df,
        OUTPUT_DIR / 'regime_analysis_report.txt'
    )

    # Save regime probabilities
    regime_df = pd.DataFrame({
        'date': results.dates,
        'crisis_probability': results.smoothed_probs,
        'regime': classify_regimes(results.smoothed_probs),
    })
    regime_df.to_csv(OUTPUT_DIR / 'regime_probabilities.csv', index=False)
    print(f"  Saved: {OUTPUT_DIR / 'regime_probabilities.csv'}")

    # Generate plots
    print("\nGenerating plots...")
    plot_regime_analysis(fsi, results, PLOTS_DIR / 'regime_analysis.png')
    plot_transition_matrix(results, PLOTS_DIR / 'transition_matrix.png')

    print("\n" + "=" * 70)
    print("REGIME ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n   Output files:")
    print(f"   - {OUTPUT_DIR}/regime_analysis_report.txt")
    print(f"   - {OUTPUT_DIR}/regime_probabilities.csv")
    print(f"   - {PLOTS_DIR}/regime_analysis.png")
    print(f"   - {PLOTS_DIR}/transition_matrix.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
