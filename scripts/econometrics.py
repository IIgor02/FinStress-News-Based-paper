#!/usr/bin/env python3
"""
Econometric Methods for Financial Stress Index
================================================
Implements academic-standard econometric techniques for FSI refinement:

1. Kalman Filter/Smoother: Treats FSI as a latent variable, separating signal from noise
2. Gap Imputation: Uses state-space framework to estimate missing observations
3. Variance Analysis: Pre/post smoothing variance decomposition

Mathematical Framework (State-Space Model):
    Observation equation:  y_t = Z_t * α_t + ε_t,  ε_t ~ N(0, Q_t)
    State equation:        α_{t+1} = T_t * α_t + R_t * η_t,  η_t ~ N(0, R_t)

    Where:
    - y_t: Observed (noisy) FSI
    - α_t: Latent (true) financial stress level
    - ε_t: Observation noise
    - η_t: State transition noise

References:
    - Durbin & Koopman (2012): Time Series Analysis by State Space Methods
    - Hamilton (1994): Time Series Analysis
    - Harvey (1990): Forecasting, Structural Time Series Models

Usage:
    from scripts.econometrics import KalmanSmoother, apply_kalman_smoothing

    # Apply to single series
    smoothed_fsi = apply_kalman_smoothing(raw_fsi)

    # Apply to DataFrame
    smoothed_df = smooth_fsi_dataframe(fsi_df)

Author: Claude Code for CAEN/UFC Master's Thesis
"""

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Statsmodels for state-space models
try:
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Kalman smoothing will be limited.")

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

# Color palette
COLORS = {
    'raw': '#A0A0A0',           # Gray for raw
    'smoothed': '#4472C4',      # Blue for smoothed
    'ci_fill': '#B4C7E7',       # Light blue for CI
    'imputed': '#C55A11',       # Orange for imputed points
    'crisis': '#D9534F',        # Red for crisis
}

# Handle paths
try:
    _SCRIPT_DIR = Path(__file__).parent
    _PROJECT_DIR = _SCRIPT_DIR.parent
except NameError:
    _PROJECT_DIR = Path.cwd()
    _SCRIPT_DIR = _PROJECT_DIR / 'scripts'

sys.path.insert(0, str(_PROJECT_DIR))

OUTPUT_DIR = _PROJECT_DIR / 'output'
PLOTS_DIR = OUTPUT_DIR / 'plots'


# =============================================================================
# DATA CLASSES FOR RESULTS
# =============================================================================

@dataclass
class KalmanResults:
    """Results from Kalman Filter/Smoother application."""
    smoothed: np.ndarray          # Smoothed state estimates
    filtered: np.ndarray          # Filtered state estimates
    smoothed_cov: np.ndarray      # Smoothed state covariance
    filtered_cov: np.ndarray      # Filtered state covariance
    observations: np.ndarray      # Original observations (with NaN)
    imputed_mask: np.ndarray      # Boolean mask of imputed values
    signal_variance: float        # Estimated signal variance
    noise_variance: float         # Estimated noise variance
    signal_to_noise: float        # Signal-to-noise ratio
    log_likelihood: float         # Model log-likelihood
    aic: float                    # Akaike Information Criterion
    bic: float                    # Bayesian Information Criterion


@dataclass
class VarianceDecomposition:
    """Variance decomposition results."""
    raw_variance: float
    smoothed_variance: float
    noise_removed: float          # Percentage of variance removed
    signal_ratio: float           # Signal / (Signal + Noise)


# =============================================================================
# KALMAN SMOOTHER CLASS
# =============================================================================

class KalmanSmoother:
    """
    Kalman Filter/Smoother for Financial Stress Index.

    Uses the Local Level Model (Random Walk + Noise):
        y_t = α_t + ε_t,    ε_t ~ N(0, σ²_ε)  [Observation equation]
        α_{t+1} = α_t + η_t,  η_t ~ N(0, σ²_η)  [State equation]

    This model treats the FSI as a latent variable (α_t) observed with noise.
    The Kalman smoother provides optimal estimates of the latent state.

    Parameters
    ----------
    level : bool
        Include stochastic level component (default: True)
    trend : str or None
        Trend specification: None, 'local' (default: None)
    seasonal : int or None
        Seasonal period (default: None for no seasonality)
    freq_seasonal : list or None
        List of seasonal periods for multiple seasonalities
    """

    def __init__(self, level: bool = True, trend: str = None,
                 seasonal: int = None, freq_seasonal: list = None):
        self.level = level
        self.trend = trend
        self.seasonal = seasonal
        self.freq_seasonal = freq_seasonal
        self.model = None
        self.results = None
        self._fitted = False

    def fit(self, y: Union[np.ndarray, pd.Series],
            freq: str = 'W') -> 'KalmanSmoother':
        """
        Fit the state-space model using MLE.

        Parameters
        ----------
        y : array-like
            Observed FSI series (can contain NaN for missing values)
        freq : str
            Frequency for pandas DatetimeIndex (default: 'W' for weekly)

        Returns
        -------
        self : KalmanSmoother
            Fitted smoother instance
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for Kalman smoothing")

        # Convert to pandas Series with proper index
        if isinstance(y, pd.Series):
            series = y.copy()
        else:
            series = pd.Series(y)

        # Ensure numeric index if not datetime
        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.RangeIndex(len(series))

        # Build Unobserved Components Model
        # Local Level Model: y_t = μ_t + ε_t, μ_{t+1} = μ_t + η_t
        self.model = UnobservedComponents(
            series,
            level='local level' if self.level else None,
            trend=self.trend,
            seasonal=self.seasonal,
            freq_seasonal=self.freq_seasonal,
        )

        # Fit with MLE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.results = self.model.fit(disp=False, maxiter=500)

        self._fitted = True
        return self

    def smooth(self, y: Union[np.ndarray, pd.Series] = None) -> KalmanResults:
        """
        Apply Kalman smoother to obtain optimal state estimates.

        Uses the Rauch-Tung-Striebel (RTS) smoother for two-pass smoothing:
        1. Forward pass: Standard Kalman filter
        2. Backward pass: Backward recursion for smoothed estimates

        Parameters
        ----------
        y : array-like, optional
            New observations to smooth. If None, uses fitted data.

        Returns
        -------
        KalmanResults
            Dataclass containing smoothed estimates and diagnostics
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before smoothing")

        # Get smoothed state
        smoothed_state = self.results.smoothed_state[0]  # Level component
        filtered_state = self.results.filtered_state[0]

        # Get state covariances
        smoothed_cov = self.results.smoothed_state_cov[0, 0]
        filtered_cov = self.results.filtered_state_cov[0, 0]

        # Original observations
        observations = np.array(self.model.endog)

        # Identify imputed values (where original was NaN)
        imputed_mask = np.isnan(observations)

        # Variance decomposition
        # σ²_ε (observation noise) and σ²_η (state noise) from model
        try:
            params = self.results.params
            # For local level model: params[0] = σ²_ε, params[1] = σ²_η
            noise_var = params[0] if len(params) > 0 else np.var(observations[~np.isnan(observations)])
            signal_var = params[1] if len(params) > 1 else noise_var * 0.1
        except Exception:
            noise_var = np.nanvar(observations - smoothed_state)
            signal_var = np.nanvar(smoothed_state)

        snr = signal_var / noise_var if noise_var > 0 else np.inf

        return KalmanResults(
            smoothed=smoothed_state,
            filtered=filtered_state,
            smoothed_cov=smoothed_cov,
            filtered_cov=filtered_cov,
            observations=observations,
            imputed_mask=imputed_mask,
            signal_variance=signal_var,
            noise_variance=noise_var,
            signal_to_noise=snr,
            log_likelihood=self.results.llf,
            aic=self.results.aic,
            bic=self.results.bic,
        )

    def get_confidence_interval(self, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get confidence interval for smoothed state.

        Parameters
        ----------
        alpha : float
            Significance level (default: 0.05 for 95% CI)

        Returns
        -------
        lower, upper : tuple of arrays
            Lower and upper confidence bounds
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")

        smoothed_state = self.results.smoothed_state[0]
        smoothed_cov = self.results.smoothed_state_cov[0, 0]

        z = stats.norm.ppf(1 - alpha/2)
        se = np.sqrt(smoothed_cov)

        lower = smoothed_state - z * se
        upper = smoothed_state + z * se

        return lower, upper


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def apply_kalman_smoothing(series: Union[np.ndarray, pd.Series],
                           return_full: bool = False) -> Union[np.ndarray, Tuple]:
    """
    Apply Kalman smoothing to a single FSI series.

    This is the main function for smoothing raw FSI data and imputing gaps.

    Parameters
    ----------
    series : array-like
        Raw FSI series (can contain NaN for missing values)
    return_full : bool
        If True, return full KalmanResults; otherwise just smoothed values

    Returns
    -------
    smoothed : array or tuple
        Smoothed FSI values, or (smoothed, KalmanResults) if return_full=True
    """
    if not STATSMODELS_AVAILABLE:
        warnings.warn("statsmodels not available, returning original series")
        return series if not return_full else (series, None)

    # Handle edge cases
    if len(series) < 10:
        warnings.warn("Series too short for Kalman smoothing, returning original")
        return series if not return_full else (series, None)

    # Initialize and fit
    smoother = KalmanSmoother(level=True, trend=None)

    try:
        smoother.fit(series)
        results = smoother.smooth()
        smoothed = results.smoothed

        # Clip to valid FSI range [0, 1]
        smoothed = np.clip(smoothed, 0, 1)

        if return_full:
            return smoothed, results
        return smoothed

    except Exception as e:
        warnings.warn(f"Kalman smoothing failed: {e}. Returning original series.")
        return series if not return_full else (series, None)


def smooth_fsi_dataframe(df: pd.DataFrame,
                         fsi_columns: List[str] = None,
                         date_column: str = 'date') -> pd.DataFrame:
    """
    Apply Kalman smoothing to all FSI columns in a DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame with FSI columns
    fsi_columns : list, optional
        Columns to smooth. If None, auto-detects columns containing 'fsi'
    date_column : str
        Name of date column

    Returns
    -------
    DataFrame
        DataFrame with original and smoothed columns (suffixed '_smoothed')
    """
    result = df.copy()

    # Auto-detect FSI columns
    if fsi_columns is None:
        fsi_columns = [c for c in df.columns if 'fsi' in c.lower() and c != date_column]

    smoothing_log = []

    for col in fsi_columns:
        if col not in df.columns:
            continue

        series = df[col].values
        smoothed, kalman_results = apply_kalman_smoothing(series, return_full=True)

        # Add smoothed column
        smoothed_col = f"{col}_smoothed"
        result[smoothed_col] = smoothed

        # Log variance decomposition
        if kalman_results is not None:
            raw_var = np.nanvar(series)
            smoothed_var = np.nanvar(smoothed)
            noise_removed = (1 - smoothed_var / raw_var) * 100 if raw_var > 0 else 0

            smoothing_log.append({
                'column': col,
                'raw_variance': raw_var,
                'smoothed_variance': smoothed_var,
                'noise_removed_pct': noise_removed,
                'n_imputed': kalman_results.imputed_mask.sum(),
                'signal_to_noise': kalman_results.signal_to_noise,
                'aic': kalman_results.aic,
                'bic': kalman_results.bic,
            })

    # Print smoothing log
    if smoothing_log:
        print("\n  Kalman Smoothing Results:")
        print("  " + "-" * 60)
        for entry in smoothing_log:
            print(f"    {entry['column']}:")
            print(f"      Raw variance:      {entry['raw_variance']:.6f}")
            print(f"      Smoothed variance: {entry['smoothed_variance']:.6f}")
            print(f"      Noise removed:     {entry['noise_removed_pct']:.1f}%")
            print(f"      Values imputed:    {entry['n_imputed']}")
            print(f"      Signal-to-noise:   {entry['signal_to_noise']:.3f}")
            print(f"      AIC: {entry['aic']:.2f}, BIC: {entry['bic']:.2f}")

    return result


def impute_gaps_kalman(series: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    Impute missing values using Kalman smoother.

    This uses the state-space framework to estimate missing observations,
    which is superior to linear interpolation as it accounts for the
    stochastic structure of the data.

    Parameters
    ----------
    series : array-like
        Series with missing values (NaN)

    Returns
    -------
    array
        Series with gaps filled via Kalman estimates
    """
    smoothed, results = apply_kalman_smoothing(series, return_full=True)

    if results is None:
        # Fallback to forward-fill then back-fill
        if isinstance(series, pd.Series):
            return series.ffill().bfill().values
        s = pd.Series(series)
        return s.ffill().bfill().values

    return smoothed


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_smoothing_comparison(raw: np.ndarray, smoothed: np.ndarray,
                              dates: pd.DatetimeIndex = None,
                              title: str = 'Kalman Smoothing Comparison',
                              save_path: Path = None) -> None:
    """
    Plot comparison of raw vs smoothed FSI.

    Parameters
    ----------
    raw : array
        Original (raw) FSI values
    smoothed : array
        Kalman-smoothed FSI values
    dates : DatetimeIndex, optional
        Date index for x-axis
    title : str
        Plot title
    save_path : Path, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    if dates is None:
        x = np.arange(len(raw))
        xlabel = 'Observation'
    else:
        x = dates
        xlabel = 'Date'

    # Plot 1: Comparison
    ax1 = axes[0]
    ax1.plot(x, raw, color=COLORS['raw'], linewidth=1, alpha=0.7, label='Raw FSI')
    ax1.plot(x, smoothed, color=COLORS['smoothed'], linewidth=1.5, label='Smoothed FSI')

    # Mark imputed values
    imputed_mask = np.isnan(raw)
    if imputed_mask.any():
        ax1.scatter(x[imputed_mask], smoothed[imputed_mask],
                   color=COLORS['imputed'], s=20, zorder=5, label='Imputed')

    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('FSI (0-1 scale)')
    ax1.set_ylim(0, 1)
    ax1.set_title(title)
    ax1.legend(loc='upper right')

    # Plot 2: Noise removed
    ax2 = axes[1]
    noise = raw - smoothed
    noise_valid = noise[~np.isnan(noise)]

    ax2.fill_between(x[~np.isnan(noise)], 0, noise_valid,
                    where=noise_valid > 0, alpha=0.5, color='#E8C1C1', label='Positive noise')
    ax2.fill_between(x[~np.isnan(noise)], 0, noise_valid,
                    where=noise_valid <= 0, alpha=0.5, color='#C1E8C1', label='Negative noise')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Noise Removed')
    ax2.set_xlabel(xlabel)
    ax2.set_title(f'Removed Noise (Variance reduced by {(1-np.nanvar(smoothed)/np.nanvar(raw))*100:.1f}%)')
    ax2.legend(loc='upper right')

    # Format x-axis
    if dates is not None:
        date_range = (dates.max() - dates.min()).days / 365
        if date_range > 10:
            ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        else:
            ax2.xaxis.set_major_locator(mdates.YearLocator(1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def generate_smoothing_report(df: pd.DataFrame,
                              smoothed_df: pd.DataFrame,
                              output_path: Path = None) -> str:
    """
    Generate detailed report of Kalman smoothing results.

    Parameters
    ----------
    df : DataFrame
        Original DataFrame
    smoothed_df : DataFrame
        DataFrame with smoothed columns
    output_path : Path, optional
        Path to save report

    Returns
    -------
    str
        Report text
    """
    lines = []
    lines.append("=" * 70)
    lines.append("KALMAN SMOOTHING REPORT")
    lines.append("=" * 70)

    lines.append("\n1. METHODOLOGY")
    lines.append("-" * 40)
    lines.append("""
    Model: Local Level (Random Walk + Noise)

    State-space representation:
      Observation: y_t = alpha_t + epsilon_t,  epsilon_t ~ N(0, sigma_eps^2)
      State:       alpha_{t+1} = alpha_t + eta_t,  eta_t ~ N(0, sigma_eta^2)

    Where:
      y_t:       Observed (noisy) FSI
      alpha_t:   Latent (true) financial stress level
      epsilon_t: Observation noise
      eta_t:     State transition noise

    Estimation: Maximum Likelihood (MLE)
    Smoothing: Rauch-Tung-Striebel (RTS) backward recursion
    """)

    lines.append("\n2. VARIANCE DECOMPOSITION")
    lines.append("-" * 40)

    fsi_cols = [c for c in df.columns if 'fsi' in c.lower() and '_smoothed' not in c]

    for col in fsi_cols:
        smoothed_col = f"{col}_smoothed"
        if smoothed_col not in smoothed_df.columns:
            continue

        raw = df[col].values
        smoothed = smoothed_df[smoothed_col].values

        raw_var = np.nanvar(raw)
        smoothed_var = np.nanvar(smoothed)
        noise_removed = (1 - smoothed_var / raw_var) * 100 if raw_var > 0 else 0
        n_gaps = np.isnan(raw).sum()

        lines.append(f"\n  {col}:")
        lines.append(f"    Observations:        {len(raw)}")
        lines.append(f"    Missing values:      {n_gaps} ({100*n_gaps/len(raw):.1f}%)")
        lines.append(f"    Raw variance:        {raw_var:.6f}")
        lines.append(f"    Smoothed variance:   {smoothed_var:.6f}")
        lines.append(f"    Variance reduced:    {noise_removed:.1f}%")
        lines.append(f"    Raw std dev:         {np.nanstd(raw):.4f}")
        lines.append(f"    Smoothed std dev:    {np.nanstd(smoothed):.4f}")

    lines.append("\n3. INTERPRETATION")
    lines.append("-" * 40)
    lines.append("""
    The Kalman smoother extracts the underlying signal from noisy FSI
    observations. Higher variance reduction indicates more noise was present
    in the raw data. The smoothed series represents the optimal estimate of
    the latent financial stress level given all observations.

    Missing values are imputed using the state-space structure, which
    provides statistically optimal estimates based on surrounding data
    points and the estimated dynamics of the series.
    """)

    lines.append("=" * 70)

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  Saved: {output_path}")

    return report


# =============================================================================
# MAIN (for testing)
# =============================================================================

def main():
    """Test Kalman smoothing on combined FSI data."""
    print("=" * 60)
    print("KALMAN SMOOTHING MODULE TEST")
    print("=" * 60)

    # Load combined FSI data
    fsi_path = OUTPUT_DIR / 'combined_fsi.csv'

    if not fsi_path.exists():
        print(f"\n  ERROR: Combined FSI not found at {fsi_path}")
        print("  Run: python scripts/combined_fsi.py first")
        return

    df = pd.read_csv(fsi_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"\n  Loaded {len(df)} observations")

    # Apply smoothing
    print("\n  Applying Kalman smoothing...")
    smoothed_df = smooth_fsi_dataframe(df)

    # Generate report
    print("\n  Generating report...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    report = generate_smoothing_report(df, smoothed_df,
                                       OUTPUT_DIR / 'kalman_smoothing_report.txt')

    # Generate plots for each FSI
    fsi_cols = [c for c in df.columns if 'fsi' in c.lower() and c != 'date' and '_smoothed' not in c]

    for col in fsi_cols:
        smoothed_col = f"{col}_smoothed"
        if smoothed_col in smoothed_df.columns:
            plot_smoothing_comparison(
                df[col].values,
                smoothed_df[smoothed_col].values,
                pd.to_datetime(df['date']),
                title=f'Kalman Smoothing: {col}',
                save_path=PLOTS_DIR / f'kalman_{col}.png'
            )

    # Save smoothed data
    smoothed_path = OUTPUT_DIR / 'combined_fsi_smoothed.csv'
    smoothed_df.to_csv(smoothed_path, index=False)
    print(f"  Saved: {smoothed_path}")

    print("\n" + "=" * 60)
    print("KALMAN SMOOTHING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
