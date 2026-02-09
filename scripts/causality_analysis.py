#!/usr/bin/env python3
"""
Granger Causality and Impulse Response Function Analysis
=========================================================
Econometric analysis of causal relationships between FSI components
and market indicators (IBOVESPA volatility, Brazil CDS).

Tests:
1. Granger Causality - Does X help predict Y?
2. Impulse Response Functions (IRF) - How does a shock to X affect Y over time?
3. Variance Decomposition - How much of Y's variance is explained by X?

Variables analyzed:
- FSI indices: Dictionary FSI, ML FSI, News FSI, Combined FSI
- Market indicators: IBOVESPA 21d volatility, Brazil 5Y CDS

Mathematical Framework:
    VAR(p) model: Y_t = c + Σ(A_i * Y_{t-i}) + ε_t

    Granger Causality: X Granger-causes Y if past values of X help predict Y
    beyond what past values of Y alone can predict.

    H0: X does not Granger-cause Y (coefficients on lagged X are zero)
    H1: X Granger-causes Y

Usage:
    python scripts/causality_analysis.py
    python scripts/causality_analysis.py --lags 4

Output:
    - output/causality_analysis_report.txt
    - output/plots/granger_causality_heatmap.png
    - output/plots/irf_*.png

References:
    - Granger, C.W.J. (1969): Investigating Causal Relations by Econometric Models
    - Hamilton, J.D. (1994): Time Series Analysis, Chapters 10-11
    - Lütkepohl, H. (2005): New Introduction to Multiple Time Series Analysis

Author: Claude Code for CAEN/UFC Master's Thesis
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Statistical libraries
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Causality analysis will not work.")

# Plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Professional plot configuration
if MATPLOTLIB_AVAILABLE:
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
    'dict_fsi': '#4472C4',
    'ml_fsi': '#ED7D31',
    'news_fsi': '#548235',
    'combined_fsi': '#2C2C2C',
    'volatility': '#7030A0',
    'cds': '#C00000',
}

# Handle paths
try:
    _SCRIPT_DIR = Path(__file__).parent
    _PROJECT_DIR = _SCRIPT_DIR.parent
except NameError:
    _PROJECT_DIR = Path.cwd()
    _SCRIPT_DIR = _PROJECT_DIR / 'scripts'

sys.path.insert(0, str(_PROJECT_DIR))

DATA_DIR = _PROJECT_DIR / 'data' / 'raw'
OUTPUT_DIR = _PROJECT_DIR / 'output'
PLOTS_DIR = OUTPUT_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Analysis start date (standardize to 2008 onwards)
ANALYSIS_START_DATE = pd.Timestamp('2008-01-01')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_combined_fsi() -> Optional[pd.DataFrame]:
    """Load combined FSI data."""
    paths = [
        OUTPUT_DIR / 'combined_fsi.csv',
        OUTPUT_DIR / 'combined_fsi_weighted.csv',
    ]

    for path in paths:
        if path.exists():
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            return df
    return None


def load_ibovespa() -> Optional[pd.DataFrame]:
    """Load IBOVESPA data."""
    path = DATA_DIR / 'ibovespa_data.csv'
    if path.exists():
        df = pd.read_csv(path)
        # Handle mixed timezones by converting to UTC then removing timezone info
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        return df
    return None


def load_brazil_cds() -> Optional[pd.DataFrame]:
    """Load Brazil CDS data (handles Investing.com format)."""
    # Try Investing.com format first
    investing_path = _PROJECT_DIR / 'Brasil CDS 5 Anos USD - Visão Geral.csv'

    if investing_path.exists():
        try:
            df = pd.read_csv(investing_path, encoding='utf-8-sig')

            # Handle column names (Portuguese)
            col_mapping = {
                'Data': 'date',
                'Último': 'cds_5y',
                'Abertura': 'open',
                'Máxima': 'high',
                'Mínima': 'low',
                'Var%': 'change_pct'
            }
            df = df.rename(columns=col_mapping)

            # Parse date (DD.MM.YYYY format)
            df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')

            # Convert values from European format (force string conversion first)
            for col in ['cds_5y', 'open', 'high', 'low']:
                if col in df.columns:
                    # Always convert to string first to ensure proper processing
                    df[col] = df[col].astype(str)
                    df[col] = df[col].str.replace('.', '', regex=False)
                    df[col] = df[col].str.replace(',', '.', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna(subset=['cds_5y'])
            df = df.sort_values('date').reset_index(drop=True)
            return df[['date', 'cds_5y']]

        except Exception as e:
            print(f"  Warning: Could not parse Investing.com CDS file: {e}")

    # Try standard format
    cds_path = DATA_DIR / 'brazil_cds_5y.csv'
    if cds_path.exists():
        df = pd.read_csv(cds_path)
        df['date'] = pd.to_datetime(df['date'])
        # Ensure cds_5y is numeric (force string conversion first)
        df['cds_5y'] = df['cds_5y'].astype(str)
        df['cds_5y'] = df['cds_5y'].str.replace('.', '', regex=False)
        df['cds_5y'] = df['cds_5y'].str.replace(',', '.', regex=False)
        df['cds_5y'] = pd.to_numeric(df['cds_5y'], errors='coerce')
        df = df.dropna(subset=['cds_5y'])
        df = df.sort_values('date').reset_index(drop=True)
        return df

    return None


def prepare_analysis_data(max_lags: int = 8) -> Optional[pd.DataFrame]:
    """Prepare aligned weekly data for causality analysis."""
    print("\nLoading data sources...")

    # Load FSI
    fsi_df = load_combined_fsi()
    if fsi_df is None:
        print("  ERROR: Combined FSI not found. Run combined_fsi.py first.")
        return None
    print(f"  FSI: {len(fsi_df)} observations")

    # Load IBOVESPA
    ibov_df = load_ibovespa()
    if ibov_df is not None:
        print(f"  IBOVESPA: {len(ibov_df)} observations")
        # Aggregate to weekly
        ibov_df['week'] = ibov_df['date'].dt.to_period('W').dt.start_time
        vol_weekly = ibov_df.groupby('week').agg({
            'realized_vol_21d': 'mean',
            'close': 'last'
        }).reset_index()
        vol_weekly.columns = ['date', 'volatility', 'ibov_close']
    else:
        vol_weekly = None
        print("  IBOVESPA: Not found")

    # Load CDS
    cds_df = load_brazil_cds()
    if cds_df is not None:
        print(f"  CDS: {len(cds_df)} observations")
        # Already weekly, just normalize date
        cds_df['date'] = cds_df['date'].dt.to_period('W').dt.start_time
        cds_df = cds_df.groupby('date')['cds_5y'].mean().reset_index()
    else:
        print("  CDS: Not found")

    # Merge all data
    print("\nMerging data sources...")

    # Start with FSI
    fsi_df['date'] = fsi_df['date'].dt.to_period('W').dt.start_time
    merged = fsi_df.groupby('date').first().reset_index()

    # Add volatility
    if vol_weekly is not None:
        merged = merged.merge(vol_weekly, on='date', how='outer')

    # Add CDS
    if cds_df is not None:
        merged = merged.merge(cds_df, on='date', how='outer')

    merged = merged.sort_values('date').reset_index(drop=True)

    # Filter to 2008 onwards
    merged = merged[merged['date'] >= ANALYSIS_START_DATE].copy()
    print(f"  Filtered to 2008+: {len(merged)} observations")

    # Select analysis variables
    analysis_cols = ['date']
    for col in ['dict_fsi', 'ml_fsi', 'news_fsi', 'combined_fsi', 'volatility', 'cds_5y']:
        if col in merged.columns:
            analysis_cols.append(col)

    merged = merged[analysis_cols]

    # Drop rows with all NaN (except date)
    merged = merged.dropna(how='all', subset=[c for c in merged.columns if c != 'date'])

    print(f"  Merged dataset: {len(merged)} observations")
    print(f"  Period: {merged['date'].min().date()} to {merged['date'].max().date()}")
    print(f"  Variables: {[c for c in merged.columns if c != 'date']}")

    return merged


# =============================================================================
# STATIONARITY TESTS
# =============================================================================

def test_stationarity(series: pd.Series, name: str) -> Dict:
    """
    Test stationarity using Augmented Dickey-Fuller test.

    Returns dict with test results.
    """
    clean = series.dropna()
    if len(clean) < 20:
        return {'name': name, 'stationary': None, 'reason': 'insufficient data'}

    try:
        result = adfuller(clean, autolag='AIC')
        return {
            'name': name,
            'adf_stat': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'n_obs': result[3],
            'critical_1%': result[4]['1%'],
            'critical_5%': result[4]['5%'],
            'critical_10%': result[4]['10%'],
            'stationary': result[1] < 0.05,
        }
    except Exception as e:
        return {'name': name, 'stationary': None, 'reason': str(e)}


def stationarity_report(df: pd.DataFrame) -> pd.DataFrame:
    """Test stationarity for all variables."""
    results = []
    for col in df.columns:
        if col != 'date':
            result = test_stationarity(df[col], col)
            results.append(result)
    return pd.DataFrame(results)


# =============================================================================
# GRANGER CAUSALITY
# =============================================================================

def granger_causality_test(data: pd.DataFrame, cause: str, effect: str,
                           max_lag: int = 4) -> Dict:
    """
    Test if 'cause' Granger-causes 'effect'.

    Returns dict with test results for each lag.
    """
    # Prepare data
    test_data = data[[effect, cause]].dropna()

    if len(test_data) < max_lag * 3:
        return {
            'cause': cause,
            'effect': effect,
            'error': 'insufficient data',
            'results': None
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

        # Extract results for each lag
        lag_results = {}
        for lag in range(1, max_lag + 1):
            if lag in results:
                test_result = results[lag][0]
                lag_results[lag] = {
                    'f_stat': test_result['ssr_ftest'][0],
                    'f_pvalue': test_result['ssr_ftest'][1],
                    'chi2_stat': test_result['ssr_chi2test'][0],
                    'chi2_pvalue': test_result['ssr_chi2test'][1],
                }

        # Find minimum p-value across lags
        min_pvalue = min([r['f_pvalue'] for r in lag_results.values()])
        best_lag = [k for k, v in lag_results.items() if v['f_pvalue'] == min_pvalue][0]

        return {
            'cause': cause,
            'effect': effect,
            'min_pvalue': min_pvalue,
            'best_lag': best_lag,
            'significant_5%': min_pvalue < 0.05,
            'significant_1%': min_pvalue < 0.01,
            'lag_results': lag_results,
        }

    except Exception as e:
        return {
            'cause': cause,
            'effect': effect,
            'error': str(e),
            'results': None
        }


def full_granger_analysis(df: pd.DataFrame, variables: List[str],
                          max_lag: int = 4) -> pd.DataFrame:
    """Run Granger causality tests between all pairs of variables."""
    results = []

    for cause in variables:
        for effect in variables:
            if cause != effect:
                result = granger_causality_test(df, cause, effect, max_lag)
                results.append({
                    'Cause': cause,
                    'Effect': effect,
                    'Min_P_Value': result.get('min_pvalue', np.nan),
                    'Best_Lag': result.get('best_lag', np.nan),
                    'Significant_5%': result.get('significant_5%', False),
                    'Significant_1%': result.get('significant_1%', False),
                })

    return pd.DataFrame(results)


# =============================================================================
# VAR MODEL AND IRF
# =============================================================================

def fit_var_model(df: pd.DataFrame, variables: List[str],
                  max_lag: int = 8) -> Tuple:
    """
    Fit Vector Autoregression model.

    Returns (model, results, optimal_lag)
    """
    # Prepare data
    data = df[variables].dropna()

    if len(data) < max_lag * 3:
        return None, None, None

    try:
        # Fit VAR model
        model = VAR(data)

        # Select optimal lag using AIC
        lag_order = model.select_order(max_lag)
        optimal_lag = lag_order.aic

        # Fit with optimal lag
        results = model.fit(optimal_lag)

        return model, results, optimal_lag

    except Exception as e:
        print(f"  VAR model fitting error: {e}")
        return None, None, None


def compute_irf(var_results, periods: int = 20,
                orthogonalized: bool = True) -> Optional[object]:
    """Compute Impulse Response Functions."""
    if var_results is None:
        return None

    try:
        irf = var_results.irf(periods=periods)
        return irf
    except Exception as e:
        print(f"  IRF computation error: {e}")
        return None


def compute_variance_decomposition(var_results, periods: int = 20) -> Optional[object]:
    """Compute Forecast Error Variance Decomposition."""
    if var_results is None:
        return None

    try:
        fevd = var_results.fevd(periods=periods)
        return fevd
    except Exception as e:
        print(f"  FEVD computation error: {e}")
        return None


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_granger_heatmap(granger_df: pd.DataFrame, output_path: Path = None):
    """Plot Granger causality p-values as heatmap."""
    if not MATPLOTLIB_AVAILABLE:
        return

    # Pivot to matrix
    pivot = granger_df.pivot(index='Cause', columns='Effect', values='Min_P_Value')

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', vmin=0, vmax=0.15)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('P-Value (lower = stronger causality)', rotation=-90, va='bottom')

    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Effect (Y)')
    ax.set_ylabel('Cause (X)')
    ax.set_title('Granger Causality Test P-Values\n(X Granger-causes Y)')

    # Add values and significance markers
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.05 else 'black'
                sig = '***' if val < 0.01 else '**' if val < 0.05 else '*' if val < 0.1 else ''
                ax.text(j, i, f'{val:.3f}{sig}', ha='center', va='center',
                       color=color, fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


def plot_irf(irf, var_names: List[str], output_path: Path = None):
    """Plot Impulse Response Functions."""
    if not MATPLOTLIB_AVAILABLE or irf is None:
        return

    n_vars = len(var_names)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(4*n_vars, 3*n_vars))

    for i, response in enumerate(var_names):
        for j, impulse in enumerate(var_names):
            ax = axes[i, j] if n_vars > 1 else axes

            # Get IRF
            irf_values = irf.irfs[:, i, j]

            # Get confidence intervals if available
            try:
                lower = irf.ci[:, i, j, 0]
                upper = irf.ci[:, i, j, 1]
                ax.fill_between(range(len(irf_values)), lower, upper, alpha=0.3)
            except:
                pass

            ax.plot(irf_values, color=COLORS.get(impulse, 'blue'), linewidth=1.5)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

            ax.set_title(f'{impulse} → {response}', fontsize=9)
            if i == n_vars - 1:
                ax.set_xlabel('Weeks')
            if j == 0:
                ax.set_ylabel('Response')

    plt.suptitle('Impulse Response Functions (Orthogonalized)', fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


def plot_single_irf(irf, var_names: List[str], impulse: str,
                    output_path: Path = None):
    """Plot IRF for a single impulse variable."""
    if not MATPLOTLIB_AVAILABLE or irf is None:
        return

    if impulse not in var_names:
        return

    j = var_names.index(impulse)
    n_responses = len(var_names)

    fig, axes = plt.subplots(1, n_responses, figsize=(4*n_responses, 4))

    for i, response in enumerate(var_names):
        ax = axes[i] if n_responses > 1 else axes

        irf_values = irf.irfs[:, i, j]

        # Confidence intervals
        try:
            lower = irf.ci[:, i, j, 0]
            upper = irf.ci[:, i, j, 1]
            ax.fill_between(range(len(irf_values)), lower, upper,
                           alpha=0.3, color=COLORS.get(response, 'blue'))
        except:
            pass

        ax.plot(irf_values, color=COLORS.get(response, 'blue'), linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_title(f'Response: {response}')
        ax.set_xlabel('Weeks')
        ax.set_ylabel('Response')

    impulse_label = impulse.replace('_', ' ').title()
    plt.suptitle(f'Impulse Response to {impulse_label} Shock', fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_causality_report(stationarity_df: pd.DataFrame,
                              granger_df: pd.DataFrame,
                              var_results,
                              optimal_lag: int,
                              output_path: Path = None) -> str:
    """Generate comprehensive causality analysis report."""
    lines = []
    lines.append("=" * 70)
    lines.append("GRANGER CAUSALITY AND IMPULSE RESPONSE ANALYSIS")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # Methodology
    lines.append("\n" + "=" * 70)
    lines.append("1. METHODOLOGY")
    lines.append("=" * 70)
    lines.append("""
    Granger Causality Test:
    -----------------------
    Tests whether past values of X improve predictions of Y beyond
    what past values of Y alone provide.

    H0: X does not Granger-cause Y
    H1: X Granger-causes Y

    Significance levels: * p<0.10, ** p<0.05, *** p<0.01

    Vector Autoregression (VAR):
    ----------------------------
    Y_t = c + Σ(A_i * Y_{t-i}) + ε_t

    Lag selection: Akaike Information Criterion (AIC)

    Impulse Response Functions:
    ---------------------------
    Show how a one-standard-deviation shock to variable X affects
    all variables over time (orthogonalized using Cholesky decomposition).
    """)

    # Stationarity
    lines.append("\n" + "=" * 70)
    lines.append("2. STATIONARITY TESTS (ADF)")
    lines.append("=" * 70)
    lines.append("\n" + stationarity_df.to_string(index=False))

    non_stationary = stationarity_df[stationarity_df['stationary'] == False]['name'].tolist()
    if non_stationary:
        lines.append(f"\n  Warning: Non-stationary series: {non_stationary}")
        lines.append("  Consider differencing or using error-correction models.")

    # Granger Causality Results
    lines.append("\n" + "=" * 70)
    lines.append("3. GRANGER CAUSALITY RESULTS")
    lines.append("=" * 70)
    lines.append("\n" + granger_df.to_string(index=False))

    # Significant relationships
    sig_01 = granger_df[granger_df['Significant_1%'] == True]
    sig_05 = granger_df[(granger_df['Significant_5%'] == True) &
                        (granger_df['Significant_1%'] == False)]

    lines.append("\n\nSignificant Causal Relationships:")
    lines.append("-" * 40)

    if len(sig_01) > 0:
        lines.append("\n  *** Highly Significant (p < 0.01):")
        for _, row in sig_01.iterrows():
            lines.append(f"      {row['Cause']} → {row['Effect']} (p={row['Min_P_Value']:.4f}, lag={row['Best_Lag']})")

    if len(sig_05) > 0:
        lines.append("\n  ** Significant (p < 0.05):")
        for _, row in sig_05.iterrows():
            lines.append(f"      {row['Cause']} → {row['Effect']} (p={row['Min_P_Value']:.4f}, lag={row['Best_Lag']})")

    # VAR Model
    lines.append("\n" + "=" * 70)
    lines.append("4. VAR MODEL SUMMARY")
    lines.append("=" * 70)

    if var_results is not None:
        lines.append(f"\n  Optimal lag (AIC): {optimal_lag}")
        lines.append(f"  Observations: {var_results.nobs}")
        lines.append(f"  AIC: {var_results.aic:.2f}")
        lines.append(f"  BIC: {var_results.bic:.2f}")
    else:
        lines.append("\n  VAR model could not be estimated.")

    # Interpretation
    lines.append("\n" + "=" * 70)
    lines.append("5. INTERPRETATION")
    lines.append("=" * 70)
    lines.append("""
    Key findings from Granger causality analysis:

    1. If FSI Granger-causes market indicators (volatility, CDS), then
       FSI contains predictive information about future market stress.

    2. If market indicators Granger-cause FSI, then markets lead the
       text-based stress measures (possible efficiency concern).

    3. Bidirectional causality suggests feedback loops between text-based
       stress perception and market outcomes.

    4. No causality suggests independence or contemporaneous relationships
       not captured by Granger tests.

    The impulse response functions show how shocks propagate through
    the system over time, providing insights into the dynamics of
    financial stress transmission.
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
    parser = argparse.ArgumentParser(description='Granger Causality and IRF Analysis')
    parser.add_argument('--lags', type=int, default=4,
                        help='Maximum lag for Granger causality tests (default: 4)')
    parser.add_argument('--irf-periods', type=int, default=20,
                        help='Periods for IRF computation (default: 20)')
    args = parser.parse_args()

    print("=" * 70)
    print("GRANGER CAUSALITY AND IMPULSE RESPONSE ANALYSIS")
    print("=" * 70)

    if not STATSMODELS_AVAILABLE:
        print("\n  ERROR: statsmodels required for causality analysis")
        print("  Install with: pip install statsmodels")
        return

    # Load and prepare data
    df = prepare_analysis_data(args.lags)
    if df is None:
        return

    # Get analysis variables
    variables = [c for c in df.columns if c != 'date']
    print(f"\nVariables for analysis: {variables}")

    # Test stationarity
    print("\nTesting stationarity (ADF test)...")
    stationarity_df = stationarity_report(df)
    print(stationarity_df[['name', 'adf_stat', 'p_value', 'stationary']].to_string(index=False))

    # Granger causality tests
    print(f"\nRunning Granger causality tests (max lag = {args.lags})...")
    granger_df = full_granger_analysis(df, variables, args.lags)

    # Print significant results
    sig = granger_df[granger_df['Significant_5%'] == True]
    if len(sig) > 0:
        print("\n  Significant causal relationships (p < 0.05):")
        for _, row in sig.iterrows():
            print(f"    {row['Cause']} → {row['Effect']} (p={row['Min_P_Value']:.4f})")
    else:
        print("\n  No significant causal relationships found at 5% level")

    # Fit VAR model
    print("\nFitting VAR model...")
    model, var_results, optimal_lag = fit_var_model(df, variables, args.lags * 2)

    if var_results is not None:
        print(f"  Optimal lag: {optimal_lag}")
        print(f"  AIC: {var_results.aic:.2f}")

        # Compute IRF
        print("\nComputing Impulse Response Functions...")
        irf = compute_irf(var_results, args.irf_periods)

        # Compute FEVD
        print("Computing Variance Decomposition...")
        fevd = compute_variance_decomposition(var_results, args.irf_periods)
    else:
        irf = None
        fevd = None

    # Generate report
    print("\nGenerating report...")
    report = generate_causality_report(
        stationarity_df, granger_df, var_results, optimal_lag,
        OUTPUT_DIR / 'causality_analysis_report.txt'
    )

    # Generate plots
    print("\nGenerating plots...")

    # Granger causality heatmap
    plot_granger_heatmap(granger_df, PLOTS_DIR / 'granger_causality_heatmap.png')

    # IRF plots
    if irf is not None:
        # Full IRF matrix
        plot_irf(irf, variables, PLOTS_DIR / 'irf_full.png')

        # Individual IRF plots for key impulses
        for impulse in ['volatility', 'cds_5y', 'combined_fsi']:
            if impulse in variables:
                plot_single_irf(irf, variables, impulse,
                               PLOTS_DIR / f'irf_{impulse}.png')

    print("\n" + "=" * 70)
    print("CAUSALITY ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n   Output files:")
    print(f"   - {OUTPUT_DIR}/causality_analysis_report.txt")
    print(f"   - {PLOTS_DIR}/granger_causality_heatmap.png")
    print(f"   - {PLOTS_DIR}/irf_*.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
