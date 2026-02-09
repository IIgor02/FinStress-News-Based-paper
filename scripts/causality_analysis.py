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


def prepare_analysis_data(max_lags: int = 8, frequency: str = 'weekly') -> Optional[pd.DataFrame]:
    """
    Prepare aligned data for causality analysis.

    Parameters
    ----------
    max_lags : int
        Maximum lags for VAR model
    frequency : str
        'weekly' or 'monthly' - monthly reduces noise and improves significance
    """
    print(f"\nLoading data sources (frequency: {frequency})...")

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

    # Determine period for aggregation
    period = 'W' if frequency == 'weekly' else 'M'
    period_name = 'week' if frequency == 'weekly' else 'month'

    # Start with FSI
    fsi_df['period'] = fsi_df['date'].dt.to_period(period).dt.start_time
    merged = fsi_df.groupby('period').mean(numeric_only=True).reset_index()
    merged = merged.rename(columns={'period': 'date'})

    # Add volatility (re-aggregate to match period)
    if vol_weekly is not None:
        vol_weekly['period'] = vol_weekly['date'].dt.to_period(period).dt.start_time
        vol_agg = vol_weekly.groupby('period')['volatility'].mean().reset_index()
        vol_agg = vol_agg.rename(columns={'period': 'date'})
        merged = merged.merge(vol_agg, on='date', how='outer')

    # Add CDS (re-aggregate to match period)
    if cds_df is not None:
        cds_df['period'] = cds_df['date'].dt.to_period(period).dt.start_time
        cds_agg = cds_df.groupby('period')['cds_5y'].mean().reset_index()
        cds_agg = cds_agg.rename(columns={'period': 'date'})
        merged = merged.merge(cds_agg, on='date', how='outer')

    merged = merged.sort_values('date').reset_index(drop=True)

    # Filter to 2008 onwards
    merged = merged[merged['date'] >= ANALYSIS_START_DATE].copy()
    print(f"  Filtered to 2008+ ({frequency}): {len(merged)} observations")

    # Select analysis variables
    # Note: Variable ordering matters for Cholesky identification in VAR/IRF
    # Order: fast-moving (volatility) → risk indicators (CDS) → stress indices
    # Keep all FSIs for Granger tests and bivariate IRF, but use subset for multivariate VAR
    analysis_cols = ['date']
    # All variables including all FSI indices
    all_vars = ['dict_fsi', 'ml_fsi', 'news_fsi', 'combined_fsi', 'volatility', 'cds_5y']
    for col in all_vars:
        if col in merged.columns:
            analysis_cols.append(col)

    merged = merged[analysis_cols]

    # Mark which variables to use for multivariate VAR (to avoid multicollinearity)
    # Order: volatility → cds_5y → combined_fsi (from most exogenous to most endogenous)
    merged.attrs['var_variables'] = ['volatility', 'cds_5y', 'combined_fsi']

    # Drop rows with all NaN (except date)
    merged = merged.dropna(how='all', subset=[c for c in merged.columns if c != 'date'])

    print(f"  Merged dataset: {len(merged)} observations")
    print(f"  Period: {merged['date'].min().date()} to {merged['date'].max().date()}")

    # ==========================================================================
    # CRITICAL: Transform to stationary and standardize
    # ==========================================================================
    print("\nApplying econometric transformations...")

    # Step 1: Transform CDS to log-returns and volatility to log scale
    merged = transform_to_stationary(merged)

    # Step 2: Drop NaN rows created by differencing
    merged = merged.dropna(subset=['cds_5y'])  # cds_5y is now log-return

    # Step 3: Standardize all analysis variables (mean=0, std=1)
    # This ensures IRF shows response to one-standard-deviation shock
    analysis_vars = [c for c in merged.columns if c not in ['date', 'cds_level', 'vol_level', 'cds_log_ret']]
    merged = standardize_variables(merged, analysis_vars)

    print(f"  After transformations: {len(merged)} observations")
    print(f"  Variables: {[c for c in merged.columns if c not in ['cds_level', 'vol_level', 'cds_log_ret']]}")
    print("  Note: CDS transformed to log-returns, volatility to log scale, all standardized")

    return merged


# =============================================================================
# DATA TRANSFORMATION FOR STATIONARITY
# =============================================================================

def transform_to_stationary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform variables to stationary representations following econometric best practices.

    Key insight from investigation:
    - CDS levels are non-stationary (ADF p~0.086) → use log-returns
    - Volatility is borderline → use log transformation for stability
    - FSI indices are already stationary bounded measures

    References:
    - Hamilton (1994): Time Series Analysis - non-stationary variables in VAR
    - Lütkepohl (2005): Variables must be stationary for valid VAR inference
    """
    result = df.copy()

    # Transform CDS to log-returns (percentage change in log scale)
    # This ensures stationarity and makes shocks comparable to FSI
    if 'cds_5y' in result.columns:
        # Log-return: 100 * (log(P_t) - log(P_{t-1}))
        result['cds_log_ret'] = np.log(result['cds_5y']).diff() * 100
        # Keep original for reference but use log-return for analysis
        result['cds_level'] = result['cds_5y']  # Save original
        result['cds_5y'] = result['cds_log_ret']  # Replace with stationary version

    # Transform volatility to log scale (stabilizes variance, improves normality)
    if 'volatility' in result.columns:
        # Small constant to avoid log(0)
        result['vol_level'] = result['volatility']  # Save original
        result['volatility'] = np.log(result['volatility'] + 0.001)

    # FSI indices are already bounded/stationary - no transformation needed
    # (They are based on sentiment proportions or normalized scores)

    return result


def standardize_variables(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """
    Standardize variables to have mean 0 and std 1.

    This ensures IRF magnitudes are comparable across variables and
    represents responses to one-standard-deviation shocks.

    Parameters
    ----------
    df : pd.DataFrame
        Data with variables to standardize
    variables : list
        Column names to standardize

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized variables
    """
    result = df.copy()

    for var in variables:
        if var in result.columns:
            series = result[var].dropna()
            if len(series) > 0 and series.std() > 0:
                result[var] = (result[var] - series.mean()) / series.std()

    return result


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
# LOCAL PROJECTIONS IRF (Jordà 2005) - ROBUST TO PERSISTENCE
# =============================================================================

def local_projection_irf(df: pd.DataFrame, shock_var: str, response_var: str,
                         max_horizon: int = 20, n_lags: int = 4,
                         control_vars: List[str] = None) -> Dict:
    """
    Compute Impulse Response Function using Local Projections (Jordà, 2005).

    Local Projections are more robust than VAR-based IRF because:
    1. They don't require correct specification of the full system
    2. They work well with highly persistent (near unit-root) variables
    3. They directly estimate the response at each horizon

    Model for each horizon h:
        y_{t+h} = α_h + β_h * shock_t + Σ(γ_k * y_{t-k}) + Σ(δ_k * x_{t-k}) + ε_{t+h}

    Parameters
    ----------
    df : pd.DataFrame
        Data with shock and response variables
    shock_var : str
        Variable that receives the shock
    response_var : str
        Variable whose response we measure
    max_horizon : int
        Maximum horizon for IRF (default: 20 periods)
    n_lags : int
        Number of lags to include as controls
    control_vars : list
        Additional control variables

    Returns
    -------
    dict with keys: 'horizons', 'irf', 'lower', 'upper', 'pvalues'

    References
    ----------
    Jordà, Ò. (2005). Estimation and Inference of Impulse Responses by Local Projections.
    American Economic Review, 95(1), 161-182.
    """
    # Prepare data - use LEVELS (not log-returns) for persistent variables
    data = df[[shock_var, response_var]].copy()

    if control_vars:
        for cv in control_vars:
            if cv in df.columns and cv not in [shock_var, response_var]:
                data[cv] = df[cv]

    data = data.dropna()

    if len(data) < max_horizon + n_lags + 10:
        return None

    # Standardize for comparability
    for col in data.columns:
        data[col] = (data[col] - data[col].mean()) / data[col].std()

    # Results storage
    horizons = list(range(max_horizon + 1))
    irf_values = []
    lower_ci = []
    upper_ci = []
    pvalues = []

    for h in horizons:
        # Dependent variable: response at t+h
        y = data[response_var].shift(-h)

        # Regressors: shock at t, plus lagged controls
        X_cols = [shock_var]

        # Add lagged values of response and shock as controls
        for lag in range(1, n_lags + 1):
            lag_col_y = f'{response_var}_lag{lag}'
            lag_col_x = f'{shock_var}_lag{lag}'
            data[lag_col_y] = data[response_var].shift(lag)
            data[lag_col_x] = data[shock_var].shift(lag)
            X_cols.extend([lag_col_y, lag_col_x])

        # Add control variables and their lags
        if control_vars:
            for cv in control_vars:
                if cv in data.columns:
                    X_cols.append(cv)
                    for lag in range(1, n_lags + 1):
                        lag_col = f'{cv}_lag{lag}'
                        data[lag_col] = data[cv].shift(lag)
                        X_cols.append(lag_col)

        # Remove duplicates
        X_cols = list(dict.fromkeys(X_cols))

        # Build design matrix
        X = data[X_cols].copy()
        X = sm.add_constant(X)

        # Align and drop NaN
        valid_idx = ~(y.isna() | X.isna().any(axis=1))
        y_clean = y[valid_idx]
        X_clean = X[valid_idx]

        if len(y_clean) < len(X_cols) + 5:
            irf_values.append(np.nan)
            lower_ci.append(np.nan)
            upper_ci.append(np.nan)
            pvalues.append(np.nan)
            continue

        try:
            # OLS with HAC standard errors (Newey-West) for autocorrelation
            model = sm.OLS(y_clean, X_clean)
            # Use HAC standard errors with bandwidth = h+1 for horizon h
            results = model.fit(cov_type='HAC', cov_kwds={'maxlags': max(1, h)})

            # Extract coefficient on shock variable
            coef = results.params[shock_var]
            se = results.bse[shock_var]
            pval = results.pvalues[shock_var]

            # 95% confidence interval
            irf_values.append(coef)
            lower_ci.append(coef - 1.96 * se)
            upper_ci.append(coef + 1.96 * se)
            pvalues.append(pval)

        except Exception:
            irf_values.append(np.nan)
            lower_ci.append(np.nan)
            upper_ci.append(np.nan)
            pvalues.append(np.nan)

    return {
        'horizons': np.array(horizons),
        'irf': np.array(irf_values),
        'lower': np.array(lower_ci),
        'upper': np.array(upper_ci),
        'pvalues': np.array(pvalues),
        'shock_var': shock_var,
        'response_var': response_var,
    }


def plot_local_projection_irf(df: pd.DataFrame, output_path: Path = None):
    """
    Create comprehensive IRF plots using Local Projections method.

    This is more robust than VAR-based IRF for persistent financial variables.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    # Define shock-response pairs to analyze
    fsi_vars = ['dict_fsi', 'ml_fsi', 'combined_fsi']
    market_vars = ['volatility', 'cds_5y']

    # Filter to available variables - use original levels
    available_fsi = [f for f in fsi_vars if f in df.columns]
    available_market = [m for m in market_vars if m in df.columns]

    if not available_fsi or not available_market:
        print("  Insufficient data for LP-IRF analysis")
        return

    # We need to use LEVEL data, not transformed data
    # Reload original data for LP analysis
    fsi_df = pd.read_csv(OUTPUT_DIR / 'combined_fsi.csv')
    fsi_df['date'] = pd.to_datetime(fsi_df['date'])
    fsi_df['week'] = fsi_df['date'].dt.to_period('W').dt.start_time
    fsi_weekly = fsi_df.groupby('week').mean(numeric_only=True).reset_index()
    fsi_weekly = fsi_weekly.rename(columns={'week': 'date'})

    # Load CDS levels
    cds_path = _PROJECT_DIR / 'Brasil CDS 5 Anos USD - Visão Geral.csv'
    if cds_path.exists():
        cds_df = pd.read_csv(cds_path, encoding='utf-8-sig')
        cds_df = cds_df.rename(columns={'Data': 'date', 'Último': 'cds_5y'})
        cds_df['date'] = pd.to_datetime(cds_df['date'], format='%d.%m.%Y')
        cds_df['cds_5y'] = cds_df['cds_5y'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        cds_df['cds_5y'] = pd.to_numeric(cds_df['cds_5y'], errors='coerce')
        cds_df['week'] = cds_df['date'].dt.to_period('W').dt.start_time
        cds_weekly = cds_df.groupby('week')['cds_5y'].mean().reset_index()
        cds_weekly = cds_weekly.rename(columns={'week': 'date'})
    else:
        cds_weekly = None

    # Load volatility levels
    ibov_path = DATA_DIR / 'ibovespa_data.csv'
    if ibov_path.exists():
        ibov_df = pd.read_csv(ibov_path)
        ibov_df['date'] = pd.to_datetime(ibov_df['date'], utc=True).dt.tz_localize(None)
        ibov_df['week'] = ibov_df['date'].dt.to_period('W').dt.start_time
        vol_weekly = ibov_df.groupby('week')['realized_vol_21d'].mean().reset_index()
        vol_weekly = vol_weekly.rename(columns={'week': 'date', 'realized_vol_21d': 'volatility'})
    else:
        vol_weekly = None

    # Merge level data
    level_data = fsi_weekly.copy()
    if vol_weekly is not None:
        level_data = level_data.merge(vol_weekly, on='date', how='outer')
    if cds_weekly is not None:
        level_data = level_data.merge(cds_weekly, on='date', how='outer')

    level_data = level_data.sort_values('date')
    level_data = level_data[level_data['date'] >= ANALYSIS_START_DATE].dropna()

    print(f"  LP-IRF using {len(level_data)} observations (level data)")

    # Update available variables based on level_data
    available_fsi = [f for f in fsi_vars if f in level_data.columns]
    available_market = [m for m in ['volatility', 'cds_5y'] if m in level_data.columns]

    n_fsi = len(available_fsi)
    n_market = len(available_market)

    # Create figure: FSI as leading indicator (FSI shock → Market response)
    fig, axes = plt.subplots(n_market, n_fsi, figsize=(4.5*n_fsi, 4*n_market))

    max_horizon = 20

    for i, response in enumerate(available_market):
        for j, shock in enumerate(available_fsi):
            ax = axes[i, j] if n_market > 1 and n_fsi > 1 else (axes[j] if n_market == 1 else axes[i])

            # Compute LP-IRF
            result = local_projection_irf(level_data, shock, response,
                                          max_horizon=max_horizon, n_lags=4)

            if result is not None and not np.all(np.isnan(result['irf'])):
                h = result['horizons']
                irf = result['irf']
                lower = result['lower']
                upper = result['upper']
                pvals = result['pvalues']

                # Plot confidence interval
                ax.fill_between(h, lower, upper, alpha=0.2, color='gray')

                # Plot IRF line
                ax.plot(h, irf, color='black', linewidth=1.5)

                # Add significance markers
                sig_5 = pvals < 0.05
                sig_10 = (pvals < 0.10) & ~sig_5
                ax.scatter(h[sig_5], irf[sig_5], color='black', s=20, zorder=5)

                # Check if significant at any horizon in first 8 weeks
                early_sig = np.any(pvals[:8] < 0.05)
                sig_marker = '*' if early_sig else ''
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                sig_marker = ''

            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

            # Labels
            shock_label = shock.replace('_', ' ').replace('fsi', 'FSI').title()
            response_label = response.replace('_', ' ').replace('5y', '5Y').title()
            ax.set_title(f'{shock_label} → {response_label}{sig_marker}', fontsize=10)

            if i == n_market - 1:
                ax.set_xlabel('Weeks')
            if j == 0:
                ax.set_ylabel('Response (std. dev.)')

    plt.suptitle('Local Projections IRF: Market Response to FSI Shocks\n(Robust method for persistent variables; * significant at 5%)',
                 fontsize=11)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


def plot_lp_irf_reverse(df: pd.DataFrame, output_path: Path = None):
    """
    Create LP-IRF plots for reverse direction: Market shock → FSI response.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    fsi_vars = ['dict_fsi', 'ml_fsi', 'combined_fsi']
    market_vars = ['volatility', 'cds_5y']

    # Reload level data (same as above)
    fsi_df = pd.read_csv(OUTPUT_DIR / 'combined_fsi.csv')
    fsi_df['date'] = pd.to_datetime(fsi_df['date'])
    fsi_df['week'] = fsi_df['date'].dt.to_period('W').dt.start_time
    fsi_weekly = fsi_df.groupby('week').mean(numeric_only=True).reset_index()
    fsi_weekly = fsi_weekly.rename(columns={'week': 'date'})

    cds_path = _PROJECT_DIR / 'Brasil CDS 5 Anos USD - Visão Geral.csv'
    if cds_path.exists():
        cds_df = pd.read_csv(cds_path, encoding='utf-8-sig')
        cds_df = cds_df.rename(columns={'Data': 'date', 'Último': 'cds_5y'})
        cds_df['date'] = pd.to_datetime(cds_df['date'], format='%d.%m.%Y')
        cds_df['cds_5y'] = cds_df['cds_5y'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        cds_df['cds_5y'] = pd.to_numeric(cds_df['cds_5y'], errors='coerce')
        cds_df['week'] = cds_df['date'].dt.to_period('W').dt.start_time
        cds_weekly = cds_df.groupby('week')['cds_5y'].mean().reset_index()
        cds_weekly = cds_weekly.rename(columns={'week': 'date'})
    else:
        cds_weekly = None

    ibov_path = DATA_DIR / 'ibovespa_data.csv'
    if ibov_path.exists():
        ibov_df = pd.read_csv(ibov_path)
        ibov_df['date'] = pd.to_datetime(ibov_df['date'], utc=True).dt.tz_localize(None)
        ibov_df['week'] = ibov_df['date'].dt.to_period('W').dt.start_time
        vol_weekly = ibov_df.groupby('week')['realized_vol_21d'].mean().reset_index()
        vol_weekly = vol_weekly.rename(columns={'week': 'date', 'realized_vol_21d': 'volatility'})
    else:
        vol_weekly = None

    level_data = fsi_weekly.copy()
    if vol_weekly is not None:
        level_data = level_data.merge(vol_weekly, on='date', how='outer')
    if cds_weekly is not None:
        level_data = level_data.merge(cds_weekly, on='date', how='outer')

    level_data = level_data.sort_values('date')
    level_data = level_data[level_data['date'] >= ANALYSIS_START_DATE].dropna()

    available_fsi = [f for f in fsi_vars if f in level_data.columns]
    available_market = [m for m in ['volatility', 'cds_5y'] if m in level_data.columns]

    n_fsi = len(available_fsi)
    n_market = len(available_market)

    # Create figure: Market shock → FSI response
    fig, axes = plt.subplots(n_market, n_fsi, figsize=(4.5*n_fsi, 4*n_market))

    max_horizon = 20

    for i, shock in enumerate(available_market):
        for j, response in enumerate(available_fsi):
            ax = axes[i, j] if n_market > 1 and n_fsi > 1 else (axes[j] if n_market == 1 else axes[i])

            # Compute LP-IRF
            result = local_projection_irf(level_data, shock, response,
                                          max_horizon=max_horizon, n_lags=4)

            if result is not None and not np.all(np.isnan(result['irf'])):
                h = result['horizons']
                irf = result['irf']
                lower = result['lower']
                upper = result['upper']
                pvals = result['pvalues']

                ax.fill_between(h, lower, upper, alpha=0.2, color='gray')
                ax.plot(h, irf, color='black', linewidth=1.5)

                sig_5 = pvals < 0.05
                ax.scatter(h[sig_5], irf[sig_5], color='black', s=20, zorder=5)

                early_sig = np.any(pvals[:8] < 0.05)
                sig_marker = '*' if early_sig else ''
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                sig_marker = ''

            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

            shock_label = shock.replace('_', ' ').replace('5y', '5Y').title()
            response_label = response.replace('_', ' ').replace('fsi', 'FSI').title()
            ax.set_title(f'{shock_label} → {response_label}{sig_marker}', fontsize=10)

            if i == n_market - 1:
                ax.set_xlabel('Weeks')
            if j == 0:
                ax.set_ylabel('Response (std. dev.)')

    plt.suptitle('Local Projections IRF: FSI Response to Market Shocks\n(Robust method for persistent variables; * significant at 5%)',
                 fontsize=11)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


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


def plot_irf(irf, var_names: List[str], output_path: Path = None, sigs: int = 2):
    """Plot Impulse Response Functions with confidence intervals."""
    if not MATPLOTLIB_AVAILABLE or irf is None:
        return

    n_vars = len(var_names)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(4*n_vars, 3*n_vars))

    periods = len(irf.irfs)

    for i, response in enumerate(var_names):
        for j, impulse in enumerate(var_names):
            ax = axes[i, j] if n_vars > 1 else axes

            # Get IRF
            irf_values = irf.irfs[:, i, j]

            # Get confidence intervals
            try:
                if hasattr(irf, 'ci') and irf.ci is not None:
                    lower = irf.ci[:, i, j, 0]
                    upper = irf.ci[:, i, j, 1]
                else:
                    stderr = irf.stderr()
                    if stderr is not None:
                        std_err = stderr[:, i, j]
                        lower = irf_values - sigs * std_err
                        upper = irf_values + sigs * std_err
                    else:
                        lower = upper = None

                if lower is not None and upper is not None:
                    ax.fill_between(range(periods), lower, upper, alpha=0.2, color='gray')
            except:
                pass

            ax.plot(irf_values, color='black', linewidth=1.5)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

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
                    output_path: Path = None, sigs: int = 2):
    """Plot IRF for a single impulse variable with confidence intervals."""
    if not MATPLOTLIB_AVAILABLE or irf is None:
        return

    if impulse not in var_names:
        return

    j = var_names.index(impulse)
    n_responses = len(var_names)

    fig, axes = plt.subplots(1, n_responses, figsize=(4*n_responses, 4))

    periods = len(irf.irfs)

    for i, response in enumerate(var_names):
        ax = axes[i] if n_responses > 1 else axes

        irf_values = irf.irfs[:, i, j]

        # Compute bootstrap confidence intervals if not available
        try:
            # Try to get pre-computed confidence intervals
            if hasattr(irf, 'ci') and irf.ci is not None:
                lower = irf.ci[:, i, j, 0]
                upper = irf.ci[:, i, j, 1]
            else:
                # Use asymptotic standard errors
                stderr = irf.stderr()
                if stderr is not None:
                    std_err = stderr[:, i, j]
                    lower = irf_values - sigs * std_err
                    upper = irf_values + sigs * std_err
                else:
                    lower = upper = None

            if lower is not None and upper is not None:
                ax.fill_between(range(periods), lower, upper,
                               alpha=0.2, color='gray', label=f'{sigs}σ CI')
        except Exception:
            pass

        # Plot IRF line in black
        ax.plot(irf_values, color='black', linewidth=1.5)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
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


def run_bivariate_irf(df: pd.DataFrame, shock_var: str, response_var: str,
                      periods: int = 20, max_lag: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Run bivariate VAR and compute IRF for a single shock-response pair.

    Returns (irf_values, lower_ci, upper_ci, n_periods) or (None, None, None, 0) if failed.
    """
    # Order: shock variable first (more exogenous), response second
    var_data = df[[shock_var, response_var]].dropna()

    if len(var_data) < max_lag * 3:
        return None, None, None, 0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VAR(var_data)
            # Select optimal lag
            results = model.fit(maxlags=max_lag, ic='aic')

            # Compute IRF
            irf = results.irf(periods=periods)

            # Get IRF values (response of second variable to shock in first)
            irf_values = irf.irfs[:, 1, 0]  # response_var response to shock_var shock
            n_periods = len(irf_values)

            # Get standard errors for confidence intervals
            try:
                stderr = irf.stderr()
                if stderr is not None:
                    std_err = stderr[:, 1, 0]
                    lower = irf_values - 2 * std_err
                    upper = irf_values + 2 * std_err
                else:
                    lower = upper = None
            except:
                lower = upper = None

            return irf_values, lower, upper, n_periods

    except Exception as e:
        return None, None, None, 0


def plot_fsi_shock_responses(df: pd.DataFrame, output_path: Path = None):
    """
    Create comprehensive IRF plot showing how each FSI responds to volatility and CDS shocks.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    fsi_indices = ['dict_fsi', 'ml_fsi', 'news_fsi', 'combined_fsi']
    shock_vars = ['volatility', 'cds_5y']

    # Filter to available FSIs
    available_fsis = [f for f in fsi_indices if f in df.columns]
    available_shocks = [s for s in shock_vars if s in df.columns]

    if not available_fsis or not available_shocks:
        print("  Insufficient data for FSI shock response analysis")
        return

    n_fsis = len(available_fsis)
    n_shocks = len(available_shocks)

    fig, axes = plt.subplots(n_shocks, n_fsis, figsize=(4*n_fsis, 4*n_shocks))

    periods = 20

    for i, shock in enumerate(available_shocks):
        for j, fsi in enumerate(available_fsis):
            ax = axes[i, j] if n_shocks > 1 and n_fsis > 1 else (axes[j] if n_shocks == 1 else axes[i])

            # Run bivariate IRF
            irf_values, lower, upper, n_periods = run_bivariate_irf(df, shock, fsi, periods=periods)

            if irf_values is not None and n_periods > 0:
                # Plot confidence interval
                if lower is not None and upper is not None:
                    ax.fill_between(range(n_periods), lower, upper, alpha=0.2, color='gray')

                # Plot IRF line
                ax.plot(range(n_periods), irf_values, color='black', linewidth=1.5)

                # Check significance (CI doesn't cross zero)
                if lower is not None and upper is not None:
                    significant = not any((lower[k] <= 0 <= upper[k]) for k in range(1, min(8, n_periods)))
                    sig_marker = '*' if significant else ''
                else:
                    sig_marker = ''
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                sig_marker = ''

            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

            # Labels
            fsi_label = fsi.replace('_', ' ').replace('fsi', 'FSI').title()
            shock_label = shock.replace('_', ' ').replace('5y', '5Y').title()
            ax.set_title(f'{shock_label} → {fsi_label}{sig_marker}', fontsize=10)

            if i == n_shocks - 1:
                ax.set_xlabel('Weeks')
            if j == 0:
                ax.set_ylabel('Response')

    plt.suptitle('FSI Responses to Market Shocks\n(* indicates significant at 95% CI for first 8 weeks)', fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close()


def plot_fsi_as_leading_indicator(df: pd.DataFrame, output_path: Path = None):
    """
    Create IRF plot showing FSI as a leading indicator - how CDS and volatility
    respond to FSI shocks. This reflects the finding that FSI leads market stress.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    fsi_indices = ['dict_fsi', 'ml_fsi', 'news_fsi', 'combined_fsi']
    response_vars = ['volatility', 'cds_5y']

    # Filter to available FSIs
    available_fsis = [f for f in fsi_indices if f in df.columns]
    available_responses = [s for s in response_vars if s in df.columns]

    if not available_fsis or not available_responses:
        print("  Insufficient data for FSI leading indicator analysis")
        return

    n_fsis = len(available_fsis)
    n_responses = len(available_responses)

    fig, axes = plt.subplots(n_responses, n_fsis, figsize=(4*n_fsis, 4*n_responses))

    periods = 20

    for i, response in enumerate(available_responses):
        for j, fsi in enumerate(available_fsis):
            ax = axes[i, j] if n_responses > 1 and n_fsis > 1 else (axes[j] if n_responses == 1 else axes[i])

            # Run bivariate IRF with FSI as shock (first) and market var as response (second)
            irf_values, lower, upper, n_periods = run_bivariate_irf(df, fsi, response, periods=periods)

            if irf_values is not None and n_periods > 0:
                # Plot confidence interval
                if lower is not None and upper is not None:
                    ax.fill_between(range(n_periods), lower, upper, alpha=0.2, color='gray')

                # Plot IRF line
                ax.plot(range(n_periods), irf_values, color='black', linewidth=1.5)

                # Check significance
                if lower is not None and upper is not None:
                    significant = not any((lower[k] <= 0 <= upper[k]) for k in range(1, min(8, n_periods)))
                    sig_marker = '*' if significant else ''
                else:
                    sig_marker = ''
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                sig_marker = ''

            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

            # Labels
            fsi_label = fsi.replace('_', ' ').replace('fsi', 'FSI').title()
            response_label = response.replace('_', ' ').replace('5y', '5Y').title()
            ax.set_title(f'{fsi_label} → {response_label}{sig_marker}', fontsize=10)

            if i == n_responses - 1:
                ax.set_xlabel('Weeks')
            if j == 0:
                ax.set_ylabel('Response')

    plt.suptitle('FSI as Leading Indicator: Market Response to FSI Shocks\n(* indicates significant at 95% CI for first 8 weeks)', fontsize=12)
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
    Data Transformations (Following Econometric Best Practices):
    ------------------------------------------------------------
    - CDS: Transformed to log-returns (ensures stationarity)
      Original CDS levels are non-stationary (ADF p~0.09), which can
      produce spurious results. Log-returns are stationary.

    - Volatility: Transformed to log scale (stabilizes variance)

    - FSI indices: No transformation (already bounded/stationary)

    - All variables: Standardized (mean=0, std=1) so IRF shows
      response to one-standard-deviation shock

    References: Hamilton (1994), Lütkepohl (2005)

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
    parser.add_argument('--frequency', type=str, default='weekly', choices=['weekly', 'monthly'],
                        help='Data frequency: weekly or monthly (monthly reduces noise)')
    args = parser.parse_args()

    print("=" * 70)
    print("GRANGER CAUSALITY AND IMPULSE RESPONSE ANALYSIS")
    print("=" * 70)

    if not STATSMODELS_AVAILABLE:
        print("\n  ERROR: statsmodels required for causality analysis")
        print("  Install with: pip install statsmodels")
        return

    # Load and prepare data
    df = prepare_analysis_data(args.lags, frequency=args.frequency)
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

    # Get VAR variables (subset to avoid multicollinearity)
    var_variables = df.attrs.get('var_variables', variables)
    var_variables = [v for v in var_variables if v in df.columns]
    print(f"\nVAR analysis variables: {var_variables}")

    # Fit VAR model
    print("\nFitting VAR model...")
    model, var_results, optimal_lag = fit_var_model(df, var_variables, args.lags * 2)

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
        plot_irf(irf, var_variables, PLOTS_DIR / 'irf_full.png')

        # Individual IRF plots for key impulses
        for impulse in ['volatility', 'cds_5y', 'combined_fsi']:
            if impulse in var_variables:
                plot_single_irf(irf, var_variables, impulse,
                               PLOTS_DIR / f'irf_{impulse}.png')

    # Bivariate IRF analysis for each FSI index
    print("\nComputing bivariate IRFs for each FSI index...")
    plot_fsi_shock_responses(df, PLOTS_DIR / 'irf_fsi_responses.png')

    # FSI as leading indicator analysis
    print("\nComputing FSI as leading indicator IRFs...")
    plot_fsi_as_leading_indicator(df, PLOTS_DIR / 'irf_fsi_leading.png')

    # =========================================================================
    # LOCAL PROJECTIONS IRF (Jordà 2005) - More robust for persistent variables
    # =========================================================================
    print("\n" + "=" * 70)
    print("LOCAL PROJECTIONS IRF (Jordà 2005)")
    print("=" * 70)
    print("  Note: LP-IRF is more robust than VAR-IRF for persistent financial variables")
    print("  Using LEVEL data (not log-returns) to capture economic relationships")

    print("\nComputing LP-IRF: FSI → Market (FSI as leading indicator)...")
    plot_local_projection_irf(df, PLOTS_DIR / 'lp_irf_fsi_to_market.png')

    print("\nComputing LP-IRF: Market → FSI (Market feedback to FSI)...")
    plot_lp_irf_reverse(df, PLOTS_DIR / 'lp_irf_market_to_fsi.png')

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
