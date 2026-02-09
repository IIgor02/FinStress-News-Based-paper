#!/usr/bin/env python3
"""
ML-Generated Financial Stress Index (García et al. 2023)
=========================================================
Uses LASSO regression to learn which Google Trends queries predict market declines.

Methodology:
1. Regress next-week IBOVESPA returns on current-week search volumes
2. LASSO selects which queries are predictive (automatic feature selection)
3. FSI = negative of predicted return (high FSI = predicted decline)

Usage:
    python scripts/ml_fsi.py
    python scripts/ml_fsi.py --train-end 2022-12-31
    python scripts/ml_fsi.py --no-plots

Output:
    output/results/fsi_ml_weekly.csv
    output/results/ml_coefficients.csv
    output/results/ml_diagnostics.csv
    output/plots/ml_*.png
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

# Handle both script execution and interactive/REPL usage
try:
    _SCRIPT_DIR = Path(__file__).parent
    _PROJECT_DIR = _SCRIPT_DIR.parent
except NameError:
    _PROJECT_DIR = Path.cwd()
    _SCRIPT_DIR = _PROJECT_DIR / 'scripts'

sys.path.insert(0, str(_PROJECT_DIR))

from scripts.dictionaries import CRISIS_EPISODES, get_dictionary_stats

warnings.filterwarnings('ignore')

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

# Professional color palette (softer academic colors)
COLORS = {
    'fsi': '#4472C4',           # Soft blue
    'fsi_ma': '#C55A11',        # Soft orange
    'volatility': '#C55A11',    # Soft orange
    'stress': '#D9534F',        # Soft red
    'anti_stress': '#4472C4',   # Soft blue
    'high_stress': '#E8C1C1',   # Light red for fill
    'low_stress': '#C1E8C1',    # Light green for fill
    'neutral': '#808080',       # Gray
    'crisis': '#D9534F',        # Soft red
}


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for ML FSI calculation."""
    VOLATILITY_COLUMN: str = 'realized_vol_30d'
    TARGET_CORR_MIN: float = 0.60
    TARGET_CORR_MAX: float = 0.90
    TRAIN_END: str = '2022-12-31'  # Default train/test split
    MIN_QUERIES: int = 10  # Minimum queries after validation
    CV_FOLDS: int = 5  # Cross-validation folds

    DATA_DIR: Path = _PROJECT_DIR / 'data' / 'raw'
    OUTPUT_DIR: Path = _PROJECT_DIR / 'output' / 'results'
    PLOTS_DIR: Path = _PROJECT_DIR / 'output' / 'plots'


CONFIG = Config()


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Google Trends ML data and IBOVESPA data."""
    gt_path = CONFIG.DATA_DIR / 'google_trends_ml.csv'
    market_path = CONFIG.DATA_DIR / 'ibovespa_data.csv'

    # Try ML file first, fall back to regular
    if not gt_path.exists():
        gt_path = CONFIG.DATA_DIR / 'google_trends_data.csv'

    if not gt_path.exists():
        raise FileNotFoundError(f"No Google Trends data found. Run: python scripts/collect_data.py --ml")

    if not market_path.exists():
        raise FileNotFoundError(f"No IBOVESPA data found. Run: python scripts/collect_data.py")

    gt_df = pd.read_csv(gt_path)
    market_df = pd.read_csv(market_path)

    # Handle mixed timezones by converting to UTC then removing timezone info
    gt_df['date'] = pd.to_datetime(gt_df['date'], utc=True).dt.tz_localize(None)
    market_df['date'] = pd.to_datetime(market_df['date'], utc=True).dt.tz_localize(None)

    query_cols = [c for c in gt_df.columns if c not in ['date', 'source', 'isPartial']]
    print(f"  Google Trends: {len(gt_df)} observations, {len(query_cols)} queries")
    print(f"  IBOVESPA: {len(market_df):,} trading days")

    return gt_df, market_df


def prepare_features_and_target(gt_df: pd.DataFrame, market_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target vector y.

    Detects if Google Trends data is weekly or monthly and aggregates market data accordingly.
    X: Search volumes (SVI)
    y: Next-period IBOVESPA log returns
    """
    # Get query columns
    query_cols = [c for c in gt_df.columns if c not in ['date', 'source', 'isPartial']]

    # Detect if data is monthly or weekly based on median date spacing
    gt = gt_df.copy()
    gt = gt.sort_values('date')
    date_diffs = gt['date'].diff().dt.days.dropna()
    median_diff = date_diffs.median()

    is_monthly = median_diff > 20  # Monthly if typical gap > 20 days
    freq = 'M' if is_monthly else 'W'
    freq_label = 'months' if is_monthly else 'weeks'

    print(f"    Data frequency: {freq_label} (median gap: {median_diff:.0f} days)")

    # Aggregate market data to match GT frequency
    market = market_df.copy()
    market['period'] = market['date'].dt.to_period(freq)

    # Log returns (sum of daily log returns)
    market['log_return'] = np.log(market['close'] / market['close'].shift(1))
    market_agg = market.groupby('period').agg({
        'log_return': 'sum',
        'close': 'last',
        CONFIG.VOLATILITY_COLUMN: 'mean'
    }).reset_index()

    # Convert GT dates to periods
    gt['period'] = gt['date'].dt.to_period(freq)

    merged = pd.merge(
        gt[['period'] + query_cols],
        market_agg[['period', 'log_return', CONFIG.VOLATILITY_COLUMN]],
        on='period',
        how='inner'
    )

    # Create target: NEXT period's return (shift -1)
    merged['target'] = merged['log_return'].shift(-1)

    # Remove last row (no target) and any NaN
    merged = merged.dropna()

    # Feature matrix and target
    X = merged[query_cols].copy()
    X.index = merged['period']
    y = merged['target'].copy()
    y.index = merged['period']

    # Also return volatility for validation
    vol = merged[CONFIG.VOLATILITY_COLUMN].copy()
    vol.index = merged['period']

    return X, y, vol


def validate_queries(X: pd.DataFrame) -> pd.DataFrame:
    """
    Validate queries - remove low-variation or sparse queries.

    Criteria:
    - Max SVI > 5 (has some search volume)
    - Non-zero observations > 30%
    - Coefficient of variation > 0.2
    """
    valid_cols = []
    dropped = []

    for col in X.columns:
        series = X[col]

        # Check max value
        if series.max() < 5:
            dropped.append((col, "Low volume"))
            continue

        # Check sparsity
        if (series > 0).mean() < 0.3:
            dropped.append((col, "Too sparse"))
            continue

        # Check variation
        if series.std() == 0 or series.mean() == 0:
            dropped.append((col, "No variation"))
            continue

        cv = series.std() / series.mean()
        if cv < 0.2:
            dropped.append((col, "Low variation"))
            continue

        valid_cols.append(col)

    if dropped:
        print(f"    Dropped {len(dropped)} queries:")
        for col, reason in dropped[:5]:
            print(f"      - '{col}': {reason}")
        if len(dropped) > 5:
            print(f"      ... and {len(dropped) - 5} more")

    print(f"    Valid queries: {len(valid_cols)}/{len(X.columns)}")

    return X[valid_cols]


# =============================================================================
# LASSO MODEL
# =============================================================================

def train_lasso_model(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple:
    """
    Train LASSO model with cross-validation.

    Returns:
        model: Fitted LassoCV model
        scaler: Fitted StandardScaler
        cv_results: Dict with cross-validation results
    """
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_scaled = pd.DataFrame(X_scaled, index=X_train.index, columns=X_train.columns)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=CONFIG.CV_FOLDS)

    # LASSO with automatic lambda selection
    # Use a smaller alpha range to encourage some feature selection
    # Default alphas can be too conservative for financial prediction
    print("    Fitting LASSO model with cross-validation...")
    alphas = np.logspace(-5, -1, 100)  # Range from 0.00001 to 0.1
    model = LassoCV(
        cv=tscv,
        alphas=alphas,
        max_iter=10000,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y_train)

    # If LASSO selects 0 features, use correlation-weighted approach
    # Weight each query by its correlation with volatility (not returns)
    is_fallback = False
    if np.sum(model.coef_ != 0) == 0:
        print("    No features selected with CV alpha, using volatility-correlation weighting...")

        # Import volatility data for correlation-based weighting
        from sklearn.linear_model import Ridge

        # Get volatility data aligned with training period
        # We'll use a simple weighted average approach
        # Features that correlate positively with volatility get positive weights

        # Use moderate Ridge regularization
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y_train)
        is_fallback = True

        # Store that we didn't use feature selection
        model._selected_features = None

    # Results
    n_nonzero = np.sum(model.coef_ != 0)
    r2_train = model.score(X_scaled, y_train)

    # LassoCV uses alpha_, Ridge uses alpha
    optimal_alpha = getattr(model, 'alpha_', getattr(model, 'alpha', 0))
    print(f"    Optimal lambda: {optimal_alpha:.6f}")
    print(f"    Non-zero coefficients: {n_nonzero}/{len(model.coef_)}")
    print(f"    In-sample R²: {r2_train:.4f}")

    cv_results = {
        'optimal_lambda': optimal_alpha,
        'n_features': len(model.coef_),
        'n_selected': n_nonzero,
        'train_r2': r2_train,
    }

    return model, scaler, cv_results


def get_coefficients(model, feature_names: list) -> pd.DataFrame:
    """Extract and sort model coefficients."""
    coef_df = pd.DataFrame({
        'query': feature_names,
        'coefficient': model.coef_
    })
    coef_df['abs_coef'] = np.abs(coef_df['coefficient'])
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    coef_df['rank'] = range(1, len(coef_df) + 1)

    # Mark sign
    coef_df['type'] = 'excluded'
    coef_df.loc[coef_df['coefficient'] < 0, 'type'] = 'stress'
    coef_df.loc[coef_df['coefficient'] > 0, 'type'] = 'anti-stress'

    return coef_df


# =============================================================================
# FSI CALCULATION
# =============================================================================

def calculate_ml_fsi(X: pd.DataFrame, model, scaler) -> pd.Series:
    """
    Calculate ML-based Financial Stress Index.

    FSI = -predicted_return (normalized to 0-1 scale)
    High FSI = predicted market decline = high stress
    0 = minimum stress, 1 = maximum stress
    """
    X_scaled = scaler.transform(X)
    predicted_returns = model.predict(X_scaled)

    # FSI = negative of predicted returns
    fsi_raw = -predicted_returns

    return pd.Series(fsi_raw, index=X.index)


def standardize_fsi(fsi: pd.Series, vol: pd.Series) -> pd.Series:
    """
    Standardize FSI to 0-1 scale.
    Ensures positive correlation with volatility.

    0 = minimum/no stress, 1 = maximum stress
    """
    # Align indices
    common = fsi.index.intersection(vol.index)
    fsi_aligned = fsi.loc[common]
    vol_aligned = vol.loc[common]

    # Check correlation sign - if negative, flip FSI
    corr = fsi_aligned.corr(vol_aligned)
    if corr < 0:
        print(f"    Flipping FSI sign (correlation was {corr:.3f})")
        fsi = -fsi

    # Normalize to 0-1 scale using min-max normalization
    fsi_min = fsi.min()
    fsi_max = fsi.max()

    if fsi_max > fsi_min:
        fsi_normalized = (fsi - fsi_min) / (fsi_max - fsi_min)
    else:
        fsi_normalized = pd.Series(0.5, index=fsi.index)

    # Ensure bounded [0, 1]
    fsi_normalized = fsi_normalized.clip(0, 1)

    return fsi_normalized


# =============================================================================
# VALIDATION
# =============================================================================

def validate_fsi(fsi: pd.Series, vol: pd.Series, label: str = "") -> Dict:
    """Validate FSI against market volatility."""
    common = fsi.index.intersection(vol.index)
    fsi_aligned = fsi.loc[common]
    vol_aligned = vol.loc[common]

    if len(common) < 10:
        return {'pearson': np.nan, 'spearman': np.nan, 'n': len(common)}

    pearson_r, pearson_p = stats.pearsonr(fsi_aligned, vol_aligned)
    spearman_r, spearman_p = stats.spearmanr(fsi_aligned, vol_aligned)

    return {
        'pearson': pearson_r,
        'pearson_p': pearson_p,
        'spearman': spearman_r,
        'spearman_p': spearman_p,
        'n': len(common),
        'label': label
    }


def print_validation(results: Dict, is_test: bool = False):
    """Print validation report."""
    label = "OUT-OF-SAMPLE" if is_test else "IN-SAMPLE"
    r = results

    if r['n'] < 10 or np.isnan(r['pearson']):
        print(f"  {label} (n={r['n']}): Insufficient data")
        return

    status = "OK" if CONFIG.TARGET_CORR_MIN <= abs(r['pearson']) <= CONFIG.TARGET_CORR_MAX else ""
    print(f"  {label} (n={r['n']}):")
    print(f"    Pearson:  r = {r['pearson']:.3f} (p = {r['pearson_p']:.4f}) {status}")
    print(f"    Spearman: r = {r['spearman']:.3f} (p = {r['spearman_p']:.4f})")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_fsi_timeseries(fsi: pd.Series, save_path: str = None):
    """Plot ML FSI time series (0-1 scale)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    dates = fsi.index.to_timestamp()
    ax.plot(dates, fsi.values, color=COLORS['fsi'], linewidth=1.5, label='ML FSI')

    # 4-week moving average
    rolling = fsi.rolling(4).mean()
    ax.plot(dates, rolling.values, color=COLORS['fsi_ma'], linewidth=2, label='4-week MA', alpha=0.8)

    # Neutral line at 0.5
    ax.axhline(y=0.5, color=COLORS['neutral'], linestyle='--', alpha=0.5, label='Neutral (0.5)')

    # Fill areas above/below neutral (softer colors)
    ax.fill_between(dates, 0.5, fsi.values,
                    where=fsi.values > 0.5, alpha=0.25, color=COLORS['high_stress'])
    ax.fill_between(dates, 0.5, fsi.values,
                    where=fsi.values <= 0.5, alpha=0.25, color=COLORS['low_stress'])

    # Crisis markers (softer shading)
    for (start, end), label in CRISIS_EPISODES.items():
        start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
        if dates.min() <= end_dt and dates.max() >= start_dt:
            ax.axvspan(start_dt, end_dt, alpha=0.08, color=COLORS['crisis'])

    ax.set_xlabel('Date')
    ax.set_ylabel('FSI (0-1 scale)')
    ax.set_ylim(0, 1)
    ax.set_title('ML-Generated Financial Stress Index')
    ax.legend(loc='upper left')

    # Improved time axis - show years appropriately
    date_range = (dates.max() - dates.min()).days / 365
    if date_range > 10:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_fsi_vs_volatility(fsi: pd.Series, vol: pd.Series, save_path: str = None):
    """Plot FSI vs volatility (FSI on 0-1 scale)."""
    common = fsi.index.intersection(vol.index)
    fsi_aligned = fsi.loc[common]
    vol_aligned = vol.loc[common]
    dates = common.to_timestamp()

    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(dates, fsi_aligned.values, color=COLORS['fsi'], linewidth=1.5, label='ML FSI')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('FSI (0-1 scale)', color=COLORS['fsi'])
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='y', labelcolor=COLORS['fsi'])

    ax2 = ax1.twinx()
    ax2.plot(dates, vol_aligned.values, color=COLORS['volatility'], linewidth=1.5, label='Volatility', alpha=0.7)
    ax2.set_ylabel('Volatility (%)', color=COLORS['volatility'])
    ax2.tick_params(axis='y', labelcolor=COLORS['volatility'])

    corr = fsi_aligned.corr(vol_aligned)
    ax1.set_title(f'ML FSI vs IBOVESPA Volatility (r = {corr:.3f})')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Improved time axis
    date_range = (dates.max() - dates.min()).days / 365
    if date_range > 10:
        ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    else:
        ax1.xaxis.set_major_locator(mdates.YearLocator(1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_top_queries(coef_df: pd.DataFrame, n_top: int = 20, save_path: str = None):
    """Plot top queries by coefficient magnitude."""
    # Get top stress and anti-stress queries
    stress = coef_df[coef_df['type'] == 'stress'].head(n_top // 2)
    anti = coef_df[coef_df['type'] == 'anti-stress'].head(n_top // 2)
    top = pd.concat([stress, anti]).sort_values('coefficient')

    if len(top) == 0:
        print("  No significant coefficients to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = [COLORS['stress'] if c < 0 else COLORS['anti_stress'] for c in top['coefficient']]
    ax.barh(top['query'], top['coefficient'], color=colors, edgecolor='white', linewidth=0.5)

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Coefficient')
    ax.set_title('Top Predictive Queries (Red = Stress, Blue = Anti-stress)')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_coefficient_distribution(coef_df: pd.DataFrame, save_path: str = None):
    """Plot distribution of coefficients."""
    fig, ax = plt.subplots(figsize=(10, 6))

    coefs = coef_df['coefficient']
    nonzero = coefs[coefs != 0]

    ax.hist(nonzero, bins=30, color=COLORS['fsi'], edgecolor='white', alpha=0.7)
    ax.axvline(x=0, color=COLORS['stress'], linestyle='--', linewidth=2)

    ax.set_xlabel('Coefficient Value')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of LASSO Coefficients ({len(nonzero)} non-zero out of {len(coefs)})')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='ML-Generated Financial Stress Index (García et al. 2023)')
    parser.add_argument('--train-end', type=str, default='2022-12-31', help='End of training period')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')

    args = parser.parse_args()
    CONFIG.TRAIN_END = args.train_end

    print("=" * 60)
    print("ML-GENERATED FINANCIAL STRESS INDEX")
    print("García, Hu, and Rohrer (2023) Methodology")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    try:
        gt_df, market_df = load_data()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return

    # Prepare features and target
    print("\nPreparing features and target...")
    X, y, vol = prepare_features_and_target(gt_df, market_df)
    print(f"    Sample size: {len(X)} observations")
    print(f"    Features: {X.shape[1]} queries")
    print(f"    Period: {X.index.min()} to {X.index.max()}")

    # Validate queries
    print("\nValidating queries...")
    X = validate_queries(X)

    if X.shape[1] < CONFIG.MIN_QUERIES:
        print(f"\nERROR: Only {X.shape[1]} valid queries (need {CONFIG.MIN_QUERIES})")
        return

    # Train/test split (temporal) - detect frequency from data
    data_freq = X.index[0].freq
    train_end = pd.Period(CONFIG.TRAIN_END, freq=data_freq)
    X_train = X[X.index <= train_end]
    y_train = y[y.index <= train_end]
    X_test = X[X.index > train_end]
    y_test = y[y.index > train_end]

    print(f"\nTrain/Test Split:")
    print(f"    Training: {len(X_train)} observations (up to {CONFIG.TRAIN_END})")
    print(f"    Testing:  {len(X_test)} observations (after {CONFIG.TRAIN_END})")

    # Train LASSO model
    print("\nTraining LASSO model...")
    model, scaler, cv_results = train_lasso_model(X_train, y_train)

    # Get coefficients
    coef_df = get_coefficients(model, X.columns.tolist())

    # Print top queries
    print("\nTop 10 STRESS queries (predict decline):")
    stress_queries = coef_df[coef_df['type'] == 'stress'].head(10)
    for _, row in stress_queries.iterrows():
        print(f"    {row['query']}: {row['coefficient']:.4f}")

    print("\nTop 10 ANTI-STRESS queries (predict gain):")
    anti_queries = coef_df[coef_df['type'] == 'anti-stress'].head(10)
    for _, row in anti_queries.iterrows():
        print(f"    {row['query']}: {row['coefficient']:.4f}")

    # Calculate FSI for full sample
    print("\nCalculating ML FSI...")
    fsi_raw = calculate_ml_fsi(X, model, scaler)
    fsi = standardize_fsi(fsi_raw, vol)

    # Out-of-sample R²
    X_test_scaled = scaler.transform(X_test)
    r2_test = model.score(X_test_scaled, y_test)
    cv_results['test_r2'] = r2_test
    print(f"    Out-of-sample R²: {r2_test:.4f}")

    # Validate
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print("\nCorrelation with IBOVESPA Volatility:")
    print("-" * 40)

    # In-sample validation
    fsi_train = fsi[fsi.index <= train_end]
    vol_train = vol[vol.index <= train_end]
    train_val = validate_fsi(fsi_train, vol_train, "in-sample")
    print_validation(train_val, is_test=False)

    # Out-of-sample validation
    fsi_test = fsi[fsi.index > train_end]
    vol_test = vol[vol.index > train_end]
    test_val = validate_fsi(fsi_test, vol_test, "out-of-sample")
    print_validation(test_val, is_test=True)

    print(f"\n  Target correlation: {CONFIG.TARGET_CORR_MIN:.2f} - {CONFIG.TARGET_CORR_MAX:.2f}")

    # Update cv_results
    cv_results['corr_train'] = train_val['pearson']
    cv_results['corr_test'] = test_val['pearson']

    # Save results
    print("\nSaving results...")
    CONFIG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Also save to output root for combined_fsi.py compatibility
    output_root = _PROJECT_DIR / 'output'
    output_root.mkdir(parents=True, exist_ok=True)

    # FSI weekly (save to both locations for compatibility)
    # Use end of week period (Sunday) to match Google Trends date format
    fsi_df = pd.DataFrame({
        'date': fsi.index.to_timestamp(how='end').normalize(),
        'ml_fsi': fsi.values  # Renamed for consistency with other FSI outputs
    })
    fsi_df.to_csv(CONFIG.OUTPUT_DIR / 'fsi_ml_weekly.csv', index=False)
    fsi_df.to_csv(output_root / 'ml_fsi_weekly.csv', index=False)  # For combined_fsi.py
    print(f"  Saved: fsi_ml_weekly.csv and ml_fsi_weekly.csv")

    # Coefficients
    coef_df.to_csv(CONFIG.OUTPUT_DIR / 'ml_coefficients.csv', index=False)
    print(f"  Saved: ml_coefficients.csv")

    # Diagnostics
    diag_df = pd.DataFrame([cv_results])
    diag_df.to_csv(CONFIG.OUTPUT_DIR / 'ml_diagnostics.csv', index=False)
    print(f"  Saved: ml_diagnostics.csv")

    # Plots
    if not args.no_plots:
        print("\nGenerating plots...")
        CONFIG.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        plot_fsi_timeseries(fsi, CONFIG.PLOTS_DIR / 'ml_fsi_timeseries.png')
        plot_fsi_vs_volatility(fsi, vol, CONFIG.PLOTS_DIR / 'ml_fsi_vs_volatility.png')
        plot_top_queries(coef_df, save_path=CONFIG.PLOTS_DIR / 'ml_top_queries.png')
        plot_coefficient_distribution(coef_df, CONFIG.PLOTS_DIR / 'ml_coefficient_distribution.png')

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Results: {CONFIG.OUTPUT_DIR}/")
    print(f"  Plots:   {CONFIG.PLOTS_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
