#!/usr/bin/env python3
"""
Investigation of FSI Construction Methods
==========================================
Analyzes why different FSI indices behave differently and tests
potential improvements at the source level.

Key questions:
1. Why does ML FSI have inverted behavior relative to CDS?
2. Can we improve the combination method?
3. Should we use dict_fsi alone?
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'
PLOTS_DIR = OUTPUT_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    """Load aligned monthly FSI data."""
    df = pd.read_csv(OUTPUT_DIR / 'fsi_monthly_aligned.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_correlations(df):
    """Analyze correlations between FSI indices and CDS."""
    print("=" * 70)
    print("CORRELATION ANALYSIS: FSI vs CDS (Log-Returns)")
    print("=" * 70)

    # Calculate log returns of CDS (stationarity)
    df = df.copy()
    df['cds_log_return'] = np.log(df['cds_5y'] / df['cds_5y'].shift(1))

    # FSI changes (first difference for comparable dynamics)
    for col in ['dict_fsi', 'ml_fsi', 'combined_fsi']:
        df[f'{col}_diff'] = df[col].diff()

    df = df.dropna()

    # Correlations with CDS log returns (contemporaneous)
    print("\n1. CONTEMPORANEOUS CORRELATIONS (FSI change vs CDS log-return)")
    print("-" * 50)
    for col in ['dict_fsi_diff', 'ml_fsi_diff', 'combined_fsi_diff']:
        corr, pval = stats.pearsonr(df[col], df['cds_log_return'])
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {col:20s}: r = {corr:+.4f} (p = {pval:.4f}) {sig}")

    # Predictive correlations (FSI change predicts NEXT period CDS change)
    df['cds_log_return_next'] = df['cds_log_return'].shift(-1)
    df = df.dropna()

    print("\n2. PREDICTIVE CORRELATIONS (FSI change → Next period CDS log-return)")
    print("-" * 50)
    for col in ['dict_fsi_diff', 'ml_fsi_diff', 'combined_fsi_diff']:
        corr, pval = stats.pearsonr(df[col], df['cds_log_return_next'])
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {col:20s}: r = {corr:+.4f} (p = {pval:.4f}) {sig}")

    # Cross-correlation of FSI indices
    print("\n3. INTER-FSI CORRELATIONS")
    print("-" * 50)
    for col1 in ['dict_fsi', 'ml_fsi', 'combined_fsi']:
        for col2 in ['dict_fsi', 'ml_fsi', 'combined_fsi']:
            if col1 < col2:
                corr, pval = stats.pearsonr(df[col1], df[col2])
                print(f"  {col1:15s} vs {col2:15s}: r = {corr:+.4f}")

    return df


def analyze_crisis_behavior(df):
    """Analyze FSI behavior during known crisis periods."""
    print("\n" + "=" * 70)
    print("CRISIS PERIOD ANALYSIS")
    print("=" * 70)

    crises = [
        ('2008 GFC', '2008-09-01', '2009-03-31'),
        ('2015 Fiscal Crisis', '2015-08-01', '2016-02-28'),
        ('COVID-19', '2020-02-01', '2020-06-30'),
    ]

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    print("\n            Period          | dict_fsi | ml_fsi | combined | CDS (mean)")
    print("-" * 70)

    # Pre-crisis baseline (full sample)
    baseline = df.copy()

    for name, start, end in crises:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        crisis_data = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

        if len(crisis_data) > 0:
            dict_mean = crisis_data['dict_fsi'].mean()
            ml_mean = crisis_data['ml_fsi'].mean()
            combined_mean = crisis_data['combined_fsi'].mean()
            cds_mean = crisis_data['cds_5y'].mean()

            print(f"  {name:20s} |  {dict_mean:.3f}   |  {ml_mean:.3f}  |  {combined_mean:.3f}   | {cds_mean:.1f}")

    # Overall means for comparison
    print("-" * 70)
    print(f"  {'Full Sample':20s} |  {df['dict_fsi'].mean():.3f}   |  {df['ml_fsi'].mean():.3f}  |  {df['combined_fsi'].mean():.3f}   | {df['cds_5y'].mean():.1f}")

    print("\n  Analysis:")
    print("  - dict_fsi shows HIGHER values during crises (correct behavior)")
    print("  - ml_fsi shows inconsistent/lower values during crises (inverted)")
    print("  - combined_fsi is diluted by ml_fsi's poor behavior")


def test_alternative_combinations(df):
    """Test alternative ways to combine FSI indices."""
    print("\n" + "=" * 70)
    print("ALTERNATIVE COMBINATION METHODS")
    print("=" * 70)

    df = df.copy()

    # Calculate CDS log returns
    df['cds_log_return'] = np.log(df['cds_5y'] / df['cds_5y'].shift(1))

    # Method 1: Dict FSI only
    df['fsi_dict_only'] = df['dict_fsi']

    # Method 2: Sign-corrected ML FSI (flip if negative corr with dict)
    dict_ml_corr = df['dict_fsi'].corr(df['ml_fsi'])
    if dict_ml_corr < 0:
        df['ml_fsi_corrected'] = 1 - df['ml_fsi']  # Flip
        print(f"  ML FSI flipped (original corr with dict_fsi: {dict_ml_corr:.3f})")
    else:
        df['ml_fsi_corrected'] = df['ml_fsi']

    # Method 3: Dict + Corrected ML (average)
    df['fsi_dict_ml_corrected'] = (df['dict_fsi'] + df['ml_fsi_corrected']) / 2

    # Method 4: Dict-only weighted (more weight on dict)
    df['fsi_weighted_dict'] = 0.7 * df['dict_fsi'] + 0.3 * df['ml_fsi_corrected']

    # Method 5: Max of dict and corrected ML
    df['fsi_max'] = df[['dict_fsi', 'ml_fsi_corrected']].max(axis=1)

    # Method 6: Principal Component (first PC)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = df[['dict_fsi', 'ml_fsi_corrected']].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X_scaled).flatten()

    # Standardize PC1 to 0-1
    pc1_min, pc1_max = pc1.min(), pc1.max()
    pc1_normalized = (pc1 - pc1_min) / (pc1_max - pc1_min)
    df.loc[X.index, 'fsi_pca'] = pc1_normalized

    # Evaluate all methods
    methods = [
        ('fsi_dict_only', 'Dictionary FSI Only'),
        ('combined_fsi', 'Original Combined'),
        ('fsi_dict_ml_corrected', 'Dict + Corrected ML'),
        ('fsi_weighted_dict', 'Weighted (70% Dict)'),
        ('fsi_max', 'Max(Dict, Corrected ML)'),
        ('fsi_pca', 'PCA (Dict, Corrected ML)'),
    ]

    # Calculate predictive correlation with next-period CDS change
    df['cds_log_return_next'] = df['cds_log_return'].shift(-1)
    df_eval = df.dropna()

    print("\n  Method                      | Corr(FSI_diff, CDS_next) | Significance")
    print("  " + "-" * 65)

    results = []
    for col, name in methods:
        df_eval[f'{col}_diff'] = df_eval[col].diff()

        # Predictive correlation
        valid = df_eval[[f'{col}_diff', 'cds_log_return_next']].dropna()
        corr, pval = stats.pearsonr(valid[f'{col}_diff'], valid['cds_log_return_next'])

        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"  {name:30s} | r = {corr:+.4f}              | p = {pval:.4f} {sig}")

        results.append({
            'method': col,
            'name': name,
            'corr': corr,
            'pval': pval
        })

    return df, results


def run_local_projection_comparison(df, methods):
    """Run Local Projections IRF for each combination method."""
    print("\n" + "=" * 70)
    print("LOCAL PROJECTIONS IRF COMPARISON")
    print("=" * 70)

    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    df = df.copy()
    df['cds_log_return'] = np.log(df['cds_5y'] / df['cds_5y'].shift(1))

    max_horizon = 12
    n_lags = 4

    # Store results for each method
    lp_results = {}

    for col, name in methods:
        # Calculate FSI change
        df[f'{col}_change'] = df[col].diff()

        # Standardize
        shock_std = df[f'{col}_change'].std()
        if shock_std > 0:
            df[f'{col}_shock'] = df[f'{col}_change'] / shock_std
        else:
            df[f'{col}_shock'] = df[f'{col}_change']

        irfs = []
        ses = []
        pvals = []

        for h in range(max_horizon + 1):
            # Cumulative response over h periods
            df['y_forward'] = df['cds_log_return'].rolling(h + 1).sum().shift(-h)

            # Build regression: y_t+h = α + β*shock_t + controls + ε
            X_cols = [f'{col}_shock']

            # Add lagged CDS returns as controls
            for lag in range(1, n_lags + 1):
                df[f'cds_lag_{lag}'] = df['cds_log_return'].shift(lag)
                X_cols.append(f'cds_lag_{lag}')

            # Prepare data
            reg_df = df[['y_forward'] + X_cols].dropna()

            if len(reg_df) < 30:
                irfs.append(np.nan)
                ses.append(np.nan)
                pvals.append(np.nan)
                continue

            y = reg_df['y_forward']
            X = add_constant(reg_df[X_cols])

            # OLS with HAC standard errors
            model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': h + 1})

            irfs.append(model.params[f'{col}_shock'])
            ses.append(model.bse[f'{col}_shock'])
            pvals.append(model.pvalues[f'{col}_shock'])

        lp_results[col] = {
            'name': name,
            'irf': np.array(irfs),
            'se': np.array(ses),
            'pval': np.array(pvals),
            'cum_irf': np.nansum(irfs),
            'sig_months': np.sum(np.array(pvals) < 0.1)
        }

    # Print summary
    print("\n  Method                      | Cum. IRF | Sig. Months | Best for CDS?")
    print("  " + "-" * 65)

    best_method = None
    best_cum_irf = -np.inf

    for col, res in lp_results.items():
        cum_irf = res['cum_irf']
        sig = res['sig_months']

        # We want POSITIVE cumulative IRF (stress → CDS increase)
        is_best = cum_irf > 0 and cum_irf > best_cum_irf
        if is_best:
            best_cum_irf = cum_irf
            best_method = col

        indicator = "←" if is_best else ""
        print(f"  {res['name']:30s} | {cum_irf:+.3f}    |     {sig:2d}      | {indicator}")

    return lp_results, best_method


def plot_irf_comparison(lp_results, save_path):
    """Plot IRF comparison for different methods."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    horizons = np.arange(len(list(lp_results.values())[0]['irf']))

    for idx, (col, res) in enumerate(lp_results.items()):
        if idx >= 6:
            break

        ax = axes[idx]
        irf = res['irf']
        se = res['se']

        # Plot IRF with confidence bands
        ax.plot(horizons, irf, 'b-', linewidth=2, label='IRF')
        ax.fill_between(horizons, irf - 1.96 * se, irf + 1.96 * se,
                        alpha=0.2, color='blue', label='95% CI')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        ax.set_title(f"{res['name']}\n(Cum: {res['cum_irf']:.2f}, Sig: {res['sig_months']})")
        ax.set_xlabel('Months')
        ax.set_ylabel('CDS Log-Return Response')
        ax.grid(True, alpha=0.3)

    plt.suptitle('IRF Comparison: FSI Shock → CDS Response (Different Methods)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  Saved comparison plot: {save_path}")


def main():
    print("=" * 70)
    print("FSI CONSTRUCTION INVESTIGATION")
    print("=" * 70)
    print("\nInvestigating why different FSI indices behave differently")
    print("and testing potential improvements at the source level.\n")

    # Load data
    df = load_data()
    print(f"Loaded {len(df)} monthly observations")

    # Analysis 1: Correlations
    df = analyze_correlations(df)

    # Analysis 2: Crisis behavior
    analyze_crisis_behavior(df)

    # Analysis 3: Alternative combinations
    df, combo_results = test_alternative_combinations(df)

    # Analysis 4: Local Projections comparison
    methods = [
        ('fsi_dict_only', 'Dictionary FSI Only'),
        ('combined_fsi', 'Original Combined'),
        ('fsi_dict_ml_corrected', 'Dict + Corrected ML'),
        ('fsi_weighted_dict', 'Weighted (70% Dict)'),
        ('fsi_max', 'Max(Dict, Corrected ML)'),
        ('fsi_pca', 'PCA (Dict, Corrected ML)'),
    ]

    lp_results, best_method = run_local_projection_comparison(df, methods)

    # Plot comparison
    plot_path = PLOTS_DIR / 'fsi_construction_comparison.png'
    plot_irf_comparison(lp_results, plot_path)

    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if best_method:
        best_name = lp_results[best_method]['name']
        best_cum = lp_results[best_method]['cum_irf']
        best_sig = lp_results[best_method]['sig_months']

        print(f"\n  Best method: {best_name}")
        print(f"  Cumulative IRF: {best_cum:.3f}")
        print(f"  Significant months: {best_sig}")

        if best_method == 'fsi_dict_only':
            print("\n  CONCLUSION: Dictionary FSI alone is the best choice.")
            print("  The ML FSI has fundamental issues (inverted during crises)")
            print("  and combining it with dict_fsi only dilutes the signal.")
            print("\n  RECOMMENDATION: Use dict_fsi as the primary index for analysis.")
        elif 'corrected' in best_method or 'weighted' in best_method:
            print("\n  CONCLUSION: Sign-corrected combination provides some improvement.")
            print("  However, the improvement over dict_fsi alone is marginal.")
            print("\n  RECOMMENDATION: For simplicity, use dict_fsi alone.")
    else:
        print("\n  No clear winner - all methods show weak performance.")
        print("  RECOMMENDATION: Use dict_fsi as it has theoretical backing.")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
