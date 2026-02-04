#!/usr/bin/env python3
"""
Financial Stress Index (FSI) - Dictionary-Based Implementation
==============================================================
Based on Baker et al. (2019) methodology adapted for Brazilian Portuguese.

This script calculates the FSI using three-way co-occurrence:
- A post is "stress" if it contains terms from ALL three categories:
  1. Financial terms (mercado, bolsa, Ibovespa, etc.)
  2. Stress terms (crise, risco, volatilidade, etc.)
  3. Negative terms (queda, perda, prejuízo, etc.)

Usage:
    python fsi_dictionary.py [--synthetic] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]

Example:
    python fsi_dictionary.py --synthetic --start-date 2020-01-01 --end-date 2024-12-31

Author: CAEN/UFC Master Research
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dictionaries import DICTIONARIES, CRISIS_EPISODES, get_dictionary_stats
from src.preprocessing import preprocess_text
from src.classifier import StressClassifier
from src.fsi_calculator import FSICalculator
from src.data_generator import (
    generate_synthetic_twitter_data,
    generate_synthetic_reddit_data,
    download_ibovespa_data,
)
from src.validation import (
    validate_fsi,
    print_validation_report,
    plot_fsi_timeseries,
    plot_fsi_vs_volatility,
    plot_scatter_correlation,
)


def setup_directories():
    """Create output directories if they don't exist."""
    dirs = [
        'data/raw',
        'data/processed',
        'output/results',
        'output/plots',
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def load_or_generate_data(
    use_synthetic: bool = True,
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31',
    twitter_path: str = None,
    reddit_path: str = None,
) -> tuple:
    """
    Load real data or generate synthetic data for testing.

    Args:
        use_synthetic: Whether to use synthetic data
        start_date: Start date for data
        end_date: End date for data
        twitter_path: Path to Twitter CSV (if not synthetic)
        reddit_path: Path to Reddit CSV (if not synthetic)

    Returns:
        Tuple of (twitter_df, reddit_df, market_df)
    """
    if use_synthetic:
        print("\n📊 Generating synthetic data for testing...")

        twitter_df = generate_synthetic_twitter_data(
            start_date=start_date,
            end_date=end_date,
            posts_per_day=(80, 150),
            stress_ratio=0.18,
            financial_ratio=0.65,
            seed=42,
        )
        print(f"  Twitter: {len(twitter_df):,} posts generated")

        reddit_df = generate_synthetic_reddit_data(
            start_date=start_date,
            end_date=end_date,
            posts_per_day=(30, 70),
            stress_ratio=0.22,
            financial_ratio=0.75,
            seed=43,
        )
        print(f"  Reddit: {len(reddit_df):,} posts generated")

    else:
        # Load from CSV files
        if twitter_path and Path(twitter_path).exists():
            twitter_df = pd.read_csv(twitter_path)
            twitter_df['date'] = pd.to_datetime(twitter_df['date'])
            print(f"  Twitter: Loaded {len(twitter_df):,} posts from {twitter_path}")
        else:
            twitter_df = pd.DataFrame()
            print("  Twitter: No data file provided")

        if reddit_path and Path(reddit_path).exists():
            reddit_df = pd.read_csv(reddit_path)
            reddit_df['date'] = pd.to_datetime(reddit_df['date'])
            print(f"  Reddit: Loaded {len(reddit_df):,} posts from {reddit_path}")
        else:
            reddit_df = pd.DataFrame()
            print("  Reddit: No data file provided")

    # Download or generate market data
    print("\n📈 Loading market data (IBOVESPA)...")
    market_df = download_ibovespa_data(
        start_date=start_date,
        end_date=end_date,
        use_synthetic=use_synthetic,
    )
    print(f"  Market: {len(market_df):,} trading days")

    return twitter_df, reddit_df, market_df


def process_platform(
    df: pd.DataFrame,
    platform: str,
    calculator: FSICalculator,
    weight_column: str,
) -> tuple:
    """
    Process data from a single platform.

    Args:
        df: DataFrame with social media posts
        platform: 'twitter' or 'reddit'
        calculator: FSICalculator instance
        weight_column: Column containing weights

    Returns:
        Tuple of (classified_df, daily_fsi, monthly_fsi)
    """
    if df.empty:
        return None, None, None

    print(f"\n  Processing {platform.title()}...")

    # Classify posts
    classified = calculator.classify_dataframe(df, text_column='text')

    n_financial = classified['is_financial'].sum()
    n_stress = classified['is_stress'].sum()
    stress_rate = n_stress / n_financial * 100 if n_financial > 0 else 0

    print(f"    Financial posts: {n_financial:,} ({n_financial/len(classified)*100:.1f}%)")
    print(f"    Stress posts: {n_stress:,} ({stress_rate:.1f}% of financial)")

    # Calculate daily FSI
    daily = calculator.calculate_daily_fsi(
        classified,
        date_column='date',
        weight_column=weight_column,
        platform=platform,
    )

    # Standardize
    daily['fsi'] = calculator.standardize_fsi(daily['fsi_raw'])

    # Calculate monthly FSI
    monthly = calculator.calculate_monthly_fsi(daily, 'fsi')

    return classified, daily, monthly


def save_results(
    twitter_daily: pd.DataFrame,
    reddit_daily: pd.DataFrame,
    combined_daily: pd.DataFrame,
    twitter_monthly: pd.DataFrame,
    reddit_monthly: pd.DataFrame,
    combined_monthly: pd.DataFrame,
    output_dir: str = 'output/results',
):
    """Save FSI results to CSV files."""
    print("\n💾 Saving results...")

    if twitter_daily is not None:
        twitter_daily.to_csv(f'{output_dir}/fsi_twitter_daily.csv', index=False)
        print(f"  Saved: {output_dir}/fsi_twitter_daily.csv")

    if reddit_daily is not None:
        reddit_daily.to_csv(f'{output_dir}/fsi_reddit_daily.csv', index=False)
        print(f"  Saved: {output_dir}/fsi_reddit_daily.csv")

    if combined_daily is not None:
        combined_daily.to_csv(f'{output_dir}/fsi_combined_daily.csv', index=False)
        print(f"  Saved: {output_dir}/fsi_combined_daily.csv")

    # Combined monthly file
    if twitter_monthly is not None or reddit_monthly is not None:
        monthly_combined = pd.DataFrame()

        if twitter_monthly is not None:
            monthly_combined['month'] = twitter_monthly['month']
            monthly_combined['fsi_twitter'] = twitter_monthly['fsi']

        if reddit_monthly is not None:
            if 'month' not in monthly_combined.columns:
                monthly_combined['month'] = reddit_monthly['month']
            monthly_combined['fsi_reddit'] = reddit_monthly['fsi']

        if combined_monthly is not None:
            monthly_combined['fsi_combined'] = combined_monthly['fsi']

        monthly_combined.to_csv(f'{output_dir}/fsi_monthly.csv', index=False)
        print(f"  Saved: {output_dir}/fsi_monthly.csv")


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Calculate Dictionary-Based Financial Stress Index'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data for testing',
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2020-01-01',
        help='Start date (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2024-12-31',
        help='End date (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--twitter-data',
        type=str,
        help='Path to Twitter CSV file',
    )
    parser.add_argument(
        '--reddit-data',
        type=str,
        help='Path to Reddit CSV file',
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots',
    )

    args = parser.parse_args()

    # Header
    print("=" * 60)
    print("FINANCIAL STRESS INDEX - DICTIONARY-BASED")
    print("Baker et al. (2019) Methodology for Brazilian Portuguese")
    print("=" * 60)

    # Setup
    setup_directories()

    # Print dictionary stats
    stats = get_dictionary_stats()
    print("\n📚 Dictionary Statistics:")
    print(f"  Financial terms: {stats['financial_terms']}")
    print(f"  Stress terms: {stats['stress_terms']}")
    print(f"  Negative terms: {stats['negative_terms']}")
    print(f"  Total unique: {stats['total_unique']}")

    # Load data
    twitter_df, reddit_df, market_df = load_or_generate_data(
        use_synthetic=args.synthetic or (not args.twitter_data and not args.reddit_data),
        start_date=args.start_date,
        end_date=args.end_date,
        twitter_path=args.twitter_data,
        reddit_path=args.reddit_data,
    )

    # Initialize calculator
    print("\n🔍 Classifying posts...")
    calculator = FSICalculator(min_posts_per_day=5)

    # Process each platform
    twitter_classified, twitter_daily, twitter_monthly = process_platform(
        twitter_df, 'twitter', calculator, 'followers'
    )

    reddit_classified, reddit_daily, reddit_monthly = process_platform(
        reddit_df, 'reddit', calculator, 'upvotes'
    )

    # Combine platforms
    combined_daily = None
    combined_monthly = None

    if twitter_daily is not None and reddit_daily is not None:
        print("\n  Combining platforms...")
        combined_daily = calculator.combine_platforms(
            twitter_daily,
            reddit_daily,
            twitter_weight=0.6,
            reddit_weight=0.4,
        )
        combined_daily['fsi'] = calculator.standardize_fsi(combined_daily['fsi_raw'])
        combined_monthly = calculator.calculate_monthly_fsi(combined_daily, 'fsi')
    elif twitter_daily is not None:
        combined_daily = twitter_daily
        combined_monthly = twitter_monthly
    elif reddit_daily is not None:
        combined_daily = reddit_daily
        combined_monthly = reddit_monthly

    # Validation
    print("\n📊 Validating against market volatility...")

    if combined_daily is not None:
        validation_results = validate_fsi(
            combined_daily,
            market_df,
            monthly_fsi=combined_monthly,
            fsi_column='fsi',
            vol_column='realized_vol_30d',
        )
        print_validation_report(validation_results)

    # Save results
    save_results(
        twitter_daily,
        reddit_daily,
        combined_daily,
        twitter_monthly,
        reddit_monthly,
        combined_monthly,
    )

    # Generate plots
    if not args.no_plots and combined_daily is not None:
        print("\n📈 Generating visualizations...")

        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend

            plot_fsi_timeseries(
                combined_daily,
                fsi_column='fsi',
                title='Financial Stress Index (Combined)',
                save_path='output/plots/fsi_timeseries.png',
            )

            plot_fsi_vs_volatility(
                combined_daily,
                market_df,
                fsi_column='fsi',
                vol_column='realized_vol_30d',
                save_path='output/plots/fsi_vs_volatility.png',
            )

            # Merge for scatter plot
            merged = pd.merge(
                combined_daily[['date', 'fsi']],
                market_df[['date', 'realized_vol_30d']],
                on='date',
            )
            plot_scatter_correlation(
                merged['fsi'],
                merged['realized_vol_30d'],
                save_path='output/plots/fsi_scatter.png',
            )

            print("  All plots saved to output/plots/")

        except Exception as e:
            print(f"  Warning: Could not generate plots: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_posts = len(twitter_df) + len(reddit_df)
    total_financial = (
        (twitter_classified['is_financial'].sum() if twitter_classified is not None else 0) +
        (reddit_classified['is_financial'].sum() if reddit_classified is not None else 0)
    )
    total_stress = (
        (twitter_classified['is_stress'].sum() if twitter_classified is not None else 0) +
        (reddit_classified['is_stress'].sum() if reddit_classified is not None else 0)
    )

    print(f"\nData Coverage: {args.start_date} to {args.end_date}")
    print(f"Total Posts Analyzed: {total_posts:,}")
    if len(twitter_df) > 0:
        print(f"  - Twitter: {len(twitter_df):,} ({len(twitter_df)/total_posts*100:.1f}%)")
    if len(reddit_df) > 0:
        print(f"  - Reddit: {len(reddit_df):,} ({len(reddit_df)/total_posts*100:.1f}%)")
    print(f"\nPosts Classified as Financial: {total_financial:,} ({total_financial/total_posts*100:.1f}%)")
    print(f"Posts Classified as Stress: {total_stress:,} ({total_stress/total_financial*100:.1f}% of financial)")

    if combined_daily is not None and 'daily' in validation_results:
        corr = validation_results['daily']['pearson']
        status = "✓ TARGET ACHIEVED" if 0.6 <= abs(corr) <= 0.9 else "○ Review methodology"
        print(f"\nCorrelation with IBOVESPA Volatility: r = {corr:.3f} {status}")

    print("\n✅ FSI calculation complete!")
    print("   Results saved to: output/results/")
    print("   Plots saved to: output/plots/")
    print("=" * 60)


if __name__ == '__main__':
    main()
