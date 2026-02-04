"""
Financial Stress Index Calculator
=================================
Calculates daily and monthly FSI from classified social media data.

Formula (Baker et al. 2019):
    Daily FSI = (weighted_stress_posts / weighted_financial_posts) × 100

Weighting:
    - Twitter: weight = log(1 + followers)
    - Reddit: weight = log(1 + upvotes)
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .classifier import StressClassifier, ClassificationResult


class FSICalculator:
    """
    Calculate Financial Stress Index from social media data.
    """

    def __init__(
        self,
        classifier: Optional[StressClassifier] = None,
        target_mean: float = 100.0,
        min_posts_per_day: int = 10,
    ):
        """
        Initialize FSI calculator.

        Args:
            classifier: StressClassifier instance (creates default if None)
            target_mean: Target mean for standardized index (default 100)
            min_posts_per_day: Minimum financial posts required per day
        """
        self.classifier = classifier or StressClassifier()
        self.target_mean = target_mean
        self.min_posts_per_day = min_posts_per_day

    def calculate_weight(
        self,
        value: Union[int, float],
        platform: str = 'twitter'
    ) -> float:
        """
        Calculate post weight based on engagement metric.

        Args:
            value: Followers (Twitter) or upvotes (Reddit)
            platform: 'twitter' or 'reddit'

        Returns:
            Log-transformed weight
        """
        # Ensure non-negative
        value = max(0, value)
        return np.log1p(value)  # log(1 + value)

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
    ) -> pd.DataFrame:
        """
        Classify all posts in a DataFrame.

        Args:
            df: DataFrame with social media posts
            text_column: Name of text column

        Returns:
            DataFrame with classification columns added
        """
        df = df.copy()

        # Classify each text
        results = []
        for text in df[text_column]:
            result = self.classifier.classify(str(text))
            results.append(result)

        # Add classification columns
        df['is_financial'] = [r.is_financial for r in results]
        df['is_stress'] = [r.is_stress_post for r in results]
        df['n_financial_terms'] = [len(r.financial_matches) for r in results]
        df['n_stress_terms'] = [len(r.stress_matches) for r in results]
        df['n_negative_terms'] = [len(r.negative_matches) for r in results]

        return df

    def calculate_daily_fsi(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        weight_column: Optional[str] = None,
        platform: str = 'twitter',
    ) -> pd.DataFrame:
        """
        Calculate daily FSI from classified data.

        Args:
            df: DataFrame with classified posts (needs is_financial, is_stress)
            date_column: Name of date column
            weight_column: Column with weight values (followers/upvotes)
            platform: 'twitter' or 'reddit' (for weight calculation)

        Returns:
            DataFrame with daily FSI values
        """
        df = df.copy()

        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column])
        df['date_only'] = df[date_column].dt.date

        # Calculate weights
        if weight_column and weight_column in df.columns:
            df['weight'] = df[weight_column].apply(
                lambda x: self.calculate_weight(x, platform)
            )
        else:
            df['weight'] = 1.0  # Equal weights if no weight column

        # Filter to financial posts only
        financial_df = df[df['is_financial']].copy()

        # Group by date and calculate FSI
        daily_stats = []
        for date, group in financial_df.groupby('date_only'):
            total_weight = group['weight'].sum()
            stress_weight = group[group['is_stress']]['weight'].sum()

            n_total = len(group)
            n_stress = group['is_stress'].sum()

            # Calculate raw FSI (percentage of stress posts by weight)
            if total_weight > 0:
                fsi_raw = (stress_weight / total_weight) * 100
            else:
                fsi_raw = np.nan

            daily_stats.append({
                'date': date,
                'fsi_raw': fsi_raw,
                'n_financial_posts': n_total,
                'n_stress_posts': n_stress,
                'total_weight': total_weight,
                'stress_weight': stress_weight,
            })

        daily_df = pd.DataFrame(daily_stats)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df = daily_df.sort_values('date').reset_index(drop=True)

        # Filter days with insufficient data
        mask = daily_df['n_financial_posts'] >= self.min_posts_per_day
        daily_df.loc[~mask, 'fsi_raw'] = np.nan

        return daily_df

    def standardize_fsi(
        self,
        fsi_series: pd.Series,
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
    ) -> pd.Series:
        """
        Standardize FSI series to have target mean.

        Args:
            fsi_series: Raw FSI values
            target_mean: Target mean (default: self.target_mean)
            target_std: Target std (if None, preserves original std)

        Returns:
            Standardized FSI series
        """
        target_mean = target_mean or self.target_mean

        # Remove NaN for calculation
        valid_values = fsi_series.dropna()

        if len(valid_values) == 0:
            return fsi_series

        # Z-score standardization
        mean = valid_values.mean()
        std = valid_values.std()

        if std == 0:
            return pd.Series(target_mean, index=fsi_series.index)

        # Standardize
        standardized = (fsi_series - mean) / std

        # Rescale to target
        if target_std:
            standardized = standardized * target_std + target_mean
        else:
            standardized = standardized * std + target_mean

        return standardized

    def calculate_monthly_fsi(
        self,
        daily_df: pd.DataFrame,
        fsi_column: str = 'fsi_raw',
        method: str = 'mean',
    ) -> pd.DataFrame:
        """
        Aggregate daily FSI to monthly.

        Args:
            daily_df: DataFrame with daily FSI
            fsi_column: Column with FSI values
            method: Aggregation method ('mean', 'median', 'weighted_mean')

        Returns:
            DataFrame with monthly FSI
        """
        df = daily_df.copy()
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')

        if method == 'mean':
            monthly = df.groupby('month').agg({
                fsi_column: 'mean',
                'n_financial_posts': 'sum',
                'n_stress_posts': 'sum',
            }).reset_index()
        elif method == 'median':
            monthly = df.groupby('month').agg({
                fsi_column: 'median',
                'n_financial_posts': 'sum',
                'n_stress_posts': 'sum',
            }).reset_index()
        elif method == 'weighted_mean':
            # Weight by number of posts
            def weighted_avg(group):
                weights = group['n_financial_posts']
                values = group[fsi_column]
                return np.average(values.dropna(), weights=weights[values.notna()])

            monthly = df.groupby('month').apply(weighted_avg).reset_index()
            monthly.columns = ['month', fsi_column]

            # Add post counts
            counts = df.groupby('month').agg({
                'n_financial_posts': 'sum',
                'n_stress_posts': 'sum',
            }).reset_index()
            monthly = monthly.merge(counts, on='month')
        else:
            raise ValueError(f"Unknown method: {method}")

        monthly['month'] = monthly['month'].dt.to_timestamp()
        monthly = monthly.rename(columns={fsi_column: 'fsi'})

        return monthly

    def combine_platforms(
        self,
        twitter_daily: pd.DataFrame,
        reddit_daily: pd.DataFrame,
        twitter_weight: float = 0.6,
        reddit_weight: float = 0.4,
    ) -> pd.DataFrame:
        """
        Combine FSI from multiple platforms.

        Args:
            twitter_daily: Daily FSI from Twitter
            reddit_daily: Daily FSI from Reddit
            twitter_weight: Weight for Twitter FSI
            reddit_weight: Weight for Reddit FSI

        Returns:
            Combined daily FSI DataFrame
        """
        # Merge on date
        combined = pd.merge(
            twitter_daily[['date', 'fsi_raw', 'n_financial_posts', 'n_stress_posts']],
            reddit_daily[['date', 'fsi_raw', 'n_financial_posts', 'n_stress_posts']],
            on='date',
            how='outer',
            suffixes=('_twitter', '_reddit')
        )

        # Calculate combined FSI (weighted average)
        combined['fsi_raw'] = (
            combined['fsi_raw_twitter'].fillna(0) * twitter_weight +
            combined['fsi_raw_reddit'].fillna(0) * reddit_weight
        )

        # Adjust for missing platforms
        twitter_present = combined['fsi_raw_twitter'].notna()
        reddit_present = combined['fsi_raw_reddit'].notna()

        # If only one platform present, use that platform's value
        only_twitter = twitter_present & ~reddit_present
        only_reddit = reddit_present & ~twitter_present

        combined.loc[only_twitter, 'fsi_raw'] = combined.loc[only_twitter, 'fsi_raw_twitter']
        combined.loc[only_reddit, 'fsi_raw'] = combined.loc[only_reddit, 'fsi_raw_reddit']

        # Total posts
        combined['n_financial_posts'] = (
            combined['n_financial_posts_twitter'].fillna(0) +
            combined['n_financial_posts_reddit'].fillna(0)
        )
        combined['n_stress_posts'] = (
            combined['n_stress_posts_twitter'].fillna(0) +
            combined['n_stress_posts_reddit'].fillna(0)
        )

        combined = combined.sort_values('date').reset_index(drop=True)

        return combined


def calculate_fsi_from_data(
    data: pd.DataFrame,
    text_column: str = 'text',
    date_column: str = 'date',
    weight_column: Optional[str] = None,
    platform: str = 'twitter',
    standardize: bool = True,
    target_mean: float = 100.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to calculate FSI from raw data.

    Args:
        data: DataFrame with social media posts
        text_column: Column containing post text
        date_column: Column containing dates
        weight_column: Column containing weights (followers/upvotes)
        platform: Platform type ('twitter' or 'reddit')
        standardize: Whether to standardize the index
        target_mean: Target mean for standardization

    Returns:
        Tuple of (daily_fsi, monthly_fsi) DataFrames
    """
    calculator = FSICalculator(target_mean=target_mean)

    # Classify posts
    classified = calculator.classify_dataframe(data, text_column)

    # Calculate daily FSI
    daily = calculator.calculate_daily_fsi(
        classified,
        date_column=date_column,
        weight_column=weight_column,
        platform=platform,
    )

    # Standardize
    if standardize:
        daily['fsi'] = calculator.standardize_fsi(daily['fsi_raw'])
    else:
        daily['fsi'] = daily['fsi_raw']

    # Calculate monthly FSI
    monthly = calculator.calculate_monthly_fsi(daily, 'fsi')

    return daily, monthly


if __name__ == '__main__':
    # Quick test with sample data
    print("FSI Calculator Module")
    print("=" * 50)

    # Create sample data
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100, freq='D').repeat(10),
        'text': [
            'Crise no mercado financeiro gera pânico com queda das ações',
            'Ibovespa fecha em alta com otimismo do investidor',
            'Risco de recessão preocupa analistas do banco',
        ] * 333 + ['Extra post'] * 1,  # 1000 posts
        'followers': np.random.randint(100, 100000, 1000),
    })

    # Calculate FSI
    calculator = FSICalculator()
    classified = calculator.classify_dataframe(sample_data)

    print(f"\nClassification Results:")
    print(f"  Total posts: {len(classified)}")
    print(f"  Financial posts: {classified['is_financial'].sum()}")
    print(f"  Stress posts: {classified['is_stress'].sum()}")
    print(f"  Stress rate: {classified['is_stress'].sum() / classified['is_financial'].sum() * 100:.1f}%")
