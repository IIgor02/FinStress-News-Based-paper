"""
Synthetic Data Generator for FSI Testing
=========================================
Generates realistic Portuguese social media posts about financial topics
for testing the FSI methodology.
"""

import random
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .dictionaries import FINANCIAL_TERMS, STRESS_TERMS, NEGATIVE_TERMS, CRISIS_EPISODES


# Template components for generating realistic posts
TWEET_TEMPLATES = {
    'stress': [
        # Full stress posts (financial + stress + negative)
        "A {stress} no {financial} está gerando {negative} entre investidores",
        "{stress} no {financial} causa {negative} nas {financial2}",
        "Analistas alertam para {stress} com {negative} no {financial}",
        "{financial} em {negative} com {stress} generalizado",
        "URGENTE: {financial} sofre {negative} em meio a {stress}",
        "{stress} afeta {financial} e gera {negative}",
        "Com a {stress} no {financial}, investidores enfrentam {negative}",
        "{financial} despenca em dia de {stress} e {negative}",
        "Temor de {stress} causa {negative} no {financial}",
        "Depois da {stress}, {financial} registra {negative}",
        "{negative} no {financial} após sinais de {stress}",
        "{financial} fecha com {negative} devido à {stress}",
        "Investidores preocupados com {stress} e {negative} no {financial}",
    ],
    'financial_only': [
        # Financial posts without stress/negative
        "{financial} fecha em alta nesta segunda-feira",
        "Investidores otimistas com {financial}",
        "{financial} tem dia de ganhos expressivos",
        "Analistas recomendam compra de {financial2}",
        "{financial} supera expectativas do mercado",
        "Bons resultados impulsionam {financial}",
        "{financial} atinge nova máxima histórica",
        "Otimismo com {financial} anima investidores",
        "{financial2} lideram altas na {financial}",
        "Perspectiva positiva para o {financial}",
    ],
    'non_financial': [
        # Non-financial posts (noise)
        "Que dia lindo hoje! Bom fim de semana a todos",
        "Alguém mais viu o jogo ontem? Que partida!",
        "Recomendo muito esse restaurante novo",
        "Feriado chegando, hora de descansar",
        "Que filme incrível, vale muito a pena",
        "Bom dia pessoal! Vamos que vamos",
        "Saudades de viajar, quando isso vai acabar",
        "Essa receita ficou deliciosa",
        "Tempo bom para praia hoje",
        "Meu novo livro favorito chegou",
    ],
}

# Sample terms for template filling
SAMPLE_FINANCIAL = [
    'mercado', 'bolsa', 'Ibovespa', 'mercado financeiro', 'setor bancário',
    'crédito', 'economia', 'investimentos', 'ações', 'bancos',
]
SAMPLE_FINANCIAL2 = [
    'ações', 'papéis', 'ativos', 'fundos', 'títulos',
    'investimentos', 'carteiras', 'cotas',
]
SAMPLE_STRESS = [
    'crise', 'risco', 'volatilidade', 'instabilidade', 'incerteza',
    'pânico', 'turbulência', 'tensão', 'recessão', 'pressão',
]
SAMPLE_NEGATIVE = [
    'queda', 'perdas', 'prejuízo', 'desvalorização', 'retração',
    'problemas', 'dificuldades', 'piora', 'deterioração',
]


def generate_tweet(
    category: str = 'stress',
    templates: dict = TWEET_TEMPLATES,
) -> str:
    """
    Generate a single synthetic tweet.

    Args:
        category: 'stress', 'financial_only', or 'non_financial'
        templates: Template dictionary

    Returns:
        Generated tweet text
    """
    template = random.choice(templates[category])

    # Fill in placeholders
    tweet = template.format(
        financial=random.choice(SAMPLE_FINANCIAL),
        financial2=random.choice(SAMPLE_FINANCIAL2),
        stress=random.choice(SAMPLE_STRESS),
        negative=random.choice(SAMPLE_NEGATIVE),
    )

    # Add random hashtags sometimes
    if random.random() < 0.3:
        hashtags = ['#economia', '#mercado', '#investimentos', '#bolsa', '#brasil']
        tweet += ' ' + random.choice(hashtags)

    return tweet


def generate_follower_count(base_stress_level: float = 0.5) -> int:
    """
    Generate realistic follower count.

    Uses power law distribution to simulate real social media.

    Args:
        base_stress_level: Higher stress = more engagement (more followers active)

    Returns:
        Follower count
    """
    # Power law parameters
    alpha = 1.5 - base_stress_level * 0.3  # More engagement during stress

    # Generate from power law
    u = random.random()
    followers = int(100 * (1 - u) ** (-1/alpha))

    # Cap at reasonable max
    return min(followers, 10_000_000)


def generate_reddit_upvotes(is_stress: bool = False) -> int:
    """
    Generate Reddit upvote count.

    Args:
        is_stress: Stress posts tend to get more engagement

    Returns:
        Upvote count
    """
    base = 10 if is_stress else 5
    scale = 100 if is_stress else 50

    upvotes = int(np.random.exponential(scale) + base)
    return min(upvotes, 50_000)


def get_stress_probability(date: datetime, base_prob: float = 0.15) -> float:
    """
    Calculate stress probability based on date (crisis periods have higher prob).

    Args:
        date: Date to check
        base_prob: Base probability of stress

    Returns:
        Probability of stress post
    """
    date_str = date.strftime('%Y-%m-%d')

    # Check if date falls in a crisis period
    for (start, end), label in CRISIS_EPISODES.items():
        if start <= date_str <= end:
            return base_prob * 3  # 3x more stress during crises

    return base_prob


def generate_synthetic_twitter_data(
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31',
    posts_per_day: Tuple[int, int] = (50, 200),
    stress_ratio: float = 0.15,
    financial_ratio: float = 0.6,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate synthetic Twitter data for FSI testing.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        posts_per_day: Range of posts per day (min, max)
        stress_ratio: Base ratio of stress posts (among financial)
        financial_ratio: Ratio of financial posts (vs non-financial)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic Twitter data
    """
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    data = []
    current_date = start

    while current_date <= end:
        # Number of posts for this day
        n_posts = random.randint(*posts_per_day)

        # Adjust stress probability based on date
        stress_prob = get_stress_probability(current_date, stress_ratio)

        for _ in range(n_posts):
            # Determine post category
            if random.random() < financial_ratio:
                # Financial post
                if random.random() < stress_prob:
                    category = 'stress'
                else:
                    category = 'financial_only'
            else:
                category = 'non_financial'

            # Generate post
            text = generate_tweet(category)
            followers = generate_follower_count(stress_prob)

            # Add timestamp within day
            hour = random.randint(6, 23)
            minute = random.randint(0, 59)
            timestamp = current_date.replace(hour=hour, minute=minute)

            data.append({
                'date': timestamp,
                'text': text,
                'followers': followers,
                'user_id': f'user_{random.randint(1, 100000)}',
                'platform': 'twitter',
                '_category': category,  # For validation
            })

        current_date += timedelta(days=1)

    df = pd.DataFrame(data)
    df = df.sort_values('date').reset_index(drop=True)

    return df


def generate_synthetic_reddit_data(
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31',
    posts_per_day: Tuple[int, int] = (20, 80),
    stress_ratio: float = 0.20,
    financial_ratio: float = 0.8,
    seed: Optional[int] = 43,
) -> pd.DataFrame:
    """
    Generate synthetic Reddit data for FSI testing.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        posts_per_day: Range of posts per day (min, max)
        stress_ratio: Base ratio of stress posts
        financial_ratio: Ratio of financial posts
        seed: Random seed

    Returns:
        DataFrame with synthetic Reddit data
    """
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    subreddits = [
        'investimentos', 'brasil', 'farialimabets', 'bolsapt',
        'economia', 'financas', 'mercadofinanceiro',
    ]

    data = []
    current_date = start

    while current_date <= end:
        n_posts = random.randint(*posts_per_day)
        stress_prob = get_stress_probability(current_date, stress_ratio)

        for _ in range(n_posts):
            if random.random() < financial_ratio:
                if random.random() < stress_prob:
                    category = 'stress'
                else:
                    category = 'financial_only'
            else:
                category = 'non_financial'

            text = generate_tweet(category)
            is_stress = category == 'stress'
            upvotes = generate_reddit_upvotes(is_stress)

            hour = random.randint(8, 22)
            minute = random.randint(0, 59)
            timestamp = current_date.replace(hour=hour, minute=minute)

            data.append({
                'date': timestamp,
                'text': text,
                'upvotes': upvotes,
                'subreddit': random.choice(subreddits),
                'platform': 'reddit',
                '_category': category,
            })

        current_date += timedelta(days=1)

    df = pd.DataFrame(data)
    df = df.sort_values('date').reset_index(drop=True)

    return df


def download_ibovespa_data(
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31',
    use_synthetic: bool = False,
) -> pd.DataFrame:
    """
    Download IBOVESPA data from Yahoo Finance.

    Falls back to synthetic data if download fails.

    Args:
        start_date: Start date
        end_date: End date
        use_synthetic: Force synthetic data

    Returns:
        DataFrame with IBOVESPA data
    """
    if not use_synthetic:
        try:
            import yfinance as yf

            ticker = yf.Ticker('^BVSP')
            df = ticker.history(start=start_date, end=end_date)

            if len(df) > 0:
                df = df.reset_index()
                df = df.rename(columns={
                    'Date': 'date',
                    'Close': 'close',
                    'Volume': 'volume',
                })
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

                # Calculate returns
                df['return'] = df['close'].pct_change()

                # Calculate realized volatility (30-day rolling std of returns)
                df['realized_vol_30d'] = (
                    df['return'].rolling(window=30).std() * np.sqrt(252) * 100
                )

                return df[['date', 'close', 'return', 'realized_vol_30d', 'volume']]

        except Exception as e:
            print(f"Could not download IBOVESPA data: {e}")
            print("Using synthetic data instead.")

    # Generate synthetic market data
    return generate_synthetic_market_data(start_date, end_date)


def generate_synthetic_market_data(
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31',
    seed: int = 44,
) -> pd.DataFrame:
    """
    Generate synthetic market data that mimics IBOVESPA behavior.

    Args:
        start_date: Start date
        end_date: End date
        seed: Random seed

    Returns:
        DataFrame with synthetic market data
    """
    np.random.seed(seed)

    dates = pd.date_range(start_date, end_date, freq='B')  # Business days
    n_days = len(dates)

    # Generate returns with regime switching
    base_vol = 0.015  # 1.5% daily vol
    returns = []

    for i, date in enumerate(dates):
        date_str = date.strftime('%Y-%m-%d')

        # Higher vol during crisis periods
        vol_multiplier = 1.0
        for (start, end), _ in CRISIS_EPISODES.items():
            if start <= date_str <= end:
                vol_multiplier = 2.5
                break

        daily_vol = base_vol * vol_multiplier
        daily_return = np.random.normal(0.0003, daily_vol)  # Small positive drift
        returns.append(daily_return)

    # Calculate prices
    initial_price = 100000  # IBOVESPA-like starting point
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Calculate realized volatility
    returns_series = pd.Series(returns)
    realized_vol = returns_series.rolling(window=30).std() * np.sqrt(252) * 100

    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'return': returns,
        'realized_vol_30d': realized_vol,
        'volume': np.random.randint(1e9, 5e9, n_days),
    })

    return df


if __name__ == '__main__':
    print("Generating synthetic data...")

    # Generate Twitter data
    twitter_df = generate_synthetic_twitter_data(
        start_date='2020-01-01',
        end_date='2020-03-31',  # Just Q1 2020 for testing
        posts_per_day=(50, 100),
    )
    print(f"\nTwitter data: {len(twitter_df)} posts")
    print(f"  Categories: {twitter_df['_category'].value_counts().to_dict()}")
    print(f"  Sample tweet: {twitter_df['text'].iloc[0]}")

    # Generate Reddit data
    reddit_df = generate_synthetic_reddit_data(
        start_date='2020-01-01',
        end_date='2020-03-31',
        posts_per_day=(20, 50),
    )
    print(f"\nReddit data: {len(reddit_df)} posts")
    print(f"  Categories: {reddit_df['_category'].value_counts().to_dict()}")

    # Generate market data
    market_df = download_ibovespa_data(
        start_date='2020-01-01',
        end_date='2020-03-31',
        use_synthetic=True,
    )
    print(f"\nMarket data: {len(market_df)} days")
    print(f"  Vol range: {market_df['realized_vol_30d'].min():.1f}% - {market_df['realized_vol_30d'].max():.1f}%")
