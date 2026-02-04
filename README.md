# Financial Stress Index (FSI) - Dictionary-Based

A Python implementation of the Baker et al. (2019) dictionary-based methodology for measuring financial stress using Brazilian Portuguese social media data (Twitter/X and Reddit).

## Overview

This project calculates a **Financial Stress Index (FSI)** by analyzing social media posts for three-way co-occurrence of:
1. **Financial terms** (mercado, bolsa, Ibovespa, banco, etc.)
2. **Stress terms** (crise, risco, volatilidade, pânico, etc.)
3. **Negative terms** (queda, perda, prejuízo, colapso, etc.)

A post is classified as indicating "financial stress" only if it contains at least one term from **each** category.

## Methodology

Based on Baker, Bloom, Davis, and Kost (2019): "Policy News and Stock Market Volatility", adapted for Brazilian Portuguese.

### Index Formula

```
Daily FSI = (weighted_stress_posts / weighted_financial_posts) × 100
```

**Weighting:**
- Twitter: `weight = log(1 + followers)`
- Reddit: `weight = log(1 + upvotes)`

**Standardization:**
- Mean = 100
- Standard deviation preserved from raw data

## Usage

### Quick Start (Synthetic Data)

Test the methodology with synthetic data:

```bash
python fsi_dictionary.py --synthetic --start-date 2020-01-01 --end-date 2024-12-31
```

### With Real Data

```bash
python fsi_dictionary.py \
    --twitter-data data/raw/twitter_data.csv \
    --reddit-data data/raw/reddit_data.csv \
    --start-date 2020-01-01 \
    --end-date 2024-12-31
```

### Expected Data Format

**Twitter CSV:**
```csv
date,text,followers,user_id
2024-01-01 10:30:00,A crise no mercado financeiro...,15000,user_123
```

**Reddit CSV:**
```csv
date,text,upvotes,subreddit
2024-01-01 14:00:00,Ibovespa despenca com volatilidade...,234,investimentos
```

## Output

### CSV Files (`output/results/`)
- `fsi_twitter_daily.csv` - Daily FSI from Twitter
- `fsi_reddit_daily.csv` - Daily FSI from Reddit
- `fsi_combined_daily.csv` - Combined daily FSI
- `fsi_monthly.csv` - Monthly FSI (all platforms)

### Plots (`output/plots/`)
- `fsi_timeseries.png` - FSI over time with crisis markers
- `fsi_vs_volatility.png` - FSI overlaid with market volatility
- `fsi_scatter.png` - Correlation scatter plot

## Project Structure

```
FinStress-News-Based-paper/
├── fsi_dictionary.py      # Main script
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── dictionaries.py    # Portuguese term dictionaries
│   ├── preprocessing.py   # Text preprocessing
│   ├── classifier.py      # Three-way co-occurrence classifier
│   ├── fsi_calculator.py  # FSI calculation with weighting
│   ├── data_generator.py  # Synthetic data generation
│   └── validation.py      # Validation and visualization
├── data/
│   ├── raw/               # Input data
│   └── processed/         # Processed data
└── output/
    ├── results/           # CSV outputs
    └── plots/             # Visualizations
```

## Dictionary Statistics

- **Financial terms:** 105 terms
- **Stress terms:** 67 terms
- **Negative terms:** 172 terms
- **Total unique terms:** 341

## Validation

Target correlation with IBOVESPA realized volatility

## Programmatic Usage

```python
from src import (
    StressClassifier,
    FSICalculator,
    validate_fsi,
    generate_synthetic_twitter_data,
)

# Generate test data
twitter_df = generate_synthetic_twitter_data(
    start_date='2020-01-01',
    end_date='2024-12-31',
)

# Initialize components
classifier = StressClassifier()
calculator = FSICalculator()

# Classify posts
classified = calculator.classify_dataframe(twitter_df, text_column='text')

# Calculate FSI
daily_fsi = calculator.calculate_daily_fsi(
    classified,
    date_column='date',
    weight_column='followers',
    platform='twitter',
)

# Standardize
daily_fsi['fsi'] = calculator.standardize_fsi(daily_fsi['fsi_raw'])
```

## Crisis Episodes

The index automatically marks these Brazilian crisis periods:
- Petrobras Scandal (2014)
- Brazilian Recession / Dilma Impeachment (2015-2016)
- JBS Scandal - Joesley Day (2017)
- Truckers Strike (2018)
- COVID-19 Crash (2020)
- Fiscal Concerns - PEC (2021)
- Election Uncertainty (2022)
- January 8 Events (2023)

## References

Baker, S. R., Bloom, N., Davis, S. J., & Kost, K. J. (2019). Policy News and Stock Market Volatility. NBER Working Paper No. 25720.

## Author

Pedro Igor
CAEN/UFC
