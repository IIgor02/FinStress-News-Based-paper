# Financial Stress Index (FSI) - Dictionary-Based

A Python implementation combining Baker et al. (2019) dictionary-based methodology with Da et al. (2011) Google Trends approach for measuring financial stress in Brazil using Twitter/X and Google Trends data.

## Overview

This project calculates a **Financial Stress Index (FSI)** using two complementary approaches:

### 1. Twitter/X - Three-Way Co-occurrence (Baker et al. 2019)
Analyzes social media posts for three-way co-occurrence of:
- **Financial terms** (mercado, bolsa, Ibovespa, banco, etc.)
- **Stress terms** (crise, risco, volatilidade, panico, etc.)
- **Negative terms** (queda, perda, prejuizo, colapso, etc.)

A post is classified as "financial stress" only if it contains at least one term from **each** category.

### 2. Google Trends - Search Volume Index (Da et al. 2011)
Uses pre-combined search queries that already capture financial stress context:
- 25 Portuguese stress queries in 5 tiers
- Tier-weighted aggregation (Tier 1 = 1.5x, Tier 2 = 1.2x, Tiers 3-5 = 1.0x)
- Weekly frequency for official, representative data

## Project Structure (3 Main Scripts)

```
FinStress-News-Based-paper/
├── scripts/
│   ├── dictionaries.py    # Script 1: Portuguese dictionaries + Google Trends queries
│   ├── collect_data.py    # Script 2: Data collection (Twitter/Google Trends/IBOVESPA)
│   └── run_fsi.py         # Script 3: FSI calculation, validation, plots
├── data/
│   └── raw/               # Input data (generated/collected)
├── output/
│   ├── results/           # CSV outputs
│   └── plots/             # Visualizations
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Data

```bash
# Generate synthetic data for testing
python scripts/collect_data.py --synthetic --start-date 2015-01-01 --end-date 2024-12-31

# Or collect real data (requires API credentials)
python scripts/collect_data.py --all --start-date 2015-01-01 --end-date 2024-12-31
```

### 3. Calculate FSI

```bash
python scripts/run_fsi.py
```

### 4. View Results

- **CSV files**: `output/results/`
- **Plots**: `output/plots/`

## Detailed Usage

### Script 1: Dictionaries (`scripts/dictionaries.py`)

View dictionary and query statistics:
```bash
python scripts/dictionaries.py
```

Output:
```
Term Counts (Twitter):
   Financial terms: 270
   Stress terms:    163
   Negative terms:  441
   Total unique:    849

Google Trends Queries: 25
   Tier 1 (core):   5 queries (weight 1.5)
   Tier 2 (market): 5 queries (weight 1.2)
   Tier 3 (macro):  5 queries (weight 1.0)
   Tier 4 (forex):  5 queries (weight 1.0)
   Tier 5 (sent.):  5 queries (weight 1.0)
```

### Script 2: Data Collection (`scripts/collect_data.py`)

```bash
# Synthetic data (for testing)
python scripts/collect_data.py --synthetic

# Real IBOVESPA data only
python scripts/collect_data.py --ibovespa

# All data with custom date range
python scripts/collect_data.py --all --start-date 2015-01-01 --end-date 2024-12-31
```

**Data Sources:**
| Source | Method | Frequency | Notes |
|--------|--------|-----------|-------|
| Twitter | snscrape / API v2 | Daily | Requires bearer token for API |
| Google Trends | pytrends | Weekly | Free, no API key needed |
| IBOVESPA | yfinance | Daily | Free, official market data |

**Environment Variables (for real Twitter data):**
```bash
export TWITTER_BEARER_TOKEN="your_token"    # Twitter API v2
```

### Script 3: FSI Calculation (`scripts/run_fsi.py`)

```bash
# Default settings
python scripts/run_fsi.py

# Custom parameters
python scripts/run_fsi.py --min-posts 10 --twitter-weight 0.6 --gt-weight 0.4 --gt-method pca

# Skip plots (faster)
python scripts/run_fsi.py --no-plots
```

**Tunable Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-posts` | 5 | Minimum posts/day for valid Twitter FSI |
| `--twitter-weight` | 0.5 | Twitter weight in combined FSI |
| `--gt-weight` | 0.5 | Google Trends weight in combined FSI |
| `--gt-method` | weighted_average | GT aggregation: `average`, `weighted_average`, `pca` |
| `--smoothing` | 4 | Rolling average window in weeks (0=none) |

## Methodology

### Twitter FSI (Baker et al. 2019)

```
Daily FSI = (weighted_stress_posts / weighted_financial_posts) x 100

where:
  - weight = log(1 + followers)
  - stress_post = has_financial AND has_stress AND has_negative
```

### Google Trends FSI (Da et al. 2011)

```
Weekly FSSVI = weighted_average(SVI_query_i * tier_weight_i)

where:
  - SVI = Search Volume Index (0-100)
  - Tier 1 weight = 1.5 (core crisis terms)
  - Tier 2 weight = 1.2 (market terms)
  - Tier 3-5 weight = 1.0 (macro, forex, sentiment)
```

### Combined Index

```
Combined FSI = (Twitter_weekly * tw_weight) + (GT_weekly * gt_weight)
```

### Google Trends Query Tiers

| Tier | Category | Queries | Weight |
|------|----------|---------|--------|
| 1 | Core Crisis | crise financeira, crise economica, panico financeiro... | 1.5 |
| 2 | Market | queda ibovespa, volatilidade bolsa, bear market brasil... | 1.2 |
| 3 | Macro | inflacao alta, juros altos, divida publica... | 1.0 |
| 4 | Currency | dolar alto, desvalorizacao real, fuga capitais... | 1.0 |
| 5 | Sentiment | medo mercado, pessimismo economia, recessao brasil... | 1.0 |

## Validation Results

Target correlation with IBOVESPA volatility: **0.60 - 0.90**

**Achieved (synthetic data, 2015-2024):**
```
Combined FSI:
  Weekly:  r = 0.771
  Monthly: r = 0.863

Twitter FSI:
  Weekly:  r = 0.836
  Monthly: r = 0.866

Google Trends FSI:
  Weekly:  r = 0.780
  Monthly: r = 0.853
```

## Output Files

### CSV Results (`output/results/`)
| File | Description |
|------|-------------|
| `fsi_twitter_daily.csv` | Daily FSI from Twitter |
| `fsi_google_trends_weekly.csv` | Weekly FSI from Google Trends (FSSVI) |
| `fsi_combined_weekly.csv` | Combined weekly FSI |
| `fsi_monthly.csv` | Monthly FSI |

### Plots (`output/plots/`)
| File | Description |
|------|-------------|
| `fsi_timeseries.png` | FSI over time with crisis markers |
| `fsi_vs_volatility.png` | FSI vs IBOVESPA volatility |
| `fsi_components.png` | Twitter vs Google Trends comparison |

## Crisis Episodes Tracked

The index marks these Brazilian crisis periods:
- **2008-2009**: Global Financial Crisis
- **2014**: Petrobras Scandal
- **2015-2016**: Brazilian Recession / Dilma Impeachment
- **2017**: JBS Scandal (Joesley Day)
- **2018**: Truckers Strike
- **2020**: COVID-19 Crash
- **2021**: Fiscal Concerns (PEC dos Precatorios)
- **2022**: Election Uncertainty
- **2023**: January 8 Events

## Programmatic Usage

```python
from scripts.dictionaries import (
    DICTIONARIES, STRESS_QUERIES_GT, QUERY_WEIGHTS,
    get_dictionary_stats, get_query_weight
)
from scripts.run_fsi import classify_post, preprocess_text

# Check dictionary stats
stats = get_dictionary_stats()
print(f"Twitter terms: {stats['total_unique']}")
print(f"Google Trends queries: {stats['n_gt_queries']}")

# Classify a single post (Twitter)
result = classify_post("A crise no mercado financeiro gera panico com queda das acoes")
print(f"Is stress: {result['is_stress']}")  # True

# Get query weight (Google Trends)
weight = get_query_weight("crise financeira")  # 1.5 (Tier 1)
```

## Data Format

### Input: Twitter CSV
```csv
date,text,followers,user_id
2024-01-01 10:30:00,A crise no mercado financeiro...,15000,user_123
```

### Input: Google Trends CSV
```csv
date,crise financeira,crise economica,panico financeiro,...
2024-01-07,45,32,28,...
2024-01-14,48,35,31,...
```

### Input: IBOVESPA CSV
```csv
date,close,return,realized_vol_30d
2024-01-02,130000,0.012,18.5
```

## References

- Baker, S. R., Bloom, N., Davis, S. J., & Kost, K. J. (2019). Policy News and Stock Market Volatility. NBER Working Paper No. 25720.
- Da, Z., Engelberg, J., & Gao, P. (2011). In Search of Attention. The Journal of Finance, 66(5), 1461-1499.

## Author

CAEN/UFC Master's Research
