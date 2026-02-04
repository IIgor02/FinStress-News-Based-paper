# Financial Stress Index (FSI) - Dictionary-Based

A Python implementation of the Baker et al. (2019) dictionary-based methodology for measuring financial stress using Brazilian Portuguese social media data (Twitter/X and Reddit).

## Overview

This project calculates a **Financial Stress Index (FSI)** by analyzing social media posts for three-way co-occurrence of:
1. **Financial terms** (mercado, bolsa, Ibovespa, banco, etc.)
2. **Stress terms** (crise, risco, volatilidade, pânico, etc.)
3. **Negative terms** (queda, perda, prejuízo, colapso, etc.)

A post is classified as indicating "financial stress" only if it contains at least one term from **each** category.

## Project Structure (3 Main Scripts)

```
FinStress-News-Based-paper/
├── scripts/
│   ├── dictionaries.py    # Script 1: Portuguese term dictionaries
│   ├── collect_data.py    # Script 2: Data collection (Twitter/Reddit/IBOVESPA)
│   └── run_fsi.py         # Script 3: FSI calculation, validation, plots
├── src/                   # Legacy modules (kept for reference)
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
python scripts/collect_data.py --synthetic --start-date 2020-01-01 --end-date 2024-12-31

# Or collect real data (requires API credentials)
python scripts/collect_data.py --all --start-date 2020-01-01 --end-date 2024-12-31
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

View dictionary statistics:
```bash
python scripts/dictionaries.py
```

Output:
```
📚 Term Counts:
   Financial terms: 270
   Stress terms:    163
   Negative terms:  441
   Total unique:    849
```

### Script 2: Data Collection (`scripts/collect_data.py`)

```bash
# Synthetic data (for testing)
python scripts/collect_data.py --synthetic

# Real IBOVESPA data only
python scripts/collect_data.py --ibovespa

# All data with custom date range
python scripts/collect_data.py --all --start-date 2020-01-01 --end-date 2024-12-31
```

**Environment Variables (for real data):**
```bash
export TWITTER_BEARER_TOKEN="your_token"    # Twitter API v2
export REDDIT_CLIENT_ID="your_client_id"    # Reddit API
export REDDIT_CLIENT_SECRET="your_secret"   # Reddit API
```

### Script 3: FSI Calculation (`scripts/run_fsi.py`)

```bash
# Default settings
python scripts/run_fsi.py

# Custom parameters
python scripts/run_fsi.py --min-posts 10 --twitter-weight 0.7 --reddit-weight 0.3 --smoothing 14

# Skip plots (faster)
python scripts/run_fsi.py --no-plots
```

**Tunable Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-posts` | 5 | Minimum posts/day for valid FSI |
| `--twitter-weight` | 0.6 | Twitter weight in combined FSI |
| `--reddit-weight` | 0.4 | Reddit weight in combined FSI |
| `--smoothing` | 7 | Rolling average window (0=none) |

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
- Standard deviation preserved

### Three-Way Co-occurrence

A post is classified as "stress" if it contains at least one term from:
1. **Financial** (270 terms): mercado, bolsa, Ibovespa, ações, banco, crédito...
2. **Stress** (163 terms): crise, risco, volatilidade, pânico, incerteza, colapso...
3. **Negative** (441 terms): queda, perda, prejuízo, despencar, falência, problema...

## Validation Results

Target correlation with market volatility: **0.60 - 0.90** (from Baker et al. 2019)

**Achieved with synthetic data:**
```
Daily:   Pearson r = 0.754 ✓
Monthly: Pearson r = 0.824 ✓
```

## Output Files

### CSV Results (`output/results/`)
| File | Description |
|------|-------------|
| `fsi_twitter_daily.csv` | Daily FSI from Twitter |
| `fsi_reddit_daily.csv` | Daily FSI from Reddit |
| `fsi_combined_daily.csv` | Combined daily FSI |
| `fsi_monthly.csv` | Monthly FSI (all platforms) |

### Plots (`output/plots/`)
| File | Description |
|------|-------------|
| `fsi_timeseries.png` | FSI over time with crisis markers |
| `fsi_vs_volatility.png` | FSI vs IBOVESPA volatility |
| `fsi_scatter.png` | Correlation scatter plot |
| `fsi_monthly_comparison.png` | Platform comparison |

## Crisis Episodes Tracked

The index marks these Brazilian crisis periods:
- **2008-2009**: Global Financial Crisis
- **2014**: Petrobras Scandal
- **2015-2016**: Brazilian Recession / Dilma Impeachment
- **2017**: JBS Scandal (Joesley Day)
- **2018**: Truckers Strike
- **2020**: COVID-19 Crash
- **2021**: Fiscal Concerns (PEC dos Precatórios)
- **2022**: Election Uncertainty
- **2023**: January 8 Events

## Programmatic Usage

```python
from scripts.dictionaries import DICTIONARIES, get_dictionary_stats
from scripts.run_fsi import classify_post, preprocess_text

# Check dictionary stats
stats = get_dictionary_stats()
print(f"Total terms: {stats['total_unique']}")

# Classify a single post
result = classify_post("A crise no mercado financeiro gera pânico com queda das ações")
print(f"Is stress: {result['is_stress']}")  # True
```

## Data Format

### Input: Twitter CSV
```csv
date,text,followers,user_id
2024-01-01 10:30:00,A crise no mercado financeiro...,15000,user_123
```

### Input: Reddit CSV
```csv
date,text,upvotes,subreddit
2024-01-01 14:00:00,Ibovespa despenca com volatilidade...,234,investimentos
```

### Input: IBOVESPA CSV
```csv
date,close,return,realized_vol_30d
2024-01-02,130000,0.012,18.5
```

## References

Baker, S. R., Bloom, N., Davis, S. J., & Kost, K. J. (2019). Policy News and Stock Market Volatility. NBER Working Paper No. 25720.

## Author

CAEN/UFC Master's Research
