# Financial Stress Index (FSI) for Brazil

Dictionary-based Financial Stress Index using Twitter and Google Trends data for Brazil.

Based on:
- Baker, Bloom, Davis & Kost (2019) - Three-way co-occurrence methodology
- Da, Engelberg & Gao (2011) - Google Trends Search Volume Index approach

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect data (synthetic for testing)
python scripts/collect_data.py --synthetic

# 3. Calculate FSI
python scripts/run_fsi.py
```

## Project Structure

```
FinStress-News-Based-paper/
├── scripts/
│   ├── dictionaries.py    # Portuguese dictionaries + Google Trends queries
│   ├── collect_data.py    # Data collection (Twitter/Google Trends/IBOVESPA)
│   └── run_fsi.py         # FSI calculation and validation
├── data/raw/              # Input data files
├── output/
│   ├── results/           # CSV outputs
│   └── plots/             # Visualizations
└── requirements.txt
```

## Data Collection

### Option 1: Synthetic Data (for testing)
```bash
python scripts/collect_data.py --synthetic
```

### Option 2: Real Data
```bash
# Google Trends only (free, no API key)
python scripts/collect_data.py --google-trends

# All sources (requires Twitter API)
python scripts/collect_data.py --all
```

### Data Sources

| Source | Method | Frequency | API Key |
|--------|--------|-----------|---------|
| Google Trends | pytrends | Weekly | Not required |
| Twitter | snscrape | Daily | Not required |
| IBOVESPA | yfinance | Daily | Not required |

## FSI Calculation

```bash
# Default settings (50% Twitter, 50% Google Trends)
python scripts/run_fsi.py

# Custom weights
python scripts/run_fsi.py --twitter-weight 0.6 --gt-weight 0.4

# Using PCA for Google Trends aggregation
python scripts/run_fsi.py --aggregation pca
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--twitter-weight` | 0.5 | Twitter weight in combined FSI |
| `--gt-weight` | 0.5 | Google Trends weight in combined FSI |
| `--aggregation` | weighted_average | GT method: average, weighted_average, pca |
| `--min-posts` | 5 | Minimum tweets per day |
| `--no-plots` | false | Skip generating plots |

## Methodology

### Twitter FSI (Baker et al. 2019)

Posts are classified using **three-way co-occurrence**:
- Contains financial term AND
- Contains stress term AND
- Contains negative term

```
Daily FSI = (weighted stress posts / weighted financial posts) × 100
Weight = log(1 + followers)
```

### Google Trends FSI (Da et al. 2011)

Pre-combined search queries in 5 tiers:

| Tier | Weight | Examples |
|------|--------|----------|
| 1 - Crisis | 1.5 | crise financeira, panico financeiro |
| 2 - Market | 1.2 | queda ibovespa, bolsa despenca |
| 3 - Macro | 1.0 | recessao brasil, divida publica |
| 4 - Currency | 1.0 | dolar dispara, desvalorizacao real |
| 5 - Sentiment | 1.0 | vender acoes, mercado vai cair |

### Combined Index

```
Weekly FSI = (Twitter × tw_weight) + (Google Trends × gt_weight)
```

## Output Files

### Results (CSV)
- `fsi_twitter_daily.csv` - Daily Twitter FSI
- `fsi_google_trends_weekly.csv` - Weekly Google Trends FSI
- `fsi_combined_weekly.csv` - Combined weekly FSI
- `fsi_monthly.csv` - Monthly FSI

### Plots (PNG)
- `fsi_timeseries.png` - FSI over time with crisis markers
- `fsi_vs_volatility.png` - FSI vs IBOVESPA volatility
- `fsi_components.png` - Twitter vs Google Trends comparison

## Validation

Target correlation with IBOVESPA volatility: **0.60 - 0.90**

Expected results (synthetic data):
```
Combined FSI:   r = 0.77 (weekly), r = 0.86 (monthly)
Twitter FSI:    r = 0.84 (weekly), r = 0.87 (monthly)
Google Trends:  r = 0.78 (weekly), r = 0.85 (monthly)
```

## Dictionary Statistics

```
Financial terms: 270
Stress terms:    163
Negative terms:  441
Total unique:    849

Google Trends queries: 25 (5 tiers)
```

## Crisis Episodes Tracked

- 2008-2009: Global Financial Crisis
- 2014: Petrobras Scandal
- 2015-2016: Brazilian Recession / Dilma Impeachment
- 2017: JBS Scandal (Joesley Day)
- 2018: Truckers Strike
- 2020: COVID-19 Crash
- 2021: Fiscal Concerns
- 2022: Election Uncertainty
- 2023: January 8 Events
- 2024: Fiscal Framework Concerns

## References

- Baker, S. R., Bloom, N., Davis, S. J., & Kost, K. J. (2019). Policy News and Stock Market Volatility. NBER Working Paper No. 25720.
- Da, Z., Engelberg, J., & Gao, P. (2011). In Search of Attention. The Journal of Finance, 66(5), 1461-1499.

## Author

Pedro Igor
CAEN/UFC
