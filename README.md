# Financial Stress Index (FSI) for Brazil

Dictionary-based Financial Stress Index using Google Trends search volume data for Brazil, following Da, Engelberg & Gao (2011) methodology.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect data (real Google Trends + IBOVESPA)
python scripts/collect_data.py

# 3. Calculate FSI
python scripts/run_fsi.py
```

## Project Structure

```
FinStress-News-Based-paper/
├── scripts/
│   ├── dictionaries.py    # 25 Portuguese stress queries (5 tiers)
│   ├── collect_data.py    # Google Trends + IBOVESPA collection
│   └── run_fsi.py         # FSI calculation and validation
├── data/raw/              # Input data
├── output/
│   ├── results/           # CSV outputs (fsi_weekly.csv, fsi_monthly.csv)
│   └── plots/             # Visualizations
└── requirements.txt
```

## Data Collection

```bash
# Collect real data from Google Trends + Yahoo Finance
python scripts/collect_data.py

# Or generate synthetic data for testing (no internet)
python scripts/collect_data.py --synthetic

# Custom date range
python scripts/collect_data.py --start-date 2015-01-01 --end-date 2025-12-31
```

## FSI Calculation

```bash
# Default (weighted average by tier)
python scripts/run_fsi.py

# Using PCA
python scripts/run_fsi.py --aggregation pca

# Skip plots
python scripts/run_fsi.py --no-plots
```

### Aggregation Methods

| Method | Description |
|--------|-------------|
| `weighted_average` | Tier 1 = 1.5x, Tier 2 = 1.2x, Tier 3-5 = 1.0x |
| `average` | Simple average of all queries |
| `pca` | First principal component |

## Methodology (Da et al. 2011)

Uses Google Trends Search Volume Index (SVI) for pre-combined stress queries:

| Tier | Weight | Queries |
|------|--------|---------|
| 1 - Crisis | 1.5 | crise financeira, crise economica, panico financeiro... |
| 2 - Market | 1.2 | queda ibovespa, bolsa despenca, volatilidade bolsa... |
| 3 - Macro | 1.0 | recessao brasil, divida publica, desemprego brasil... |
| 4 - Currency | 1.0 | dolar dispara, desvalorizacao real, crise bancaria... |
| 5 - Sentiment | 1.0 | vender acoes, mercado vai cair, protecao carteira... |

## Output Files

- `output/results/fsi_weekly.csv` - Weekly FSI values
- `output/results/fsi_monthly.csv` - Monthly FSI values
- `output/plots/fsi_timeseries.png` - FSI over time
- `output/plots/fsi_vs_volatility.png` - FSI vs IBOVESPA volatility

## Crisis Episodes Tracked

- 2008-2009: Global Financial Crisis
- 2015-2016: Brazilian Recession
- 2017: JBS Scandal
- 2020: COVID-19 Crash
- 2021-2024: Various fiscal/political events

## References

- Da, Z., Engelberg, J., & Gao, P. (2011). In Search of Attention. The Journal of Finance, 66(5), 1461-1499.

## Author

Pedro Igor - CAEN/UFC
