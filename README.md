# Financial Stress Index (FSI) for Brazil

Multiple complementary Financial Stress Indices for Brazil using Google Trends, LASSO regression, and news sentiment analysis.

## Methods

### 1. Dictionary-Based FSI (Da et al. 2011)
- Pre-defined stress queries with tier-based weights
- 25 queries in 5 tiers
- Weighted aggregation of Google Trends SVI

### 2. ML-Generated FSI (García et al. 2023)
- LASSO regression learns which queries predict market declines
- ~100 queries with automatic feature selection
- FSI = negative of predicted return

### 3. News-Based FSI (Baker et al. 2019 style)
- Scrapes news from G1, Valor Econômico, Folha de S. Paulo
- Three-way co-occurrence of financial + stress + negative terms
- Aggregates article-level stress to weekly index

### 4. Combined FSI
- Combines all methodologies using weighted average, PCA, or dynamic weights
- Validates against IBOVESPA realized volatility

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# === DICTIONARY-BASED FSI ===
python scripts/collect_data.py
python scripts/run_fsi.py

# === ML-BASED FSI ===
python scripts/collect_data.py --ml
python scripts/ml_fsi.py

# === NEWS-BASED FSI ===
python scripts/collect_news.py
python scripts/news_fsi.py

# === COMBINED FSI ===
python scripts/combined_fsi.py --method weighted
```

## Project Structure

```
FinStress-News-Based-paper/
├── scripts/
│   ├── dictionaries.py    # Dictionary definitions (270+ terms, 100+ queries)
│   ├── collect_data.py    # Google Trends + IBOVESPA collection
│   ├── collect_news.py    # News scraping (G1, Valor, Folha)
│   ├── run_fsi.py         # Dictionary-based FSI (Da et al.)
│   ├── ml_fsi.py          # ML-based FSI (García et al.)
│   ├── news_fsi.py        # News-based FSI (Baker et al.)
│   └── combined_fsi.py    # Combined FSI (all methods)
├── data/raw/              # Input data
├── output/                # Results and plots
└── requirements.txt
```

## Data Collection

### Google Trends Data
```bash
# Dictionary method (25 queries)
python scripts/collect_data.py

# ML method (104 queries)
python scripts/collect_data.py --ml

# Custom date range
python scripts/collect_data.py --start-date 2015-01-01 --end-date 2025-12-31
```

### News Data
```bash
# All sources
python scripts/collect_news.py

# Specific source
python scripts/collect_news.py --source g1
python scripts/collect_news.py --source valor
python scripts/collect_news.py --source folha

# Search with keywords
python scripts/collect_news.py --keywords "crise,inflação" --pages 10
```

## FSI Calculation

### Dictionary-Based
```bash
python scripts/run_fsi.py
python scripts/run_fsi.py --aggregation pca
```

### ML-Based (LASSO)
```bash
python scripts/ml_fsi.py
python scripts/ml_fsi.py --train-end 2020-12-31
```

### News-Based
```bash
python scripts/news_fsi.py
python scripts/news_fsi.py --scrape --pages 10
```

### Combined
```bash
python scripts/combined_fsi.py --method average
python scripts/combined_fsi.py --method weighted
python scripts/combined_fsi.py --method pca
python scripts/combined_fsi.py --method dynamic
```

## Methodology

### Dictionary FSI (Da et al. 2011)
```
FSI = weighted_average(SVI × tier_weight)

Tier 1 (Crisis):   1.5x weight
Tier 2 (Market):   1.2x weight
Tier 3-5:          1.0x weight
```

### ML FSI (García et al. 2023)
```
Step 1: r_{t+1} = γ₀ + Σ φ_k × SVI_{k,t}
Step 2: LASSO (L1 penalty) selects predictive queries
Step 3: FSI = -predicted_return
```

### News FSI (Baker et al. 2019 style)
```
For each article:
  1. Count financial terms (mercado, bolsa, etc.)
  2. Count stress terms (crise, pânico, etc.)
  3. Count negative terms (queda, perda, etc.)
  4. Article has co-occurrence if all three > 0

FSI = aggregate(stress_scores) per week
```

### Combined FSI
```
Method 1: Simple average of standardized FSIs
Method 2: Weighted average (Dict: 40%, ML: 35%, News: 25%)
Method 3: PCA first principal component
Method 4: Dynamic weights based on rolling correlation with volatility
```

## Output Files

| Script | Output |
|--------|--------|
| `run_fsi.py` | `output/results/fsi_weekly.csv`, `fsi_monthly.csv` |
| `ml_fsi.py` | `output/ml_fsi_weekly.csv`, `ml_coefficients.csv` |
| `news_fsi.py` | `output/news_fsi_weekly.csv`, `news_fsi_articles.csv` |
| `combined_fsi.py` | `output/combined_fsi.csv` |

## Query Categories

| Category | Queries | Examples |
|----------|---------|----------|
| Crisis | 15 | crise financeira, crash financeiro |
| Stock Market | 18 | queda ibovespa, circuit breaker |
| Banking | 12 | crise bancária, inadimplência |
| Sovereign | 14 | risco default, rebaixamento rating |
| Currency | 12 | dólar dispara, fuga capitais |
| Political | 11 | impeachment, lava jato |
| Macro | 12 | inflação alta, desemprego |
| Sentiment | 10 | vender ações, mercado vai cair |

## Text Dictionaries (for News Analysis)

| Dictionary | Terms | Examples |
|------------|-------|----------|
| Financial | ~270 | mercado, bolsa, ibovespa, ações |
| Stress | ~160 | crise, pânico, risco, instabilidade |
| Negative | ~440 | queda, perda, prejuízo, falência |

## References

- Da, Z., Engelberg, J., & Gao, P. (2011). In Search of Attention. *The Journal of Finance*.
- García, D., Hu, X., & Rohrer, M. (2023). The Color of Finance Words. *Journal of Financial Economics*.
- Baker, S.R., Bloom, N., & Davis, S.J. (2016). Measuring Economic Policy Uncertainty. *Quarterly Journal of Economics*.

## Author

Pedro Igor - CAEN/UFC
