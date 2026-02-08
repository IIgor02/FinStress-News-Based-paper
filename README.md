# Financial Stress Index (FSI) for Brazil

Multiple complementary Financial Stress Indices for Brazil using Google Trends, LASSO regression, and news sentiment analysis.

**Analysis Period**: All analysis is standardized to start from **2008 onwards** for consistency across methodologies.

**All FSI outputs are scaled 0-1 where:**
- **0** = No/minimum financial stress
- **1** = Maximum financial stress
- **0.5** = Neutral/average stress level

## Methods

### 1. Dictionary-Based FSI
- Pre-defined stress queries with tier-based weights
- 25 queries in 5 tiers
- Weighted aggregation of Google Trends SVI
- Output: 0-1 scale

### 2. ML-Generated FSI (LASSO)
- LASSO regression learns which queries predict market declines
- ~100 queries with automatic feature selection
- FSI = negative of predicted return (normalized to 0-1)
- Handles large datasets with batch processing
- Output: 0-1 scale

### 3. News-Based FSI
- Scrapes news from G1, Valor Econômico, Folha de S. Paulo
- **Two-way co-occurrence**: financial + (stress OR negative) terms
- More inclusive than three-way co-occurrence
- Weighted scoring: stress terms count 2x more than negative terms
- Output: 0-1 scale

### 4. Combined FSI
- Combines all methodologies using weighted average, PCA, or dynamic weights
- Validates against IBOVESPA realized volatility
- All components normalized to 0-1 before combining
- Output: 0-1 scale

### 5. Econometric Enhancements

#### Kalman Filter Smoothing
- Treats FSI as latent variable observed with noise
- Separates signal from noise using state-space model
- Imputes missing values via Kalman smoothing (not linear interpolation)
- Provides variance decomposition analysis

#### Markov Switching Regime Analysis
- Identifies "Crisis" vs "Calm" regimes using Hamilton (1989) methodology
- Estimates regime-specific means and variances
- Provides transition probabilities and expected regime durations
- Validates against known historical crisis periods

#### CDS Benchmark Integration
- Compares FSI against Brazil 5Y CDS spread
- CDS serves as market-based sovereign stress indicator
- Calculates correlations and generates comparison plots

#### Granger Causality and IRF Analysis
- Tests causal relationships between FSI and market indicators
- Vector Autoregression (VAR) model with AIC-based lag selection
- Impulse Response Functions (IRF) with confidence intervals
- Variance decomposition (FEVD)

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
python scripts/collect_news.py --start-year 2002 --end-year 2025
python scripts/news_fsi.py

# === COMBINED FSI ===
python scripts/combined_fsi.py --method weighted

# === ADVANCED ANALYSIS ===
# Kalman smoothing
python scripts/analyze_fsi.py --smooth

# With CDS benchmark
python scripts/analyze_fsi.py --smooth --cds

# Regime analysis (Markov Switching)
python scripts/regime_analysis.py

# Granger Causality and IRF
python scripts/causality_analysis.py
```

## Project Structure

```
FinStress-News-Based-paper/
├── scripts/
│   ├── dictionaries.py        # Dictionary definitions (270+ terms, 100+ queries)
│   ├── collect_data.py        # Google Trends + IBOVESPA collection
│   ├── collect_news.py        # News scraping (G1, Valor, Folha)
│   ├── run_fsi.py             # Dictionary-based FSI
│   ├── ml_fsi.py              # ML-based FSI (LASSO)
│   ├── news_fsi.py            # News-based FSI
│   ├── combined_fsi.py        # Combined FSI (all methods)
│   ├── analyze_fsi.py         # Comprehensive FSI analysis + CDS benchmark
│   ├── econometrics.py        # Kalman Filter smoothing & gap imputation
│   ├── regime_analysis.py     # Markov Switching regime identification
│   ├── causality_analysis.py  # Granger Causality and IRF analysis
│   ├── fetch_cds_data.py      # Brazil CDS data utilities
│   └── fetch_ibov_data.py     # IBOVESPA data utilities
├── data/raw/                  # Input data
├── output/                    # Results and plots
│   ├── results/               # FSI CSVs
│   └── plots/                 # Visualization PNGs
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
python scripts/collect_data.py --start-date 2002-01-01 --end-date 2025-12-31
```

### News Data
```bash
# All sources (G1, Valor, Folha)
python scripts/collect_news.py --start-year 2002 --end-year 2025

# Folha only (fastest, no Selenium required)
python scripts/collect_news.py --folha-only

# Skip Selenium (Folha only)
python scripts/collect_news.py --no-selenium
```

**Note**: G1 and Valor require Selenium for JavaScript-rendered pages. Folha works with regular HTTP requests.

### CDS Data
Place Brazil CDS 5Y data at `Brasil CDS 5 Anos USD - Visão Geral.csv` (Investing.com format) or `data/raw/brazil_cds_5y.csv`.

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
python scripts/news_fsi.py --data path/to/news.csv
```

### Combined
```bash
python scripts/combined_fsi.py --method average
python scripts/combined_fsi.py --method weighted
python scripts/combined_fsi.py --method pca
python scripts/combined_fsi.py --method dynamic
```

## Methodology

### FSI Scale (0-1)

All FSI outputs use a normalized 0-1 scale:
```
0.0 = Minimum/no financial stress
0.5 = Neutral/average stress level
1.0 = Maximum financial stress
```

This allows direct comparison between different FSI methodologies.

### Dictionary FSI
```
FSI_raw = weighted_average(SVI × tier_weight)
FSI = min-max_normalize(FSI_raw) → [0, 1]

Tier 1 (Crisis):   1.5x weight
Tier 2 (Market):   1.2x weight
Tier 3-5:          1.0x weight
```

### ML FSI (LASSO)
```
Step 1: r_{t+1} = γ₀ + Σ φ_k × SVI_{k,t}
Step 2: LASSO (L1 penalty) selects predictive queries
Step 3: FSI_raw = -predicted_return
Step 4: FSI = min-max_normalize(FSI_raw) → [0, 1]
```

### News FSI
```
For each article:
  1. Count financial terms (mercado, bolsa, etc.)
  2. Count stress terms (crise, pânico, etc.)
  3. Count negative terms (queda, perda, etc.)
  4. TWO-WAY co-occurrence: financial + (stress OR negative)
  5. Weighted score = (stress × 2) + negative

FSI_raw = aggregate(stress_scores) per week
FSI = min-max_normalize(FSI_raw) → [0, 1]
```

### Combined FSI
```
Method 1: Simple average of normalized FSIs
Method 2: Weighted average (Dict: 40%, ML: 35%, News: 25%)
Method 3: PCA first principal component (normalized to 0-1)
Method 4: Dynamic weights based on rolling correlation with volatility
```

## Output Files

| Script | Output | Scale |
|--------|--------|-------|
| `run_fsi.py` | `output/results/fsi_weekly.csv` | 0-1 |
| `ml_fsi.py` | `output/ml_fsi_weekly.csv` | 0-1 |
| `news_fsi.py` | `output/news_fsi_weekly.csv` | 0-1 |
| `combined_fsi.py` | `output/combined_fsi.csv` | 0-1 |
| `analyze_fsi.py` | `output/fsi_analysis_report.txt` | - |
| `regime_analysis.py` | `output/regime_probabilities.csv` | 0-1 |
| `causality_analysis.py` | `output/causality_analysis_report.txt` | - |

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

## Performance Notes

- **Large datasets**: ML FSI uses batch processing for datasets with >10,000 samples
- **News scraping**: Folha is fastest (HTTP only); G1/Valor require Selenium
- **Memory**: For very large news datasets, FSI calculation is optimized with efficient pandas operations

## Advanced Econometric Methods

### Kalman Filter Smoothing

The state-space model treats FSI as a latent variable:

```
Observation: y_t = α_t + ε_t,    ε_t ~ N(0, σ²_ε)
State:       α_{t+1} = α_t + η_t,  η_t ~ N(0, σ²_η)
```

Where:
- `y_t` = Observed (noisy) FSI
- `α_t` = Latent (true) financial stress level
- `ε_t` = Observation noise
- `η_t` = State transition noise

Usage:
```bash
# Apply Kalman smoothing during analysis
python scripts/analyze_fsi.py --smooth

# Or use the econometrics module directly
python scripts/econometrics.py
```

### Markov Switching Regime Analysis

The Hamilton (1989) model assumes FSI depends on a latent regime:

```
y_t = μ_{S_t} + σ_{S_t} * ε_t,    ε_t ~ N(0, 1)
```

Where `S_t ∈ {0, 1}` follows a Markov chain with transition matrix P:

```
P = | p_00  p_01 |
    | p_10  p_11 |
```

This identifies:
- **Regime 0 (Calm)**: Low mean, low variance
- **Regime 1 (Crisis)**: High mean, high variance

Usage:
```bash
# Run regime analysis
python scripts/regime_analysis.py

# Use Kalman-smoothed data
python scripts/regime_analysis.py --smooth

# Custom probability threshold
python scripts/regime_analysis.py --threshold 0.7
```

### CDS Benchmark

Brazil 5Y CDS spread serves as market-based validation:

```bash
# Include CDS in analysis
python scripts/analyze_fsi.py --cds

# Full analysis with smoothing and CDS
python scripts/analyze_fsi.py --smooth --cds
```

To use CDS data, place a CSV file at project root:
- `Brasil CDS 5 Anos USD - Visão Geral.csv` (Investing.com format)
- Or `data/raw/brazil_cds_5y.csv` (standard format with date, cds_5y columns)

### Granger Causality and IRF

Tests whether FSI helps predict market indicators:

```bash
# Run causality analysis
python scripts/causality_analysis.py

# Custom lag order
python scripts/causality_analysis.py --lags 8
```

Output includes:
- Granger causality p-value heatmap
- Impulse Response Functions with confidence intervals
- Variance decomposition (FEVD)
- Stationarity tests (ADF)

## References

- Da, Z., Engelberg, J., & Gao, P. (2011). In Search of Attention. *The Journal of Finance*.
- García, D., Hu, X., & Rohrer, M. (2023). The Color of Finance Words. *Journal of Financial Economics*.
- Baker, S.R., Bloom, N., & Davis, S.J. (2016). Measuring Economic Policy Uncertainty. *Quarterly Journal of Economics*.
- Hamilton, J.D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. *Econometrica*.
- Granger, C.W.J. (1969). Investigating Causal Relations by Econometric Models and Cross-Spectral Methods. *Econometrica*.
- Durbin, J., & Koopman, S.J. (2012). Time Series Analysis by State Space Methods. *Oxford University Press*.
- Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis. *Springer*.

## Author

Pedro Igor - CAEN/UFC
