# Replication Package: Financial Stress Index for Brazil

**A News-Based Financial Stress Index: Evidence from Brazilian Markets**

Pedro Igor - CAEN/UFC

---

## Overview

This repository contains the replication code and data for constructing a Financial Stress Index (FSI) for Brazil using Google Trends search behavior, and analyzing its predictive power for sovereign credit risk (CDS spreads).

### Key Findings

1. **Dictionary-based FSI outperforms ML and combined approaches** for predicting CDS movements
2. **Positive and significant IRF**: A one-standard-deviation FSI shock predicts cumulative CDS increase of ~0.85 log-points over 12 months
3. **13 out of 13 months show significant predictive power** (p < 0.10) for dictionary-based FSI
4. **ML-based FSI has inverted behavior**: Negative predictive correlation with CDS (r = -0.24)

---

## Quick Replication

To replicate the main results, run the following commands:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate publication figures and tables (main results)
python scripts/generate_paper_figures.py
```

The main outputs will be in `output/figures/`:
- `figure_3_irf_main.png` - Main IRF result
- `table_2_irf_results.csv` - IRF summary statistics

---

## Full Replication Guide

### Step 1: Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

**Required packages**: pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib

### Step 2: Data Collection

The raw data is included in `data/raw/`. To recollect from sources:

```bash
# Google Trends data (dictionary queries)
python scripts/collect_data.py

# Google Trends data (ML queries - extended set)
python scripts/collect_data.py --ml

# News articles (optional - requires Selenium)
python scripts/collect_news.py --start-year 2008 --end-year 2025

# IBOVESPA market data
python scripts/fetch_ibov_data.py

# Brazil CDS data
python scripts/fetch_cds_data.py
```

### Step 3: FSI Construction

```bash
# Dictionary-based FSI (Da et al. 2011 methodology)
python scripts/run_fsi.py

# ML-based FSI (García et al. 2023 methodology)
python scripts/ml_fsi.py

# News-based FSI (optional)
python scripts/news_fsi.py

# Combined FSI (all methods)
python scripts/combined_fsi.py --method weighted
```

### Step 4: Econometric Analysis

```bash
# Main causality analysis (Local Projections IRF)
python scripts/causality_analysis.py

# Regime analysis (Markov Switching)
python scripts/regime_analysis.py

# FSI analysis with Kalman smoothing
python scripts/analyze_fsi.py --smooth --cds
```

### Step 5: Generate Paper Figures

```bash
# Generate all publication-quality figures
python scripts/generate_paper_figures.py
```

---

## Repository Structure

```
FinStress-News-Based-paper/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── data/
│   └── raw/                     # Raw input data
│       ├── google_trends_data.csv
│       ├── google_trends_ml.csv
│       ├── ibovespa_data.csv
│       ├── brazil_cds_5y.csv
│       └── news_*.csv
│
├── scripts/
│   ├── plot_style.py            # Academic plot styling
│   ├── dictionaries.py          # Search query dictionaries
│   │
│   │   # Data Collection
│   ├── collect_data.py          # Google Trends collection
│   ├── collect_news.py          # News scraping
│   ├── fetch_ibov_data.py       # IBOVESPA data
│   ├── fetch_cds_data.py        # CDS data
│   │
│   │   # FSI Construction
│   ├── run_fsi.py               # Dictionary-based FSI
│   ├── ml_fsi.py                # ML-based FSI (LASSO)
│   ├── news_fsi.py              # News-based FSI
│   ├── combined_fsi.py          # Combined FSI
│   │
│   │   # Econometric Analysis
│   ├── causality_analysis.py    # Local Projections IRF
│   ├── regime_analysis.py       # Markov Switching
│   ├── analyze_fsi.py           # FSI analysis
│   ├── econometrics.py          # Kalman Filter
│   │
│   │   # Paper Output
│   ├── generate_paper_figures.py # Publication figures
│   └── investigate_fsi_construction.py  # Methodology comparison
│
└── output/
    ├── figures/                 # Publication-ready figures
    │   ├── figure_1_fsi_timeseries.png
    │   ├── figure_2_fsi_vs_cds.png
    │   ├── figure_3_irf_main.png
    │   ├── figure_4_methodology_comparison.png
    │   ├── figure_5_crisis_analysis.png
    │   ├── figure_6_predictive_correlation.png
    │   ├── table_1_summary_statistics.csv
    │   └── table_2_irf_results.csv
    │
    ├── results/                 # Intermediate results
    │   ├── fsi_weekly.csv
    │   ├── fsi_ml_weekly.csv
    │   └── ml_coefficients.csv
    │
    ├── plots/                   # Additional plots
    └── fsi_monthly_aligned.csv  # Aligned monthly data
```

---

## Methodology

### Financial Stress Index Construction

#### Dictionary-Based FSI (Primary Method)

Based on Da, Engelberg, and Gao (2011), we construct an FSI using Google Trends Search Volume Index (SVI) for financial stress-related queries in Portuguese.

**Query categories**:
- Crisis terms (e.g., "crise financeira", "crash bolsa")
- Stock market terms (e.g., "queda ibovespa", "circuit breaker")
- Banking terms (e.g., "crise bancária", "falência banco")
- Sovereign risk terms (e.g., "default Brasil", "rebaixamento rating")
- Currency terms (e.g., "dólar dispara", "fuga capitais")

**Aggregation**:
```
FSI_t = Σ (w_i × SVI_{i,t}) / Σ w_i

where w_i = tier weight (1.5 for crisis, 1.2 for market, 1.0 for others)
```

**Normalization**: Min-max scaling to [0, 1] range.

#### ML-Based FSI (Comparison)

Based on García, Hu, and Rohrer (2023), we use LASSO regression to select predictive queries:

```
r_{t+1} = α + Σ β_i × SVI_{i,t} + ε_t

FSI_t = -ŷ_{t+1}  (negative of predicted return)
```

**Finding**: ML FSI has negative predictive correlation with CDS (r = -0.24), indicating inverted behavior relative to true financial stress.

### Econometric Analysis

#### Local Projections IRF (Jordà 2005)

We estimate impulse response functions using Local Projections:

```
Σ_{j=0}^{h} Δlog(CDS)_{t+j} = α_h + β_h × ΔFS_t + Σ_{k=1}^{p} γ_{h,k} × Δlog(CDS)_{t-k} + ε_{t+h}
```

Where:
- h = forecast horizon (0 to 12 months)
- β_h = cumulative IRF coefficient at horizon h
- Standard errors: Newey-West HAC with h+1 lags

#### Key Specifications

- **Data frequency**: Monthly
- **Sample period**: 2008-02 to 2025-11 (214 observations)
- **CDS transformation**: Log-returns (stationarity)
- **FSI transformation**: First differences (changes)
- **Control variables**: 4 lags of CDS log-returns

---

## Main Results

### Table 2: IRF Results Summary

| Method | Cumulative IRF | Significant Months | Peak Response |
|--------|----------------|-------------------|---------------|
| Dictionary FSI | **+0.846** | **13/13** | 0.087 (Month 1) |
| ML FSI | -0.281 | 0/13 | 0.033 (Month 5) |
| Combined FSI | +0.429 | 5/13 | 0.057 (Month 0) |

### Interpretation

A one-standard-deviation increase in the Dictionary FSI predicts:
- **Immediate effect**: 8.7% increase in CDS spread (Month 1)
- **Cumulative effect**: 84.6% cumulative increase over 12 months
- **Persistence**: Significant effects in all 13 months (p < 0.10)

---

## Data Sources

| Data | Source | Period |
|------|--------|--------|
| Google Trends SVI | Google Trends API | 2008-present |
| IBOVESPA | Yahoo Finance | 2008-present |
| Brazil 5Y CDS | Investing.com | 2008-present |
| News Articles | G1, Valor, Folha | 2008-present |

---

## FSI Scale Interpretation

All FSI outputs use a standardized 0-1 scale:

| Value | Interpretation |
|-------|----------------|
| 0.0 | Minimum/no financial stress |
| 0.5 | Neutral/average stress level |
| 1.0 | Maximum financial stress |

---

## Figures Description

### Figure 1: FSI Time Series
Shows the Dictionary-based FSI for Brazil (2008-2025) with 12-month moving average and crisis period shading (GFC, Brazilian Recession, COVID-19).

### Figure 2: FSI vs CDS
Dual-axis comparison of FSI with Brazil 5-Year CDS spread, demonstrating strong co-movement (r = 0.67).

### Figure 3: Impulse Response Function (Main Result)
Local Projections IRF showing CDS response to FSI shock with 90% and 95% confidence intervals.

### Figure 4: Methodology Comparison
Side-by-side IRF comparison of Dictionary, ML, and Combined FSI approaches.

### Figure 5: Crisis Period Analysis
Detailed FSI behavior during major crisis episodes (GFC, Brazilian Recession, COVID-19).

### Figure 6: Predictive Correlation
Bar chart showing FSI correlation with future CDS returns at different horizons.

---

## References

- Da, Z., Engelberg, J., & Gao, P. (2011). In Search of Attention. *The Journal of Finance*, 66(5), 1461-1499.
- García, D., Hu, X., & Rohrer, M. (2023). The Color of Finance Words. *Journal of Financial Economics*, 147(3), 525-549.
- Jordà, Ò. (2005). Estimation and Inference of Impulse Responses by Local Projections. *American Economic Review*, 95(1), 161-182.
- Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series. *Econometrica*, 57(2), 357-384.
- Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703-708.

---

## Citation

If you use this code or data, please cite:

```bibtex
@misc{igor2025fsi,
  author = {Igor, Pedro},
  title = {A News-Based Financial Stress Index: Evidence from Brazilian Markets},
  year = {2025},
  institution = {CAEN/UFC},
  url = {https://github.com/IIgor02/FinStress-News-Based-paper}
}
```

---

## License

This project is for academic purposes. Please cite appropriately if using in research.

---

## Contact

Pedro Igor - CAEN/UFC

For questions about the replication package, please open an issue on GitHub.
