# Financial Stress Index (FSI) for Brazil

Two complementary Financial Stress Indices using Google Trends search volume data for Brazil.

## Methods

### 1. Dictionary-Based FSI (Da et al. 2011)
- Pre-defined stress queries with tier-based weights
- 25 queries in 5 tiers
- Simple weighted aggregation

### 2. ML-Generated FSI (García et al. 2023)
- LASSO regression learns which queries predict market declines
- ~100 queries, automatic feature selection
- Data-driven approach

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# === DICTIONARY-BASED FSI ===
python scripts/collect_data.py --synthetic
python scripts/run_fsi.py

# === ML-BASED FSI ===
python scripts/collect_data.py --ml --synthetic
python scripts/ml_fsi.py
```

## Project Structure

```
FinStress-News-Based-paper/
├── scripts/
│   ├── dictionaries.py    # Query definitions (25 + 104 queries)
│   ├── collect_data.py    # Data collection
│   ├── run_fsi.py         # Dictionary-based FSI
│   └── ml_fsi.py          # ML-based FSI (LASSO)
├── data/raw/              # Input data
├── output/
│   ├── results/           # CSV outputs
│   └── plots/             # Visualizations
└── requirements.txt
```

## Data Collection

```bash
# Dictionary method (25 queries)
python scripts/collect_data.py

# ML method (104 queries)
python scripts/collect_data.py --ml

# Synthetic data for testing
python scripts/collect_data.py --synthetic
python scripts/collect_data.py --ml --synthetic
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

## Methodology

### Dictionary FSI (Da et al. 2011)
```
FSI = weighted_average(SVI × tier_weight)

Tier 1 (Crisis):   1.5x
Tier 2 (Market):   1.2x
Tier 3-5:          1.0x
```

### ML FSI (García et al. 2023)
```
Step 1: r_{t+1} = γ₀ + Σ φ_k × SVI_{k,t}
Step 2: LASSO selects predictive queries
Step 3: FSI = -predicted_return
```

## Output Files

### Dictionary FSI
- `fsi_weekly.csv`, `fsi_monthly.csv`

### ML FSI
- `fsi_ml_weekly.csv` - Weekly ML FSI
- `ml_coefficients.csv` - Query coefficients
- `ml_diagnostics.csv` - Model metrics

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

## References

- Da, Z., Engelberg, J., & Gao, P. (2011). In Search of Attention. The Journal of Finance.
- García, D., Hu, X., & Rohrer, M. (2023). The Color of Finance Words. Journal of Financial Economics.

## Author

Pedro Igor - CAEN/UFC
