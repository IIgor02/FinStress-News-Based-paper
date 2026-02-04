"""
Financial Stress Index (FSI) - Dictionary-Based Implementation
==============================================================
Based on Baker et al. (2019) methodology adapted for Brazilian Portuguese.

This package provides tools to:
1. Classify social media posts as indicating financial stress
2. Calculate daily and monthly FSI values
3. Validate against market volatility
4. Generate visualizations

Main Components:
- dictionaries: Portuguese financial, stress, and negative term dictionaries
- preprocessing: Text preprocessing for Portuguese social media
- classifier: Three-way co-occurrence stress classifier
- fsi_calculator: FSI calculation with weighting
- data_generator: Synthetic data generation for testing
- validation: Validation metrics and visualizations
"""

from .dictionaries import (
    FINANCIAL_TERMS,
    STRESS_TERMS,
    NEGATIVE_TERMS,
    DICTIONARIES,
    CRISIS_EPISODES,
    get_dictionary_stats,
)

from .preprocessing import (
    preprocess_text,
    extract_tokens_for_matching,
    clean_text,
    remove_stopwords,
)

from .classifier import (
    StressClassifier,
    ClassificationResult,
    classify_stress,
)

from .fsi_calculator import (
    FSICalculator,
    calculate_fsi_from_data,
)

from .data_generator import (
    generate_synthetic_twitter_data,
    generate_synthetic_reddit_data,
    download_ibovespa_data,
)

from .validation import (
    validate_fsi,
    print_validation_report,
    plot_fsi_timeseries,
    plot_fsi_vs_volatility,
    create_all_plots,
)

__version__ = '1.0.0'
__author__ = 'CAEN/UFC Master Research'
