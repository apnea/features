# Features

## Chaos vs Noise Analysis for USD/EUR Exchange Rates

This module implements the methodology from Rosso et al. (2007) "Distinguishing Noise from Chaos" to analyze whether USD/EUR exchange rate time series exhibit chaotic or stochastic behavior using the complexity-entropy causality plane.

### Architecture Overview

The implementation consists of three main classes organized in a hierarchical structure:

```
ExchangeRateAnalyzer (Main Interface)
├── ChaosNoiseClassifier
    └── BandtPompeAnalysis (Core Algorithm)
```

#### Core Components

**1. BandtPompeAnalysis**
- **Purpose**: Implements the Bandt-Pompe ordinal pattern methodology
- **Key Methods**:
  - `ordinal_patterns()`: Extracts ordinal patterns from time series using embedding dimension D
  - `probability_distribution()`: Computes probability distribution P = {p_π} of ordinal patterns
  - `shannon_entropy()`: Calculates normalized Shannon entropy H_S ∈ [0,1]
  - `statistical_complexity()`: Computes Jensen-Shannon statistical complexity C_JS
- **Parameters**: Embedding dimension D (typically 3-7, default=6)

**2. ChaosNoiseClassifier**
- **Purpose**: Classifies time series based on position in complexity-entropy (CH) plane
- **Classification Rules** (from paper findings):
  - **Chaotic**: 0.45 < H_S < 0.7 + high C_JS (>0.4)
  - **Stochastic**: H_S > 0.7 + low C_JS (<0.3)
  - **White Noise**: H_S ≈ 1 + C_JS ≈ 0
  - **Periodic**: H_S < 0.45
- **Analysis Modes**: Full series or sliding window

**3. ExchangeRateAnalyzer**
- **Purpose**: Main interface for USD/EUR exchange rate analysis
- **Data Preprocessing Options**:
  - `'raw'`: Exchange rates as-is
  - `'returns'`: Price returns (Δp/p)
  - `'log_returns'`: Log returns (Δln(p)) - recommended for financial data
  - `'differenced'`: Simple differences (Δp)
- **Visualization**: CH plane plots, time evolution, statistical reports

### Theoretical Foundation

#### Complexity-Entropy Causality Plane
The method plots statistical complexity C_JS (y-axis) versus normalized Shannon entropy H_S (x-axis):

- **Entropy H_S**: Measures randomness/unpredictability (0 = deterministic, 1 = maximum randomness)
- **Complexity C_JS**: Measures structural organization within randomness (0 = no structure, max = optimal structure)

#### Ordinal Patterns (Bandt-Pompe Method)
1. **Embedding**: Create D-dimensional vectors from consecutive time series values
2. **Ordering**: Rank values within each vector to create ordinal patterns π
3. **Probability**: Count pattern frequencies to build distribution P = {p_π}
4. **Analysis**: Compute entropy and complexity from P

### Usage Example

```python
from chaos_noise_analysis import ExchangeRateAnalyzer

# Initialize analyzer
analyzer = ExchangeRateAnalyzer(embedding_dimension=6)

# Load USD/EUR data
analyzer.load_data('usd_eur_rates.csv', date_column='Date', rate_column='Rate')

# Run analysis
results = analyzer.run_analysis(preprocessing_method='log_returns')

# Generate report and visualizations
print(analyzer.generate_report())
fig = analyzer.plot_ch_plane()
fig.show()
```

### Key Insights from Methodology

**Distinguishing Characteristics:**
- **Chaotic systems**: Exhibit medium entropy with high complexity (structured but unpredictable)
- **Random noise**: Shows high entropy with low complexity (unpredictable but unstructured)
- **Periodic signals**: Display low entropy with minimal complexity (predictable and structured)

**Financial Applications:**
- Identify market regime changes (chaos ↔ noise transitions)
- Detect structural breaks in exchange rate dynamics
- Assess predictability windows in financial time series
- Complement traditional econometric analysis

### Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib`: Visualization
- `scipy`: Statistical functions
- `itertools`: Permutation generation

### References

Rosso, O. A., Larrondo, H. A., Martin, M. T., Plastino, A., & Fuentes, M. A. (2007). *Distinguishing Noise from Chaos*. Physical Review Letters, 99(15), 154102.