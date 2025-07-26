# Features

## Chaos vs Noise Analysis for EUR/USD Exchange Rates

This module implements the methodology from Rosso et al. (2007) "Distinguishing Noise from Chaos" to analyze whether EUR/USD exchange rate time series exhibit chaotic or stochastic behavior using the complexity-entropy causality plane. Data is ingested from an Apache Arrow Flight server.

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
- **Purpose**: Main interface for EUR/USD exchange rate analysis with Arrow Flight integration
- **Data Sources**: 
  - Arrow Flight server at `grpc://localhost:8815` (primary)
  - Legacy file loading (CSV/Excel) for backward compatibility
- **Data Types Supported**:
  - `'mid_price'`: (Ask + Bid) / 2 - most common for analysis
  - `'ask_price'`: EUR/USD ask prices
  - `'bid_price'`: EUR/USD bid prices  
  - `'spread'`: Ask - Bid (absolute spread)
  - `'relative_spread'`: Spread / Mid-price (normalized spread)
- **Data Preprocessing Options**:
  - `'raw'`: Exchange rates as-is
  - `'returns'`: Price returns (Δp/p)
  - `'log_returns'`: Log returns (Δln(p)) - recommended for prices
  - `'differenced'`: Simple differences (Δp) - recommended for spreads
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

# Initialize analyzer with Flight server connection
analyzer = ExchangeRateAnalyzer(embedding_dimension=6, 
                               flight_server="grpc://localhost:8815")

# Load EUR/USD data from Arrow Flight server
analyzer.load_data_from_flight("EURUSD")

# Run analysis on mid-price log returns
results = analyzer.run_analysis(data_type='mid_price', 
                               preprocessing_method='log_returns')

# Run analysis on spread dynamics
spread_results = analyzer.run_analysis(data_type='spread', 
                                      preprocessing_method='differenced')

# Generate report and visualizations
print(analyzer.generate_report())
fig = analyzer.plot_ch_plane()
fig.show()

# Compare multiple analyses
analysis_configs = [
    {'data_type': 'mid_price', 'method': 'log_returns'},
    {'data_type': 'spread', 'method': 'differenced'},
    {'data_type': 'relative_spread', 'method': 'differenced'}
]

for config in analysis_configs:
    results = analyzer.run_analysis(**config)
    print(f"Analysis: {config}")
    print(analyzer.generate_report())
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
- Analyze bid-ask spread microstructure behavior
- Compare price vs spread dynamics for market efficiency insights
- Complement traditional econometric analysis

### Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib`: Visualization
- `scipy`: Statistical functions
- `itertools`: Permutation generation
- `pyarrow`: Arrow Flight client for data ingestion

### Data Schema

The Arrow Flight server should provide EUR/USD data with the following schema:
- `UTC`: Timestamp with timezone
- `AskPrice`: EUR/USD ask price (double)
- `BidPrice`: EUR/USD bid price (double)
- `AskVolume`: Ask volume (double)
- `BidVolume`: Bid volume (double)

### References

Rosso, O. A., Larrondo, H. A., Martin, M. T., Plastino, A., & Fuentes, M. A. (2007). *Distinguishing Noise from Chaos*. Physical Review Letters, 99(15), 154102.